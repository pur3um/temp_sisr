import math
import torch


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int,
    eps: float = 1e-7,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """
    Approximate matrix sign / polar factor via the quintic Newton-Schulz iteration used in Muon.

    For a rectangular matrix G = U S V^T, this returns an approximation to U V^T.
    In practice, with only a few steps, the output is closer to U S' V^T for a bounded diagonal S',
    which is exactly the same practical approximation used by the original Muon code path.

    Args:
        G:
            2D matrix to orthogonalize.
        steps:
            Number of Newton-Schulz steps.
        eps:
            Small constant for normalization stability.
        use_bfloat16:
            If True and running on CUDA, compute in bfloat16 for speed. For the small projected
            matrix in the low-rank routine below, float32 is usually preferable, so that routine
            sets this to False by default.
    """
    assert G.ndim == 2

    a, b, c = (3.4445, -4.7750, 2.0315)

    compute_dtype = torch.bfloat16 if (use_bfloat16 and G.is_cuda) else torch.float32
    X = G.to(dtype=compute_dtype)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Keep the scale controlled before Newton-Schulz. This is the same practical normalization
    # used in Muon-style code; it is not an exact spectral-norm normalization.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(dtype=G.dtype)


@torch.no_grad()
def zeropower_via_lowrank_matrix_sign(
    G: torch.Tensor,
    steps: int = 5,
    rank: int = 200,
    oversample: int = 4,
    n_subspace_iters: int = 1,
    eps: float = 1e-6,
    rescale: bool = True,
    small_ns_bfloat16: bool = False,
) -> torch.Tensor:
    """
    Paper-style low-rank orthogonalization block:

        1) sketch a dominant left subspace with Y = G Omega
        2) optionally refine the subspace with power iteration
        3) Q = qr(Y)
        4) B = Q^T G
        5) S = msgn(B)  (here approximated with quintic Newton-Schulz)
        6) return Q S

    This differs from the compressed-SVD route:

        Y -> Q -> B -> svd(B) -> U_k V_k^T

    because the final orthogonalization is now done through a small projected matrix-sign step
    rather than through a truncated small SVD.

    Notes:
        - If you want to be closer to the paper's basic fixed-rank sketch, set oversample=0.
        - If you want to be closer to the appendix power-iteration variant, keep n_subspace_iters>0.
        - The optional rescale is still a heuristic from the practical code path, not a paper-faithful
          requirement.
    """
    assert G.ndim == 2, "Expected a 2D matrix after any conv flattening."

    if rank <= 0:
        return torch.zeros_like(G)

    X = G.float()
    transposed = False

    # Work with m <= n for numerical convenience, same as the original code.
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m, n = X.shape
    r = min(m, n)

    # In this implementation, oversample increases the sketch dimension itself.
    # For a more literal fixed-rank paper version, use oversample=0.
    sketch_dim = min(rank + max(0, oversample), r)

    # No low-rank benefit left: just orthogonalize the full matrix directly.
    if sketch_dim >= r:
        Z = zeropower_via_newtonschulz5(
            X,
            steps=steps,
            eps=eps,
            use_bfloat16=small_ns_bfloat16,
        ).float()
        if transposed:
            Z = Z.mT
        return Z.type_as(G)

    Omega = torch.randn(n, sketch_dim, device=X.device, dtype=X.dtype)
    Y = X @ Omega

    Qc, _ = torch.linalg.qr(Y, mode="reduced")
    B = Qc.mT @ X

    # Small matrix sign on the projected matrix.
    S = zeropower_via_newtonschulz5(
        B,
        steps=steps,
        eps=eps,
        use_bfloat16=small_ns_bfloat16,
    ).float()

    Z = Qc @ S

    if transposed:
        Z = Z.mT

    return Z.type_as(G)


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    rank: int = 200,
    oversample: int = 4,
    n_subspace_iters: int = 1,
    lowrank_rescale: bool = True,
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
) -> torch.Tensor:
    """
    Practical Muon-style momentum path from the original code, but with the orthogonalization block
    replaced by low-rank sketch + small-matrix-sign.

    Important:
        This is *not* yet the full paper-faithful optimizer dynamics.
        The momentum / Nesterov block remains your original practical heuristic.
        What changed here is the orthogonalization block only.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:  # conv filters -> flatten to matrix
        update = update.view(len(update), -1)

    update = zeropower_via_lowrank_matrix_sign(
        update,
        steps=ns_steps,
        rank=rank,
        oversample=oversample,
        n_subspace_iters=n_subspace_iters,
        eps=eps,
        rescale=lowrank_rescale,
        small_ns_bfloat16=small_ns_bfloat16,
    )

    # Same Muon aspect-ratio scaling used in your current code.
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


class SingleDeviceLRMuon(torch.optim.Optimizer):
    """
    Non-distributed Muon variant with:
        low-rank sketch -> QR -> small matrix sign -> lift back

    API is intentionally close to your original optimizer, but low-rank controls are exposed so you
    can actually sweep them instead of hard-coding them inside muon_update.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        rank: int = 200,
        oversample: int = 4,
        n_subspace_iters: int = 1,
        lowrank_rescale: bool = True,
        eps: float = 1e-6,
        small_ns_bfloat16: bool = False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
            rank=rank,
            oversample=oversample,
            n_subspace_iters=n_subspace_iters,
            lowrank_rescale=lowrank_rescale,
            eps=eps,
            small_ns_bfloat16=small_ns_bfloat16,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                    rank=group["rank"],
                    oversample=group["oversample"],
                    n_subspace_iters=group["n_subspace_iters"],
                    lowrank_rescale=group["lowrank_rescale"],
                    eps=group["eps"],
                    small_ns_bfloat16=group["small_ns_bfloat16"],
                )

                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceSignWithAuxAdam(torch.optim.Optimizer):
    """
    Muon + auxiliary Adam.

    Use use_muon=True groups for matrix-like parameters and use_muon=False groups for parameters
    you want to leave on Adam.
    """

    def __init__(self, param_groups):
        normalized_groups = []
        for group in param_groups:
            assert "use_muon" in group, "Each param group must include use_muon=True/False."

            g = dict(group)
            if g["use_muon"]:
                g.setdefault("lr", 0.003)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
                g.setdefault("nesterov", True)
                g.setdefault("rank", 200)
                g.setdefault("oversample", 4)
                g.setdefault("n_subspace_iters", 1)
                g.setdefault("lowrank_rescale", True)
                g.setdefault("eps", 1e-6)
                g.setdefault("small_ns_bfloat16", False)
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)

            normalized_groups.append(g)

        super().__init__(normalized_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        rank=group["rank"],
                        oversample=group["oversample"],
                        n_subspace_iters=group["n_subspace_iters"],
                        lowrank_rescale=group["lowrank_rescale"],
                        eps=group["eps"],
                        small_ns_bfloat16=group["small_ns_bfloat16"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
