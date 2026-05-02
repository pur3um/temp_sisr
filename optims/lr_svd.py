import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.no_grad()
def zeropower_via_lowrank_dominant_subspace(
    G: torch.Tensor,
    steps: int = 5,
    rank: int = 8,
    oversample: int = 4,
    n_subspace_iters: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Low-rank dominant-subspace approximation of the polar factor.

    Goal:
        G ~= U_k S_k V_k^T  ->  return U_k V_k^T

    Why:
        - cheaper than full SVD
        - keeps dominant input/output directional information
        - often better aligned with "direction-only" hypothesis than few-step NS

    Args:
        G: (..., m, n) or (m, n). In your current Muon flow, usually 2D.
        steps:
            kept for interface compatibility with zeropower_via_newtonschulz5.
            If n_subspace_iters is None-like in your future edits, you may map steps to it.
        rank:
            target dominant rank k
        oversample:
            randomized SVD oversampling
        n_subspace_iters:
            number of power/subspace iterations
        eps:
            numerical stability

    Returns:
        Approximate polar factor with rank-k dominant subspace:
            U_k V_k^T
        same shape / dtype as G
    """
    assert G.ndim >= 2

    X = G.float()
    transposed = False

    # Work on the wider-last form for numerical convenience
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m, n = X.shape[-2], X.shape[-1]
    r = min(m, n)

    # print(rank, r)
    # import pdb; pdb.set_trace()
    # Fallback: if requested rank is too large, just do exact economy SVD
    if rank >= r:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        Q = U @ Vh
        if transposed:
            Q = Q.mT
        return Q.type_as(G)

    k = min(rank + oversample, r)

    # Random test matrix
    Omega = torch.randn(n, k, device=X.device, dtype=X.dtype)

    # Sample dominant column space: Y = X Omega
    Y = X @ Omega

    # Optional subspace iteration
    # Y <- X (X^T Y) repeatedly
    for _ in range(max(0, int(n_subspace_iters))):
        Y = X @ (X.mT @ Y)

    # Orthonormal basis for dominant left subspace
    Qc, _ = torch.linalg.qr(Y, mode="reduced")   # (m, k)

    # Small matrix
    B = Qc.mT @ X                                # (k, n)

    # Small SVD
    Uh, S, Vh = torch.linalg.svd(B, full_matrices=False)

    # Keep only top-rank components
    rk = min(rank, Uh.shape[-1], Vh.shape[-2])
    Uh = Uh[:, :rk]                              # (k, rk)
    Vh = Vh[:rk, :]                              # (rk, n)

    # Lift back to original space
    U = Qc @ Uh                                 # (m, rk)

    # Truncated polar factor
    Z = U @ Vh                                  # (m, n)

    # Optional rescale to avoid too-small updates when rank is tiny
    # Frobenius norm of U@Vh is sqrt(rk), while full orthogonal factor would be sqrt(r).
    # This compensates partially for rank truncation.
    if rk > 0:
        Z = Z * ((float(r) / float(rk)) ** 0.5)

    if transposed:
        Z = Z.mT

    return Z.type_as(G)

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    # update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = zeropower_via_lowrank_dominant_subspace(
        update,
        steps=ns_steps,
        rank=200,  # 150 not bad
        oversample=4,
        n_subspace_iters=1,
    )
    #update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update



class SingleDeviceLRMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
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
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceLrSVDWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.003)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

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
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    
    