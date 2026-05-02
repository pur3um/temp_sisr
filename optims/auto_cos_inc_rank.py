import math
import statistics
from typing import Optional

import torch
from time import time

def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int,
    eps: float = 1e-7,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    assert G.ndim == 2

    a, b, c = (3.4445, -4.7750, 2.0315)

    compute_dtype = torch.bfloat16 if (use_bfloat16 and G.is_cuda) else torch.float32
    X = G.to(dtype=compute_dtype)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

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
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
    rescale: bool = False,
) -> torch.Tensor:
    assert G.ndim == 2, "Expected a 2D matrix after any conv flattening."

    # start_time = time()
    
    if rank <= 0:
        return torch.zeros_like(G)

    X = G.float()
    transposed = False

    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    m, n = X.shape
    r = min(m, n)
    sketch_dim = min(rank + max(0, oversample), r)

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

    # omega_time = time()
    # print("Multiple Omega : ", omega_time - start_time)

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.mT @ X

    # qr_time = time()
    # print("QR deomp : ", qr_time - omega_time)

    S = zeropower_via_newtonschulz5(
        B,
        steps=steps,
        eps=eps,
        use_bfloat16=small_ns_bfloat16,
    ).float()

    # ns_time = time()
    # print("NS iter : ", ns_time - qr_time)

    Z = Q @ S

    if rescale and sketch_dim > 0:
        Z = Z * math.sqrt(float(r) / float(sketch_dim))

    if transposed:
        Z = Z.mT

    # end_time = time()
    # func_whole_time = end_time - start_time
    # print("Total time : ", func_whole_time)

    return Z.type_as(G)



def _round_up_to_multiple(value: int, multiple: int = 8) -> int:
    multiple = max(1, int(multiple))
    return int(math.ceil(int(value) / float(multiple)) * multiple)


def _clamp_rank(value: int, floor_rank: int, ceil_rank: int) -> int:
    floor_rank = int(floor_rank)
    ceil_rank = max(floor_rank, int(ceil_rank))
    return max(floor_rank, min(int(value), ceil_rank))


@torch.no_grad()
def build_muon_search_matrix(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim > 2:
        update = update.view(update.shape[0], -1)
    elif update.ndim < 2:
        update = update.view(1, -1)

    return update


@torch.no_grad()
def preview_muon_search_matrix(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    """
    Preview the Muon search matrix without mutating the live momentum buffer.
    This keeps the probe branch from changing the original optimizer dynamics.
    """
    momentum_preview = momentum.detach().clone()
    grad_preview = grad.detach().clone()

    momentum_preview.lerp_(grad_preview, 1 - beta)
    update = grad_preview.lerp_(momentum_preview, beta) if nesterov else momentum_preview

    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim > 2:
        update = update.view(update.shape[0], -1)
    elif update.ndim < 2:
        update = update.view(1, -1)

    return update


@torch.no_grad()
def choose_auto_rank_start(
    update: torch.Tensor,
    floor_rank: int,
    probe_rank: int,
    energy_tau: float = 0.90,
    round_multiple: int = 8,
    eps: float = 1e-12,
) -> int:
    """
    Cheap Frobenius-energy rank chooser.

    We form B = Q^T U from a randomized sketch of the current Muon search matrix U,
    then choose the smallest r such that

        sum_{i<=r} sigma_i(B)^2 / ||U||_F^2 >= energy_tau.

    The returned rank is then floored by floor_rank and rounded up.
    """
    assert update.ndim == 2

    X = update.float()
    if X.size(-2) > X.size(-1):
        X = X.mT

    m, n = X.shape
    limit = min(m, n)
    floor_rank = max(1, min(int(floor_rank), limit))
    probe_rank = max(floor_rank, min(int(probe_rank), limit))

    Omega = torch.randn(n, probe_rank, device=X.device, dtype=X.dtype)
    Y = X @ Omega

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.mT @ X

    svals = torch.linalg.svdvals(B.float())
    if svals.numel() == 0:
        return floor_rank

    capture = torch.cumsum(svals.square(), dim=0) / ((X.square().sum()) + eps)

    if float(capture[-1].item()) < float(energy_tau):
        return probe_rank

    threshold = torch.tensor(float(energy_tau), device=capture.device, dtype=capture.dtype)
    r_hat = int(torch.searchsorted(capture, threshold).item()) + 1
    r_hat = max(floor_rank, r_hat)
    r_hat = min(r_hat, probe_rank)
    r_hat = _round_up_to_multiple(r_hat, round_multiple)
    return _clamp_rank(r_hat, floor_rank, probe_rank)



def get_cosine_rank(step: int, start_rank: int, end_rank: int, warmup_steps: int) -> int:
    step = int(step)
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    warmup_steps = int(warmup_steps)

    if warmup_steps <= 1:
        return end_rank
    if step <= 1:
        return start_rank
    if step >= warmup_steps:
        return end_rank

    t = (step - 1) / float(warmup_steps - 1)
    progress = 0.5 * (1.0 - math.cos(math.pi * t))

    rank = start_rank + (end_rank - start_rank) * progress
    return int(round(rank))


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    rank: int = 200,
    oversample: int = 4,
    lowrank_rescale: bool = False,
    eps: float = 1e-6,
    small_ns_bfloat16: bool = False,
    step: int = 1,
    rank_start: int = 200,
    rank_end: int = 250,
    warmup_steps: int = 100000,
    current_rank: Optional[int] = None,
) -> torch.Tensor:
    update = build_muon_search_matrix(
        grad,
        momentum,
        beta=beta,
        nesterov=nesterov,
    )

    # Backward-compatible path: if current_rank is not provided, compute it here.
    if current_rank is None:
        scheduled_rank = get_cosine_rank(
            step=step,
            start_rank=rank_start,
            end_rank=rank_end,
            warmup_steps=warmup_steps,
        )
        applied_rank = max(int(rank), int(scheduled_rank))
    else:
        # Preferred path: caller computes the rank and passes it directly.
        applied_rank = int(current_rank)

    update = zeropower_via_lowrank_matrix_sign(
        update,
        steps=ns_steps,
        rank=applied_rank,
        oversample=oversample,
        eps=eps,
        small_ns_bfloat16=small_ns_bfloat16,
        rescale=lowrank_rescale,
    )

    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update



def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceAutoCosIncWithAuxAdam(torch.optim.Optimizer):
    """
    Muon + auxiliary Adam.

    When auto_init_rank_start=True:
      (1) estimate initial rank for the first init_probe_steps steps,
      (2) compare it against the floor rank via max(rank, estimated_start),
      (3) grow it with the cosine schedule over warmup_steps,
      (4) pass the actually used rank directly into muon_update.
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
                g.setdefault("ns_steps", 5) #! KU server는 여기가 10
                g.setdefault("nesterov", True)
                g.setdefault("rank", 200)
                g.setdefault("rank_start", g["rank"])
                g.setdefault("rank_end", g["rank"])
                g.setdefault("warmup_steps", 1)
                g.setdefault("oversample", 4)
                g.setdefault("lowrank_rescale", False)
                g.setdefault("eps", 1e-6)
                g.setdefault("small_ns_bfloat16", False)
                g.setdefault("step", 0)
                g.setdefault("current_rank", g["rank_start"])
                g.setdefault("current_target_rank", max(int(g["rank"]), int(g["rank_end"])))
                g.setdefault("current_method", "cosine_inc_closed_form")

                # Auto-initialize rank_start from the early sketched Frobenius-energy curve.
                g.setdefault("auto_init_rank_start", False)
                g.setdefault("init_probe_steps", 8)
                g.setdefault("init_energy", 0.90)
                g.setdefault("init_round_multiple", 8)
                g.setdefault("auto_rank_start_final", None)
                g.setdefault("_init_rank_candidates", [])
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
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
                group_step = int(group.get("step", 0)) + 1
                group["step"] = group_step

                # (1) Initial-rank estimation for the first init_probe_steps steps.
                if bool(group.get("auto_init_rank_start", False)) and group.get("auto_rank_start_final") is None:
                    if group_step <= int(group["init_probe_steps"]):
                        step_candidates = []
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)

                            # Preview only: do not mutate the live momentum buffer.
                            search_matrix = preview_muon_search_matrix(
                                p.grad,
                                state["momentum_buffer"],
                                beta=group["momentum"],
                                nesterov=group["nesterov"],
                            )
                            candidate = choose_auto_rank_start(
                                search_matrix,
                                floor_rank=group["rank"],
                                probe_rank=group["rank_end"],
                                energy_tau=group["init_energy"],
                                round_multiple=group["init_round_multiple"],
                            )
                            step_candidates.append(int(candidate))

                        if step_candidates:
                            step_median = int(statistics.median(step_candidates))
                            step_median = max(int(group["rank"]), step_median)
                            step_median = min(int(group["rank_end"]), step_median)
                            group["_init_rank_candidates"].append(step_median)

                            provisional_start = int(statistics.median(group["_init_rank_candidates"]))
                            provisional_start = max(int(group["rank"]), provisional_start)
                            provisional_start = min(int(group["rank_end"]), provisional_start)
                            group["rank_start"] = _clamp_rank(
                                _round_up_to_multiple(provisional_start, int(group["init_round_multiple"])),
                                int(group["rank"]),
                                max(int(group["rank"]), int(group["rank_end"])),
                            )

                    if group_step == int(group["init_probe_steps"]):
                        if group["_init_rank_candidates"]:
                            final_start = int(statistics.median(group["_init_rank_candidates"]))
                            final_start = max(int(group["rank"]), final_start)
                            final_start = min(int(group["rank_end"]), final_start)
                            final_start = _clamp_rank(
                                _round_up_to_multiple(final_start, int(group["init_round_multiple"])),
                                int(group["rank"]),
                                max(int(group["rank"]), int(group["rank_end"])),
                            )
                        else:
                            final_start = int(group["rank"])
                        group["auto_rank_start_final"] = final_start
                        group["rank_start"] = final_start

                # (2) and (3): compare against floor rank, then apply cosine growth.
                scheduled_rank = get_cosine_rank(
                    step=group_step,
                    start_rank=group["rank_start"],
                    end_rank=group["rank_end"],
                    warmup_steps=group["warmup_steps"],
                )
                applied_rank = max(int(group["rank"]), int(scheduled_rank))

                group["current_rank"] = applied_rank
                group["current_target_rank"] = max(int(group["rank"]), int(group["rank_end"]))
                group["current_method"] = (
                    "cosine_inc_auto_start"
                    if bool(group.get("auto_init_rank_start", False))
                    else "cosine_inc_closed_form"
                )

                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    # (4) Pass the rank directly into muon_update and use it for the
                    # low-rank subspace projection.
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        rank=group["rank"],
                        oversample=group["oversample"],
                        lowrank_rescale=group["lowrank_rescale"],
                        eps=group["eps"],
                        small_ns_bfloat16=group["small_ns_bfloat16"],
                        step=group_step,
                        rank_start=group["rank_start"],
                        rank_end=group["rank_end"],
                        warmup_steps=group["warmup_steps"],
                        current_rank=applied_rank,
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

"""
python sisr_auto_cos_inc_rank_wsd.py \
  --task super_resolution \
  --optimizer auto_cos_inc \
  --scheduler rank_wsd \
  --epochs 5000 \
  --rank 64 \
  --rank_start 64 \
  --rank_end 256 \
  --rank_warmup_steps 4000 \
  --rank_wsd_decay_start_step 4000 \
  --rank_wsd_min_lr_ratio 0.1 \
  --muon_lr 1e-1 \
  --lr 1e-3
"""