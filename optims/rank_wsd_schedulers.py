import math
from typing import Dict, Optional


# 아주 짧은 warmup 후, rank가 자라는 동안 LR을 유지하고, 그 뒤에만 늦게 linear decay
class RankAwareWarmupStableLinearScheduler:
    """
    WSD-like scheduler for INR / progressive-rank optimizers.

    Shape:
        micro warmup -> stable plateau -> late linear decay

    The intended use is to keep the peak LR unchanged while the optimizer's
    effective subspace is still growing, and only start decaying after the
    rank schedule has essentially saturated.
    """

    def __init__(
        self,
        base_lr_adam: float,
        base_lr_muon: float,
        total_iters: int,
        base_N_rand: int,
        warmup_steps: int = 0,
        decay_start_step: Optional[int] = None,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if total_iters <= 0:
            raise ValueError(f"total_iters must be > 0, got {total_iters}")
        if base_N_rand <= 0:
            raise ValueError(f"base_N_rand must be > 0, got {base_N_rand}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        self.base_lr_adam = float(base_lr_adam)
        self.base_lr_muon = float(base_lr_muon)
        self.total_iters = int(total_iters)
        self.base_N_rand = int(base_N_rand)
        self.warmup_steps = min(int(warmup_steps), max(0, self.total_iters - 1))
        self.min_lr_ratio = float(min_lr_ratio)

        if decay_start_step is None:
            resolved_decay_start = max(self.warmup_steps, int(round(0.8 * self.total_iters)))
        else:
            resolved_decay_start = int(decay_start_step)
        resolved_decay_start = max(self.warmup_steps, resolved_decay_start)
        resolved_decay_start = min(self.total_iters - 1, resolved_decay_start)
        self.decay_start_step = resolved_decay_start

        self._decay_span = max(1, (self.total_iters - 1) - self.decay_start_step)

    @property
    def effective_total_iters(self) -> int:
        return self.total_iters

    def _lr_ratio(self, step: int) -> float:
        step = int(max(0, step))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(max(1, self.warmup_steps))

        if step < self.decay_start_step:
            return 1.0

        decay_step = min(max(step - self.decay_start_step, 0), self._decay_span)
        progress = float(decay_step) / float(self._decay_span)
        return 1.0 - (1.0 - self.min_lr_ratio) * progress

    def step(self, global_step: int) -> Dict[str, float]:
        step = int(max(0, global_step))
        ratio = self._lr_ratio(step)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            phase = -1
            phase_name = "warmup"
        elif step < self.decay_start_step:
            phase = 0
            phase_name = "stable"
        else:
            phase = 1
            phase_name = "linear_decay"

        return {
            "lr_adam": self.base_lr_adam * ratio,
            "lr_muon": self.base_lr_muon * ratio,
            "N_rand": self.base_N_rand,
            "phase": phase,
            "phase_name": phase_name,
            "lr_ratio": ratio,
        }

    def describe(self) -> str:
        return (
            "[RankWSD] "
            f"total_iters={self.total_iters} warmup_steps={self.warmup_steps} "
            f"decay_start_step={self.decay_start_step} min_lr_ratio={self.min_lr_ratio:.6g} "
            f"base_N_rand={self.base_N_rand}"
        )
