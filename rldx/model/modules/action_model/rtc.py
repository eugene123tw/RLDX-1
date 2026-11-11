"""Real-Time Chunking (RTC) for flow-matching action heads.

Implements two mutually-reinforcing variants:

- **Training RTC** (arXiv 2512.05964): Per-sample inference-delay simulation.
  At training time, sample a delay d_i in [0, max_delay] for each batch item;
  set t=1 on the first d_i action-horizon positions (they stay clean ground
  truth), sample normal t for the rest, and mask loss on the clean prefix.
  The model learns to predict the postfix conditioned on a clean prefix.

- **Inference RTC**
    * ``trained`` (2512.05964): Hard-inpaint the frozen prefix at each
      Euler step and feed the model per-token time with t=1 on the prefix.
      Only valid for models trained with training-RTC.
    * ``guided`` (2506.07339 Eq. 2): Jacobian universal-guidance modification
      of the velocity field via the vector-Jacobian product of the predicted
      clean trajectory ``Â¹ = x_τ + (1-τ)·v`` with respect to ``x_τ``. Works
      with any pre-trained flow-matching VLA, no retraining required.

The soft mask ``W`` (Eq. 5) ramps from 1 on the frozen prefix down to 0 on the
free postfix with an exponential taper across the middle band.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional

import torch

from rldx.configs.model.inference import GuidedRtcConfig, InferenceRtc, TrainedRtcConfig


_EPS_TAU = 1e-2  # Keep the guidance scale finite at the τ endpoints.
_EPS_DENOM = 1e-8


@dataclass(frozen=True)
class RtcTiming:
    rtc: InferenceRtc
    exec_horizon: int
    horizon: int

    def __post_init__(self) -> None:
        d, s, h = self.delay, self.exec_horizon, self.horizon
        if not 1 <= d <= s <= h - d:
            raise ValueError(f"must satisfy 1 <= d ({d}) <= s ({s}) <= H - d ({h - d}); H={h}")

    @property
    def delay(self) -> int:
        return self.rtc.delay

    @property
    def guided(self) -> bool:
        return isinstance(self.rtc, GuidedRtcConfig)

    @property
    def ramp(self) -> int:
        return self.horizon - self.exec_horizon - self.delay

    def soft_mask(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return compute_soft_mask_weights(
            self.horizon, self.delay, self.exec_horizon, device=device, dtype=dtype
        )


def rtc_timing(rtc: InferenceRtc | None, training_max_delay: int, horizon: int) -> RtcTiming | None:
    """Resolve the serving RTC against what the checkpoint was trained with."""
    if rtc is None:
        return None
    # A prefix longer than any delay sampled in training asks the checkpoint to
    # honor a conditioning it has never seen (2512.05964 Alg. 1).
    if isinstance(rtc, TrainedRtcConfig) and not 0 < rtc.delay <= training_max_delay:
        raise ValueError(
            f"trained RTC with delay={rtc.delay} needs a checkpoint trained with "
            f"rtc_training_max_delay >= {rtc.delay}, this one has {training_max_delay}. "
            "Guided RTC needs no RTC training."
        )
    return RtcTiming(rtc, rtc.exec_horizon, horizon)


class RtcActions(NamedTuple):
    actions: torch.Tensor
    present: torch.Tensor


class RtcInput(NamedTuple):
    prefix: RtcActions
    postfix: RtcActions | None


def sample_training_prefix(
    batch_size: int,
    horizon: int,
    max_delay: int,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a boolean prefix mask for training RTC (Algorithm 1 of
    arXiv 2512.05964).

    For each sample draws ``d_i ~ U[0, max_delay]`` and marks positions
    ``[0, d_i)`` as frozen prefix. ``d_i = 0`` naturally recovers the
    standard flow-matching case, so no extra probability gate is needed.
    """
    if max_delay <= 0:
        return torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)

    kwargs = {"device": device}
    if generator is not None:
        kwargs["generator"] = generator

    delays = torch.randint(0, max_delay + 1, (batch_size,), **kwargs)
    positions = torch.arange(horizon, device=device).unsqueeze(0)  # (1, H)
    return positions < delays.unsqueeze(1)  # (B, H)


def build_per_token_time(
    scalar_tau: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct per-token time with t=1 on the prefix, scalar τ elsewhere.

    Args:
        scalar_tau: ``(B,)`` per-sample noise time.
        prefix_mask: ``(B, H)`` boolean mask marking clean-prefix positions.
    Returns:
        ``(B, H)`` per-token time tensor.
    """
    ones = torch.ones_like(prefix_mask, dtype=scalar_tau.dtype)
    tau_bh = scalar_tau.unsqueeze(1).expand_as(prefix_mask).to(dtype=scalar_tau.dtype)
    return torch.where(prefix_mask, ones, tau_bh)


def build_noisy_trajectory_rtc(
    actions: torch.Tensor,
    noise: torch.Tensor,
    t_per_token: torch.Tensor,
) -> torch.Tensor:
    """Flow-matching linear-interpolation noise schedule with per-token t.

    ``x_t = t * action + (1 - t) * noise`` so that prefix positions
    (t = 1) stay at the ground-truth clean action.

    Args:
        actions: ``(B, H, D)``.
        noise: ``(B, H, D)``.
        t_per_token: ``(B, H)`` time per action step.
    """
    t_exp = t_per_token.unsqueeze(-1).to(dtype=actions.dtype)
    return t_exp * actions + (1.0 - t_exp) * noise


def compute_soft_mask_weights(
    horizon: int,
    d: torch.Tensor | int,
    s: torch.Tensor | int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eq. 5 of arXiv 2506.07339 — soft mask weights W_i.

    The three regions (frozen / ramp / free) are produced vectorised so that
    per-sample ``d`` and ``s`` are supported. Returns ``(B, H)``.
    """
    if isinstance(d, int):
        d = torch.full((1,), d, device=device)
    if isinstance(s, int):
        s = torch.full((1,), s, device=device)
    if d.dim() == 0:
        d = d.unsqueeze(0)
    if s.dim() == 0:
        s = s.unsqueeze(0)
    B = int(max(d.shape[0], s.shape[0]))
    if d.shape[0] == 1 and B > 1:
        d = d.expand(B)
    if s.shape[0] == 1 and B > 1:
        s = s.expand(B)

    d = d.to(device=device, dtype=torch.long)
    s = s.to(device=device, dtype=torch.long)
    positions = torch.arange(horizon, device=device).unsqueeze(0).expand(B, -1)  # (B, H)

    # Regions
    frozen = positions < d.unsqueeze(1)  # i < d
    free = positions >= (horizon - s).unsqueeze(1)  # i >= H - s
    ramp = (~frozen) & (~free)  # d <= i < H - s

    # c_i = (H - s - i) / (H - s - d + 1)
    numer = (horizon - s).unsqueeze(1) - positions  # (B, H)
    denom = (horizon - s - d + 1).unsqueeze(1).clamp_min(1)  # avoid /0 when s+d=H+1
    c = numer.to(dtype=dtype) / denom.to(dtype=dtype)
    # c_i should be in [0, 1] within ramp band; clamp for numerical safety
    c = c.clamp(min=0.0, max=1.0)

    ramp_val = c * (torch.exp(c) - 1.0) / (math.e - 1.0)

    W = torch.zeros(B, horizon, device=device, dtype=dtype)
    W = torch.where(frozen, torch.ones_like(W), W)
    W = torch.where(ramp, ramp_val, W)
    return W


def guidance_scale(tau: float, beta: float) -> float:
    """min(β, (1-τ)/(τ·r_τ²)) per Eq. 2 + Eq. 4 of arXiv 2506.07339.

    Algebraic equivalent: min(β, (τ²+(1-τ)²)/(τ(1-τ))).
    """
    t = min(max(tau, _EPS_TAU), 1.0 - _EPS_TAU)
    inv_r2 = (t * t + (1.0 - t) * (1.0 - t)) / ((1.0 - t) * (1.0 - t))
    c = (1.0 - t) / t
    return float(min(c * inv_r2, beta))


__all__ = [
    "GuidedRtcConfig",
    "InferenceRtc",
    "RtcActions",
    "RtcInput",
    "RtcTiming",
    "TrainedRtcConfig",
    "rtc_timing",
    "sample_training_prefix",
    "build_per_token_time",
    "build_noisy_trajectory_rtc",
    "compute_soft_mask_weights",
    "guidance_scale",
]
