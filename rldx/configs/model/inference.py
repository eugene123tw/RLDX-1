# SPDX-License-Identifier: Apache-2.0
"""How to run a checkpoint now. Kept apart from RLDXConfig, which says what the weights are."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TypeVar
import warnings

from .rldx import RLDXConfig


T = TypeVar("T")


def _check_exec_horizon(exec_horizon: int | None) -> None:
    if exec_horizon is None or exec_horizon < 1:
        raise ValueError(
            f"RTC needs the actions the robot executes per chunk: exec_horizon={exec_horizon}"
        )


@dataclass(frozen=True)
class TrainedRtcConfig:
    """Inpaint the prefix at t=1 (2512.05964); needs a checkpoint trained with RTC."""

    delay: int  # frozen prefix carried over from the previous chunk
    exec_horizon: int  # actions the robot executes per chunk

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError(f"RTC needs a frozen prefix: delay={self.delay}")
        _check_exec_horizon(self.exec_horizon)


@dataclass(frozen=True)
class GuidedRtcConfig:
    """Jacobian guidance toward the prefix (2506.07339); runs on any checkpoint."""

    delay: int  # frozen prefix carried over from the previous chunk
    exec_horizon: int  # actions the robot executes per chunk
    # The unclipped Eq. 2 + Eq. 4 scale has minimum 2 at τ=0.5.
    # β <= 2 makes the clipped scale constant; β=5 retains its τ dependence.
    jacobian_beta: float = 5.0
    # Guide only the first N denoising steps; autograd through the DiT costs
    # several times the inference memory. None = every step.
    jacobian_steps_only: int | None = 3

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError(f"RTC needs a frozen prefix: delay={self.delay}")
        _check_exec_horizon(self.exec_horizon)


InferenceRtc = TrainedRtcConfig | GuidedRtcConfig


@dataclass(frozen=True)
class InferenceConfig:
    # Overrides of what the checkpoint was trained with; None keeps the checkpoint's value.
    deactivate_memory: bool = False
    mask_padded_action_dims: bool | None = None
    # Settings a checkpoint does not carry.
    rtc: InferenceRtc | None = None
    sample_timestep_from_beta_dist: bool = False
    denoising_timesteps: tuple[float, ...] | None = None

    def model_config(self, checkpoint: RLDXConfig) -> RLDXConfig:
        if self.rtc is None and checkpoint.rtc_training_max_delay > 0:
            warnings.warn(
                f"checkpoint was trained with RTC (rtc_training_max_delay="
                f"{checkpoint.rtc_training_max_delay}) but serves without it; pass "
                "rtc:trained if this was not intended",
                stacklevel=2,
            )
        if not (self.deactivate_memory and checkpoint.memory is not None):
            return checkpoint
        config = copy.deepcopy(checkpoint)
        config.memory = None
        return config

    def modality_configs(self, saved: dict, model: RLDXConfig) -> dict:
        physics_keys = [] if model.physics is None else model.physics.keys
        served = {
            key: copy.deepcopy(modality)
            for key, modality in saved.items()
            if key in _CORE_MODALITIES or key in physics_keys
        }
        for key in physics_keys:
            if key not in served:
                continue
            history = [d for d in served[key].delta_indices if d <= 0]
            if history:
                served[key].delta_indices = history
            else:
                del served[key]
        if model.memory is not None:
            served["video"].delta_indices = (
                recurrent_video_deltas(model.video_length, model.video_stride)
                if model.use_video
                else [0]
            )
        return served

    def processor_config(self, model: RLDXConfig, served: dict) -> RLDXConfig:
        physics = model.physics
        if physics is None:
            return model
        kept = [(k, d) for k, d in zip(physics.keys, physics.dims) if k in served]
        if len(kept) == len(physics.keys):
            return model
        config = copy.deepcopy(model)
        config.physics = (
            replace(physics, keys=[k for k, _ in kept], dims=[d for _, d in kept]) if kept else None
        )
        return config


_CORE_MODALITIES = ("video", "state", "action", "language")


def recurrent_video_deltas(video_length: int, video_stride: int) -> list[int]:
    """Frame deltas for recurrent video inference, laid out like ``features/video.py``."""
    if video_length < 1:
        raise ValueError(f"video_length must be >= 1, got {video_length}")
    if video_stride < 1:
        raise ValueError(f"video_stride must be >= 1, got {video_stride}")
    return [(i - (video_length - 1)) * video_stride for i in range(video_length)]


def overridden(checkpoint: T, override: T | None) -> T:
    return checkpoint if override is None else override
