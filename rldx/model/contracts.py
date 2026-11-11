"""Tensor contracts at the backbone/action-model boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, TypedDict

import torch

from rldx.model.modules.action_model.rtc import RtcInput
from rldx.planner.input.contracts import PreparedQwen3VLInputs


class ActionModelInput(TypedDict, total=False):
    state: torch.Tensor
    embodiment_id: torch.Tensor
    action: torch.Tensor
    action_mask: torch.Tensor
    action_dim_mask: torch.Tensor
    physics: torch.Tensor
    physics_mask: torch.Tensor
    rtc: RtcInput
    reset_memory: bool


class TrainingOutput(NamedTuple):
    loss: torch.Tensor
    action_loss: torch.Tensor
    physics_loss: torch.Tensor | None = None


class ActionOutput(NamedTuple):
    action_pred: torch.Tensor
    memory: torch.Tensor | None = None


class RLDXInputBatch(ActionModelInput, PreparedQwen3VLInputs, total=False):
    """Semantically processed inputs accepted by the RLDX model boundary."""


def validate_rldx_input_batch(inputs: Mapping[str, Any]) -> None:
    """Reject raw processor inputs at the model boundary.

    Detailed backbone and action invariants remain owned by their respective
    consumers; this boundary only distinguishes processed model input from raw
    multimodal content.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError(f"RLDX input batch must be a mapping, got {type(inputs).__name__}")
    if "vlm_content" in inputs:
        raise ValueError(
            "RLDX expects processed inputs; pass vlm_content through the processor "
            "and collator in the calling runtime"
        )
