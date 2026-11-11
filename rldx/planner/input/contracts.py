"""Lightweight typing for planner-complete Qwen3-VL inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, cast

import torch

from rldx.planner.input.deepstack import DeepStackPlan
from rldx.planner.input.text_attention import TextAttentionPlan
from rldx.planner.input.token_compression import PreparedCompression
from rldx.planner.input.vision_layout import VisionLayoutPlan
from rldx.planner.input.wire import parse_mapping
from rldx.planner.token_layout import TokenBatchLayout


class Qwen3VLHostLayout(TypedDict, total=False):
    image_wise_encoding: bool
    num_views: int
    num_frames: int
    token_layout: TokenBatchLayout


class PreparedQwen3VLInputs(TypedDict, total=False):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    pixel_values: torch.Tensor
    pixel_values_videos: torch.Tensor
    image_grid_thw: torch.Tensor
    video_grid_thw: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Any
    inputs_embeds: torch.Tensor
    cache_position: torch.Tensor
    compression: PreparedCompression
    text_attention: TextAttentionPlan
    host_layout: Qwen3VLHostLayout
    vision_layout: VisionLayoutPlan
    deepstack_plan: DeepStackPlan


class PreparedQwen3VLBatch(TypedDict):
    inputs: PreparedQwen3VLInputs


def parse_routed_batch(value: Mapping[str, Any]) -> PreparedQwen3VLBatch:
    """Restore planner records at the routing boundary, preserving all batch fields.

    Routing has consumed host_layout.token_layout before sending the batch.
    Tensor contents are trusted planner outputs; parsers only read host metadata
    and tensor descriptions, retaining the received storage and device.
    """
    batch = parse_mapping(value, "batch")
    inputs = parse_mapping(batch["inputs"], "batch.inputs")
    parsed = dict(inputs)
    parsed["deepstack_plan"] = DeepStackPlan.from_wire(inputs["deepstack_plan"])
    parsed["compression"] = PreparedCompression.from_wire(inputs["compression"])
    if "text_attention" in inputs:
        parsed["text_attention"] = TextAttentionPlan.from_wire(inputs["text_attention"])
    if inputs.get("image_grid_thw") is not None or "vision_layout" in inputs:
        parsed["vision_layout"] = VisionLayoutPlan.from_wire(inputs["vision_layout"])
    return cast(PreparedQwen3VLBatch, {**batch, "inputs": parsed})


def project_qwen3vl_inputs(inputs: Mapping[str, Any]) -> PreparedQwen3VLInputs:
    """Select model-facing Qwen3-VL fields from the composite input mapping."""
    fields = PreparedQwen3VLInputs.__annotations__
    return cast(PreparedQwen3VLInputs, {key: inputs[key] for key in fields if key in inputs})
