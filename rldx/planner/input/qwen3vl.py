"""Model-aware host input planning for Qwen3-VL."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from rldx.planner.input.contracts import PreparedQwen3VLBatch, Qwen3VLHostLayout
from rldx.planner.input.deepstack import compute_deepstack_plan
from rldx.planner.input.positioning import extend_for_cog_tokens, get_rope_index
from rldx.planner.input.spec import Qwen3VLInputSpec
from rldx.planner.input.text_attention import compute_text_attention_plan
from rldx.planner.input.token_compression import PreparedCompression, compute_token_compression_plan
from rldx.planner.input.vision_layout import compute_vision_layout_plan
from rldx.planner.lru import lru_get
from rldx.planner.token_layout import TokenBatchLayout
from rldx.planner.tree import tensor_map


LAYOUT_KEYS = (
    "input_ids",
    "attention_mask",
    "image_grid_thw",
    "video_grid_thw",
    "image_wise_encoding",
    "num_views",
    "num_frames",
)
CONTROL_KEYS = ("image_wise_encoding", "num_views", "num_frames")


def _cast_floating_inputs(inputs: Mapping[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    return tensor_map(lambda value: value.to(dtype) if value.is_floating_point() else value, inputs)


def _position_ids(
    inputs: Mapping[str, Any],
    extended_ids: torch.Tensor,
    extended_mask: torch.Tensor,
    spec: Qwen3VLInputSpec,
) -> torch.Tensor:
    if "video_grid_thw" in inputs:
        raise ValueError("Qwen3VLInputPlanner does not support video_grid_thw")
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None and image_grid_thw.ndim == 3:
        image_grid_thw = image_grid_thw.reshape(-1, 3)
    position_ids, _ = get_rope_index(
        extended_ids,
        image_grid_thw,
        None,
        extended_mask,
        spatial_merge_size=spec.spatial_merge_size,
        image_token_id=spec.image_token_id,
        video_token_id=spec.video_token_id,
        vision_start_token_id=spec.vision_start_token_id,
    )
    return position_ids


def _validate_image_placeholders(
    input_ids: torch.Tensor, vision_layout: Mapping[str, Any], image_token_id: int
) -> None:
    """Validate the image-token/layout contract while inputs are still on the host."""
    if input_ids.device.type != "cpu":
        raise ValueError("image placeholder counts must be validated from host input_ids")
    actual = int((input_ids == image_token_id).sum())
    expected = sum(vision_layout["split_sizes"])
    if actual != expected:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {actual}, "
            f"expected features: {expected}"
        )


def _uniform_host_value(value: Any, name: str, cast_type: type, batch_size: int) -> Any:
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be resolved from a host tensor")
        values = value.reshape(-1)
        if values.numel() != batch_size:
            raise ValueError(f"{name} must contain one value per sample")
        if not torch.all(values == values[0]):
            raise ValueError(f"prepared Qwen3-VL batch requires uniform {name}")
        value = values[0]
    return cast_type(value)


def _per_sample_host_values(
    value: Any, name: str, cast_type: type, batch_size: int
) -> tuple[Any, Any]:
    """Return the per-sample value and its legacy first-sample execution scalar."""
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be resolved from a host tensor")
        values = value.reshape(-1)
        if values.numel() != batch_size:
            raise ValueError(f"{name} must contain one value per sample")
        per_sample = [cast_type(item) for item in values]
        return per_sample, per_sample[0]
    scalar = cast_type(value)
    return scalar, scalar


def _token_compression(
    inputs: Mapping[str, Any], spec: Qwen3VLInputSpec
) -> tuple[PreparedCompression, Qwen3VLHostLayout]:
    batch_size = inputs["input_ids"].shape[0]
    image_wise = inputs.get("image_wise_encoding")
    if image_wise is not None:
        image_wise = _uniform_host_value(image_wise, "image_wise_encoding", bool, batch_size)

    raw_num_views = inputs.get("num_views")
    num_views = None
    execution_num_views = None
    if raw_num_views is not None:
        num_views, execution_num_views = _per_sample_host_values(
            raw_num_views, "num_views", int, batch_size
        )
        if spec.motion_layout == "uniform":
            execution_num_views = _uniform_host_value(raw_num_views, "num_views", int, batch_size)
    compression_views = num_views
    if not image_wise:
        compression_views = None

    plan, attention_mask = compute_token_compression_plan(
        inputs["input_ids"],
        inputs["attention_mask"],
        spec.n_cog_tokens,
        spec.compression_pattern_token_id,
        compression_views,
    )
    compression = PreparedCompression(plan, attention_mask)
    host_layout: Qwen3VLHostLayout = {}
    if image_wise is not None:
        host_layout["image_wise_encoding"] = image_wise
    if execution_num_views is not None:
        host_layout["num_views"] = execution_num_views
    raw_num_frames = inputs.get("num_frames")
    if raw_num_frames is not None:
        _, execution_num_frames = _per_sample_host_values(
            raw_num_frames, "num_frames", int, batch_size
        )
        if spec.motion_layout == "uniform":
            execution_num_frames = _uniform_host_value(
                raw_num_frames, "num_frames", int, batch_size
            )
        host_layout["num_frames"] = execution_num_frames
    return compression, host_layout


def _fingerprint(layout_inputs: Mapping[str, Any]) -> tuple:
    def value_key(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.dtype, tuple(value.shape), value.numpy().tobytes()
        return value

    return tuple((key, value_key(value)) for key, value in sorted(layout_inputs.items()))


@dataclass(frozen=True)
class Qwen3VLInputPlanner:
    """Derive a complete device-transfer-ready Qwen3-VL input on the host."""

    dtype: torch.dtype
    spec: Qwen3VLInputSpec
    text_attention: bool = False
    memo_size: int = 0
    _memo: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False, compare=False)
    _vision_memo: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False, compare=False
    )

    def __call__(self, batch: Any) -> PreparedQwen3VLBatch:
        if not isinstance(batch, Mapping):
            raise TypeError(f"collated batch must be a Mapping, got {type(batch).__name__}")
        inputs = batch.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError(f"batch.inputs must be a Mapping, got {type(inputs).__name__}")
        planned_inputs = dict(_cast_floating_inputs(inputs, self.dtype))
        layout_inputs = {key: planned_inputs[key] for key in LAYOUT_KEYS if key in planned_inputs}
        planned_inputs.update(self._layout(layout_inputs))
        for control_key in CONTROL_KEYS:
            planned_inputs.pop(control_key, None)
        return cast(PreparedQwen3VLBatch, {**batch, "inputs": planned_inputs})

    def _layout(self, layout_inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.memo_size:
            return self._derive_layout(layout_inputs)
        layout = lru_get(
            self._memo,
            _fingerprint(layout_inputs),
            self.memo_size,
            lambda: self._derive_layout(layout_inputs),
        )
        return {**layout, "host_layout": dict(layout["host_layout"])}

    def _vision_layout(self, grid: torch.Tensor) -> Any:
        def plan():
            return compute_vision_layout_plan(
                grid, self.spec.spatial_merge_size, max_seqlen=self.spec.vision_max_seqlen
            )

        if not self.memo_size:
            return plan()
        return lru_get(self._vision_memo, tuple(map(tuple, grid.tolist())), self.memo_size, plan)

    def _derive_layout(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layout: dict[str, Any] = {}
        extended_ids, extended_mask = extend_for_cog_tokens(
            inputs["input_ids"], inputs["attention_mask"], self.spec.n_cog_tokens
        )
        layout["position_ids"] = _position_ids(inputs, extended_ids, extended_mask, self.spec)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            vision_layout = self._vision_layout(image_grid_thw.reshape(-1, 3))
            _validate_image_placeholders(
                inputs["input_ids"], vision_layout, self.spec.image_token_id
            )
            layout["vision_layout"] = vision_layout
        layout["deepstack_plan"] = compute_deepstack_plan(
            inputs["input_ids"], self.spec.n_cog_tokens, self.spec.image_token_id
        )
        compression, host_layout = _token_compression(inputs, self.spec)
        token_layout = TokenBatchLayout.parse(inputs["attention_mask"])
        host_layout["token_layout"] = token_layout
        if token_layout.all_valid:
            if not self.text_attention:
                layout["attention_mask"] = None
            compression = PreparedCompression(
                compression.plan, torch.ones_like(compression.attention_mask)
            )
        layout["compression"] = compression
        if self.text_attention:
            layout["text_attention"] = compute_text_attention_plan(
                extended_mask, compression.attention_mask
            )
        layout["host_layout"] = host_layout
        return layout
