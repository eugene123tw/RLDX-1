"""Host-side layout planning for Qwen3-VL vision inputs."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rldx.planner.input.wire import parse_mapping, parse_positive_int, parse_tensor
from rldx.planner.tree import register_mapping


@register_mapping
@dataclass(frozen=True, init=False)
class VisionLayoutPlan(Mapping[str, Any]):
    shapes: tuple[tuple[int, int, int], ...]
    rotary_position_ids: torch.Tensor
    merge_index: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    split_sizes: tuple[int, ...]

    def __init__(self, values: Mapping[str, Any] | None = None, **fields: Any) -> None:
        source = dict(values or fields)
        source["shapes"] = tuple(tuple(shape) for shape in source["shapes"])
        source["split_sizes"] = tuple(source["split_sizes"])
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, source[name])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "VisionLayoutPlan":
        """Restore the plan representation; index values remain planner-owned."""
        raw = parse_mapping(value, "vision_layout")
        shapes = raw["shapes"]
        if not isinstance(shapes, (tuple, list)) or not shapes:
            raise ValueError("vision_layout.shapes must be a non-empty sequence")
        parsed_shapes = []
        for shape in shapes:
            if not isinstance(shape, (tuple, list)) or len(shape) != 3:
                raise ValueError("vision_layout.shapes entries must have three dimensions")
            parsed_shapes.append(
                tuple(parse_positive_int(dim, "vision dimension") for dim in shape)
            )
        split_sizes = raw["split_sizes"]
        if not isinstance(split_sizes, (tuple, list)) or len(split_sizes) != len(parsed_shapes):
            raise ValueError("vision_layout.split_sizes must have one entry per grid")
        split_sizes = tuple(parse_positive_int(size, "vision split size") for size in split_sizes)
        max_seqlen = parse_positive_int(raw["max_seqlen"], "vision_layout.max_seqlen")
        if max_seqlen < max(h * w for _, h, w in parsed_shapes):
            raise ValueError("vision_layout.max_seqlen is smaller than a frame")
        patches = sum(t * h * w for t, h, w in parsed_shapes)
        frames = sum(t for t, _, _ in parsed_shapes)
        return cls(
            shapes=tuple(parsed_shapes),
            split_sizes=split_sizes,
            max_seqlen=max_seqlen,
            rotary_position_ids=parse_tensor(
                raw["rotary_position_ids"],
                "vision_layout.rotary_position_ids",
                shape=(patches, 2),
                dtype=torch.int64,
            ),
            merge_index=parse_tensor(
                raw["merge_index"],
                "vision_layout.merge_index",
                shape=(patches,),
                dtype=torch.int64,
            ),
            cu_seqlens=parse_tensor(
                raw["cu_seqlens"],
                "vision_layout.cu_seqlens",
                shape=(frames + 1,),
                dtype=torch.int32,
            ),
        )


def compute_vision_layout_plan(
    grid_thw: torch.Tensor, spatial_merge_size: int, *, max_seqlen: int
) -> VisionLayoutPlan:
    """Resolve grid-dependent indexing once, before device transfer."""
    if grid_thw.device.type != "cpu":
        raise ValueError("vision layout must be planned from a host grid")
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [images, 3]")

    shapes = tuple(tuple(int(value) for value in row) for row in grid_thw.tolist())
    if not shapes:
        raise ValueError("image_grid_thw must contain at least one image")

    observed_max_seqlen = max(height * width for _, height, width in shapes)
    if observed_max_seqlen > max_seqlen:
        max_seqlen = 1 << (observed_max_seqlen - 1).bit_length()

    merge = spatial_merge_size
    rotary_parts = []
    merge_parts = []
    patch_offset = 0
    for frames, height, width in shapes:
        if min(frames, height, width) <= 0:
            raise ValueError("vision grid dimensions must be positive")
        if height % merge or width % merge:
            raise ValueError("vision grid dimensions must be divisible by spatial_merge_size")

        rows = torch.arange(height // merge)
        columns = torch.arange(width // merge)
        inner_rows = torch.arange(merge)
        inner_columns = torch.arange(merge)
        row = rows[:, None, None, None] * merge + inner_rows[None, None, :, None]
        column = columns[None, :, None, None] * merge + inner_columns[None, None, None, :]
        positions = torch.stack(
            (
                row.expand(height // merge, width // merge, merge, merge).reshape(-1),
                column.expand(height // merge, width // merge, merge, merge).reshape(-1),
            ),
            dim=-1,
        )
        rotary_parts.append(positions.repeat(frames, 1))

        source = torch.arange(height * width).reshape(height // merge, merge, width // merge, merge)
        source = source.permute(0, 2, 1, 3).reshape(-1) + patch_offset
        merge_parts.append(source.repeat(frames))
        patch_offset += height * width

    frame_lengths = torch.tensor(
        [height * width for frames, height, width in shapes for _ in range(frames)],
        dtype=torch.int32,
    )
    # FlashAttention-2 requires int32 boundaries.  Integer cumsum otherwise
    # promotes int32 inputs to int64, so constrain the accumulator explicitly.
    cu_seqlens = torch.nn.functional.pad(
        frame_lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0
    )
    return VisionLayoutPlan(
        shapes=shapes,
        rotary_position_ids=torch.cat(rotary_parts),
        merge_index=torch.cat(merge_parts),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        split_sizes=tuple(frames * height * width // merge**2 for frames, height, width in shapes),
    )
