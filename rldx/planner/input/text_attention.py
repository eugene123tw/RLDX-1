"""Host-planned FlashAttention varlen layouts for the Qwen3-VL text layers."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rldx.planner.input.wire import parse_mapping, parse_tensor
from rldx.planner.tree import register_mapping


@register_mapping
@dataclass(frozen=True, init=False)
class TextAttentionLayout(Mapping[str, Any]):
    cu_seqlens: torch.Tensor

    def __init__(self, values: Mapping[str, Any] | None = None, **fields: Any) -> None:
        source = dict(values or fields)
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, source[name])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> "TextAttentionLayout":
        if mask.device.type != "cpu" or mask.ndim != 2:
            raise ValueError("text attention layout must be planned from a 2D host mask")
        rows, width = mask.shape
        lengths = mask.sum(dim=-1, dtype=torch.int32)
        if not torch.equal(mask.bool(), torch.arange(width) >= width - lengths[:, None]):
            raise ValueError("text attention layout needs a left-padded mask")
        ends = torch.arange(1, rows + 1, dtype=torch.int32) * width
        bounds = torch.stack((ends - lengths, ends), dim=1).flatten()
        return cls(cu_seqlens=F.pad(bounds, (1, 0)))

    @classmethod
    def from_wire(cls, value: Mapping[str, Any], name: str) -> "TextAttentionLayout":
        raw = parse_mapping(value, name)
        return cls(
            cu_seqlens=parse_tensor(
                raw["cu_seqlens"], f"{name}.cu_seqlens", shape=(None,), dtype=torch.int32
            ),
        )


@register_mapping
@dataclass(frozen=True, init=False)
class TextAttentionPlan(Mapping[str, Any]):
    before_compression: TextAttentionLayout
    after_compression: TextAttentionLayout

    def __init__(self, values: Mapping[str, Any] | None = None, **fields: Any) -> None:
        source = dict(values or fields)
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, source[name])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "TextAttentionPlan":
        raw = parse_mapping(value, "text_attention")
        return cls(
            {
                name: TextAttentionLayout.from_wire(raw[name], f"text_attention.{name}")
                for name in cls.__dataclass_fields__
            }
        )


def compute_text_attention_plan(
    extended_mask: torch.Tensor, compressed_mask: torch.Tensor
) -> TextAttentionPlan:
    return TextAttentionPlan(
        before_compression=TextAttentionLayout.from_mask(extended_mask),
        after_compression=TextAttentionLayout.from_mask(compressed_mask),
    )
