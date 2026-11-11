"""Pure host-side planning for Qwen token compression."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rldx.planner.input.positioning import extend_for_cog_tokens
from rldx.planner.input.wire import parse_mapping, parse_positive_int, parse_tensor
from rldx.planner.tree import register_mapping


@register_mapping
@dataclass(frozen=True, init=False)
class TokenCompressionPlan(Mapping[str, Any]):
    gather_index: torch.Tensor
    pad_mask: torch.Tensor
    motion_slot: torch.Tensor
    begin: torch.Tensor
    end: torch.Tensor
    compress: torch.Tensor
    extended_len: int

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
    def from_wire(cls, value: Mapping[str, Any]) -> "TokenCompressionPlan":
        """Restore compression fields without reading or recomputing device indices."""
        raw = parse_mapping(value, "compression.plan")
        extended_len = parse_positive_int(raw["extended_len"], "compression.plan.extended_len")
        gather_index = parse_tensor(
            raw["gather_index"],
            "compression.plan.gather_index",
            shape=(None, None),
            dtype=torch.int64,
        )
        batch_size, width = gather_index.shape
        if batch_size <= 0 or not 0 < width <= extended_len:
            raise ValueError("compression.plan.gather_index has invalid batch or sequence extent")
        fields = {"gather_index": gather_index, "extended_len": extended_len}
        for name, shape, dtype in (
            ("pad_mask", (batch_size, width), torch.bool),
            ("motion_slot", (batch_size, width), torch.bool),
            ("begin", (batch_size,), torch.int64),
            ("end", (batch_size,), torch.int64),
            ("compress", (batch_size,), torch.bool),
        ):
            fields[name] = parse_tensor(
                raw[name], f"compression.plan.{name}", shape=shape, dtype=dtype
            )
        return cls(fields)


@register_mapping
@dataclass(frozen=True, init=False)
class PreparedCompression(Mapping[str, Any]):
    plan: TokenCompressionPlan
    attention_mask: torch.Tensor

    def __init__(
        self,
        values: Mapping[str, Any] | TokenCompressionPlan,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        if isinstance(values, Mapping) and attention_mask is None:
            source = values
        else:
            source = {"plan": values, "attention_mask": attention_mask}
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, source[name])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "PreparedCompression":
        raw = parse_mapping(value, "compression")
        plan = TokenCompressionPlan.from_wire(raw["plan"])
        attention_mask = parse_tensor(
            raw["attention_mask"],
            "compression.attention_mask",
            shape=tuple(plan.gather_index.shape),
        )
        return cls(plan, attention_mask)


def compute_token_compression_plan(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_cog_tokens: int,
    pattern_token_id: int,
    num_views: int | list[int] | None = None,
) -> tuple[TokenCompressionPlan, torch.Tensor]:
    """Build the layer-4 compression layout using host tensors only."""
    extended_ids, extended_mask = extend_for_cog_tokens(input_ids, attention_mask, n_cog_tokens)
    if extended_ids.device.type != "cpu":
        raise ValueError("token compression plan must be computed on host tensors")
    if extended_mask.ndim != 2:
        raise ValueError("token compression plan requires a 2D attention_mask")

    bsz, seq_len = extended_ids.shape
    matches = extended_ids.eq(pattern_token_id)
    begin = torch.empty(bsz, dtype=torch.long)
    end = torch.empty(bsz, dtype=torch.long)
    views_per_sample = None
    if num_views is not None:
        views_per_sample = (
            [int(v) for v in num_views] if not isinstance(num_views, int) else [num_views] * bsz
        )
        if len(views_per_sample) != bsz:
            raise ValueError("num_views must be an int or one value per sample")
    for batch_idx in range(bsz):
        match_indices = torch.nonzero(matches[batch_idx], as_tuple=False).squeeze(-1)
        if match_indices.numel() == 0:
            raise ValueError("token compression plan requires an image token in every sample")
        begin[batch_idx] = match_indices[0]
        if views_per_sample is not None:
            views = views_per_sample[batch_idx]
            if views < 1 or match_indices.numel() < views:
                raise ValueError("num_views exceeds the image-token matches in a sample")
            end[batch_idx] = match_indices[-views]
        else:
            end[batch_idx] = match_indices[-1]

    compress = end > begin
    new_len = seq_len - (end - begin) + compress.to(torch.long)
    out_len = int(new_len.max())
    gather_index = torch.zeros((bsz, out_len), dtype=torch.long)
    pad_mask = torch.ones((bsz, out_len), dtype=torch.bool)
    motion_slot = torch.zeros((bsz, out_len), dtype=torch.bool)
    compressed_mask = torch.zeros((bsz, out_len), dtype=extended_mask.dtype)

    for batch_idx in range(bsz):
        front = torch.arange(begin[batch_idx])
        back = torch.arange(end[batch_idx], seq_len)
        source = (
            torch.cat((front, begin[batch_idx : batch_idx + 1], back))
            if compress[batch_idx]
            else torch.cat((front, back))
        )
        offset = out_len - source.numel()
        gather_index[batch_idx, offset:] = source
        pad_mask[batch_idx, offset:] = False
        if compress[batch_idx]:
            motion_slot[batch_idx, offset + front.numel()] = True
        compressed_mask[batch_idx, offset:] = extended_mask[batch_idx].gather(0, source)
        if compress[batch_idx]:
            compressed_mask[batch_idx, offset + front.numel()] = 1

    plan = TokenCompressionPlan(
        gather_index=gather_index,
        pad_mask=pad_mask,
        motion_slot=motion_slot,
        begin=begin,
        end=end,
        compress=compress,
        extended_len=seq_len,
    )
    return plan, compressed_mask
