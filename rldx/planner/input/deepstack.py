"""Host-planned visual rows in the cognition-extended language sequence."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rldx.planner.input.positioning import extend_for_cog_tokens
from rldx.planner.input.wire import parse_mapping, parse_tensor
from rldx.planner.tree import register_mapping


@register_mapping
@dataclass(frozen=True, init=False)
class DeepStackPlan(Mapping[str, Any]):
    visual_rows: torch.Tensor

    def __init__(self, values: Mapping[str, Any] | None = None, **fields: Any) -> None:
        source = values if values is not None else fields
        object.__setattr__(self, "visual_rows", source["visual_rows"])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "DeepStackPlan":
        raw = parse_mapping(value, "deepstack_plan")
        return cls(
            visual_rows=parse_tensor(
                raw["visual_rows"], "deepstack_plan.visual_rows", shape=(None,), dtype=torch.int64
            )
        )


def compute_deepstack_plan(
    input_ids: torch.Tensor, n_cog_tokens: int, image_token_id: int
) -> DeepStackPlan:
    if input_ids.device.type != "cpu":
        raise ValueError("DeepStack must be planned from host input_ids")
    extended_ids, _ = extend_for_cog_tokens(input_ids, None, n_cog_tokens)
    return DeepStackPlan(
        visual_rows=extended_ids.eq(image_token_id).flatten().nonzero(as_tuple=True)[0]
    )
