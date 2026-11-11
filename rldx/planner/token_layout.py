"""Host-only token layout facts shared by input and routing planners."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TokenBatchLayout:
    """Per-backbone-row token lengths parsed from one host attention mask."""

    lengths: tuple[int, ...]
    padded_length: int

    def __post_init__(self) -> None:
        if not self.lengths:
            raise ValueError("token layout requires at least one row")
        if self.padded_length <= 0:
            raise ValueError("token layout requires a positive padded length")
        if any(
            type(length) is not int or not 0 <= length <= self.padded_length
            for length in self.lengths
        ):
            raise ValueError("token lengths must be integers within the padded length")

    @classmethod
    def parse(cls, attention_mask: torch.Tensor) -> TokenBatchLayout:
        if not torch.is_tensor(attention_mask) or attention_mask.device.type != "cpu":
            raise ValueError("token layout must be parsed from a host attention_mask tensor")
        if attention_mask.ndim != 2 or 0 in attention_mask.shape:
            raise ValueError("token layout requires a non-empty 2D attention_mask")
        if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise ValueError("attention_mask must contain only zero and one")
        return cls(
            tuple(int(length) for length in attention_mask.sum(dim=1).tolist()),
            int(attention_mask.shape[1]),
        )

    @property
    def row_count(self) -> int:
        return len(self.lengths)

    @property
    def all_valid(self) -> bool:
        return all(length == self.padded_length for length in self.lengths)

    @property
    def valid_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def routing_padded_tokens(self) -> int:
        return self.row_count * max(self.lengths)
