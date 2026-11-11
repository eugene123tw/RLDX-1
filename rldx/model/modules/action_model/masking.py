"""Attention-mask helpers for the action model."""

from __future__ import annotations

import torch


def to_additive_attention_mask(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Map visible/non-visible entries to 0/-inf without device scalar tensors."""
    additive_mask = torch.zeros_like(mask, dtype=dtype)
    return additive_mask.masked_fill_(mask == 0, float("-inf"))
