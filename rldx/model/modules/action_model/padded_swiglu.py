"""RLDX-specific padded execution form of the upstream SwiGLU block."""

import torch
from torch import nn

from rldx.configs.execution import DEFAULT_HIDDEN_ALIGNMENT
from rldx.model.modules.action_model.ops import SwiGLUFFN
from rldx.model.modules.hidden_padding import logical_rows, pad_hidden


def canonicalize_padded_swiglu_checkpoint(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return canonical checkpoint tensors for every padded SwiGLU module."""
    canonical = state_dict.copy()
    for module_name, module in model.named_modules():
        if not isinstance(module, PaddedSwiGLUFFN):
            continue
        logical = module.logical_hidden_features
        prefix = f"{module_name}." if module_name else ""
        shapes = {
            f"{prefix}w12.weight": ((2 * logical, module.w12.in_features), module.w12.weight.shape),
            f"{prefix}w3.weight": ((module.w3.out_features, logical), module.w3.weight.shape),
        }
        if module.w12.bias is not None:
            shapes[f"{prefix}w12.bias"] = ((2 * logical,), module.w12.bias.shape)

        for key, (canonical_shape, runtime_shape) in shapes.items():
            value = canonical.get(key)
            if value is None or value.shape == canonical_shape:
                continue
            if value.shape != runtime_shape:
                raise ValueError(
                    f"invalid padded SwiGLU checkpoint shape for {key}: "
                    f"got {list(value.shape)}, expected canonical {list(canonical_shape)} "
                    f"or runtime {list(runtime_shape)}"
                )
            if key.endswith("w3.weight"):
                canonical[key] = value[:, :logical]
            else:
                canonical[key] = logical_rows(value, logical, 2)
    return canonical


class PaddedSwiGLUFFN(SwiGLUFFN):
    """Pad the hidden width to an alignment while preserving canonical checkpoints."""

    def __init__(self, *args, alignment: int = DEFAULT_HIDDEN_ALIGNMENT, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pad_hidden(self, "w12", "w3", alignment, chunks=2)

    @property
    def logical_hidden_features(self) -> int:
        return self.hidden_layout.logical

    @property
    def runtime_hidden_features(self) -> int:
        return self.w3.in_features

    def zero_padding_(self) -> None:
        """Restore zero padding after model-wide parameter initialization."""
        if self.runtime_hidden_features == self.logical_hidden_features:
            return
        logical_width = self.logical_hidden_features
        with torch.no_grad():
            self.w12.weight[logical_width : self.runtime_hidden_features].zero_()
            self.w12.weight[
                self.runtime_hidden_features + logical_width : 2 * self.runtime_hidden_features
            ].zero_()
            if self.w12.bias is not None:
                self.w12.bias[logical_width : self.runtime_hidden_features].zero_()
                self.w12.bias[
                    self.runtime_hidden_features + logical_width : 2 * self.runtime_hidden_features
                ].zero_()
            self.w3.weight[:, logical_width:].zero_()


def restore_padded_swiglu_invariants(model: nn.Module) -> None:
    """Restore padding after all RLDX model-specific initializers have run."""
    for module in model.modules():
        if isinstance(module, PaddedSwiGLUFFN):
            module.zero_padding_()
