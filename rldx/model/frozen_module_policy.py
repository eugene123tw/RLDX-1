"""Training-runtime policy for frozen subtree modes."""

from __future__ import annotations

from typing import Any

from torch import nn


def _unwrap_model(model: nn.Module) -> Any:
    """Unwrap common distributed containers without guessing recursively."""
    wrapped = getattr(model, "module", None)
    return wrapped if wrapped is not None else model


class FrozenModulePolicy:
    """Restore configured frozen subtrees after a runtime enters train mode."""

    @staticmethod
    def supports(model: nn.Module) -> bool:
        root = _unwrap_model(model)
        return hasattr(root, "backbone") and hasattr(root, "action_model")

    @classmethod
    def apply_training(cls, model: nn.Module) -> None:
        root = _unwrap_model(model)
        if not cls.supports(root):
            raise TypeError("FrozenModulePolicy requires an RLDX model")
        if not root.training:
            raise ValueError("FrozenModulePolicy.apply_training requires a model in train mode")

        root.backbone.restore_frozen_modes()
        root.action_model.restore_frozen_modes()
