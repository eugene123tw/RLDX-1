"""RLDX — RLWRLD RLDX-1 Vision-Language-Action model.

Public symbols are exposed as lazy attributes (``rldx.RLDX``,
``rldx.RLDXConfig``, ``rldx.RLDXProcessor``, ...), resolved on first access.

Keep this file import-free. Python runs ``__init__.py`` before any sub-module,
so an import here also reaches the sim venvs, which install ``rldx`` with
``--no-deps`` and carry no model stack.
"""

from importlib import import_module
from typing import TYPE_CHECKING


__version__ = "1.1.0"

_LAZY_ATTRS = {
    "Config": ("rldx.configs.base_config", "Config"),
    "RLDXConfig": ("rldx.configs.model.rldx", "RLDXConfig"),
    "TrainArgs": ("rldx.configs.training.cli", "TrainArgs"),
    "EmbodimentTag": ("rldx.data.embodiment_tags", "EmbodimentTag"),
    "RLDXProcessor": ("rldx.model.core.processor", "RLDXProcessor"),
    "RLDX": ("rldx.model.core.rldx", "RLDX"),
    "RLDXPolicy": ("rldx.serving.policy", "RLDXPolicy"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr = _LAZY_ATTRS[name]
        value = getattr(import_module(module_path), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rldx' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = ["__version__", *sorted(_LAZY_ATTRS)]


if TYPE_CHECKING:
    from rldx.configs.base_config import Config
    from rldx.configs.model.rldx import RLDXConfig
    from rldx.configs.training.cli import TrainArgs
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.model.core.processor import RLDXProcessor
    from rldx.model.core.rldx import RLDX
    from rldx.serving.policy import RLDXPolicy
