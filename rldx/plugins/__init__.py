# SPDX-License-Identifier: Apache-2.0
"""Optional packages that extend the built-in registries.

Each plugin is a subpackage here, loaded only when named in ``RLDX_PLUGINS``
(comma-separated). A registry imports only its own role file (``tags``,
``slots``, ``mixes``, ``presets``) and calls into it, never the plugin as a
whole: ``EmbodimentTag`` is fixed when its module finishes importing, and later
roles import it.
"""

import importlib
import importlib.util
import os
import pkgutil
from types import ModuleType


ENV = "RLDX_PLUGINS"


def enabled() -> list[str]:
    """Plugins named in ``RLDX_PLUGINS``, refusing a name with no package here."""
    names = sorted({n.strip() for n in os.environ.get(ENV, "").split(",") if n.strip()})
    available = {m.name for m in pkgutil.iter_modules(__path__) if m.ispkg}
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"{ENV} names unknown plugins {unknown}; available: {sorted(available)}")
    return names


def modules(role: str) -> list[ModuleType]:
    """Each enabled plugin's ``role`` submodule, in plugin-name order."""
    found = []
    for plugin in enabled():
        name = f"{__name__}.{plugin}.{role}"
        if importlib.util.find_spec(name) is not None:
            found.append(importlib.import_module(name))
    return found


def add_new(target: dict, entries: dict, what: str) -> None:
    """Merge ``entries`` into ``target``, refusing to replace a key."""
    clash = sorted(set(target) & set(entries))
    if clash:
        raise ValueError(f"{what} already defined: {clash}")
    target.update(entries)


def replace_existing(target: dict, entries: dict, what: str) -> None:
    """Overwrite keys of ``target`` with ``entries``, refusing to add a key."""
    absent = sorted(set(entries) - set(target))
    if absent:
        raise ValueError(f"{what} not defined, cannot replace: {absent}")
    target.update(entries)
