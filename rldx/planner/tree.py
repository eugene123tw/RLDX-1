"""Nested inputs as structure plus leaves, with one registry of how each container opens."""

from collections.abc import Mapping
from typing import Any, Callable

import optree
import torch
from transformers.feature_extraction_utils import BatchFeature


NAMESPACE = "rldx"


def register_mapping(cls: type) -> type:
    optree.register_pytree_node(
        cls,
        lambda mapping: (list(mapping.values()), tuple(mapping.keys())),
        lambda keys, values: cls(dict(zip(keys, values))),
        namespace=NAMESPACE,
    )
    return cls


register_mapping(BatchFeature)


def flatten(tree: Any) -> tuple[list[Any], optree.PyTreeSpec]:
    leaves, spec = optree.tree_flatten(tree, namespace=NAMESPACE)
    for leaf in leaves:
        if isinstance(leaf, Mapping):
            raise TypeError(f"{type(leaf).__name__} is a mapping without a tree registration")
    return leaves, spec


def tensors(tree: Any) -> list[torch.Tensor]:
    return [leaf for leaf in flatten(tree)[0] if isinstance(leaf, torch.Tensor)]


def tensor_map(fn: Callable[[torch.Tensor], Any], tree: Any) -> Any:
    leaves, spec = flatten(tree)
    mapped = [fn(leaf) if isinstance(leaf, torch.Tensor) else leaf for leaf in leaves]
    return spec.unflatten(mapped)
