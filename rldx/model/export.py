"""Trace-friendly ``RLDX.get_action`` for graph exporters such as OpenVINO.

Like the forward CUDA graphs, one export serves one planned input layout: every non-empty
tensor leaf of the planned batch becomes a graph input, and every other leaf (host layout,
plan sizes, empty tensors) is baked into the graph.
"""

from __future__ import annotations

from typing import Any

import optree
import torch
from torch import nn
import torch.nn.functional as F

from rldx.model.inference_noise import noise_source
from rldx.planner import tree
from rldx.planner.input.tensor_image import TensorImageSpec


def _is_input(leaf: Any) -> bool:
    """Whether a leaf of the planned batch becomes a graph input (otherwise it is baked in)."""
    # Empty tensors carry no data, and OpenVINO constant folding crashes on them as inputs.
    return isinstance(leaf, torch.Tensor) and leaf.numel() > 0


def layout_signature(inputs: dict) -> dict:
    """Structure, baked constants and tensor input shapes that one exported graph serves.

    JSON-serializable, so it is stored next to the exported graph and compared on every request
    by ``graph_inputs``. ``tensors`` also fixes the order (and names) of the graph inputs.
    """
    leaves, spec = tree.flatten(inputs)
    paths = optree.tree_paths(inputs, namespace=tree.NAMESPACE)
    tensors, constants = [], []
    for path, leaf in zip(paths, leaves, strict=True):
        # Dotted pytree path of the leaf, used as the graph input name.
        name = ".".join(str(p) for p in path) or "input"
        if _is_input(leaf):
            tensors.append([name, list(leaf.shape), str(leaf.dtype)])
        elif isinstance(leaf, torch.Tensor):
            constants.append([name, f"empty{list(leaf.shape)} {leaf.dtype}"])
        else:
            constants.append([name, repr(leaf)])
    return {"structure": str(spec), "constants": constants, "tensors": tensors}


def graph_inputs(inputs: dict, layout: dict) -> list[torch.Tensor]:
    """Tensor inputs of ``inputs`` in graph order; raise if its layout is not ``layout``."""
    actual = layout_signature(inputs)
    if actual != layout:
        diffs = [
            f"{key}: exported {layout[key]!r:.300} != request {actual[key]!r:.300}"
            for key in ("structure", "constants", "tensors")
            if actual[key] != layout[key]
        ]
        raise ValueError(
            "request layout differs from the exported graph (instruction in another text "
            "bucket, different camera resolution or batch size?)\n  " + "\n  ".join(diffs)
        )
    return [leaf for leaf in tree.flatten(inputs)[0] if _is_input(leaf)]


class _FirstCallOwnWeight(nn.Module):
    """Linear whose first call per export forward reads its own copy of the weights.

    The denoising loop is unrolled at trace time, so every step's matmul reads the same
    weight constant. This wrapper makes the first step read a separate (but identical)
    constant; ``RLDXExport.forward`` resets ``calls`` so the split is the same on each trace.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.first_weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        self.first_bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        )
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            return F.linear(x, self.first_weight, self.first_bias)
        return self.linear(x)


class RLDXExport(nn.Module):
    """``model.get_action`` over the tensor inputs of one planned layout, plus the initial noise.

    ``forward(*graph_inputs(inputs, self.layout), noise)`` returns the normalized
    ``action_pred``. Build it from a model prepared for eager inference with an SDPA backbone.

    ``inputs`` is a template: its non-input leaves are kept and baked into the traced graph,
    and its tensor leaves only serve as example inputs. Noise is a graph input so callers can
    reproduce torch results with the same noise (parity checks).
    """

    def __init__(
        self, model: nn.Module, inputs: dict, image_spec: TensorImageSpec | None = None
    ):
        super().__init__()
        self.model = model
        self.layout = layout_signature(inputs)
        self._leaves, self._structure = tree.flatten(inputs)
        self._input_index = [i for i, leaf in enumerate(self._leaves) if _is_input(leaf)]
        self._image_spec = image_spec
        self._first_step_linears: list[_FirstCallOwnWeight] = []
        action_model = model.action_model
        self.noise_shape = (
            int(inputs["embodiment_id"].shape[0]),
            action_model.action_horizon,
            action_model.action_dim,
        )

    @property
    def input_names(self) -> list[str]:
        """Graph input names in positional order: the layout's tensor paths, then ``noise``."""
        return [name for name, _, _ in self.layout["tensors"]] + ["noise"]

    def example_inputs(self, noise: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The template's own tensor inputs plus ``noise``, e.g. for tracing."""
        return (*(self._leaves[i] for i in self._input_index), noise)

    def untie_first_denoising_step(self) -> None:
        """Give the first denoising step its own copy of every double block's ``vl_mlp.w3``.

        Works around an OpenVINO GPU float16 bug on weights shared by the unrolled steps
        (physical-ai-studio#1428). Identical constants are stored once, so the IR does not grow.
        """
        if self._first_step_linears:
            return
        for block in self.model.action_model.model.double_blocks:
            block.vl_mlp.w3 = _FirstCallOwnWeight(block.vl_mlp.w3)
            self._first_step_linears.append(block.vl_mlp.w3)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        *tensors, noise = args
        for linear in self._first_step_linears:
            linear.calls = 0
        # Rebuild the planned batch: template leaves with the tensor inputs swapped in.
        leaves = list(self._leaves)
        for index, tensor in zip(self._input_index, tensors, strict=True):
            leaves[index] = tensor
        inputs = self._structure.unflatten(leaves)
        if "raw_images" in inputs:
            # Tensor-image path: uint8 frames are normalized and patchified inside the graph.
            if self._image_spec is None:
                raise ValueError("raw_images inputs need the processor's image_spec")
            inputs["pixel_values"] = self._image_spec.pixel_values(
                self._image_spec.lookup(noise.device),
                inputs.pop("raw_images"),
                inputs.pop("raw_image_order"),
                self.model.dtype,
            )
        # get_action draws its initial noise through noise_source; feed it the graph input.
        with noise_source(lambda shape, dtype, device: noise.to(dtype=dtype, device=device)):
            return self.model.get_action(inputs).action_pred
