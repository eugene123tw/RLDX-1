"""Inference execution boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import threading
from typing import Any

import torch

from rldx.configs.execution import InferenceRuntimeProfile
from rldx.model.contracts import ActionOutput
from rldx.planner.input.tensor_image import TensorImageSpec
from rldx.runtime.forward_graph import ForwardGraphs
from rldx.runtime.tensor_image import TensorImages
from rldx.telemetry.step_profiler import profile_section as _profile
from rldx.transport.device import move_tensors


@dataclass(frozen=True)
class InferenceSpec:
    action_dim: int
    memory_shape: tuple[int, int] | None
    dtype: torch.dtype

    @property
    def supports_memory(self) -> bool:
        return self.memory_shape is not None


@dataclass(frozen=True)
class InferenceCall:
    planned_batch: Mapping[str, Any]
    controls: Mapping[str, Any] = field(default_factory=dict)
    memory: torch.Tensor | None = None
    reset_memory: tuple[bool, ...] | None = None
    manage_memory: bool = False


class InferenceRuntime:
    def __init__(
        self,
        model: Any,
        *,
        dtype: torch.dtype,
        profile: InferenceRuntimeProfile | None = None,
        gemm_rows: int | None = None,
        tensor_image_path: TensorImageSpec | None = None,
        inference_mode: bool = True,
        lock: Any | None = None,
    ) -> None:
        module = _execution_module(model)
        action_model = getattr(module, "action_model", None)
        self.spec = InferenceSpec(
            action_dim=int(getattr(action_model, "action_dim", 0)),
            memory_shape=module.inference_memory_shape
            if getattr(module, "use_memory", False)
            else None,
            dtype=dtype,
        )
        self._module = module
        self._device = _model_device(module)
        self._profile = profile
        self._gemm_rows = gemm_rows
        self._images = (
            TensorImages(tensor_image_path, self._device) if tensor_image_path is not None else None
        )
        self._inference_mode = inference_mode
        self._forward_graphs = (
            ForwardGraphs(self._forward)
            if profile is not None and profile.text_bucket is not None
            else None
        )
        self._lock = lock if lock is not None else threading.Lock()

    @property
    def inference_mode(self) -> bool:
        return self._inference_mode

    def prepare(self) -> None:
        if self._profile is None:
            return
        self._module.prepare_for_execution(self._profile, gemm_rows=self._gemm_rows)
        if self._images is not None:
            self._images = TensorImages(self._images.spec, self._device, self._profile.backend)

    def warm_up(self, calls: Iterable[InferenceCall]) -> None:
        """Run each served input shape once so its graph is captured before the first request."""
        for call in calls:
            self.execute(call)

    def execute(self, call: InferenceCall) -> ActionOutput:
        inputs = call.planned_batch["inputs"]
        collision = sorted(inputs.keys() & call.controls.keys())
        if collision:
            raise ValueError(f"control keys collide with the planned inputs: {collision}")

        controls = {**call.controls, **self._memory_controls(call)}
        inputs = move_tensors({**inputs, **controls}, self._device)
        if self._forward_graphs is not None:
            with self._lock, self._grad_context():
                return self._forward_graphs.run(_graph_inputs(inputs))

        with self._lock:
            self._prepare_tensor_images(inputs)
            with self._grad_context():
                return self._module.get_action(inputs)

    def _memory_controls(self, call: InferenceCall) -> dict:
        if not call.manage_memory:
            if call.reset_memory is not None or call.memory is not None:
                raise ValueError("memory controls need manage_memory")
            return {}
        if not self.spec.supports_memory:
            raise ValueError("inference call requested memory for a model without memory support")
        reset = call.reset_memory
        if call.memory is None and not all(reset):
            raise ValueError("a call without memory must reset every sample")
        memory = (
            call.memory
            if call.memory is not None
            else torch.zeros(
                len(reset), *self.spec.memory_shape, dtype=self.spec.dtype, device=self._device
            )
        )
        return {
            "memory": memory,
            "reset_memory": torch.tensor(reset, dtype=torch.bool, device=self._device),
        }

    def _forward(self, inputs: dict) -> ActionOutput:
        self._prepare_tensor_images(inputs)
        return self._module.get_action(inputs)

    def _prepare_tensor_images(self, inputs: dict) -> None:
        if self._images is None:
            return
        with _profile("inference.tensor_image_patchify"):
            self._images.to_pixel_values(inputs, self.spec.dtype)

    def _grad_context(self):
        if self._inference_mode:
            return torch.inference_mode()
        return torch.no_grad()


def _graph_inputs(inputs: dict) -> dict:
    host_layout = inputs.get("host_layout")
    if host_layout is None:
        return inputs
    return {
        **inputs,
        "host_layout": {key: value for key, value in host_layout.items() if key != "token_layout"},
    }


def _execution_module(model: Any) -> Any:
    module = getattr(model, "module", None)
    return module if isinstance(module, torch.nn.Module) else model


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")
