# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

from __future__ import annotations

import itertools
import json
import math
from typing import Any
import warnings

import numpy as np
import torch

from rldx.configs.execution import InferenceRuntimeProfile, ServingLimits
from rldx.configs.model.inference import InferenceConfig
from rldx.configs.model.rldx import RLDXConfig
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.state_action.layout import join, split
from rldx.data.types import MessageType, ModalityConfig
from rldx.model.core.collator import PromptEncoder, RLDXDataCollator
from rldx.model.core.processor import RLDXProcessor
from rldx.model.modules.action_model.rtc import RtcActions, RtcInput, RtcTiming
from rldx.planner import BatchPlanner
from rldx.planner.input.qwen3vl import Qwen3VLInputPlanner
from rldx.runtime.inference import InferenceCall, InferenceRuntime
from rldx.serving import loader
from rldx.serving.sessions import SessionRegistry
from rldx.telemetry.step_profiler import (
    mark_step as _profile_mark_step,
    mark_step_start as _profile_mark_step_start,
    profile_section as _profile,
)

from .schema import Action, InputContract, OutputContract, PolicyOptions, PolicyStep, RawObservation


class RLDXPolicy:
    """Serves a checkpoint without keeping anything between calls.

    With RTC on, the client sends as ``action_prefix`` the actions the robot is
    executing from its previous chunk, in robot units, from the current step on:
    ``get_rtc()["prefix_steps"]`` of them. ``rtc_prefix_len`` counts, per sample,
    how many of the first ``delay`` are committed; 0 marks a sample without a
    previous chunk.
    """

    @classmethod
    def from_checkpoint(
        cls,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        inference_config: InferenceConfig = InferenceConfig(),
        runtime_profile: InferenceRuntimeProfile | None = None,
        serving_limits: ServingLimits | None = None,
    ) -> "RLDXPolicy":
        dtype = torch.bfloat16
        runtime_profile = runtime_profile or InferenceRuntimeProfile()
        if runtime_profile.text_bucket is not None:
            if serving_limits is None:
                raise ValueError("text_bucket needs serving limits to bound its CUDA graphs")
            if inference_config.sample_timestep_from_beta_dist:
                raise ValueError(
                    "sample_timestep_from_beta_dist draws a new schedule every call, "
                    "which a forward CUDA graph records once"
                )
        if runtime_profile.backend == "eager":
            warnings.warn("inference runs eager: no compile regions or CUDA graphs", stacklevel=2)
        model_dir = loader.resolve_model_dir(model_path)
        saved = RLDXConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
        if inference_config.deactivate_memory and saved.memory is not None:
            print("[RLDXPolicy] deactivate_memory=True: disabling memory for inference...")
        model_config = inference_config.model_config(saved)
        tag = embodiment_tag.value
        modality_configs = inference_config.modality_configs(
            RLDXProcessor.load_config(model_dir).modality_configs[tag], model_config
        )
        processor = RLDXProcessor.from_pretrained(
            model_dir,
            rldx_config=inference_config.processor_config(model_config, modality_configs),
            modality_configs={tag: modality_configs},
        )
        processor.eval()
        model = loader.load_model(model_dir, model_config, device, inference_config, dtype=dtype)
        encoder = PromptEncoder(
            processor.collator.encoder.processor,
            image_max_area=model_config.image.max_area,
            image_first=model_config.conversation_image_first,
            instruction_limit=None if serving_limits is None else serving_limits.instruction_tokens,
            text_bucket=runtime_profile.text_bucket,
        )
        collate_fn = BatchPlanner(
            collator=RLDXDataCollator(encoder),
            input_planner=Qwen3VLInputPlanner(
                dtype=dtype,
                spec=model.backbone.input_spec,
                text_attention=True,
                memo_size=loader.INFERENCE_PLANNER_MEMO_SIZE,
            ),
        )
        rtc_timing = model.action_model.rtc_timing
        if rtc_timing is not None:
            print(f"[RLDXPolicy] RTC enabled: {rtc_timing}")
        loader.require_video_keys(modality_configs)
        gemm_rows = None
        warm_up_instructions: tuple[int, ...] = ()
        if serving_limits is not None:
            gemm_rows = loader.gemm_rows(serving_limits, encoder, modality_configs, model)
            warm_up_instructions = tuple(
                encoder.warm_up_instructions(
                    modality_configs["video"], serving_limits.instruction_tokens
                )
            )
        guided = rtc_timing is not None and rtc_timing.guided
        inference = loader.build_inference_runtime(
            model,
            guided,
            dtype=dtype,
            profile=runtime_profile,
            gemm_rows=gemm_rows,
            tensor_image_path=processor.eval_image_spec,
        )
        inference.prepare()
        if guided:
            model.action_model.prepare_input_gradients()
        return cls(
            processor=processor,
            inference=inference,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            collate_fn=collate_fn,
            physics_keys=[] if model_config.physics is None else list(model_config.physics.keys),
            rtc_timing=rtc_timing,
            serving_limits=serving_limits,
            warm_up_instructions=warm_up_instructions,
        )

    def __init__(
        self,
        *,
        processor: Any,
        inference: InferenceRuntime,
        modality_configs: dict,
        embodiment_tag: EmbodimentTag,
        collate_fn: Any,
        physics_keys: list[str],
        rtc_timing: RtcTiming | None,
        serving_limits: ServingLimits | None = None,
        warm_up_instructions: tuple[int, ...] = (),
    ):
        if inference.spec.supports_memory and rtc_timing is not None:
            raise ValueError(
                "memory and RTC are served apart: a memory checkpoint runs without RTC, "
                "since memory snapshots assume calls a fixed number of steps apart"
            )
        sap = processor.state_action_processor
        tag = embodiment_tag.value
        physics = processor.rldx_config.physics
        self.inference = inference
        self.processor = processor
        self.modality_configs = modality_configs
        self.embodiment_tag = embodiment_tag
        self.collate_fn = collate_fn
        self.physics_keys = physics_keys
        self.rtc_timing = rtc_timing
        self._input = InputContract(
            modality_configs=modality_configs,
            physics_keys=physics_keys,
            embodiment_tag=embodiment_tag,
            allow_missing_physics=physics is not None and physics.allow_missing,
            expected_state_dims=sap.layout(tag, "state"),
            serving_limits=serving_limits,
            rtc_delay=None if rtc_timing is None else rtc_timing.delay,
        )
        self._output = OutputContract(
            modality_configs=modality_configs,
            action_dims=sap.layout(tag, "action"),
        )
        if serving_limits is not None:
            self.inference.warm_up(self._served_shape_calls(serving_limits, warm_up_instructions))

    def get_action(
        self, observation: RawObservation, options: PolicyOptions | None = None
    ) -> tuple[Action, dict[str, Any]]:
        if self.inference.spec.supports_memory:
            raise ValueError("a memory checkpoint is served through RLDXMemoryPolicy's sessions")
        action, _ = self.step(self.parse(observation, options))
        return action, {}

    def parse(
        self, observation: RawObservation, options: PolicyOptions | None = None
    ) -> PolicyStep:
        return self._input.parse(observation, options)

    def step(
        self,
        request: PolicyStep,
        memory: torch.Tensor | None = None,
        cold: tuple[bool, ...] | None = None,
    ) -> tuple[Action, torch.Tensor | None]:
        """One call; a memory checkpoint also takes each sample's memory and returns the next."""
        _profile_mark_step_start(None)
        with _profile("step.total"):
            raw, memory = self._step(request, memory, cold)
        # After step.total closes, so the heartbeat reads this step's sample.
        _profile_mark_step()
        return self._output.parse(raw), memory

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if options:
            raise ValueError(f"a stateless policy takes no reset options, got {sorted(options)}")
        return {}

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def get_memory(self) -> None:
        return None

    def get_rtc(self) -> dict[str, int] | None:
        timing = self.rtc_timing
        if timing is None:
            return None
        return {
            "delay": timing.delay,
            "exec_horizon": timing.exec_horizon,
            "prefix_steps": self._prefix_span,
        }

    def _step(
        self,
        request: PolicyStep,
        memory: torch.Tensor | None,
        cold: tuple[bool, ...] | None,
    ) -> tuple[dict[str, np.ndarray], torch.Tensor | None]:
        with _profile("step.prepare_inputs_total"):
            states, collated = self._prepare_inputs(request)
        if self.rtc_timing is not None:
            with _profile("step.rtc_prefix_inject"):
                collated["rtc"] = self._rtc_input(self._client_prefix(request, states))
        with _profile("step.inference"):
            output = self.inference.execute(self._inference_call(collated, memory, cold))
        with _profile("step.decode"):
            return self._decode(output.action_pred.float(), states), output.memory

    def _prepare_inputs(self, request: PolicyStep) -> tuple[list[dict], dict]:
        processed_inputs = []
        with _profile("prepare_inputs.processor_per_sample_total"):
            for sample in request.samples:
                messages = [{"type": MessageType.EPISODE_STEP.value, "content": sample}]
                with _profile("prepare_inputs.processor_single_call"):
                    processed_inputs.append(self.processor(messages))
        with _profile("prepare_inputs.collate"):
            collated = self.collate_fn(processed_inputs)
        return list(request.states), collated

    @property
    def _prefix_span(self) -> int:
        timing = self.rtc_timing
        return timing.delay + (timing.ramp if timing.guided else 0)

    def _client_prefix(self, request: PolicyStep, states: list[dict]) -> RtcActions:
        span = self._prefix_span
        B = request.batch_size
        normalized = torch.zeros(B, span, self.inference.spec.action_dim)
        if request.rtc is None:
            return self._rtc_actions(normalized, torch.zeros(B, span, dtype=torch.bool))
        sap = self.processor.state_action_processor
        layout = sap.layout(self.embodiment_tag.value, "action")
        width = sum(layout.values())
        prefix = torch.as_tensor(request.rtc.actions, dtype=torch.float32)
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(0)
        if prefix.shape[1] < span or prefix.shape[2] != width:
            raise ValueError(
                f"action_prefix must hold at least {span} decoded actions of width {width} "
                f"the robot still holds; got {tuple(prefix.shape)}"
            )
        prefix = prefix[:, :span]
        if not bool(prefix.isfinite().all()):
            raise ValueError("action_prefix contains NaN or Inf values")
        for b, state in enumerate(states):
            action = sap.apply_action(
                split(prefix[b].numpy(), layout),
                embodiment_tag=self.embodiment_tag.value,
                state=state,
                # Per-step normalization stats: the prefix opens the new chunk.
                position_slice=slice(0, span),
            )
            normalized[b, :, :width] = torch.from_numpy(join(action, layout))
        d = self.rtc_timing.delay
        frozen = torch.tensor(request.rtc.prefix_len)[:, None]
        steps = torch.arange(span)
        present = (steps < frozen) | ((steps >= d) & (frozen > 0))
        return self._rtc_actions(normalized, present)

    def _rtc_input(self, prefix: RtcActions) -> RtcInput:
        timing = self.rtc_timing
        d = timing.delay

        def steps(window: slice) -> RtcActions:
            return RtcActions(
                prefix.actions[:, window].contiguous(), prefix.present[:, window].contiguous()
            )

        postfix = steps(slice(d, d + timing.ramp)) if timing.guided and timing.ramp > 0 else None
        return RtcInput(steps(slice(0, d)), postfix)

    def _rtc_actions(self, actions: torch.Tensor, present: torch.Tensor) -> RtcActions:
        return RtcActions(actions.to(self.inference.spec.dtype).contiguous(), present.contiguous())

    def _inference_call(
        self,
        collated: dict,
        memory: torch.Tensor | None = None,
        cold: tuple[bool, ...] | None = None,
    ) -> InferenceCall:
        control = {key: value for key, value in collated.items() if key != "inputs"}
        if not self.inference.spec.supports_memory:
            return InferenceCall(planned_batch=collated, controls=control)
        return InferenceCall(
            planned_batch=collated,
            controls=control,
            memory=memory,
            reset_memory=cold,
            manage_memory=True,
        )

    def _decode(self, normalized_action: torch.Tensor, states: list[dict]) -> dict[str, np.ndarray]:
        batched_states = {
            key: np.stack([state[key] for state in states])
            for key in self.modality_configs["state"].modality_keys
        }
        action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        return {key: value.astype(np.float32) for key, value in action.items()}

    def _served_shape_calls(self, limits: ServingLimits, instructions: tuple[int, ...]):
        """A cold and a warm call per accepted batch size and instruction bucket; only
        the shapes matter, so the warm prefix and memory are zeros."""
        spec = self.inference.spec
        for batch, words in itertools.product(range(1, limits.batch + 1), instructions):
            request = self._input.parse(self._blank_observation(batch, " ".join(["go"] * words)))
            for warm in (False, True):
                _, collated = self._prepare_inputs(request)
                if self.rtc_timing is not None:
                    span = self._prefix_span
                    prefix = self._rtc_actions(
                        torch.zeros(batch, span, spec.action_dim),
                        torch.full((batch, span), warm),
                    )
                    collated["rtc"] = self._rtc_input(prefix)
                if not spec.supports_memory:
                    yield self._inference_call(collated)
                    continue
                memory = torch.zeros(batch, *spec.memory_shape, dtype=spec.dtype) if warm else None
                yield self._inference_call(collated, memory, (not warm,) * batch)

    def _blank_observation(self, batch: int, instruction: str) -> RawObservation:
        side = math.isqrt(self.processor.rldx_config.image.max_area)
        sap = self.processor.state_action_processor
        modalities = self.modality_configs

        def frames(group: str) -> int:
            return len(modalities[group].delta_indices)

        def zeros(group: str) -> dict[str, np.ndarray]:
            return {
                key: np.zeros((batch, frames(group), dim), np.float32)
                for key, dim in sap.layout(self.embodiment_tag.value, group).items()
            }

        return {
            "video": {
                key: np.zeros((batch, frames("video"), side, side, 3), np.uint8)
                for key in modalities["video"].modality_keys
            },
            "state": zeros("state"),
            "language": {
                key: [[instruction] * frames("language") for _ in range(batch)]
                for key in modalities["language"].modality_keys
            },
            **{group: zeros(group) for group in self._input.physics_groups},
        }


class RLDXMemoryPolicy:
    """Serves a memory checkpoint: each session's memory carries from one call to the next.

    Every request names one session per sample (``session_ids``) and marks the samples
    that start an episode (``reset_memory``). The client runs ``get_memory()["stride"]``
    steps of each chunk before calling again: memory keeps one snapshot per call, and
    training spaced them that far apart.
    """

    def __init__(self, policy: RLDXPolicy):
        if not policy.inference.spec.supports_memory:
            raise ValueError(
                "a checkpoint without memory has no sessions; serve it with RLDXPolicy"
            )
        self._policy = policy
        self._sessions = SessionRegistry()

    def get_action(
        self, observation: RawObservation, options: dict[str, Any] | None = None
    ) -> tuple[Action, dict[str, Any]]:
        options = dict(options or {})
        session_ids = options.pop("session_ids", None)
        starts = options.pop("reset_memory", None)
        request = self._policy.parse(observation, options)
        sids = _parse_session_ids(session_ids, request.batch_size)
        starts = _parse_starts(starts, request.batch_size)
        self._sessions.reset([sid for sid, start in zip(sids, starts) if start])
        memory, cold = self._sessions.load_memory_batch(sids)
        action, memory = self._policy.step(request, memory, cold)
        self._sessions.save_memory_batch(sids, memory)
        return action, {}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if options is None:
            return {"cleared_sessions": self._sessions.clear()}
        if options.keys() != {"session_ids"}:
            raise ValueError(f"reset takes only session_ids, got {sorted(options)}")
        return {"cleared_sessions": self._sessions.reset(list(options["session_ids"]))}

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self._policy.get_modality_config()

    def get_rtc(self) -> None:
        return None

    def get_memory(self) -> dict[str, int]:
        memory = self._policy.processor.rldx_config.memory
        return {"length": memory.length, "stride": memory.stride}


def load_policy(
    embodiment_tag: EmbodimentTag,
    model_path: str,
    *,
    device: int | str,
    inference_config: InferenceConfig = InferenceConfig(),
    runtime_profile: InferenceRuntimeProfile | None = None,
    serving_limits: ServingLimits | None = None,
) -> RLDXPolicy | RLDXMemoryPolicy:
    policy = RLDXPolicy.from_checkpoint(
        embodiment_tag,
        model_path,
        device=device,
        inference_config=inference_config,
        runtime_profile=runtime_profile,
        serving_limits=serving_limits,
    )
    return RLDXMemoryPolicy(policy) if policy.inference.spec.supports_memory else policy


def _parse_session_ids(value: Any, batch_size: int) -> list[str]:
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError("a memory checkpoint needs session_ids, one str per sample")
    sids = list(value)
    if len(sids) != batch_size or not all(isinstance(sid, str) for sid in sids):
        raise ValueError(f"session_ids must be {batch_size} str, one per sample, got {sids!r}")
    return sids


def _parse_starts(value: Any, batch_size: int) -> list[bool]:
    if value is None:
        return [False] * batch_size
    starts = list(value)
    if len(starts) != batch_size or not all(isinstance(start, bool) for start in starts):
        raise ValueError(f"reset_memory must be {batch_size} bool, one per sample, got {starts!r}")
    return starts
