# SPDX-License-Identifier: Apache-2.0
"""What a policy call carries in and out, and the one parse that admits it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NewType, TypeAlias, TypedDict

import numpy as np
import torch

from rldx.configs.execution import ServingLimits
from rldx.data.state_action.layout import Layout
from rldx.data.types import VLAStepData


RawObservation: TypeAlias = Mapping[str, Any]
RawAction: TypeAlias = dict[str, np.ndarray]
Action = NewType("Action", RawAction)


class PolicyOptions(TypedDict, total=False):
    action_prefix: "np.ndarray | torch.Tensor"
    rtc_prefix_len: int | Sequence[int]


@dataclass(frozen=True)
class RtcPrefix:
    actions: Any
    prefix_len: tuple[int, ...]


@dataclass(frozen=True)
class PolicyStep:
    samples: tuple[VLAStepData, ...]
    batch_size: int
    rtc: RtcPrefix | None = None

    @property
    def states(self) -> tuple[dict, ...]:
        return tuple(sample.states for sample in self.samples)


class InputContract:
    def __init__(
        self,
        *,
        modality_configs: dict,
        physics_keys: list[str],
        embodiment_tag: Any,
        expected_state_dims: Layout,
        allow_missing_physics: bool = False,
        serving_limits: ServingLimits | None = None,
        rtc_delay: int | None = None,
    ) -> None:
        self.modality_configs = modality_configs
        self.rtc_delay = rtc_delay
        self.serving_limits = serving_limits
        self.physics_keys = physics_keys
        self.language_key = _one_instruction_key(modality_configs["language"])
        self.embodiment_tag = embodiment_tag
        self.allow_missing_physics = allow_missing_physics
        # Groups the client must send: keys pruned from modality_configs
        # (fut-only mode) are not part of the wire contract.
        self.physics_groups = [pk for pk in physics_keys if pk in modality_configs]
        self.expected_state_dims = expected_state_dims
        supported = {"video", "state", "action", "language", *self.physics_groups}
        unknown = sorted(set(modality_configs) - supported)
        if unknown:
            raise ValueError(f"Unsupported observation groups {unknown}")

    def parse(
        self,
        observation: RawObservation,
        options: PolicyOptions | None = None,
    ) -> PolicyStep:
        if not isinstance(observation, dict):
            raise ValueError(f"observation must be a dict, got {type(observation).__name__}")
        allowed = set(self.modality_configs) - {"action"}
        unknown = sorted(set(observation) - allowed)
        if unknown:
            message = (
                f"Unknown observation keys {unknown}. The observation contract "
                f"allows {sorted(allowed)}."
            )
            if "physics" in unknown and self.physics_groups:
                message += (
                    " The observation['physics'] envelope (RLDX-1 v1.0.2 clients) is "
                    "no longer accepted. Send each physics group exactly as advertised "
                    f"by get_modality_config() — top-level groups {self.physics_groups} "
                    "with their modality_keys as sub-keys, e.g. "
                    "observation['tactile'] = {'left': ..., 'right': ...}."
                )
            raise ValueError(message)
        for group in ("video", "state", "language"):
            if group not in observation:
                raise ValueError(f"Observation must contain a '{group}' key")
            if not isinstance(observation[group], dict):
                raise ValueError(
                    f"Observation '{group}' must be a dictionary, "
                    f"got {type(observation[group]).__name__}"
                )
            expected = set(self.modality_configs[group].modality_keys)
            actual = set(observation[group])
            if actual != expected:
                raise ValueError(
                    f"Observation '{group}' keys must be {sorted(expected)}, got {sorted(actual)}"
                )
        batch_size = self._check_arrays(observation)
        limits = self.serving_limits
        if limits is not None and batch_size > limits.batch:
            raise ValueError(f"batch size {batch_size} exceeds the serving limit {limits.batch}")
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError(f"options must be a dict, got {type(options).__name__}")
        unknown_options = sorted(options.keys() - PolicyOptions.__annotations__.keys())
        if unknown_options:
            raise ValueError(f"Unknown policy option keys {unknown_options}")

        return PolicyStep(
            samples=self._parse_samples(observation, batch_size),
            batch_size=batch_size,
            rtc=self._parse_rtc(options, batch_size),
        )

    def _parse_rtc(self, options: PolicyOptions, batch_size: int) -> RtcPrefix | None:
        actions = options.get("action_prefix")
        lengths = options.get("rtc_prefix_len")
        if actions is None:
            if lengths is not None:
                raise ValueError("rtc_prefix_len counts the frozen steps of an action_prefix")
            return None
        delay = self.rtc_delay
        if delay is None:
            raise ValueError("action_prefix needs a server with RTC enabled")
        _check_action_prefix(actions, batch_size)
        prefix_len = (
            (delay,) * batch_size if lengths is None else _parse_prefix_lengths(lengths, batch_size)
        )
        if max(prefix_len) > delay:
            raise ValueError(
                f"rtc_prefix_len must count 0 to {delay} frozen steps, got {prefix_len}"
            )
        return RtcPrefix(actions, prefix_len)

    def _parse_samples(self, observation: dict, batch_size: int) -> tuple[VLAStepData, ...]:
        samples = []
        for index in range(batch_size):
            physics = {
                f"{group}.{key}": value[index]
                for group in self.physics_groups
                if group in observation
                for key, value in observation[group].items()
            }
            samples.append(
                VLAStepData(
                    images={key: value[index] for key, value in observation["video"].items()},
                    states={key: value[index] for key, value in observation["state"].items()},
                    actions={},
                    text=observation["language"][self.language_key][index][0],
                    embodiment=self.embodiment_tag,
                    physics=physics,
                )
            )
        return tuple(samples)

    def _check_arrays(self, observation: dict[str, Any]) -> int:
        name = f"Language key '{self.language_key}'"
        language = observation["language"][self.language_key]
        if not isinstance(language, list):
            raise ValueError(f"{name} must be a list. Got {type(language)}")
        for item in language:
            if not (isinstance(item, list) and len(item) == 1 and isinstance(item[0], str)):
                raise ValueError(f"{name} items must be [str], got {item!r}")
        batch = len(language)

        for key in self.modality_configs["video"].modality_keys:
            name = f"Video key '{key}'"
            video = _array(name, observation["video"][key], np.uint8, ("B", "T", "H", "W", "C"))
            _same_batch(name, len(video), batch)
            self._check_video_horizon(key, video)
            if video.shape[-1] != 3:
                raise ValueError(f"Video key '{key}'s channel 'C' must be 3. Got {video.shape[-1]}")

        for key in self.modality_configs["state"].modality_keys:
            name = f"State key '{key}'"
            state = _array(name, observation["state"][key], np.float32, ("B", "T", "D"))
            _same_batch(name, len(state), batch)
            horizon = len(self.modality_configs["state"].delta_indices)
            if state.shape[1] != horizon:
                raise ValueError(
                    f"State key '{key}'s horizon must be {horizon}. Got {state.shape[1]}"
                )
            expected = self.expected_state_dims[key]
            if state.shape[-1] != expected:
                raise ValueError(
                    f"{name} shape[-1]={state.shape[-1]} != expected_state_dim={expected}"
                )
            _check_finite(f"state key '{key}'", state)

        present = [group for group in self.physics_groups if group in observation]
        # physics_mask is one bit per sample, so a partial set would be zero-filled whole.
        if self.allow_missing_physics and present and len(present) < len(self.physics_groups):
            raise ValueError(
                f"This checkpoint takes physics groups {self.physics_groups} all or "
                f"none, got only {present}. A partial set cannot be used: the sent "
                "groups would be dropped too and inference would run without physics."
            )
        for group in self.physics_groups:
            if group not in observation:
                if not self.allow_missing_physics:
                    raise ValueError(
                        f"Model requires physics group '{group}' "
                        f"(physics groups: {self.physics_groups}) but observation "
                        f"has no '{group}' key. Send it as advertised by "
                        f"get_modality_config()."
                    )
                continue
            values = observation[group]
            if not isinstance(values, dict):
                raise ValueError(f"Observation '{group}' must be a dictionary. Got {type(values)}")
            expected_keys = list(self.modality_configs[group].modality_keys)
            if sorted(values) != sorted(expected_keys):
                raise ValueError(
                    f"Physics group '{group}' sub-keys must match the handshake "
                    f"modality_keys {expected_keys}. Got {sorted(values)}"
                )
            horizon = len(self.modality_configs[group].delta_indices)
            for key in expected_keys:
                name = f"Physics key '{group}.{key}'"
                physics = _array(name, values[key], np.float32, ("B", "T", "D"))
                _same_batch(name, len(physics), batch)
                if physics.shape[1] != horizon:
                    raise ValueError(
                        f"{name} horizon must be {horizon} "
                        f"(len(delta_indices) from the handshake). Got {physics.shape[1]}"
                    )
                _check_finite(f"physics key '{group}.{key}'", physics)
        return batch

    def _check_video_horizon(self, key: str, video: np.ndarray) -> None:
        deltas = self.modality_configs["video"].delta_indices
        if video.shape[1] == len(deltas):
            return
        message = (
            f"Video key '{key}' shape mismatch: "
            f"got T={video.shape[1]}, expected T={len(deltas)} (delta_indices={deltas})."
        )
        raise ValueError(message)


def _one_instruction_key(language: Any) -> str:
    if len(language.modality_keys) != 1 or len(language.delta_indices) != 1:
        raise ValueError(
            "language must carry one instruction per sample (one key, one delta index), "
            f"got keys {list(language.modality_keys)} and delta_indices "
            f"{list(language.delta_indices)}"
        )
    return language.modality_keys[0]


def _array(name: str, value: Any, dtype: type, dims: tuple[str, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy array. Got {type(value)}")
    if value.dtype != dtype:
        raise ValueError(
            f"{name} must be a numpy array of type np.{dtype.__name__}. Got {value.dtype}"
        )
    if value.ndim != len(dims):
        raise ValueError(
            f"{name} must be a numpy array of shape ({', '.join(dims)}), got {value.shape}"
        )
    return value


def _same_batch(name: str, size: int, batch: int) -> None:
    if size != batch:
        raise ValueError(f"{name} must have batch size {batch}. Got {size}")


def _check_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite values in {name}")


class OutputContract:
    def __init__(self, *, modality_configs: dict, action_dims: Layout) -> None:
        self.horizon = len(modality_configs["action"].delta_indices)
        self.action_dims = action_dims

    def parse(self, raw: RawAction) -> Action:
        if raw.keys() != self.action_dims.keys():
            raise ValueError(f"action keys {sorted(raw)} != {sorted(self.action_dims)}")
        for key, dim in self.action_dims.items():
            value = raw[key]
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.float32
                and value.ndim == 3
                and value.shape[1:] == (self.horizon, dim)
            ):
                got = (
                    f"{value.dtype}{value.shape}"
                    if isinstance(value, np.ndarray)
                    else type(value).__name__
                )
                raise ValueError(
                    f"action '{key}' must be float32 (B, {self.horizon}, {dim}), got {got}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite values in action key '{key}'")
        return Action(raw)


def _parse_prefix_lengths(value: Any, batch_size: int) -> tuple[int, ...]:
    lengths = (value,) * batch_size if isinstance(value, int) else tuple(value)
    if len(lengths) != batch_size:
        raise ValueError(f"rtc_prefix_len length {len(lengths)} != batch size {batch_size}")
    return tuple(_parse_non_negative_int(length, "rtc_prefix_len") for length in lengths)


def _check_action_prefix(value: Any, batch_size: int) -> None:
    if value is None:
        return
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(
            "action_prefix must have .shape attribute "
            f"(ndarray or Tensor), got {type(value).__name__}"
        )
    if len(shape) == 2 and batch_size != 1:
        raise ValueError(
            f"a 2-D action_prefix (d, D) is one sample's prefix; batch size {batch_size} "
            "needs (B, d, D)"
        )
    if len(shape) not in (2, 3):
        raise ValueError(f"action_prefix shape must be 2 (d,D) or 3 (B,d,D), got {tuple(shape)}")
    if len(shape) == 3 and shape[0] != batch_size:
        raise ValueError(f"action_prefix batch dim {shape[0]} != batch size {batch_size}")


def _parse_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be non-negative int, got {value!r}")
    return value
