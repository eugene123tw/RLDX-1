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

"""
Unified processor for robot state and action data.

Handles:
- State normalization (min/max, mean/std, sin/cos encoding)
- Action normalization
- Absolute <-> Relative action representation conversion
- Action processing with state dependency
"""

from copy import deepcopy
import logging

import numpy as np
from rldx.configs.data.embodiment_configs import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from rldx.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from rldx.data.state_action.action_codec import ActionKeyCodec
from rldx.data.state_action.layout import Layout, join, split
from rldx.data.state_action.norm_stats import NormStats
from rldx.data.state_action.pose import EndEffectorPose, JointPose
from rldx.data.utils import (
    apply_sin_cos_encoding,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
)


def _group(values: dict[str, np.ndarray], key: str, embodiment_tag: str) -> np.ndarray:
    if key not in values:
        raise KeyError(
            f"Joint group '{key}' not found in action dict for embodiment '{embodiment_tag}'"
        )
    return values[key]


def _same_stats(a: dict[str, list[float]], b: dict[str, list[float]]) -> bool:
    if a.keys() != b.keys():
        return False
    for stat in a:
        x, y = np.asarray(a[stat], dtype=np.float64), np.asarray(b[stat], dtype=np.float64)
        if x.shape != y.shape or not np.allclose(x, y):
            return False
    return True


def _check_kept_statistics(embodiment_tag: str, kept: dict, incoming: dict) -> None:
    """Fail or warn when ``incoming`` statistics are dropped in favour of ``kept`` ones."""
    flag = "--override-pretraining-statistics"
    # Physics groups missing from the base are left out: apply_physics passes them raw (midtrain).
    missing = {
        modality: sorted(set(incoming[modality]) - set(kept.get(modality, {})))
        for modality in ("state", "action")
        if modality in incoming
    }
    missing = {modality: keys for modality, keys in missing.items() if keys}
    if missing:
        raise ValueError(
            f"The base checkpoint already has statistics for embodiment tag '{embodiment_tag}', "
            f"so this run's dataset statistics are not used, but they have joint groups the "
            f"checkpoint's do not: {missing}. Normalizing would fail on the first batch. "
            f"Pass {flag} to normalize with this run's statistics."
        )
    differing = sorted(
        f"{modality}/{group}"
        for modality, groups in incoming.items()
        for group, stats in groups.items()
        if group in kept.get(modality, {}) and not _same_stats(kept[modality][group], stats)
    )
    if differing:
        logging.warning(
            f"[StateActionProcessor] Embodiment tag '{embodiment_tag}' keeps the base checkpoint's "
            f"statistics, which differ from this run's dataset statistics for {differing}. "
            f"Pass {flag} to normalize with this run's statistics."
        )


class StateActionProcessor:
    """
    Unified processor for robot state and action data.

    Handles:
    - State normalization (min/max, mean/std, sin/cos encoding)
    - Action normalization
    - Absolute <-> Relative action representation conversion
    - Action processing with state dependency
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        """
        Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
                where modality in ["state", "action"]
                Example: {"gr1": {"state": ModalityConfig(...), "action": ModalityConfig(...)}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
                where modality in ["state", "action", "relative_action"]
                and stat_type in ["min", "max", "mean", "std", "q01", "q99"]
                Example: {"gr1": {"state": {"left_arm": {"min": [...], "max": [...], ...}}}}
            use_percentiles: True normalizes to q01/q99, False to min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Global flag to enable sin/cos encoding for states
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, NormStats]]] = {}

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """
        Set dataset statistics for normalization.

        Args:
            statistics: Nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
        """
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                _check_kept_statistics(key, self.statistics[key], statistics[key])
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics for all embodiments and modalities."""
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    low, high = ("q01", "q99") if self.use_percentiles else ("min", "max")
                    self.norm_params[embodiment_tag][modality][joint_group] = NormStats.from_stats(
                        stats, low, high
                    )

            # Physics modalities (tactile, torque, etc.): q01/q99 for normalize_values_q99
            for modality, stats_per_group in self.statistics[embodiment_tag].items():
                if modality in ("state", "action", "relative_action"):
                    continue  # Already handled above
                self.norm_params[embodiment_tag][modality] = {}
                for joint_group, stats in stats_per_group.items():
                    self.norm_params[embodiment_tag][modality][joint_group] = NormStats.from_stats(
                        stats, "q01", "q99"
                    )

            # Override absolute action stats with relative stats where specified
            embodiment_modality_configs = self.modality_configs.get(embodiment_tag, {})
            if "action" in embodiment_modality_configs:
                modality_keys = embodiment_modality_configs["action"].modality_keys
                action_configs = embodiment_modality_configs["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs):
                        if (
                            action_config.rep == ActionRepresentation.RELATIVE
                            and self.use_relative_action
                        ):
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            # TODO: raw min/max whatever use_percentiles, as in GR00T N1.6.
                            # Following the mode changes relative-trained checkpoints' numerics.
                            self.norm_params[embodiment_tag]["action"][key] = NormStats.from_stats(
                                self.statistics[embodiment_tag]["relative_action"][key],
                                "min",
                                "max",
                                dim=self.norm_params[embodiment_tag]["action"][key].dim,
                            )

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1")

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)
        """
        normalized_values = {}
        state = deepcopy(state)  # Avoid modifying input

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config:
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding (doubles dimension)
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])

            # Strategy 2: Mean/std normalization
            elif (
                self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Min/max normalization to [-1, 1]
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

        return normalized_values

    def action_codecs(self, embodiment_tag: str) -> dict[str, ActionKeyCodec]:
        config = self.modality_configs[embodiment_tag]["action"]
        mean_std = set(config.mean_std_embedding_keys or ())
        action_configs = config.action_configs or [None] * len(config.modality_keys)
        return {
            key: ActionKeyCodec(
                key=key,
                embodiment_tag=embodiment_tag,
                params=self.norm_params[embodiment_tag]["action"][key],
                mean_std=key in mean_std,
                clip=self.clip_outliers,
                relative=action_config if self._codes_relative(action_config) else None,
                to_relative=self._convert_to_relative_action,
                to_absolute=self._convert_to_absolute_action,
            )
            for key, action_config in zip(config.modality_keys, action_configs)
        }

    def _codes_relative(self, action_config: ActionConfig | None) -> bool:
        return (
            self.use_relative_action
            and action_config is not None
            and action_config.rep == ActionRepresentation.RELATIVE
        )

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
        position_slice: slice | None = None,
    ) -> dict[str, np.ndarray]:
        return {
            key: codec.encode(_group(action, key, embodiment_tag), state, position_slice)
            for key, codec in self.action_codecs(embodiment_tag).items()
        }

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
        position_slice: slice | None = None,
    ) -> dict[str, np.ndarray]:
        return {
            key: codec.decode(_group(action, key, embodiment_tag), state, position_slice)
            for key, codec in self.action_codecs(embodiment_tag).items()
        }

    def reframe_action(
        self,
        action: np.ndarray,
        embodiment_tag: str,
        prev_state: dict[str, np.ndarray],
        new_state: dict[str, np.ndarray],
        prev_position: slice,
        new_position: slice,
    ) -> np.ndarray:
        layout = self.layout(embodiment_tag, "action")
        parts = split(action, layout)
        moved = {
            key: codec.reframe(parts[key], prev_state, new_state, prev_position, new_position)
            for key, codec in self.action_codecs(embodiment_tag).items()
        }
        padding = action[..., sum(layout.values()) :]
        return np.concatenate([join(moved, layout), padding], axis=-1)

    def _has_params(self, embodiment_tag: str, modality: str, key: str) -> bool:
        return (
            embodiment_tag in self.norm_params
            and modality in self.norm_params[embodiment_tag]
            and key in self.norm_params[embodiment_tag][modality]
        )

    def apply_physics(
        self,
        physics: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """Normalize physics signals (tactile, torque, etc.) using q99 normalization.

        Args:
            physics: Dict mapping "modality.key" -> raw values (e.g. "tactile.left" -> ndarray)
            embodiment_tag: Embodiment identifier

        Returns:
            Dict of normalized values. Always a dict (possibly empty) — never
            None. In training, all-zero tactile/torque modalities are skipped
            (zero-padded columns of sensor-less robots in mixed datasets),
            which may produce a smaller dict than the input. Callers must
            detect missing modalities explicitly rather than relying on a
            None sentinel.
        """
        result = {}
        for compound_key, values in physics.items():
            values = np.array(values, dtype=np.float32)
            # compound_key is e.g., "tactile.left" -> modality="tactile", key="left"
            parts = compound_key.split(".", 1)
            if len(parts) == 2:
                modality, key = parts
            else:
                modality, key = compound_key, compound_key

            if self._has_params(embodiment_tag, modality, key):
                # Absent-sensor heuristic is TRAINING-only: mixed datasets
                # carry zero-padded physics columns for robots without the
                # sensor and cannot omit them (fixed schema). At inference
                # absence is structural — a client without the sensor omits
                # the group (allow_missing_physics) — so values that arrive are real
                # readings: zero means no contact, not no sensor.
                if self.training and modality in ("torque", "tactile") and np.all(values == 0):
                    continue
                from rldx.data.utils import normalize_values_q99

                params = self.norm_params[embodiment_tag][modality][key]
                normalized = normalize_values_q99(values, params)
                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)
                result[compound_key] = normalized
            elif self.training:
                # Documented midtrain flow: adding physics on top of a base
                # checkpoint whose statistics.json already covers the
                # embodiment tag — set_statistics skips the tag wholesale, so
                # the dataset's physics stats never merge and raw values are
                # what this flow has always trained on. Keep it, loudly.
                logging.warning(
                    f"[StateActionProcessor] No stats for physics "
                    f"'{modality}/{key}' in '{embodiment_tag}' — passing raw "
                    "values through (training path)"
                )
                result[compound_key] = values
            else:
                raise ValueError(
                    f"No normalization stats for physics '{modality}/{key}' in "
                    f"embodiment '{embodiment_tag}'. Passing raw values through "
                    "would silently feed the model out-of-distribution inputs — "
                    "the checkpoint's stats and the requested physics keys "
                    "disagree."
                )
        return result

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Apply both state and action processing together.

        Convenience method that processes state and action in one call,
        automatically passing raw state to action processor for relative conversion.

        Args:
            state: Dict mapping joint_group -> raw state values
            action: Dict mapping joint_group -> raw action values
            embodiment_tag: Embodiment identifier

        Returns:
            Tuple of (processed_state, processed_action)
        """
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def layout(self, embodiment_tag: str, modality: str) -> Layout:
        params = self.norm_params[embodiment_tag][modality]
        keys = self.modality_configs[embodiment_tag][modality].modality_keys
        return {key: params[key].dim for key in keys}

    def get_action_dim(self, embodiment_tag: str) -> int:
        return sum(self.layout(embodiment_tag, "action").values())

    def _convert_to_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert absolute action to relative action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        if action_type == ActionType.EEF:
            action_chunking = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            action_chunking = JointActionChunk([JointPose(m) for m in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        relative_action_chunking = action_chunking.relative_chunking(
            reference_frame=reference_frame
        )
        return relative_action_chunking.to(action_format)

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert relative action to absolute action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert reference_state.shape[0] == action.shape[1], (
            f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"
        )

        if action_type == ActionType.EEF:
            rel_action = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            rel_action = JointActionChunk([JointPose(pose) for pose in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        abs_action = rel_action.to_absolute_chunking(reference_frame=reference_frame)
        return abs_action.to(action_format)

    def __str__(self) -> str:
        return (
            "StateActionProcessor("
            f"modality_configs={self.modality_configs}, "
            f"statistics={self.statistics}, "
            f"use_percentiles={self.use_percentiles}, "
            f"clip_outliers={self.clip_outliers}, "
            f"apply_sincos_state_encoding={self.apply_sincos_state_encoding}, "
            f"use_relative_action={self.use_relative_action})"
        )
