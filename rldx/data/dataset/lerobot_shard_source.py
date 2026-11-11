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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rldx.data.interfaces import ShardedDataset
from rldx.data.types import (
    EmbodimentTag,
    EpisodeBoundaryPolicy,
    MessageType,
    ModalityConfig,
    VLAStepData,
)

from .lerobot_episode_loader import LeRobotEpisodeLoader


@dataclass(frozen=True)
class ShardPlan:
    shards: list[list[tuple[int, np.ndarray]]]
    lengths: np.ndarray


@dataclass(frozen=True)
class StepShardSampler:
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    seed: int = 42

    def build(self, episode_lengths: list[int], *, dataset_path: str | Path) -> ShardPlan:
        """Deterministically assign episode steps to balanced shards."""
        rng = np.random.default_rng(self.seed)
        episode_indices = rng.permutation(len(episode_lengths))
        assert len(episode_indices) > 0, f"No valid trajectories found for dataset {dataset_path}"

        num_splits = int(1 / self.episode_sampling_rate)
        total_steps = int(np.sum([episode_lengths[index] for index in episode_indices]))
        num_shards = int(np.ceil(total_steps / self.shard_size))
        shards: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(num_shards)]
        lengths = np.zeros(num_shards, dtype=int)

        for episode_index in episode_indices:
            step_indices = np.arange(episode_lengths[episode_index])
            rng.shuffle(step_indices)
            for split_index in range(num_splits):
                sampled_steps = step_indices[split_index::num_splits]
                shard_index = int(np.argmin(lengths))
                shards[shard_index].append((int(episode_index), sampled_steps))
                lengths[shard_index] += len(sampled_steps)

        assert np.all(lengths > 0), "All shards must have length greater than 0"
        return ShardPlan(shards=shards, lengths=lengths)


def required_video_frame_indices(
    step_indices: np.ndarray,
    episode_length: int,
    modality_configs: dict[str, ModalityConfig],
) -> np.ndarray:
    """Return the sorted frame union shared by all configured video views."""
    video_config = modality_configs.get("video")
    if video_config is None:
        return np.empty(0, dtype=np.int64)

    raw = (
        np.asarray(step_indices, dtype=np.int64)[:, None]
        + np.asarray(video_config.delta_indices, dtype=np.int64)[None, :]
    )
    raw = np.clip(raw, 0, episode_length - 1)
    return np.unique(raw.reshape(-1))


def extract_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    boundary_policy: EpisodeBoundaryPolicy = EpisodeBoundaryPolicy.MASK,
) -> VLAStepData:
    step_data = {}

    # Extract data for each configured modality
    for modality, config in modality_configs.items():
        # Skip modalities whose keys are entirely absent from the episode data
        available_keys = [
            k for k in config.modality_keys if f"{modality}.{k}" in episode_data.columns
        ]
        if not available_keys:
            continue

        step_data[modality] = {}
        # Calculate indices
        raw_indices = [step_index + delta_index for delta_index in config.delta_indices]
        max_idx = len(episode_data) - 1

        # Determine padding mask (True if out of bounds)
        out_of_bounds = [idx < 0 or idx > max_idx for idx in raw_indices]

        indices_to_load = [max(0, min(idx, max_idx)) for idx in raw_indices]

        for i, key in enumerate(config.modality_keys):
            if f"{modality}.{key}" in episode_data.columns:
                modality_data = episode_data[f"{modality}.{key}"].iloc[indices_to_load]
            else:
                continue

            if modality not in ("video", "language"):
                # Numerical modalities: state, action, and any physics signals
                # Determine padding strategy; default is absolute (clamp to first/last) unless specified otherwise
                zero_pad = False
                if config.action_configs is not None and i < len(config.action_configs):
                    from rldx.data.types import ActionRepresentation

                    act_cfg = config.action_configs[i]
                    # DELTA only. A RELATIVE key holds an absolute value here and becomes
                    # ``action - state`` downstream, so zeroing it asks the joint to
                    # travel to the origin; clamping holds the last pose.
                    if act_cfg.rep is ActionRepresentation.DELTA:
                        zero_pad = True

                # Stack arrays for numerical modalities
                arrays = []
                for j in range(len(modality_data)):
                    val = np.array(modality_data.iloc[j]).astype(np.float32)
                    if out_of_bounds[j] and zero_pad:
                        # A DELTA step displacement past the end is no movement.
                        val = np.zeros_like(val)
                    arrays.append(val)
                step_data[modality][key] = np.vstack(arrays)
            else:
                # Keep as lists for non-numerical modalities (video / language).
                # The processor consumes them as list[PIL] or list[np.ndarray].
                step_data[modality][key] = modality_data.tolist()

    # Parse extracted data into VLAStepData structure
    video_data = step_data.get("video", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    assert len(language_data) == 1, f"Expected 1 language, got {len(language_data)}"
    text = language_data[list(language_data.keys())[0]][0]

    # Collect physics modalities (any modality that isn't video/state/action/language)
    physics_data = {}
    for mod_key, mod_vals in step_data.items():
        if mod_key not in ("video", "state", "action", "language"):
            physics_data.update({f"{mod_key}.{k}": v for k, v in mod_vals.items()})

    # The processor zeroes ``action_mask`` at and beyond this index, so this one
    # value is what separates MASK from PAD.
    if "action" in modality_configs:
        action_horizon = len(modality_configs["action"].delta_indices)
        in_episode_action_horizon = max(0, min(action_horizon, len(episode_data) - step_index))
        if boundary_policy == EpisodeBoundaryPolicy.PAD:
            valid_action_horizon = action_horizon
        else:
            valid_action_horizon = in_episode_action_horizon
    else:
        action_horizon = 0
        in_episode_action_horizon = 0
        valid_action_horizon = 0

    vla_step_data = VLAStepData(
        images=video_data,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
        physics=physics_data,
        metadata={
            "valid_action_horizon": valid_action_horizon,
            # The boundary itself, which PAD removes from ``valid_action_horizon``.
            # Physics cannot see past the episode end under either policy.
            "in_episode_action_horizon": in_episode_action_horizon,
        },
    )
    return vla_step_data


class LeRobotShardSource(ShardedDataset):
    """Materialize planned LeRobot shards into processed training samples.

    ``StepShardSampler`` owns step selection and balancing. This source owns
    episode I/O, selective video decoding, step extraction, and processing.

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        num_ffmpeg_threads: FFmpeg threads per decoder.
        shard_size: Target number of timesteps per shard
        episode_sampling_rate: Fraction of episode timesteps to use (for efficiency)
        seed: Random seed for reproducible sharding and sampling

    Example:
        >>> dataset = LeRobotShardSource(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(8)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed timesteps
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        num_ffmpeg_threads: int = 1,
        shard_size: int = 2**10,  # 1024 steps
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        excluded_episode_ids: set[int] | None = None,
        boundary_policy: EpisodeBoundaryPolicy = EpisodeBoundaryPolicy.MASK,
    ):
        """Initialize the episode loader and its deterministic shard plan."""
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.boundary_policy = boundary_policy
        self.processor = None
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            num_ffmpeg_threads=num_ffmpeg_threads,
        )

        if excluded_episode_ids:
            known_ids = {row["episode_index"] for row in self.episode_loader.episodes_metadata}
            missing = excluded_episode_ids - known_ids
            if missing:
                raise ValueError(
                    f"Validation holdout episodes missing from {dataset_path}: {sorted(missing)}"
                )
            # Filter before planning so no shard can include a held-out frame.
            # Loader indices remain dense, while metadata retains physical IDs.
            self.episode_loader.episodes_metadata = [
                row
                for row in self.episode_loader.episodes_metadata
                if row["episode_index"] not in excluded_episode_ids
            ]
            self.episode_loader.episode_lengths = self.episode_loader.get_episode_lengths()
            print(
                f"Excluded {len(excluded_episode_ids)} validation episodes from training: {dataset_path}"
            )

        self.shard_plan = StepShardSampler(
            shard_size=shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=seed,
        ).build(self.episode_loader.episode_lengths, dataset_path=self.dataset_path)
        print(f"Generated {len(self.shard_plan.shards)} shards for dataset {self.dataset_path}")

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_plan.shards)

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        """
        Extract and process a single timestep from episode data.

        Converts raw episode data into a VLAStepData structure and applies
        the configured processor to create model-ready inputs.

        Args:
            episode_data: Complete episode DataFrame from LeRobotEpisodeLoader
            step_index: Timestep index within the episode to extract

        Returns:
            Processed datapoint ready for model training

        Raises:
            AssertionError: If processor is not set before calling this method
        """
        assert self.processor is not None, "Processor must be set before getting datapoints"
        vla_step_data = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            self.boundary_policy,
        )
        # Apply processor to convert to model inputs
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)

    def get_shard_length(self, idx: int) -> int:
        """Get the number of timesteps in a specific shard."""
        return int(self.shard_plan.lengths[idx])

    def get_shard(self, idx: int) -> list:
        """
        Load and process all timesteps in a specific shard.

        Loads the required episodes and extracts all timesteps assigned to this shard,
        applying the configured processor to each timestep.

        Args:
            idx: Shard index to load

        Returns:
            List of processed timesteps ready for model training
        """
        return self.get_shard_with_decode_strategy(idx, decode_strategy="union")

    def get_shard_with_decode_strategy(self, idx: int, decode_strategy: str) -> list:
        """Build a shard using ``full`` (reference) or ``union`` decoding.

        The explicit reference path exists for reproducibility/performance
        harnesses. Production callers continue through :meth:`get_shard` and
        therefore always use union decoding.
        """
        if decode_strategy not in ("full", "union"):
            raise ValueError(f"Unknown decode strategy: {decode_strategy}")

        episodes = self.shard_plan.shards[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            if decode_strategy == "full":
                episode_data = self.episode_loader[ep_idx]
            else:
                # Sampling is already fixed by shard_dataset. Decode only the
                # shared frame union that those exact samples will consume.
                video_indices = required_video_frame_indices(
                    step_indices,
                    self.episode_loader.get_episode_length(ep_idx),
                    self.modality_configs,
                )
                episode_data = self.episode_loader.load_episode(ep_idx, video_indices=video_indices)
            for step_index in step_indices:
                datapoints.append(self.get_datapoint(episode_data, step_index))
        return datapoints

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        """Get initial actions from the underlying episode loader."""
        return self.episode_loader.get_initial_actions()
