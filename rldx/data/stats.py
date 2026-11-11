#!/usr/bin/env python
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
Calculate dataset statistics for LeRobot datasets.
Note: Please update the `rldx/configs/data/embodiment_configs.py` file with the correct modality configurations for the dataset you are using before running this script.

Usage:
    python rldx/data/stats.py <dataset_path> <embodiment_tag>

Args:
    dataset_path: Path to the dataset.
    embodiment_tag: Embodiment tag to use to load modality configurations from `rldx/configs/data/embodiment_configs.py`.
"""

import contextlib
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rldx.configs.data.embodiment_configs import MODALITY_CONFIGS
from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from rldx.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from rldx.data.state_action.pose import EndEffectorPose, JointPose
from rldx.data.types import ActionRepresentation, ActionType, EmbodimentTag, ModalityConfig
from rldx.data.utils import to_json_serializable, write_json_atomic


LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"

REQUIRED_STATS = ("mean", "std", "min", "max", "q01", "q99")


def _has_required_relative_stats(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    for stat in REQUIRED_STATS:
        if stat not in entry:
            return False
    return True


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


@contextlib.contextmanager
def _dataset_lock(dataset_path: Path):
    """Serialise stats writes for one dataset across processes and nodes.

    Yields False when the lock cannot be taken; the caller still does the work.
    """
    lock_path = dataset_path / "meta" / ".stats.lock"
    fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
    except OSError as exc:
        print(f"[warn] could not lock {lock_path} ({exc}); proceeding unserialised")
        locked = False

    try:
        yield locked
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    """Whether ``dataset_path`` already holds the statistics a run will read."""
    stats_path = Path(dataset_path) / LE_ROBOT_STATS_FILENAME
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in REQUIRED_STATS:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(dataset_path: Path | str):
    """Fill in missing statistics, keeping every entry this cannot recompute.

    Only float features can be summarised here, so this merges rather than
    replaces: overwriting would drop the entries it cannot produce.
    """
    dataset_path = Path(dataset_path)
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "observation.velocity" in feature:  # NOTE(TY):SKIP VELOCITY for ALLEX
            continue
        # A zero in the declared shape is a placeholder; its rows do not stack
        # into an array and `calculate_dataset_statistics` would raise.
        if 0 in (le_features[feature].get("shape") or []):
            continue
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)
    if check_stats_validity(dataset_path, lowdim_features):
        return

    with _dataset_lock(dataset_path):
        # Re-check now that we hold the lock: another job may have just done it,
        # and repeating the parquet pass would only risk racing its result.
        if check_stats_validity(dataset_path, lowdim_features):
            return

        print(f"Generating stats for {str(dataset_path)}")
        parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        computed = calculate_dataset_statistics(parquet_files, lowdim_features)

        stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
        stats = {}
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
        stats.update(computed)
        write_json_atomic(stats_path, stats)


class RelativeActionLoader:
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        action_key: str,
    ):
        self.dataset_path = Path(dataset_path)
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        action, state = modality_configs["action"], modality_configs["state"]
        # Check action config
        assert action_key in action.modality_keys
        assert action.action_configs is not None, action
        self.action_config = action.action_configs[action.modality_keys.index(action_key)]
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=action.delta_indices,
            modality_keys=[action_key],
        )
        # Check state config
        state_key = self.action_config.state_key or action_key
        assert state_key in state.modality_keys
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=state.delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency
        assert (
            self.modality_configs["state"].delta_indices[-1]
            == self.modality_configs["action"].delta_indices[0]
        )
        self.loader = LeRobotEpisodeLoader(dataset_path, self.modality_configs)

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        df = self.loader[trajectory_id]

        # OPTIMIZATION: Extract columns once and convert to numpy arrays
        # This eliminates repeated DataFrame.__getitem__ and Series.__getitem__ calls
        if self.action_config.state_key is not None:
            state_key = f"state.{self.action_config.state_key}"
        else:
            state_key = f"state.{self.action_key}"
        action_key = f"action.{self.action_key}"

        # Convert to numpy arrays once - this is much faster than repeated pandas access
        state_data = df[state_key].values  # Shape: (episode_length, joint_dim)
        action_data = df[action_key].values  # Shape: (episode_length, joint_dim)
        trajectories = []
        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices)
        for i in range(usable_length):
            state_ind = self.modality_configs["state"].delta_indices[-1] + i
            action_inds = action_delta_indices + i
            last_state = state_data[state_ind]
            actions = action_data[action_inds]
            if self.action_config.type == ActionType.EEF:
                action_format = self.action_config.format
                reference_frame = EndEffectorPose.from_action_format(last_state, action_format)
                traj = EndEffectorActionChunk.from_array(actions, action_format).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(traj.to(action_format).astype(np.float32))
            elif self.action_config.type == ActionType.NON_EEF:
                reference_frame = JointPose(last_state)
                traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(np.stack([p.joints for p in traj.poses], dtype=np.float32))
            else:
                raise ValueError(f"Unknown ActionType: {self.action_config.type}")
        return trajectories

    def __len__(self) -> int:
        return len(self.loader)


def calculate_stats_for_key(
    dataset_path: Path | str,
    modality_configs: dict[str, ModalityConfig],
    group_key: str,
    max_episodes: int = -1,
) -> dict:
    loader = RelativeActionLoader(dataset_path, modality_configs, group_key)
    trajectories = []
    for episode_id in tqdm(range(len(loader)), desc=f"Loading trajectories for key {group_key}"):
        if max_episodes != -1 and episode_id >= max_episodes:
            break
        trajectories.extend(loader.load_relative_actions(episode_id))
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }


def generate_rel_stats(
    dataset_path: Path | str, modality_configs: dict[str, ModalityConfig]
) -> None:
    dataset_path = Path(dataset_path)
    action_config = modality_configs["action"]
    if action_config.action_configs is None:
        return
    action_keys = [
        key
        for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
        if action_config.rep == ActionRepresentation.RELATIVE
    ]
    stats_path = Path(dataset_path) / LE_ROBOT_REL_STATS_FILENAME
    with _dataset_lock(dataset_path):
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
        else:
            stats = {}
        missing = [k for k in sorted(action_keys) if not _has_required_relative_stats(stats.get(k))]
        if not missing:
            return
        for action_key in missing:
            print(f"Generating relative stats for {dataset_path} {action_key}")
            stats[action_key] = calculate_stats_for_key(dataset_path, modality_configs, action_key)
        write_json_atomic(stats_path, to_json_serializable(dict(stats)))


def main(dataset_path: Path | str, embodiment_tag: EmbodimentTag):
    generate_stats(dataset_path)
    if embodiment_tag.value in MODALITY_CONFIGS:
        generate_rel_stats(dataset_path, MODALITY_CONFIGS[embodiment_tag.value])


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
