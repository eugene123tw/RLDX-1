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

"""Replay Policy implementation for replaying actions from a dataset.

This module provides a policy that replays recorded actions from a LeRobot-style dataset,
with observation validation matching the RLDXPolicy interface.
"""

from pathlib import Path
from typing import Any

import numpy as np
from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import ModalityConfig
from rldx.serving.schema import Action, InputContract, OutputContract


class ReplayPolicy:
    """Policy that replays recorded actions from a LeRobot-style dataset.

    This policy loads actions from a dataset and replays them in sequence. It validates
    that incoming observations match the expected format (compatible with RLDXPolicy).

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the dataset's modality configuration.

    Example:
        >>> policy = ReplayPolicy(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...         "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        ...     },
        ...     exec_horizon=16,
        ...     embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        ... )
        >>> action, info = policy.get_action(observation)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        exec_horizon: int,
        *,
        embodiment_tag: EmbodimentTag,
        num_ffmpeg_threads: int = 1,
    ):
        """Initialize the Replay Policy.

        Args:
            dataset_path: Path to the LeRobot-style dataset directory
            modality_configs: Dictionary mapping modality names to ModalityConfig objects
                that specify temporal sampling and data keys to load
            exec_horizon: Actions the robot executes from each chunk; the replay advances by it.
            embodiment_tag: Embodiment the observations are parsed for, as in RLDXPolicy.
            num_ffmpeg_threads: FFmpeg threads per decoder.
        """
        self.dataset_path = Path(dataset_path)
        self.modality_configs = modality_configs
        self.episode_index = 0
        self.exec_horizon = exec_horizon

        # Validate modality configs
        for modality in ["video", "state", "action", "language"]:
            if modality not in modality_configs:
                raise ValueError(f"Modality config must contain '{modality}' key")

        # The policy can predict more steps than it executes.
        self.action_horizon = len(modality_configs["action"].delta_indices)
        if exec_horizon is None or not 1 <= exec_horizon <= self.action_horizon:
            raise ValueError(
                f"exec_horizon must be between 1 and the action horizon {self.action_horizon}, "
                f"got {exec_horizon}"
            )

        # Initialize episode loader for data access
        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            num_ffmpeg_threads=num_ffmpeg_threads,
        )

        # Load the episode data
        self.episode_data = self.episode_loader[self.episode_index]
        self.episode_length = len(self.episode_data)

        # Current step index in the episode
        self.current_step = 0

        # Preload all actions from the episode
        self._preload_actions()

        self._input_contract = InputContract(
            modality_configs=modality_configs,
            physics_keys=[],
            embodiment_tag=embodiment_tag,
            expected_state_dims={
                key: np.asarray(self.episode_data[f"state.{key}"].iloc[0]).shape[-1]
                for key in modality_configs["state"].modality_keys
            },
        )
        self._output_contract = OutputContract(
            modality_configs=modality_configs,
            action_dims={key: actions.shape[-1] for key, actions in self.actions.items()},
        )

    def _preload_actions(self) -> None:
        """Preload all actions from the current episode for efficient replay."""
        action_keys = self.modality_configs["action"].modality_keys
        self.actions: dict[str, np.ndarray] = {}

        for key in action_keys:
            col_name = f"action.{key}"
            if col_name in self.episode_data.columns:
                # Stack all actions: shape (episode_length, action_dim)
                action_list = []
                for i in range(self.episode_length):
                    action_array = np.array(self.episode_data[col_name].iloc[i]).astype(np.float32)
                    action_list.append(action_array)
                self.actions[key] = np.stack(action_list, axis=0)
            else:
                raise ValueError(f"Action key '{col_name}' not found in episode data")

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[Action, dict[str, Any]]:
        """Replay the next action chunk from the dataset.

        The observation is parsed exactly as RLDXPolicy parses it; only its batch size is used.

        Returns:
            Tuple of (actions_dict, info_dict) where actions_dict contains action chunks
            with shape (B, action_horizon, D) for each action key
        """
        if options:
            raise ValueError(f"ReplayPolicy reads no get_action options, got {sorted(options)}")
        batch_size = self._input_contract.parse(observation).batch_size

        action_horizon = self.action_horizon
        # Extract action chunk starting from current step
        action_chunk = {}
        for key, actions in self.actions.items():
            if self.current_step >= self.episode_length:
                # Past the end of episode: return last action repeated
                chunk = np.tile(actions[-1:], (action_horizon, 1))  # (action_horizon, D)
            else:
                end_step = self.current_step + action_horizon
                if end_step <= self.episode_length:
                    # Normal case: extract without padding
                    chunk = actions[self.current_step : end_step]  # (action_horizon, D)
                else:
                    # Near end of episode: pad with last action
                    remaining = self.episode_length - self.current_step
                    valid_chunk = actions[self.current_step :]  # (remaining, D)
                    padding = np.tile(
                        actions[-1:], (action_horizon - remaining, 1)
                    )  # (action_horizon - remaining, D)
                    chunk = np.concatenate([valid_chunk, padding], axis=0)

            # Expand to batch dimension: (action_horizon, D) -> (B, action_horizon, D)
            action_chunk[key] = np.tile(chunk[np.newaxis, :, :], (batch_size, 1, 1))

        info = {
            "current_step": self.current_step,
            "episode_length": self.episode_length,
            "episode_index": self.episode_index,
        }

        # Advance step counter by exec_horizon (the number of actions the policy will execute)
        self.current_step += self.exec_horizon

        return self._output_contract.parse(action_chunk), info

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to replay from the beginning of the episode.

        Args:
            options: Dictionary containing optional reset options:
                - episode_index: int - Switch to a different episode
                - step_index: int - Start from a specific step within the episode

        Returns:
            Dictionary containing info about the reset state
        """
        # Check if we should switch episodes
        if options is not None:
            if "episode_index" in options:
                new_episode_index = options["episode_index"]
                if new_episode_index != self.episode_index:
                    self.episode_index = new_episode_index
                    self.episode_data = self.episode_loader[self.episode_index]
                    self.episode_length = len(self.episode_data)
                    self._preload_actions()

            if "step_index" in options:
                self.current_step = options["step_index"]
            else:
                self.current_step = 0
        else:
            self.current_step = 0

        return {
            "episode_index": self.episode_index,
            "episode_length": self.episode_length,
            "current_step": self.current_step,
        }

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.modality_configs

    def get_rtc(self) -> None:
        return None

    def get_memory(self) -> None:
        return None

    @property
    def num_episodes(self) -> int:
        """Return the total number of episodes in the dataset."""
        return len(self.episode_loader)
