"""Validation episodes of a LeRobot v2.1 dataset, read outside the training dataloader."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rldx.data.dataset.lerobot_episode_loader import (
    LEROBOT_INFO_FILENAME,
    LEROBOT_META_DIR_NAME,
    LEROBOT_MODALITY_FILENAME,
    LeRobotEpisodeLoader,
)
from rldx.data.dataset.lerobot_shard_source import extract_step_data, required_video_frame_indices
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import MessageType, ModalityConfig, VLAStepData


@dataclass(frozen=True)
class Episode:
    tag: str
    episode_id: int
    length: int
    starts: tuple[int, ...]
    task: str | None


@dataclass(frozen=True)
class ValidationStep:
    record: dict
    raw_state: dict[str, np.ndarray]
    gt: dict[str, np.ndarray]
    valid_horizon: int


def group_ranges(chunks: dict[str, np.ndarray], keys: list[str]) -> list[tuple[str, int, int]]:
    widths = [chunks[key].shape[-1] for key in keys]
    ends = np.cumsum(widths)
    return [(key, int(end - width), int(end)) for key, width, end in zip(keys, widths, ends)]


class ValidationEpisodeDataset:
    """Checked when built, so a misconfigured validation set fails before training starts."""

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: EmbodimentTag,
        action_horizon: int,
    ) -> None:
        self._dataset_path = Path(dataset_path)
        self._modality_configs = modality_configs
        self.embodiment_tag = embodiment_tag
        self._action_horizon = action_horizon

        self._preflight_validate()
        self._loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
        )
        if len(self._loader) == 0:
            raise ValueError(f"Validation dataset at {dataset_path} contains 0 episodes.")

        self._episode: pd.DataFrame | None = None
        self._obs_modality_configs: dict[str, ModalityConfig] = {
            k: v for k, v in modality_configs.items() if k != "action"
        }
        self.action_keys: list[str] = modality_configs["action"].modality_keys
        self._index = {
            row["episode_index"]: i for i, row in enumerate(self._loader.episodes_metadata)
        }

    def _preflight_validate(self) -> None:
        meta_dir = self._dataset_path / LEROBOT_META_DIR_NAME

        for fname in (LEROBOT_INFO_FILENAME, LEROBOT_MODALITY_FILENAME):
            fpath = meta_dir / fname
            if not fpath.exists():
                raise ValueError(f"Validation dataset pre-flight failed: {fpath} does not exist.")

        for required_key in ("state", "video", "language"):
            if required_key not in self._modality_configs:
                raise ValueError(
                    f"Validation dataset pre-flight failed: modality_configs is missing "
                    f"required key '{required_key}'."
                )

        if "action" not in self._modality_configs:
            raise ValueError(
                "Validation dataset pre-flight failed: modality_configs is missing 'action'."
            )

        offsets = list(self._modality_configs["action"].delta_indices)
        if offsets != list(range(self._action_horizon)):
            raise ValueError(
                f"Validation dataset pre-flight failed: action offsets {offsets} are not "
                f"0..{self._action_horizon - 1}, so a GT chunk would not be [t, t + H)."
            )

    def episodes(self) -> list[tuple[int, int]]:
        """(episode_index, length) of every episode."""
        return [
            (episode_id, self._loader.get_episode_length(i))
            for episode_id, i in self._index.items()
        ]

    def open(self, episode: Episode) -> None:
        """Load the frames ``episode.starts`` needs; the episode in use is the only one held."""
        self.close()
        index = self._index[episode.episode_id]
        if self._loader.get_episode_length(index) != episode.length:
            raise ValueError(f"Validation episode {episode.episode_id} length changed")
        frames = required_video_frame_indices(
            np.asarray(episode.starts), episode.length, self._modality_configs
        )
        df = self._loader.load_episode(index, video_indices=frames)
        if episode.task is not None:
            # Datasets carry several annotation variants; a frozen selection fixes the text.
            for key in self._modality_configs["language"].modality_keys:
                df[f"language.{key}"] = episode.task
        self._episode = df

    def step(self, t: int, processor: Any) -> ValidationStep:
        """The processed observation at ``t``, its raw state, and the GT action chunk."""
        episode_df = self._episode

        obs_modality_configs = deepcopy(self._obs_modality_configs)
        vla_step_data: VLAStepData = extract_step_data(
            episode_df, t, obs_modality_configs, self.embodiment_tag
        )
        vla_step_data.actions = {}

        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_record = processor(messages)

        # Match training boundary padding without exposing actions to inference.
        gt_step = extract_step_data(
            episode_df,
            t,
            {key: self._modality_configs[key] for key in ("action", "language")},
            self.embodiment_tag,
        )

        raw_state: dict[str, np.ndarray] = {
            k: v.astype(np.float32) for k, v in vla_step_data.states.items()
        }

        return ValidationStep(
            processed_record,
            raw_state,
            {key: gt_step.actions[key] for key in self.action_keys},
            gt_step.metadata["in_episode_action_horizon"],
        )

    def close(self) -> None:
        self._episode = None
