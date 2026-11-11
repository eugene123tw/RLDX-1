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

import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional

from rldx.data.types import EpisodeBoundaryPolicy, ModalityConfig

from .embodiment_configs import MODALITY_CONFIGS


@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset in a mixed-training setup.

    A list of these objects can be supplied in ``DataConfig.datasets`` to mix
    multiple datasets at arbitrary ratios. For convenience the
    single-dataset fields are also accepted; if ``datasets`` is non-empty
    they take precedence.
    """

    # Path to the dataset root directory (can be strings or dicts for complex configs)
    dataset_paths: List[Any]

    # Robot embodiment identifier (e.g. "gr1", "franka")
    embodiment_tag: Optional[str] = None

    # Relative sampling probability (will be normalised across the list)
    mix_ratio: float = 1.0

    dataset_type: str = "physical_embodiment"


@dataclass
class DataArgs:
    """Data settings the command line sets."""

    dataset_weight_by_steps: bool = False
    """
    If True, multiply each dataset's mix_ratio by ``len(dataset)`` (total
    step count) before normalisation, so that batches sample each dataset
    proportionally to its total steps. With this flag, mix_ratios act as
    *boost / suppress multipliers* on top of length-proportional sampling
    (e.g. ``--data.mix-ratios 1 1 0.2`` keeps the third dataset's
    natural step share but multiplies it by 0.2). When False (default),
    mix_ratios are used as raw weights regardless of dataset length —
    short datasets are sampled at the same rate as long ones.
    """

    projector_grouping: str | None = None
    """Optional projector-sharing policy. ``None`` preserves the tag mapping."""

    shard_size: int = 2**10
    """Size of the shard to use for dataset preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for episode steps."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited (sharded mode only)."""

    episode_boundary_policy: EpisodeBoundaryPolicy = EpisodeBoundaryPolicy.MASK
    """How action-chunk positions past the end of an episode are treated.

    ``MASK`` is the default and keeps the shipped behaviour. See
    ``EpisodeBoundaryPolicy`` for what each option does.
    """

    override_pretraining_statistics: bool = False
    """Use this run's dataset statistics even for tags the base checkpoint already has statistics for."""


@dataclass
class DataConfig(DataArgs):
    """Dataset configuration (supports single or multiple datasets)."""

    # Leave empty by default for backwards-compatibility with the original
    # single-dataset workflow.  Users can supply one or more configs via CLI or
    # YAML when they need mixing.
    datasets: List[SingleDatasetConfig] = field(default_factory=list)

    # Modality configs
    # There are three sources of modality configs:
    # 1. Default modality configs in code: rldx/configs/data/embodiment_configs.py
    # 2. Modality configs supplied through command line: --data.modality_configs (although rare and inconvenient)
    # 1 and 2 are unified through `config.data.modality_configs`.
    # 3. modality configs saved in the pretrained checkpoint.
    modality_configs: dict[str, dict[str, ModalityConfig]] = field(
        default_factory=lambda: copy.deepcopy(MODALITY_CONFIGS)
    )

    download_cache: bool = False

    # General task / mode config (shared across datasets)
    mode: str = "single_turn"

    # Data loading
    shuffle: bool = True
    seed: int = 42
    multiprocessing_context: str = "fork"  # Options: "fork", "spawn", and "forkserver"
