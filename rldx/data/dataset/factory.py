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

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from tqdm import tqdm

from rldx.configs.base_config import Config
from rldx.data.dataset.lerobot_shard_source import LeRobotShardSource
from rldx.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.interfaces import BaseProcessor
from rldx.data.stats import generate_rel_stats, generate_stats
from rldx.dist import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.

    Builds the canonical shard-planned dataset pipeline.
    """

    def __init__(self, config: Config, *, held_out: Mapping[Path, frozenset[int]]):
        self.config = config
        # Validation episodes by resolved path; ValidationSpec.check made every key a training path.
        self.held_out = held_out

    def _ensure_stats(self, dataset_path: str, embodiment_tag: str) -> None:
        """Generate dataset statistics if not already present (rank-0 only, then barrier)."""
        modality_configs = self.config.data.modality_configs[embodiment_tag]
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                generate_stats(dataset_path)
                generate_rel_stats(dataset_path, modality_configs)
        else:
            generate_stats(dataset_path)
            generate_rel_stats(dataset_path, modality_configs)
        barrier()

    def build(self, processor: BaseProcessor) -> ShardedMixtureDataset:
        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"

                self._ensure_stats(dataset_path, embodiment_tag)

                dataset = LeRobotShardSource(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    excluded_episode_ids=self.held_out.get(Path(dataset_path).resolve()),
                    boundary_policy=self.config.data.episode_boundary_policy,
                )

                datasets.append(dataset)

            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            weight_by_steps = self.config.data.dataset_weight_by_steps
            for dataset, relative_length, raw_length in zip(
                datasets, dataset_relative_lengths, dataset_lengths
            ):
                weight = relative_length * dataset_spec.mix_ratio
                if weight_by_steps:
                    # Multiply by raw step count so cross-dataset sampling
                    # is proportional to total steps; mix_ratio then acts as
                    # a boost/suppress multiplier on top of length-proportional
                    # sampling.
                    weight *= float(raw_length)
                all_datasets.append(dataset)
                all_weights.append(weight)

        return ShardedMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            weight_by_steps=weight_by_steps,
        )
