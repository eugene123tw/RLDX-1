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

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf
import yaml

from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    EpisodeBoundaryPolicy,
)
from rldx.data.utils import write_text_atomic

from .data.data_config import DataConfig
from .model import create_model_union_type, get_model_config
from .model.rldx import RLDXConfig
from .training.runtime import TrainingConfig


ModelUnionType = create_model_union_type()


@dataclass
class Config:
    """Complete configuration."""

    load_config_path: Optional[str] = None
    model: ModelUnionType = field(default_factory=lambda: RLDXConfig())
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration."""
        stage = self.training.deepspeed_stage

        rldx_dir = Path(__file__).parent.parent
        if stage == 2:
            config = json.load(open(rldx_dir / "configs/deepspeed/zero2_config.json"))
        else:
            raise ValueError(f"Invalid DeepSpeed stage: {stage}")

        return config

    def validate(self):
        """Validate configuration."""
        # Check dataset path(s)
        embodiment_tags = set()
        for d_cfg in self.data.datasets:
            # (Disable missing data check because we now support caching PDX data sources.)
            # if not Path(d_cfg.dataset_path).exists():
            #     raise ValueError(f"Dataset path does not exist: {d_cfg.dataset_path}")
            if d_cfg.dataset_type == "physical_embodiment" and not d_cfg.embodiment_tag:
                raise ValueError(f"Embodiment tag is empty for dataset {d_cfg.dataset_path}")
            if d_cfg.embodiment_tag is not None:
                embodiment_tags.add(d_cfg.embodiment_tag)
        stripped_modality_configs = {}
        for embodiment_tag in embodiment_tags:
            stripped_modality_configs[embodiment_tag] = self.data.modality_configs[embodiment_tag]
        self.data.modality_configs = stripped_modality_configs

        # ensure mix ratios are valid
        total_ratio = sum(d.mix_ratio for d in self.data.datasets)
        if total_ratio <= 0:
            raise ValueError("Sum of mix_ratio must be greater than zero")

        # Fill in default values for action configs
        for embodiment_tag in self.data.modality_configs:
            # Fill in default values for action representation, type and format
            if self.data.modality_configs[embodiment_tag]["action"].action_configs is None:
                self.data.modality_configs[embodiment_tag]["action"].action_configs = [
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ] * len(self.data.modality_configs[embodiment_tag]["action"].modality_keys)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_config(model_type: str) -> Config:
    model_config_type = get_model_config(model_type)
    return Config(model=model_config_type())


@dataclass(frozen=True)
class RunRecord:
    experiment_name: str
    max_steps: int
    # None: written before the field existed, so the policy is unknown.
    episode_boundary_policy: EpisodeBoundaryPolicy | None


def record_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "experiment_cfg" / "conf.yaml"


def write_run(config: Config, cfg_dir: Path, held_out: str | None) -> None:
    record = OmegaConf.create(config.__dict__)
    record["max_steps"] = config.training.max_steps
    record["save_steps"] = config.training.save_steps
    record["held_out"] = held_out
    cfg_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(cfg_dir / "conf.yaml", OmegaConf.to_yaml(record, resolve=True))


def read_run(run_dir: Path | str) -> RunRecord | None:
    path = record_path(run_dir)
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text())
    return RunRecord(
        experiment_name=raw["training"]["experiment_name"],
        max_steps=raw["training"]["max_steps"],
        episode_boundary_policy=EpisodeBoundaryPolicy[raw["data"]["episode_boundary_policy"]]
        if "episode_boundary_policy" in raw["data"]
        else None,
    )


def read_held_out(run_dir: Path | str) -> str | None:
    """Digest of the validation episodes the run held out of training; None if none."""
    path = record_path(run_dir)
    raw = yaml.safe_load(path.read_text()) if path.is_file() else {}
    if "held_out" not in raw:
        raise ValueError(
            f"{path} does not record which validation episodes were held out of training, "
            "so resuming could validate on trained episodes. Use a fresh --output-dir."
        )
    return raw["held_out"]
