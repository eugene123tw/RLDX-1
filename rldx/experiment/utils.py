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
import os
from pathlib import Path
import shutil

from termcolor import colored
from transformers import TrainerCallback

from rldx.dist import rank_zero_print as _print


# `write_json_atomic` cleans up after an exception but not after a kill, so a
# temporary can outlive the writer and ride into every checkpoint from here.
_IGNORE_TMP = shutil.ignore_patterns("*.tmp")


# NOTE(MK): Model architecture keys based on `RLDXConfig`, excluding:
#   - Training-only keys (e.g., `tune_llm`, `state_dropout_prob`)
#   - Processing/augmentation keys (e.g., `image`, `augmentation`)
#   - Runtime keys (e.g., `model_dtype`, `load_bf16`)
#   - Inference keys (e.g., `num_inference_timesteps`, `noise_beta_alpha/beta/s`)
MODEL_ARCHITECTURE_KEYS = [
    "model_type",
    "model_name",
    "backbone_model_type",
    "backbone_embedding_dim",
    "n_cog_tokens",
    "memory",
    "use_video",
    "video_length",
    "video_stride",
    "motion",
    "physics",
    "hidden_size",
    "input_embedding_dim",
    "max_state_dim",
    "max_action_dim",
    "action_horizon",
    "select_layer",
    "max_num_embodiments",
]


def snapshot_model_config(config_model) -> dict:
    """Take a deep-copy snapshot of architecture-relevant fields from a model config."""
    return {key: copy.deepcopy(getattr(config_model, key, None)) for key in MODEL_ARCHITECTURE_KEYS}


def compare_model_configs(ckpt_snapshot: dict, final_cfg) -> dict:
    """Compare checkpoint model config snapshot with final model config.

    Returns a dict of {key: (old_val, new_val)} for any fields that differ.
    Logs differences as warnings (CLI overrides always take effect).
    """
    diffs = {}
    for key in MODEL_ARCHITECTURE_KEYS:
        ckpt_val = ckpt_snapshot.get(key)
        final_val = getattr(final_cfg, key, None)
        if ckpt_val != final_val:
            diffs[key] = (ckpt_val, final_val)

    if not diffs:
        _print(colored("[i] No model architecture changes detected from checkpoint.", "green"))
        return diffs

    msg_lines = ["[w] Model config differences (checkpoint -> CLI override):"]
    for key, (old, new) in diffs.items():
        msg_lines.append(f"  {key}: {old} -> {new}")
    _print(colored("\n".join(msg_lines), "yellow"))
    return diffs


# ---------------------------------------------------------------------------
# Trainer callbacks
# ---------------------------------------------------------------------------


class CheckpointFormatCallback(TrainerCallback):
    """Copies the run's config and processor directories into each checkpoint-N/ so it loads alone."""

    def __init__(self, dirs: tuple[Path, ...]):
        self.dirs = dirs

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            for src in self.dirs:
                print(f"Copying {src} to {checkpoint_dir / src.name}")
                shutil.copytree(
                    src, checkpoint_dir / src.name, dirs_exist_ok=True, ignore=_IGNORE_TMP
                )

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class NewParamWarmupCallback(TrainerCallback):
    """Trains only the parameters the checkpoint left unloaded for the first ``steps`` steps."""

    def __init__(self, steps: int, model, unloaded: set[str]):
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.warm = trainable & unloaded
        if not self.warm:
            raise ValueError(
                "--new-param-warmup-steps: the checkpoint left no trainable parameter unloaded"
            )
        self.steps = steps
        self.frozen = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.frozen = [
            p for name, p in model.named_parameters() if p.requires_grad and name not in self.warm
        ]
        for p in self.frozen:
            p.requires_grad_(False)
        _print(
            colored(
                f"\n[NewParamWarmup] {len(self.warm)} params train alone for {self.steps} steps, "
                f"{len(self.frozen)} frozen until then",
                "cyan",
            )
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.frozen and state.global_step >= self.steps:
            for p in self.frozen:
                p.requires_grad_(True)
            _print(
                colored(
                    f"\n[NewParamWarmup] Step {state.global_step}: {len(self.frozen)} params unfrozen",
                    "cyan",
                )
            )
            self.frozen = []


def assert_world_size_matches(config) -> None:
    """Check the launcher's GPU count against the process group actually formed.

    ``per_device`` is derived from ``--num-gpus`` but the trainer multiplies by
    the real world size, so if the two disagree the effective batch is off by
    their ratio -- silently, with a plausible loss curve.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != config.training.num_gpus:
        raise ValueError(
            f"--num-gpus is {config.training.num_gpus} but the process group has world size "
            f"{world_size}. per-device batch is derived from --num-gpus while the "
            "effective batch scales with the real world size, so these must agree. "
            "Launch through torchrun, or pass --num-gpus 1."
        )
