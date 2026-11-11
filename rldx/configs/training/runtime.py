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
from enum import Enum
from typing import Annotated, Optional

import tyro

from rldx.configs.execution import ExecutionBackend


class TrainingPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases logging."""

    project: str = "RLDX-1"
    """Project the run logs to."""


@dataclass
class TrainingArgs:
    """Training settings the command line sets."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str = "debug"

    max_steps: int = 30000
    """Total number of training steps to run before stopping."""

    global_batch_size: int = 64
    """Sum of per-device micro-batches across all GPUs; effective batch is this x gradient_accumulation_steps."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    execution_backend: Annotated[str, tyro.conf.arg(constructor=ExecutionBackend)] = "compile"
    """Tensor execution backend used to prepare the model for this runtime."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    lr_scheduler_type: str = "cosine"
    """Learning rate scheduler type."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    max_grad_norm: float = 1.0
    """Gradient clipping threshold (passed to ``TrainingArguments.max_grad_norm``)."""

    new_param_warmup_steps: int = 0
    """Number of initial training steps where only newly-added parameters (from motion module, memory,
    physics modules) are trained. Set to 0 to disable (all trainable params from step 0)."""

    optim: str = "adamw_torch_fused"
    """Optimizer choice forwarded to ``TrainingArguments.optim``. Default is
    the fused AdamW kernel (matches ``TrainingConfig`` default)."""

    skip_weight_loading: bool = False
    """Build the architecture but skip loading ``--base-model-path``'s weights.
    The processor (tokenizer/config) is still loaded from the base model, and
    the VLM backbone still loads its own pretrained weights, so the data
    pipeline and model construction paths run exactly as in a real run — only
    the checkpoint shard read is skipped. Intended for smoke-testing a config,
    dataset or collator change without paying for the checkpoint load.

    Use a fresh ``--output-dir``, or pass ``--no-resume-from-checkpoint``.
    Resume is on by default, so a checkpoint left behind by a real run would
    otherwise be resumed into an architecture that was never given its
    weights."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    save_only_model: bool = False
    """Save model weights only — no optimizer, scheduler or RNG state.

    Shrinks each checkpoint substantially — a 2-GPU ZeRO-2 smoke run measured
    61 GB down to 13 GB (~4.7x) — multiplied by ``--save-total-limit``. The
    exact ratio depends on the ZeRO stage and the optimizer. Worth it for a run
    that will never be resumed.

    Incompatible with ``--resume-from-checkpoint``, which is on by default, so
    this flag has to be paired with ``--no-resume-from-checkpoint``. Resuming
    from such a checkpoint under DeepSpeed fails outright — the engine looks
    for a ``global_step*/`` directory that was never written."""

    resume_from_checkpoint: bool = True
    """Resume from the newest ``checkpoint-*`` in ``--output-dir`` when one is
    there. On by default because a requeued or preempted SLURM job has to pick
    its run back up without the submitting script knowing it was interrupted.

    Pass ``--no-resume-from-checkpoint`` to start fresh against a directory
    that already holds checkpoints."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    validation: list[str] = field(default_factory=list)
    """Validation spec files (JSON). Each names its datasets, frozen samples or every
    episode, cadence, memory, comparison space and metrics; frozen samples are held
    out of training."""

    state_dropout_prob_final: float | None = None
    """If set, anneal state dropout from --state-dropout-prob to this value
    over the full --max-steps run. Default None means no schedule (constant
    dropout = --state-dropout-prob)."""

    state_dropout_schedule: str = "cosine"
    """Annealing curve when --state-dropout-prob-final is set: 'cosine' or
    'linear'. Ignored when --state-dropout-prob-final is None."""


@dataclass
class TrainingConfig(TrainingArgs):
    """Training configuration."""

    # Experiment tracking; None logs nowhere.
    wandb: WandbConfig | None = None

    # Base checkpoint the run starts from; None trains from scratch.
    start_from_checkpoint: Optional[str] = None

    batch_size: Optional[int] = None
    warmup_steps: int = 0  # This will override warmup_ratio

    # Hardware precision
    precision: TrainingPrecision = TrainingPrecision.BF16
    tf32: bool = True

    def __post_init__(self) -> None:
        self.precision = TrainingPrecision(self.precision)

    @property
    def fp16(self) -> bool:
        return self.precision is TrainingPrecision.FP16

    @property
    def bf16(self) -> bool:
        return self.precision is TrainingPrecision.BF16

    # Logging and saving
    logging_steps: int = 10

    # DeepSpeed (default)
    deepspeed_stage: int = 2  # ZeRO stage (only 2 is supported)
    gradient_checkpointing: bool = False

    # Transformers loading parameters
    transformers_trust_remote_code: bool = True
    transformers_local_files_only: bool = False
    transformers_cache_dir: str | None = None
    transformers_access_token: str | None = None  # Access token for HuggingFace Hub

    # Data handling
    remove_unused_columns: bool = False

    # Profiling
    enable_profiling: bool = False

    # For testing.
    assert_loss_less_than: float | None = None
