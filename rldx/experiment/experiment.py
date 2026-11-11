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

from functools import partial
import json
import logging
import os
from pathlib import Path
import warnings

import torch
import torch.distributed as dist
from transformers import TrainingArguments, set_seed
import wandb

from rldx.configs.base_config import Config, read_held_out, read_run, record_path, write_run
from rldx.configs.execution import TrainingRuntimeProfile
from rldx.configs.training.runtime import TrainingPrecision
from rldx.data.dataset.factory import DatasetFactory
from rldx.data.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions
from rldx.data.types import EpisodeBoundaryPolicy
from rldx.dist import barrier, get_global_rank as get_rank, rank_zero_print as _print
from rldx.experiment.callbacks.state_dropout_schedule import StateDropoutScheduleCallback
from rldx.experiment.loader import load_model, load_processor
from rldx.experiment.utils import CheckpointFormatCallback, NewParamWarmupCallback
from rldx.experiment.validation.callback import ValidationCallback, open_datasets
from rldx.experiment.validation.spec import ValidationSpec, check_names, held_out, held_out_digest
from rldx.model.core.processor import PROCESSOR_SUBDIR
from rldx.planner import BatchGroupPlanner, BatchPlanner
from rldx.planner.input.contracts import parse_routed_batch
from rldx.planner.input.qwen3vl import Qwen3VLInputPlanner
from rldx.planner.routing import Predictor, load_predictor, plan_batch_routing_metadata
from rldx.runtime.training import ProfCallback, TrainingRuntime
from rldx.transport.distributed import route_batches


#: Distributed timeout, in seconds. A slow shard read is a stall, not a hang.
DIST_TIMEOUT_SECONDS = 72_000


def _training_dtype(training_config):
    return {
        TrainingPrecision.FP32: torch.float32,
        TrainingPrecision.FP16: torch.float16,
        TrainingPrecision.BF16: torch.bfloat16,
    }[training_config.precision]


def _batch_routing_enabled(config: Config) -> bool:
    """Select canonical routing from the training structure, not a feature flag."""
    training = config.training
    return (
        training.num_gpus > 1
        and training.gradient_accumulation_steps > 1
        and training.deepspeed_stage == 2
    )


def _build_training_batch_planner(
    data_collator,
    *,
    dtype,
    routing_enabled=False,
    micro_batch_size=None,
    batch_group_size=None,
    backbone=None,
    predictor: Predictor | None = None,
):
    """Build the host-side training input contract."""
    if backbone is None:
        raise ValueError("the training input contract requires a Qwen3-VL backbone")
    batch_planner = BatchPlanner(
        collator=data_collator,
        input_planner=Qwen3VLInputPlanner(dtype=dtype, spec=backbone.input_spec),
        metadata_planners=(plan_batch_routing_metadata,) if routing_enabled else (),
    )
    if routing_enabled:
        if predictor is None:
            raise TypeError("batch routing requires an explicit Predictor")
        if micro_batch_size is None:
            raise ValueError("batch routing requires a micro batch size")
        if batch_group_size is None:
            raise ValueError("batch routing requires a batch group size")
        return BatchGroupPlanner(
            batch_planner,
            micro_batch_size,
            batch_group_size,
        )
    return batch_planner


def widen_default_collective_timeout(seconds: int = DIST_TIMEOUT_SECONDS) -> None:
    """Make sub-process-groups inherit the timeout, not torch's compiled default.

    Rebound in ``distributed_c10d``, not ``constants``: it holds its own
    reference and would never see the new value.
    """
    import datetime

    import torch.distributed.distributed_c10d as c10d

    c10d.default_pg_nccl_timeout = datetime.timedelta(seconds=seconds)
    c10d.default_pg_timeout = datetime.timedelta(seconds=seconds)


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def warn_configs(config: Config):
    # updates to batch size
    assert config.training.global_batch_size % config.training.num_gpus == 0, (
        "global_batch_size must be divisible by num_gpus"
    )

    if config.training.batch_size is not None:
        warnings.warn(
            "batch_size will be deprecated in the future, please use global_batch_size instead. For now, this will override global_batch_size."
        )

    if config.training.warmup_steps > 0:
        warnings.warn(
            "warmup_steps will be deprecated in the future, please use warmup_ratio instead. For now, this will override warmup_ratio."
        )

    if (
        config.model.image.crop_fraction is not None
        and not 0.0 < config.model.image.crop_fraction <= 1.0
    ):
        raise ValueError(
            f"random_crop_fraction must be in (0.0, 1.0], got {config.model.image.crop_fraction!r}"
        )

    if config.training.save_only_model and config.training.resume_from_checkpoint:
        raise ValueError(
            "save_only_model=True is incompatible with resume_from_checkpoint=True. "
            "A save_only_model checkpoint carries no optimizer, scheduler or RNG "
            "state, so there is nothing to resume from: under DeepSpeed the resume "
            'fails outright ("Can\'t find a valid checkpoint at ..." — the engine '
            "looks for a global_step*/ directory that was never written), and "
            "without DeepSpeed it silently restarts the optimizer and LR schedule "
            "at the restored global_step. Either drop --save-only-model, or start "
            "fresh with --no-resume-from-checkpoint."
        )


def _resumed_checkpoint(config: Config, output_dir: Path) -> Path | None:
    """The checkpoint this run continues from, if any.

    Resume is on by default, so relaunching with the same ``--output-dir``
    continues the previous run even with no base model named — and there the
    optimizer state carries over. ``--no-resume-from-checkpoint`` inherits nothing.
    """
    from transformers.trainer import get_last_checkpoint

    if not config.training.resume_from_checkpoint or not output_dir.exists():
        return None
    try:
        last_checkpoint = get_last_checkpoint(str(output_dir))
    except Exception as exc:
        raise ValueError(
            f"Could not inspect {output_dir} for existing checkpoints ({exc}). Training "
            "resumes from the last checkpoint in --output-dir, so what that run was "
            "trained with cannot be confirmed. Use a fresh --output-dir."
        ) from exc
    return Path(last_checkpoint) if last_checkpoint else None


def assert_resume_matches_boundary_policy(config: Config, output_dir: Path) -> None:
    """Refuse to auto-resume a run under a different episode-boundary policy.

    A requeue that forgets the flag would pair PAD weights with a MASK
    dataloader and raise nothing, so this fails closed, also when the record
    predates the field and cannot say which policy trained it.
    """
    last_checkpoint = _resumed_checkpoint(config, output_dir)
    if last_checkpoint is None:
        return

    current = EpisodeBoundaryPolicy(config.data.episode_boundary_policy)
    previous = read_run(last_checkpoint)
    if previous is None or previous.episode_boundary_policy is None:
        raise ValueError(
            f"{last_checkpoint} will be resumed from but {record_path(last_checkpoint)} does "
            f"not record it, so the episode_boundary_policy it was trained with cannot be confirmed against "
            f"the requested '{current.value}'. Use a fresh --output-dir to start a new run."
        )
    previous_policy = previous.episode_boundary_policy

    if previous_policy != current:
        raise ValueError(
            f"Refusing to resume {output_dir}: the existing checkpoints were trained "
            f"with episode_boundary_policy='{previous_policy.value}' "
            f"but this run asks for '{current.value}'. This run resumes from the last "
            "checkpoint in --output-dir, so continuing would mix the two policies in "
            "one model. Re-run with --episode-boundary-policy "
            f"{previous_policy.name}, or pick a different "
            "--output-dir to start a new run."
        )


def assert_resume_matches_held_out(config: Config, current: str | None, output_dir: Path) -> None:
    """Refuse to auto-resume when the episodes held out of training would change.

    Resuming with a different set would validate on episodes the checkpoint
    trained on, or start training on ones it held out.
    """
    last_checkpoint = _resumed_checkpoint(config, output_dir)
    if last_checkpoint is None:
        return
    previous = read_held_out(last_checkpoint)
    if previous != current:
        raise ValueError(
            f"Refusing to resume {output_dir}: its checkpoints held out validation "
            f"episodes {previous} but this run holds out {current}. Pass the same "
            "--validation specs, or pick a different --output-dir to start a new run."
        )


def run(config: Config):
    warn_configs(config)

    """Main training function."""
    # Before anything can make a group: every later group inherits the value.
    widen_default_collective_timeout()

    # If using distributed training, initialize the process group
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import datetime

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=DIST_TIMEOUT_SECONDS)
        )
        # only meaningful for torchrun, for ray it is always 0
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    # Setup
    setup_logging()
    set_seed(config.data.seed)

    # Validate config
    config.validate()

    # Create output directory
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / PROCESSOR_SUBDIR
    validations = [ValidationSpec.from_path(path) for path in config.training.validation]
    check_names(validations)
    train_paths = {
        Path(path).resolve() for dataset in config.data.datasets for path in dataset.dataset_paths
    }
    for spec in validations:
        spec.check(
            train_paths=train_paths,
            modality_configs=config.data.modality_configs,
            action_horizon=config.model.action_horizon,
        )
    excluded = held_out(validations)
    # Must run before the save below: resume is on by default, so relaunching
    # with the same --output-dir silently continues the previous run. Saving the
    # config first would also erase the evidence of what that run used.
    assert_resume_matches_boundary_policy(config, output_dir)
    assert_resume_matches_held_out(config, held_out_digest(excluded), output_dir)

    config.model._fill_missing_defaults()
    write_run(config, save_cfg_dir, held_out_digest(excluded))
    wandb_config = config.training.wandb
    if wandb_config is not None:
        with open(output_dir / "wandb_config.json", "w") as f:
            json.dump({"project": wandb_config.project, "run_id": experiment_name}, f)

    _print(f"\n[i] Saved config to {save_cfg_dir}")

    # Initialize wandb if configured, but only on the main process
    if wandb_config is not None and global_rank == 0:
        # Add git commit hash and version info to config
        config_dict = {
            **config.__dict__,
            "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown"),
        }

        wandb.init(
            project=wandb_config.project,
            name=os.environ.get("WANDB_RUN_NAME") or experiment_name,
            config=config_dict,
            tags=[config.data.mode],
        )

    model, unloaded = load_model(config)
    processor = load_processor(config)
    train_dataset = DatasetFactory(config, held_out=excluded).build(processor)
    # Rank-0 only, then barrier: every rank runs `run()` and would corrupt these.
    if get_rank() == 0:
        processor.save_pretrained(processor_dir)
    barrier()
    _print("[i] Model type:", type(model))

    # DeepSpeed on every GPU count: one precision regime (bf16 module, fp32
    # master copy in the optimizer) for training, validation, and serving.
    deepspeed_config = config.get_deepspeed_config()
    dtype = _training_dtype(config.training)
    # For now we will let batch_size override global_batch_size, in future we will deprecate batch_size
    if config.training.batch_size is None:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    else:
        per_device_train_batch_size = config.training.batch_size

    print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    routing_predictor = load_predictor() if _batch_routing_enabled(config) else None
    train_batch_planner = _build_training_batch_planner(
        processor.collator,
        dtype=dtype,
        routing_enabled=_batch_routing_enabled(config),
        micro_batch_size=per_device_train_batch_size,
        batch_group_size=config.training.gradient_accumulation_steps,
        backbone=getattr(model, "backbone", None),
        predictor=routing_predictor,
    )
    batch_planner = (
        train_batch_planner.batch_planner
        if isinstance(train_batch_planner, BatchGroupPlanner)
        else train_batch_planner
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        save_only_model=config.training.save_only_model,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="none" if wandb_config is None else "wandb",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_timeout=72000,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    # Create trainer
    trainer = TrainingRuntime(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=batch_planner,
        train_batch_planner=train_batch_planner,
        multiprocessing_context=config.data.multiprocessing_context,
        runtime_profile=TrainingRuntimeProfile(backend=config.training.execution_backend),
        tensor_image_path=processor.train_image_spec,
        input_dtype=dtype,
        batch_router=(
            partial(route_batches, predictor=routing_predictor, parse_batch=parse_routed_batch)
            if routing_predictor is not None
            else None
        ),
    )

    trainer.add_callback(CheckpointFormatCallback(dirs=(save_cfg_dir, processor_dir)))

    for spec in validations:
        callback = ValidationCallback(
            spec,
            open_datasets(spec, config.data.modality_configs, config.model.action_horizon),
            processor=processor,
            collate_fn=batch_planner,
            action_horizon=config.model.action_horizon,
            dtype=dtype,
            use_wandb=wandb_config is not None,
        )
        trainer.add_callback(callback)
        _print(
            f"[i] Validation {spec.name}: every {spec.every} steps, "
            f"{len(callback.episodes)} episodes, {spec.memory.value} memory, {spec.space.value} space"
        )

    # ── State-dropout annealing callback (default disabled) ────────────────
    sd_final = config.training.state_dropout_prob_final
    if sd_final is not None:
        sd_start = float(config.model.state_dropout_prob)
        sd_schedule = config.training.state_dropout_schedule
        trainer.add_callback(
            StateDropoutScheduleCallback(
                start=sd_start,
                end=float(sd_final),
                max_steps=config.training.max_steps,
                schedule=sd_schedule,
            )
        )
        _print(
            f"[i] StateDropoutScheduleCallback registered: "
            f"{sd_schedule} {sd_start:.3f} → {float(sd_final):.3f} "
            f"over {config.training.max_steps} steps"
        )

    if config.training.new_param_warmup_steps > 0:
        trainer.add_callback(
            NewParamWarmupCallback(config.training.new_param_warmup_steps, model, unloaded)
        )

    initial_actions = train_dataset.get_initial_actions()
    if initial_actions:
        initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
        save_initial_actions(initial_actions, initial_actions_path)
        _print(f"Saved {len(initial_actions)} initial actions to {initial_actions_path}")

    # Train
    _print("🚀 Starting training...")
    if config.training.enable_profiling:
        _print(f"{global_rank} Starting training with profiling...")

        def on_trace_ready_handler(trainer, profile_dir, prof):
            output_path = (
                profile_dir / f"trace_rank_{global_rank}_iter_{trainer.state.global_step}.json"
            )
            prof.export_chrome_trace(str(output_path))
            _print(f"Trace saved to {output_path}")

        profile_dir = output_dir / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)

        # The trace owns intra-step compute attribution only (H2D overlap,
        # kernel gaps); periodic loader stalls are owned by the always-on
        # perf/* and [LOADER] metrics, so a few steady-state steps suffice.
        # Stacks stay off: they inflate step time enough to distort the
        # input-bound verdict the trace exists to support.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=2, active=4, repeat=1),
            # profile_memory=True,
            with_stack=False,
            # record_shapes=True,
            on_trace_ready=partial(on_trace_ready_handler, trainer, profile_dir),
        ) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    # Save final model
    trainer.save_model()
    _print(f"Model saved to {output_dir}")

    if config.training.assert_loss_less_than is not None:
        final_loss = trainer.loss
        if final_loss.item() > config.training.assert_loss_less_than:
            raise AssertionError(
                f"Loss too high: {final_loss.item()} vs {config.training.assert_loss_less_than})"
            )

    _print("🚀 Training completed!")
