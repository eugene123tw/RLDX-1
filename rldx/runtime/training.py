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

"""HF Trainer integration for training metrics."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.distributed as dist
from transformers.integrations.integration_utils import WandbCallback, rewrite_logs
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback

from rldx.configs.execution import TrainingRuntimeProfile
from rldx.dist import rank_zero_print as _print
from rldx.planner import BatchGroupPlanner, BatchPlanner, WorkerBatchDataset
from rldx.runtime.heap import freeze_static_heap, unfreeze_worker_heap
from rldx.runtime.tensor_image import TensorImages
from rldx.telemetry.train_metrics import (
    CudaMemoryStats,
    GcCollectionStats,
    RankMetricGather,
    StepMetrics,
)
from rldx.transport.device import AsyncDeviceTransfer
from rldx.transport.distributed import ExchangeChannel
from rldx.transport.shared_storage import SharedStorageCollate


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _OptStepClock(TrainerCallback):
    def __init__(self, metrics: StepMetrics):
        self._metrics = metrics

    def on_step_end(self, args, state, control, **kwargs):
        self._metrics.finish_step()

    def on_save(self, args, state, control, **kwargs):
        self._metrics.skip_next_step()

    def on_evaluate(self, args, state, control, **kwargs):
        self._metrics.skip_next_step()


class _GcFreezeCallback(TrainerCallback):
    """Freeze the parent heap once the loader workers have forked.

    Only the first freeze belongs here.  Whatever repopulates the heap later
    freezes it again at its own boundary -- see ``freeze_static_heap``.
    """

    def __init__(self):
        self._frozen = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._frozen:
            freeze_static_heap()
            self._frozen = True


def _wandb_keys(logs: dict[str, float]) -> dict[str, float]:
    """``rewrite_logs``, except ``perf/*`` and ``mem/*`` are not nested under ``train/``.

    They describe how the run executed rather than what it learned, so they
    get their own wandb sections instead of sharing one with the loss.
    """
    top_level = {k: v for k, v in logs.items() if k.startswith(("perf/", "mem/"))}
    rest = {k: v for k, v in logs.items() if k not in top_level}
    return {**rewrite_logs(rest), **top_level}


class _WandbCallback(WandbCallback):
    """HF 4.57 ``WandbCallback.on_log`` with ``_wandb_keys`` for key naming.

    The row is still one ``wandb.log`` call carrying ``train/global_step``,
    so ``perf/*`` and ``mem/*`` share the x-axis and step of the loss.
    """

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            self._wandb.log(
                {**_wandb_keys(non_scalar_logs), "train/global_step": state.global_step}
            )


def _replace_wandb_callback(callbacks: list[TrainerCallback]) -> None:
    """Swap report_to's ``WandbCallback`` in place, keeping its handler position."""
    for index, callback in enumerate(callbacks):
        if type(callback) is WandbCallback:
            callbacks[index] = _WandbCallback()


def _single_dataset_frame_count(d: Any) -> Optional[int]:
    """Raw single-step frame count of one (non-mixture) dataset, or ``None``.

    - Sharded sources (``LeRobotShardSource``): ``__len__`` is the shard
      *count*, so frames = sum of per-shard lengths (``get_shard_length``).
      This sum equals the dataset's ``total_steps`` (every episode timestep).
    """
    if hasattr(d, "get_shard_length") and hasattr(d, "__len__"):
        try:
            return int(sum(d.get_shard_length(i) for i in range(len(d))))
        except (TypeError, IndexError, AttributeError):
            return None
    return None


def _compute_total_frames(dataset: Any) -> Optional[int]:
    """Best-effort count of raw single-step frames in the training dataset.

    Used as the denominator for the frame-based ``epoch`` metric
    (``epoch = global_step * effective_global_batch / total_frames``).

    Counts unique ``(episode, step)`` samples *before* any mix-ratio
    over/under-sampling. ``ShardedMixtureDataset`` exposes the underlying
    shard loaders via ``.datasets``;
    we sum each sub-dataset's raw frame count (see ``_single_dataset_frame_count``)
    rather than ``len(mixture)`` (post-resample / shard count). Returns ``None``
    when any sub-count is unavailable so callers can skip epoch logging.
    """
    sub_datasets = getattr(dataset, "datasets", None)
    if sub_datasets is not None:
        counts = [_single_dataset_frame_count(d) for d in sub_datasets]
        if any(c is None for c in counts):
            return None
        return int(sum(counts))
    return _single_dataset_frame_count(dataset)


class TrainingRuntime(Trainer):
    """Training runtime built on Hugging Face Trainer.

    Owns the training input lifecycle, asynchronous device transfer, frozen
    module policy, and the integration with the underlying training engine.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        self.batch_router = kwargs.pop("batch_router", None)
        self.train_batch_planner = kwargs.pop("train_batch_planner")
        data_collator = kwargs.get("data_collator")
        if not isinstance(data_collator, BatchPlanner):
            raise TypeError(
                f"TrainingRuntime data_collator must be BatchPlanner, got {type(data_collator).__name__}"
            )
        if isinstance(self.train_batch_planner, BatchGroupPlanner):
            self._batch_group_size = self.train_batch_planner.group_size
        elif isinstance(self.train_batch_planner, BatchPlanner):
            self._batch_group_size = None
        else:
            raise TypeError(
                "TrainingRuntime train_batch_planner must be BatchPlanner or BatchGroupPlanner"
            )
        if self.batch_router is not None and self._batch_group_size is None:
            raise TypeError("batch routing requires BatchGroupPlanner")
        runtime_profile = kwargs.pop("runtime_profile", TrainingRuntimeProfile())
        tensor_image_path = kwargs.pop("tensor_image_path")
        self._input_dtype = kwargs.pop("input_dtype")
        model = kwargs.get("model", args[0] if args else None)
        if model is None:
            raise ValueError("TrainingRuntime requires model as a keyword argument")
        prepare_for_execution = getattr(model, "prepare_for_execution", None)
        if not callable(prepare_for_execution):
            raise TypeError("TrainingRuntime model must implement prepare_for_execution(profile)")
        prepare_for_execution(runtime_profile)
        super().__init__(
            *args,
            **kwargs,
        )
        self._images = TensorImages(tensor_image_path, self.args.device)

        # Raw frame count for the frame-based ``epoch`` metric (see ``log``).
        # Computed once; ``None`` when unavailable (IterableDataset shards).
        self._total_frames = _compute_total_frames(self.train_dataset)
        if self.args.local_rank in (-1, 0):
            _print(f"[i] Total raw frames (epoch denominator): {self._total_frames}")

        self._step_metrics = StepMetrics()
        self._gc_stats = GcCollectionStats()
        self._cuda_metrics = CudaMemoryStats()
        self._rank_metrics = RankMetricGather()
        self.add_callback(_OptStepClock(self._step_metrics))
        self.add_callback(_GcFreezeCallback())
        _replace_wandb_callback(self.callback_handler.callbacks)
        self._exchange_channel = None

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        timed_iterator = self._step_metrics.timed_iterator(epoch_iterator)
        result = super().get_batch_samples(timed_iterator, num_batches, device)
        if self.batch_router is None:
            self._step_metrics.add_batches(result[0])
            return result
        if self.model_accepts_loss_kwargs is not False:
            raise RuntimeError("batch routing requires model_accepts_loss_kwargs=False")
        if num_batches != self._batch_group_size:
            raise RuntimeError(
                f"batch routing expected {self._batch_group_size} batches, got {num_batches}"
            )
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("batch routing requires torch.distributed")

        batch_samples, num_items_in_batch = result
        if num_items_in_batch is not None:
            raise RuntimeError("batch routing requires num_items_in_batch=None")
        if self._exchange_channel is None:
            self._exchange_channel = ExchangeChannel.host_local(device)
        with self._step_metrics.routing():
            routed = self.batch_router(batch_samples, self._exchange_channel)
        self._step_metrics.add_batches(routed)
        return routed, num_items_in_batch

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide HF's auto epoch (misleading for Iterable datasets), then inject a
        # frame-based epoch in its place: how many times the optimizer has swept
        # the raw dataset = global_step * effective_global_batch / total_frames.
        # Logged at wandb step == global_step (WandbCallback uses state.global_step).
        epoch = self.state.epoch
        self.state.epoch = None
        total_frames = self._total_frames
        if total_frames is not None and total_frames > 0:
            eff_global_batch = self.get_total_train_batch_size(self.args)
            # Log global_step explicitly and place epoch right after it so the
            # logged record / wandb panels order as (... , global_step, epoch).
            logs["global_step"] = self.state.global_step
            logs["epoch"] = self.state.global_step * eff_global_batch / total_frames
        # HF 4.57's periodic train log has "loss"; eval, final, and our
        # train_accuracy logs do not. Drain windows only with that train log.
        training_log = "loss" in logs
        if training_log:
            local_metrics = self._step_metrics.log_values()
            local_metrics.update(self._gc_stats.log_values())
            if self.args.device.type == "cuda":
                local_metrics.update(self._cuda_metrics.log_values(self.args.device))
            logs.update(local_metrics)
            logs.update(self._rank_metrics.gather(local_metrics))
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch
        if training_log:
            self._step_metrics.resume_after_log()

    def get_train_dataloader(self):  # noqa: D401
        """Return a dataloader without skipping data on resume; reseeds the dataset instead.

        ``ShardedMixtureDataset`` manages iteration and distributed ordering;
        no external sampler is added.
        """
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        _print(f"[i] Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            _print(
                f"[i] Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        _print("[i] Creating custom train dataloader")
        data_collator = self.train_batch_planner
        if isinstance(data_collator, BatchGroupPlanner):
            loader_batch_size = None
            loader_collator = data_collator.batch_planner
            loader_dataset = WorkerBatchDataset(self.train_dataset, data_collator.micro_batch_size)
        elif isinstance(data_collator, BatchPlanner):
            loader_batch_size = self._train_batch_size
            loader_collator = data_collator
            loader_dataset = self.train_dataset
        else:
            raise TypeError(
                f"TrainingRuntime requires a batch planner, got {type(data_collator).__name__}"
            )
        loader_collator = SharedStorageCollate(
            self._get_collator_with_removed_columns(loader_collator, description="training")
        )
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": loader_batch_size,
            "collate_fn": loader_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
            "worker_init_fn": unfreeze_worker_heap,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            raise TypeError("TrainingRuntime requires the canonical iterable shard dataset")

        loader = torch.utils.data.DataLoader(loader_dataset, **dataloader_params)
        if self.args.device.type == "cuda":
            if not self.args.dataloader_pin_memory:
                raise ValueError("CUDA H2D prefetch requires dataloader_pin_memory=True")
            loader = AsyncDeviceTransfer(loader, self.args.device)
            _print("[i] CUDA H2D batch prefetch enabled")
        return loader

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                _print(
                    f"[w] No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            _print(f"[i] Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        self._step_metrics.reset()
        if self.args.device.type == "cuda":
            self._cuda_metrics.reset(self.args.device)
        with self._gc_stats.recording(), self._rank_metrics.recording():
            return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        # The host input planner casts the other floating inputs to this dtype.
        self._images.to_pixel_values(inputs["inputs"], self._input_dtype)
        return inputs

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        self.loss = loss
        return (loss, outputs) if return_outputs else loss
