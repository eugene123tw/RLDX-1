"""Score a validation spec every ``spec.every`` optimizer steps."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from transformers.modeling_utils import unwrap_model
import wandb

from rldx.data.dataset.validation_episode_dataset import (
    Episode,
    ValidationEpisodeDataset,
    ValidationStep,
    group_ranges,
)
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.experiment.callbacks._mode_switch import validation_mode
from rldx.experiment.validation.spec import Memory, Metric, Space, ValidationSpec
from rldx.planner import BatchPlanner
from rldx.runtime.inference import InferenceCall, InferenceRuntime


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    tag: str
    pred: np.ndarray
    gt: np.ndarray
    ranges: list[tuple[str, int, int]]

    def errors(self, metric: Metric) -> tuple[float, float]:
        columns = [
            np.arange(start, end)
            for key, start, end in self.ranges
            if metric.keys is None or key in metric.keys
        ]
        error = self.pred[:, np.concatenate(columns)] - self.gt[:, np.concatenate(columns)]
        return float(np.abs(error).mean()), float((error**2).mean())


@dataclass(frozen=True)
class ScoredEpisode:
    episode: Episode
    frames: list[Frame]


def open_datasets(
    spec: ValidationSpec, modality_configs, action_horizon: int
) -> dict[str, ValidationEpisodeDataset]:
    return {
        tag: ValidationEpisodeDataset(
            dataset_path=str(source.path),
            modality_configs=modality_configs[tag],
            embodiment_tag=EmbodimentTag(tag),
            action_horizon=action_horizon,
        )
        for tag, source in spec.datasets.items()
    }


def _every_chunk(tag: str, dataset, action_horizon: int) -> list[Episode]:
    episodes = [
        Episode(
            tag,
            episode_id,
            length,
            tuple(range(0, length - action_horizon + 1, action_horizon)),
            None,
        )
        for episode_id, length in dataset.episodes()
        if length >= action_horizon
    ]
    if not episodes:
        raise ValueError(f"{tag}: no episode holds a full action chunk of {action_horizon}")
    return episodes


def balanced_assignments(episodes, world_size: int) -> list[list[Episode]]:
    """Deterministic LPT per embodiment: balance frames, keep each episode on one rank."""
    shards: list[list[Episode]] = [[] for _ in range(world_size)]
    total = [0] * world_size
    for tag in sorted({episode.tag for episode in episodes}):
        counts = [0] * world_size
        mine = [episode for episode in episodes if episode.tag == tag]
        for episode in sorted(mine, key=lambda e: (-len(e.starts), e.episode_id)):
            rank = min(range(world_size), key=lambda r: (counts[r], total[r], r))
            shards[rank].append(episode)
            counts[rank] += len(episode.starts)
            total[rank] += len(episode.starts)
    return shards


class ValidationCallback(TrainerCallback):
    def __init__(
        self,
        spec: ValidationSpec,
        datasets: Mapping[str, ValidationEpisodeDataset],
        *,
        processor,
        collate_fn,
        action_horizon: int,
        dtype: torch.dtype,
        use_wandb: bool,
    ):
        self.spec = spec
        self.datasets = datasets
        self.processor = processor
        # The training planner's routing descriptor needs GT action_mask, which
        # an observation-only call does not carry.
        self.collate_fn = (
            replace(collate_fn, metadata_planners=())
            if isinstance(collate_fn, BatchPlanner)
            else collate_fn
        )
        self.dtype = dtype
        self.use_wandb = use_wandb
        self.episodes = (
            list(spec.samples)
            if spec.samples is not None
            else [
                episode
                for tag, dataset in datasets.items()
                for episode in _every_chunk(tag, dataset, action_horizon)
            ]
        )
        self._cpu_group = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Every rank creates the group in the same order; results stay on CPU.
        if dist.is_available() and dist.is_initialized():
            self._cpu_group = dist.new_group(backend="gloo")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0 or step % self.spec.every:
            return control
        started = time.monotonic()
        model = unwrap_model(kwargs["model"])
        sharded = dist.is_available() and dist.is_initialized()
        rank, world_size = (dist.get_rank(), dist.get_world_size()) if sharded else (0, 1)
        with validation_mode(model, self.processor):
            # no_grad, not inference_mode: the model goes back to training after this.
            runtime = InferenceRuntime(
                model,
                dtype=self.dtype,
                tensor_image_path=self.processor.eval_image_spec,
                inference_mode=False,
            )
            scored = [
                self._score(model, runtime, episode)
                for episode in balanced_assignments(self.episodes, world_size)[rank]
            ]
        part = {"scored": scored, "seconds": time.monotonic() - started}
        parts = [part]
        if sharded:
            parts = [None] * world_size if state.is_world_process_zero else None
            dist.gather_object(part, parts, dst=0, group=self._cpu_group)
        if state.is_world_process_zero:
            self._report(parts, Path(args.output_dir), step)
        return control

    def _score(self, model, runtime, episode: Episode) -> ScoredEpisode:
        dataset = self.datasets[episode.tag]
        dataset.open(episode)
        carry = self.spec.memory is Memory.EPISODE
        memory, frames = None, []
        for t in episode.starts:
            step = dataset.step(t, self.processor)
            result = runtime.execute(
                InferenceCall(
                    planned_batch=self.collate_fn([step.record]),
                    memory=memory if carry else None,
                    reset_memory=(t == episode.starts[0] or not carry,)
                    if model.use_memory
                    else None,
                    manage_memory=model.use_memory,
                )
            )
            memory = result.memory
            prediction = result.action_pred.detach().float().cpu().numpy()
            frames.append(self._compare(prediction, step, dataset))
        dataset.close()
        return ScoredEpisode(episode, frames)

    def _compare(self, prediction: np.ndarray, step: ValidationStep, dataset) -> Frame:
        keys = dataset.action_keys
        if self.spec.space is Space.NORMALIZED:
            # The training representation, statistics and clipping; predictions already live here.
            gt = self.processor.state_action_processor.apply_action(
                step.gt, dataset.embodiment_tag.value, state=step.raw_state
            )
            # Heads pad to the longest horizon and widest action; keep this embodiment's part.
            width = sum(gt[key].shape[-1] for key in keys)
            pred = prediction[0, : len(gt[keys[0]]), :width]
        else:
            gt = step.gt
            decoded = self.processor.decode_action(
                prediction,
                dataset.embodiment_tag,
                {key: value[None].astype(np.float32) for key, value in step.raw_state.items()},
            )
            pred = np.concatenate([decoded[key][0] for key in keys], axis=-1)
        gt_array = np.concatenate([gt[key] for key in keys], axis=-1)
        if pred.shape != gt_array.shape:
            raise ValueError(f"Action shape mismatch: prediction {pred.shape}, GT {gt_array.shape}")
        valid = step.valid_horizon
        return Frame(
            dataset.embodiment_tag.value, pred[:valid], gt_array[:valid], group_ranges(gt, keys)
        )

    def _report(self, parts, output_dir: Path, step: int) -> None:
        scored = [episode for part in parts for episode in part["scored"]]
        frames = [frame for episode in scored for frame in episode.frames]
        values = {}
        for name, metric in self.spec.metrics.items():
            l1, mse = np.mean(
                [frame.errors(metric) for frame in frames if metric.covers(frame.tag)], axis=0
            )
            values[f"{self.spec.name}/{name}_L1"] = float(l1)
            values[f"{self.spec.name}/{name}_MSE"] = float(mse)
        directory = output_dir / "validation" / self.spec.name / f"step_{step:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        for item in scored:
            episode = item.episode
            np.savez_compressed(
                directory / f"{episode.tag}_{episode.episode_id}.npz",
                starts=np.asarray(episode.starts, dtype=np.int64),
                valid=np.asarray([len(frame.gt) for frame in item.frames], dtype=np.int64),
                pred=_stack([frame.pred for frame in item.frames]),
                gt=_stack([frame.gt for frame in item.frames]),
            )
        summary = {
            "step": step,
            "spec_sha256": self.spec.sha256,
            "rank_seconds": [part["seconds"] for part in parts],
            **values,
        }
        (directory / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        logger.info("Validation %s: %s", self.spec.name, summary)
        if self.use_wandb:
            wandb.log({"train/global_step": step, **values})


def _stack(chunks: list[np.ndarray]) -> np.ndarray:
    """Tail chunks are shorter; pad with NaN so the episode stays one array."""
    horizon = max(len(chunk) for chunk in chunks)
    out = np.full((len(chunks), horizon, chunks[0].shape[-1]), np.nan, dtype=np.float32)
    for i, chunk in enumerate(chunks):
        out[i, : len(chunk)] = chunk
    return out
