"""Pure cost observation and assignment planning for batch routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import torch

from rldx.planner.token_layout import TokenBatchLayout


FEATURE_NAMES = (
    "valid_tokens",
    "padded_tokens",
    "action_elements",
    "vision_patches",
    "image_count",
)
SCORE_SCALE = 1_000_000
COEFFICIENTS_PATH = Path(__file__).with_name("routing_coefficients.json")
BATCH_ROUTING_DESCRIPTOR_KEY = "_rldx_batch_routing_descriptor"


class RoutingHostLayout(TypedDict):
    token_layout: TokenBatchLayout


class RoutingInputs(TypedDict):
    action_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    host_layout: RoutingHostLayout


class PreparedRoutingBatch(TypedDict):
    inputs: RoutingInputs


class BatchWork(NamedTuple):
    """Model-work features consumed by the routing cost model."""

    valid_tokens: int
    padded_tokens: int
    action_elements: int
    vision_patches: int
    image_count: int

    @classmethod
    def parse(cls, value: Any) -> BatchWork:
        if not isinstance(value, (tuple, list)) or len(value) != len(cls._fields):
            raise RuntimeError("batch routing batch is missing its host descriptor")
        if any(type(item) is not int for item in value):
            raise RuntimeError("batch routing descriptor fields must be integers")
        return cls(*value)


@dataclass(frozen=True, order=True)
class Slot:
    rank: int
    micro: int


@dataclass(frozen=True)
class Assignment:
    source: Slot
    destination: Slot
    score: int


@dataclass(frozen=True)
class RoutePlan:
    assignments: tuple[Assignment, ...]
    world: int
    group_size: int

    @classmethod
    def parse(
        cls,
        assignments: list[Assignment] | tuple[Assignment, ...],
        world: int,
        group_size: int,
    ) -> RoutePlan:
        expected = {Slot(rank, micro) for rank in range(world) for micro in range(group_size)}
        sources = [item.source for item in assignments]
        destinations = [item.destination for item in assignments]
        if len(sources) != len(expected) or set(sources) != expected:
            raise RuntimeError("batch routing plan is not a source-slot bijection")
        if len(destinations) != len(expected) or set(destinations) != expected:
            raise RuntimeError("batch routing plan is not a destination-slot bijection")
        return cls(tuple(assignments), world, group_size)

    def __iter__(self):
        return iter(self.assignments)

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index):
        return self.assignments[index]


@dataclass(frozen=True)
class Predictor:
    """Immutable routing cost model calibrated outside the training run.

    Training deliberately does not time batches or update coefficients online:
    synchronization would perturb overlap and divergent rank-local models could
    produce incompatible collective plans.
    """

    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def predict(self, work: BatchWork) -> float:
        return self.intercept + sum(
            coefficient * (value - mean) / scale
            for coefficient, value, mean, scale in zip(
                self.coefficients, work, self.means, self.scales
            )
        )

    def fixed_score(self, work: BatchWork) -> int:
        return round(self.predict(work) * SCORE_SCALE)


def load_predictor(path: Path = COEFFICIENTS_PATH) -> Predictor:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported batch routing predictor schema version")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError(
            "batch routing predictor feature names do not match the production schema"
        )
    vectors = [payload[name] for name in ("coefficients", "means", "scales")]
    if any(len(vector) != len(FEATURE_NAMES) for vector in vectors):
        raise RuntimeError("batch routing predictor coefficient width mismatch")
    intercept = float(payload["intercept"])
    coefficients, means, scales = (tuple(map(float, vector)) for vector in vectors)
    if not all(math.isfinite(value) for value in (intercept, *coefficients, *means, *scales)):
        raise RuntimeError("batch routing predictor values must be finite")
    if any(scale == 0 for scale in scales):
        raise RuntimeError("batch routing predictor scales must be non-zero")
    return Predictor(intercept, coefficients, means, scales)


def make_route_plan(scores: list[list[int]]) -> RoutePlan:
    """Group similar-cost batches; prioritize rank locality over cost balance."""
    world = len(scores)
    group_size = len(scores[0]) if world else 0
    if (
        world == 0
        or group_size == 0
        or any(len(rank_scores) != group_size for rank_scores in scores)
    ):
        raise ValueError("batch routing requires a fixed positive group size")
    sources = [Slot(rank, micro) for rank in range(world) for micro in range(group_size)]
    ordered = sorted(
        sources,
        key=lambda slot: (-scores[slot.rank][slot.micro], slot.micro, slot.rank),
    )
    strata = []
    for micro in range(group_size):
        stratum = ordered[micro * world : (micro + 1) * world]
        assignments = []
        available_ranks = set(range(world))
        deferred = []
        for source in stratum:
            if source.rank in available_ranks:
                assignments.append(
                    Assignment(source, Slot(source.rank, micro), scores[source.rank][source.micro])
                )
                available_ranks.remove(source.rank)
            else:
                deferred.append(source)
        for source, rank in zip(deferred, sorted(available_ranks)):
            assignments.append(
                Assignment(source, Slot(rank, micro), scores[source.rank][source.micro])
            )
        strata.append(assignments)
    _balance_rank_totals_without_extra_transfers(strata, world)
    return RoutePlan.parse(
        [item for stratum in strata for item in stratum],
        world,
        group_size,
    )


def _balance_rank_totals_without_extra_transfers(
    strata: list[list[Assignment]], world: int
) -> None:
    totals = [0] * world
    for stratum in strata:
        for item in stratum:
            totals[item.destination.rank] += item.score
    while True:
        best = None
        best_delta = 0
        for stratum in strata:
            for left, a in enumerate(stratum):
                rank_a = a.destination.rank
                for right in range(left + 1, len(stratum)):
                    b = stratum[right]
                    rank_b = b.destination.rank
                    current_transfers = (a.source.rank != rank_a) + (b.source.rank != rank_b)
                    swapped_transfers = (a.source.rank != rank_b) + (b.source.rank != rank_a)
                    if swapped_transfers != current_transfers:
                        continue
                    cost_shift = b.score - a.score
                    variance_delta = 2 * cost_shift * (totals[rank_a] - totals[rank_b] + cost_shift)
                    if variance_delta < best_delta:
                        best_delta = variance_delta
                        best = stratum, left, right
        if best is None:
            return
        stratum, left, right = best
        a, b = stratum[left], stratum[right]
        totals[a.destination.rank] += b.score - a.score
        totals[b.destination.rank] += a.score - b.score
        stratum[left] = Assignment(a.source, b.destination, a.score)
        stratum[right] = Assignment(b.source, a.destination, b.score)


def observe_batch_work(batch: PreparedRoutingBatch) -> BatchWork:
    """Derive host-only routing features from a prepared model batch."""
    inputs = batch["inputs"]
    action_mask = inputs["action_mask"]
    if not torch.is_tensor(action_mask) or action_mask.device.type != "cpu":
        raise ValueError("batch routing requires a host action_mask tensor")
    token_layout = inputs["host_layout"]["token_layout"]
    grid = inputs["image_grid_thw"]
    if not torch.is_tensor(grid) or grid.device.type != "cpu":
        raise ValueError("batch routing requires host image_grid_thw")
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("inputs.image_grid_thw must have shape [N, 3]")
    patches = int(grid.to(torch.int64).prod(dim=1).sum())
    image_count = int(grid.shape[0])
    return BatchWork(
        token_layout.valid_tokens,
        token_layout.routing_padded_tokens,
        int(action_mask.to(torch.int64).sum()),
        patches,
        image_count,
    )


def plan_batch_routing_metadata(batch: PreparedRoutingBatch) -> dict[str, BatchWork]:
    """Consume transient host facts and return the durable routing descriptor."""
    descriptor = observe_batch_work(batch)
    del batch["inputs"]["host_layout"]["token_layout"]
    return {BATCH_ROUTING_DESCRIPTOR_KEY: descriptor}
