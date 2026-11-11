"""Worker-side composition of collation and pure batch planning."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch.utils.data import IterableDataset


PlannedBatch = TypeVar("PlannedBatch")


@dataclass(frozen=True)
class BatchPlanner(Generic[PlannedBatch]):
    """Collate samples, prepare model inputs, and attach planned metadata."""

    collator: Callable[[Sequence[dict[str, Any]]], Any]
    input_planner: Callable[[Any], PlannedBatch]
    metadata_planners: tuple[Callable[[PlannedBatch], Mapping[str, Any]], ...] = ()

    def __call__(self, features: Sequence[dict[str, Any]]) -> PlannedBatch:
        batch = self.input_planner(self.collator(features))
        if not isinstance(batch, MutableMapping):
            raise TypeError(f"planned batch must be mutable, got {type(batch).__name__}")
        for metadata_planner in self.metadata_planners:
            metadata = metadata_planner(batch)
            overlap = batch.keys() & metadata.keys()
            if overlap:
                raise ValueError(f"planned batch metadata keys already exist: {sorted(overlap)}")
            batch.update(metadata)
        return batch


@dataclass(frozen=True)
class BatchGroupPlanner(Generic[PlannedBatch]):
    """Configure worker-built batches for one routed accumulation group."""

    batch_planner: BatchPlanner[PlannedBatch]
    micro_batch_size: int
    group_size: int


class WorkerBatchDataset(IterableDataset):
    """Form physical batches inside each data-loader worker."""

    def __init__(self, dataset: IterableDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) < self.batch_size:
                continue
            yield batch
            batch = []
