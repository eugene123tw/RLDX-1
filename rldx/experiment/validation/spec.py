"""What a validation scores and how, read once from a JSON file."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from rldx.data.dataset.validation_episode_dataset import Episode
from rldx.data.types import ModalityConfig


class Memory(Enum):
    EPISODE = "episode"
    RESET = "reset"


class Space(Enum):
    RAW = "raw"
    NORMALIZED = "normalized"


@dataclass(frozen=True)
class Source:
    path: Path
    metadata_sha256: Mapping[str, str] | None


@dataclass(frozen=True)
class Metric:
    tags: frozenset[str] | None
    keys: tuple[str, ...] | None

    def covers(self, tag: str) -> bool:
        return self.tags is None or tag in self.tags


@dataclass(frozen=True)
class ValidationSpec:
    name: str
    every: int
    memory: Memory
    space: Space
    datasets: Mapping[str, Source]
    samples: tuple[Episode, ...] | None
    metrics: Mapping[str, Metric]
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> ValidationSpec:
        raw = json.loads(Path(path).read_text())
        _fields(
            raw,
            {"name", "every", "memory", "space", "datasets", "metrics"},
            {"samples", "provenance"},
            str(path),
        )
        if raw["every"] <= 0:
            raise ValueError(f"{path}: every must be positive")
        space = Space(raw["space"])
        datasets = {}
        for tag, source in raw["datasets"].items():
            _fields(source, {"path"}, {"metadata_sha256"}, f"dataset {tag}")
            datasets[tag] = Source(Path(source["path"]), source.get("metadata_sha256"))
        samples = None
        if "samples" in raw:
            samples = tuple(_episode(sample, datasets) for sample in raw["samples"])
            ids = [(episode.tag, episode.episode_id) for episode in samples]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{path}: an episode is selected twice")
        if not raw["metrics"]:
            raise ValueError(f"{path}: at least one metric is required")
        metrics = {name: _metric(name, m, datasets, space) for name, m in raw["metrics"].items()}
        if samples is not None:
            for name, metric in metrics.items():
                if not any(metric.covers(episode.tag) for episode in samples):
                    raise ValueError(f"{path}: metric {name} covers no sample")
        return cls(
            name=raw["name"],
            every=raw["every"],
            memory=Memory(raw["memory"]),
            space=space,
            datasets=datasets,
            samples=samples,
            metrics=metrics,
            sha256=digest(raw),
        )

    def check(
        self,
        *,
        train_paths: set[Path],
        modality_configs: Mapping[str, Mapping[str, ModalityConfig]],
        action_horizon: int,
    ) -> None:
        """Refuse a spec this run cannot score honestly, before training starts."""
        for tag, source in self.datasets.items():
            if tag not in modality_configs:
                raise ValueError(f"{self.name}: {tag} has no modality config in this run")
            trained = source.path.resolve() in train_paths
            if self.samples is not None and not trained:
                raise ValueError(
                    f"{self.name}: {source.path} is not a training dataset, so its samples "
                    "cannot be held out; name the build this run trains on"
                )
            if self.samples is None and trained:
                raise ValueError(f"{self.name}: this run trains on {source.path}")
            if source.metadata_sha256 is not None:
                current = metadata_digests(source.path, source.metadata_sha256)
                if current != source.metadata_sha256:
                    raise ValueError(f"{self.name}: {source.path}/meta changed since selection")
        for name, metric in self.metrics.items():
            for tag in metric.tags or self.datasets:
                missing = set(metric.keys or ()) - set(
                    modality_configs[tag]["action"].modality_keys
                )
                if missing:
                    raise ValueError(
                        f"{self.name}: metric {name} keys {sorted(missing)} not in {tag}"
                    )
        if self.memory is Memory.EPISODE and self.samples is not None:
            for episode in self.samples:
                if any(b - a != action_horizon for a, b in zip(episode.starts, episode.starts[1:])):
                    raise ValueError(
                        f"{self.name}: carried memory needs chunks {action_horizon} apart "
                        f"({episode.tag}/{episode.episode_id})"
                    )


def digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def metadata_digests(root: str | Path, names) -> dict[str, str]:
    return {
        name: hashlib.sha256((Path(root) / "meta" / name).read_bytes()).hexdigest()
        for name in names
    }


def _fields(raw: dict, required: set[str], optional: set[str], where: str) -> dict:
    unknown = raw.keys() - required - optional
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"{where}: missing field(s) {sorted(missing)}")
    return raw


def _episode(raw: dict, datasets: Mapping[str, Source]) -> Episode:
    _fields(raw, {"tag", "episode_id", "episode_length", "frame_starts"}, {"task"}, "sample")
    tag, starts, length = raw["tag"], tuple(raw["frame_starts"]), raw["episode_length"]
    where = f"sample {tag}/{raw['episode_id']}"
    if tag not in datasets:
        raise ValueError(f"{where}: {tag} is not one of the datasets")
    if not starts:
        raise ValueError(f"{where}: frame_starts is empty")
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError(f"{where}: frame_starts must be increasing")
    if starts[0] < 0 or starts[-1] >= length:
        raise ValueError(f"{where}: frame_starts must lie in [0, episode_length)")
    return Episode(tag, raw["episode_id"], length, starts, raw.get("task"))


def _metric(name: str, raw: dict, datasets: Mapping[str, Source], space: Space) -> Metric:
    _fields(raw, set(), {"tags", "keys"}, f"metric {name}")
    tags = frozenset(raw["tags"]) if "tags" in raw else None
    keys = tuple(raw["keys"]) if "keys" in raw else None
    if tags is not None and (not tags or tags - datasets.keys()):
        raise ValueError(f"metric {name}: tags {sorted(tags)} must name datasets of this spec")
    if keys is not None and not keys:
        raise ValueError(f"metric {name}: keys, when given, must not be empty")
    if space is Space.RAW and len(tags if tags is not None else datasets) != 1:
        raise ValueError(f"metric {name}: raw space cannot average embodiments of different units")
    return Metric(tags, keys)


def check_names(specs) -> None:
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Validation specs share a name, so their results would collide: {names}")


def held_out(specs) -> dict[Path, frozenset[int]]:
    excluded: dict[Path, set[int]] = {}
    for spec in specs:
        for episode in spec.samples or ():
            path = spec.datasets[episode.tag].path.resolve()
            excluded.setdefault(path, set()).add(episode.episode_id)
    return {path: frozenset(ids) for path, ids in excluded.items()}


def held_out_digest(excluded: Mapping[Path, frozenset[int]]) -> str | None:
    if not excluded:
        return None
    return digest(sorted([str(path), sorted(ids)] for path, ids in excluded.items()))
