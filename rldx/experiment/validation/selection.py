"""Task-balanced held-out episodes, chosen from dataset metadata alone."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from rldx.experiment.validation.spec import digest


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _task_texts(episode: dict, tasks: dict[int, str]) -> list[str]:
    # IDs are local to each dataset; compare their text across datasets of a group.
    texts = [tasks[item] if isinstance(item, int) else item for item in episode["tasks"]]
    return sorted({" ".join(text.split()) for text in texts if text.strip()})


def select_samples(
    sources: dict[str, str],
    groups: dict[str, list[str]],
    *,
    top_k: int,
    seed: int,
    action_horizon: int,
    frame_stride: int,
) -> list[dict]:
    """One episode for each of a group's ``top_k`` most-recorded tasks, every frame_stride frame."""
    if min(top_k, action_horizon, frame_stride) <= 0:
        raise ValueError("top_k, action_horizon and frame_stride must be positive")
    group_of = {tag: group for group, tags in groups.items() for tag in tags}
    candidates = {group: defaultdict(list) for group in groups}
    for tag, path in sorted(sources.items()):
        meta = Path(path).resolve() / "meta"
        tasks = {row["task_index"]: row["task"] for row in _read_jsonl(meta / "tasks.jsonl")}
        episodes = _read_jsonl(meta / "episodes.jsonl")
        ids = [row["episode_index"] for row in episodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate episode IDs in {path}")
        for row in episodes:
            for task in _task_texts(row, tasks):
                candidates[group_of[tag]][task].append((tag, row["episode_index"], row["length"]))
    samples = []
    for group, by_task in candidates.items():
        # Count distinct episodes, including short ones; only multi-episode tasks
        # with at least one full action chunk are eligible.
        ranked = sorted(
            (
                task
                for task, rows in by_task.items()
                if len(rows) >= 2 and any(r[2] >= action_horizon for r in rows)
            ),
            key=lambda task: (-len(by_task[task]), task),
        )
        if len(ranked) < top_k:
            raise ValueError(f"{group}: only {len(ranked)} eligible tasks, need {top_k}")
        used = set()
        for task in ranked[:top_k]:
            rows = [r for r in by_task[task] if r[2] >= action_horizon and r[:2] not in used]
            if not rows:
                raise ValueError(f"{group}: no unused full-length episode for task {task!r}")
            tag, episode_id, length = min(rows, key=lambda r: digest([seed, group, task, *r[:2]]))
            used.add((tag, episode_id))
            samples.append(
                {
                    "tag": tag,
                    "episode_id": episode_id,
                    "episode_length": length,
                    "frame_starts": list(range(0, length, frame_stride)),
                    "task": task,
                }
            )
    return samples
