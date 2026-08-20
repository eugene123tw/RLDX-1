#!/usr/bin/env python
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

"""
Convert a HuggingFace LeRobot **v3.0** dataset (multi-episode chunked parquet +
video files, the format produced by current ``lerobot`` recording tools such
as SO-100 / SO-101 teleop) into the **per-episode v2.1-style** layout that
RLDX-1's ``LeRobotEpisodeLoader`` expects (one parquet + one mp4 per episode,
JSONL metadata).

This mirrors what ``download_pusht.py`` gets "for free" by pinning the
``v2.1`` tag on the Hub -- custom user datasets recorded with recent
``lerobot`` only exist in v3.0, so we re-chunk them ourselves.

Requires the ``lerobot`` package, which is NOT part of RLDX-1's main
environment (Python/deps clash -- see ``pyproject.toml``). Run this script
from a separate venv, e.g. reusing the one already declared for the SO100
real-robot eval path:

    cd rldx/eval/real_robot/SO100
    uv venv --python 3.10 .venv-lerobot
    uv pip install --python .venv-lerobot -e .
    cd -
    rldx/eval/real_robot/SO100/.venv-lerobot/bin/python \\
        run_scripts/data/convert_lerobot_v3_to_v2.py \\
        --repo-id eugene123tw/lerobot-so101-pick-and-place-potato \\
        --output-path examples/so101_pick_and_place_potato

The output only contains plain parquet/mp4/JSON files, so every later step
(``rldx/data/stats.py``, training) runs in RLDX-1's normal environment --
``lerobot`` is only needed for this one-time conversion.

Also writes ``meta/modality.json`` for the SO-101 6-DOF arm+gripper
convention used by ``rldx/configs/data/so101_config.py``:
state/action = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper] -> sliced into ``single_arm`` (5) + ``gripper`` (1).
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def _write_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    """Write an (T, H, W, 3) uint8 RGB frame stack to an h264 mp4 via ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        "18",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed (exit {ret}) writing {path}")


def convert(repo_id: str, output_path: str, v3_root: str | None = None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames

    out = Path(output_path)
    v3_root = v3_root or str(out) + ".v3_cache"

    print(f"Downloading/loading v3.0 dataset {repo_id!r} -> {v3_root}")
    ds = LeRobotDataset(repo_id, root=v3_root, video_backend="pyav")
    meta = ds.meta

    video_keys = meta.video_keys  # e.g. ["observation.images.arm", "observation.images.overhead"]
    # Logical (short) camera names used in modality.json / the python modality config.
    cam_names = {vk: vk.removeprefix("observation.images.") for vk in video_keys}

    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    for vk in video_keys:
        (out / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)

    raw_hf_dataset = ds.hf_dataset.with_format(None)

    episodes_lines = []
    for ep_idx in range(meta.total_episodes):
        ep = meta.episodes[ep_idx]
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        length = end - start
        print(f"[{ep_idx + 1}/{meta.total_episodes}] episode {ep_idx}: {length} frames")

        # --- tabular data (state/action/timestamp/task_index/...) ---
        df = raw_hf_dataset.select(range(start, end)).to_pandas()
        parquet_path = out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path)

        timestamps = np.asarray(df["timestamp"].tolist(), dtype=np.float64)

        # --- video: cut this episode's clip out of the concatenated per-chunk mp4 ---
        for vk in video_keys:
            from_ts = float(ep[f"videos/{vk}/from_timestamp"])
            shifted_ts = (timestamps + from_ts).tolist()
            video_path = meta.root / meta.get_video_file_path(ep_idx, vk)
            frames = decode_video_frames(video_path, shifted_ts, 1e-4, ds.video_backend)
            # (T, C, H, W) float in [0, 1] -> (T, H, W, C) uint8
            frames_np = (
                frames.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).contiguous().numpy()
            )
            out_video_path = out / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"
            _write_mp4(frames_np, out_video_path, meta.fps)

        episodes_lines.append(
            {"episode_index": ep_idx, "tasks": list(ep["tasks"]), "length": length}
        )

    # --- meta/episodes.jsonl ---
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for line in episodes_lines:
            f.write(json.dumps(line) + "\n")

    # --- meta/tasks.jsonl ---
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        for task_str, row in meta.tasks.iterrows():
            f.write(json.dumps({"task_index": int(row["task_index"]), "task": task_str}) + "\n")

    # --- meta/info.json ---
    total_episodes = meta.total_episodes
    total_frames = meta.total_frames
    features = dict(meta.features)
    info = {
        "codebase_version": "v2.1",
        "robot_type": meta.robot_type or "unknown",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(meta.tasks),
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": 1,
        "chunks_size": max(1000, total_episodes),
        "fps": meta.fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # --- meta/modality.json (SO-101: single_arm[0:5] + gripper[5:6]) ---
    modality = {
        "state": {
            "single_arm": {"start": 0, "end": 5},
            "gripper": {"start": 5, "end": 6},
        },
        "action": {
            "single_arm": {"start": 0, "end": 5},
            "gripper": {"start": 5, "end": 6},
        },
        "video": {
            cam_names[vk]: {"original_key": vk} for vk in video_keys
        },
    }
    with open(out / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    print(f"\nDone. RLDX-1-ready dataset written to: {out}")
    print("Next: generate meta/stats.json with rldx/data/stats.py (in the main RLDX-1 env).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF Hub dataset repo id, e.g. user/dataset")
    parser.add_argument("--output-path", required=True, help="Output dir for the converted dataset")
    parser.add_argument(
        "--v3-root",
        default=None,
        help="Where to cache the raw v3.0 download (default: <output-path>.v3_cache)",
    )
    args = parser.parse_args()
    convert(args.repo_id, args.output_path, args.v3_root)
