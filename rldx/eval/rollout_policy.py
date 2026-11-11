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

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
import os
from pathlib import Path
import random
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
import uuid

import gymnasium as gym
import numpy as np
import pandas as pd
from rldx.eval.sim import rldx_format
from rldx.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from tqdm import tqdm


if TYPE_CHECKING:
    from rldx.serving.server import Policy


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    # Per-env recording cap: env_idx stops writing mp4s after this many.
    # Auto-computed as n_episodes // n_envs when left None.
    max_episodes_per_env: int | None = None
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_robocasa_env_fn(
    env_name: str,
    seed: int = 0,
    robocasa_split: str = "pretrain",
):
    def env_fn():
        import os

        import robocasa  # noqa: F401
        import robosuite  # noqa: F401

        # gym_utils exists only in the Kitchen fork; RoboCasa365 has no such module.
        if not env_name.startswith("robocasa/"):
            from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        # ``robocasa/`` prefix envs are RoboCasa365 gymnasium envs that
        # require a split argument; the older ``robocasa_panda_omron/``
        # prefix uses enable_render + seed instead.
        if env_name.startswith("robocasa/"):
            return gym.make(env_name, split=robocasa_split, seed=seed)
        return gym.make(env_name, enable_render=True, seed=seed)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from rldx.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_gym_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    seed: int = 42,
    robocasa_split: str = "pretrain",
):
    """Create Ray environment factory function without wrappers."""

    if env_name.startswith("robocasa") or env_name.startswith("gr1_unified"):
        env_fn = get_robocasa_env_fn(env_name, seed=seed, robocasa_split=robocasa_split)
    elif env_name.startswith("libero"):
        env_fn = get_libero_env_fn(env_name)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    wrapper_configs: WrapperConfigs,
    start_episode_id: int = 0,
    seed: int = 42,
    robocasa_split: str = "pretrain",
    episode_id_queue: Optional[List[int]] = None,
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording). This is the GLOBAL
            index -- the caller has already applied env_idx_offset -- so seeds and video
            file names follow the lane.
        wrapper_configs: Configuration for environment wrappers
        episode_id_queue: Explicit episode ids for this env, in order. None counts up.
    Returns:
        Wrapped gymnasium environment
    """

    # Create base environment
    env_seed = seed + int(env_idx)

    # Important: In AsyncVectorEnv(spawn), each env is a separate process,
    # so for safety, we also fix the process-global RNG to avoid issues with np/random global RNG.
    random.seed(env_seed)
    np.random.seed(env_seed)
    print(
        f"[i] Creating environment {env_name} (env index: {env_idx}) with seed={env_seed}, "
        f"start_episode_id={start_episode_id}"
    )

    env = get_gym_env(env_name, env_idx, total_n_envs, seed=env_seed, robocasa_split=robocasa_split)

    # --- GR-1 tabletop reset protocol ---
    # Inert unless GR1_PROTOCOL_SEED is set. Must wrap the raw env.
    _gr1_seed = os.environ.get("GR1_PROTOCOL_SEED")
    if _gr1_seed:
        from gr1_protocol import GR1ProtocolWrapper

        env = GR1ProtocolWrapper(
            env,
            base_seed=int(_gr1_seed) + env_idx,
            start_episode_id=start_episode_id - 1,
            episode_id_queue=episode_id_queue,
        )

    if wrapper_configs.video.video_dir is not None:
        from rldx.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
            name_prefix=f"{env_name.replace('/', '_')}_env{env_idx:02d}",
            # Per-episode deterministic reset seed.
            base_seed=None if _gr1_seed else env_seed,
            seed_stride=100000,  # avoid collisions across env/episode ids
            start_episode_id=start_episode_id - 1,  # reset() will add 1, so set -1 here
            max_episodes_per_env=wrapper_configs.video.max_episodes_per_env,
            episode_id_queue=episode_id_queue,  # exact ids this env owes (hole-aware resume)
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """AsyncVectorEnv that tolerates variable-shaped info arrays across envs.

    Gymnasium's default _add_info pre-allocates a numpy array based on the
    first env's value shape and then assigns subsequent envs into it.  When
    envs return differently-shaped values (e.g. variable-length contact arrays)
    the assignment raises ValueError.  We catch that and fall back to a plain
    Python list for that key so the rest of the step can proceed normally.
    """

    def _add_info(self, infos, info, env_num):
        for k, v in info.items():
            if k not in infos:
                infos[k] = [None] * self.num_envs
                infos[f"_{k}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[k], np.ndarray):
                try:
                    infos[k][env_num] = v
                except (ValueError, TypeError):
                    lst = list(infos[k])
                    lst[env_num] = v
                    infos[k] = lst
            else:
                infos[k][env_num] = v
            infos[f"_{k}"][env_num] = True
        return infos


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: "Policy",
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    video_dir: str | None = None,
    seed: int = 42,
    robocasa_split: str = "pretrain",
    verbose: bool = False,
    env_layout: int = 0,
    env_idx_offset: int = 0,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
        env_layout: Lane count of the full layout. >0 marks this run as one chunk of it
            and enables hole-aware resume even at n_envs=1 (the repair pass). 0 keeps
            the plain global count-up.
        env_idx_offset: Lane id this chunk starts at.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    # Auto-set the per-env cap so recording splits evenly across envs.
    if (n_envs > 1 or env_layout > 0) and wrapper_configs.video.max_episodes_per_env is None:
        wrapper_configs.video.max_episodes_per_env = n_episodes // n_envs
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    existing_count, existing_successes, env_to_max_ep, env_to_recorded = (
        _load_existing_episode_metadata(n_envs, video_dir, env_idx_offset=env_idx_offset)
    )
    print(f"[i] Detected {existing_count} recorded episode(s) across envs; skipping them.")
    print(f"[i] Detected existing_successes: {existing_successes}")
    # Explicit per-env work list. None on the single-env path, which keeps counting up.
    env_missing_ids: Optional[List[List[int]]] = None
    if n_envs > 1 or env_layout > 0:
        target_per_env = int(n_episodes / n_envs)
        target_total = n_episodes

        env_missing_ids = [
            [ep for ep in range(target_per_env) if ep not in env_to_recorded.get(i, set())]
            for i in range(n_envs)
        ]
        env_episode_counts = [target_per_env - len(m) for m in env_missing_ids]

        total_existing = sum(env_episode_counts)

        if total_existing > 0:
            print(f"[i] Detected {total_existing} recorded episode(s) across envs; skipping them.")
        for i, missing in enumerate(env_missing_ids):
            if missing and missing != list(range(len(missing))):
                print(f"[i] Env {i}: refilling non-contiguous episode ids {missing}")

        if total_existing >= target_total:
            print("[i] Requested episodes per env already recorded. Nothing to run.")
            # Return success list truncated to the number of requested episodes.
            return env_name, existing_successes[:target_total]

        # First id each env will run; the wrapper gets the whole list and pops per reset.
        env_start_episode_ids = [m[0] if m else target_per_env for m in env_missing_ids]
        completed_episodes = total_existing
    else:
        # Basic mode: n_episodes = total (global) episode count
        target_per_env = None
        target_total = n_episodes
        if existing_count > 0:
            print(f"[i] Detected {existing_count} recorded episode(s); skipping them.")

        if existing_count >= n_episodes:
            print("[i] Requested episodes already recorded. Nothing to run.")
            return env_name, existing_successes[:n_episodes]

        # Calculate the episode index for each environment to start (previous maximum episode + 1)
        env_start_episode_ids = [env_to_max_ep.get(i, -1) + 1 for i in range(n_envs)]
        completed_episodes = existing_count

    env_fns = [
        partial(
            create_eval_env,
            # env_idx_offset shifts the GLOBAL env index; create_eval_env derives the
            # seed and the video file name from it.
            env_idx=idx + env_idx_offset,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
            start_episode_id=env_start_episode_ids[idx],
            seed=seed,
            robocasa_split=robocasa_split,
            episode_id_queue=(list(env_missing_ids[idx]) if env_missing_ids is not None else None),
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )
    translation = rldx_format.parse(
        env_name,
        policy.get_modality_config(),
        env.single_observation_space.keys(),
        env.single_action_space.keys(),
    )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = list(existing_successes)
    episode_infos = defaultdict(list)

    # Initial reset
    observations, _ = env.reset()
    policy.reset()
    i = 0

    # Add CSV file for recording simulation results
    csv_path = None
    if video_dir is not None:
        csv_path = f"{video_dir}/simulation_results.csv"

    env_episode_indices = env_start_episode_ids.copy()
    completed_episodes = existing_count
    env_pending: Optional[List[List[int]]] = (
        [list(m) for m in env_missing_ids] if env_missing_ids is not None else None
    )

    from rldx.serving.server import memory_options, remaining_chunk, require_cadence, rtc_options

    is_first_step = [True] * n_envs
    rtc = policy.get_rtc()
    memory = policy.get_memory()
    require_cadence(rtc, memory, wrapper_configs.multistep.n_action_steps)
    previous_chunk = None
    session_ids = [f"{env_name}_env{idx + env_idx_offset}" for idx in range(n_envs)]

    pbar = tqdm(total=n_episodes, desc="Episodes")
    # Multi-env stops on the per-env work list, which an index cap cannot
    # express: a resume may still owe episode 0 out of {0..9}.
    while True:
        if n_envs > 1:
            if all(not env_pending[i] for i in range(n_envs)):
                break
        else:
            if completed_episodes >= n_episodes:
                break

        options = {
            **rtc_options(rtc, is_first_step, previous_chunk),
            **memory_options(memory, session_ids, is_first_step),
        }

        if verbose:
            print(f"\n[CLIENT-LOG] === Step {i} ===")
            print(f"[CLIENT-LOG] Options to send: {options}")
            if isinstance(observations, dict):
                for k, v in observations.items():
                    if isinstance(v, np.ndarray):
                        print(f"[CLIENT-LOG] Obs[{k}] shape: {v.shape}, dtype: {v.dtype}")
                    elif isinstance(v, list) or isinstance(v, tuple):
                        print(f"[CLIENT-LOG] Obs[{k}] type: {type(v)}, len: {len(v)}")
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, np.ndarray):
                                print(
                                    f"[CLIENT-LOG] Obs[{k}][{sub_k}] shape: {sub_v.shape}, dtype: {sub_v.dtype}"
                                )
                            else:
                                print(f"[CLIENT-LOG] Obs[{k}][{sub_k}] type: {type(sub_v)}")
            else:
                print(f"[CLIENT-LOG] Obs type: {type(observations)}")

        actions, _ = policy.get_action(translation.observation(observations), options=options)
        if rtc is not None:
            previous_chunk = remaining_chunk(actions, rtc)

        is_first_step = [False] * n_envs
        next_obs, rewards, terminations, truncations, env_infos = env.step(
            translation.env_action(actions)
        )
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if env_pending is not None and not env_pending[env_idx]:
                    current_successes[env_idx] = False
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
                    is_first_step[env_idx] = True
                    continue

                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])

                if csv_path is not None:
                    result_stem = "success" if current_successes[env_idx] else "failure"
                    _global_env_idx = env_idx + env_idx_offset
                    _update_prediction_csv(
                        csv_path=csv_path,
                        env_idx=_global_env_idx,
                        episode_idx=env_episode_indices[env_idx],
                        success=current_successes[env_idx],
                        reward=current_rewards[env_idx],
                        steps=current_lengths[env_idx],
                        video_path=f"{env_name.replace('/', '_')}_env{_global_env_idx:02d}_episode{env_episode_indices[env_idx]:02d}_{result_stem}.mp4",
                    )

                if env_pending is not None:
                    if env_pending[env_idx]:
                        env_pending[env_idx].pop(0)
                    env_episode_indices[env_idx] = (
                        env_pending[env_idx][0] if env_pending[env_idx] else target_per_env
                    )
                else:
                    env_episode_indices[env_idx] += 1

                # Reset trackers for this environment.
                current_successes[env_idx] = False
                is_first_step[env_idx] = (
                    True  # Next step in this env will be the first step of a new episode
                )

                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    if video_dir is not None:
        with open(f"{video_dir}/summary.txt", "w") as f:
            f.write(
                f"Collecting {n_episodes} episodes took {time.time() - start_time:.2f} seconds\n"
            )
            f.write(f"Success rate: {np.mean(episode_successes):.2f}\n")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def _update_prediction_csv(
    csv_path: str,
    env_idx: int,
    episode_idx: int,
    success: bool,
    reward: float,
    steps: int,
    video_path: str = "",
):
    """Save results to CSV with (env_idx, episode_idx) as key (overwrite if exists)."""
    # Read existing file
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(
            columns=["env_idx", "episode_idx", "success", "reward", "steps", "video_path"]
        )

    # Remove same (env_idx, episode_idx) row
    if not df.empty:
        mask = ~((df["env_idx"] == env_idx) & (df["episode_idx"] == episode_idx))
        df = df[mask]

    # Add new row
    new_row = {
        "env_idx": env_idx,
        "episode_idx": episode_idx,
        "success": int(success),
        "reward": float(reward),
        "steps": int(steps),
        "video_path": video_path,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save again
    df.to_csv(csv_path, index=False)


def _load_existing_episode_metadata(
    n_envs: int,
    video_dir: str,
    env_idx_offset: int = 0,
) -> Tuple[int, List[bool], Dict[int, int], Dict[int, Set[int]]]:
    """
    Inspect video directory to determine how many episodes were already recorded.

    File name format (example):
    gr1_unified_..._Env_env00-episode_1-success.mp4
    gr1_unified_..._Env_env03-episode_12-failure.mp4

    `env_idx_offset` shifts file-name env ids into the local 0..n_envs-1 range, so a
    chunk that runs lane 3 alone bookkeeps it as its own env 0.

    Returns:
    - existing_count: number of existing episodes
    - existing_successes: list of success flags for completed episodes
    - env_to_max_ep: dictionary of maximum episode index for each environment (Dict[env_idx, max_episode_idx])
    - env_to_recorded: recorded episode indices per env (Dict[env_idx, Set[episode_idx]]).
      A set, not a max: a hole in the middle is invisible to max_ep + 1.
    """
    if video_dir is None:
        return 0, [], {i: -1 for i in range(n_envs)}, {i: set() for i in range(n_envs)}

    path = Path(video_dir)
    if not path.exists():
        return 0, [], {i: -1 for i in range(n_envs)}, {i: set() for i in range(n_envs)}
    # gr1_unified_..._Env_env00-episode_1-success.mp4
    pattern = re.compile(
        r".*_Env_env(?P<env>\d+)-episode_(?P<episode>\d+)-(?P<status>success|failure)\.mp4$"
    )

    # env-wise max episode index (for done_max_ep_id calculation)
    env_to_max_ep: Dict[int, int] = {i: -1 for i in range(n_envs)}
    # env-wise recorded episode indices (hole-aware resume)
    env_to_recorded: Dict[int, Set[int]] = {i: set() for i in range(n_envs)}
    # flat episode list: (episode_idx, env_idx, success_flag)
    episodes: List[Tuple[int, int, bool]] = []
    n_other_chunk = 0

    for file in path.glob("*.mp4"):
        match = pattern.match(file.name)
        if not match:
            print(f"[warn] Video name not recognised by the resume scan: {file.name}")
            continue

        env_idx = int(match.group("env")) - env_idx_offset
        episode_idx = int(match.group("episode"))  # 0-based or 1-based, consistent only
        status = match.group("status")
        success_flag = status == "success"

        if env_idx < 0 or env_idx >= n_envs:
            n_other_chunk += 1
            continue

        # Update the maximum episode index for each environment
        prev_max = env_to_max_ep.get(env_idx, -1)
        env_to_max_ep[env_idx] = max(prev_max, episode_idx)
        env_to_recorded.setdefault(env_idx, set()).add(episode_idx)

        # Add one flat episode
        episodes.append((episode_idx, env_idx, success_flag))

    if n_other_chunk:
        print(
            f"[i] Skipped {n_other_chunk} video(s) outside this call's env range "
            f"[{env_idx_offset}, {env_idx_offset + n_envs}) -- other chunk(s) of the layout."
        )

    if not episodes:
        return 0, [], env_to_max_ep, env_to_recorded

    # Sort by episode_idx, env_idx to create a deterministic global order
    episodes.sort(key=lambda x: (x[0], x[1]))

    # Total episode count
    total_episodes = len(episodes)
    existing_count = total_episodes

    # Flat success list (global episode idx = 0..total_episodes-1)
    existing_successes: List[bool] = [episodes[i][2] for i in range(total_episodes)]
    print(f"[i] Found {existing_count} total episodes")

    for env_idx in range(n_envs):
        max_ep = env_to_max_ep.get(env_idx, -1)
        if max_ep >= 0:
            print(f"[i] Env {env_idx}: max episode index = {max_ep} (next will be {max_ep + 1})")

    return existing_count, existing_successes, env_to_max_ep, env_to_recorded


def run_rldx_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    policy_client_host: str,
    policy_client_port: int,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    seed: int = 42,
    robocasa_split: str = "pretrain",
    verbose: bool = False,
    env_layout: int = 0,
    env_idx_offset: int = 0,
):
    from rldx.serving.server import PolicyClient

    if video_dir is None:
        video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"

    policy = PolicyClient(host=policy_client_host, port=policy_client_port)

    modality_configs = policy.get_modality_config()
    video_delta_indices = np.array(modality_configs["video"].delta_indices)
    state_delta_indices = (
        np.array(modality_configs["state"].delta_indices) if "state" in modality_configs else None
    )

    print(f"Using video_delta_indices: {video_delta_indices}")
    if state_delta_indices is not None:
        print(f"Using state_delta_indices: {state_delta_indices}")

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video_dir=video_dir,
        seed=seed,
        robocasa_split=robocasa_split,
        verbose=verbose,
        env_layout=env_layout,
        env_idx_offset=env_idx_offset,
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept both `--foo-bar` (docs style) and `--foo_bar` (code style).
    parser.add_argument(
        "--max-episode-steps",
        "--max_episode_steps",
        dest="max_episode_steps",
        type=int,
        default=504,
    )
    parser.add_argument("--n-episodes", "--n_episodes", dest="n_episodes", type=int, default=50)
    parser.add_argument(
        "--policy-client-host",
        "--policy_client_host",
        dest="policy_client_host",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--policy-client-port",
        "--policy_client_port",
        dest="policy_client_port",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--env-name",
        "--env_name",
        dest="env_name",
        type=str,
        default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    )
    parser.add_argument("--n-envs", "--n_envs", dest="n_envs", type=int, default=8)
    parser.add_argument(
        "--n-action-steps", "--n_action_steps", dest="n_action_steps", type=int, default=8
    )
    parser.add_argument("--video-dir", "--video_dir", dest="video_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--robocasa-split",
        "--robocasa_split",
        dest="robocasa_split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--env-layout",
        "--env_layout",
        dest="env_layout",
        type=int,
        default=0,
        help="Lane count of the full layout. >0 enables hole-aware resume even at "
        "--n_envs 1 (repair pass). 0 = plain global count-up.",
    )
    parser.add_argument(
        "--env-idx-offset",
        "--env_idx_offset",
        dest="env_idx_offset",
        type=int,
        default=0,
        help="Lane id this chunk starts at. Seeds and video file names follow the lane.",
    )

    args = parser.parse_args()

    results = run_rldx_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        seed=args.seed,
        robocasa_split=args.robocasa_split,
        verbose=args.verbose,
        env_layout=args.env_layout,
        env_idx_offset=args.env_idx_offset,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
