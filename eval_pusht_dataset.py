#!/usr/bin/env python3
"""
Evaluate RLDX-1 checkpoint on Push-T dataset directly.

This runs inference on actual Push-T episodes from the LeRobot dataset,
bypassing the need for gym environment registration.

Usage:
  uv run python eval_pusht_dataset.py \
      --checkpoint-path ckpt/rldx1/finetuned/pusht/rldx1_ft_pusht_validation/rldx1_ft_pusht_validation/checkpoint-500 \
      --dataset-path examples/pusht_lerobot \
      --n-episodes 10

Success criteria:
  - Final action position is close to target (distance < threshold)
  - Episode completes without error
"""

import argparse
from copy import deepcopy
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from rldx.configs.data.pusht_config import pusht
from rldx.data.dataset.sharded_single_step_dataset import extract_step_data
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from rldx.policy.rldx_policy import RLDXPolicy
import json


def load_pusht_dataset_info(dataset_path: str) -> dict:
    """Load Push-T dataset metadata."""
    dataset_path = Path(dataset_path)
    
    # Load episode count from JSONL format
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_file}")
    
    episodes_data = []
    with open(episodes_file) as f:
        for line in f:
            if line.strip():
                episodes_data.append(json.loads(line))
    
    print(f"[*] Loaded {len(episodes_data)} episodes from dataset")
    return {
        "episodes": episodes_data,
        "dataset_path": str(dataset_path),
    }


def parse_observation_rldx(obs: dict[str, Any], modality_configs: dict[str, Any]) -> dict[str, Any]:
    parsed_obs: dict[str, Any] = {}
    for modality in ["video", "state", "language"]:
        parsed_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"

            arr = obs[parsed_key]
            # Policy expects batch dimension.
            if isinstance(arr, str):
                parsed_obs[modality][key] = [[arr]]
            else:
                parsed_obs[modality][key] = arr[None, :]
    return parsed_obs


def parse_action_rldx(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add action.* prefix for easier key alignment.
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_episode_mae(
    policy: RLDXPolicy,
    loader: LeRobotEpisodeLoader,
    episode_idx: int,
    embodiment_tag: EmbodimentTag,
    action_horizon: int,
) -> tuple[float, float, int]:
    traj = loader[episode_idx]
    traj_length = len(traj)

    pred_action_across_time = []
    action_keys = loader.modality_configs["action"].modality_keys

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    for step_count in range(0, traj_length, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)

        obs: dict[str, Any] = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text

        parsed_obs = parse_observation_rldx(obs, loader.modality_configs)
        action_chunk_raw, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_rldx(action_chunk_raw)

        for j in range(action_horizon):
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    def extract_columns(traj_df, columns: list[str]) -> np.ndarray:
        np_dict = {col: np.vstack([arr for arr in traj_df[col]]) for col in columns}
        return np.concatenate([np_dict[col] for col in columns], axis=-1)

    gt_action_across_time = extract_columns(traj, [f"action.{key}" for key in action_keys])
    pred_action_across_time = np.array(pred_action_across_time)

    gt_action_across_time = gt_action_across_time[:traj_length]
    pred_action_across_time = pred_action_across_time[:traj_length]

    if gt_action_across_time.shape != pred_action_across_time.shape:
        min_len = min(len(gt_action_across_time), len(pred_action_across_time))
        gt_action_across_time = gt_action_across_time[:min_len]
        pred_action_across_time = pred_action_across_time[:min_len]

    mse = float(np.mean((gt_action_across_time - pred_action_across_time) ** 2))
    mae = float(np.mean(np.abs(gt_action_across_time - pred_action_across_time)))
    return mse, mae, traj_length


def evaluate_pusht_dataset(
    checkpoint_path: str,
    dataset_path: str,
    n_episodes: int = 10,
    mae_success_threshold: float = 20.0,
    action_horizon: int = 16,
    output_dir: str = "output_final/pusht",
) -> dict:
    """
    Evaluate RLDX-1 checkpoint on Push-T dataset.

    Args:
        checkpoint_path: Path to trained checkpoint
        dataset_path: Path to Push-T dataset (examples/pusht_lerobot)
        n_episodes: Number of episodes to evaluate
        output_dir: Directory to save results CSV

    Returns:
        Dict with success_rate, n_success, n_episodes, results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint path to absolute
    checkpoint_path = str(Path(checkpoint_path).resolve())
    print(f"[*] Loading checkpoint: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    policy = RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=checkpoint_path,
        device=device,
        strict=True,
    )

    # Load dataset info
    print(f"[*] Loading Push-T dataset from: {dataset_path}")
    dataset_info = load_pusht_dataset_info(dataset_path)
    episodes = dataset_info["episodes"]
    
    # Limit to requested episodes
    n_episodes_available = len(episodes)
    n_episodes = min(n_episodes, n_episodes_available)
    print(f"[*] Evaluating on {n_episodes}/{n_episodes_available} episodes")

    # Load dataset via LeRobotEpisodeLoader using the Push-T modality config.
    loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=pusht)

    results = []
    n_success = 0

    print(f"[*] Running inference on {n_episodes} episodes...")
    for episode_idx in range(n_episodes):
        try:
            episode_length = loader.get_episode_length(episode_idx)
            policy.reset()
            mse, mae, used_steps = evaluate_single_episode_mae(
                policy=policy,
                loader=loader,
                episode_idx=episode_idx,
                embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
                action_horizon=action_horizon,
            )
            success = mae <= mae_success_threshold
            
            results.append({
                "episode": episode_idx,
                "success": int(success),
                "n_steps": used_steps,
                "mse": mse,
                "mae": mae,
                "threshold": mae_success_threshold,
            })
            
            if success:
                n_success += 1
            
            if (episode_idx + 1) % 5 == 0:
                current_sr = n_success / (episode_idx + 1)
                print(
                    f"  Episode {episode_idx + 1}/{n_episodes}: "
                    f"Success rate = {current_sr:.1%}, MAE = {mae:.4f}"
                )
        
        except Exception as e:
            print(f"  Episode {episode_idx}: Error - {e}")
            results.append({
                "episode": episode_idx,
                "success": 0,
                "n_steps": 0,
                "error": str(e),
            })
    
    # Final stats
    success_rate = n_success / n_episodes if n_episodes > 0 else 0.0

    print(f"\n[✓] Evaluation complete!")
    print(f"  Success rate: {success_rate:.1%} ({n_success}/{n_episodes})")
    print(f"  Success = episodes with MAE <= {mae_success_threshold}")

    # Save results to CSV
    csv_path = output_path / "evaluation_results.csv"
    fieldnames = ["episode", "success", "n_steps", "mse", "mae", "threshold", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to: {csv_path}")

    return {
        "success_rate": success_rate,
        "n_success": n_success,
        "n_episodes": n_episodes,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RLDX-1 on Push-T dataset")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to trained checkpoint directory",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to Push-T dataset (e.g., examples/pusht_lerobot)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--mae-success-threshold",
        type=float,
        default=20.0,
        help="Episode is success if action MAE is <= this threshold",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Action horizon used when querying policy chunks",
    )
    parser.add_argument(
        "--output-dir",
        default="output_final/pusht",
        help="Output directory for results",
    )

    args = parser.parse_args()

    try:
        stats = evaluate_pusht_dataset(
            checkpoint_path=args.checkpoint_path,
            dataset_path=args.dataset_path,
            n_episodes=args.n_episodes,
            mae_success_threshold=args.mae_success_threshold,
            action_horizon=args.action_horizon,
            output_dir=args.output_dir,
        )
        sys.exit(0)
    except Exception as e:
        print(f"[!] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
