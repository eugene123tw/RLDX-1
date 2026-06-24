#!/usr/bin/env python3
"""
Evaluate an RLDX-1 checkpoint on Push-T using lerobot's gym environment.

Usage:
  uv run python eval_pusht_checkpoint.py \
      --checkpoint-path ckpt/rldx1/finetuned/pusht/rldx1_ft_pusht_validation/checkpoint-500 \
      --n-episodes 50

Outputs:
  - Push-T success rate (%age) and raw counts
  - Per-episode metrics (success, episode length)
  - Summary CSV at output_final/pusht/evaluation_results.csv
"""

import argparse
import csv
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy


def evaluate_pusht_checkpoint(
    checkpoint_path: str,
    n_episodes: int = 50,
    max_episode_steps: int = 200,
    output_dir: str = "output_final/pusht",
) -> dict:
    """
    Evaluate RLDX-1 checkpoint on Push-T.

    Args:
        checkpoint_path: Path to trained checkpoint (contains policy weights + processor)
        n_episodes: Number of evaluation episodes
        max_episode_steps: Max steps per episode (Push-T default is ~161)
        output_dir: Directory to save results CSV

    Returns:
        Dict with keys: success_rate, n_success, n_episodes, results (list of dicts)
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

    print(f"[*] Setting up Push-T environment...")
    env = gym.make("lerobot/PushT-v0")

    results = []
    n_success = 0

    print(f"[*] Running {n_episodes} evaluation episodes...")
    for episode_idx in range(n_episodes):
        obs, info = env.reset()
        policy.reset()  # Reset policy state/memory for new episode
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done and episode_length < max_episode_steps:
            # Forward pass through policy
            # get_action expects observation dict and returns (action_dict, info_dict)
            action_dict, policy_info = policy.get_action(obs)
            
            # Extract the action vector from the dict
            # Push-T action is under "agent_pos" key based on modality config
            if isinstance(action_dict, dict):
                # The action should be in the format {"agent_pos": [dx, dy], ...}
                action = action_dict.get("agent_pos", None)
                if action is None:
                    # Fallback: try to find the first action key
                    for k, v in action_dict.items():
                        if isinstance(v, (np.ndarray, list)):
                            action = v
                            break
            else:
                action = action_dict

            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        # Push-T success is typically indicated by info["success"] or reward > threshold
        success = info.get("success", False) or episode_reward > 0.5

        results.append({
            "episode": episode_idx,
            "success": int(success),
            "episode_length": episode_length,
            "reward": episode_reward,
        })

        if success:
            n_success += 1

        if (episode_idx + 1) % 10 == 0:
            current_sr = n_success / (episode_idx + 1)
            print(f"  Episode {episode_idx + 1}/{n_episodes}: Success rate = {current_sr:.1%}")

    env.close()

    # Final stats
    success_rate = n_success / n_episodes if n_episodes > 0 else 0.0

    print(f"\n[✓] Evaluation complete!")
    print(f"  Success rate: {success_rate:.1%} ({n_success}/{n_episodes})")

    # Save results to CSV
    csv_path = output_path / "evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "success", "episode_length", "reward"])
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
    parser = argparse.ArgumentParser(description="Evaluate RLDX-1 on Push-T")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to trained checkpoint directory (e.g., ckpt/rldx1/finetuned/pusht/...)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--output-dir",
        default="output_final/pusht",
        help="Output directory for results",
    )

    args = parser.parse_args()

    try:
        stats = evaluate_pusht_checkpoint(
            checkpoint_path=args.checkpoint_path,
            n_episodes=args.n_episodes,
            max_episode_steps=args.max_episode_steps,
            output_dir=args.output_dir,
        )
        sys.exit(0)
    except Exception as e:
        print(f"[!] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
