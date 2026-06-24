#!/bin/bash
# Evaluate trained RLDX-1 checkpoint on Push-T dataset
#
# Usage:
#   bash run_scripts/eval/eval_pusht.sh [checkpoint_path] [n_episodes]
#
# Defaults:
#   checkpoint_path = ckpt/rldx1/finetuned/pusht/rldx1_ft_pusht_validation/rldx1_ft_pusht_validation/checkpoint-500
#   n_episodes = 10
#   dataset_path = examples/pusht_lerobot
#
# Example:
#   bash run_scripts/eval/eval_pusht.sh checkpoint-500 50
#   bash run_scripts/eval/eval_pusht.sh /path/to/checkpoint 20

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

# Defaults
CHECKPOINT_PATH="${1:-ckpt/rldx1/finetuned/pusht/rldx1_ft_pusht_validation/rldx1_ft_pusht_validation/checkpoint-500}"
N_EPISODES="${2:-10}"
DATASET_PATH="${DATASET_PATH:-examples/pusht_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-output_final/pusht}"

# If checkpoint path is short (e.g., "checkpoint-500"), assume it's under the default training path
if [[ ! "$CHECKPOINT_PATH" = /* ]] && [[ "$CHECKPOINT_PATH" != *"/"* ]]; then
    CHECKPOINT_PATH="ckpt/rldx1/finetuned/pusht/rldx1_ft_pusht_validation/rldx1_ft_pusht_validation/$CHECKPOINT_PATH"
fi

echo "[*] Evaluating Push-T checkpoint"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Episodes: $N_EPISODES"
echo "  Output dir: $OUTPUT_DIR"

uv run python eval_pusht_dataset.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --dataset-path "$DATASET_PATH" \
    --n-episodes "$N_EPISODES" \
    --output-dir "$OUTPUT_DIR"

echo "[✓] Push-T evaluation complete!"
echo "    Results: $OUTPUT_DIR/evaluation_results.csv"
