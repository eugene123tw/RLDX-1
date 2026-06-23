#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on Push-T"
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --output=slurm_out/%j-rldx1_ft_pusht.out
#SBATCH --error=slurm_out/%j-rldx1_ft_pusht.err

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_pusht_validation"
RUN_NAME="$CKPT_NAME"

NUM_GPUS=8
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# Local pusht dataset (downloaded via download_pusht.py)
DATA_DIR="${DATA_DIR:-$BASE_DIR/examples/pusht_lerobot}"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/pusht/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/pusht_config.py"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        --video-length 4 \
        --dataset-path "$DATA_DIR" \
        --dataloader-num-workers 8 \
        --embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH" \
        --color-jitter-params $COLOR_JITTER_PARAMS \
        --base-model-path "$BASE_MODEL_PATH" \
        --output-dir "$CKPT_DIR" \
        --num-gpus $NUM_GPUS \
        --save-total-limit 5 \
        --save-steps 500 \
        --max-steps 5000 \
        --global-batch-size 256 \
        --use-wandb \
        --wandb-project "$WANDB_PROJECT" \
        --experiment-name "$RUN_NAME"
