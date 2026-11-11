#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1.1 on custom dataset"
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --output=slurm_out/%j-rldx1_ft_custom.out
#SBATCH --error=slurm_out/%j-rldx1_ft_custom.err

set -euo pipefail

# Custom-dataset fine-tune template — vanilla video VLA on your LeRobot
# dataset(s). Toggle motion / memory / physics by uncommenting lines in the
# ARGS array below. Edit dataset paths, modality config, and step counts
# in-place. See run_scripts/train/examples/README.md for details.

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1.1-PT}"
CKPT_NAME="rldx1_ft_custom"
RUN_NAME="$CKPT_NAME"

NUM_GPUS=8
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
EMBODIMENT_TAG="GENERAL_EMBODIMENT"  # or OXE_FRACTAL / OXE_BRIDGE_ORIG / ...
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/your_dataset_config.py"  # see rldx/configs/data/ for examples
CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"

ARGS=(
    --n-cog-tokens 64
    --video-length 4

    # Dataset (list several paths to mix them, with optional per-path weights:
    # --data.paths "/path/a" "/path/b" --data.mix-ratios 1.0 0.5)
    data:paths --data.paths "/path/to/your/lerobot_dataset"
    --data.embodiment-tag "$EMBODIMENT_TAG"

    --dataloader-num-workers 8
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --base-model-path "$BASE_MODEL_PATH"
    --output-dir "$CKPT_DIR"
    --num-gpus "$NUM_GPUS"
    --save-total-limit 5
    --save-steps 1000
    --max-steps 60000
    --global-batch-size 256

    # Add-ons (uncomment any to enable)

    # Motion: discrete action-prefix tokens injected mid-backbone.
    # motion:on --motion.insert-layer 9

    # Memory: cross-clip context tokens for long-horizon manipulation.
    # memory:on --memory.length 4 --memory.stride 16 --memory.n-cog-tokens 16 --memory.concat

    # Physics: per-step force/torque head; match keys/dims to your schema.
    # physics:on --physics.keys torque --physics.dims 48 --physics.allow-missing

    # Smoke test only: build the architecture but skip the checkpoint read.
    # Needs a fresh --output-dir (see docs/training.md).
    # --skip-weight-loading

    wandb:on
    --wandb.project "$WANDB_PROJECT"
    --experiment-name "$RUN_NAME"
)

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
# jemalloc keeps DataLoader worker RSS from ratcheting up over long runs.
JEMALLOC_PATH="${JEMALLOC_PATH:-/usr/lib/x86_64-linux-gnu/libjemalloc.so.2}"
if [ -r "$JEMALLOC_PATH" ]; then
    export LD_PRELOAD="$JEMALLOC_PATH${LD_PRELOAD:+:$LD_PRELOAD}"
else
    echo "[!] jemalloc not found: $JEMALLOC_PATH; continuing with glibc malloc (apt install libjemalloc2, or set JEMALLOC_PATH)" >&2
fi
uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py "${ARGS[@]}"
