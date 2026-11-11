#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1.1 on GR-1 Tabletop"
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --output=slurm_out/%j-rldx1_ft_gr1_tabletop_1000demo_bsz1024_60k_sd05.out
#SBATCH --error=slurm_out/%j-rldx1_ft_gr1_tabletop_1000demo_bsz1024_60k_sd05.err

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

# ── Recipe ─────────────────────────────────────────────
# video reproduces RLDX-1.1-FT-*, img reproduces RLDX-1.1-FT-*-IMG (single frame, no motion).
VARIANT="${VARIANT:-video}"
case "$VARIANT" in
    video) DEFAULT_BASE_MODEL_PATH="RLWRLD/RLDX-1.1-PT"; VIDEO_LENGTH=4; MOTION_ARGS=(motion:on); VARIANT_SUFFIX="" ;;
    img) DEFAULT_BASE_MODEL_PATH="RLWRLD/RLDX-1.1-PT-IMG"; VIDEO_LENGTH=1; MOTION_ARGS=(); VARIANT_SUFFIX="_img" ;;
    *) echo "[!] VARIANT must be video or img, got: $VARIANT" >&2; exit 1 ;;
esac
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$DEFAULT_BASE_MODEL_PATH}"
CKPT_NAME="rldx1_ft_gr1_tabletop_1000demo_bsz1024_60k_sd05$VARIANT_SUFFIX"
RUN_NAME="$CKPT_NAME"
# ───────────────────────────────────────────────────────

NUM_GPUS=8
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the GR00T-X-Embodiment-Sim dataset root}"
DATA_MIX="gr1_tabletop_1000demo"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/gr1_tabletop/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/gr1_h40_config.py"

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
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        --video-length "$VIDEO_LENGTH" \
        --action-horizon 40 \
        data:mix --data.root "$DATA_ROOT" \
        --data.name "$DATA_MIX" \
        --dataloader-num-workers 8 \
        --modality-config-path "$MODALITY_CONFIG_PATH" \
        --state-dropout-prob 0.5 \
        --base-model-path "$BASE_MODEL_PATH" \
        --output-dir "$CKPT_DIR" \
        --num-gpus $NUM_GPUS \
        --save-total-limit 10 \
        --save-steps 1000 \
        --max-steps 60000 \
        --global-batch-size 256 \
        --gradient-accumulation-steps 4 \
        "${MOTION_ARGS[@]}" \
        wandb:on \
        --wandb.project "$WANDB_PROJECT" \
        --experiment-name "$RUN_NAME"
