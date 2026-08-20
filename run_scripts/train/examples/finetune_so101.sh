#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on SO-101 pick-and-place-potato"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=slurm_out/%j-rldx1_ft_so101.out
#SBATCH --error=slurm_out/%j-rldx1_ft_so101.err

set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_so101_pick_and_place_potato"
RUN_NAME="$CKPT_NAME"

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# Local dataset produced by run_scripts/data/convert_lerobot_v3_to_v2.py
DATA_DIR="${DATA_DIR:-$BASE_DIR/examples/so101_pick_and_place_potato}"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/so101/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/so101_config.py"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

detect_visible_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a _cuda_devs <<< "$CUDA_VISIBLE_DEVICES"
        echo "${#_cuda_devs[@]}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
        return
    fi

    echo 1
}

DETECTED_GPUS="$(detect_visible_gpus)"
NUM_GPUS="${NUM_GPUS:-$DETECTED_GPUS}"
TARGET_PER_DEVICE_BATCH="${TARGET_PER_DEVICE_BATCH:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * TARGET_PER_DEVICE_BATCH))}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
# 50 episodes is a small dataset; a few thousand steps is enough to see loss converge.
MAX_STEPS="${MAX_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
OPTIM="${OPTIM:-adafactor}"
TUNE_TOP_LLM_LAYERS="${TUNE_TOP_LLM_LAYERS:-0}"
ACTION_MODEL_USE_LORA="${ACTION_MODEL_USE_LORA:-1}"
ACTION_MODEL_LORA_RANK="${ACTION_MODEL_LORA_RANK:-8}"
ACTION_MODEL_LORA_ALPHA="${ACTION_MODEL_LORA_ALPHA:-16}"
ACTION_MODEL_LORA_DROPOUT="${ACTION_MODEL_LORA_DROPOUT:-0.0}"

if (( NUM_GPUS < 1 )); then
    echo "[warn] Detected NUM_GPUS=$NUM_GPUS. Falling back to 1."
    NUM_GPUS=1
fi

if (( GLOBAL_BATCH_SIZE < NUM_GPUS )); then
    echo "[warn] GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is too small for NUM_GPUS=$NUM_GPUS. Raising to $NUM_GPUS."
    GLOBAL_BATCH_SIZE=$NUM_GPUS
fi

if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    adjusted="$(( (GLOBAL_BATCH_SIZE / NUM_GPUS) * NUM_GPUS ))"
    if (( adjusted < NUM_GPUS )); then
        adjusted=$NUM_GPUS
    fi
    echo "[warn] GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS. Adjusting $GLOBAL_BATCH_SIZE -> $adjusted."
    GLOBAL_BATCH_SIZE="$adjusted"
fi

PER_DEVICE_BATCH="$((GLOBAL_BATCH_SIZE / NUM_GPUS))"
EFFECTIVE_BATCH_SIZE="$((GLOBAL_BATCH_SIZE * GRAD_ACC_STEPS))"
echo "[info] num_gpus=$NUM_GPUS per_device_batch=$PER_DEVICE_BATCH global_batch_size=$GLOBAL_BATCH_SIZE grad_acc_steps=$GRAD_ACC_STEPS effective_batch_size=$EFFECTIVE_BATCH_SIZE optim=$OPTIM tune_top_llm_layers=$TUNE_TOP_LLM_LAYERS action_model_use_lora=$ACTION_MODEL_USE_LORA"

if (( NUM_GPUS == 1 && PER_DEVICE_BATCH > 4 )); then
    echo "[warn] per_device_batch=$PER_DEVICE_BATCH may be too large for 1x A100-40GB; consider TARGET_PER_DEVICE_BATCH=2 or enable 8-bit optimizer."
fi

EXTRA_ARGS=(
    --optim "$OPTIM"
    --tune-top-llm-layers "$TUNE_TOP_LLM_LAYERS"
)

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    EXTRA_ARGS+=(--gradient-checkpointing)
else
    EXTRA_ARGS+=(--no-gradient-checkpointing)
fi

if [[ "$ACTION_MODEL_USE_LORA" == "1" ]]; then
    EXTRA_ARGS+=(
        --action-model-use-lora
        --action-model-lora-rank "$ACTION_MODEL_LORA_RANK"
        --action-model-lora-alpha "$ACTION_MODEL_LORA_ALPHA"
        --action-model-lora-dropout "$ACTION_MODEL_LORA_DROPOUT"
    )
fi

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        --video-length 4 \
        --dataset-path "$DATA_DIR" \
        --dataloader-num-workers $DATALOADER_NUM_WORKERS \
        --embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH" \
        --color-jitter-params $COLOR_JITTER_PARAMS \
        --base-model-path "$BASE_MODEL_PATH" \
        --output-dir "$CKPT_DIR" \
        --num-gpus $NUM_GPUS \
        --save-total-limit 5 \
        --save-steps $SAVE_STEPS \
        --max-steps $MAX_STEPS \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --gradient-accumulation-steps $GRAD_ACC_STEPS \
        "${EXTRA_ARGS[@]}" \
        --use-wandb \
        --wandb-project "$WANDB_PROJECT" \
        --experiment-name "$RUN_NAME"
