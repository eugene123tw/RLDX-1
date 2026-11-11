#!/bin/bash
# Multi-node fine-tune on RoboCasa365 (16 GPU, effective bsz=1024, 60k steps).
# Reproduces the RLDX-1.1-FT-RC365 release recipe (VARIANT=img: RLDX-1.1-FT-RC365-IMG).
#
# Required environment variables:
#   DATA_ROOT      Path to robocasa365/v1.0/pretrain (lerobot dataset root)
#   NODE_RANK      Per-node rank (0 = master, 1+ = worker) for the rendezvous
#
# Optional:
#   VARIANT          video (default) or img
#   BASE_MODEL_PATH  HF repo or local checkpoint dir (default: RLWRLD/RLDX-1.1-PT,
#                    or RLWRLD/RLDX-1.1-PT-IMG with VARIANT=img)
#   NUM_GPUS         GPUs per node (default 8)
#   NUM_NODES        Total nodes (default 2)
#   MASTER_PORT      Rendezvous port (default 29500)
#   RLDX_LOG_DIR     Shared filesystem path used to publish the master IP
#                    (defaults to "$BASE_DIR/.rdzv")
#   WANDB_PROJECT    Defaults to "rldx-finetune"

set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

# NCCL / IB defaults — override per-cluster as needed
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-eth0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

# ── Recipe ─────────────────────────────────────────────
# video reproduces RLDX-1.1-FT-*, img reproduces RLDX-1.1-FT-*-IMG (single frame, no motion).
VARIANT="${VARIANT:-video}"
case "$VARIANT" in
    video) DEFAULT_BASE_MODEL_PATH="RLWRLD/RLDX-1.1-PT"; VIDEO_LENGTH=4; MOTION_ARGS=(motion:on); VARIANT_SUFFIX="" ;;
    img) DEFAULT_BASE_MODEL_PATH="RLWRLD/RLDX-1.1-PT-IMG"; VIDEO_LENGTH=1; MOTION_ARGS=(); VARIANT_SUFFIX="_img" ;;
    *) echo "[!] VARIANT must be video or img, got: $VARIANT" >&2; exit 1 ;;
esac
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$DEFAULT_BASE_MODEL_PATH}"
CKPT_NAME="rldx1_ft_robocasa365_human300_bsz1024_60k_16gpu$VARIANT_SUFFIX"
RUN_NAME="$CKPT_NAME"
# ───────────────────────────────────────────────────────

NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-2}"
TOTAL_GPUS=$(( NUM_GPUS * NUM_NODES ))
NODE_RANK="${NODE_RANK:?Set NODE_RANK to the rank of this node (0 for master, 1+ for workers)}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to robocasa365/v1.0/pretrain}"
MASTER_PORT="${MASTER_PORT:-29500}"

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RLDX_LOG_DIR="${RLDX_LOG_DIR:-$BASE_DIR/.rdzv}"
mkdir -p "$RLDX_LOG_DIR"
MASTER_ADDR_FILE="$RLDX_LOG_DIR/master_addr_${SLURM_JOB_ID:-$$}.txt"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/robocasa365_config.py"

# Discover all lerobot datasets under DATA_ROOT (exclude mimicgen-synth)
mapfile -t DATASETS < <(python3 -c "
from pathlib import Path
root = Path('$DATA_ROOT')
for p in sorted(root.glob('**/lerobot/meta/info.json')):
    if '/mg/' in str(p):
        continue
    print(str(p.parent.parent))
")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo '[x] No lerobot datasets found under DATA_ROOT' >&2
  exit 1
fi
MIX_RATIOS=()
for _ in "${DATASETS[@]}"; do MIX_RATIOS+=(1); done

# Multi-node rendezvous via shared filesystem
if [ "$NODE_RANK" = "0" ]; then
  MY_IP=$(ip -4 addr show "$NCCL_SOCKET_IFNAME" | grep -oP '(?<=inet )\d+(\.\d+){3}')
  echo "$MY_IP" > "$MASTER_ADDR_FILE"
  echo "[i] Master IP: $MY_IP -> $MASTER_ADDR_FILE"
fi
for i in $(seq 1 200); do
  [ -f "$MASTER_ADDR_FILE" ] && break
  sleep 5
done
[ -f "$MASTER_ADDR_FILE" ] || { echo "[x] Master IP file missing after timeout"; exit 1; }
export MASTER_ADDR=$(cat "$MASTER_ADDR_FILE")
export MASTER_PORT
echo "[i] NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

cd "$BASE_DIR"
# jemalloc keeps DataLoader worker RSS from ratcheting up over long runs.
JEMALLOC_PATH="${JEMALLOC_PATH:-/usr/lib/x86_64-linux-gnu/libjemalloc.so.2}"
if [ -r "$JEMALLOC_PATH" ]; then
    export LD_PRELOAD="$JEMALLOC_PATH${LD_PRELOAD:+:$LD_PRELOAD}"
else
    echo "[!] jemalloc not found: $JEMALLOC_PATH; continuing with glibc malloc (apt install libjemalloc2, or set JEMALLOC_PATH)" >&2
fi
uv run torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes="$NUM_NODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        --video-length "$VIDEO_LENGTH" \
        data:paths --data.paths "${DATASETS[@]}" \
        --data.mix-ratios "${MIX_RATIOS[@]}" \
        --dataloader-num-workers 20 \
        --data.embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH" \
        --state-dropout-prob 0.0 \
        --base-model-path "$BASE_MODEL_PATH" \
        --output-dir "$CKPT_DIR" \
        --num-gpus "$TOTAL_GPUS" \
        --save-total-limit 10 \
        --save-steps 1000 \
        --max-steps 60000 \
        --global-batch-size 256 \
        --gradient-accumulation-steps 4 \
        "${MOTION_ARGS[@]}" \
        wandb:on \
        --wandb.project "${WANDB_PROJECT:-rldx-finetune}" \
        --experiment-name "$RUN_NAME"
