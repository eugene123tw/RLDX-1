#!/bin/bash
#SBATCH --job-name="convert_to_lerobot"
#SBATCH --partition=${SBATCH_PARTITION:-cpu}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_out/%j-convert_teleop_to_lerobot.out
#SBATCH --error=slurm_out/%j-convert_teleop_to_lerobot.err

set -euo pipefail
shopt -s nullglob

USAGE="usage: TASK_MAP=<task_map.json> $0 <raw_root> <output_root>  (one sub-directory per dataset)"
SRC_ROOT="${1:?$USAGE}"
DST_ROOT="${2:?$USAGE}"
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TASK_MAP="${TASK_MAP:?Set TASK_MAP to a JSON mapping each dataset (sub-directory) name to its instruction; see run_scripts/data/task_map.json}"
PHYSICS_MAP="${PHYSICS_MAP:-$BASE_DIR/run_scripts/data/physics_map.json}"

for SRC_DIR in "$SRC_ROOT"/*/; do
  TASK_ID="$(basename "$SRC_DIR")"
  echo "=== Converting $TASK_ID ==="
  uv run --project "$BASE_DIR" python "$BASE_DIR/run_scripts/data/convert_h5_to_v2.py" \
    --raw-dir "$SRC_DIR" \
    --output-path "$DST_ROOT/$TASK_ID" \
    --repo-id "$TASK_ID" \
    --task-map "$TASK_MAP" \
    --physics-map "$PHYSICS_MAP" \
    --robot-type franka_panda \
    --mode video \
    --action-type eef \
    --fps 10
done
