#!/bin/bash
#SBATCH --job-name="Convert raw dataset to LeRobot format"
#SBATCH --partition=${SBATCH_PARTITION:-cpu}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_out/%j-convert_teleop_to_lerobot.out
#SBATCH --error=slurm_out/%j-convert_teleop_to_lerobot.err

set -euo pipefail

USAGE="usage: TASK_INSTRUCTION=\"...\" $0 <raw_dir> <output_dir>"
SRC_DIR="${1:?$USAGE}"
DST_DIR="${2:?$USAGE}"
TASK_INSTRUCTION="${TASK_INSTRUCTION:?Set TASK_INSTRUCTION to the language instruction shared by every episode}"
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

uv run --project "$BASE_DIR" python "$BASE_DIR/run_scripts/data/convert_h5_to_v2.py" \
  --raw-dir "$SRC_DIR" \
  --output-path "$DST_DIR" \
  --repo-id "$(basename "$DST_DIR")" \
  --task "$TASK_INSTRUCTION" \
  --robot-type franka_panda \
  --mode video \
  --action-type eef \
  --fps 10
