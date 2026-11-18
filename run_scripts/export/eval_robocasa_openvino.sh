#!/bin/bash
# RoboCasa kitchen evaluation against the OpenVINO policy server (rldx_openvino.py serve).
# Same 24 tasks and protocol as run_scripts/eval/robocasa_kitchen/eval_robocasa.sh, which stays
# upstream: seed 42, 5-lane seed layout, 16 actions per call, 720 steps (1000 for OpenDoubleDoor).
# Exports are batch 1, so each lane runs alone (--n_envs 1 --env_layout 5 --env_idx_offset <lane>),
# the mode upstream's swap chain re-runs with: same 1,200 scenes, and no camera-view swap.
#
# Usage: eval_robocasa_openvino.sh <RUN_NAME> <EXPORT_DIR> [OV_DEVICE] [MAX_PARALLEL]
#   N_EPISODES=50   episodes per task, a multiple of 5 (one fifth per lane). Smaller values run
#                   the first episodes of every lane, so a later larger run resumes and extends them.
#   TASKS=a,b,...   task subset (default: all 24); TASK_LIMIT=n keeps the first n of them.
#   OV_PRECISION    OpenVINO inference precision hint (default: the device's, f16 on GPU).
#
# One-time setup on a fresh machine (from the repo root, branch with run_scripts/export/):
#   1. Model venv:     uv sync --python 3.10
#                      uv pip install --python .venv/bin/python openvino==2026.4.1 nncf==3.4.0
#      Later `uv sync` / `uv run` remove openvino and nncf again; reinstall them after.
#   2. Intel GPU:      compute runtime installed, so the device shows in
#                      .venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"
#   3. Simulator:      INSTALL_FLASH_ATTN=0 bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh
#                      (submodule, robosuite 1.5.1, ~5 GB of kitchen assets)
#   4. Export:         S=run_scripts/export/rldx_openvino.py
#                      .venv/bin/python $S export --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
#                          --hidden-alignment 64 --output exports/rldx11_robocasa_ov_a64
#                      .venv/bin/python $S quantize --export-dir exports/rldx11_robocasa_ov_a64 \
#                          --output exports/rldx11_robocasa_ov_a64_int8asym --mode int8_asym
#      CPU only (~30 GB RAM, ~25 min) and downloads the checkpoint; no recorded observation is needed,
#      the blank one gives the same input layout. Or copy an existing export directory.
# The checks below stop with the step to run if any of these is missing.
set -u

export NO_ALBUMENTATIONS_UPDATE=1

RUN_NAME="${1:?RUN_NAME required}"
EXPORT_DIR="$(realpath "${2:?EXPORT_DIR required}")"
OV_DEVICE="${3:-GPU.0}"
MAX_PARALLEL="${4:-4}"

N_EPISODES="${N_EPISODES:-50}"
TASKS="${TASKS:-}"
TASK_LIMIT="${TASK_LIMIT:-}"
OV_PRECISION="${OV_PRECISION:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-120}"  # seconds between running success-rate lines

# -- Protocol (upstream eval_robocasa.sh) --------------------------------------
SEED=42
LANES=5
N_ACTION_STEPS=16
MAX_EPISODE_STEPS=720
MAX_STEPS_OVERRIDES="OpenDoubleDoor=1000"

if ! [ "$N_EPISODES" -gt 0 ] 2>/dev/null || [ $((N_EPISODES % LANES)) -ne 0 ]; then
  echo "[ERROR] N_EPISODES=$N_EPISODES must be a positive multiple of $LANES" >&2
  exit 1
fi

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
UPSTREAM_EVAL="$BASE_DIR/run_scripts/eval/robocasa_kitchen/eval_robocasa.sh"
SIM_PY="$BASE_DIR/run_scripts/eval/robocasa_kitchen/robocasa_uv/.venv/bin/python"
# Not `uv run`: a project sync would drop openvino/nncf, which are not in the lockfile.
SERVER_PY="${SERVER_PY:-$BASE_DIR/.venv/bin/python}"
OUT_BASE="${MODEL_OUTPUT_DIR:-$BASE_DIR/output_final}"
OUT_ROOT="$OUT_BASE/robocasa/$RUN_NAME"

if [ ! -f "$EXPORT_DIR/rldx_openvino.json" ]; then
  echo "[ERROR] No OpenVINO export at $EXPORT_DIR (missing rldx_openvino.json); see setup step 4" >&2
  exit 1
fi
if [ ! -x "$SIM_PY" ]; then
  echo "[ERROR] RoboCasa venv not found; see setup step 3:" >&2
  echo "        INSTALL_FLASH_ATTN=0 bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh" >&2
  exit 1
fi
if [ ! -x "$SERVER_PY" ]; then
  echo "[ERROR] No model venv at $SERVER_PY; see setup step 1 (uv sync --python 3.10)" >&2
  exit 1
fi
OV_DEVICE="$OV_DEVICE" "$SERVER_PY" - <<'PY' || exit 1
import os, sys
try:
    import openvino as ov
except ImportError:
    sys.exit("[ERROR] openvino is not in the model venv; see setup step 1:\n"
             "        uv pip install --python .venv/bin/python openvino==2026.4.1 nncf==3.4.0")
device = os.environ["OV_DEVICE"]
available = ov.Core().available_devices
if device not in available and device not in {d.split(".")[0] for d in available}:
    sys.exit(f"[ERROR] OpenVINO device {device} not found (have {available}); see setup step 2")
try:
    import rldx.model.export  # noqa: F401
except ImportError:
    sys.exit("[ERROR] rldx.model.export missing: check out the branch with the OpenVINO export patches")
PY

# The task list comes from the upstream script so both evaluations stay on the same tasks.
eval "$(sed -n '/^TASK_NAMES=(/,/^)/p' "$UPSTREAM_EVAL")"
if [ -n "$TASKS" ]; then
  IFS=, read -r -a SELECTED <<< "$TASKS"
  for t in "${SELECTED[@]}"; do
    [[ " ${TASK_NAMES[*]} " == *" $t "* ]] || { echo "[ERROR] unknown task $t" >&2; exit 1; }
  done
else
  SELECTED=("${TASK_NAMES[@]}")
fi
[ -n "$TASK_LIMIT" ] && SELECTED=("${SELECTED[@]:0:$TASK_LIMIT}")
TOTAL=${#SELECTED[@]}

# -- Manifest: refuse to resume a tree made with another export or protocol ------
mkdir -p "$OUT_ROOT"
if ! OUT_ROOT="$OUT_ROOT" EXPORT_DIR="$EXPORT_DIR" OV_DEVICE="$OV_DEVICE" \
     OV_PRECISION="$OV_PRECISION" SEED="$SEED" LANES="$LANES" N_ACTION_STEPS="$N_ACTION_STEPS" \
     MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" MAX_STEPS_OVERRIDES="$MAX_STEPS_OVERRIDES" \
     python3 - <<'PY'
import json, os, sys
from pathlib import Path

env = os.environ
meta = json.loads((Path(env["EXPORT_DIR"]) / "rldx_openvino.json").read_text())
new = {
    "backend": "openvino",
    "export_dir": env["EXPORT_DIR"],
    "model_path": meta["model_path"],
    "weight_compression": (meta.get("weight_compression") or {}).get("mode", "fp16"),
    "ov_device": env["OV_DEVICE"],
    "ov_precision": env["OV_PRECISION"] or "default",
    "seed": int(env["SEED"]),
    "lane_count": int(env["LANES"]),
    "n_action_steps": int(env["N_ACTION_STEPS"]),
    "max_episode_steps": int(env["MAX_EPISODE_STEPS"]),
    "max_steps_overrides": env["MAX_STEPS_OVERRIDES"],
}
path = Path(env["OUT_ROOT"]) / "manifest.json"
if path.exists():
    old = json.loads(path.read_text())
    diff = [(k, old.get(k), v) for k, v in new.items() if old.get(k) != v]
    if diff:
        print(f"[ERROR] {path.parent} holds results from another setting:", file=sys.stderr)
        for k, was, now in diff:
            print(f"    {k}: recorded {was!r} -> requested {now!r}", file=sys.stderr)
        print("    Use another RUN_NAME, or delete the directory.", file=sys.stderr)
        sys.exit(1)
path.write_text(json.dumps(new, indent=2) + "\n")
PY
then
  exit 1
fi

port_is_free() {
  python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(1 if s.connect_ex(('127.0.0.1', $1)) == 0 else 0)
" 2>/dev/null
}
PORT=$((20000 + RANDOM % 40000))
while ! port_is_free "$PORT"; do PORT=$((PORT + 1)); done

echo "[i] RUN_NAME=$RUN_NAME  OUT_ROOT=$OUT_ROOT"
echo "[i] EXPORT_DIR=$EXPORT_DIR  OV_DEVICE=$OV_DEVICE  OV_PRECISION=${OV_PRECISION:-default}"
echo "[i] tasks=$TOTAL  n_episodes=$N_EPISODES/task ($((N_EPISODES / LANES)) per lane)  MAX_PARALLEL=$MAX_PARALLEL  PORT=$PORT"

cd "$BASE_DIR"
SERVE_ARGS=(--backend openvino --export-dir "$EXPORT_DIR" --ov-device "$OV_DEVICE")
[ -n "$OV_PRECISION" ] && SERVE_ARGS+=(--ov-precision "$OV_PRECISION")
"$SERVER_PY" run_scripts/export/rldx_openvino.py serve "${SERVE_ARGS[@]}" \
    --host 127.0.0.1 --port "$PORT" >> "$OUT_ROOT/server.log" 2>&1 &
SERVE_PID=$!
PROGRESS_PID=""
trap 'kill $SERVE_PID $PROGRESS_PID 2>/dev/null' EXIT

echo "[i] Waiting for server (PID=$SERVE_PID, log $OUT_ROOT/server.log)..."
for i in $(seq 1 900); do
  port_is_free "$PORT" || break
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[!] Server died before binding to :$PORT; see $OUT_ROOT/server.log"; exit 1
  fi
  sleep 1
done
port_is_free "$PORT" && { echo "[!] Server never bound to :$PORT within 900s"; exit 1; }
echo "[i] Server listening on :$PORT after ${i}s"

# Episodes are counted from the recorded videos, as upstream does; quarantined ones do not count.
tally() {
  local dir=$1
  local s f
  s=$(find "$dir" -path "*_quarantine*" -prune -o -name "*-success.mp4" -print 2>/dev/null | wc -l)
  f=$(find "$dir" -path "*_quarantine*" -prune -o -name "*-failure.mp4" -print 2>/dev/null | wc -l)
  echo "$s $f"
}
progress() {
  local s=0 n=0 t ts tf
  for t in "${SELECTED[@]}"; do
    [ -d "$OUT_ROOT/$t" ] || continue
    read -r ts tf < <(tally "$OUT_ROOT/$t")
    s=$((s + ts)); n=$((n + ts + tf))
  done
  if [ "$n" -eq 0 ]; then
    echo "[progress] 0/$((TOTAL * N_EPISODES)) episodes"
  else
    awk -v s="$s" -v n="$n" -v total=$((TOTAL * N_EPISODES)) \
      'BEGIN { printf "[progress] %d/%d episodes, success %d (%.1f%%)\n", n, total, s, 100 * s / n }'
  fi
}
(while sleep "$PROGRESS_EVERY"; do progress; done) &
PROGRESS_PID=$!

# Lanes of one task run in turn: they share the task's results CSV.
run_task() {
  local task=$1 out="$OUT_ROOT/$1" max_steps=$MAX_EPISODE_STEPS kv lane
  for kv in $MAX_STEPS_OVERRIDES; do
    [ "${kv%%=*}" = "$task" ] && max_steps="${kv##*=}"
  done
  mkdir -p "$out"
  for lane in $(seq 0 $((LANES - 1))); do
    "$SIM_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --n_episodes $((N_EPISODES / LANES)) \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --max_episode_steps "$max_steps" \
        --env_name "robocasa_panda_omron/${task}_PandaOmron_Env" \
        --n_action_steps "$N_ACTION_STEPS" \
        --n_envs 1 \
        --env_layout "$LANES" \
        --env_idx_offset "$lane" \
        --seed "$SEED" \
        --video_dir "$out" \
        >> "$out/eval.log" 2>&1 || { echo "[!] $task lane $lane exited with $?" >&2; return 1; }
  done
}

RUN_PIDS=()
for idx in "${!SELECTED[@]}"; do
  echo "[i] [$((idx + 1))/$TOTAL] ${SELECTED[$idx]}"
  run_task "${SELECTED[$idx]}" &
  RUN_PIDS+=($!)
  while :; do
    running=0
    for pid in "${RUN_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null && running=$((running + 1))
    done
    [ "$running" -lt "$MAX_PARALLEL" ] && break
    sleep 5
  done
done

echo "[i] All tasks launched, waiting for completion..."
FAILED=()
for idx in "${!RUN_PIDS[@]}"; do
  wait "${RUN_PIDS[$idx]}" || FAILED+=("${SELECTED[$idx]}")
done

echo "[i] ===== Summary ($RUN_NAME) ====="
SUMMARY="$OUT_ROOT/summary.csv"
echo "task,success,episodes,success_rate" > "$SUMMARY"
for t in "${SELECTED[@]}"; do
  read -r ts tf < <(tally "$OUT_ROOT/$t")
  n=$((ts + tf))
  rate=$(awk -v s="$ts" -v n="$n" 'BEGIN { printf "%.3f", n ? s / n : 0 }')
  echo "$t,$ts,$n,$rate" >> "$SUMMARY"
  printf "[i] %-24s %3d/%-3d %s\n" "$t" "$ts" "$n" "$rate"
done
progress
echo "[i] Per-task table: $SUMMARY"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[!] ${#FAILED[@]} task(s) failed: ${FAILED[*]} (see <task>/eval.log, server.log)"
  exit 1
fi
