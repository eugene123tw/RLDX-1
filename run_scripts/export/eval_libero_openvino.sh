#!/bin/bash
# LIBERO evaluation against the OpenVINO policy server (rldx_openvino.py serve).
# Same tasks and rollout settings as run_scripts/eval/libero/eval_libero.sh, which stays upstream.
# Usage: eval_libero_openvino.sh <CKPT_NAME> <EXPORT_DIR> [OV_DEVICE] [MAX_PARALLEL]
set -u

export NO_ALBUMENTATIONS_UPDATE=1

CKPT_NAME="${1:?CKPT_NAME required}"
EXPORT_DIR="${2:?EXPORT_DIR required}"
OV_DEVICE="${3:-GPU.0}"
MAX_PARALLEL="${4:-4}"

N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
# Exports have a fixed batch size (1 by default), so each rollout must send that many observations.
N_ENVS="${N_ENVS:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-60}"  # seconds between running success-rate lines

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
UPSTREAM_EVAL="$BASE_DIR/run_scripts/eval/libero/eval_libero.sh"
LIBERO_PY="$BASE_DIR/run_scripts/eval/libero/libero_uv/.venv/bin/python"
# Not `uv run`: a project sync would drop openvino/nncf, which are not in the lockfile.
SERVER_PY="${SERVER_PY:-$BASE_DIR/.venv/bin/python}"
OUT_BASE="${MODEL_OUTPUT_DIR:-$BASE_DIR/output_final}"

if [ ! -f "$EXPORT_DIR/rldx_openvino.json" ]; then
  echo "[ERROR] No OpenVINO export at $EXPORT_DIR (missing rldx_openvino.json)" >&2
  exit 1
fi
if [ ! -f "$LIBERO_PY" ]; then
  echo "[ERROR] LIBERO venv not found. Run: bash run_scripts/eval/libero/setup_libero.sh" >&2
  exit 1
fi
export LIBERO_CONFIG_PATH="$BASE_DIR/run_scripts/eval/libero/libero_uv/.venv/libero_config"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  echo "[ERROR] No LIBERO config at $LIBERO_CONFIG_PATH. Re-run: bash run_scripts/eval/libero/setup_libero.sh" >&2
  exit 1
fi

# Task lists come from the upstream script so both evaluations stay on the same tasks.
eval "$(sed -n '/^LIBERO_[A-Z0-9_]*_TASKS=(/,/^)/p' "$UPSTREAM_EVAL")"

port_is_free() {
  "$LIBERO_PY" -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(1 if s.connect_ex(('127.0.0.1', $1)) == 0 else 0)
" 2>/dev/null
}

find_free_port() {
  local port=$1
  while ! port_is_free "$port"; do
    port=$((port + 1))
    [ "$port" -gt 65000 ] && port=20000
  done
  echo "$port"
}
PORT=$(find_free_port $((20000 + RANDOM % 40000)))

echo "[i] CKPT_NAME=$CKPT_NAME"
echo "[i] EXPORT_DIR=$EXPORT_DIR  OV_DEVICE=$OV_DEVICE"
echo "[i] MAX_PARALLEL=$MAX_PARALLEL  PORT=$PORT"
echo "[i] n_episodes=$N_EPISODES  n_action_steps=$N_ACTION_STEPS  n_envs=$N_ENVS"

cd "$BASE_DIR"
OUT_ROOT="$OUT_BASE/libero/$CKPT_NAME"
mkdir -p "$OUT_ROOT"
"$SERVER_PY" run_scripts/export/rldx_openvino.py serve \
    --backend openvino \
    --export-dir "$EXPORT_DIR" \
    --ov-device "$OV_DEVICE" \
    --host 127.0.0.1 \
    --port "$PORT" \
    >& "$OUT_ROOT/server.log" &
SERVE_PID=$!
trap 'echo "[i] killing server PID=$SERVE_PID"; kill $SERVE_PID 2>/dev/null' EXIT

echo "[i] Waiting for server (PID=$SERVE_PID, log $OUT_ROOT/server.log) readiness..."
READY=0
for i in $(seq 1 600); do
  if ! port_is_free "$PORT"; then
    echo "[i] Server listening on :$PORT after ${i}s"
    READY=1
    break
  fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "[!] Server died before binding to :$PORT; see $OUT_ROOT/server.log"; exit 1
  fi
  sleep 1
done
if [ "$READY" != 1 ]; then
  echo "[!] Server never bound to :$PORT within 600s"; exit 1
fi
sleep 5  # give server a moment past port-open

ALL_TASKS=() ALL_SUITES=() ALL_TASK_INDICES=()
for i in "${!LIBERO_10_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_10_TASKS[$i]}"); ALL_SUITES+=("libero_10"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_GOAL_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_GOAL_TASKS[$i]}"); ALL_SUITES+=("libero_goal"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_OBJECT_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_OBJECT_TASKS[$i]}"); ALL_SUITES+=("libero_object"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_SPATIAL_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_SPATIAL_TASKS[$i]}"); ALL_SUITES+=("libero_spatial"); ALL_TASK_INDICES+=($i)
done

TOTAL=${#ALL_TASKS[@]}
echo "[i] Total tasks: $TOTAL, batching $MAX_PARALLEL at a time"

# rollout_policy.py appends one row per finished episode; column 3 is success (0/1).
progress() {
    # /dev/null keeps awk from reading stdin before any CSV exists.
    # shellcheck disable=SC2046
    awk -F, -v total=$((TOTAL * N_EPISODES)) '
        FNR > 1 { n++; s += $3; seen[FILENAME] = 1 }
        END {
            if (n == 0) { printf "[progress] 0/%d episodes\n", total; exit }
            printf "[progress] %d/%d episodes, success %d (%.1f%%) over %d tasks\n", n, total, s, 100 * s / n, length(seen)
        }' /dev/null $(find "$OUT_ROOT" -name simulation_results.csv)
}
(while sleep "$PROGRESS_EVERY"; do progress; done) &
PROGRESS_PID=$!
trap 'echo "[i] killing server PID=$SERVE_PID"; kill $SERVE_PID $PROGRESS_PID 2>/dev/null' EXIT

RUN_PIDS=()
for idx in "${!ALL_TASKS[@]}"; do
    TASK="${ALL_TASKS[$idx]}"
    SUITE="${ALL_SUITES[$idx]}"
    TIDX="${ALL_TASK_INDICES[$idx]}"
    CLEAN="${TASK#libero_sim/}"
    OUT="$OUT_ROOT/$SUITE/$CLEAN"
    mkdir -p "$OUT"
    echo "[i] [$((idx+1))/$TOTAL] $SUITE :: $CLEAN (n_ep=$N_EPISODES)"
    "$LIBERO_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --n_episodes "$N_EPISODES" \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --max_episode_steps 720 \
        --env_name "$TASK" \
        --n_action_steps "$N_ACTION_STEPS" \
        --n_envs "$N_ENVS" \
        --video_dir "$OUT" \
        >& "$OUT/eval-local-$TIDX.log" &
    RUN_PIDS+=($!)
    # throttle to MAX_PARALLEL rollouts in flight
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
    if ! wait "${RUN_PIDS[$idx]}"; then
        FAILED+=("${ALL_SUITES[$idx]} :: ${ALL_TASKS[$idx]#libero_sim/}")
    fi
done

echo "[i] ===== Summary ====="
progress
for idx in "${!ALL_TASKS[@]}"; do
    SUITE="${ALL_SUITES[$idx]}"
    CLEAN="${ALL_TASKS[$idx]#libero_sim/}"
    TIDX="${ALL_TASK_INDICES[$idx]}"
    LOG="$OUT_ROOT/$SUITE/$CLEAN/eval-local-$TIDX.log"
    SR=$(grep -oE "success rate[^0-9]*[0-9.]+" "$LOG" 2>/dev/null | tail -1)
    echo "[i] $SUITE :: $CLEAN -> ${SR:-N/A}"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[!] ${#FAILED[@]} task(s) failed:"
    printf '[!]   %s\n' "${FAILED[@]}"
    exit 1
fi
