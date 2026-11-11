#!/bin/bash
# RoboCasa kitchen eval -- 24 tasks, 50 episodes each, split across 8 shards.
# Usage: eval_robocasa.sh <CKPT_NAME>
#
# Runs every shard on this machine, N_GPUS at a time (default: every visible
# card), then the swap chain, which reports the score. SHARD_ID=<k> runs shard k alone, for
# a scheduler that gives each shard a card of its own.
set -u
export NO_ALBUMENTATIONS_UPDATE=1

CKPT_NAME=${1:?"Usage: $0 <CKPT_NAME>"}
# A scheduler may run a copy of this script, so RLDX_ROOT can name the clone.
BASE_DIR="${RLDX_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
[ -d "$BASE_DIR/rldx/eval" ] || {
  echo "[!] BASE_DIR is not an RLDX clone: $BASE_DIR" >&2
  echo "    Pass RLDX_ROOT=<clone path>." >&2
  exit 2
}

# A relative checkpoint path resolves from the clone root; elsewhere it is read as a hub id.
cd "$BASE_DIR"
MODEL_PATH="$CKPT_NAME"

# The server's settings print would otherwise sit in a block buffer for the whole run.
export PYTHONUNBUFFERED=1

# -- EVALUATION PROTOCOL -----------------------------------------------------
# The simulator's default split B and fixed layout/style pairs.
# N_EPISODES and TASK_LIMIT exist for smoke runs; the rest is the protocol.
N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS=16
MAX_EPISODE_STEPS=720
MAX_STEPS_OVERRIDES=OpenDoubleDoor=1000
# Task subset for smokes: at most this many of each shard's tasks. Empty means all of them.
TASK_LIMIT="${TASK_LIMIT:-}"

# Episode seeds are (SEED + lane) * 100000 + episode_id, so the seed and the lane
# count (5) select the 1200 scenes. Concurrent rendering across lanes is what
# causes the view swap the swap chain repairs.
SEED=42
N_ENVS="${N_ENVS:-5}"
# Set only by the swap chain: it re-runs at N_ENVS=1 while ENV_LAYOUT keeps the
# lane seeds of the 5-lane layout.
ENV_LAYOUT="${ENV_LAYOUT:-0}"
ENV_IDX_OFFSET="${ENV_IDX_OFFSET:-0}"
OUT_BASE="${MODEL_OUTPUT_DIR:-$BASE_DIR/output_final}"
OUT_ROOT="$OUT_BASE/robocasa/$CKPT_NAME"
LOG_ROOT="$OUT_ROOT/_launcher_logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "[i] Output   : $OUT_ROOT"
echo "[i] Envs     : n_envs=$N_ENVS  env_layout=$ENV_LAYOUT  env_idx_offset=$ENV_IDX_OFFSET  seed=$SEED"

TASK_NAMES=(
  "TurnSinkSpout"  "TurnOnStove"  "TurnOnSinkFaucet"  "TurnOnMicrowave"
  "TurnOffStove"  "TurnOffSinkFaucet"
  "TurnOffMicrowave"  "PnPStoveToCounter"  "PnPSinkToCounter"  "PnPMicrowaveToCounter"
  "PnPCounterToStove"  "PnPCounterToSink"
  "PnPCounterToMicrowave"  "PnPCounterToCab"  "PnPCabToCounter"  "OpenSingleDoor"
  "OpenDrawer"  "OpenDoubleDoor"
  "CoffeeSetupMug"  "CoffeeServeMug"  "CoffeePressButton"  "CloseSingleDoor"
  "CloseDrawer"  "CloseDoubleDoor"
)
N_TASKS=${#TASK_NAMES[@]}
# 8 shards of 3 tasks, each with its own server; N_GPUS only sets how many run at once.
N_SHARDS=8

# -- PROTOCOL MANIFEST -------------------------------------------------------
# rollout_policy resumes whatever episodes a task directory holds, so a tree
# recorded under other values is refused. n_envs is not compared (the swap chain
# re-runs at 1 into a 5-lane tree); lane_count is.
MANIFEST="$OUT_ROOT/manifest.json"
GIT_COMMIT=$(git -C "$BASE_DIR" rev-parse --verify -q HEAD 2>/dev/null \
             || cat "$BASE_DIR/SOURCE_COMMIT" 2>/dev/null \
             || echo unknown)
# `|| true`: head -c1 closes the pipe early, and the SIGPIPE must not end the launcher.
GIT_DIRTY=$(git -C "$BASE_DIR" status --porcelain 2>/dev/null | head -c1 || true)
echo "[i] Commit   : $GIT_COMMIT${GIT_DIRTY:+ (dirty)}"

NEW_MANIFEST=$(
  MODEL_PATH="$CKPT_NAME" N_EPISODES="$N_EPISODES" \
  N_ACTION_STEPS="$N_ACTION_STEPS" SEED="$SEED" N_ENVS="$N_ENVS" \
  MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" MAX_STEPS_OVERRIDES="$MAX_STEPS_OVERRIDES" \
  ENV_LAYOUT="$ENV_LAYOUT" ENV_IDX_OFFSET="$ENV_IDX_OFFSET" N_SHARDS="$N_SHARDS" \
  TASK_LIMIT="${TASK_LIMIT:-}" \
  GIT_COMMIT="$GIT_COMMIT" GIT_DIRTY="$GIT_DIRTY" python3 -c '
import json, os
n_envs = int(os.environ["N_ENVS"])
env_layout = int(os.environ["ENV_LAYOUT"])
print(json.dumps({
    "model_path": os.environ["MODEL_PATH"],
    "n_episodes": int(os.environ["N_EPISODES"]),
    "n_action_steps": int(os.environ["N_ACTION_STEPS"]),
    "seed": int(os.environ["SEED"]),
    "max_episode_steps": int(os.environ["MAX_EPISODE_STEPS"]),
    "max_steps_overrides": os.environ["MAX_STEPS_OVERRIDES"],
    "lane_count": env_layout if env_layout > 0 else n_envs,
    # Recorded, not compared.
    "n_envs": n_envs,
    "env_layout": env_layout,
    "env_idx_offset": int(os.environ["ENV_IDX_OFFSET"]),
    "n_shards": int(os.environ["N_SHARDS"]),
    "task_limit": int(os.environ["TASK_LIMIT"] or 0),
    "git_commit": os.environ["GIT_COMMIT"],
    "git_dirty": bool(os.environ["GIT_DIRTY"]),
}, indent=2, sort_keys=True))
'
)

# Shards started as separate processes compare-then-write one at a time.
exec 8>"$OUT_ROOT/.manifest.lock"
flock -x -w 600 8 || { echo "[!] timed out waiting on the manifest lock" >&2; exit 3; }

if ! MANIFEST="$MANIFEST" OUT_ROOT="$OUT_ROOT" NEW_MANIFEST="$NEW_MANIFEST" python3 -c '
import json, os, re, sys
from pathlib import Path

COMPARED = (
    "model_path", "n_episodes", "n_action_steps", "seed",
    "max_episode_steps", "max_steps_overrides", "lane_count",
)
EPISODE = re.compile(r".*_Env_env\d+-episode_\d+-(success|failure)\.mp4$")

new = json.loads(os.environ["NEW_MANIFEST"])
path = Path(os.environ["MANIFEST"])
out_root = Path(os.environ["OUT_ROOT"])
recorded = [p for p in out_root.glob("*/*.mp4") if EPISODE.match(p.name)]
adopted = 0

if path.exists():
    old = json.loads(path.read_text())
    adopted = int(old.get("adopted_without_manifest", 0) or 0)
    diff = [(k, old.get(k, "<missing>"), new[k]) for k in COMPARED if old.get(k) != new[k]]
    if diff:
        print("[!] this directory holds results from a different protocol:", file=sys.stderr)
        for k, was, now in diff:
            print(f"      {k}: recorded {was!r} -> requested {now!r}", file=sys.stderr)
        print(f"    {len(recorded)} episode file(s) are already there and would be resumed,", file=sys.stderr)
        print("    reporting the recorded protocol under the requested one. Evaluate into a", file=sys.stderr)
        print(f"    fresh directory, or delete {out_root} if the old results are not wanted.", file=sys.stderr)
        sys.exit(3)
    was_commit, now_commit = old.get("git_commit"), new["git_commit"]
    if was_commit != now_commit:
        print(f"[warn] resuming results recorded at commit {was_commit}", file=sys.stderr)
        print(f"       under {now_commit}; the protocol matches, the code may not.", file=sys.stderr)
elif recorded:
    adopted = len(recorded)
    print(f"[warn] {adopted} episode file(s) here have no manifest, so the protocol that", file=sys.stderr)
    print("       produced them is unknown; resume will report them under the requested", file=sys.stderr)
    print("       one. Recorded as adopted_without_manifest -- delete the directory if", file=sys.stderr)
    print("       the run has to be clean.", file=sys.stderr)

if adopted:
    new["adopted_without_manifest"] = adopted
path.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
'; then
  exit 3
fi
exec 8>&-
# One 100-port band per run below the ephemeral range, so two runs on a node do
# not score against each other's server. Separately started shards share one BASE_PORT.
BASE_PORT="${BASE_PORT:-$(( 20000 + ($$ % 120) * 100 ))}"

# Unset runs every shard here. Set, it runs that one shard and nothing else.
SHARD_ID="${SHARD_ID:-}"
if [ -n "$SHARD_ID" ] && ! { [ "$SHARD_ID" -ge 0 ] && [ "$SHARD_ID" -lt "$N_SHARDS" ]; } 2>/dev/null; then
  echo "[!] SHARD_ID=$SHARD_ID is outside 0..$((N_SHARDS - 1))" >&2
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

# Polled from python (`ss` is absent on some images). An open port may be
# another run's, so a dead server fails the wait instead of passing it.
SERVER_WAIT_POLLS="${SERVER_WAIT_POLLS:-450}"   # x2s = 15 minutes: eight shards read the checkpoint at once

wait_for_server() {
  local port=$1 pid=$2 log=$3
  for _ in $(seq 1 "$SERVER_WAIT_POLLS"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[!] policy server for port $port exited before serving -- see $log" >&2
      return 1
    fi
    if ! port_is_free "$port"; then
      return 0
    fi
    sleep 2
  done
  echo "[!] server on port $port never came up -- see $log" >&2
  return 1
}

# The python that starts the server. Stop here rather than waste eight shards.
MODEL_PY="$BASE_DIR/.venv/bin/python"
[ -x "$MODEL_PY" ] || { echo "[!] no model venv at $MODEL_PY" >&2; exit 2; }

run_shard() {
  local gpu_id=$1
  local port=$((BASE_PORT + gpu_id))
  local shard_log="$LOG_ROOT/shard-gpu${gpu_id}.log"
  local serve_pid=""

  # Wait for the GPU memory to come back: the swap chain's re-runs follow on the
  # same card. Every return path goes through here.
  stop_server() {
    [ -n "${serve_pid:-}" ] || return 0
    kill "$serve_pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$serve_pid" 2>/dev/null && [ "$waited" -lt 30 ]; do
      sleep 1; waited=$((waited + 1))
    done
    kill -9 "$serve_pid" 2>/dev/null || true
    serve_pid=""
  }

  # Spread the remainder over the first shards so no shard is a task behind.
  local base=$((N_TASKS / N_SHARDS))
  local extra=$((N_TASKS % N_SHARDS))
  local start end
  if [ "$gpu_id" -lt "$extra" ]; then
    start=$((gpu_id * (base + 1)))
    end=$((start + base + 1))
  else
    start=$((extra * (base + 1) + (gpu_id - extra) * base))
    end=$((start + base))
  fi
  # A smoke runs at most TASK_LIMIT of this shard's own tasks; never into the next shard's.
  [ -n "$TASK_LIMIT" ] && [ $((start + TASK_LIMIT)) -lt "$end" ] && end=$((start + TASK_LIMIT))
  [ "$start" -ge "$end" ] && return 0

  # Never adopt an occupied port -- it is another run's model.
  local probe=0
  while ! port_is_free "$port"; do
    probe=$((probe + 1))
    if [ "$probe" -gt 40 ]; then
      echo "[!] shard $gpu_id: no free port in ${BASE_PORT}..$((port)) -- giving up" >&2
      return 1
    fi
    port=$((port + N_SHARDS))
  done
  if [ "$probe" -gt 0 ]; then
    echo "[shard ${gpu_id}] port $((BASE_PORT + gpu_id)) was busy; using $port instead" | tee -a "$shard_log"
  fi

  echo "[shard ${gpu_id}] PORT=${port} tasks=${start}..$((end-1)) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$shard_log"

  # The venv python, not `uv run`, which could sync the venv under eight shards at once.
  "$MODEL_PY" "$BASE_DIR/rldx/eval/run_rldx_server.py" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 \
    --port "$port" >> "$shard_log" 2>&1 &
  serve_pid=$!
  echo "[shard ${gpu_id}] server PID=${serve_pid}" >> "$shard_log"

  wait_for_server "$port" "$serve_pid" "$shard_log" || { stop_server; return 1; }
  echo "[shard ${gpu_id}] server ready on port ${port}" >> "$shard_log"

  # The swap chain's one-env re-run visits each lane of the layout in turn to keep
  # its seeds; lanes that owe nothing return at once.
  local lanes
  if [ "$ENV_LAYOUT" -gt 0 ] && [ "$N_ENVS" -eq 1 ]; then
    lanes=$(seq 0 $((ENV_LAYOUT - 1)))
    # Per lane, not per run: the lane owns 1/ENV_LAYOUT of the episodes.
    local ep_per_call=$((N_EPISODES / ENV_LAYOUT))
  else
    lanes="$ENV_IDX_OFFSET"
    local ep_per_call="$N_EPISODES"
  fi

  for i in $(seq $start $((end - 1))); do
    local task_name=${TASK_NAMES[$i]}
    local out_dir="$OUT_ROOT/$task_name"
    local max_steps=$MAX_EPISODE_STEPS
    for kv in $MAX_STEPS_OVERRIDES; do
      [ "${kv%%=*}" = "$task_name" ] && max_steps="${kv##*=}"
    done
    mkdir -p "$out_dir"
    for lane in $lanes; do
      echo "[shard ${gpu_id}] running ${task_name} lane ${lane}" >> "$shard_log"
      "$BASE_DIR/run_scripts/eval/robocasa_kitchen/robocasa_uv/.venv/bin/python" \
        "$BASE_DIR/rldx/eval/rollout_policy.py" \
          --n_episodes "$ep_per_call" \
          --policy_client_host 127.0.0.1 \
          --policy_client_port "$port" \
          --max_episode_steps "$max_steps" \
          --env_name "robocasa_panda_omron/${task_name}_PandaOmron_Env" \
          --n_action_steps "$N_ACTION_STEPS" \
          --n_envs "$N_ENVS" \
          --env_layout "$ENV_LAYOUT" \
          --env_idx_offset "$lane" \
          --seed "$SEED" \
          --video_dir "$out_dir" \
          >> "$out_dir/eval.log" 2>&1
      rc=$?
      echo "[shard ${gpu_id}] ${task_name} lane ${lane} done (exit=$rc)" >> "$shard_log"
      if [ "$rc" -ne 0 ]; then
        echo "[!] shard ${gpu_id}: ${task_name} lane ${lane} ended with exit=$rc" \
          | tee -a "$shard_log" >&2
        stop_server; return 1
      fi
    done
    # A rollout can record fewer episodes without failing; the chain must not see
    # this shard as complete then.
    local have
    have=$(ls "$out_dir"/*-success.mp4 "$out_dir"/*-failure.mp4 2>/dev/null | wc -l | tr -d " ")
    if [ "$have" -lt "$N_EPISODES" ]; then
      echo "[!] shard ${gpu_id}: ${task_name} recorded only $have/$N_EPISODES -- $out_dir/eval.log" \
        | tee -a "$shard_log" >&2
      stop_server; return 1
    fi
  done

  stop_server
  echo "[shard ${gpu_id}] shard complete" >> "$shard_log"
}

# Card g takes shards g, g+n, ... in turn, so the shard-to-task assignment is the
# same at any card count.
run_local() {
  local cards=() n
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
    n="${N_GPUS:-${#cards[@]}}"
  else
    n="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d " ")}"
  fi
  if ! [ "$n" -ge 1 ] 2>/dev/null; then
    echo "[!] no GPU found -- set N_GPUS or CUDA_VISIBLE_DEVICES" >&2
    return 1
  fi
  [ "$n" -gt "$N_SHARDS" ] && n="$N_SHARDS"
  echo "[i] Cards    : $n for $N_SHARDS shards, up to $(( (N_SHARDS + n - 1) / n )) in turn per card"

  local pids=() g s
  for g in $(seq 0 $((n - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="${cards[$g]:-$g}"
      for s in $(seq "$g" "$n" $((N_SHARDS - 1))); do
        run_shard "$s" || exit 1
      done
    ) &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  return "$rc"
}

if [ -n "$SHARD_ID" ]; then
  run_shard "$SHARD_ID" || {
    echo "[!] shard $SHARD_ID failed -- check $LOG_ROOT" >&2
    exit 1
  }
  echo "[i] Shard $SHARD_ID of $N_SHARDS done. Outputs under $OUT_ROOT"
else
  run_local || {
    echo "[!] a shard failed -- check $LOG_ROOT" >&2
    exit 1
  }
  echo "[i] All $N_SHARDS shards done. Outputs under $OUT_ROOT"
fi

# ---- Whoever finishes the last shard runs the swap chain --------------------
# A failed shard writes no completion line, so the chain never runs on a tree
# with missing episodes.
if [ "${SWAP_CHAIN_INLINE:-1}" = "1" ]; then
  # A shared filesystem can briefly hide another shard's completion line; re-check.
  waited=0
  while [ "$waited" -lt "${SWAP_CHAIN_CLAIM_WAIT:-180}" ]; do
    done_n=$(grep -l "shard complete" "$LOG_ROOT"/shard-gpu*.log 2>/dev/null | wc -l | tr -d " ")
    [ "$done_n" -ge "$N_SHARDS" ] && break
    sleep 15; waited=$((waited + 15))
  done
  # mkdir is atomic, so exactly one shard claims the chain.
  if [ "${done_n:-0}" -ge "$N_SHARDS" ]; then
    if mkdir "$OUT_ROOT/.chain_claim" 2>/dev/null; then
      echo "[i] ${SHARD_ID:+shard $SHARD_ID is the last -- }continuing into the swap chain"
      MODEL_OUTPUT_DIR="$OUT_BASE" bash "$BASE_DIR/run_scripts/eval/robocasa_kitchen/swap_chain.sh" \
        "$OUT_ROOT" || exit $?
    else
      echo "[i] the chain was claimed by another shard"
    fi
  fi
fi
