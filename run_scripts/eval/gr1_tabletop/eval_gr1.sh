#!/usr/bin/env bash
# GR-1 Tabletop evaluation -- 24 tasks, 50 episodes each, split across 8 shards.
# Usage: eval_gr1.sh [MODEL_PATH]   (default RLWRLD/RLDX-1.1-FT-GR1)
#
# Runs every shard on this machine, N_GPUS at a time (default: every visible
# card), then the score. SHARD_ID=<k> runs shard k alone, for a scheduler that
# gives each shard a card of its own.
#
# The 24 tasks are split across N_SHARDS shards, each with its own policy server
# on its own port. Sharding is per task and every rollout runs with n_envs=1, so
# `env_seed = seed + env_idx` is 42 in every shard, as it is when the tasks run
# one after another.
#
# Results land under a directory named for the protocol, not just the model --
# see PROTOCOL ISOLATION below.

set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

# The policy server confirms the denoising schedule with a plain print(), which
# block-buffers into the shard log and would otherwise stay invisible for the
# whole run. That line is the only in-run evidence that beta is active.
export PYTHONUNBUFFERED=1

MODEL_PATH="${1:-RLWRLD/RLDX-1.1-FT-GR1}"
# Clone root, derived from `$0` by default. A batch scheduler may copy a
# submitted script elsewhere and run it from there, so `dirname $0` is not the
# clone; `RLDX_ROOT` lets the caller name it and wins when set.
BASE_DIR="${RLDX_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
[ -d "$BASE_DIR/rldx/eval" ] || {
  echo "[!] BASE_DIR is not an RLDX clone: $BASE_DIR" >&2
  echo "    Pass RLDX_ROOT=<clone path>." >&2
  exit 2
}

# Run from the clone root: a checkpoint is usually a relative symlink there and
# only its name is passed, so elsewhere the path does not resolve and is read as
# a HuggingFace hub repo id instead.
cd "$BASE_DIR"
SCRIPT_DIR="$BASE_DIR/run_scripts/eval/gr1_tabletop"
GR1_PY="$BASE_DIR/run_scripts/eval/gr1_tabletop/robocasa_uv/.venv/bin/python"

# Shard count is a protocol axis, not a throughput knob. Each shard is its own
# policy server with its own RNG stream and the count decides which tasks land
# on which server, so two runs at different shard counts are not the same
# measurement. 8 is the standard for this track; anything else warns. The
# number of cards is a separate knob (N_GPUS, below) and changes only how long
# the run takes.
N_SHARDS="${N_SHARDS:-8}"
if [ "$N_SHARDS" -ne 8 ]; then
  echo "[!] N_SHARDS=$N_SHARDS, not the 8 this track compares at -- comparable only within this shard count" >&2
fi

# Unset runs every shard here. Set, it runs that one shard and nothing else.
SHARD_ID="${SHARD_ID:-}"
if [ -n "$SHARD_ID" ] && ! { [ "$SHARD_ID" -ge 0 ] && [ "$SHARD_ID" -lt "$N_SHARDS" ]; } 2>/dev/null; then
  echo "[!] SHARD_ID=$SHARD_ID is outside 0..$((N_SHARDS - 1))" >&2
  exit 2
fi

# Port band, one per run. Two evaluations can put their shard N on one node, and
# a fixed base would then have one run's rollout scored against the other run's
# checkpoint. The probe below cannot close that on its own: it is a connect()
# test, so two shards can both read free in the same instant, and a server
# answering may be the neighbour's while ours is still loading. A caller that
# starts the shards as separate processes passes them one BASE_PORT.
#
# `% 120 * 100` keeps the base under 32768, out of the ephemeral range, and
# leaves 100 ports per run -- enough for 8 shards and the retry below.
BASE_PORT="${BASE_PORT:-$(( 20000 + ($$ % 120) * 100 ))}"
N_EPISODES="${N_EPISODES:-50}"
SEED="${SEED:-42}"

# Action execution horizon: 8 for GR-1, 16 for RoboCasa. Stated rather than
# inherited so the difference between the two tracks is visible.
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

# Inference denoising timestep spacing. `uniform` is the protocol: an even grid
# over the step count, which is what the published GR-1 numbers were produced
# on. `beta` redraws the grid from Beta(1.5, 1.0) on every `get_action` call and
# so carries per-call discretisation variance that uniform does not; it is worth
# trying on an underperforming checkpoint, but it is not the protocol.
#
# The step count is unchanged either way, so NFE matches and the two stay
# comparable on identical seeds -- which is why SEED is part of the output path.
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-uniform}"


# GR-1 reset protocol. Reset ends with a fixed 5 s physics settle so the policy's
# first observation is a still scene, and the slots still broken after it are
# swapped for pre-validated seeds from a frozen manifest. On by default.
#
# Results are not comparable across the switch, since pre-protocol runs started
# their episodes mid-collapse. The protocol is in the output path for that
# reason.
GR1_PROTOCOL="${GR1_PROTOCOL:-on}"
if [ "$GR1_PROTOCOL" = "on" ]; then
  export GR1_RESET_SETTLE_S="${GR1_RESET_SETTLE_S:-5.0}"
  export GR1_PROTOCOL_SEED="$SEED"
  export GR1_EPISODE_SEED_MANIFEST="$SCRIPT_DIR/gr1_episode_seed_manifest.json"
  # The rollout runs under the sim venv, which has no gr1_protocol installed.
  export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

  # The settle lives in a patch against the robocasa submodule, and robocasa
  # ignores GR1_RESET_SETTLE_S when the patch is absent. Nothing in the output
  # distinguishes the two cases, so the patch is applied here rather than assumed
  # -- idempotent, and the marker is checked afterwards.
  #
  # The patch is the one in patches/ next to this script, which setup_gr1.sh globs
  # and applies too, so both paths use one file. The marker check below is what
  # keeps them from double-applying.
  _tt="$BASE_DIR/external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/environments/tabletop/tabletop.py"
  if ! grep -q GR1_RESET_SETTLE_S "$_tt" 2>/dev/null; then
    _rc_dir="$BASE_DIR/external_dependencies/robocasa-gr1-tabletop-tasks"
    _p="$BASE_DIR/run_scripts/eval/gr1_tabletop/patches/01-robocasa-reset-settle.patch"
    echo "[gr1-proto] settle patch missing -- applying $(basename "$_p")"
    [ -f "$_p" ] || {
      echo "[!] settle patch not found at $_p" >&2
      exit 2
    }
    if patch -p1 -d "$_rc_dir" --dry-run --silent < "$_p" 2>/dev/null; then
      patch -p1 -d "$_rc_dir" --silent < "$_p"
    else
      echo "[!] $(basename "$_p") does not apply to this robocasa revision" >&2
      exit 2
    fi
  fi
  grep -q GR1_RESET_SETTLE_S "$_tt" || {
    echo "[!] settle marker still absent in $_tt -- refusing to run a half protocol" >&2
    exit 2
  }
  echo "[gr1-proto] settle=${GR1_RESET_SETTLE_S}s | manifest=$(basename "$GR1_EPISODE_SEED_MANIFEST") | patch OK"
  # The manifest was measured on 5 lanes of N_EPISODES/5 episodes each; its keys
  # are (SEED + lane), so the run has to be laid out the same way to match them.
  ENV_LAYOUT="${ENV_LAYOUT:-5}"
else
  ENV_LAYOUT="${ENV_LAYOUT:-0}"
fi
# Guarded expansion below: bash 3.2 (still the default on macOS) treats an
# empty array as unset under `set -u`, and the uniform arm is empty.
SAMPLING_ARGS=()
case "$TIMESTEP_SAMPLING" in
  beta)    SAMPLING_ARGS+=(--sample-timestep-from-beta-dist) ;;
  uniform) ;;
  *)
    echo "[!] TIMESTEP_SAMPLING must be 'beta' or 'uniform' (got '$TIMESTEP_SAMPLING')" >&2
    exit 2
    ;;
esac

# Do not set --denoising-timesteps here. It outranks both schedules (explicit
# list > beta > uniform), so it would undo the setting above, and a bare integer
# is parsed as a list of timesteps rather than a step count.

TASKS=(
  "PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
)
N_TASKS=${#TASKS[@]}

# Smoke-test knob: run only the first N tasks. The protocol is 24, so the count
# goes in the output path and a partial run cannot land where a full one would.
TASK_LIMIT="${TASK_LIMIT:-}"
if [ -n "$TASK_LIMIT" ]; then
  if ! { [ "$TASK_LIMIT" -ge 1 ] && [ "$TASK_LIMIT" -le "$N_TASKS" ]; } 2>/dev/null; then
    echo "[!] TASK_LIMIT must be between 1 and $N_TASKS (got '$TASK_LIMIT')" >&2
    exit 2
  fi
  TASKS=("${TASKS[@]:0:$TASK_LIMIT}")
  N_TASKS=${#TASKS[@]}
  echo "[!] TASK_LIMIT=$TASK_LIMIT -- $N_TASKS tasks only. Not a comparable result." >&2
fi

# -- PROTOCOL ISOLATION ------------------------------------------------------
# `rollout_policy.py` resumes: a task directory that already holds the requested
# episodes is returned as-is. That makes a re-run cheap, and it is also how a
# protocol change would report the old protocol's numbers -- a horizon-16
# directory satisfies a horizon-8 request. So the protocol goes in the path.
TAG=$(echo "$MODEL_PATH" | tr '/' '_')
PROTOCOL="h${N_ACTION_STEPS}_${TIMESTEP_SAMPLING}_ep${N_EPISODES}_seed${SEED}${TASK_LIMIT:+_t$TASK_LIMIT}"
# Pre- and post-protocol episodes are different scenes, so they must not land in
# the same directory: resume would fill one run from the other.
[ "$GR1_PROTOCOL" = "on" ] && PROTOCOL="${PROTOCOL}_resetproto"
OUT_ROOT="${MODEL_OUTPUT_DIR:-$BASE_DIR/output_final}/gr1_tabletop/$TAG/$PROTOCOL"
LOG_ROOT="$OUT_ROOT/_launcher_logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

GIT_COMMIT=$(git -C "$BASE_DIR" rev-parse --verify -q HEAD 2>/dev/null \
             || cat "$BASE_DIR/SOURCE_COMMIT" 2>/dev/null \
             || echo unknown)
GIT_DIRTY=$(git -C "$BASE_DIR" status --porcelain 2>/dev/null | head -c1 || true)

# The path carries the axes varied deliberately; the manifest carries the rest.
# An axis that lives only in the manifest -- `max_episode_steps`, `n_envs` --
# could otherwise change under an unchanged path and resume would return the old
# protocol's episodes, so the manifest is compared before the directory is read.
#
# Two fields are recorded rather than enforced, because refusing on either would
# stop runs that are not wrong: `git_commit`, since a changed commit with an
# unchanged protocol is a warning and not a wrong number; and a missing
# manifest, which every tree predating this file has. Episodes adopted that way
# are counted in `adopted_without_manifest` so a mixed tree says so.
MANIFEST="$OUT_ROOT/manifest.json"
NEW_MANIFEST=$(
  MODEL_PATH="$MODEL_PATH" N_ACTION_STEPS="$N_ACTION_STEPS" \
  TIMESTEP_SAMPLING="$TIMESTEP_SAMPLING" N_EPISODES="$N_EPISODES" SEED="$SEED" \
  N_SHARDS="$N_SHARDS" ENV_LAYOUT="$ENV_LAYOUT" \
  GR1_RESET_SETTLE_S="${GR1_RESET_SETTLE_S:-}" \
  GIT_COMMIT="$GIT_COMMIT" GIT_DIRTY="$GIT_DIRTY" python3 -c '
import json, os
print(json.dumps({
    "model_path": os.environ["MODEL_PATH"],
    "n_action_steps": int(os.environ["N_ACTION_STEPS"]),
    "timestep_sampling": os.environ["TIMESTEP_SAMPLING"],
    "n_episodes": int(os.environ["N_EPISODES"]),
    "seed": int(os.environ["SEED"]),
    "max_episode_steps": 720,
    "n_envs": 1,
    # The scene selector: episode seeds key on (SEED + lane), so the lane count
    # decides which initial scenes the run saw.
    "lane_count": int(os.environ["ENV_LAYOUT"]) or 1,
    # The manifest of replacement seeds was judged at this settle, so a different
    # value evaluates different scenes under the same substitutions.
    "reset_settle_s": float(os.environ["GR1_RESET_SETTLE_S"] or 0),
    "n_shards": int(os.environ["N_SHARDS"]),
    "git_commit": os.environ["GIT_COMMIT"],
    "git_dirty": bool(os.environ["GIT_DIRTY"]),
}, indent=2, sort_keys=True))
'
)

# Compare-then-write, so eight shard processes arriving at once must not interleave.
# The lock makes the first write and the rest verify what it wrote, which is the
# sequence a single job produces.
exec 8>"$OUT_ROOT/.manifest.lock"
flock -x -w 600 8 || { echo "[!] timed out waiting on the manifest lock" >&2; exit 3; }

if ! MANIFEST="$MANIFEST" OUT_ROOT="$OUT_ROOT" NEW_MANIFEST="$NEW_MANIFEST" python3 -c '
import json, os, re, sys
from pathlib import Path

PROTOCOL = (
    "model_path", "n_action_steps", "timestep_sampling",
    "n_episodes", "seed", "max_episode_steps", "n_envs", "n_shards",
    "lane_count", "reset_settle_s",
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
    diff = [(k, old.get(k, "<missing>"), new[k]) for k in PROTOCOL if old.get(k) != new[k]]
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

echo "[i] Model path  : $MODEL_PATH"
echo "[i] Tasks       : $N_TASKS across $N_SHARDS shards"
echo "[i] Episodes    : $N_EPISODES per task (seed $SEED)"
echo "[i] Exec horizon: $N_ACTION_STEPS"
echo "[i] Timesteps   : $TIMESTEP_SAMPLING"
echo "[i] Commit      : $GIT_COMMIT${GIT_DIRTY:+ (dirty)}"
echo "[i] Output      : $OUT_ROOT"

cd "$BASE_DIR"

port_is_free() {
  python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(1 if s.connect_ex(('127.0.0.1', $1)) == 0 else 0)
" 2>/dev/null
}

# Poll the port from python rather than with `ss`, which is absent on some
# images. An open port is not proof that our server owns it, so the server
# process is checked for liveness on every poll: if it died the wait fails here
# instead of succeeding against somebody else's model.
#
# The budget is set by how long the checkpoint takes to read off shared storage
# with every shard starting at once; four servers on one node took over 5 minutes.
SERVER_WAIT_POLLS="${SERVER_WAIT_POLLS:-450}"   # x2s = 15 minutes

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
  local shard=$1
  local port=$((BASE_PORT + shard))
  local shard_log="$LOG_ROOT/shard_${shard}.log"
  # Not local: a shard that returns early must leave the pid for the EXIT trap,
  # and a card that runs several shards in turn must not keep the last one's.
  serve_pid=""

  # `trap ... RETURN` does not fire when `set -e` aborts the shard, which is the
  # case that leaves a server behind. EXIT covers abort, normal end and eviction.
  # shellcheck disable=SC2064
  trap 'if [ -n "${serve_pid:-}" ]; then kill "$serve_pid" 2>/dev/null || true; wait "$serve_pid" 2>/dev/null || true; fi' EXIT INT TERM

  # Spread the remainder over the first shards so no shard is a task behind.
  local base=$((N_TASKS / N_SHARDS))
  local extra=$((N_TASKS % N_SHARDS))
  local start end
  if [ "$shard" -lt "$extra" ]; then
    start=$((shard * (base + 1)))
    end=$((start + base + 1))
  else
    start=$((extra * (base + 1) + (shard - extra) * base))
    end=$((start + base))
  fi
  [ "$start" -ge "$end" ] && return 0

  # An occupied port must never be adopted: whatever is listening belongs to
  # another run, and rolling out against it would evaluate a different model.
  # Refusing outright is wrong too, since shards land wherever a card frees and
  # share this range with other evaluations. The port decides nothing about
  # the numbers, so the shard takes the next free one and reports which.
  local probe=0
  while ! port_is_free "$port"; do
    probe=$((probe + 1))
    if [ "$probe" -gt 40 ]; then
      echo "[!] shard $shard: no free port in ${BASE_PORT}..$((port)) -- giving up" >&2
      return 1
    fi
    port=$((port + N_SHARDS))
  done
  if [ "$probe" -gt 0 ]; then
    echo "[shard $shard] port $((BASE_PORT + shard)) was busy; using $port instead" | tee -a "$shard_log"
  fi

  # CUDA_VISIBLE_DEVICES is not set here: run_local sets it per card, and a
  # scheduler that grants each shard one card already exposes it as device 0.
  echo "[shard $shard] PORT=$port tasks=${start}..$((end - 1))" | tee -a "$shard_log"
  # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so name the card explicitly.
  echo "[shard $shard] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$shard_log"
  nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} 2>/dev/null | sed "s/^/[shard $shard] /" | tee -a "$shard_log"

  # Call the venv python directly rather than `uv run`. uv would have to be on
  # PATH, and it would also be free to sync the venv inside the job -- eight
  # shards editing one venv at once.
  "$MODEL_PY" "$BASE_DIR/rldx/eval/run_rldx_server.py" \
      --model-path "$MODEL_PATH" \
      --embodiment-tag GENERAL_EMBODIMENT \
      ${SAMPLING_ARGS[@]+"${SAMPLING_ARGS[@]}"} \
      --host 127.0.0.1 --port "$port" >> "$shard_log" 2>&1 &
  serve_pid=$!

  wait_for_server "$port" "$serve_pid" "$shard_log" || return 1

  # The seed manifest is keyed by (SEED + env lane), so the run has to visit the
  # same lanes it was measured on. One env at a time, ENV_LAYOUT lanes in turn.
  local lanes ep_per_call
  if [ "$ENV_LAYOUT" -gt 0 ]; then
    lanes=$(seq 0 $((ENV_LAYOUT - 1)))
    ep_per_call=$((N_EPISODES / ENV_LAYOUT))
  else
    lanes=0
    ep_per_call="$N_EPISODES"
  fi

  local i task out_dir lane
  for i in $(seq "$start" $((end - 1))); do
    task=${TASKS[$i]}
    out_dir="$OUT_ROOT/$task"
    mkdir -p "$out_dir"
    for lane in $lanes; do
      echo "[shard $shard] running $task lane $lane" >> "$shard_log"
      # The simulator venv omits the model stack -- its numpy/numba pins conflict
      # with the model environment's -- so importing `rldx.data` must not pull
      # that stack in. Set on the sim process only, never on the server.
      "$GR1_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
          --policy_client_host 127.0.0.1 --policy_client_port "$port" \
          --env_name "gr1_unified/$task" \
          --n_episodes "$ep_per_call" \
          --max_episode_steps 720 \
          --n_action_steps "$N_ACTION_STEPS" \
          --seed "$SEED" \
          --n_envs 1 \
          --env_layout "$ENV_LAYOUT" \
          --env_idx_offset "$lane" \
          --video_dir "$out_dir" >> "$out_dir/eval.log" 2>&1
      rc=$?
      # bash disables errexit inside a function whose status is being tested, and
      # `run_shard` is invoked that way. A failing rollout therefore has to be
      # propagated by hand, or the shard exits 0 with nothing recorded. Status is
      # checked first, then the artifact.
      if [ "$rc" != 0 ]; then
        echo "[!] shard $shard: $task lane $lane exited $rc -- see $out_dir/eval.log" \
          | tee -a "$shard_log" >&2
        return 1
      fi
      echo "[shard $shard] $task lane $lane done" >> "$shard_log"
    done
    # An exit status of 0 is not the result. Count what was recorded.
    if ! EXPECTED="$N_EPISODES" CSV="$out_dir/simulation_results.csv" python3 -c '
import csv, os, sys
from pathlib import Path
path = Path(os.environ["CSV"])
if not path.is_file():
    sys.exit("no simulation_results.csv")
rows = list(csv.DictReader(path.open()))
done = {(r.get("env_idx"), r.get("episode_idx")) for r in rows if r.get("success") in ("0", "1")}
expected = int(os.environ["EXPECTED"])
if len(done) != expected:
    sys.exit(f"expected {expected} episodes, found {len(done)}")
' 2>>"$shard_log"; then
      echo "[!] shard $shard: $task did not record $N_EPISODES episodes -- see $out_dir/eval.log" \
        | tee -a "$shard_log" >&2
      return 1
    fi
  done
  kill "$serve_pid" 2>/dev/null || true
  wait "$serve_pid" 2>/dev/null || true
  serve_pid=""
  echo "[shard $shard] complete" >> "$shard_log"
}

# Every shard on this machine. Card g takes shards g, g+N_GPUS, ... in turn, each
# with a fresh server, so the shard-to-task assignment -- and the measurement --
# is the same at any card count; fewer cards only take longer.
#
# Each card's loop is its own process group (`set -m`), so an interrupt can take
# down its server and rollout along with it instead of leaving them on the GPU.
run_local() {
  local cards=() n
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
    n="${N_GPUS:-${#cards[@]}}"
  else
    n="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d " " || true)}"
  fi
  if ! [ "$n" -ge 1 ] 2>/dev/null; then
    echo "[!] no GPU found -- set N_GPUS or CUDA_VISIBLE_DEVICES" >&2
    return 1
  fi
  [ "$n" -gt "$N_SHARDS" ] && n="$N_SHARDS"
  echo "[i] Cards       : $n for $N_SHARDS shards, up to $(( (N_SHARDS + n - 1) / n )) in turn per card"

  local pids=() g s
  set -m
  for g in $(seq 0 $((n - 1))); do
    (
      # Off again inside, so the server and rollout stay in this card's group.
      set +m
      export CUDA_VISIBLE_DEVICES="${cards[$g]:-$g}"
      for s in $(seq "$g" "$n" $((N_SHARDS - 1))); do
        run_shard "$s" || exit 1
      done
    ) &
    pids+=($!)
  done
  # shellcheck disable=SC2064
  trap 'for p in "${pids[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done; sleep 2
        for p in "${pids[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; exit 130' INT TERM

  local rc=0 p
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  trap - INT TERM
  set +m
  return "$rc"
}

if [ -n "$SHARD_ID" ]; then
  # This process succeeds or fails on its own shard only; score the tree once
  # every shard is done.
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
  # Every shard already checked its task CSVs hold N_EPISODES episodes each.
  python3 - "$OUT_ROOT" <<'PY'
import csv, glob, os, sys
ok = n = 0
for f in glob.glob(os.path.join(sys.argv[1], "*", "simulation_results.csv")):
    rows = {(r["env_idx"], r["episode_idx"]): r["success"] for r in csv.DictReader(open(f))}
    n += len(rows)
    ok += sum(s == "1" for s in rows.values())
print(f"[i] pooled SR: {ok}/{n} = {100 * ok / max(n, 1):.2f}%")
PY
fi
