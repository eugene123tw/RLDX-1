#!/bin/bash
# RoboCasa Kitchen view-swap chain: scan -> quarantine and re-evaluate -> report,
# looped in one process until it converges.
#
# repair.py's apply step re-evaluates through SUBMIT_CMD, a template run through
# subprocess.run(shell=True). The chain passes an inline loop, so the
# re-evaluation runs on the GPU this process already holds.
#
# Usage:
#   MODEL_OUTPUT_DIR=<results root> bash swap_chain.sh <run directory>
# Not normally called directly -- eval_robocasa.sh runs it once every shard is done.
set -uo pipefail

RUN_DIR=${1:?"Usage: $0 <run directory>"}
[ "$#" -eq 1 ] || { echo "Usage: $0 <run directory>" >&2; exit 2; }
: "${MODEL_OUTPUT_DIR:?set MODEL_OUTPUT_DIR to the results root}"

# The clone that ran the eval, so the repair does not mix code versions in one tree.
BASE_DIR="${RLDX_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV="$BASE_DIR/run_scripts/eval/robocasa_kitchen/robocasa_uv/.venv/bin/python"
[ -x "$VENV" ] || { echo "[!] no simulator venv at $VENV" >&2; exit 2; }

export GR00T_ROOT="$BASE_DIR"
export PYTHONPATH="$BASE_DIR/run_scripts/eval/robocasa_kitchen"
export PATH="$HOME/.local/bin:$PATH"
# The re-run must render on the backend the run used; egl is only the fallback.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_NO_SYNC=1

MANIFEST="$RUN_DIR/manifest.json"
[ -f "$MANIFEST" ] || { echo "[!] no manifest at $MANIFEST" >&2; exit 2; }

# The re-run uses the values the run recorded. Tab-separated: the default IFS
# would drop an empty field and shift the rest.
IFS=$'\t' read -r CKPT N_SHARDS N_EPISODES LANES RUN_COMMIT < <(
  MANIFEST="$MANIFEST" python3 -c '
import json, os
m = json.load(open(os.environ["MANIFEST"]))
print("\t".join(str(m.get(k, "")) for k in
      ("model_path", "n_shards", "n_episodes", "lane_count", "git_commit")))
')
for v in CKPT N_SHARDS N_EPISODES LANES; do
  [ -n "${!v}" ] || { echo "[!] manifest has no $v -- the run was not made by this launcher" >&2; exit 2; }
done

# Do not mix episodes from different code into one tree.
HEAD_COMMIT=$(git -C "$BASE_DIR" rev-parse --verify -q HEAD 2>/dev/null \
              || cat "$BASE_DIR/SOURCE_COMMIT" 2>/dev/null || echo unknown)
if [ -n "$RUN_COMMIT" ] && [ "$RUN_COMMIT" != "$HEAD_COMMIT" ]; then
  echo "[!] this tree was made by commit $RUN_COMMIT; this clone is $HEAD_COMMIT." >&2
  echo "    The repair adds episodes to that tree, so the two would be mixed." >&2
  echo "    Run from the clone that ran the eval, or pass SWAP_CHAIN_ALLOW_COMMIT_DRIFT=1." >&2
  [ "${SWAP_CHAIN_ALLOW_COMMIT_DRIFT:-0}" = "1" ] || exit 3
  echo "    -> SWAP_CHAIN_ALLOW_COMMIT_DRIFT=1 is set; continuing." >&2
fi

echo "[chain] clone    : $BASE_DIR ($HEAD_COMMIT)"
echo "[chain] run      : $RUN_DIR"
echo "[chain] manifest : shards=$N_SHARDS ep=$N_EPISODES lanes=$LANES"

# One env avoids the view swap; ENV_LAYOUT keeps the lane seeds, and the runner
# names the same directory from it.
export SUBMIT_CMD='for s in $(echo {array} | tr , " "); do
  echo "[inline-repair] shard $s";
  SHARD_ID=$s MODEL_OUTPUT_DIR='"$MODEL_OUTPUT_DIR"' \
  N_EPISODES='"$N_EPISODES"' N_ENVS=1 ENV_LAYOUT='"$LANES"' \
  bash '"$BASE_DIR"'/run_scripts/eval/robocasa_kitchen/eval_robocasa.sh '"$CKPT"' \
    || echo "[inline-repair] shard $s rc=$?";
done'

# The re-run inherits this CWD, where a relative checkpoint path resolves.
cd "$BASE_DIR"
say() { echo "[chain $(date -Is)] $*"; }

# Scorable episodes; quarantined ones do not count.
episodes_now() {
  find "$RUN_DIR" -path "*_quarantine*" -prune -o -name "*-success.mp4" -print -o \
       -path "*_quarantine*" -prune -o -name "*-failure.mp4" -print 2>/dev/null | wc -l
}
# A task directory is one holding an mp4; bookkeeping directories would inflate the total.
N_TASKS_SEEN=$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -not -name "_*" \
  -exec sh -c 'find "$1" -name "*.mp4" -print -quit | grep -q .' _ {} \; -print | wc -l)
EXPECTED=$(( N_EPISODES * N_TASKS_SEEN ))
BEFORE=$(episodes_now)
say "tree: tasks $N_TASKS_SEEN, episodes $BEFORE of $EXPECTED expected"

# A re-run can swap again, so loop -- but at most three times.
for cycle in 1 2 3; do
  say "=== cycle $cycle | scan ==="
  "$VENV" -m kitchen_protocol.repair scan "$RUN_DIR" --n-gpus "$N_SHARDS" \
    || { say "scan failed"; exit 1; }

  n=$(RUN_DIR="$RUN_DIR" python3 -c '
import json, os
m = json.load(open(os.path.join(os.environ["RUN_DIR"], "swap_manifest.json")))
print(len(m["contaminated"]))')
  have=$(episodes_now)
  say "cycle $cycle: $n contaminated, episodes $have/$EXPECTED"
  # Zero alone is not convergence: quarantined files are gone even if the re-run failed.
  if [ "$n" -eq 0 ]; then
    if [ "$have" -ge "$EXPECTED" ]; then
      say "converged: zero contaminated and the tree is whole"; break
    fi
    say "[!] zero contaminated but episodes are $have/$EXPECTED -- the quarantined ones were not re-evaluated"
    if [ "$cycle" -eq 3 ]; then
      say "[!] the re-evaluation never refilled them. See $RUN_DIR/_launcher_logs"
      exit 1
    fi
  fi

  say "=== cycle $cycle: apply (quarantine and inline re-evaluation) ==="
  "$VENV" -m kitchen_protocol.repair apply "$RUN_DIR" --n-gpus "$N_SHARDS" --submit \
    || { say "apply failed"; exit 1; }

  say "=== cycle $cycle | report ==="
  "$VENV" -m kitchen_protocol.repair report "$RUN_DIR" --n-gpus "$N_SHARDS" || true
done

say "=== final report ==="
"$VENV" -m kitchen_protocol.repair report "$RUN_DIR" --n-gpus "$N_SHARDS" || true

FINAL=$(episodes_now)
if [ "$FINAL" -lt "$EXPECTED" ]; then
  say "[!] episodes $FINAL/$EXPECTED -- the tree is incomplete. Do not use this SR."
  say "[!] quarantined: $(find "$RUN_DIR/_quarantine_swap" -name "*.mp4" 2>/dev/null | wc -l)"
  exit 1
fi
say "chain complete -- episodes $FINAL/$EXPECTED"
