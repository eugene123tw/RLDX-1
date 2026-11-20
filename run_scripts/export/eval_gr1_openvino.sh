#!/bin/bash
# GR-1 Tabletop evaluation against the OpenVINO policy server (rldx_openvino.py serve).
# Same 24 tasks and protocol as run_scripts/eval/gr1_tabletop/eval_gr1.sh, which stays upstream:
# seed 42, 5-lane seed layout, 8 actions per call, 720 steps, n_envs 1, and the reset protocol
# (5 s settle patch + frozen seed manifest, GR1_PROTOCOL=on). The denoising schedule is the uniform
# one baked into the export, as upstream's TIMESTEP_SAMPLING=uniform default.
# Upstream runs 8 shards, each with its own server; here one server takes MAX_PARALLEL tasks at
# once. Scenes are the same (the seed manifest is keyed on lane and episode, not shard), but the
# noise stream differs anyway (OpenVINO draws it from the CPU torch RNG), so compare success rates.
#
# Usage: eval_gr1_openvino.sh <RUN_NAME> <EXPORT_DIR> [OV_DEVICE] [MAX_PARALLEL]
#   N_EPISODES=50   episodes per task, a multiple of 5 (one fifth per lane). Smaller values run
#                   the first episodes of every lane, so a later larger run resumes and extends them.
#   TASKS=a,b,...   task subset, with or without the _GR1ArmsAndWaistFourierHands_Env suffix
#                   (default: all 24); TASK_LIMIT=n keeps the first n of them.
#   OV_PRECISION    OpenVINO inference precision hint (default: the device's, f16 on GPU).
#   RECORD_DIR      also save every RECORD_EVERY-th (10) observation as a parity fixture.
#
# One-time setup on a fresh machine (from the repo root, branch with run_scripts/export/):
#   1. Model venv:     uv sync --python 3.10
#                      uv pip install --python .venv/bin/python openvino==2026.4.1 nncf==3.4.0
#      Later `uv sync` / `uv run` remove openvino and nncf again; reinstall them after.
#   2. Intel GPU:      compute runtime installed, so the device shows in
#                      .venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"
#   3. Simulator:      INSTALL_FLASH_ATTN=0 bash run_scripts/eval/gr1_tabletop/setup_gr1.sh
#                      (submodule + determinism and reset-settle patches, robosuite 1.5.1, assets)
#   4. Export:         S=run_scripts/export/rldx_openvino.py
#                      .venv/bin/python $S export --model-path RLWRLD/RLDX-1.1-FT-GR1 \
#                          --hidden-alignment 64 --output exports/rldx11_gr1_ov_a64
#                      .venv/bin/python $S quantize --export-dir exports/rldx11_gr1_ov_a64 \
#                          --output exports/rldx11_gr1_ov_a64_int8 --mode int8_sym
#                      .venv/bin/python $S quantize --export-dir exports/rldx11_gr1_ov_a64 \
#                          --output exports/rldx11_gr1_ov_a64_int8asym --mode int8_asym
#      CPU only (~30 GB RAM) and downloads the checkpoint; no recorded observation is needed.
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
RECORD_DIR="${RECORD_DIR:-}"
RECORD_EVERY="${RECORD_EVERY:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-120}"  # seconds between running success-rate lines

# -- Protocol (upstream eval_gr1.sh, GR1_PROTOCOL=on) ----------------------------
SEED=42
LANES=5
N_ACTION_STEPS=8
MAX_EPISODE_STEPS=720
RESET_SETTLE_S=5.0
TASK_SUFFIX="_GR1ArmsAndWaistFourierHands_Env"

if ! [ "$N_EPISODES" -gt 0 ] 2>/dev/null || [ $((N_EPISODES % LANES)) -ne 0 ]; then
  echo "[ERROR] N_EPISODES=$N_EPISODES must be a positive multiple of $LANES" >&2
  exit 1
fi

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
GR1_DIR="$BASE_DIR/run_scripts/eval/gr1_tabletop"
UPSTREAM_EVAL="$GR1_DIR/eval_gr1.sh"
SEED_MANIFEST="$GR1_DIR/gr1_episode_seed_manifest.json"
SIM_PY="$GR1_DIR/robocasa_uv/.venv/bin/python"
TABLETOP="$BASE_DIR/external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/environments/tabletop"
# Not `uv run`: a project sync would drop openvino/nncf, which are not in the lockfile.
SERVER_PY="${SERVER_PY:-$BASE_DIR/.venv/bin/python}"
OUT_BASE="${MODEL_OUTPUT_DIR:-$BASE_DIR/output_final}"
OUT_ROOT="$OUT_BASE/gr1_tabletop/$RUN_NAME"

if [ ! -f "$EXPORT_DIR/rldx_openvino.json" ]; then
  echo "[ERROR] No OpenVINO export at $EXPORT_DIR (missing rldx_openvino.json); see setup step 4" >&2
  exit 1
fi
if [ ! -x "$SIM_PY" ]; then
  echo "[ERROR] GR-1 simulator venv not found; see setup step 3:" >&2
  echo "        INSTALL_FLASH_ATTN=0 bash run_scripts/eval/gr1_tabletop/setup_gr1.sh" >&2
  exit 1
fi
# setup_gr1.sh applies both patches to the submodule; robocasa silently ignores the protocol without them.
if ! grep -q GR1_RESET_SETTLE_S "$TABLETOP/tabletop.py" 2>/dev/null \
   || ! grep -q "all_obj_cats = sorted(set(all_obj_cats))" "$TABLETOP/tabletop_24dc.py" 2>/dev/null; then
  echo "[ERROR] robocasa-gr1-tabletop-tasks lacks the settle or determinism patch; see setup step 3" >&2
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
eval "$(sed -n '/^TASKS=(/,/^)/p' "$UPSTREAM_EVAL" | sed '1s/^TASKS=/TASK_NAMES=/')"
if [ -n "$TASKS" ]; then
  IFS=, read -r -a SELECTED <<< "$TASKS"
  for i in "${!SELECTED[@]}"; do
    t="${SELECTED[$i]%"$TASK_SUFFIX"}$TASK_SUFFIX"
    [[ " ${TASK_NAMES[*]} " == *" $t "* ]] || { echo "[ERROR] unknown task ${SELECTED[$i]}" >&2; exit 1; }
    SELECTED[$i]="$t"
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
     MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" RESET_SETTLE_S="$RESET_SETTLE_S" \
     SEED_MANIFEST="$SEED_MANIFEST" python3 - <<'PY'
import hashlib, json, os, sys
from pathlib import Path

env = os.environ
meta = json.loads((Path(env["EXPORT_DIR"]) / "rldx_openvino.json").read_text())
new = {
    "backend": "openvino",
    "export_dir": env["EXPORT_DIR"],
    "model_path": meta["model_path"],
    "weight_compression": (meta.get("weight_compression") or {}).get("mode", "fp16"),
    "vl_row_scale": meta.get("vl_row_scale"),
    "ov_device": env["OV_DEVICE"],
    "ov_precision": env["OV_PRECISION"] or "default",
    "seed": int(env["SEED"]),
    "lane_count": int(env["LANES"]),
    "n_action_steps": int(env["N_ACTION_STEPS"]),
    "max_episode_steps": int(env["MAX_EPISODE_STEPS"]),
    "timestep_sampling": "uniform",
    "reset_settle_s": float(env["RESET_SETTLE_S"]),
    "seed_manifest_sha256": hashlib.sha256(Path(env["SEED_MANIFEST"]).read_bytes()).hexdigest(),
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
echo "[gr1-proto] settle=${RESET_SETTLE_S}s | manifest=$(basename "$SEED_MANIFEST") | patch OK"

cd "$BASE_DIR"
SERVE_ARGS=(--backend openvino --export-dir "$EXPORT_DIR" --ov-device "$OV_DEVICE")
[ -n "$OV_PRECISION" ] && SERVE_ARGS+=(--ov-precision "$OV_PRECISION")
[ -n "$RECORD_DIR" ] && SERVE_ARGS+=(--record-dir "$RECORD_DIR" --record-every "$RECORD_EVERY")
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

# Live progress from the episode videos; the final score comes from the task CSVs, as upstream.
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
  local task=$1 out="$OUT_ROOT/$1" lane
  mkdir -p "$out"
  for lane in $(seq 0 $((LANES - 1))); do
    # The protocol variables go to the rollout only; gr1_protocol is imported from GR1_DIR.
    GR1_PROTOCOL_SEED="$SEED" GR1_RESET_SETTLE_S="$RESET_SETTLE_S" \
    GR1_EPISODE_SEED_MANIFEST="$SEED_MANIFEST" PYTHONPATH="$GR1_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$SIM_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --env_name "gr1_unified/$task" \
        --n_episodes $((N_EPISODES / LANES)) \
        --max_episode_steps "$MAX_EPISODE_STEPS" \
        --n_action_steps "$N_ACTION_STEPS" \
        --seed "$SEED" \
        --n_envs 1 \
        --env_layout "$LANES" \
        --env_idx_offset "$lane" \
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
if ! OUT_ROOT="$OUT_ROOT" SUMMARY="$SUMMARY" EXPECTED="$N_EPISODES" SUFFIX="$TASK_SUFFIX" \
     python3 - "${SELECTED[@]}" <<'PY'
import csv, os, sys
from pathlib import Path

root, expected, suffix = Path(os.environ["OUT_ROOT"]), int(os.environ["EXPECTED"]), os.environ["SUFFIX"]
ok = n = 0
incomplete = []
lines = ["task,success,episodes,success_rate"]
for task in sys.argv[1:]:
    path = root / task / "simulation_results.csv"
    rows = {}
    if path.is_file():
        rows = {(r.get("env_idx"), r.get("episode_idx")): r.get("success")
                for r in csv.DictReader(path.open()) if r.get("success") in ("0", "1")}
    s = sum(v == "1" for v in rows.values())
    ok, n = ok + s, n + len(rows)
    if len(rows) != expected:
        incomplete.append(f"{task} ({len(rows)}/{expected})")
    rate = s / len(rows) if rows else 0.0
    lines.append(f"{task},{s},{len(rows)},{rate:.3f}")
    print(f"[i] {task.removesuffix(suffix):<56} {s:3d}/{len(rows):<3d} {rate:.3f}")
Path(os.environ["SUMMARY"]).write_text("\n".join(lines) + "\n")
print(f"[i] pooled SR: {ok}/{n} = {100 * ok / max(n, 1):.2f}%")
if incomplete:
    print(f"[!] tasks without {expected} episodes: {', '.join(incomplete)}", file=sys.stderr)
    sys.exit(1)
PY
then
  FAILED+=("(incomplete CSVs)")
fi
SUBS=$(cat "$OUT_ROOT"/*/eval.log 2>/dev/null | grep -c "\[seed-manifest\]")
echo "[i] [seed-manifest] substitutions logged: $SUBS (a full 50-episode run logs about 130)"
echo "[i] Per-task table: $SUMMARY"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[!] ${#FAILED[@]} task(s) failed: ${FAILED[*]} (see <task>/eval.log, server.log)"
  exit 1
fi
