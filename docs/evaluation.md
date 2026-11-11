# Evaluation

This doc covers how to run a trained RLDX-1.1 checkpoint on each of the
simulator benchmarks the repo ships with, plus the general
server/client architecture those evals use.

## Architecture (why there are two processes)

RLDX-1.1 evaluation splits cleanly into two processes:

```
  ┌─────────────────────────┐       ZeroMQ         ┌───────────────────────┐
  │ run_rldx_server.py      │ ───────────────────▶ │  simulator rollout    │
  │ (training env .venv)    │ ◀─────────────────── │  (per-sim eval venv)  │
  │ loads RLDX-1.1 on GPU   │   observations /     │  mujoco/robosuite/... │
  │ answers policy queries  │   actions            │  no model weights     │
  └─────────────────────────┘                      └───────────────────────┘
```

The **model server** lives in the main training venv (`.venv/`), loads
the RLDX-1.1 checkpoint onto the GPU, and listens on a TCP port via ZeroMQ.
The **rollout process** lives in a **simulator-specific uv venv** (for
example `run_scripts/eval/robocasa_kitchen/robocasa_uv/.venv`), steps the
environment, serialises observations to the server, and applies the
returned action chunks. This split exists because the mujoco/robosuite
stacks conflict with the `torch==2.7 + flash-attn` pins required by
training, and keeping them separated means neither env has to compromise.

Every benchmark below follows the same pattern: (1) launch the model
server, (2) launch the rollout worker against that server, (3) the
rollout writes videos + a CSV summary under `output_final/{benchmark}/`.

## Shared flags

### Model server — `rldx/eval/run_rldx_server.py`

| Flag | Default | Notes |
|---|---|---|
| `--model-path` | *(required)* | HF hub id or local checkpoint path. The processor is loaded from `{model_path}/processor` automatically. |
| `--embodiment-tag` | `GENERAL_EMBODIMENT` | Which category-specific encoder/decoder head to use. See `rldx/data/embodiment_tags.py` for the full list. |
| `--host` / `--port` | `0.0.0.0 / 5555` | Where the server listens. `5555` is `DEFAULT_MODEL_SERVER_PORT` in `rldx/serving/server.py`; the examples below use an arbitrary port like `20100` when running multiple servers on one host. |
| `--deactivate-memory` | False | Loads a memory-trained checkpoint as non-memory model for the "ablation without memory" study. |
| `--sample-timestep-from-beta-dist` | False | Use a Beta-distributed denoising timestep instead of uniform. Not with a forward-CUDA-graph runtime profile. |
| `--denoising-timesteps` | None | Explicit list of denoising timesteps; overrides the uniform and Beta schedules. |

### Rollout runner — `rldx/eval/rollout_policy.py`

| Flag | Default | Notes |
|---|---|---|
| `--env-name` | a GR-1 task | Gym env id. Format depends on the benchmark (see below). |
| `--n-episodes` | 50 | Total episodes to run. Split across `n_envs`. |
| `--n-envs` | 8 | Vectorised sim envs. >1 requires robosuite to support async; set to 1 if unsure. |
| `--n-action-steps` | 8 | Execution horizon per action chunk (must be ≤ `action_horizon`). |
| `--max-episode-steps` | 504 | Max steps per episode before truncation. |
| `--policy-client-host/port` | *(required)* | Must match the server. |
| `--video-dir` | None | Output directory for `.mp4` files and `simulation_results.csv`. Resume logic auto-skips already-recorded episodes in the same dir. |

## RoboCasa kitchen (24 tasks)

Launcher: [`eval_robocasa.sh`](../run_scripts/eval/robocasa_kitchen/eval_robocasa.sh)
— 24 tasks, 50 episodes each, 8 shards. GR-1's
[`eval_gr1.sh`](../run_scripts/eval/gr1_tabletop/eval_gr1.sh) takes the same shape.

```bash
bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh RLWRLD/RLDX-1.1-FT-ROBOCASA
```

The launcher runs all 8 shards on this machine, one per visible GPU at a time
(`N_GPUS=<n>` limits that), then the view-swap repair and the score. Each shard
starts its own policy server, so the card count changes only the wall-clock.
`SHARD_ID=<k>` runs one shard alone, for a scheduler that gives each shard a
card of its own; see the
[RoboCasa README](../run_scripts/eval/robocasa_kitchen/README.md).

Single-task example, outside the protocol (one env, no manifest, no swap
repair) -- useful for debugging, not for the reported number:

```bash
# Terminal 1 — model server
uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port 20100

# Terminal 2 — rollout in the robocasa venv
run_scripts/eval/robocasa_kitchen/robocasa_uv/.venv/bin/python \
    rldx/eval/rollout_policy.py \
        --n-episodes 50 \
        --policy-client-host 127.0.0.1 --policy-client-port 20100 \
        --max-episode-steps 720 \
        --env-name "robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env" \
        --n-action-steps 16 \
        --n-envs 1 \
        --video-dir output_final/robocasa/my_ckpt/TurnSinkSpout
```

The 24 kitchen tasks are:

```
TurnSinkSpout       TurnOnStove        TurnOnSinkFaucet    TurnOnMicrowave
TurnOffStove        TurnOffSinkFaucet  TurnOffMicrowave
PnPStoveToCounter   PnPSinkToCounter   PnPMicrowaveToCounter
PnPCounterToStove   PnPCounterToSink   PnPCounterToMicrowave
PnPCounterToCab     PnPCabToCounter
OpenSingleDoor      OpenDrawer         OpenDoubleDoor
CoffeeSetupMug      CoffeeServeMug     CoffeePressButton
CloseSingleDoor     CloseDrawer        CloseDoubleDoor
```

Each one is a separate gym env id of the form
`robocasa_panda_omron/{task}_PandaOmron_Env`. Each shard runs its three tasks
one after another against a single policy server, and results land under
`output_final/robocasa/{CKPT_NAME}/{TASK_NAME}/`
— the runner records its settings in the checkpoint directory's `manifest.json`
and refuses to resume a directory filled under others.

## RoboCasa 365 (larger task set)

[`run_scripts/eval/robocasa_365/`](../run_scripts/eval/robocasa_365/) holds the
launcher [`eval_robocasa365.sh`](../run_scripts/eval/robocasa_365/eval_robocasa365.sh)
and `task_sets.yaml`, which lists the task splits. See the
[RoboCasa365 README](../run_scripts/eval/robocasa_365/README.md) for setup and
the run command.

## LIBERO (4 suites × 10 tasks)

Canonical script:
[`run_scripts/eval/libero/eval_libero.sh`](../run_scripts/eval/libero/eval_libero.sh)

Setup the sim env once, then run:

```bash
bash run_scripts/eval/libero/setup_libero.sh

bash run_scripts/eval/libero/eval_libero.sh <CKPT_NAME> <MODEL_PATH> [GPU_ID] [MAX_PARALLEL]
```

LIBERO env ids look like `libero_sim/<task_name>`, for example
`libero_sim/turn_on_the_stove`. The script runs one model server on `GPU_ID`, and all 40 tasks (10 per suite × 4 suites)
rolled out `MAX_PARALLEL` at a time from the LIBERO venv at
`run_scripts/eval/libero/libero_uv/.venv/bin/python`.

## GR1 tabletop (humanoid)

[`run_scripts/eval/gr1_tabletop/`](../run_scripts/eval/gr1_tabletop/):

```bash
# Setup (one-time)
bash run_scripts/eval/gr1_tabletop/setup_gr1.sh

# All 24 tasks across 8 shards, on every visible GPU
bash run_scripts/eval/gr1_tabletop/eval_gr1.sh RLWRLD/RLDX-1.1-FT-GR1
```

Same shape as RoboCasa: 8 shards, one per card at a time, then the score. See
the [GR1 README](../run_scripts/eval/gr1_tabletop/README.md) for the protocol
variables.

GR-1 Tabletop runs robocasa-gr1-tabletop-tasks on the GR-1 humanoid.
The embodiment tag is `GENERAL_EMBODIMENT`, which
`run_scripts/eval/gr1_tabletop/eval_gr1.sh` already passes to the
server.

## Collecting results

Every rollout writes per-episode `.mp4` files with either `-success` or
`-failure` in the filename, plus a `simulation_results.csv` per task with one
row per episode. The RoboCasa launcher ends with the swap chain's report, which
prints the pooled success rate once the tree holds every episode; the GR-1
launcher ends by printing the pooled success rate over its task CSVs.

## Where to next

- [`inference_server.md`](inference_server.md) — running the same
  `run_rldx_server.py` for real-robot deployment
- [`training.md`](training.md) — produce the checkpoint you are
  evaluating
- [`architecture.md`](architecture.md) — what the model does under the
  hood
