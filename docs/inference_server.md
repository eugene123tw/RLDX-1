# Inference server

`rldx/eval/run_rldx_server.py` is the canonical way to serve an RLDX-1.1
checkpoint. It holds the model on the GPU and answers policy queries
over a simple TCP / ZeroMQ REQ/REP protocol with msgpack payloads, so
any client — simulator rollout, real-robot driver, tele-op bridge —
can use it as long as it speaks the protocol.

This doc covers the server CLI, the client-side handshake, and two
deployment recipes (simulator eval, real robot).

## Quick start

```bash
# Start the server (holds the GPU)
uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port 5555
```

On first launch the server downloads the backbone, loads the action
head, runs the MSAT init, and starts listening. Expect ~30-60 s for
an 8B Qwen3-VL backbone on a warm disk cache, up to several minutes on
a cold cache.

Once you see

```
Starting RLDX-1 inference server...
  Embodiment tag: EmbodimentTag.GENERAL_EMBODIMENT
  Model path: RLWRLD/RLDX-1.1-FT-ROBOCASA
  Device: cuda
  Host: 127.0.0.1
  Port: 5555
```

followed by the MSAT init log and
`✓ Server ready — listening on 127.0.0.1:5555`, the server is ready to
accept clients.

## Server CLI reference

All flags come from `ServerArgs` in `rldx/eval/run_rldx_server.py`
and are parsed by `tyro`.

### Loading the policy

| Flag | Default | Notes |
|---|---|---|
| `--model-path` | required | HF hub id or absolute path to a trained RLDX-1.1 checkpoint. Processor is loaded from `{model_path}/processor`. |
| `--embodiment-tag` | `GENERAL_EMBODIMENT` | Category-specific MLP head to use. Complete list in `rldx/data/embodiment_tags.py`. |
| `--device` | `cuda` | Pass `cuda:1` to pin to a specific GPU. |

### Replaying a dataset

`rldx/eval/run_replay_server.py` serves a [`ReplayPolicy`](../rldx/eval/replay_policy.py)
over the same protocol: it answers queries with a LeRobot dataset's recorded
actions instead of model inference, which is useful for debugging the wire
protocol without loading the real model. It takes `--embodiment-tag`, `--host`
and `--port` like the model server, and:

| Flag | Default | Notes |
|---|---|---|
| `--dataset-path` | required | Path to a LeRobot dataset. |
| `--execution-horizon` | required | Actions the robot executes from each chunk; the replay advances by it (`1 ≤ s ≤ action_horizon`). The server cannot know it, so it is not defaulted. |
| `--modality-config-path` | None | Either a `.py` config that calls `register_modality_config` (the form training takes — see `rldx/configs/data/*_config.py`) or a `.json` mapping of modality name to [`ModalityConfig`](../rldx/data/types.py) fields, not the dataset's own `meta/modality.json`. Defaults to the `MODALITY_CONFIGS` entry for `--embodiment-tag`. |

### Inference-time sampling knobs

| Flag | Default | Notes |
|---|---|---|
| `--sample-timestep-from-beta-dist` | False | Draw the inference denoising schedule from the training Beta every call instead of uniform. Used for the beta-dist ablation; rejected with a forward-CUDA-graph `--runtime-profile`. |
| `--denoising-timesteps` | None | Fixed denoising schedule, e.g. `0.0 0.1 0.3 0.6`. Overrides the default `num_inference_timesteps = 4` Euler steps. |
| `--deactivate-memory` | False | Load a memory-trained checkpoint as if it were a non-memory model. Memory weights are skipped on load. Useful for "does the memory module actually help" ablations. |

### Real-Time Chunking (RTC)

How to run RTC is chosen here, not read from the checkpoint; the checkpoint
only records `rtc_training_max_delay`. RTC is **off by default** — give
`rtc:trained` or `rtc:guided` with `--rtc.delay` and `--rtc.exec-horizon` to
engage it:

```bash
python rldx/eval/run_rldx_server.py --model-path … --embodiment-tag … \
    rtc:guided --rtc.delay 3 --rtc.exec-horizon 8
```

A flag the chosen mode does not take is rejected. `trained` with a delay above the
checkpoint's `rtc_training_max_delay` is rejected at load. A memory checkpoint
is served without RTC: its memory keeps one snapshot per call and assumes calls
`stride` steps apart, which RTC's varying delays do not keep, so RTC flags with
a memory checkpoint are rejected at load.

| Flag | Default | Notes |
|---|---|---|
| `rtc:guided` / `rtc:trained` | off | `guided` runs the VJP guidance term per denoising step (training-free, [Black et al. (Real-Time Execution of Action Chunking Flow Policies)](https://arxiv.org/abs/2506.07339)). `trained` inpaints the prefix at `τ=1` and requires a checkpoint trained with `--rtc-training-max-delay > 0` ([Black et al. (Training-Time Action Conditioning for Efficient Real-Time Chunking)](https://arxiv.org/abs/2512.05964)). |
| `--rtc.delay` | required | Frozen-prefix length `d` carried over from the previous chunk; at least 1. |
| `--rtc.exec-horizon` | required | Actions the robot executes from each chunk before it asks again (`d ≤ s ≤ action_horizon − d`). |
| `--rtc.jacobian-beta` | `5.0` | Guidance scale clip for `guided` mode (Eq. 4 of [Black et al. (Real-Time Execution of Action Chunking Flow Policies)](https://arxiv.org/abs/2506.07339)). Inference-only knob; not stored on the checkpoint. |
| `--rtc.jacobian-steps-only` | `3` | `guided` only. Apply VJP guidance only on the first `N` denoising steps. `1` = cheapest single-step variant, `None` = all steps. |

`guided` and `trained` are not equivalent. Pick `guided` when the
checkpoint was trained without RTC (any flow-matching checkpoint
qualifies); pick `trained` when the checkpoint was trained with
`--rtc-training-max-delay > 0`. The `trained` path is faster; `guided`
runs a VJP per guided denoising step.

### Inference runtime (`--runtime-profile`)

The policy loads through `InferenceRuntimeProfile`
(`rldx/configs/execution.py`), which decides compile regions, quantization
and CUDA graphs.

| Flag | Default | Notes |
|---|---|---|
| `--runtime-profile` | `compile` | A preset per recipe, and per device for nvfp4: `compile` (bf16), `eager` (bf16, no compile), `any:bf16`, `any:fp8`, `thor:nvfp4-mix`, `thor:nvfp4`, `gb10:nvfp4-mix`, `gb10:nvfp4`. See `PRESETS` in `rldx/configs/execution.py`. |
| `limits:on --limits.batch B --limits.instruction-tokens T` | off | Largest request batch and instruction (in tokens) the server accepts; both are required. Every preset other than `compile` and `eager` needs it to bound its forward CUDA graphs. With limits on, larger requests are rejected, the nvfp4 GEMM is tuned for that range, and every batch size up to the limit is compiled at load. |

The first request after start compiles; send one warm-up request before
serving. Changing the instruction length does not add compiles.

### Server / protocol

| Flag | Default | Notes |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address. Listens on every interface by default; pass `127.0.0.1` for loopback-only. |
| `--port` | `5555` (`DEFAULT_MODEL_SERVER_PORT`) | Server port. |

## Client protocol (`PolicyServer` / `PolicyClient`)

The server is built on
[`rldx/serving/server.py`](../rldx/serving/server.py).
`PolicyServer` is a ZeroMQ REQ/REP server (`zmq.REP` socket bound to
`tcp://{host}:{port}`); `PolicyClient` is the matching `zmq.REQ`
client. Each request is a msgpack-encoded dict
`{"endpoint": "<name>", "data": {...}, "api_token": "..."}`; the reply
is the endpoint's return value, also msgpack-encoded.

Default endpoints registered by `PolicyServer.__init__`:

| Endpoint | Purpose |
|---|---|
| `ping` | Liveness check; returns immediately. No input. |
| `kill` | Stop the server. No input. |
| `get_action` | Forward a single observation batch and return the predicted action chunk. |
| `get_modality_config` | The observation and action keys the checkpoint serves. No input. |
| `get_rtc` | `{delay, exec_horizon, prefix_steps}` when RTC is on, else `None`. No input. |
| `get_memory` | `{length, stride}` for a memory checkpoint, else `None`. No input. |
| `reset` | Memory checkpoint: drop the named `session_ids`, or every session. Otherwise nothing to drop. |

`rldx.serving.policy.load_policy` serves a checkpoint without memory with the
stateless `RLDXPolicy` and a memory checkpoint with `RLDXMemoryPolicy`, which
keeps each session's memory between calls.

Numpy arrays and `ModalityConfig` are tunneled through msgpack via the
custom `MsgSerializer.encode_custom_classes` hook in `rldx/serving/server.py`.

### Python client

```python
import numpy as np
from rldx.serving.server import PolicyClient

client = PolicyClient(host="127.0.0.1", port=5555)

observations = {
    "video": {
        "primary": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "secondary": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    },
    "state": {"joint_position": np.zeros((1, 1, 16), dtype=np.float32)},
    "language": {"annotation.human.action.task_description": [["pick up the cup"]]},
}
action, info = client.get_action(observations)              # no RTC, no memory
# RTC on:  options = {"action_prefix": prev[:, s:]}          # what the robot still holds
# memory:  options = {"session_ids": ["robot-0"], "reset_memory": [True]}   # episode start
# action["joint_position"]: (1, 16, 16) float32 — 16 chunk steps
```

`PolicyClient.get_action(obs, options=…)` is the idiomatic call — the
in-tree simulator client (`rollout_policy.py`) uses it. It wraps
`call_endpoint("get_action", …)` above; use the low-level form only when reaching a custom endpoint.

Observations are grouped by modality, `observation[modality][key]`, with
the keys of the embodiment's `ModalityConfig`. Video arrays are uint8
`(B, T, H, W, 3)`; state is float32 `(B, T, dim)`; language is one list of
strings per batch element. The exact set of keys depends on the
embodiment — consult the modality config you trained with. Simulator envs
emit flat `video.<key>` / `state.<key>` observations; `rollout_policy.py`
translates them with `rldx.eval.sim.rldx_format.parse`, which fixes the key
mapping for the env family once and rejects a mismatch before the first step.

Physics-trained checkpoints (`physics:on`) additionally require one
top-level group per physics key, exactly as `get_modality_config()`
advertises it — e.g. `observation["tactile"] = {"left": ..., "right": ...}`
with float32 `(B, T, dim)` arrays where `T = len(delta_indices)` from the
handshake. The group is validated against the handshake (sub-keys, horizon,
dtype, finiteness) and a missing group is an error unless the checkpoint
was trained with `--physics.allow-missing`. Such a checkpoint takes its
physics groups all or none: omitting every group runs inference without
physics, but sending only some is rejected — the model has one physics-present
bit per sample, so the groups that were sent could not be used either.
Observation keys outside the advertised contract are rejected — the
handshake is the whole contract, so the RLDX-1 v1.0.2 `observation["physics"]`
envelope with compound `"tactile.left"` keys fails as an unknown key, with an
error that names the group format to send instead.

### Options dict

| Key | Served by | Type | Purpose |
|---|---|---|---|
| `action_prefix` | RTC on | `(B, T, width)` float | The actions the robot still holds from its previous chunk, in robot units, from the current step on; `T ≥ get_rtc()["prefix_steps"]`. Omit for a cold batch. |
| `rtc_prefix_len` | RTC on | `int` or `list[int]` | Per sample, how many of the first `delay` actions are committed (default `delay`); `0` marks a sample without a previous chunk. |
| `session_ids` | memory | `list[str]` | Required: one stable id per robot or env, one per sample. |
| `reset_memory` | memory | `list[bool]` | Mark the samples that start an episode. |

### RTC variants

Two real-time-chunking flavors are implemented (see
`rldx/model/modules/action_model/rtc.py`); the CLI flags that select
between them are documented under
[Real-Time Chunking (RTC)](#real-time-chunking-rtc) above.

- **`guided`** ([Black et al. (Real-Time Execution of Action Chunking Flow Policies)](https://arxiv.org/abs/2506.07339)): no training change.
  At each denoising step the flow-matching velocity is augmented with a
  VJP guidance term that pulls the new chunk toward the still-overlapping
  positions of the previous one, hard-fixing the first `d` already-
  executed entries. Adds latency (one VJP per guided step).
- **`trained`** ([Black et al. (Training-Time Action Conditioning for Efficient Real-Time Chunking)](https://arxiv.org/abs/2512.05964)): a delay
  `d ~ U{0, …, rtc_training_max_delay}` was sampled at each training
  step and the model was conditioned on a clean prefix of length `d`,
  so inference reduces to the standard denoising loop with the prefix
  inpainted at `τ=1` — no guidance term and no extra forward passes.
  Requires a checkpoint trained with `--rtc-training-max-delay > 0`
  (see [`training.md`](training.md#real-time-chunking-training-time)).

If a checkpoint trained with `--rtc-training-max-delay > 0` is served
with `rtc:trained`, the trained path is faster. If you
need to serve any other flow-matching checkpoint with RTC, use
`rtc:guided`.

## Two canonical deployments

### Simulator eval

`run_rldx_server.py` + `rollout_policy.py` pair. See
[`evaluation.md`](evaluation.md) for the full recipe. Key points:

- The rollout worker runs in a separate simulator-specific venv and
  does not load the model.
- The rollout worker translates the simulator's flat observations and
  actions itself (`rldx/eval/sim/rldx_format.py`, one `SimFormat` per env
  family); the server only sees the RLDX format.
- Bind to `127.0.0.1` since both processes live on the same box.

### Real-robot (DROID, ALLEX, ...)

`run_scripts/deploy/droid_deploy.sh` runs DROID with the policy loaded
in-process (`load_policy`, see [Embedded vs server use](#embedded-vs-server-use)).
To keep the model on a separate GPU host instead, start the server there:

```bash
uv run python rldx/eval/run_rldx_server.py \
    --model-path /path/to/your/checkpoint \
    --embodiment-tag GENERAL_EMBODIMENT \
    --host 0.0.0.0 --port 5555
```

On the robot side, wire the DROID driver into a `PolicyClient`
(ZMQ REQ over TCP) and step at whatever rate the controller can
sustain. The server's action chunk is 16 steps at the training
timestep dt; the client decides how many to execute before asking for
the next chunk (with RTC on, the server's `--rtc.exec-horizon`).

Caveats for real-robot deployment:

- **Steps per chunk.** With RTC on, execute `get_rtc()["exec_horizon"]`
  actions of each chunk before asking again; with a memory checkpoint,
  `get_memory()["stride"]`. Either assumes that spacing.
- **Memory resets.** With a memory checkpoint, pass
  `options["reset_memory"] = [True]` on the first step of every
  episode, otherwise the memory from the previous episode bleeds into
  the new one.
- **Session ids.** Use a stable unique string per controller; if you
  regenerate it every query the memory starts cold every step.
- **Latency.** For measured per-call latency see
  [`optimization.md`](optimization.md). Chunks of 16 steps mean you
  can amortise that over 16 controller ticks.

## Embedded vs server use

If your client lives in the same Python process (so no wire protocol
is needed), you can skip `run_rldx_server.py` entirely and use
`load_policy` directly:

```python
import rldx
from rldx.serving.policy import load_policy

policy = load_policy(
    model_path="RLWRLD/RLDX-1.1-FT-ROBOCASA",
    embodiment_tag=rldx.EmbodimentTag.GENERAL_EMBODIMENT,
    device="cuda",
)
actions, _ = policy.get_action(observations)
```

The server and the embedded path go through the exact same
`get_action` code path, just with or without the ZMQ round-trip.

## Troubleshooting

### The client hangs on its first request

ZMQ connects lazily, so a client started before the server does not fail:
its first request waits, and `PolicyClient` sets no receive timeout, so it
waits indefinitely if the server never comes up. The server takes 30-60 s to
load the backbone on a warm cache, much longer on cold. Wait until it prints
`✓ Server ready` before connecting.

### `Unrecognized processing class`

Checkpoint processor files are not inside a `processor/` subdir.
Re-upload the checkpoint with the current layout — see the
`CheckpointFormatCallback` section of [`training.md`](training.md#checkpoint-format).

### Memory cache not resetting between episodes

Either `session_ids` is being regenerated every query (so every step
looks like a new session and starts cold), or `reset_memory` is never
set, so one episode's memory carries into the next.

### Action chunks look random at inference

Most likely the processor is running in `train` mode and applying
random crops. `RLDXPolicy` calls `processor.eval()` on load, so this
only happens if you are using `RLDXPolicy` directly and accidentally
called `processor.train()` later in your code.

## Where to next

- [`installation.md`](installation.md) — env setup
- [`evaluation.md`](evaluation.md) — simulator benchmark recipes
- [`training.md`](training.md) — produce a checkpoint to serve
- [`architecture.md`](architecture.md) — model internals
