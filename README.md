<div align="center">

# RLDX-1.1

[![arXiv](https://img.shields.io/badge/arXiv-2605.03269-b31b1b.svg)](https://arxiv.org/abs/2605.03269)
[![Project Page](https://img.shields.io/badge/Project-Page-1f72b1.svg)](https://rlwrld.ai/rldx-1)
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow.svg)](https://huggingface.co/collections/RLWRLD/rldx-11)

<img src="assets/rldx_overview.png" width="100%" alt="RLDX-1.1 overview">

</div>

RLDX-1.1 is a Vision-Language-Action model (VLA) for human-like
dexterous manipulation. Beyond the *versatile intelligence* inherited
from pre-trained VLM backbones, RLDX-1.1 adds three **functional
capabilities** — motion awareness, long-term memory, and physical
sensing — through a unified **Multi-Stream Action Transformer (MSAT)**
architecture, a synthetic-augmented training pipeline, and a real-time
inference stack.

Compared with RLDX-1, RLDX-1.1 scores higher on all four simulation
benchmarks, reduces action jitter on low-DoF robots by up to 57%, and trains
2.27× and infers up to 3.7× faster. See
[What's New in RLDX-1.1](#whats-new-in-rldx-11).

> Previous release: [RLDX-1 (`v1.0.2`)](https://github.com/RLWRLD/RLDX-1/tree/v1.0.2). All RLDX-1 checkpoints still load with this codebase.

## News

- **[2026.11.12]** RLDX-1.1 code and model checkpoints released! See [What's New in RLDX-1.1](#whats-new-in-rldx-11).
- **[2026.06.16]** RLDX-1-PT-**IMG** (lightweight image-input version) released!
- **[2026.05.06]** Technical report, blog post, codes and model checkpoints for RLDX-1 released!

## What's New in RLDX-1.1

### Better Performance

We developed a stronger pre-training recipe for RLDX-1.1 by carefully
refining the existing RLDX-1 corpus and incorporating a broad collection
of open-source robot datasets. This improved data mixture provides a
stronger foundation for downstream fine-tuning. To further improve the
video-input model, we enable the motion module during fine-tuning, which
was not used for RLDX-1 video-input checkpoints.

The table below compares RLDX-1.1 with RLDX-1 and recent VLAs across
four public benchmarks. RLDX-1.1 improves on RLDX-1 on all four benchmarks,
and an RLDX-1.1 model scores highest on each.

#### Comparison with Other VLAs

Success rates (%) of each model fine-tuned on each benchmark's training set.
Note that some scores were re-evaluated with our current evaluation setup.

| Method | RoboCasa Kitchen | GR-1 Tabletop | RoboCasa365 (Avg) | LIBERO (Avg) |
|---|---|---|---|---|
| π0.5 | 70.1 | 43.3 | 16.9 | 96.9 |
| GR00T N1.7 | 64.8 | 40.2 | 24.6 | 95.1 |
| RLDX-1 | 70.8 | 58.9 | 36.0 | 97.8 |
| **RLDX-1.1-IMG (ours)** | 70.4 | 60.6 | 33.2 | 97.2 |
| **RLDX-1.1 (ours)** | 72.3 | 60.3 | 40.2 | 97.9 |

Per-benchmark checkpoints, embodiment tags, and evaluation guides are listed
under [Reproducing Benchmark Results](#reproducing-benchmark-results).

### Smoother Actions in low DoF

For low-DoF embodiments, flow-matching noise in padded action dimensions
can affect joint commands from the action DiT and increase jitter.
RLDX-1.1 applies masking from pre-training to keep these unused dimensions
at zero during training and inference.

We evaluated single-frame image models on a 13-DoF real robot with 50 actions
per chunk. Values show mean direction changes in executed joint commands
(reversals/s; lower is better).

| Model | Masking | 4 denoising steps ↓ | 10 denoising steps ↓ |
|---|---|---:|---:|
| RLDX-1-IMG | Off | 9.69 | 10.38 |
| RLDX-1.1-IMG | On | 6.45 | 4.47 |

Compared with RLDX-1-IMG, RLDX-1.1-IMG reduced reversals by 33% at four
denoising steps and 57% at ten.

Each result averages metrics across 13 joints and five successful episodes
at 30 Hz. The direction-change count excludes velocities with an absolute
value of 0.06 rad/s or less.

This comparison includes other changes between model versions.

**Keep `mask_padded_action_dims=True` during fine-tuning and inference
with the resulting checkpoint.**

### Faster Training and Inference

RLDX-1.1 trains 2.27× faster than RLDX-1 on the same model and data, with
the same training math. Every training optimization is on by default.

Inference is up to 3.7× faster on the RTX 5090, each release at its
fastest setting on RoboCasa. It now runs batched and
quantized from edge to cloud: fp8 on every supported GPU, plus nvfp4 on
Jetson Thor, GB10 and the RTX 5090. RLDX-1 ran batch 1 in bf16, with optimized
inference on the RTX 5090 only.

| | Hardware | RLDX-1 | RLDX-1.1 |
|---|---|---|---|
| Training | H200 × 8 | 338 samples/s | 767 samples/s (2.27×) |
| Inference | RTX 5090 | 75 ms | 20 ms (3.7×) |
| Inference | Jetson Thor | not supported | 86 ms |

See [Optimization](docs/optimization.md) for conditions, every preset on every
device, and how to choose one.

<details>
<summary>Upgrading from RLDX-1</summary>

**Checkpoints.** Every RLDX-1 checkpoint loads with this codebase and keeps
its outputs: a `config.json` without `mask_padded_action_dims` loads
unmasked, and its 36 embodiment slots are kept. Checkpoints saved by this
codebase use `config.json` format v2, which RLDX-1 cannot read.

**Environment.** Pinned dependencies changed (`tyro`, `deepspeed`, `wandb`
and others), so run `uv sync` again. `gymnasium` moved to the `eval` extra.

**Flags and APIs.** Optional features and the data source are now
subcommands (`memory:on`, `data:paths`, ...). A flag that does not apply to
the chosen subcommands is rejected instead of silently ignored. Flags not
listed below, such as `--learning-rate` and `--n-cog-tokens`, are unchanged;
see the [training flags](docs/training.md#training-knobs-cheat-sheet).

| RLDX-1 | RLDX-1.1 |
|---|---|
| `--dataset-path(s) … --embodiment-tag …` | `data:paths --data.paths … --data.embodiment-tag …` |
| `--dataset-mix-ratios …` | `--data.mix-ratios …` |
| `--pt-dataset-root … --pt-dataset-mix …` | `data:mix --data.root … --data.name …` |
| `--use-memory --memory-length 4 --concat-memory` | `memory:on --memory.length 4 --memory.concat` |
| `--blockwise-attn-for-memory` | `--memory.blockwise-attn` |
| `--use-motion --motion-insert-layer 9` | `motion:on --motion.insert-layer 9` |
| `--use-physics --physics-keys … --physics-dims …` | `physics:on --physics.keys … --physics.dims …` |
| `--action-model-use-lora --action-model-lora-rank 16` | `action-model-lora:on --action-model-lora.rank 16` |
| `--backbone-use-lora --backbone-lora-rank 16` | `backbone-lora:on --backbone-lora.rank 16` |
| `--use-wandb --wandb-project …` | `wandb:on --wandb.project …` |
| `--dataset-mode …` | Removed; data is always sharded |
| Training `--rtc-inference-*` | Removed; RTC is chosen when serving |
| Server `--compile …` | `--runtime-profile …` (default `compile`), see [runtime profiles](docs/inference_server.md#inference-runtime---runtime-profile) |
| Server `--rtc-inference-mode … --rtc-inference-delay …` | `rtc:trained` or `rtc:guided` with `--rtc.delay … --rtc.exec-horizon …` |
| Server `--dataset-path … --modality-config-path …` (replay) | `rldx/eval/run_replay_server.py --dataset-path … --execution-horizon … --modality-config-path …` |
| Server `--strict`, `--use-sim-policy-wrapper`, `--verbose` | Removed |
| `rollout_policy.py --model-path …` | Removed; start `run_rldx_server.py` and pass `--policy-client-host … --policy-client-port …` |
| `rldx.policy.rldx_policy.RLDXPolicy(...)` | `rldx.serving.policy.load_policy(...)`; `get_action` returns `(actions, info)` |

Validation during training is new: pass `--validation a.json b.json`
(one spec per file); see the [training flags](docs/training.md#training-knobs-cheat-sheet).

**Behavior.**

- Run directories created by RLDX-1 cannot be resumed. Fine-tuning from a
  checkpoint into a new `--output-dir` is unaffected.
- The policy server binds to `0.0.0.0` by default (RLDX-1: `127.0.0.1`), so
  it listens on every interface; pass `--host 127.0.0.1` for loopback only.
- With RTC, the client sends the part of the previous chunk the robot has not
  executed yet as `action_prefix`. Memory checkpoints need `session_ids`. See
  the [client protocol](docs/inference_server.md#client-protocol-policyserver--policyclient).

</details>

## Model Architecture

The architecture is unchanged from RLDX-1.

<div align="center">
<img src="assets/overview_architecture.png" width="90%" alt="RLDX-1.1 architecture">
</div>

- **Multi-Stream Action Transformer (MSAT).** Cognition, physics, and
  action each get a dedicated stream coupled by joint self-attention —
  an extension of MM-DiT to action modeling.
- **Motion awareness.** Multi-frame observations + a motion module
  capture temporal dynamics; intermediate VLM layers compress video
  tokens to keep the policy efficient.
- **Long-term memory.** A memory module fuses past cognition features
  with the current ones for history-grounded decisions beyond a short
  multi-frame window.
- **Physical sensing.** Tactile and torque enter as a dedicated physics
  stream; the decoder is jointly trained to predict future physical
  signals.
- **Three-stage training.** Pre-training (generalization) → mid-training
  (functionality) → post-training (task adaptation), with synthetic data
  augmenting rare manipulation scenarios.
- **Real-time inference.** Inference runtime profiles combine regional
  `torch.compile`, one CUDA graph per call shape, and FP8 / NVFP4
  quantization.

A five-stage walkthrough with config flags is in
[`docs/architecture.md`](docs/architecture.md).

## Model Checkpoints

**Video input**

| Checkpoint | Type | Description |
|---|---|---|
| [`RLWRLD/RLDX-1.1-PT`](https://huggingface.co/RLWRLD/RLDX-1.1-PT) | Base | Pre-trained |
| [`RLWRLD/RLDX-1.1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-ROBOCASA) | Fine-tuned | RoboCasa Kitchen |
| [`RLWRLD/RLDX-1.1-FT-GR1`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-GR1) | Fine-tuned | GR-1 Tabletop |
| [`RLWRLD/RLDX-1.1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-LIBERO) | Fine-tuned | LIBERO |
| [`RLWRLD/RLDX-1.1-FT-RC365`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-RC365) | Fine-tuned | RoboCasa365 |

**Image input**

| Checkpoint | Type | Description |
|---|---|---|
| [`RLWRLD/RLDX-1.1-PT-IMG`](https://huggingface.co/RLWRLD/RLDX-1.1-PT-IMG) | Base | Pre-trained |
| [`RLWRLD/RLDX-1.1-FT-ROBOCASA-IMG`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-ROBOCASA-IMG) | Fine-tuned | RoboCasa Kitchen |
| [`RLWRLD/RLDX-1.1-FT-GR1-IMG`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-GR1-IMG) | Fine-tuned | GR-1 Tabletop |
| [`RLWRLD/RLDX-1.1-FT-LIBERO-IMG`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-LIBERO-IMG) | Fine-tuned | LIBERO |
| [`RLWRLD/RLDX-1.1-FT-RC365-IMG`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-RC365-IMG) | Fine-tuned | RoboCasa365 |

All RLDX-1.1 checkpoints have 6.6B parameters.

RLDX-1 checkpoints (PT, PT-IMG, MT-\*, FT-\*) are listed in the
[RLDX-1 README](https://github.com/RLWRLD/RLDX-1/tree/v1.0.2) and all load
with this codebase.

## Quick Start

**Requirements**: Python 3.10, CUDA 12.x, [uv](https://github.com/astral-sh/uv) v0.8.4+

```bash
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
uv sync --python 3.10
```

Verify installation:
```bash
uv run python -c "import rldx; print(rldx.__version__)"
```

For simulator setup, dev tooling, and full troubleshooting, see
[`docs/installation.md`](docs/installation.md).

Load a checkpoint and predict an action chunk in-process:

```python
from rldx.serving.policy import load_policy
from rldx.data.embodiment_tags import EmbodimentTag

policy = load_policy(
    model_path="RLWRLD/RLDX-1.1-FT-ROBOCASA",
    embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
    device="cuda:0",
)

# observation: {"video": {...}, "state": {...}, "language": {...}}
actions, info = policy.get_action(observation)
```

[`getting_started/01_quickstart.ipynb`](getting_started/01_quickstart.ipynb)
builds a dummy observation for this checkpoint and runs the call end to end.

## Inference and Evaluation

### Policy Server

`rldx/eval/run_rldx_server.py` serves the same policy over ZeroMQ for
simulator rollouts and real-robot deployment:

```bash
uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --host 0.0.0.0 --port 20000
```

Two orthogonal knobs control real-time behavior:

- **`--runtime-profile`** — a preset per recipe: `compile` (bf16, default), `eager` (bf16, no compile), `any:bf16` or `any:fp8` on any device, and `<device>:nvfp4-mix` (text fp8 except `gate_up_proj` in nvfp4) or `<device>:nvfp4` for `thor` and `gb10`. Presets other than `compile` and `eager` turn on forward CUDA graphs and need `limits:on --limits.batch B --limits.instruction-tokens T`, which bounds requests.
- **`rtc:guided` / `rtc:trained`** — Real-Time Chunking for chunk-boundary stitching. Off when omitted.
  - `guided` — works with any flow-matching checkpoint.
  - `trained` — requires a checkpoint trained with `--rtc-training-max-delay > 0`.
  - Implementation follows [Black et al. (Real-Time Execution of Action Chunking Flow Policies)](https://arxiv.org/abs/2506.07339). The `trained` mode uses the integration from [Black et al. (Training-Time Action Conditioning for Efficient Real-Time Chunking)](https://arxiv.org/abs/2512.05964).

The client protocol, full flag list, and a walkthrough of the trade-offs
are in [`docs/inference_server.md`](docs/inference_server.md#real-time-chunking-rtc).

### Reproducing Benchmark Results

Each benchmark has a self-contained eval README; this table maps each
column in [Comparison with Other VLAs](#comparison-with-other-vlas) to
the fine-tuned checkpoint, the embodiment tag the server expects, and the
runnable guide.

| Benchmark | Fine-tuned Checkpoint | Embodiment Tag | Eval Guide |
|---|---|---|---|
| LIBERO | [RLWRLD/RLDX-1.1-FT-LIBERO](https://huggingface.co/RLWRLD/RLDX-1.1-FT-LIBERO) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/libero/README.md`](run_scripts/eval/libero/README.md) |
| RoboCasa Kitchen (24 tasks) | [RLWRLD/RLDX-1.1-FT-ROBOCASA](https://huggingface.co/RLWRLD/RLDX-1.1-FT-ROBOCASA) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/robocasa_kitchen/README.md`](run_scripts/eval/robocasa_kitchen/README.md) |
| GR-1 Tabletop | [RLWRLD/RLDX-1.1-FT-GR1](https://huggingface.co/RLWRLD/RLDX-1.1-FT-GR1) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/gr1_tabletop/README.md`](run_scripts/eval/gr1_tabletop/README.md) |
| RoboCasa365 | [RLWRLD/RLDX-1.1-FT-RC365](https://huggingface.co/RLWRLD/RLDX-1.1-FT-RC365) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/robocasa_365/README.md`](run_scripts/eval/robocasa_365/README.md) |

Shared mechanics (server + rollout split, common flags, troubleshooting)
are documented in [`docs/evaluation.md`](docs/evaluation.md).

## Fine-tuning

This section covers how to fine-tune RLDX-1.1 from a pre-trained checkpoint
(`RLWRLD/RLDX-1.1-PT`) on your own LeRobot v2.1 dataset. The training entry
point is a single CLI (`rldx/experiment/launch_train.py`) where flags
toggle the optional functional capabilities described in
[Model Architecture](#model-architecture):

- `--video-length N` — temporal frames per observation (motion awareness)
- `memory:on` — temporal memory module (long-term memory)
- `motion:on` — motion module inside the VLM backbone
- `physics:on --physics.keys ...` — tactile / torque streams (physical sensing)

LoRA, training-time RTC, and the full flag list are documented in
[`docs/training.md`](docs/training.md). Below are the canonical recipes.

### Data Preparation

RLDX-1.1 uses [LeRobot](https://github.com/huggingface/lerobot) v2.1 format datasets.
[`run_scripts/data/`](run_scripts/data/) has a converter from HDF5 teleoperation recordings
(`convert_h5_to_v2.py`) and wrappers for one dataset or a directory of datasets; the scripts
define the expected input layout and arguments. The converter needs LeRobot 0.3.3, the last
release that writes v2.1 (`uv pip install --no-deps lerobot==0.3.3 jsonlines h5py`).

Each dataset must carry a `meta/modality.json` that slices the flat
state / action vectors into named joint groups and remaps video columns
to modality keys — see
[`docs/training.md`](docs/training.md#dataset-layout-metamodalityjson).
For a custom robot, pass your modality config with
`--modality-config-path` together with an `EmbodimentTag` that selects the
per-robot MLP head slot (default: `GENERAL_EMBODIMENT`; see
[`docs/embodiment_tags.md`](docs/embodiment_tags.md)). The `EmbodimentTag`
design follows [NVIDIA GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release).

### Basic Recipe

```bash
uv run torchrun --nproc_per_node=<N_GPUS> rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1.1-PT \
    data:paths --data.paths /path/to/your/dataset \
    --data.embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path /path/to/your_modality_config.py \
    --video-length 4 \
    --n-cog-tokens 64 \
    --num-gpus <N_GPUS> \
    --global-batch-size 64 \
    --learning-rate 1e-4 \
    --max-steps 60000 \
    --save-steps 5000 \
    --output-dir ./outputs/my_finetune
```

`--modality-config-path` is required: it registers the modality config for
the embodiment tag (see [Data Preparation](#data-preparation)).

### With Memory, Motion, and Physics

Recommended for embodiments where memory, motion awareness, or contact
sensing matter. To enable a *single* add-on instead of all three, keep
just the corresponding `memory:on` / `motion:on` / `physics:on` group(s) and drop the rest.

```bash
uv run torchrun --nproc_per_node=<N_GPUS> rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1.1-PT \
    data:paths --data.paths /path/to/your/dataset \
    --data.embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path /path/to/your_modality_config.py \
    --video-length 4 \
    memory:on --memory.length 4 --memory.concat \
    motion:on --motion.insert-layer 9 \
    physics:on --physics.keys tactile torque --physics.dims 30 7 \
    --new-param-warmup-steps 2000 \
    --n-cog-tokens 64 \
    --num-gpus <N_GPUS> \
    --global-batch-size 64 \
    --max-steps 60000 \
    --output-dir ./outputs/my_finetune_all
```

### Key Training Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--video-length` | Number of video frames (video token compression is always on; set to `1` for single-frame) | `4` |
| `--video-stride` | Stride between frames in action-step units | `2` |
| `memory:on` | Enable temporal memory module | off |
| `--memory.length` | Memory context window (timesteps) | `4` |
| `motion:on` | Enable motion module | off |
| `physics:on` | Enable physics signal conditioning | off |
| `--n-cog-tokens` | Number of cognition tokens | `64` |
| `--global-batch-size` | Total batch size across GPUs per step, split evenly over `--num-gpus` | `64` |
| `--new-param-warmup-steps` | Warmup steps for newly added modules | `0` |

### LoRA

For memory-constrained fine-tunes you can replace full-parameter tuning
of the action model (MSAT) and/or the backbone VLM with PEFT
LoRA adapters:

```bash
action-model-lora:on --action-model-lora.rank 16 --action-model-lora.alpha 32
backbone-lora:on --backbone-lora.rank 16 --backbone-lora.alpha 32 --backbone-lora.num-layers -1
```

Only the adapters train in the part a LoRA covers, so `action-model-lora:on`
does not take `--tune-diffusion-model`, and `backbone-lora:on` does not take
`--tune-llm`, `--tune-visual`, `--tune-top-llm-layers` or `--freeze-cog-tokens`;
giving one is an error. Full flag list
and target-module defaults are in
[`docs/training.md`](docs/training.md#lora-fine-tuning).

### Training for Real-Time Chunking

If you intend to serve the checkpoint with `rtc:trained`
(faster, fullgraph-compatible), enable training-time RTC at training
time:

```bash
--rtc-training-max-delay 4
```

The training-time RTC formulation follows
[Black et al. (Training-Time Action Conditioning for Efficient Real-Time Chunking)](https://arxiv.org/abs/2512.05964); the inference-side
counterpart is
[Black et al. (Real-Time Execution of Action Chunking Flow Policies)](https://arxiv.org/abs/2506.07339). See
[`docs/training.md`](docs/training.md#real-time-chunking-training-time)
and [`docs/inference_server.md`](docs/inference_server.md#real-time-chunking-rtc)
for usage details.

## Documentation

Hands-on guides live under [`docs/`](docs/):

| Guide | What it covers |
|---|---|
| [`installation.md`](docs/installation.md) | Environment setup, simulator venvs, dev tooling, common pitfalls |
| [`architecture.md`](docs/architecture.md) | Five-stage walkthrough of the RLDX-1.1 model and its config flags |
| [`training.md`](docs/training.md) | `launch_train.py` recipes (fine-tune / mid-train), LoRA, training-time RTC, dataset layout |
| [`embodiment_tags.md`](docs/embodiment_tags.md) | What `EmbodimentTag` is and how to pick one for a custom robot |
| [`evaluation.md`](docs/evaluation.md) | RoboCasa / LIBERO / GR-1 eval, server + rollout split, results aggregation |
| [`inference_server.md`](docs/inference_server.md) | `run_rldx_server.py` CLI, wire protocol, RTC modes, runtime profiles, simulator + real-robot deployment |
| [`optimization.md`](docs/optimization.md) | Training speedups and what they change for your runs, inference execution path, fp8 / nvfp4 quantization, choosing a runtime profile |

Step-by-step notebooks are in [`getting_started/`](getting_started/README.md).

## Acknowledgments

RLDX-1.1 builds upon the following open-source projects:

- [NVIDIA GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release) — Training Codebase
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Vision-language backbone
- [FLUX](https://github.com/black-forest-labs/flux) — MMDiT architecture

## Contributing

We currently do not accept external pull requests on this repository.
If you encounter a bug, broken reproduction step, or have a question
about RLDX-1.1, please **open an issue** at
[github.com/RLWRLD/RLDX-1/issues](https://github.com/RLWRLD/RLDX-1/issues)
and we will follow up there.

## License

- **Code**: released under the [Apache License 2.0](LICENSE.md). The codebase
  is built on the [NVIDIA Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release)
  framework. Third-party code included in this repository is listed, with its
  copyright notices and license texts, in [NOTICE](NOTICE).
- **Model weights**: distributed on Hugging Face under the
  [RLWRLD Model License v2.0](https://huggingface.co/RLWRLD/RLDX-1.1-PT/blob/main/LICENSE.md)
  (a non-commercial license with attribution and share-alike terms). By using
  any `RLWRLD/RLDX-1.1-*` checkpoint you agree to those terms. `RLWRLD/RLDX-1-*`
  checkpoints remain under the
  [RLWRLD Model License v1.0](https://huggingface.co/RLWRLD/RLDX-1-PT/blob/main/LICENSE.md).

## Citation

```bibtex
@article{rldx2026,
  title={RLDX-1 Technical Report},
  author={Dongyoung Kim and Huiwon Jang and Myungkyu Koo and Suhyeok Jang and Taeyoung Kim and others},
  year={2026},
  journal={arXiv preprint arXiv:2605.03269},
  eprint={2605.03269},
  archivePrefix={arXiv}
}
```
