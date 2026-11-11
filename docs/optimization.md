# Optimization

RLDX-1.1 trains 2.27× faster than RLDX-1 on the same
recipe, and serves faster through compiled, CUDA-graph inference and fp8 /
nvfp4 quantization.

This doc covers what the training and inference paths do for you, what each
change means for your runs, and how to pick an inference runtime profile.

## At a glance

| | RLDX-1 | RLDX-1.1 |
|---|---|---|
| Training step, GR-1 (H200 × 8) | 3.03 s | **1.34 s (2.27×)** |
| Inference, RTX 5090, default | 111.3 ms | 42.8 ms (2.6×) |
| Inference, RTX 5090, fastest without quantization | 75.1 ms | 42.4 ms (1.8×) |
| Inference, RTX 5090, fastest | 75.1 ms | **20.2 ms (3.7×)** |

Training fine-tunes `RLDX-1-PT` at per-GPU batch 64 × accumulation 2
([Results](#results)).[^training] Inference serves `RLWRLD/RLDX-1-FT-ROBOCASA`
on RoboCasa observations (3 cameras × 4 frames × 256²) at batch 1; a cell is
the median of one `get_action` call.[^inference] "Fastest without quantization"
is `any:bf16`; "fastest" is `gb10:nvfp4`, whose presets apply to the RTX 5090.

RLDX-1 runs uncompiled by default; its fastest setting is `fullgraph`.
Details are in [Compared with RLDX-1](#compared-with-rldx-1).

## Training

There is almost nothing to configure: every change below is on by default.
They keep the training math — on the same data and seed, loss and weight
updates match the previous code within seed-to-seed noise.

| Change | What it means for you |
|---|---|
| [Host-side input planning](#host-side-input-planning) | CUDA training needs `pin_memory=True`; it fails at startup otherwise |
| [Asynchronous host-to-device copy](#asynchronous-host-to-device-copy) | Nothing |
| [Compilation](#compilation) | `--execution-backend compile` (default); `eager` for debugging |
| [Tensor Core shapes](#tensor-core-shapes) | Nothing; checkpoints keep their shapes |
| [Selective video decoding](#selective-video-decoding) | A frame that fails to decode now fails the run instead of being skipped |
| [Micro-batch routing](#micro-batch-routing) | On by itself with more than one GPU, gradient accumulation and ZeRO-2 |
| [Memory allocator](#memory-allocator-jemalloc) | Install `libjemalloc2`, or set `JEMALLOC_PATH` |
| [Garbage collection](#garbage-collection) | Nothing |

### Host-side input planning

Every value the forward pass needs to decide shapes — vision grid sizes,
mRoPE position ids, which tokens video token compression keeps — is computed
on the CPU in the DataLoader workers (`rldx/planner/`) and handed to the
model as a plan. The forward pass then runs only device kernels: it never
calls `.item()` or `.tolist()` on a CUDA tensor, so the GPU never waits for
the host mid-step.

Inputs the planner does not support (4D attention masks, incremental decode)
raise instead of falling back to a slow path.

### Asynchronous host-to-device copy

The collator builds each batch in its final dtype in pinned memory, and the
training runtime copies the next batch on a separate CUDA stream while the
current one computes (`rldx/transport/`). The whole batch reaches the main
process as one shared-memory buffer, so receiving it costs one file
descriptor instead of one per tensor.

### Compilation

`--execution-backend compile`, the default, compiles three regions with
`torch.compile`:

| Region | Boundary |
|---|---|
| Text RMSNorm, RoPE | Function |
| Vision encoder | One vision block |
| Action model | The whole MSAT forward |

Shapes entering a region are kept stable so a new batch does not trigger a
recompile: MSAT inputs are made contiguous at the boundary, and the vision
encoder receives a fixed sequence-length bucket instead of each batch's
exact maximum.

`--execution-backend eager` runs the same model without compiled kernels —
useful when a stack trace inside a compiled region is hard to read.

### Tensor Core shapes

The action model's SwiGLU hidden width, 10,922, is not a multiple of 8, so
the GPU picked a slow fallback GEMM for it. RLDX-1.1 pads it with zeros to
10,928 at load. The padding stays zero through training and is stripped on
save, so checkpoints keep the 10,922 shape and load in either release.

### Selective video decoding

The loader used to decode every frame of an episode and keep the few a
sample uses. It now decodes only the union of frames the shard's samples
need, one FFmpeg thread per decoder (`num_ffmpeg_threads=1`), which keeps CPU
use predictable when many workers decode at once. Which samples are drawn,
and in what order, is unchanged.

### Micro-batch routing

With gradient accumulation, every rank waits at each micro-step for the
slowest rank. When training runs on more than one GPU with
`--gradient-accumulation-steps` above 1 and ZeRO-2, the runtime estimates each
micro-batch's cost from its token, image and action counts, then reorders
which rank runs which micro-batch so expensive ones share a micro-step.
Micro-batches move only between GPUs on the same node, whole, and the
optimizer step still sees exactly the same batches.

RLDX-1.1 also moves to DeepSpeed 0.19.6. Versions 0.17.6 – 0.18.2 applied only
the last micro-batch's gradient under ZeRO-2 with gradient accumulation
([deepspeed#7718](https://github.com/deepspeedai/DeepSpeed/issues/7718)), so
`grad_norm` now reads higher than in RLDX-1 runs. Do not compare the two.

### Memory allocator (jemalloc)

With glibc malloc, DataLoader workers keep freed shard and frame buffers in
their arenas, so host memory grows step after step until the job is
OOM-killed. The training scripts preload jemalloc
(`/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`, or `JEMALLOC_PATH`), which keeps
host memory from growing without changing step time. Without
the library, training still runs and prints a warning; see
[`training.md`](training.md#-jemalloc-not-found-).

### Garbage collection

After the first optimizer step, each training process freezes the objects it
built during setup (`gc.freeze()`, `rldx/runtime/heap.py`), so Python's
full collection no longer walks them. Before this, a collection paused one
rank for over a second while every other rank waited for it at the next
all-reduce. DataLoader workers unfreeze and collect as usual.

### Results

| Code (model `RLDX-1-PT`) | Step period (s) | Samples / s | Tensor Core active | GPU idle | Peak GPU memory (GiB) |
|---|---|---|---|---|---|
| RLDX-1 release (v1.0.2) | 3.03 | 338 | 22 % | 36 % | 79.7 |
| RLDX-1.1 release | 1.34 (2.27× faster) | 767 | 55 % | 5 % | 66.7 |

Same model and data in both rows, so the difference is the code.[^training]

## Inference

Inference is fast in two layers: an execution path every runtime profile
shares, and the quantization a profile selects.

### Execution path

| Change | What it means for you |
|---|---|
| Compiled denoising step | One Euler step is compiled once and reused for every step |
| Forward CUDA graph | On for every profile except `compile` and `eager` when serving limits are set |
| Text bucket | Instruction length changes do not recompile |
| GPU image preprocessing | Raw `uint8` frames go to the GPU; crop, normalize and patchify run there |
| Fused projections | Q/K/V and gate/up GEMMs are merged at load |
| No autocast | Weights and inputs are bf16 end to end |

**Forward CUDA graph.** With `limits:on --limits.batch B --limits.instruction-tokens T`,
the server captures one CUDA graph per batch size up to `B` and per
instruction width up to `T` at load, then replays it for each request. A
call replays in one launch instead of thousands. On H100, this alone takes
batch-1 latency from 39.4 ms (`compile`) to 22.2 ms (`any:bf16`).

**Text bucket.** The prompt is left-padded to a multiple of 16 tokens, so the
set of shapes within the limits is small and fixed. Requests beyond the
limits are rejected.

**GPU image preprocessing.** The host only resizes to the target area and
sends `uint8`, 8× smaller than float32 pixels.

### Quantization

A profile quantizes four groups of linear layers: `text` (language model),
`vision` (vision encoder), `msat_vl` and `msat_sa` (the action model's
vision-language and state-action streams).

| Recipe | `text` | `vision` | `msat_vl` | `msat_sa` | Devices |
|---|---|---|---|---|---|
| `fp8` | fp8 | fp8 | fp8 | fp8 | Any |
| `nvfp4-mix` | fp8, `gate_up_proj` nvfp4 | nvfp4 | nvfp4 | fp8 | GB10, Thor |
| `nvfp4` | nvfp4 | nvfp4 (GB10), fp8 (Thor) | nvfp4 | fp8 | GB10, Thor |

- **fp8** quantizes weights and activations to 8-bit floats (W8A8) with a
  per-row activation scale.
- **nvfp4** quantizes weights and activations to 4-bit floats (W4A4). Layers
  whose input range is known use a fixed scale; the rest compute it per
  call. No calibration data is needed.
- **Why `nvfp4-mix`.** The success-rate loss of `nvfp4` comes from 4-bit
  text activations; the vision encoder loses nothing at nvfp4. `nvfp4-mix`
  keeps the text model at fp8 except `gate_up_proj`, its largest GEMM, which
  buys back most of the speed.
- nvfp4 needs a GPU with FP4 Tensor Cores. On GB10 the text GEMMs use the
  b12x kernels, on Thor CUTLASS; H100 has no nvfp4 profile.

### Latency and success rate

| Preset | GB10 (ms) | Thor (ms) | H100 (ms) | RTX 5090 (ms) | Success rate (Thor) |
|---|---|---|---|---|---|
| `compile` (default) | 167.7 | 189.2 | 39.4 | 43.4 | 73.1 % |
| `eager` | 235.3 | 239.1 | 116.4 | 58.9 | — |
| `any:bf16` | 159.9 | 155.9 | 22.2 | 43.0 | — |
| `any:fp8` | 106.1 | 96.4 | 21.7 | 31.7 | 71.4 % |
| `<device>:nvfp4-mix` | 88.8 | 86.1 | — | 22.8¹ | 71.5 % |
| `<device>:nvfp4` | 85.7 | 87.2 | — | 20.8¹ | 70.5 % |

¹ The `gb10:` presets apply to the RTX 5090.


#### Batch size

Latency per call (ms) at batch 1 / 2 / 3, same conditions:

| Preset | GB10 | Thor | H100 | RTX 5090 |
|---|---|---|---|---|
| `compile` | 167.7 / 274.5 / 374.6 | 189.2 / 255.1 / 392.9 | 39.4 / 42.0 / 50.1 | 43.4 / 74.5 / 103.1 |
| `eager` | 235.3 / 414.0 / 588.1 | 239.1 / 407.3 / 616.7 | 116.4 / 114.2 / 115.2 | 58.9 / 94.5 / 131.8 |
| `any:bf16` | 159.9 / 269.1 / 374.9 | 155.9 / 236.8 / 379.4 | 22.2 / 34.7 / 48.8 | 43.0 / 76.2 / 103.2 |
| `any:fp8` | 106.1 / 176.6 / 258.9 | 96.4 / 160.0 / 249.4 | 21.7 / 33.3 / 44.6 | 31.7 / 49.4 / 66.6 |
| `<device>:nvfp4-mix` | 88.8 / 153.6 / 228.5 | 86.1 / 148.2 / 227.9 | — | 22.8 / 35.9 / 47.4¹ |
| `<device>:nvfp4` | 85.7 / 146.9 / 217.8 | 87.2 / 147.2 / 213.4 | — | 20.8 / 31.8 / 41.0¹ |

#### Compared with RLDX-1

With the checkpoint fixed to
[`RLWRLD/RLDX-1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA),
batch 1, all rows of a device in one container:

| Code path | H100 (ms) | RTX 5090 (ms) |
|---|---|---|
| RLDX-1 (v1.0.2), no compile | 142.5 – 163.9² | 111.3 |
| RLDX-1 (v1.0.2), `fullgraph` compile | — | 75.1 |
| RLDX-1.1 `compile` | 32.1 | 42.8 |
| RLDX-1.1 `any:bf16` | 21.7 | 42.4 |
| RLDX-1.1 `any:fp8` | 20.1 | 31.1 |
| RLDX-1.1 `gb10:nvfp4` | — | 20.2¹ |

² RLDX-1 inference is host-bound, so it moves with the host CPU; this is the
range over the first and last run of one container. RLDX-1 serves batch 1
only and does not install on GB10 or Thor.

### Serving with optimization and quantization

```
  dynamic shapes            fixed shapes (limits:on), CUDA graph
  ──────────────            ────────────────────────────────────────────────────
  eager ─▶ compile    ─▶    any:bf16 ─▶ any:fp8 ─▶ <device>:nvfp4-mix ─▶ <device>:nvfp4
                            ◀────────── faster ─────────────────────────────────▶
  —        73.1 %           —           71.4 %     71.5 %                70.5 %
```

The bottom row is the success rate on Thor from [the table above](#latency-and-success-rate).

1. Start with `compile` while integrating; it accepts any batch size and
   instruction length:

   ```bash
   pixi run -e quantize python rldx/eval/run_rldx_server.py \
       --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
       --embodiment-tag GENERAL_EMBODIMENT
   ```

2. When the serving shapes are known, bound them with `limits:on` and move
   to `any:bf16`, then `any:fp8`.
3. On GB10 or Thor, use `<device>:nvfp4-mix`. Pick `<device>:nvfp4` only if
   its extra speed is worth a lower success rate:

   ```bash
   pixi run -e quantize python rldx/eval/run_rldx_server.py \
       --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
       --embodiment-tag GENERAL_EMBODIMENT \
       --runtime-profile gb10:nvfp4-mix \
       limits:on --limits.batch 1 --limits.instruction-tokens 64
   ```

- Every preset other than `compile` and `eager` needs `limits:on`; without
  it the server refuses to load.
- `--limits.batch` is the largest request batch, `--limits.instruction-tokens`
  the longest instruction in tokens. Larger requests are rejected.
- The server compiles and captures every shape within the limits at load,
  so loading takes minutes; the first request is then as fast as the rest.
- The `quantize` environment is in
  [`installation.md`](installation.md#quantized-inference-quantize).

### How these were measured

- Checkpoint: [`RLWRLD/RLDX-1.1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-ROBOCASA)
  (RLDX-1.1, RoboCasa fine-tune), 3 cameras at 256² × 4 frames.
- Latency: median wall time of one `get_action` call, RTC off; batch 1
  unless the table says otherwise. Compared with RLDX-1 uses
  [`RLWRLD/RLDX-1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) instead.
- Devices: GB10 and Thor on our nodes, H100 on Modal, RTX 5090 on a
  workstation.
- Success rate: RoboCasa kitchen on Thor, 24 tasks × 50 episodes, seed 42,
  success within 720 steps (1000 for `OpenDoubleDoor`). Runs of the same
  setting differ by up to 0.3 %p, so each cell is the mean of several runs.

Full conditions are in the footnote.[^inference]

## Where to next

- [`training.md`](training.md) — launch training and the flags above
- [`inference_server.md`](inference_server.md) — `--runtime-profile` and serving limits on the server CLI
- [`architecture.md`](architecture.md) — what the compiled regions belong to

[^training]: Checkpoint `RLWRLD/RLDX-1-PT` for both releases, GR-1 recipe
    (`gr1_tabletop_1000demo`, 4 frames). Data: the official GR-1 datasets of
    [`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim),
    of which only the first 12 episodes of each of the mix's 24 datasets were
    used, since the run measures throughput only.
    Modal H200 × 8, per-device batch 64 × gradient accumulation 2 (1024),
    12 DataLoader workers per rank. Median over 8 ranks of the
    optimizer-step period, steps 12 to 50; RLDX-1 ran before and after
    RLDX-1.1 in the same container. Tensor Core active is the nsys
    GPU-metrics mean over steps 16 to 27; GPU idle is the share of the step
    with no kernel running on the GPU (nsys trace, steps 16 to 27).

[^inference]: Checkpoints `RLWRLD/RLDX-1.1-FT-ROBOCASA`, and
    `RLWRLD/RLDX-1-FT-ROBOCASA` for Compared with RLDX-1. Input: observations
    recorded from RoboCasa simulation episodes, not synthetic, 3 cameras ×
    4 frames × 256². Latency is one
    `get_action` call from observation to action, median of 2 × 100 calls after
    10 warm-up calls; graph presets with `--limits.batch 3
    --limits.instruction-tokens 64`, RTC off. All rows of a device in one
    container. GB10 and Thor on our nodes, H100 on Modal, RTX 5090 on a
    workstation. Success rate: served on Thor, 24 tasks × 5 scene layouts × 10
    episodes, seeds 42 to 46, at most 720 steps (1,000 for `OpenDoubleDoor`),
    mean of 2 runs.
