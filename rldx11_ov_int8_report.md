# RLDX-1.1 on OpenVINO (Intel Arc Pro B70): INT8 report

Date: 2026-11-18. Scope: `RLWRLD/RLDX-1.1-FT-LIBERO-IMG`, OpenVINO export from the upstream model
(`RLDX-1/run_scripts/export/rldx_openvino.py`), served on the Arc Pro B70 (`GPU.0`).

## TL;DR

| | OV f16 (B70) | **OV INT8 (B70)** |
|---|---|---|
| `get_action` median, batch 1 | 54.7 ms | **38.5 ms (-30 %)** |
| Open-loop parity vs torch fp32 (max \|diff\| normalized) | 2.6e-2 | 1.07e-1 |
| LIBERO closed loop (40 tasks × 10 ep) | 97.5 % (390/400) | **97.25 % (389/400)** |
| IR size | 12 GB | 5.7 GB |

INT8 is 30 % faster and loses no measurable closed-loop accuracy: 389/400 vs 390/400 for f16 on the same
protocol, a single episode, well inside run-to-run noise. Both are close to the 97.9 % upstream reports for
`RLDX-1.1-FT-LIBERO` (a different checkpoint and episode count). The open-loop parity max-diff misses
our 5e-2 bar, but mean error and cosine are small, and the closed loop shows no practical loss.

RoboCasa (`RLDX-1.1-FT-ROBOCASA`, full upstream protocol, 1,200 episodes): INT8 asym on the B70 scores
**71.8 %**, between upstream's 73.1 % `compile` and 71.4 % `any:fp8` on Thor, at 89.6 ms per call vs
107.3 ms for f16.

GR-1 Tabletop (`RLDX-1.1-FT-GR1`, full upstream protocol, 1,200 episodes): INT8 asym on the B70 scores
**58.75 %** against upstream's 60.3 % (bf16 `compile`), inside the ±2.8-point interval. The OV f16 and
torch bf16 runs on the same protocol are still to do.

## Setup

| Item | Value |
|---|---|
| Checkpoint | `RLWRLD/RLDX-1.1-FT-LIBERO-IMG` (2 cameras × 1 frame, 256², text bucket 64) |
| Device | Intel Arc Pro B70 (`GPU.0`). `GPU.1` is the RTX 3090 through OpenCL; do not use it |
| Host | i9-10980XE, 125 GB RAM; RTX 3090 used for torch references |
| Software | OpenVINO 2026.4.1, NNCF 3.4.0, RLDX-1 `v1.1.0` (`dev-1.1` branch + local export patches) |
| Export | fp32 CPU trace → IR with FP16 weights, static shapes, batch 1 |

### Pipeline

```bash
cd ~/git/RLDX-1
S=run_scripts/export/rldx_openvino.py
# 1. Export (f16-safe action head, SwiGLU/vision-MLP widths padded to 64)
.venv/bin/python $S export --model-path RLWRLD/RLDX-1.1-FT-LIBERO-IMG --hidden-alignment 64 \
    --output /tmp/rldx11_libero_img_ov_a64
# 2. INT8 weight compression (data-free, per-channel symmetric)
.venv/bin/python $S quantize --export-dir /tmp/rldx11_libero_img_ov_a64 \
    --output /tmp/rldx11_libero_img_ov_a64_int8 --mode int8_sym
# 3. Latency (blank observation; shapes are static, so input content does not change timing)
.venv/bin/python $S bench  --backend openvino --export-dir <dir> --ov-device GPU.0 --ov-precision f16 \
    --iters 50
# 4. Parity fixtures: record real observations from a torch server while a LIBERO eval drives it
#    (every 10th request, up to 64), then compare
.venv/bin/python $S serve --model-path RLWRLD/RLDX-1.1-FT-LIBERO-IMG --runtime-profile eager \
    --record-dir fixtures/libero --port 20000
.venv/bin/python $S parity --export-dir <dir> --fixtures fixtures/libero \
    --ov-devices GPU.0 --ref-device cpu --ref-dtype fp32
# 5. Closed loop (upstream eval_libero.sh untouched; batch-1 export -> N_ENVS=1)
N_EPISODES=10 bash run_scripts/export/eval_libero_openvino.sh rldx11_ov_int8 \
    /tmp/rldx11_libero_img_ov_a64_int8 GPU.0
```

## Precision

| Part | f16 export | INT8 export |
|---|---|---|
| Linear weights (all 274 MatMuls with constant weights) | FP16 | INT8, symmetric, one scale per output channel (NNCF `int8_sym`) |
| Linear activations | FP16 | INT8, quantized per token at runtime by the GPU plugin (on by default once weights are INT8) |
| Everything else (norms, SDPA, RoPE, residual adds) | FP16 | FP16 |
| MSAT state-action residual stream | rescaled by 1/16 | rescaled by 1/16 |

The IRs measured in this report use the 1/16 rescale below. Later exports use upstream's recipe
instead (see [f16 overflow: upstream's fix](#f16-overflow-upstreams-fix)).

- **The OV analogue of RLDX's `any:fp8`.** `any:fp8` is W8A8 e4m3 with one scale per weight channel and
  activation scales computed per row on every call, with no calibration (`torch._scaled_mm`). Xe2 has no
  FP8 matrix units, so the closest OV recipe is INT8 W8A8 with the same scale structure: per-channel
  weights, dynamic per-token activations, and no calibration data.
- **f16 overflow.** Without the fix, the MSAT state-action residual reaches about 111k, above the f16
  limit of 65,504. The export divides every writer to that stream by 16 and adjusts the norm eps
  (exact in fp32; peak after rescaling ~7k). Quantization neither fixes nor worsens this: INT8 matmul
  outputs go back to f16 before the residual add. The rescale must therefore stay, and quantization is
  applied to the rescaled IR.
- **Hidden alignment matters.** The action model's SwiGLU width is 10,922, not a multiple of 64. With
  INT8 weights, `vl_mlp.w3` (K=10,922) fell back to a slow path at 1.35 ms per call, which made INT8
  slower than f16 overall. `--hidden-alignment 64` zero-pads the width to 10,944, the same padding
  RLDX-1.1 applies at load for its `any:*` presets. The padding is exact (0.0 difference in fp32) and
  removes the fallback.

## Latency

Median `get_action` time per call: batch 1, 2 × 50 calls after 10 warm-up calls. "Model" is time
inside the OV infer request. The input is a blank observation: the graph has static shapes and a fixed
number of denoising steps, so timing does not depend on the input content (a recorded observation and
a blank one both give 38.5 ms on the padded INT8 IR).

| Variant (B70, f16 inference precision) | get_action (ms) | model (ms) |
|---|---|---|
| f16 IR, unpadded widths | 55.6 | 54.3 |
| f16 IR, `--hidden-alignment 64` | 54.7 | 53.5 |
| INT8 weights, unpadded | 59.5 | 58.2 |
| INT8 weights, unpadded, action-model `w3` kept f16 | 40.5 | 39.3 |
| **INT8 weights, `--hidden-alignment 64`** | **38.5** | **37.4** |
| INT8 + explicit `DYNAMIC_QUANTIZATION_GROUP_SIZE=32` | 64.3 | 63.1 |
| INT8 + explicit `DYNAMIC_QUANTIZATION_GROUP_SIZE=128` or per-token | 59.5 | 58.3 |
| *Reference: torch `eager` bf16 on RTX 3090 (same checkpoint, 5-call smoke test)* | *110.7* | *108.4* |

The explicit group-size rows were measured on the unpadded INT8 IR. Group 32 is slower; 128 and
per-token match the default (the GPU already quantizes activations dynamically).

GPU profile, device time by region (unpadded IRs; the padded INT8 IR has not been profiled):

| Region | f16 | INT8 (unpadded) |
|---|---|---|
| Text backbone (18 layers) | 19.1 ms | 11.8 ms |
| Vision encoder | 7.5 ms | 6.2 ms |
| Action model (MSAT, 4 denoising steps) | 23.0 ms | 34.3 ms (`w3` fallback) |
| Total device time | 50.7 ms | 54.5 ms |

In the f16 IR, 83 % of device time is FullyConnected. The text `gate_up_proj` layers run at roughly
80 % of the B70's 608 GB/s, i.e. they are bandwidth-bound, which is why INT8 weights pay off there.

## Open-loop parity

Normalized action chunk compared with torch fp32 on CPU, 8 recorded LIBERO observations, shared noise.

| Export (B70 f16 precision) | max \|diff\| | mean \|diff\| | min cosine | NaN | Result (atol 5e-2) |
|---|---|---|---|---|---|
| f16 (unpadded) | 2.58e-2 | 7.4e-5 | 0.99998 | no | PASS |
| INT8, `w3` kept f16 (unpadded) | 1.25e-1 | 9.3e-4 | 0.99813 | no | FAIL |
| **INT8, padded to 64** | **1.07e-1** | **9.6e-4** | **0.99839** | no | FAIL |

Decoded max \|diff\| for INT8 (padded): `eef_pos_delta` 6.2e-2, `eef_rot_delta` 8.4e-3,
`gripper_close` 1.07e-1 (the gripper output is thresholded downstream).

## Closed-loop accuracy (LIBERO, OV on the B70)

40 tasks × 10 episodes, max 720 steps, 5 actions per call, batch 1, 4 tasks in parallel, same script and
settings for both runs. Outputs are in `RLDX-1/output_final/libero/rldx11_ov_f16/` and
`RLDX-1/output_final/libero/rldx11_ov_int8/`.

| Suite | OV f16 | OV INT8 | Mean steps f16 / INT8 (all) |
|---|---|---|---|
| LIBERO-Long (`libero_10`) | 96.0 % (96/100) | 96.0 % (96/100) | 55 / 55 |
| LIBERO-Goal | 97.0 % (97/100) | 96.0 % (96/100) | 27 / 27 |
| LIBERO-Object | 98.0 % (98/100) | 98.0 % (98/100) | 30 / 30 |
| LIBERO-Spatial | 99.0 % (99/100) | 99.0 % (99/100) | 23 / 22 |
| **Average** | **97.5 % (390/400)** | **97.25 % (389/400)** | 34 / 34 |

"Steps" are policy calls, 5 actions each. Episode lengths are the same, so INT8 does not make the
policy slower to finish tasks.

Tasks where the two runs differ (successes out of 10); the flips go both ways:

| Task | f16 | INT8 |
|---|---|---|
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 9 | 10 |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 10 | 9 |
| `open_the_top_drawer_and_put_the_bowl_inside` | 9 | 8 |
| `push_the_plate_to_the_front_of_the_stove` | 10 | 9 |
| `put_the_bowl_on_the_plate` | 9 | 10 |
| `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 8 | 9 |
| `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 10 | 9 |

The hardest task is the same in both runs: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` (8/10).

| Run | Policy calls | Summed task time (4 parallel) | Server errors / rejected requests |
|---|---|---|---|
| OV f16 | 13,542 | 3,744 s | none |
| OV INT8 | 13,447 | 3,367 s (-10 %) | none |

Caveats:
- With n=400, the 95 % binomial interval is about ±1.6 percentage points, so the 0.25-point gap is
  noise. Per-task rates (n=10) are noisy.
- Upstream reports 97.9 % for `RLDX-1.1-FT-LIBERO`, which is a different checkpoint from `-IMG`, with
  50 episodes per task. A torch run of `-IMG` has not been done yet; the OV f16 export already matches
  torch fp32 open-loop (max \|diff\| 2.6e-2), so it serves as the baseline here.

## RoboCasa readiness: export and open-loop precision (`RLDX-1.1-FT-ROBOCASA`)

Goal: get the best working OV model before spending time on closed-loop RoboCasa runs.

**Export.** The checkpoint has 3 cameras × 4 frames at 256², a 64-dim state and 832 input tokens (text bucket 64).
It exports at batch 1 with `--hidden-alignment 64` after one source fix. `visual.motion_block` used
`torch.diagonal`, and OpenVINO has no conversion for `aten::diagonal`. `STSSTransformation._convert_global_to_local`
in `rldx/model/modules/backbone/motion.py` now gathers the same window with plain indexing. The new version
matches the old one bit for bit: `torch.equal` on the function for both correlation types and several
shapes, and on the full `forward`. OV CPU f32 vs torch fp32 after export: 1.1e-3.

**Overflow.** Same pattern as LIBERO. The MSAT state-action residual peaks at 73.1k in fp32, above the f16
limit. The /16 rescale is exact (0.0 difference in fp32) and brings the peak down to about 4.6k. The
next-largest activations are in text layers 14 and 16, at 12.3k. No NaN or inf appeared on any device or precision.

**Fixtures.** 33 real observations recorded from the torch `eager` server, one episode each on
`PnPCounterToCab`, `OpenDrawer`, `TurnOnSinkFaucet`, `CoffeePressButton`, `OpenDoubleDoor` and
`TurnOnStove` (5 of 6 succeeded). They are in `RLDX-1/fixtures/robocasa` (75 MB, untracked; the LIBERO
ones are in `fixtures/libero`). They were recorded by running
`rldx_openvino.py serve --runtime-profile eager --record-dir fixtures/robocasa` (saves every 10th
request) while `rollout_policy.py` drove it; there is no separate recording script. Only `parity` uses
them. To check parity on another machine, copy `fixtures/` there instead of re-recording.

**Thresholds.** These come from what upstream itself ships, measured against the same torch fp32 CPU
reference with shared noise:
- f32 / f16: at most 5e-2 max error, and mean error no worse than torch bf16 (the upstream `compile`/`eager` dtype).
- INT8: no worse than upstream `any:fp8` (W8A8 e4m3) on max error, mean error and cosine.
  `any:fp8` scores 71.4 % vs 73.1 % on RoboCasa. On the 3090 it runs through upstream's portable
  `w8a8_reference` kernel, which uses the same math as `_scaled_mm`.

Normalized action chunk, 33 fixtures:

| Variant | max \|diff\| | mean \|diff\| | min cosine | Threshold | Result |
|---|---|---|---|---|---|
| *torch bf16, RTX 3090 (calibration)* | *2.59e-2* | *3.45e-4* | *0.99989* | | |
| *torch `any:fp8` W8A8, RTX 3090 (calibration)* | *1.34e-1* | *1.09e-3* | *0.99698* | | |
| OV f32, CPU | 3.05e-3 | 2.2e-5 | 1.00000 | f16 bar | PASS |
| OV f16, B70 | 2.77e-2 | 2.0e-4 | 0.99987 | f16 bar | PASS |
| OV INT8 `int8_sym`, B70 | 1.50e-1 | 1.33e-3 | 0.99707 | `any:fp8` | FAIL (slightly worse than fp8) |
| OV INT8 `int8_sym`, dynamic activation quantization off | 1.29e-1 | 1.17e-3 | 0.99735 | `any:fp8` | FAIL (mean) |
| OV INT8 `int8_sym`, action model kept f16 | 1.37e-1 | 1.08e-3 | 0.99691 | `any:fp8` | FAIL |
| OV INT8 `int8_sym`, vision kept f16 | 1.89e-1 | 1.29e-3 | 0.99682 | `any:fp8` | FAIL |
| OV INT8 `int8_sym`, text kept f16 | 7.36e-2 | 8.6e-4 | 0.99891 | `any:fp8` | PASS |
| **OV INT8 `int8_asym`, B70** | **6.64e-2** | **9.2e-4** | **0.99919** | `any:fp8` | **PASS** |

The largest decoded error is always `end_effector_position`. For `int8_asym` the decoded max errors are
position 6.6e-2, rotation below 4.5e-2 and gripper below 6e-3.

INT8 error mostly comes from weights, not activations: turning off dynamic activation quantization barely
helps. The error is spread across the model, with the text backbone contributing most. Asymmetric
per-channel weights fix it without keeping anything in f16, and land inside the `any:fp8` drift on all
three metrics. They still miss the 5e-2 absolute bar.

Latency, `get_action` median, batch 1, 2 × 50 calls after 10 warm-up calls, OV inference precision
f16, on the Arc Pro B70 (discrete) and the Arc B390 (integrated GPU of a Core Ultra X7 358H, 30 GB
shared RAM, NUC). The B70 column was measured on a recorded observation; a blank one gives the same
timing (INT8 asym: 89.1 vs 89.6 ms), so `bench` needs no fixture. The B390 column used a blank
observation. The same command measures every IR; quantized IRs carry their compression mode in
`rldx_openvino.json`. Reports are in `RLDX-1/results/ov_bench/robocasa_bench_<ir>.json` (same file
names on both machines).

The `benchmark_app` column is OpenVINO's `benchmark_app` on the B70, with the Python host code left out:
median of 100 iterations, `-hint latency -nireq 1`, f16 inference precision, fed the real inputs of
RoboCasa fixture `000000`. Logs and JSON reports are in `RLDX-1/results/ov_bench/benchmark_app/<ir>/`.

| IR | B70 get_action (ms) | B70 model (ms) | B70 `benchmark_app` (ms) | B390 get_action (ms) | B390 model (ms) | size |
|---|---|---|---|---|---|---|
| f16 | 107.3 | 105.7 | 104.5 | 395.1 | 393.6 | 12 GB |
| INT8 `int8_sym` | 86.7 (-19 %) | 85.2 | 83.6 (-20 %) | 279.0 (-29 %) | 277.5 | 5.7 GB |
| **INT8 `int8_asym`** | **89.6 (-16 %)** | **88.1** | **86.3 (-17 %)** | **290.7 (-26 %)** | **289.2** | 5.7 GB |

The B390 is about 3.3 to 3.7 times slower than the B70. It shares system memory bandwidth, and the
model is bandwidth-bound, so INT8 weights save more there than on the B70.

The `benchmark_app` median is 1 to 2 ms below the `bench` model time, which also counts copying inputs
and outputs through the Python API. Its spread is under ±1 ms (min / max: f16 103.5 / 105.2, sym
83.0 / 84.1, asym 86.0 / 86.9). It also reports first inference (137, 140, 157 ms) and B70 compile
time (26.3, 15.6, 15.9 s). Notes for repeating it:
- Do not let `benchmark_app` fill inputs with random data. Random values in `input_ids`,
  `position_ids` and `vision_layout` index out of range. Dump one real feed to `.npy` files with
  `physical-ai-studio/tmp_scripts/dump_ov_bench_inputs.py <export> <fixture> <out_dir>`, which prints the
  `-i` mapping on its last line.
- Pass `-iop position_ids:i64`. `benchmark_app` takes the 3-D `position_ids` for an image and expects u8.
- Ignore the reported FPS. It counts the 12 camera frames as a batch of 12.

```bash
MAP=$(.venv/bin/python ../physical-ai-studio/tmp_scripts/dump_ov_bench_inputs.py \
    exports/rldx11_robocasa_ov_a64 fixtures/robocasa/000000 results/ov_bench/inputs | tail -1)
for d in rldx11_robocasa_ov_a64 rldx11_robocasa_ov_a64_int8 rldx11_robocasa_ov_a64_int8asym; do
  .venv/bin/benchmark_app -m exports/$d/rldx.xml -d GPU.0 -infer_precision f16 -hint latency \
      -nireq 1 -niter 100 -i "$MAP" -iop position_ids:i64 \
      -report_type no_counters -json_stats -report_folder results/ov_bench/benchmark_app/$d
done
```

Choice for the RoboCasa runs: **f16** (`exports/rldx11_robocasa_ov_a64`) and **INT8 asym**
(`exports/rldx11_robocasa_ov_a64_int8asym`). Asym costs about 3 ms over sym and is the only fully
quantized IR within upstream's fp8 drift.

```bash
cd ~/git/RLDX-1; S=run_scripts/export/rldx_openvino.py; F=fixtures/robocasa
.venv/bin/python $S export --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
    --hidden-alignment 64 --output exports/rldx11_robocasa_ov_a64
.venv/bin/python $S quantize --export-dir exports/rldx11_robocasa_ov_a64 \
    --output exports/rldx11_robocasa_ov_a64_int8 --mode int8_sym
.venv/bin/python $S quantize --export-dir exports/rldx11_robocasa_ov_a64 \
    --output exports/rldx11_robocasa_ov_a64_int8asym --mode int8_asym
# fp32 CPU reference is cached; torch candidates calibrate the bar
.venv/bin/python $S parity --export-dir exports/rldx11_robocasa_ov_a64 --fixtures $F \
    --ref-device cpu --ref-dtype fp32 --ref-cache fixtures/rldx11_robocasa_ref_fp32.npz \
    --torch-candidates cuda:bf16 cuda:bf16:any:fp8 --ov-devices CPU GPU.0
# Latency: run once per IR (f16, int8, int8asym)
for d in rldx11_robocasa_ov_a64 rldx11_robocasa_ov_a64_int8 rldx11_robocasa_ov_a64_int8asym; do
  .venv/bin/python $S bench --backend openvino --export-dir exports/$d --ov-device GPU.0 \
      --ov-precision f16 --iters 50 \
      --report results/ov_bench/robocasa_bench_$d.json
done
```

## Running latency and parity on a new machine

These steps were checked on a fresh clone of `dev-1.1` without flash-attn. On the B70, `bench` gave
89.0 ms for INT8 asym and `parity` reproduced the numbers above (max 6.64e-2, mean 9.2e-4, cos 0.99919).

Requirements: Linux x86_64 with `git` and [`uv`](https://docs.astral.sh/uv/), an Intel GPU with its
compute runtime (driver) installed, and a user that can open `/dev/dri/renderD*` (usually the `render`
group). No CUDA is needed. Disk: venv 6.5 GB, checkpoint 13 GB, and 12 GB (f16) or 5.7 GB (INT8) per IR.

```bash
# 1. Code: the fork's dev-1.1 branch (upstream v1.1.0 lacks run_scripts/export and the export patches)
git clone -b dev-1.1 git@github.com:eugene123tw/RLDX-1.git ~/git/RLDX-1
cd ~/git/RLDX-1

# 2. Model venv. flash-attn needs CUDA to build and the export script runs PyTorch SDPA, so skip it
uv sync --python 3.10 --no-install-package flash-attn
# openvino/nncf are not in uv.lock: reinstall them after every later `uv sync` or `uv run`
uv pip install --python .venv/bin/python openvino==2026.4.1 nncf==3.4.0

# 3. The Intel GPU must be listed (e.g. ['CPU', 'GPU.0']); use its name for --ov-device below
.venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"

# 4. Checkpoint (public, 13 GB, into ~/.cache/huggingface). bench and parity need it even for
#    OpenVINO runs: config, processor and normalization stats are read from it
.venv/bin/hf download RLWRLD/RLDX-1.1-FT-ROBOCASA

# 5. Exports: copy them from a machine that has them ...
rsync -a <host>:~/git/RLDX-1/exports/ exports/
#    ... or build them here (CPU only, about 30 GB RAM, about 25 min for the f16 export)
S=run_scripts/export/rldx_openvino.py
.venv/bin/python $S export --model-path RLWRLD/RLDX-1.1-FT-ROBOCASA \
    --hidden-alignment 64 --output exports/rldx11_robocasa_ov_a64
.venv/bin/python $S quantize --export-dir exports/rldx11_robocasa_ov_a64 \
    --output exports/rldx11_robocasa_ov_a64_int8 --mode int8_sym
.venv/bin/python $S quantize --export-dir exports/rldx11_robocasa_ov_a64 \
    --output exports/rldx11_robocasa_ov_a64_int8asym --mode int8_asym

# 6. Parity fixtures and fp32 reference cache (76 MB); recording them needs CUDA and the simulator
rsync -a <host>:~/git/RLDX-1/fixtures/ fixtures/

# 7. Latency (no fixtures needed)
mkdir -p results/ov_bench
for d in rldx11_robocasa_ov_a64 rldx11_robocasa_ov_a64_int8 rldx11_robocasa_ov_a64_int8asym; do
  .venv/bin/python $S bench --backend openvino --export-dir exports/$d --ov-device GPU.0 \
      --ov-precision f16 --iters 50 --report results/ov_bench/robocasa_bench_$d.json
done

# 8. Parity against the cached fp32 reference; INT8 exits 1 because it misses the 5e-2 bar
for d in rldx11_robocasa_ov_a64 rldx11_robocasa_ov_a64_int8asym; do
  .venv/bin/python $S parity --export-dir exports/$d --fixtures fixtures/robocasa \
      --ref-device cpu --ref-dtype fp32 --ref-cache fixtures/rldx11_robocasa_ref_fp32.npz \
      --ov-devices GPU.0 --report results/ov_bench/robocasa_parity_$d.json
done
```

Compare the parity output with the table above. `--torch-candidates` is left out because it needs CUDA;
its calibration rows (torch bf16 and `any:fp8`) are already in the table.

## RoboCasa success rate (OV on the B70)

**Where upstream's Thor numbers come from.** In `docs/optimization.md`, the success-rate column is RoboCasa
kitchen, scored by `run_scripts/eval/robocasa_kitchen/eval_robocasa.sh` (the launcher `docs/evaluation.md`
names). The footnote gives the conditions: 24 tasks × 5 scene layouts × 10 episodes, seeds 42 to 46, at
most 720 steps (1,000 for `OpenDoubleDoor`), served on Thor, mean of 2 runs. That launcher starts
`rldx/eval/run_rldx_server.py` without `--runtime-profile`, so it always measures `compile`. The
`any:fp8` and nvfp4 rows needed the server started with `--runtime-profile <preset>`, and the repo does
not contain that variant. The launcher also runs 5 environments per process and needs CUDA; it cannot
serve an OV export.

**OV runner.** `run_scripts/export/eval_robocasa_openvino.sh` is the RoboCasa counterpart of
`eval_libero_openvino.sh`. Upstream `eval_robocasa.sh` is untouched. The OV runner:
- starts one `rldx_openvino.py serve --backend openvino` server on the export, then runs
  `rldx/eval/rollout_policy.py` from the RoboCasa venv;
- reads the 24 task names from upstream `eval_robocasa.sh` and uses its protocol: seed 42, 16 actions
  per call, 720 / 1000 steps, 5-lane seed layout;
- runs each lane at batch 1 (`--n_envs 1 --env_layout 5 --env_idx_offset 0..4`), which matches the
  batch-1 export. This is the mode upstream's swap chain re-runs with, so it selects the same 1,200
  scenes, and it cannot hit the camera-view swap. No swap chain is needed;
- runs up to `MAX_PARALLEL` tasks at once against the one server, with a task's 5 lanes in turn because
  they share a results CSV;
- writes `manifest.json` and refuses to resume a directory made with another export, device,
  precision or protocol;
- prints `[progress]` lines while running and a per-task table at the end (`summary.csv`), counted
  from the episode videos as upstream does.

| Variable | Default | Meaning |
|---|---|---|
| `N_EPISODES` | 50 | Episodes per task, a multiple of 5 (one fifth per lane) |
| `TASKS` | all 24 | Comma-separated task subset |
| `TASK_LIMIT` | all | Keep the first n tasks of the list |
| `OV_PRECISION` | device default (f16 on GPU) | OV inference precision hint |
| `MAX_PARALLEL` (4th argument) | 4 | Tasks in flight against the one server |

**Setup on a fresh machine.** The script does not install anything. It checks the prerequisites
first and stops with the missing step:

1. **Repo, model venv, Intel GPU:** steps 1 to 3 of
   [Running latency and parity on a new machine](#running-latency-and-parity-on-a-new-machine).
   On a machine with CUDA, plain `uv sync --python 3.10` also works and installs flash-attn.
2. **Simulator:** `INSTALL_FLASH_ATTN=0 bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh`.
   It initializes the submodule, builds the venv with robosuite 1.5.1 and downloads about 5 GB of
   kitchen assets.
3. **Exports:** run `export` without `--observation`, then `quantize --mode int8_asym`; the commands
   are in the script header. This needs only the CPU, about 30 GB of RAM and about 25 min, and it
   downloads the checkpoint. No CUDA or recorded observations are needed: a blank observation gives
   the same input layout as a recorded RoboCasa one (checked against the existing export). Recorded
   fixtures are needed only for `parity`. Copying an existing export directory also works.

Disk needed: about 40 GB (checkpoint 13 GB, f16 IR 12 GB, INT8 IR 5.7 GB, assets and venv about 10 GB).

Partial runs are a prefix of the full run. Episode seeds are `(42 + lane) × 100000 + episode`, so
`N_EPISODES=10` runs episodes 0 and 1 of every lane. Running the same `RUN_NAME` again with a larger
`N_EPISODES` resumes and adds only the missing episodes.

```bash
cd ~/git/RLDX-1
R=run_scripts/export/eval_robocasa_openvino.sh
# Quick check: 2 tasks x 5 episodes (1 per lane), about 6 min on the B70
N_EPISODES=5 TASKS=OpenDrawer,TurnOnStove bash $R rldx11_ov_int8asym_smoke \
    exports/rldx11_robocasa_ov_a64_int8asym GPU.0 2
# Small scale: all 24 tasks x 10 episodes (240 episodes), roughly 1 to 1.5 h at MAX_PARALLEL=4
N_EPISODES=10 bash $R rldx11_ov_int8asym exports/rldx11_robocasa_ov_a64_int8asym GPU.0
# Full protocol (24 x 50 = 1,200 episodes); resumes the 10-episode run above
N_EPISODES=50 bash $R rldx11_ov_int8asym exports/rldx11_robocasa_ov_a64_int8asym GPU.0
# Same for f16
N_EPISODES=50 bash $R rldx11_ov_f16 exports/rldx11_robocasa_ov_a64 GPU.0
```

Outputs go to `output_final/robocasa/<RUN_NAME>/`: per-task videos, `simulation_results.csv`,
`eval.log`, plus `server.log`, `manifest.json` and `summary.csv` at the top.

The quick check ran end to end with the INT8 asym export: `OpenDrawer` 5/5, `TurnOnStove` 3/5, 8/10
pooled, no server errors. Lanes 0 to 4 produced `env00` to `env04`, as upstream's layout does. Ten
episodes are too few to compare with the 72.3 % upstream reports for this checkpoint.

### Result: INT8 asym, 24 tasks × 10 episodes

```bash
N_EPISODES=10 bash $R rldx11_ov_int8asym exports/rldx11_robocasa_ov_a64_int8asym GPU.0
```

Export `exports/rldx11_robocasa_ov_a64_int8asym` (`int8_asym`) on the B70 (`GPU.0`), f16 inference precision
(device default), seed 42, 5 lanes × 2 episodes per task, 16 actions per call, 720 steps (1,000 for
`OpenDoubleDoor`), `MAX_PARALLEL=4`. Outputs are in `output_final/robocasa/rldx11_ov_int8asym/`.

| | Value |
|---|---|
| **Success** | **72.9 % (175/240)** |
| 95 % binomial interval | about ±5.6 points (67.3 % to 78.5 %) |
| Upstream, same checkpoint (Thor, 1,200 episodes, mean of 2 runs) | 73.1 % `compile`, 71.4 % `any:fp8`; README 72.3 % |
| Policy calls | 5,921 (mean 24.7 per episode) |
| Wall time | about 60 min (17:31 to 18:31) |
| Server errors / rejected requests | none |

INT8 asym matches upstream: 72.9 % is between upstream's `compile` and `any:fp8` numbers, and both are
well inside the interval. Ten episodes per task cannot resolve a gap of 1 to 2 points; that needs the
full 50-episode run and an f16 run on the same protocol.

By task family:

| Family | Tasks | Success |
|---|---|---|
| Sink (faucet on/off, spout) | 3 | 100 % (30/30) |
| Drawer (open/close) | 2 | 95.0 % (19/20) |
| Microwave (on/off) | 2 | 95.0 % (19/20) |
| Doors (open/close, single/double) | 4 | 77.5 % (31/40) |
| Coffee (setup mug, serve mug, press button) | 3 | 66.7 % (20/30) |
| Stove (on/off) | 2 | 60.0 % (12/20) |
| Pick and place | 8 | 55.0 % (44/80) |

Per task (mean policy calls over all episodes and over successful ones; the limit is 45 calls, 63 for
`OpenDoubleDoor`):

| Task | Success | Mean calls (all / success) |
|---|---|---|
| CloseDrawer | 10/10 | 12.5 / 12.5 |
| CloseSingleDoor | 10/10 | 16.5 / 16.5 |
| TurnOffMicrowave | 10/10 | 17.0 / 17.0 |
| TurnOffSinkFaucet | 10/10 | 11.1 / 11.1 |
| TurnOnSinkFaucet | 10/10 | 14.1 / 14.1 |
| TurnSinkSpout | 10/10 | 10.2 / 10.2 |
| CoffeePressButton | 9/10 | 11.8 / 8.1 |
| OpenDrawer | 9/10 | 16.5 / 13.3 |
| TurnOnMicrowave | 9/10 | 20.0 / 17.2 |
| OpenDoubleDoor | 8/10 | 47.5 / 43.6 |
| PnPCounterToCab | 8/10 | 23.6 / 18.2 |
| PnPCounterToStove | 8/10 | 24.0 / 18.8 |
| CoffeeServeMug | 7/10 | 27.0 / 19.3 |
| OpenSingleDoor | 7/10 | 26.2 / 18.1 |
| PnPCounterToSink | 7/10 | 30.8 / 24.7 |
| PnPSinkToCounter | 7/10 | 24.1 / 15.1 |
| PnPStoveToCounter | 7/10 | 24.6 / 15.9 |
| TurnOnStove | 7/10 | 21.2 / 11.0 |
| CloseDoubleDoor | 6/10 | 32.8 / 24.7 |
| PnPCabToCounter | 5/10 | 33.9 / 22.8 |
| TurnOffStove | 5/10 | 28.2 / 11.4 |
| CoffeeSetupMug | 4/10 | 33.2 / 15.5 |
| PnPMicrowaveToCounter | 2/10 | 40.3 / 21.5 |
| PnPCounterToMicrowave | 0/10 | 45.0 / — |

Observations:
- Every one of the 65 failures ran to the step limit. None ended early, so failures are the policy not
  finishing the task, not crashes or server faults.
- The microwave pick-and-place tasks are the weak spot: `PnPCounterToMicrowave` 0/10 and
  `PnPMicrowaveToCounter` 2/10. Whether this is INT8 or the checkpoint itself needs the f16 or torch run
  on the same seeds.
- Per lane (scene layout seed), success ranges from 31/48 (lane 0) to 39/48 (lane 3).
- Per-task rates have n=10 (±about 30 points), so do not read single tasks as regressions.

For a torch baseline on the 3090, use the upstream launcher unchanged:
`N_GPUS=1 bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh RLWRLD/RLDX-1.1-FT-ROBOCASA` (`compile`, 5
environments per process, then its swap chain). `N_EPISODES` and `TASK_LIMIT` work the same way for
smoke runs.

### Result: INT8 asym, full protocol (24 tasks × 50 episodes)

```bash
N_EPISODES=50 bash $R rldx11_ov_int8asym exports/rldx11_robocasa_ov_a64_int8asym GPU.0
```

Same export, device, f16 inference precision and protocol as the 10-episode run (the manifest is
unchanged), 5 lanes × 10 episodes per task. The run directory held no earlier episodes (every lane logged
`Detected 0 recorded episode(s)`), so all 1,200 episodes were run fresh here instead of resuming the
10-episode run. Outputs are in `output_final/robocasa/rldx11_ov_int8asym/` (`summary.csv`, and
`simulation_results.csv`, `eval.log` and videos per task).

| | Value |
|---|---|
| **Success** | **71.8 % (861/1,200)** |
| 95 % binomial interval | about ±2.5 points (69.2 % to 74.3 %) |
| Upstream, same checkpoint (Thor, 1,200 episodes, mean of 2 runs) | 73.1 % `compile` (-1.3 points), 71.4 % `any:fp8` (+0.4 points); README 72.3 % |
| Policy calls | 30,560 (mean 25.5 per episode) |
| Wall time | about 4 h 12 min (19:18 to 23:31), `MAX_PARALLEL=4` |
| Server errors / rejected requests | none (no error or traceback in `server.log` or any `eval.log`) |

INT8 asym on the B70 reproduces upstream on the full protocol. 71.8 % falls between upstream's `compile`
and `any:fp8` scores and is closest to `any:fp8`, the W8A8 preset it is meant to match. All three upstream
numbers are inside the interval. A 1.3-point gap to `compile` is too small to resolve with 1,200 episodes.
Upstream ran 5 environments per process, while this run used batch 1 per lane. Both setups use the
same 1,200 scenes.

By task family (the 10-episode column is the separate run above):

| Family | Tasks | Success, 50 ep | Success, 10 ep |
|---|---|---|---|
| Drawer (open/close) | 2 | 95.0 % (95/100) | 95.0 % |
| Sink (faucet on/off, spout) | 3 | 94.7 % (142/150) | 100 % |
| Microwave (on/off) | 2 | 90.0 % (90/100) | 95.0 % |
| Doors (open/close, single/double) | 4 | 85.5 % (171/200) | 77.5 % |
| Coffee (setup mug, serve mug, press button) | 3 | 74.7 % (112/150) | 66.7 % |
| Pick and place | 8 | 55.0 % (220/400) | 55.0 % |
| Stove (on/off) | 2 | 31.0 % (31/100) | 60.0 % |

Per task, sorted by success. The mean policy calls are given over all episodes and over successful
ones. The limit is 45 calls, or 63 for `OpenDoubleDoor`:

| Task | Success | Rate | Mean calls (all / success) | 10-ep run |
|---|---|---|---|---|
| CloseDrawer | 50/50 | 100 % | 12.1 / 12.1 | 10/10 |
| TurnOffMicrowave | 50/50 | 100 % | 14.8 / 14.8 | 10/10 |
| CloseSingleDoor | 49/50 | 98 % | 16.9 / 16.3 | 10/10 |
| CoffeePressButton | 49/50 | 98 % | 8.8 / 8.1 | 9/10 |
| TurnSinkSpout | 49/50 | 98 % | 9.3 / 8.6 | 10/10 |
| TurnOffSinkFaucet | 47/50 | 94 % | 15.4 / 13.5 | 10/10 |
| TurnOnSinkFaucet | 46/50 | 92 % | 17.8 / 15.4 | 10/10 |
| OpenDrawer | 45/50 | 90 % | 16.6 / 13.5 | 9/10 |
| CloseDoubleDoor | 44/50 | 88 % | 26.8 / 24.3 | 6/10 |
| CoffeeServeMug | 42/50 | 84 % | 24.4 / 20.5 | 7/10 |
| PnPCounterToSink | 42/50 | 84 % | 28.9 / 25.9 | 7/10 |
| OpenDoubleDoor | 40/50 | 80 % | 48.0 / 44.2 | 8/10 |
| TurnOnMicrowave | 40/50 | 80 % | 21.2 / 15.2 | 9/10 |
| OpenSingleDoor | 38/50 | 76 % | 23.4 / 16.6 | 7/10 |
| PnPSinkToCounter | 37/50 | 74 % | 24.1 / 16.8 | 7/10 |
| PnPStoveToCounter | 34/50 | 68 % | 25.9 / 17.0 | 7/10 |
| PnPCounterToCab | 28/50 | 56 % | 28.3 / 15.2 | 8/10 |
| PnPCounterToStove | 26/50 | 52 % | 33.3 / 22.5 | 8/10 |
| CoffeeSetupMug | 21/50 | 42 % | 33.0 / 16.5 | 4/10 |
| PnPCabToCounter | 19/50 | 38 % | 35.6 / 20.3 | 5/10 |
| TurnOnStove | 19/50 | 38 % | 33.5 / 14.7 | 7/10 |
| PnPMicrowaveToCounter | 18/50 | 36 % | 36.0 / 20.1 | 2/10 |
| PnPCounterToMicrowave | 16/50 | 32 % | 39.4 / 27.6 | 0/10 |
| TurnOffStove | 12/50 | 24 % | 37.6 / 14.1 | 5/10 |

Observations:
- All 339 failures ran to the step limit. No episode ended early, and nothing crashed or errored on the
  server.
- The stove is the weakest family: `TurnOffStove` 24 %, `TurnOnStove` 38 %. A successful stove episode is
  short (about 14 calls), and every failure times out. The policy either turns the knob quickly or never
  does. The 60 % in the 10-episode run came from a small sample.
- Microwave pick and place is still weak (`PnPCounterToMicrowave` 32 %, `PnPMicrowaveToCounter` 36 %),
  but not as bad as the 0/10 and 2/10 seen in the 10-episode run. Pick and place overall is 55.0 %, the
  same as before.
- Per lane (scene layout seed), success ranges from 166/240 (69.2 %, lane 3) to 177/240 (73.8 %, lane 2).
  This spread is within the per-lane noise (about ±5.7 points).
- Episodes 0 and 1 of every lane use the same seeds as the 10-episode run. Here they scored 182/240
  (75.8 %), against 175/240 (72.9 %) then, with the same export, seeds and protocol. So the closed loop
  is not bit-reproducible across runs. Differences of a few points between runs are expected.
- This run alone cannot say whether the stove and microwave weakness comes from INT8 or from the
  checkpoint. That needs the f16 run (or torch) on the same protocol.

## f16 overflow: upstream's fix

RLWRLD confirmed the overflow is expected on every released RLDX-1.1 checkpoint and located it. It
is not in the state and action tokens. The MSAT single-stream blocks run one sequence of 64 cognition
tokens followed by the time, state and action tokens. The cognition tokens enter through
`action_model.model.vl_proj_to_sa`, and a few of its output channels (channel 42 is the largest) have
large weights. Only the cognition rows exceed the f16 range. The state and action rows stay below
about 2k. Upstream's fp32 peaks on synthetic inputs: RoboCasa 58k to 66k, LIBERO about 84k, GR-1 up
to 385k. Ours (73k to 111k) are in line.

The output reads only the state and action rows, and the cognition rows reach them only through each
block's pre-norm, a LayerNorm without affine parameters. So the cognition rows can live at a smaller
scale. The export (`rldx_openvino.py export`, `--vl-row-scale`, default 1/64) now does:

1. Multiply the weight and bias of `vl_proj_to_sa` by s = 1/64 (`MSAT.rescale_vl_rows`).
2. In each of the 8 single-stream blocks, scale the residual update per row before the add:
   `x = x + r * F(LN(x))`, with r = s for the 64 cognition rows and 1 for the rest
   (`SingleStreamBlock(row_scale=...)`).

The only change in exact arithmetic is the norm eps on the cognition rows (effectively eps / s²).
Upstream's numbers: fp32 relative difference at most 1.6e-6; GR-1 peak 385k down to about 6k; f16 vs
fp32 0.08 % to 0.11 % relative L2 (their bf16 path: 0.7 % to 1.0 %). Our synthetic check of the code
(cognition rows pushed to 162k): fp32 relative difference 2e-7, the original gives non-finite f16
outputs, the rescaled one 7e-4 relative L2. The state and action rows are untouched.

Compared with the 1/16 rescale used for the IRs above: 1/16 would leave GR-1 at about 24k. The new
recipe also touches one layer's weights instead of every writer to the stream, and leaves the double
blocks alone. The export prints the single-stream peaks split into cognition rows
(`<msat.single.vl_rows>`) and state/action rows (`<msat.single.sa_rows>`), before and after the
rescale, and warns if anything is still above the f16 limit.

Upstream also asked that the backbone `LayerNorm` / `RMSNorm` variance stays in fp32: the residual
stream peaks around 12k, which fits f16, but its square does not. The OV GPU plugin already does this.
On the RoboCasa f16 IR every norm compiles to a fused `rms` (197) or `MVN` (156) kernel, and a
stand-alone test at 12k magnitude stays finite with about 4e-4 relative L2 error.

## GR-1 Tabletop (`RLDX-1.1-FT-GR1`): validation plan

Upstream asked for GR-1 tabletop success rates of the f16 and INT8 builds next to a bf16 run on the
same setup: this checkpoint has the largest peak they measured (385k), so it also tests the fix.

**The benchmark.** GR-1 Tabletop is a simulated benchmark: NVIDIA's `robocasa-gr1-tabletop-tasks`
(from GR00T), a GR-1 humanoid with Fourier hands, one ego camera at 256². It has 24 tasks: 18 object
rearrangement and 6 articulated objects. `RLDX-1.1-FT-GR1` is fine-tuned on the GR-1 sets of
`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`, embodiment tag `GENERAL_EMBODIMENT`. Upstream
reports **60.3 %** (24 tasks × 50 episodes; README, `run_scripts/eval/gr1_tabletop/README.md`).

**Where upstream's number comes from.** `run_scripts/eval/gr1_tabletop/eval_gr1.sh`:
- 24 tasks split over 8 shards, each with its own `run_rldx_server.py` (`compile`, bf16), one shard
  per card at a time;
- every rollout at `n_envs 1`, seed 42, 5 lanes × 10 episodes per task, 8 actions per call, at most
  720 steps, uniform denoising timesteps;
- the reset protocol (`GR1_PROTOCOL=on`): a 5 s physics settle after reset (a patch to the simulator)
  and a frozen seed manifest that swaps 130 broken scene slots for pre-validated seeds. Results are not
  comparable across the switch;
- score: pooled success over the per-task `simulation_results.csv`.

**OV runner.** `run_scripts/export/eval_gr1_openvino.sh` is the GR-1 counterpart of
`eval_robocasa_openvino.sh`. Upstream `eval_gr1.sh` is untouched. The OV runner:
- starts one `rldx_openvino.py serve --backend openvino` server on the export, then runs
  `rldx/eval/rollout_policy.py` from the GR-1 venv;
- reads the 24 task names from upstream `eval_gr1.sh` and uses its protocol: seed 42, 5 lanes
  (`--n_envs 1 --env_layout 5 --env_idx_offset 0..4`), 8 actions per call, 720 steps;
- turns on the reset protocol the way upstream does (`GR1_PROTOCOL_SEED`, `GR1_RESET_SETTLE_S=5.0`,
  `GR1_EPISODE_SEED_MANIFEST`, `gr1_protocol` on `PYTHONPATH`, on the rollout only), and refuses to run
  if the settle or determinism patch is missing from the submodule;
- uses one server for all tasks instead of 8 shard servers. The scenes are the same, since the seed
  manifest is keyed on lane and episode, not shard. The noise stream differs from upstream anyway
  (OV draws it from the CPU torch RNG), so the comparison is on success rate, not per episode;
- runs up to `MAX_PARALLEL` tasks at once, with a task's 5 lanes in turn;
- writes `manifest.json` (export, compression, `vl_row_scale`, device, precision, protocol, seed
  manifest hash) and refuses to resume a directory made with another setting;
- prints `[progress]` lines from the episode videos, and at the end a per-task table (`summary.csv`)
  and the pooled rate from the task CSVs, as upstream does. It fails if a task did not record
  `N_EPISODES` episodes, and prints how many `[seed-manifest]` substitutions were logged (a full
  50-episode run logs about 130);
- with `RECORD_DIR`, also saves every `RECORD_EVERY`-th observation as a parity fixture.

| Variable | Default | Meaning |
|---|---|---|
| `N_EPISODES` | 50 | Episodes per task, a multiple of 5 (one fifth per lane) |
| `TASKS` | all 24 | Comma-separated subset, with or without the `_GR1ArmsAndWaistFourierHands_Env` suffix |
| `TASK_LIMIT` | all | Keep the first n tasks of the list |
| `OV_PRECISION` | device default (f16 on GPU) | OV inference precision hint |
| `RECORD_DIR` / `RECORD_EVERY` | off / 10 | Save observations as parity fixtures |
| `MAX_PARALLEL` (4th argument) | 4 | Tasks in flight against the one server |

**Setup on a fresh machine.** As for RoboCasa, the script installs nothing and stops with the
missing step:

1. **Repo, model venv, Intel GPU:** steps 1 to 3 of
   [Running latency and parity on a new machine](#running-latency-and-parity-on-a-new-machine).
2. **Simulator:** `INSTALL_FLASH_ATTN=0 bash run_scripts/eval/gr1_tabletop/setup_gr1.sh`. It checks out
   the `robocasa-gr1-tabletop-tasks` submodule, applies the determinism and reset-settle patches, builds
   the venv with robosuite 1.5.1 and downloads the tabletop assets. The patches edit the submodule, so
   `git pull` does not bring them.
3. **Checkpoint:** `.venv/bin/hf download RLWRLD/RLDX-1.1-FT-GR1`. The GR-1 README warns that
   `max_num_embodiments` in `config.json` can disagree with the weights; check the first dimension of
   `action_model.action_decoder.layer1.W` before exporting.
4. **Exports:** `export` with the new rescale, then `quantize --mode int8_sym` and `--mode int8_asym`
   (commands below and in the script header). CPU only, about 30 GB of RAM.

**Steps.**

```bash
cd ~/git/RLDX-1; S=run_scripts/export/rldx_openvino.py; R=run_scripts/export/eval_gr1_openvino.sh
# 1. Export. Check the log: <msat.single.vl_rows> should drop from ~3.85e5 to ~6e3, sa_rows unchanged
.venv/bin/python $S export --model-path RLWRLD/RLDX-1.1-FT-GR1 --hidden-alignment 64 \
    --output exports/rldx11_gr1_ov_a64
.venv/bin/python $S quantize --export-dir exports/rldx11_gr1_ov_a64 \
    --output exports/rldx11_gr1_ov_a64_int8 --mode int8_sym
.venv/bin/python $S quantize --export-dir exports/rldx11_gr1_ov_a64 \
    --output exports/rldx11_gr1_ov_a64_int8asym --mode int8_asym
# Optional evidence for upstream: without the fix, GPU f16 overflows (expect NaN in parity).
# --no-f16-safe also skips the GPU first-step untie, so a NaN here is not only the rescale's doing.
.venv/bin/python $S export --model-path RLWRLD/RLDX-1.1-FT-GR1 --hidden-alignment 64 \
    --no-f16-safe --output exports/rldx11_gr1_ov_a64_nofix
# 2. Smoke runs (2 tasks x 5 episodes each); the f16 one also records parity fixtures
N_EPISODES=5 TASKS=PnPCanToDrawerClose,PnPCupToDrawerClose RECORD_DIR=fixtures/gr1 \
    bash $R rldx11_gr1_ov_f16_smoke exports/rldx11_gr1_ov_a64 GPU.0 2
for q in int8 int8asym; do
  N_EPISODES=5 TASKS=PnPCanToDrawerClose,PnPCupToDrawerClose \
      bash $R rldx11_gr1_ov_${q}_smoke exports/rldx11_gr1_ov_a64_$q GPU.0 2
done
# 3. Open-loop parity vs torch fp32 CPU (cached), with torch bf16 as the calibration row
.venv/bin/python $S parity --export-dir exports/rldx11_gr1_ov_a64 --fixtures fixtures/gr1 \
    --ref-device cpu --ref-dtype fp32 --ref-cache fixtures/rldx11_gr1_ref_fp32.npz \
    --torch-candidates cuda:bf16 cuda:bf16:any:fp8 --ov-devices CPU GPU.0
.venv/bin/python $S parity --export-dir exports/rldx11_gr1_ov_a64_int8asym --fixtures fixtures/gr1 \
    --ref-device cpu --ref-dtype fp32 --ref-cache fixtures/rldx11_gr1_ref_fp32.npz --ov-devices GPU.0
.venv/bin/python $S parity --export-dir exports/rldx11_gr1_ov_a64_int8 --fixtures fixtures/gr1 \
    --ref-device cpu --ref-dtype fp32 --ref-cache fixtures/rldx11_gr1_ref_fp32.npz --ov-devices GPU.0
# 4. Closed loop, full protocol: OV f16 and INT8 asym on the B70 (INT8 sym too if its parity passes)
N_EPISODES=50 bash $R rldx11_gr1_ov_f16 exports/rldx11_gr1_ov_a64 GPU.0
N_EPISODES=50 bash $R rldx11_gr1_ov_int8asym exports/rldx11_gr1_ov_a64_int8asym GPU.0
# 5. Torch bf16 baseline on the RTX 3090: upstream launcher unchanged (8 shards in turn on one card)
N_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash run_scripts/eval/gr1_tabletop/eval_gr1.sh RLWRLD/RLDX-1.1-FT-GR1
# 6. Latency, as for RoboCasa: bench per IR on the B70 (blank observation is enough; shapes are static)
for d in rldx11_gr1_ov_a64 rldx11_gr1_ov_a64_int8 rldx11_gr1_ov_a64_int8asym; do
  .venv/bin/python $S bench --backend openvino --export-dir exports/$d --ov-device GPU.0 \
      --ov-precision f16 --iters 50 --report results/ov_bench/gr1_bench_$d.json
done
# Torch reference on the 3090, same checkpoint
.venv/bin/python $S bench --backend torch --model-path RLWRLD/RLDX-1.1-FT-GR1 --device cuda \
    --runtime-profile compile --iters 50 --report results/ov_bench/gr1_bench_torch_compile.json
# benchmark_app without the Python host code, fed one recorded GR-1 fixture (needs step 2)
MAP=$(.venv/bin/python ../physical-ai-studio/tmp_scripts/dump_ov_bench_inputs.py \
    exports/rldx11_gr1_ov_a64 fixtures/gr1/000000 results/ov_bench/inputs_gr1 | tail -1)
for d in rldx11_gr1_ov_a64 rldx11_gr1_ov_a64_int8 rldx11_gr1_ov_a64_int8asym; do
  .venv/bin/benchmark_app -m exports/$d/rldx.xml -d GPU.0 -infer_precision f16 -hint latency \
      -nireq 1 -niter 100 -i "$MAP" -iop position_ids:i64 \
      -report_type no_counters -json_stats -report_folder results/ov_bench/benchmark_app/$d
done
```

The `bench` and `benchmark_app` notes from the RoboCasa section apply unchanged: feed real inputs, pass
`-iop position_ids:i64`, and ignore the reported FPS (it counts camera frames as the batch). For the
B390, run the same `bench` loop on the NUC with the exports copied over.

The recorded fixtures come from the OV f16 policy's trajectories. They are still real simulator
observations, which is all `parity` needs. To record from torch instead, start
`rldx_openvino.py serve --runtime-profile eager --record-dir fixtures/gr1` and drive it with
`rollout_policy.py` and the protocol variables above, as for RoboCasa.

Outputs: the OV runs go to `output_final/gr1_tabletop/<RUN_NAME>/` (per-task videos,
`simulation_results.csv`, `eval.log`, plus `server.log`, `manifest.json` and `summary.csv`). The torch
baseline goes to `output_final/gr1_tabletop/RLWRLD_RLDX-1.1-FT-GR1/h8_uniform_ep50_seed42_resetproto/`.

**How to read the results.**
- Pass for the fix: the export shows the cognition-row peak under the f16 limit, f16 parity has no NaN,
  and `server.log` shows no NaN or inf.
- Parity bar: the absolute 5e-2 max does not work on GR-1. The hand actions are discrete (targets of
  ±1.5 and ±3, normalized to ±1), so at an open/close transition any precision change flips a hand
  joint by the full range: normalized max |diff| about 2.0, decoded about 6. Torch bf16 does the same
  (2.01), so compare with the calibration rows instead: f16 mean and min cosine no worse than torch
  bf16, INT8 no worse than `any:fp8`. OV f32 on CPU must still be tight (no flips).
- First parity run (64 fixtures from the f16 smoke run, torch fp32 CPU reference):

  | Variant | max \|diff\| | mean \|diff\| | min cosine |
  |---|---|---|---|
  | *torch bf16, RTX 3090* | *2.01* | *1.10e-3* | *0.98096* |
  | *torch `any:fp8`, RTX 3090* | *2.02* | *2.10e-3* | *0.98079* |
  | OV f32, CPU | 8.7e-3 | 1.5e-5 | 1.00000 |
  | OV f16, B70 | 2.02 | 4.8e-4 | 0.98126 |

  OV f16 is at least as close to fp32 as upstream's bf16 path; the max comes from the same hand flips.
- INT8 asym: within upstream `any:fp8` drift on parity, as on RoboCasa.
- Closed loop: at about 60 % over 1,200 episodes the 95 % interval is about ±2.8 points, so the f16,
  INT8 and bf16 runs agree if they are inside each other's interval and near 60.3 %. Gaps of 1 to 2
  points cannot be resolved. Closed loop is not bit-reproducible (RoboCasa INT8 moved 3 points between
  identical runs).
- Report upstream's 60.3 % alongside ours, but the bf16 baseline is our own run on the same scenes.

**Status.** Done: exports, the OV f16 smoke run with fixtures, f16 parity (above) and the INT8 asym
full run (below). Not run yet: the OV f16 full run, the torch bf16 baseline, INT8 parity and latency.

### Result: INT8 asym, full protocol (24 tasks × 50 episodes)

```bash
N_EPISODES=50 bash $R rldx11_gr1_ov_int8asym exports/rldx11_gr1_ov_a64_int8asym GPU.0
```

Export `exports/rldx11_gr1_ov_a64_int8asym` (`int8_asym`, `vl_row_scale` 1/64) on the B70 (`GPU.0`), f16
inference precision (device default), seed 42, 5 lanes × 10 episodes per task, 8 actions per call,
720 steps (90 calls), uniform timesteps, reset protocol on (5 s settle, seed manifest), `MAX_PARALLEL=4`.
Outputs are in `output_final/gr1_tabletop/rldx11_gr1_ov_int8asym/`.

| | Value |
|---|---|
| **Success** | **58.75 % (705/1,200)** |
| 95 % binomial interval | about ±2.8 points (56.0 % to 61.5 %) |
| Upstream, same checkpoint (torch bf16 `compile`, 1,200 episodes) | 60.3 % (-1.55 points) |
| Policy calls | 66,931 (mean 55.8 per episode) |
| Wall time | about 3 h 47 min (17:16 to 21:03), `MAX_PARALLEL=4` |
| Seed-manifest substitutions | 130, as expected for a full run |
| Server errors / rejected requests | none (no error or traceback in `server.log` or any `eval.log`) |

INT8 asym is 1.55 points below upstream's 60.3 %. Upstream's number is inside the interval, so the gap
is not resolvable at 1,200 episodes. This is the checkpoint with the largest f16 overflow (385k), and
the run had no NaN-driven failures: every episode either succeeded or ran to the step limit.

By task group:

| Group | Tasks | Success |
|---|---|---|
| Articulated (put in drawer/cabinet/microwave, then close) | 6 | 60.0 % (180/300) |
| Rearrangement (`PosttrainPnPNovel*`) | 18 | 58.3 % (525/900) |

Rearrangement by destination:

| Destination | Tasks | Success |
|---|---|---|
| Plate | 3 | 78.0 % (117/150) |
| Basket | 2 | 65.0 % (65/100) |
| Pot | 2 | 61.0 % (61/100) |
| Pan | 2 | 60.0 % (60/100) |
| Cardboard box | 3 | 58.7 % (88/150) |
| Bowl | 2 | 54.0 % (54/100) |
| Tiered basket | 2 | 44.0 % (44/100) |
| Tiered shelf | 2 | 36.0 % (36/100) |

Per task, sorted by success. The mean policy calls are given over all episodes and over successful
ones. The limit is 90 calls:

| Task | Success | Rate | Mean calls (all / success) |
|---|---|---|---|
| PosttrainPnPNovelFromPlateToPlateSplitA | 42/50 | 84 % | 36.4 / 26.2 |
| PnPBottleToCabinetClose | 41/50 | 82 % | 52.3 / 44.0 |
| PosttrainPnPNovelFromTrayToPlateSplitA | 40/50 | 80 % | 39.9 / 27.4 |
| PnPCanToDrawerClose | 39/50 | 78 % | 49.1 / 37.6 |
| PnPWineToCabinetClose | 39/50 | 78 % | 66.3 / 59.7 |
| PosttrainPnPNovelFromPlacematToPlateSplitA | 35/50 | 70 % | 44.0 / 24.3 |
| PosttrainPnPNovelFromPlacematToBasketSplitA | 33/50 | 66 % | 52.5 / 33.2 |
| PosttrainPnPNovelFromCuttingboardToBasketSplitA | 32/50 | 64 % | 51.9 / 30.5 |
| PosttrainPnPNovelFromTrayToCardboardboxSplitA | 32/50 | 64 % | 48.1 / 24.6 |
| PosttrainPnPNovelFromTrayToPotSplitA | 31/50 | 62 % | 50.0 / 25.5 |
| PosttrainPnPNovelFromCuttingboardToPanSplitA | 30/50 | 60 % | 51.5 / 25.8 |
| PosttrainPnPNovelFromCuttingboardToPotSplitA | 30/50 | 60 % | 51.6 / 26.0 |
| PosttrainPnPNovelFromPlateToPanSplitA | 30/50 | 60 % | 51.0 / 24.9 |
| PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA | 29/50 | 58 % | 56.1 / 31.6 |
| PosttrainPnPNovelFromPlacematToBowlSplitA | 29/50 | 58 % | 51.9 / 24.4 |
| PosttrainPnPNovelFromPlateToCardboardboxSplitA | 27/50 | 54 % | 53.9 / 23.2 |
| PosttrainPnPNovelFromPlateToBowlSplitA | 25/50 | 50 % | 58.6 / 27.2 |
| PnPMilkToMicrowaveClose | 24/50 | 48 % | 66.8 / 41.7 |
| PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA | 23/50 | 46 % | 62.1 / 29.3 |
| PosttrainPnPNovelFromTrayToTieredbasketSplitA | 21/50 | 42 % | 65.4 / 31.5 |
| PnPCupToDrawerClose | 19/50 | 38 % | 65.9 / 26.5 |
| PosttrainPnPNovelFromTrayToTieredshelfSplitA | 19/50 | 38 % | 67.0 / 29.5 |
| PnPPotatoToMicrowaveClose | 18/50 | 36 % | 77.0 / 53.8 |
| PosttrainPnPNovelFromPlacematToTieredshelfSplitA | 17/50 | 34 % | 69.0 / 28.3 |

Observations:
- All 495 failures ran to the step limit (90 calls). No episode ended early, and nothing crashed or
  errored on the server.
- The weak spots are tiered destinations (shelf 36 %, basket 44 %), the microwave tasks (milk 48 %,
  potato 36 %) and `PnPCupToDrawerClose` (38 %). Plate destinations are the easiest (78 %).
- Per lane (scene layout seed), success ranges from 127/240 (52.9 %, lane 1) to 151/240 (62.9 %,
  lane 2). This spread is within the per-lane noise (about ±6.2 points).
- Against the OV f16 smoke run (episode 0 of every lane, same scenes): `PnPCanToDrawerClose` 5/5 in
  both, `PnPCupToDrawerClose` 0/5 here vs 1/5 for f16. Five episodes say nothing about precision.
- This run alone cannot say whether the 1.55-point gap or the weak tasks come from INT8 or from the
  checkpoint. That needs the OV f16 and torch bf16 runs on the same protocol.

## Artifacts

| Path | What |
|---|---|
| `RLDX-1/run_scripts/export/rldx_openvino.py` | `export` (`--hidden-alignment`), `quantize` (NNCF), `parity`, `bench`, `serve`, `--ov-config KEY=VALUE` |
| `RLDX-1/run_scripts/export/eval_libero_openvino.sh` | Closed-loop LIBERO with the OV server; `N_ENVS`, `[progress]` lines |
| `RLDX-1/run_scripts/export/eval_robocasa_openvino.sh` | Closed-loop RoboCasa with the OV server; full or partial (`N_EPISODES`, `TASKS`, `TASK_LIMIT`) |
| `RLDX-1/run_scripts/export/eval_gr1_openvino.sh` | Closed-loop GR-1 Tabletop with the OV server and the reset protocol; `RECORD_DIR` for fixtures |
| `RLDX-1/results/ov_bench/*.json` | Bench and parity reports |
| `/tmp/rldx11_libero_img_ov_a64`, `/tmp/rldx11_libero_img_ov_a64_int8` | f16 and INT8 IRs (in `/tmp`; move them out to keep them) |
| `/tmp/ov_profile.py` | Per-layer GPU profile script (`PERF_COUNT`) |

## TODO

Accuracy:
- [x] OV f16 closed-loop baseline (`_a64`), `N_EPISODES=10`: 97.5 % vs INT8 97.25 %.
- [ ] Torch `compile` closed-loop baseline of `-IMG` on the RTX 3090, same settings.
- [ ] Repeat f16 and INT8 with `N_EPISODES=50` to match upstream reporting and tighten the interval.
- [ ] Calibrate the parity bar: measure torch bf16 vs fp32 drift on the same fixtures and set `atol`
      from it instead of the fixed 5e-2.
- [ ] If INT8 costs accuracy: keep sensitive layers in f16 (`quantize --ignore`, e.g. MSAT `sa_*`) or
      try `int8_asym`.

Latency:
- [ ] Profile the padded INT8 IR to see where the remaining 37 ms goes (action model vs text vs vision).
- [ ] Try INT4 (`int4_sym --group-size 128 --ratio ...`) on the text MLPs, which are bandwidth-bound.
- [ ] Check smaller costs: `masked_scatter/NonZero` (0.77 ms), Transpose/permute (~1 ms).
- [ ] Test OV `ACTIVATIONS_SCALE_FACTOR` as a replacement for the manual residual rescale (needs a
      `--no-f16-safe` export).

RoboCasa (the benchmark RLDX's `docs/optimization.md` table uses: `RLDX-1.1-FT-ROBOCASA`, 3 cameras × 4
frames, 24 kitchen tasks × 50 episodes, success 73.1 % `compile` / 71.4 % `any:fp8` on Thor):
- [x] Free disk (265 GB free now).
- [x] Set up the simulator: `run_scripts/eval/robocasa_kitchen/setup_robocasa.sh` (robosuite 1.5.1;
      assets already on disk; `INSTALL_FLASH_ATTN=0`).
- [x] Record RoboCasa fixtures with the torch server (33 observations, 6 tasks).
- [x] Export `RLDX-1.1-FT-ROBOCASA` at batch 1 with `--hidden-alignment 64`, f16 and INT8 (sym, asym).
      `motion_block` needed the `aten::diagonal` rewrite; the residual rescale still covers the action model.
- [x] Open-loop precision: f32/f16 pass the bf16-calibrated bar; INT8 asym is within upstream `any:fp8` drift.
- [ ] Optional: get INT8 under the 5e-2 absolute bar (asym with text `down_proj` kept f16, or a
      data-aware NNCF pass) if closed loop shows a gap to f16.
- [ ] Latency, same checkpoint and input as the upstream table: `bench` on the RTX 3090 (`eager`,
      `compile`, `any:bf16`, `--limits-batch 3`) and on the B70 (OV f16, INT8).
- [x] OV closed-loop runner `run_scripts/export/eval_robocasa_openvino.sh` (upstream `eval_robocasa.sh`
      untouched): same protocol, each lane at batch 1. Quick check passed (2 tasks × 5 episodes).
- [x] OV INT8 asym on the B70, `N_EPISODES=10`: 72.9 % (175/240).
- [ ] OV f16 on the B70, `N_EPISODES=10`, same protocol, to compare with INT8 asym.
- [ ] Torch baseline on the RTX 3090 with upstream `eval_robocasa.sh` (`N_GPUS=1`).
- [x] Full protocol (`N_EPISODES=50`) for INT8 asym: 71.8 % (861/1,200).
- [ ] Full protocol (`N_EPISODES=50`) for f16, to compare with INT8 asym per task (stove, microwave PnP).
- [ ] Re-export with the VL-row rescale (the current IRs use 1/16) and re-check parity.

GR-1 Tabletop (`RLDX-1.1-FT-GR1`, requested by upstream; 24 tasks × 50 episodes, reported 60.3 %):
- [x] OV closed-loop runner `run_scripts/export/eval_gr1_openvino.sh` (upstream `eval_gr1.sh` untouched).
- [ ] Set up the simulator (`INSTALL_FLASH_ATTN=0 setup_gr1.sh`) and download the checkpoint.
- [ ] Export f16 (VL-row rescale), INT8 sym and INT8 asym; record the cognition-row peak before and after.
- [ ] Optional: `--no-f16-safe` export to show the overflow on GPU f16.
- [ ] Smoke runs (f16 with `RECORD_DIR`, INT8 sym, INT8 asym), then open-loop parity (f32, f16, INT8
      sym and asym, torch bf16 / `any:fp8`).
- [ ] Full protocol: OV f16 and OV INT8 asym on the B70, torch bf16 (`eval_gr1.sh`, `N_GPUS=1`) on the 3090.
- [ ] Latency: `bench` for f16, INT8 sym and INT8 asym on the B70 (and B390), torch `compile` on the 3090,
      `benchmark_app` on the B70.
- [ ] Same three runs on RoboCasa with the new rescale, and report both benchmarks to upstream.

Engineering:
- [ ] Batch > 1 export, so `N_ENVS=5` works and throughput can be measured.
- [ ] Port `--hidden-alignment` and the INT8 path into the Studio library export
      (`physicalai-train`) and the Runtime manifest.
- [ ] Move exports out of `/tmp`; free disk (the disk was at 100 % during LIBERO setup).
- [ ] Commit `run_scripts/export/` (currently untracked in RLDX-1).
