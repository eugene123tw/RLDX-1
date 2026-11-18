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

| IR | B70 get_action (ms) | B70 model (ms) | B390 get_action (ms) | B390 model (ms) | size |
|---|---|---|---|---|---|
| f16 | 107.3 | 105.7 | 395.1 | 393.6 | 12 GB |
| INT8 `int8_sym` | 86.7 (-19 %) | 85.2 | 279.0 (-29 %) | 277.5 | 5.7 GB |
| **INT8 `int8_asym`** | **89.6 (-16 %)** | **88.1** | **290.7 (-26 %)** | **289.2** | 5.7 GB |

The B390 is about 3.3 to 3.7 times slower than the B70. It shares system memory bandwidth, and the
model is bandwidth-bound, so INT8 weights save more there than on the B70.

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

1. **Repo:** a checkout of the branch with `run_scripts/export/` and the export patches
   (`dev-1.1` on the `eugene` fork), not upstream `v1.1.0`.
2. **Model venv:** `uv sync --python 3.10`, then
   `uv pip install --python .venv/bin/python openvino==2026.4.1 nncf==3.4.0`. openvino and nncf are
   not in the lockfile, so every later `uv sync` or `uv run` removes them; reinstall them afterwards.
3. **Intel GPU:** the compute runtime is installed, and
   `.venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"` lists the device.
4. **Simulator:** `INSTALL_FLASH_ATTN=0 bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh`.
   It initializes the submodule, builds the venv with robosuite 1.5.1 and downloads about 5 GB of
   kitchen assets.
5. **Exports:** run `export` without `--observation`, then `quantize --mode int8_asym`; the commands
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

For a torch baseline on the 3090, use the upstream launcher unchanged:
`N_GPUS=1 bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh RLWRLD/RLDX-1.1-FT-ROBOCASA` (`compile`, 5
environments per process, then its swap chain). `N_EPISODES` and `TASK_LIMIT` work the same way for
smoke runs.

## Artifacts

| Path | What |
|---|---|
| `RLDX-1/run_scripts/export/rldx_openvino.py` | `export` (`--hidden-alignment`), `quantize` (NNCF), `parity`, `bench`, `serve`, `--ov-config KEY=VALUE` |
| `RLDX-1/run_scripts/export/eval_libero_openvino.sh` | Closed-loop LIBERO with the OV server; `N_ENVS`, `[progress]` lines |
| `RLDX-1/run_scripts/export/eval_robocasa_openvino.sh` | Closed-loop RoboCasa with the OV server; full or partial (`N_EPISODES`, `TASKS`, `TASK_LIMIT`) |
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
- [ ] Test OV `ACTIVATIONS_SCALE_FACTOR` as a replacement for the manual 1/16 residual rescale (needs a
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
- [ ] Torch baseline on the RTX 3090 with upstream `eval_robocasa.sh` (`N_GPUS=1`), then OV f16 and
      INT8 on the B70. Smoke with `N_EPISODES=10` first (must be a multiple of 5).

Engineering:
- [ ] Batch > 1 export, so `N_ENVS=5` works and throughput can be measured.
- [ ] Port `--hidden-alignment` and the INT8 path into the Studio library export
      (`physicalai-train`) and the Runtime manifest.
- [ ] Move exports out of `/tmp`; free disk (the disk was at 100 % during LIBERO setup).
- [ ] Commit `run_scripts/export/` (currently untracked in RLDX-1).
