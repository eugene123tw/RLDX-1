# Installation

This guide walks through a fresh install of RLDX-1.1 on a Linux machine with
an NVIDIA GPU. The short version is at the top; subsequent sections cover
simulator environments, developer tooling, and common pitfalls.

## TL;DR

There are two install paths. Pick the one that matches your GPU:

| Path | Tool |
|---|---|
| [Standard install](#standard-install) | `uv` |
| [SM_120 (RTX 5090 / Blackwell)](#rtx-5090--blackwell-sm_120) | `pixi` |

For nvfp4 quantized inference or aarch64 (GB10 / Thor), see
[Quantized inference](#quantized-inference-quantize).

```bash
# Standard install
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
uv sync --python 3.10
python -c "import rldx; print(rldx.__version__)"
```

```bash
# RTX 5090 / Blackwell (SM_120) — flash-attn must be built from source
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
pixi install
pixi run -e default postinstall
pixi run python -c "import rldx; print(rldx.__version__)"
```

If `import rldx` prints the installed rldx version (no import error) you are done for training and inference
against pre-trained checkpoints. Simulator eval stacks install separately
(see [Simulator environments](#simulator-environments)).

## Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Linux | x86_64 | The model has only been exercised on Linux. |
| Python | `3.10.*` (pinned) | pyproject.toml constrains `requires-python`. |
| CUDA toolkit | 12.x | Needed to build `flash-attn==2.7.4.post1`. |
| NVIDIA driver | supports CUDA 12 | For `torch==2.7.0` + `flash-attn`. |
| [uv](https://github.com/astral-sh/uv) | `>= 0.8.4` | Dependency + virtualenv manager (Standard path). |
| [pixi](https://pixi.sh) | `>= 0.68` | Conda-backed env manager (SM_120 path; required for lockfile v7). |
| `git` / `git-lfs` | recent | Submodules + HF LFS assets. |
| `libjemalloc2` | optional, recommended | Preloaded by the training scripts to keep host RSS flat on long runs. |

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install `pixi` only if you are on the SM_120 path:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Install jemalloc for training:

```bash
sudo apt install libjemalloc2
```

The training scripts under `run_scripts/train/` preload it from
`/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`. On another distribution or
architecture, point `JEMALLOC_PATH` at the library, e.g.
`JEMALLOC_PATH=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2`. Without it the
scripts print a warning and train with glibc malloc.

## Standard install

Default path. flash-attn ships prebuilt wheels and `uv sync` installs
them directly without compilation. Use the
[SM_120 path](#rtx-5090--blackwell-sm_120) instead if you're on
RTX 5090 / Blackwell.

```bash
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
uv sync --python 3.10          # creates .venv and resolves deps
```

`uv sync` materialises the environment described by `pyproject.toml`,
including the `[tool.uv.extra-build-dependencies]` override that pins
`torch==2.7.0 + numpy==1.26.4` at build time for `flash-attn`. The
resulting virtualenv lives at `.venv/` and is activated automatically
by every `uv run` call.

### Verify

```bash
# Python import smoke test — should print the installed rldx version (no ImportError)
uv run python -c "import rldx; print(rldx.__version__)"
```

## RTX 5090 / Blackwell (SM_120)

flash-attn upstream does not publish prebuilt wheels for `sm_120`. The
Standard path's `uv sync` therefore installs a wheel without sm_120
kernels, which fails at the first kernel launch. The pixi path solves this
by providing CUDA 12.8 toolkit through the `nvidia` conda channel and
compiling flash-attn from source with `FLASH_ATTN_CUDA_ARCHS="120"`.

```bash
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1

# Resolve and install the pixi-managed conda + PyPI environment.
pixi install

# Run the chained postinstall task — builds flash-attn from source for
# sm_120 and editable-installs the rldx package against the pixi torch.
# MAX_JOBS caps parallel nvcc jobs; without it the build can run out of host RAM.
MAX_JOBS=4 pixi run -e default postinstall
```

The `postinstall` task chains `install-flash-attn` (source build) before
`pip install -e . --no-deps` so a single command bootstraps the entire
env. The arch target is `FLASH_ATTN_CUDA_ARCHS="120"` in
`[feature.base.tasks.install-flash-attn]`; flash-attn ignores
`TORCH_CUDA_ARCH_LIST`.

### Verify (pixi)

```bash
# Python import smoke test — should print the installed rldx version (no ImportError)
pixi run python -c "import rldx; print(rldx.__version__)"

# flash-attn was built against the pixi torch and targets sm_120
pixi run python -c "
import torch
from flash_attn import flash_attn_func
print(torch.__version__, torch.cuda.get_device_capability())
"
```

### Notes

- `pixi.toml` pins **torch 2.8.0 + cu128** (vs the uv path's `torch 2.7.0`).
  The two environments are deliberately separate; do not mix them.
- The first `postinstall` takes about 30 minutes because flash-attn
  compiles from source. Subsequent runs reuse the cached build.
- If you are not on Blackwell, prefer the Standard install — pixi will
  work but pulls a larger conda payload (CUDA 12.8 toolkit) than uv needs.

## Quantized inference (`quantize`)

The `quantize` pixi environment pins CUDA 13 torch / triton / flashinfer /
flash-attn for quantized inference, on x86 and aarch64 (GB10 / Thor).
flashinfer carries the nvfp4 GEMM kernels. Training pins stay on the
Standard path.

```bash
pixi install -e quantize
pixi run -e quantize postinstall
```

`postinstall` builds flash-attn from a pinned main commit
(`run_scripts/install/flash_attn.sh`). It compiles sm_90/100/120 on x86 and
sm_110/120 on aarch64; set `FLASH_ATTN_CUDA_ARCHS` (e.g. `120`) to build
only the GPU you run. Both platforms need the host's CUDA 13.0 toolkit at
`/usr/local/cuda-13.0` (as in the `nvidia/cuda:13.0.0-devel` images); flashinfer
JIT-compiles against it at first use.

## Simulator environments

Simulator eval stacks are maintained as git submodules under
`external_dependencies/`, pinned to specific upstream commits. They are
optional — you only need the ones that match the benchmarks you want to
run — and most of them have their own Python environments to avoid
dependency conflicts with the main training env.

You do not need to initialise the submodules by hand. Each setup script
below pulls its own submodule at the pinned commit and, where we carry
local changes, applies the patches next to that script under
`run_scripts/eval/<bench>/patches/`. Run the setup script, not
`git submodule update` — going around it skips the patches.

### RoboCasa

```bash
bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh
```

The setup script creates a separate `robocasa_uv/.venv` under
`run_scripts/eval/robocasa_kitchen/` and installs robocasa, robosuite, and mujoco
into that venv. The robocasa eval scripts (for example
`run_scripts/eval/robocasa_kitchen/eval_robocasa.sh`) explicitly invoke
that venv's Python so the main training env is untouched.

### LIBERO

```bash
bash run_scripts/eval/libero/setup_libero.sh
```

Separate venv at `run_scripts/eval/libero/libero_uv/.venv`.

### RoboCasa365

```bash
bash run_scripts/eval/robocasa_365/setup_robocasa365.sh
```

Creates a separate `robocasa365_uv/.venv` under
`run_scripts/eval/robocasa_365/` and downloads the RoboCasa365 kitchen
assets. It also disables the `flash-attn` wheel inside that venv, which the sim client
never loads and whose ABI does not match that venv's torch. See
[`run_scripts/eval/robocasa_365/README.md`](../run_scripts/eval/robocasa_365/README.md)
for fine-tuning and evaluation.

## Developer install

For anyone editing the code rather than only consuming it, also install
the optional `dev` extra and set up the pre-commit hooks:

```bash
uv sync --extra dev
uv tool install ruff
uv tool install pre-commit
pre-commit install
```

The pre-commit config (`.pre-commit-config.yaml`) runs the same
`ruff check` and `ruff format --check` steps that CI enforces, scoped to
`rldx/`. Run them against the whole tree ad-hoc with:

```bash
pre-commit run --all-files
```

## Common pitfalls

### `flash-attn` build fails with "nvcc not found"

The Standard path downloads a prebuilt wheel; this error means none was
fetched and the install fell back to a source build, which needs the CUDA
toolkit, not just a matching driver. Make sure `nvcc --version` works on the
machine, and that the version matches the version `torch` was built
against. On RTX 5090 (Blackwell) follow the
[SM_120 install](#rtx-5090--blackwell-sm_120) — the pixi env ships
the CUDA 12.8 toolkit so this error cannot occur.

### `flash-attn` takes a long time to install

Expected only when flash-attn compiles from source: on the Standard path
when no prebuilt wheel was fetched (see above), and always on the SM_120
path. Subsequent installs reuse the cached wheel.

### `flash-attn` import fails with "no kernel image is available for execution on the device" on RTX 5090

The default `uv sync` pulled the prebuilt `flash-attn` wheel which only
ships kernels for SM_80 and SM_90. On Blackwell that wheel imports but
crashes at the first kernel launch. Switch to the
[SM_120 install](#rtx-5090--blackwell-sm_120) (pixi) — it builds
flash-attn from source with `FLASH_ATTN_CUDA_ARCHS="120"`.

### `ImportError: libcuda.so.1: cannot open shared object file`

The driver is missing or the `LD_LIBRARY_PATH` does not point at the
NVIDIA userspace libraries. On a typical Ubuntu install the fix is to
add `/usr/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH`, or to invoke the
Python interpreter inside an NVIDIA container.

### `HF_HUB_OFFLINE=1` + first-time model load

Several code paths download the Qwen3-VL backbone or the reference
RLDX-1.1 checkpoint from the Hugging Face Hub on first use. Either run the
initial pull online, or pre-download the checkpoint to
`$HF_HOME/hub/models--...` before switching to offline mode.

## Where to next

- [`training.md`](training.md) — run fine-tuning and mid-training
- [`evaluation.md`](evaluation.md) — run LIBERO / RoboCasa / GR-1 evals
- [`inference_server.md`](inference_server.md) — serve a checkpoint over ZeroMQ
- [`architecture.md`](architecture.md) — high-level model walkthrough
