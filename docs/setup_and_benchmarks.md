# RLDX-1: End-to-End Setup & Benchmark Guide

This document walks through a clean install of RLDX-1 from scratch — the
main training/inference env, every simulator sub-environment, and the
runnable command for each benchmark. It also captures the submodule and
external-dependency fixes we landed during onboarding so you don't have
to rediscover them.

If you only want the bare-minimum install or a single benchmark, the
existing [`installation.md`](installation.md) and the per-benchmark
READMEs under [`run_scripts/eval/`](../run_scripts/eval/) are the source
of truth. This guide consolidates them and adds the workarounds we hit.

---

## 0. Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Linux x86_64 | — | Only platform exercised |
| Python | `3.10.*` | Pinned by `pyproject.toml` and every sim venv |
| CUDA toolkit | 12.x | `flash-attn==2.7.4.post1` builds against `nvcc` |
| NVIDIA driver | supports CUDA 12 | For `torch==2.7.0` + `flash-attn` |
| [`uv`](https://github.com/astral-sh/uv) | `>= 0.8.4` | Resolver/venv for the main env and every sim venv |
| [`pixi`](https://pixi.sh) | `>= 0.40` | Only needed for RTX 5090 / Blackwell (`sm_120`) |
| `git`, `git-lfs` | recent | Submodules and HF LFS assets |
| `xvfb` (or `MUJOCO_GL=egl`) | — | Headless rendering for sim evals |

```bash
# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install pixi only on Blackwell
curl -fsSL https://pixi.sh/install.sh | sh

# Install git-lfs only if GR00T-WholeBodyControl is on your list
sudo apt-get install -y git-lfs && git lfs install
```

---

## 1. Clone & set up the main RLDX env

```bash
git clone https://github.com/RLWRLD/RLDX-1.git
cd RLDX-1
```

### 1a. Standard install (most GPUs)

```bash
uv sync --python 3.10
uv pip install -e .
```

`uv sync` materialises `.venv/` against `pyproject.toml`, including the
`[tool.uv.extra-build-dependencies]` override that pins
`torch==2.7.0 + numpy==1.26.4` while building `flash-attn`.

### 1b. RTX 5090 / Blackwell (`sm_120`) install

Upstream `flash-attn` ships no `sm_120` wheels. The pixi env supplies
the CUDA 12.8 toolkit through conda and compiles `flash-attn` from
source with `TORCH_CUDA_ARCH_LIST=sm_120` baked in.

```bash
pixi install
pixi run --environment rldx postinstall
```

The first run takes 10–20 min for the `flash-attn` source build.

### 1c. Smoke test

```bash
# Standard path
uv run python -c "import rldx; print(rldx.__version__)"

# Pixi path
pixi run --environment rldx python -c "import rldx; print(rldx.__version__)"
```

Both should print `0.1.0`. From here on, "Standard install" commands use
`uv run …`; the pixi equivalent is `pixi run --environment rldx …`.

---

## 2. External dependencies

Despite the name and the `.gitmodules` file, the entries under
[`external_dependencies/`](../external_dependencies/) are **not all real
submodules**. The current state of `HEAD` is:

| Path | How it's tracked | What to do |
|---|---|---|
| `LIBERO` | **Vendored** (plain `tree` in `HEAD`) | Already present after `git clone`. No init needed. |
| `robocasa` | **Vendored** (plain `tree` in `HEAD`) | Already present. No init needed. |
| `robocasa-gr1-tabletop-tasks` | **Vendored** (plain `tree` in `HEAD`) | Already present. No init needed. |
| `robocasa365` | **Real submodule** (gitlink) | `git submodule update --init external_dependencies/robocasa365` |
| `SimplerEnv` | **Orphaned `.gitmodules` entry** — declared but **no gitlink in `HEAD`** | Clone manually: `git clone https://github.com/allenzren/SimplerEnv.git external_dependencies/SimplerEnv` |
| `GR00T-WholeBodyControl` | **Orphaned `.gitmodules` entry** — declared but no gitlink | Clone manually: `git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git external_dependencies/GR00T-WholeBodyControl` |

A blanket `git submodule update --init --recursive` only brings in
`robocasa365`; the orphaned entries silently no-op. The per-sim setup
scripts under `rldx/eval/sim/<NAME>/setup_<NAME>.sh` already do the
right thing for each path:

- They call `git submodule update --init <path>` (a no-op on the
  orphaned and vendored entries — harmless).
- For SimplerEnv specifically, `setup_SimplerEnv.sh` detects whether
  `external_dependencies/SimplerEnv` is populated; if it is, it then
  recurses into it to init the **inner** `ManiSkill2_real2sim`
  submodule (which is a real submodule of the SimplerEnv repo, pointing
  at the allenzren fork).

### Bring-up sequence we actually use

```bash
# 1. Real submodule
git submodule update --init external_dependencies/robocasa365

# 2. Manual clones for the orphaned entries (only the ones you need)
git clone https://github.com/allenzren/SimplerEnv.git \
    external_dependencies/SimplerEnv
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git \
    external_dependencies/GR00T-WholeBodyControl

# 3. Vendored entries (LIBERO, robocasa, robocasa-gr1-tabletop-tasks)
#    are already present — no action.
```

If you only run one benchmark, skip steps 2/3 for paths you don't need
and just run that benchmark's `setup_<bench>.sh` — it'll surface a
clear error if a required clone is missing.

### Per-submodule fixes & gotchas

These are issues that bit us during onboarding; the per-sim setup
scripts have been updated to handle each one. If you are pulling from a
fork or working around a network restriction, you may still hit them.

| Path | Issue | Workaround |
|---|---|---|
| `SimplerEnv` | `.gitmodules` declares it but `HEAD` carries no gitlink, so submodule init silently no-ops. | Clone manually from `allenzren/SimplerEnv` (see above). The allenzren fork is required, not upstream `simpler-env/SimplerEnv`, because it adds end-effector proprio (`eef_pos`) used by our env adapter — see [`simpler_eval_fixes.md`](simpler_eval_fixes.md). |
| `SimplerEnv/ManiSkill2_real2sim` | Inner submodule whose `.gitmodules` points to `allenzren/ManiSkill2_real2sim`. Won't be init'd by anything operating at the RLDX-1 repo level. | `setup_SimplerEnv.sh` recurses into the cloned SimplerEnv directory and runs `git submodule update --init --recursive`. If you cloned SimplerEnv manually, that recursive init still happens via the setup script. |
| `GR00T-WholeBodyControl` | Same orphan-gitlink issue as SimplerEnv. Also: LFS-backed assets, and the in-repo `gr00trobosuite` is a placeholder that the setup replaces with a specific branch clone. | Clone manually (see above). `setup_GR00T_WholeBodyControl.sh` then requires `git-lfs`, runs `git lfs pull`, and clones [`xieleo5/robosuite@leo/support_g1_locomanip`](https://github.com/xieleo5/robosuite/tree/leo/support_g1_locomanip) into `gr00t_wbc/dexmg/gr00trobosuite`. |
| `robocasa` (vendored) | Gymnasium 0.29's `SyncVectorEnv` forwards 64-bit seeds that numpy's legacy seeding rejects. | `setup_robocasa.sh` applies `seed_clamp_64bit.patch` from [`run_scripts/eval/robocasa_kitchen/patches/`](../run_scripts/eval/robocasa_kitchen/patches/). |
| `robocasa-gr1-tabletop-tasks` (vendored) | Same 64-bit seed issue; sim venv also ends up with a `flash-attn` wheel mismatched against its torch. | `setup_gr1.sh` applies determinism patches and renames `flash_attn*` site-packages so `transformers` falls back to SDPA. |
| `robocasa365` (real submodule) | Independent fork of `robocasa/robocasa` for the extended kitchen tasks; separate venv. | `setup_RoboCasa365.sh` initialises it and downloads kitchen assets non-interactively. |

### Why each sim has its own venv

The sim environments pin **different `torch`, `numpy`, and `gymnasium`
versions** than the main training env (e.g. `torch==2.5.1`,
`numpy==1.26.4`, `gymnasium==0.29.1`) and several of them are
incompatible with each other. The split avoids dependency conflicts and
lets the main env serve the model over ZeroMQ while the sim venv runs
only the rollout client.

---

## 3. Per-benchmark setup & run

The runtime split for every sim benchmark is the same:

1. **Main env (`.venv/`) launches the model server** —
   `uv run python rldx/eval/run_rldx_server.py --model-path … --port N`.
2. **Sim venv (`<sim>_uv/.venv/`) launches the rollout client** —
   `<sim venv python> rldx/eval/rollout_policy.py --policy_client_host 127.0.0.1 --policy_client_port N --env_name …`.

The wrapper scripts under `run_scripts/eval/<bench>/eval_*.sh` handle
both halves: they start the server, wait for the port to bind, run the
rollouts, and clean up on exit. Outputs land under
`output_final/<bench>/<ckpt-tag>/<task>/`.

### 3.1 LIBERO (40 tasks, Franka Panda)

| | |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| Checkpoint | [`RLWRLD/RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) |
| Sim venv | `rldx/eval/sim/LIBERO/libero_uv/.venv` |

Setup:

```bash
bash run_scripts/eval/libero/setup_libero.sh
```

Run:

```bash
bash run_scripts/eval/libero/eval_libero.sh \
    libero_release \
    RLWRLD/RLDX-1-FT-LIBERO
```

Arguments: `<run_label> <MODEL_PATH> [GPU_ID] [MAX_PARALLEL]`. Throttles
to `MAX_PARALLEL` concurrent task rollouts against a single server.

### 3.2 LIBERO-Plus (10,300 perturbation variants)

| | |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| Checkpoint | [`RLWRLD/RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) (reuses LIBERO) |
| Sim venv | `rldx/eval/sim/LIBERO_PLUS/libero_plus_uv/.venv` |
| Asset source | [`Sylvest/LIBERO-plus`](https://huggingface.co/datasets/Sylvest/LIBERO-plus) (6.4 GB) |

Setup:

```bash
bash run_scripts/eval/libero_plus/setup_libero_plus.sh
```

Run:

```bash
LIBERO_PLUS_DATA_DIR=/path/to/libero_plus \
bash run_scripts/eval/libero_plus/eval_libero_plus.sh \
    RLWRLD/RLDX-1-FT-LIBERO
```

Optional second positional argument restricts to a single LIBERO suite
(e.g. `libero_10`).

### 3.3 SimplerEnv (Google Fractal + WidowX Bridge)

| Variant | Embodiment | Checkpoint |
|---|---|---|
| `google_vm` (Visual Matching) | `OXE_FRACTAL` | [`RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE) |
| `google_va` (Variant Aggregation, 28 envs) | `OXE_FRACTAL` | same |
| `widowx` | `OXE_BRIDGE_ORIG` | [`RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX) |

Sim venv: `rldx/eval/sim/SimplerEnv/simpler_uv/.venv`.

Setup:

```bash
bash run_scripts/eval/simpler/setup_simpler.sh
```

The setup also installs the full set of rldx import-time deps (tyro,
einops, peft, diffusers, accelerate, deepspeed, …) into the sim venv
and pins `transformers==4.57.0` — required because
`rldx.model.modules.backbone.modeling_qwen3_vl` imports
`transformers.masking_utils`, added in 4.55+. See
[`simpler_eval_fixes.md`](simpler_eval_fixes.md) for the full list of
fixes (state/action key packing, env action_space declarations, removed
`--num-inference-timesteps` flag, etc.).

Run:

```bash
# Visual Matching
bash run_scripts/eval/simpler/eval_simpler.sh google_vm

# Variant Aggregation (4 tasks × 7 visual variants = 28 envs)
bash run_scripts/eval/simpler/eval_simpler.sh google_va

# WidowX Bridge
bash run_scripts/eval/simpler/eval_simpler.sh widowx
```

Second positional arg is `MODEL_PATH` and defaults to the matching
released checkpoint per variant.

### 3.4 RoboCasa Kitchen (24 tasks, PandaOmron)

| | |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| Checkpoint | [`RLWRLD/RLDX-1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) |
| Sim venv | `rldx/eval/sim/robocasa/robocasa_uv/.venv` |

Setup:

```bash
bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh
```

Applies the seed-clamp patch, disables the stale `flash-attn` in the sim
venv (so `transformers` falls back to SDPA), and downloads kitchen
assets.

Run:

```bash
bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh \
    RLWRLD/RLDX-1-FT-ROBOCASA
```

Parallel 4-GPU runner: 24 tasks sharded across `N_GPUS=4` (6 tasks per
GPU), 50 episodes per task.

### 3.5 GR-1 Tabletop (24 humanoid tasks)

| | |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` (the table says `GR1`; the eval script passes `GENERAL_EMBODIMENT`) |
| Checkpoint | [`RLWRLD/RLDX-1-FT-GR1`](https://huggingface.co/RLWRLD/RLDX-1-FT-GR1) |
| Sim venv | `rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv` |

Setup:

```bash
bash run_scripts/eval/gr1_tabletop/setup_gr1.sh
```

Same `flash-attn` disable and seed-clamp patches as RoboCasa Kitchen.

Run:

```bash
bash run_scripts/eval/gr1_tabletop/eval_gr1.sh RLWRLD/RLDX-1-FT-GR1
```

Sequential loop over the 24 tasks (`N_EPISODES=50` per task) against a
single server on `PORT=20100`.

### 3.6 RoboCasa365 (target50, household)

| | |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| Checkpoint | unreleased (tracked separately) |
| Sim venv | `rldx/eval/sim/robocasa365/robocasa365_uv/.venv` |

Setup:

```bash
bash run_scripts/eval/robocasa_365/setup_robocasa365.sh
```

Run:

```bash
bash run_scripts/eval/robocasa_365/eval_robocasa365.sh \
    --model-path <RC365 checkpoint or HF repo> \
    --task-set target50 \
    --split target
```

Key flags: `--task-set {atomic_seen|composite_seen|composite_unseen|target50}`,
`--split {pretrain|target}`, `--n-episodes`, `--n-envs`,
`--n-action-steps`, `--num-shards / --shard-index` for sharded parallel
runs.

### 3.7 GR00T Whole-Body Control (BEHAVIOR / locomanip)

| | |
|---|---|
| Sim venv | `rldx/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv` |
| LFS required | Yes |

Setup:

```bash
bash rldx/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

Notes: needs `git-lfs`, pulls a large asset payload, then replaces the
in-repo `gr00trobosuite` placeholder with the
`xieleo5/robosuite@leo/support_g1_locomanip` branch. Wrapper eval
scripts live under `run_scripts/eval/` per task and follow the same
server + rollout-client pattern.

---

## 4. Repro recipe — all benchmarks against a single checkpoint

For a single fine-tuned checkpoint that covers multiple benchmarks
(rare; usually each row uses a benchmark-specific checkpoint), the
sequence is:

```bash
# One-time: build every sim venv you care about
bash run_scripts/eval/libero/setup_libero.sh
bash run_scripts/eval/libero_plus/setup_libero_plus.sh
bash run_scripts/eval/simpler/setup_simpler.sh
bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh
bash run_scripts/eval/gr1_tabletop/setup_gr1.sh
bash run_scripts/eval/robocasa_365/setup_robocasa365.sh

# Per checkpoint, run only the benchmarks it was fine-tuned for.
# Each script starts/stops its own model server on a benchmark-specific port.
```

The full benchmark-to-checkpoint mapping is in the
[`README.md` reproducibility table](../README.md#reproducing-benchmark-results).

---

## 5. Common pitfalls

**`flash-attn` build fails with "nvcc not found".** The wheel compiles
on first install and needs `nvcc`, not just a matching driver. Verify
`nvcc --version`. On Blackwell, use the pixi path.

**`flash-attn` import fails with "no kernel image is available …" on
RTX 5090.** Upstream wheels ship only `sm_80/sm_90`. Switch to the pixi
install which builds with `TORCH_CUDA_ARCH_LIST=sm_120`.

**Sim venv fails with `ModuleNotFoundError: flash_attn` after RoboCasa /
GR-1 setup.** Expected — the setup script renames `flash_attn*` in the
sim venv site-packages so `transformers` falls back to SDPA, avoiding an
ABI mismatch with the sim-venv torch. The model still runs flash-attn
in the main env that serves it over ZeroMQ.

**SimplerEnv directory empty / `setup_SimplerEnv.sh` says nothing to
init.** `external_dependencies/SimplerEnv` is declared in
`.gitmodules` but has no gitlink in `HEAD`, so `git submodule update
--init external_dependencies/SimplerEnv` silently does nothing. Clone
the fork manually:

```bash
git clone https://github.com/allenzren/SimplerEnv.git \
    external_dependencies/SimplerEnv
```

Then re-run the setup script. The same applies to
`external_dependencies/GR00T-WholeBodyControl`.

**SimplerEnv inner submodule empty.** Once `SimplerEnv` itself is
present, it has its own `.gitmodules` pointing at
`allenzren/ManiSkill2_real2sim`. `setup_SimplerEnv.sh` handles the
recursive init; if you bypassed the script, run
`git submodule update --init --recursive` from inside
`external_dependencies/SimplerEnv`.

**GR00T-WholeBodyControl `git lfs pull` is slow / fails.** Network
limits on the LFS asset host. If you only need other benchmarks, skip
this submodule entirely — the other setup scripts don't touch it.

**`Could not find the Transformers classes you have set` at processor
load.** `AutoProcessor` registration is a side effect of `import rldx`.
Always `import rldx` (or call one of the public `rldx.policy.*` helpers)
before `AutoProcessor.from_pretrained(...)`.

**Rollouts hang at "Waiting for server".** Either the model server died
during load (check the server log: usually OOM or a checkpoint download
failure) or the rollout venv can't reach `127.0.0.1:<PORT>`. The
wrappers wait up to 120 s; if the port never binds, the server log has
the cause.

**`Seed must be between 0 and 2**32 - 1` from robocasa / GR-1.** Means
the seed-clamp patch did not apply. Re-run the setup script and inspect
the `Applying patches to robocasa…` block — `git apply --check` shows a
conflict if the patch can't be applied cleanly.

---

## 6. Where to next

- [`installation.md`](installation.md) — bare-minimum install reference.
- [`training.md`](training.md) — `launch_train.py` recipes for fine-tune
  and mid-train.
- [`evaluation.md`](evaluation.md) — shared eval mechanics (server /
  rollout split, common flags, results aggregation).
- [`inference_server.md`](inference_server.md) — server CLI, wire
  protocol, RTC and `--compile` flags.
- [`simpler_eval_fixes.md`](simpler_eval_fixes.md) — itemised SIMPLER
  schema/routing fixes (state/action packing, env action_space,
  embodiment routing, etc.).
- [`embodiment_tags.md`](embodiment_tags.md) — picking an
  `EmbodimentTag` for a custom robot.
- [`architecture.md`](architecture.md) — model walkthrough.
