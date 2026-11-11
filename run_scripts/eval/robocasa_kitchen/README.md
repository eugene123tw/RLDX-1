# RoboCasa Kitchen (24 tasks)

Single-arm kitchen manipulation on a mobile Panda manipulator
(`PandaOmron`). 24 tasks: pick-and-place, doors, drawers, appliances.
Two fixed external cameras (left/right) + one wrist camera, 256×256.

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1.1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-ROBOCASA) |
| Reported success rate | 72.3 % |
| Simulator venv | `run_scripts/eval/robocasa_kitchen/robocasa_uv/.venv` |
| robosuite | `1.5.1` (pinned — see below) |

> **robosuite is pinned to 1.5.1 on purpose.** 1.5.2 deleted
> `MujocoXMLObject._get_geoms` and stopped dispatching it through `self`, which
> silently disabled RoboCasa's override that applies the `ObjCat` registry's
> physics to object geoms. Measured effect: objects keep the asset-baked
> `solref="0.001 1"` instead of RoboCasa's `(0.001, 2)` — critically damped
> contacts instead of overdamped, so objects bounce off plates or wedge at
> reset and PnP is unsolvable from step 0, with no error raised.
> **RoboCasa numbers produced on robosuite 1.5.2 are not comparable to these.**
> Note the version string is not a proxy: upstream master reported `1.5.1` for
> five months after the hook died. Before bumping robosuite, build an
> `MJCFObject` with `solref=(0.001, 2)` and confirm its geoms report `"0.001 2"`,
> not the asset's `"0.001 1"`.

## 1. Setup (one-time)

```bash
bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh
```

Builds the simulator venv, checks out the simulator submodule at its
pinned commit, and applies the patches under
`run_scripts/eval/robocasa_kitchen/patches/` — `seed_clamp_64bit.patch` for
`gymnasium 0.29` 64-bit seed forwarding, and
`panda_omron_action_shape.patch` for the PandaOmron action shapes our
policy emits.

It downloads the kitchen assets (~5 GB) only when they are missing or
incomplete; to fetch them again, delete
`external_dependencies/robocasa/robocasa/models/assets/{textures,generative_textures,objects/objaverse}`
and re-run.

## 2. Fine-tune from RLDX-1.1-PT

300-demo recipe — matches the technical-report number:

```bash
DATA_DIR=/path/to/robocasa_mg_gr00t_300 \
bash run_scripts/train/benchmarks/finetune_robocasa.sh
```

## 3. Run evaluation

```bash
bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh RLWRLD/RLDX-1.1-FT-ROBOCASA
```

The 24 tasks are split across 8 shards, each with its own policy server. The
runner uses every visible GPU, one shard per card at a time; `N_GPUS=<n>` (or
`CUDA_VISIBLE_DEVICES`) limits that. The card count changes only the wall-clock
-- one card runs the eight shards in turn -- not which tasks share a server.
After the shards it runs the swap chain (below), whose final report is the
pooled success rate.

Outputs land in `output_final/robocasa/<ckpt>/<task>/`. A
re-run resumes the episodes already there, so `manifest.json` records the
settings and the runner refuses a directory filled under others.

`SHARD_ID=<k>` runs shard `k` alone, for a scheduler that gives each shard a
card of its own. Run all eight into the same output root: whichever finishes
last runs the swap chain and reports the score.

### Default evaluation settings

| | |
|---|---|
| tasks x episodes | 24 x 50 = 1200 |
| lanes | 5 |
| execution horizon | 16 |
| seed | 42 |
| episode steps | 720, OpenDoubleDoor 1000 |
| denoising steps | the checkpoint's (4 uniform on every released RoboCasa checkpoint) |

Scenes are the simulator's default object split and layout/style pairs. These
are fixed in the runner; only `N_EPISODES` and `TASK_LIMIT` can be lowered, for
a smoke run. The lane count is not a throughput knob: episode seeds are
`(seed + lane) * 100000 + episode_id`, so it decides which 1200 initial scenes
are evaluated.

### The swap chain

Concurrent rendering across lanes occasionally swaps camera views between
envs, so the swap repair is part of every run. Once every shard is done the
runner continues into `swap_chain.sh`, which scans the tree, quarantines the
affected episodes, re-evaluates exactly those at one env, and repeats until
nothing is flagged and the tree is whole. A failed shard stops the chain rather
than letting it score an incomplete tree. It needs `ffmpeg` on `PATH`.
