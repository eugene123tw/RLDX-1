# GR-1 Tabletop

GR-1 humanoid tabletop manipulation (24 tasks: 18 object rearrangement +
6 articulated object). Ego-centric camera, 256x256 resolution.

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1.1-FT-GR1`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-GR1) |
| Reported success rate | 60.3 % (24-task average) |
| Simulator venv | `run_scripts/eval/gr1_tabletop/robocasa_uv/.venv` |

## 1. Setup (one-time per clone)

```bash
bash run_scripts/eval/gr1_tabletop/setup_gr1.sh
```

Checks out the simulator submodule at its pinned commit and applies the
patches under `run_scripts/eval/gr1_tabletop/patches/`:

- `tabletop_24dc_determinism.patch` — replaces the process-randomized `hash()`
  task seed and two set-ordering dependencies with deterministic equivalents.
- `01-robocasa-reset-settle.patch` — the reset protocol's extended settle, read
  through `GR1_RESET_SETTLE_S`.

Both edit the submodule rather than this repo, so `git pull` does not deliver
them. The settle patch is also applied by `eval_gr1.sh` itself when
`GR1_PROTOCOL=on` (guarded by a marker check, so the two paths cannot
double-apply) — the protocol therefore works on a fresh clone without this
step, but setup is still what provides the venv and the assets.

## 2. Fine-tune from RLDX-1.1-PT

```bash
DATA_ROOT=/path/to/GR00T-X-Embodiment-Sim \
bash run_scripts/train/benchmarks/finetune_gr1.sh
```

Defaults to `RLWRLD/RLDX-1.1-PT` as the base; override with
`BASE_MODEL_PATH=...`.

## 3. Run evaluation

```bash
bash run_scripts/eval/gr1_tabletop/eval_gr1.sh RLWRLD/RLDX-1.1-FT-GR1
```

The 24 tasks are split across 8 shards, each with its own policy server. The
runner uses every visible GPU, one shard per card at a time; `N_GPUS=<n>` (or
`CUDA_VISIBLE_DEVICES`) limits that. The card count changes only the wall-clock
-- one card runs the eight shards in turn -- not which tasks share a server.

Outputs land in `output_final/gr1_tabletop/<ckpt>/<protocol>/<task>/`,
and the runner ends by printing the pooled success rate over the task CSVs.
Each shard first checks that every one of its tasks recorded `N_EPISODES`
episodes.

`SHARD_ID=<k>` runs shard `k` alone, for a scheduler that gives each shard a
card of its own. Run all eight into the same output root; a shard run alone
prints no pooled score.

### Check the protocol actually engaged

```
[gr1-proto] settle=5.0s | manifest=gr1_episode_seed_manifest.json | patch OK
[seed-manifest] PnPWineToCabinetClose base=42 ep=3: 4200003 -> 4200011
```

Both lines are required. The `[seed-manifest]` lines alone prove only the repo
half: they appear with or without the settle, as does the `_resetproto` path
suffix, and six runs once finished looking correct with no settle at all.

### What breaks comparability

Shard count, the protocol switch, and the code base each change the numbers, so
a set has to hold all three fixed:

- **Shard count** (`N_SHARDS`, 8) decides which tasks land on which policy
  server, and with the noise draw free that makes it a different noise stream
  as well.
- **Protocol on/off** starts episodes from a different state, and 115 of 1200
  scored slots run different scenes. It is in the output path for that reason.
- **`GR1_PROTOCOL=on` needs the settle**, since the manifest was judged at
  `v4@5.0s` -- the substitutions and the run condition have to agree.

The manifest is keyed `(42 + lane, episode)` over 5 lanes, so `ENV_LAYOUT=5` and
`N_EPISODES % 5 == 0`.

The offsets it is keyed on are **-1..8** for a 10-episode lane, because the
episode counter is read before it is incremented: video `episode_k` carries
offset `k-1`. `_meta` reads `eval_offsets: -1..9 (9 dormant)`, which names
**offset 9** as dormant rather than counting nine entries -- upstream keeps
those rows so the table stays correct if the numbering convention ever changes
([upstream `docs/PROTOCOL.md`](https://github.com/huiwon-jang/gr1-tabletop-eval-protocol/blob/main/docs/PROTOCOL.md),
not ported here). A run still logs substitutions at offset 9: the rollout
resets once more after the last episode, and that reset looks up the manifest
but records no video: all 130 entries apply, and the 15 at offset 9 land on
that trailing reset rather than on a scored slot. Lane count is independent of
the shard count -- the scene a key names is the same at any number of shards.

### Per-checkpoint width

`config.json` now always records `max_num_embodiments`, and one without the
key loads as 36, the default it was saved under. A checkpoint saved by code
whose default was something else can still disagree with its own weights.
Read the real width off the weights rather than guessing:

```python
# action_model.action_decoder.layer1.W -> [width, 1024, 1024]
```

If it differs from what `config.json` says, point the eval at a directory whose `config.json` is a real
file carrying `max_num_embodiments`, with everything else symlinked to the
checkpoint.
