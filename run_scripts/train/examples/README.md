# Training examples

Reference scripts that build on `RLWRLD/RLDX-1.1-PT`:

| Script | Purpose |
|---|---|
| [`finetune.sh`](finetune.sh) | Custom-dataset fine-tune template — vanilla video VLA on your LeRobot dataset(s). Toggle motion / memory / physics by uncommenting lines in the `ARGS` array. Takes one dataset or several (`data:paths --data.paths …`, with optional `--data.mix-ratios`). |
| [`midtrain_droid.sh`](midtrain_droid.sh) | Mid-train on the DROID dataset (memory + motion + physics enabled). |
| [`midtrain_allex.sh`](midtrain_allex.sh) | Mid-train on ALLEX humanoid data (memory + motion + physics enabled). |

The default `BASE_MODEL_PATH` for all three is `RLWRLD/RLDX-1.1-PT`;
override with `BASE_MODEL_PATH=<path>` to start from a different base.
The RLDX-1 mid-trained checkpoints `RLWRLD/RLDX-1-MT-DROID` and
`RLWRLD/RLDX-1-MT-ALLEX` still load as a base.

## Add-on CLI flags

The same flag set is used across `finetune.sh` (commented out by
default — uncomment to enable) and the mid-train scripts (always on):

```bash
# Memory: cross-clip context tokens for long-horizon manipulation
memory:on --memory.length 4 --memory.stride 16 --memory.n-cog-tokens 16 --memory.concat

# Motion: discrete action-prefix tokens injected mid-backbone
motion:on --motion.insert-layer 9

# Physics: per-step force/torque head; --physics.keys / --physics.dims
# must be aligned with your dataset's physics schema
physics:on --physics.keys torque --physics.dims 48 --physics.allow-missing

# Recommended whenever any add-on above is enabled — warms up the
# newly-initialized weights for the first N steps
--new-param-warmup-steps 2000
```

## Multi-dataset training (finetune.sh)

`finetune.sh` gives its datasets with `data:paths --data.paths …`: one
LeRobot dataset root, or several (with optional `--data.mix-ratios` for
sampling weights), all of the `--data.embodiment-tag` embodiment.

The mid-train scripts give theirs with `data:mix --data.root … --data.name …`,
where the name is a mix registered in `rldx/configs/data/` that sets each
dataset's embodiment and weight.

## Embodiment & modality config

Update `EMBODIMENT_TAG` and `MODALITY_CONFIG_PATH` in `finetune.sh` to
match your dataset. Built-in tags include `GENERAL_EMBODIMENT`,
`OXE_FRACTAL`, `OXE_BRIDGE_ORIG`, etc.; copy / edit one of the configs
under [`rldx/configs/data/`](../../../rldx/configs/data/) as a starting
point for the modality config.
