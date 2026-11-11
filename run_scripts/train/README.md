# RLDX Training Scripts

The default `BASE_MODEL_PATH` for fine-tune scripts is
`RLWRLD/RLDX-1.1-PT`; override with `BASE_MODEL_PATH=<path>` to start from a
different base.

All four benchmark scripts also take
`VARIANT=img`, which reproduces the image-input `RLWRLD/RLDX-1.1-FT-*-IMG`
checkpoints: it starts from `RLWRLD/RLDX-1.1-PT-IMG` with `--video-length 1`
and without `motion:on`.

## Directory layout

| Directory | What's inside |
|---|---|
| [`benchmarks/`](benchmarks/) | One script per benchmark — the recipes that produced each released `RLWRLD/RLDX-1.1-FT-*` checkpoint (LIBERO, GR-1, RoboCasa, RoboCasa365). |
| [`examples/`](examples/) | Mid-training recipes (DROID, ALLEX), plus a [`finetune.sh`](examples/finetune.sh) template to copy + edit for fine-tuning on your own custom dataset. |

## Per-benchmark fine-tune

The benchmark guides under [`run_scripts/eval/<bench>/README.md`](../eval/)
each link to the matching script in `benchmarks/`. As a quick
reference:

| Benchmark | Script |
|---|---|
| LIBERO | `benchmarks/finetune_libero.sh` |
| GR-1 Tabletop | `benchmarks/finetune_gr1.sh` |
| RoboCasa Kitchen | `benchmarks/finetune_robocasa.sh` |
| RoboCasa365 | `benchmarks/finetune_robocasa365.sh` |

## Fine-tune on your own dataset

Copy [`examples/finetune.sh`](examples/finetune.sh), edit dataset paths
and modality config to match your data, and toggle motion / memory /
physics add-ons by uncommenting lines in the `ARGS=(...)` block. See
[`examples/README.md`](examples/) for the full add-on flag reference and
multi-dataset usage.
