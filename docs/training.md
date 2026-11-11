# Training

RLDX-1.1 has a single unified launcher — `rldx/experiment/launch_train.py` —
that covers fine-tuning and mid-training. The difference
between the two is just which dataset-selection arguments and which
base checkpoint you pass.

This doc covers the CLI surface, the three common training modes, and
where to look for canonical example scripts.

## Entry point

```bash
uv run torchrun --nproc_per_node=<N_GPUS> --master_port=<PORT> \
    rldx/experiment/launch_train.py \
        [ CLI args here ]
```

`launch_train.py` parses a single `TrainArgs` dataclass
(`rldx/configs/training/cli.py`) via
[`tyro`](https://github.com/brentyi/tyro). Plain fields are flags with
underscores turned into dashes (e.g. `global_batch_size` →
`--global-batch-size`). The data source and optional features are
subcommands whose fields carry the subcommand's prefix: `data:paths
--data.paths …`, `memory:on --memory.length …`, `wandb:on
--wandb.project …`. A flag that belongs to a subcommand you did not pick
is rejected.

The launcher internally reuses `run()` from
`rldx/experiment/experiment.py`, which handles dataset construction,
model setup, DeepSpeed initialisation, and the HuggingFace `Trainer`
main loop.

## Three training modes

### (a) Fine-tune from a base checkpoint

The common case: take a pre-trained RLDX-1.1 checkpoint and specialise it
on a single dataset.

```bash
uv run torchrun --nproc_per_node=8 rldx/experiment/launch_train.py \
    --num-gpus 8 \
    --base-model-path RLWRLD/RLDX-1.1-PT \
    data:paths --data.paths /path/to/robocasa_dataset \
    --data.embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path rldx/configs/data/robocasa_config.py \
    --video-length 4 \
    --n-cog-tokens 64 \
    --global-batch-size 64 \
    --learning-rate 1e-4 \
    --max-steps 60000 \
    --save-steps 1000 \
    --save-total-limit 5 \
    --output-dir ./ckpt/rldx/finetuned \
    --experiment-name my_run \
    wandb:on --wandb.project rldx-finetune
```

> **Choosing the base checkpoint.** Match `--video-length` to the base:
> `RLWRLD/RLDX-1.1-PT` (video input) uses `--video-length 4`, while
> `RLWRLD/RLDX-1.1-PT-IMG` (image input) uses `--video-length 1`.
> The benchmark scripts switch both together with `VARIANT=img`.

Key fine-tune flags:

| Flag | Purpose |
|---|---|
| `--base-model-path` | HF hub id or local path to the pre-trained checkpoint. Processor + config + weights are all loaded from it. |
| `data:paths --data.paths …` | LeRobot-format dataset directories; several are mixed, weighted by `--data.mix-ratios` (default equal). |
| `--data.embodiment-tag` | With `data:paths`, which embodiment slot to train. `GENERAL_EMBODIMENT` routes through the category-specific MLP's "new" head. |
| `data:mix --data.root … --data.name …` | A registered mix (`rldx/configs/data/dataset_mix.py`) under a root; the mix sets each dataset's embodiment and weight. |
| `--modality-config-path` | Python file declaring the modality (video / state / action) config for this embodiment. See [`rldx/configs/data/`](../rldx/configs/data/) for the built-ins. |

Canonical script:
[`run_scripts/train/benchmarks/finetune_robocasa.sh`](../run_scripts/train/benchmarks/finetune_robocasa.sh).

### (b) Mid-train with optional add-ons (memory / motion / physics)

Mid-training resumes from a pre-trained checkpoint and additionally
turns on one or more add-on components for supervised continuation. All
three add-ons are additive flags on top of the base fine-tune
command lines:

```bash
# Memory
memory:on \
--memory.length 4 \
--memory.n-cog-tokens 16 \
--memory.concat \
--memory.dropout-prob 0.3

# Motion module
motion:on \
--motion.insert-layer 9 \
--motion.injection-point vision_encoder \
--motion.pool-type avg

# Physics (tactile + torque)
physics:on \
--physics.keys tactile torque \
--physics.dims 30 7 \
--physics.loss-weight 0.1
```

All three can be combined in a single run. Canonical scripts:
[`run_scripts/train/examples/midtrain_droid.sh`](../run_scripts/train/examples/midtrain_droid.sh) and
[`run_scripts/train/examples/midtrain_allex.sh`](../run_scripts/train/examples/midtrain_allex.sh).

## Dataset layout: `meta/modality.json`

Beyond the LeRobot-format parquet/video files, every dataset RLDX-1.1
trains on must carry a `meta/modality.json` describing how to slice
the flat state/action vectors into named joint groups and how to
remap video columns to modality keys. This file is loaded by
[`rldx/data/dataset/lerobot_episode_loader.py`](../rldx/data/dataset/lerobot_episode_loader.py)
on every dataset open; without it the launcher stops at startup with
`FileNotFoundError`.

```
my_dataset/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/
│   └── chunk-000/
│       └── observation.images.<cam>/episode_000000.mp4
└── meta/
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── info.json
    ├── modality.json        ← required
    ├── stats.json
    └── tasks.jsonl
```

Schema (`{modality}.{key}.start/end` for state and action; video uses
`original_key` to remap a parquet column to a logical name):

```json
{
    "state": {
        "right_arm_joints": { "start": 0,  "end": 7 },
        "left_arm_joints":  { "start": 7,  "end": 14 },
        "neck_joints":      { "start": 44, "end": 46 }
    },
    "action": {
        "right_arm_joints": { "start": 0,  "end": 7 },
        "left_arm_joints":  { "start": 7,  "end": 14 },
        "neck_joints":      { "start": 44, "end": 46 }
    },
    "video": {
        "camera_ego_left":  { "original_key": "observation.images.camera_ego_left" },
        "camera_ego_right": { "original_key": "observation.images.camera_ego_right" }
    }
}
```

The slice ranges in `state` and `action` must match the joint groups
declared in the Python `MODALITY_CONFIGS` entry you pass via
`--modality-config-path`. The dataset's flat `observation.state` /
`action` parquet columns are sliced according to these ranges and
re-keyed under the joint-group name before reaching the model.

For the embodiment slot itself (which selects which Python
`MODALITY_CONFIGS` entry is active), see
[`embodiment_tags.md`](embodiment_tags.md).

## Image pipeline flags

See [`architecture.md`](architecture.md#stage-1--image-pipeline) for the
geometry. The three knobs are:

```bash
--image-max-area 65536       # area budget, default 256*256
--image-resize-m 32          # alignment multiple
--random-crop-fraction 0.9   # None (default) = no-op
--random-rotation-angle 5    # optional train-only rotate
--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08  # this is the default
```

`--random-crop-fraction=None` is the production default — the refactor
confirmed that the current datasets are 256×256 and the crop is
skipped. Set it only when you want extra augmentation.

## Training knobs cheat-sheet

| Flag | Default | Notes |
|---|---|---|
| `--num-gpus` | 1 | Also needs matching `torchrun --nproc_per_node`. |
| `--global-batch-size` | 64 | Divided by `num_gpus`; effective batch = `global_batch_size * gradient_accumulation_steps`. |
| `--gradient-accumulation-steps` | 1 | |
| `--learning-rate` | 1e-4 | Warmup via `--warmup-ratio` (default 0.05). |
| `--lr-scheduler-type` | cosine | |
| `--weight-decay` | 1e-5 | |
| `--max-steps` | 30000 | Total training steps. |
| `--save-steps` | 1000 | Checkpoint cadence. |
| `--save-total-limit` | 5 | Rolling window of saved checkpoints. |
| `--save-only-model` | off | Weights only — no optimizer/scheduler/RNG state. Much smaller checkpoints (measured 61 GB → 13 GB on a 2-GPU ZeRO-2 run), but they cannot be resumed from. Must be paired with `--no-resume-from-checkpoint`. |
| `--resume-from-checkpoint` | **on** | Resume from the newest `checkpoint-*` in the run directory `{output_dir}/{experiment_name}` (`--experiment-name` defaults to `debug`). On by default so a requeued or preempted SLURM job picks its run back up; pass `--no-resume-from-checkpoint` to start fresh against a used directory. |
| `--dataloader-num-workers` | 2 | Workers per rank. |
| `--dataset-weight-by-steps` | off | Multiply each dataset's weight by its step count before normalising, so datasets are sampled in proportion to their length and `--data.mix-ratios` become multipliers on that share. Off, the weights are used as given regardless of length. |
| `--override-pretraining-statistics` | off | Normalize with this run's dataset statistics even for tags `--base-model-path` already has statistics for. Off, the checkpoint's are kept: missing joint groups fail at startup, differing values warn. |
| `--allow-dim-mismatch` | off | Accept a modality config wider than the checkpoint's `max_state_dim` / `max_action_dim` instead of failing. The affected projector weights are re-initialised for every slot; ease them in with `--new-param-warmup-steps`. |
| `--new-param-warmup-steps` | 0 | For the first N steps, train only the parameters the checkpoint did not supply (new add-ons, re-initialised weights). |
| `--episode-boundary-policy` | `MASK` | What to do with the action-chunk positions that run past the end of an episode — the last `action_horizon - 1` steps of every episode have them. `MASK` keeps them out of the loss (the shipped behaviour). `PAD` trains on the padded targets instead, which read as "stay put", so the policy learns to hold still once the episode is over. Pass the member name, as with `--data.embodiment-tag`. Recorded in the checkpoint; a resume under a different value is refused. |
| `--validation` | none | Validation spec JSON files, given after one flag (`--validation a.json b.json`); a repeated flag keeps only the last. Each spec names its datasets, frozen samples or every episode, cadence, memory, comparison space and metrics, and its frozen samples are held out of training. See [Validation during training](#validation-during-training) for the format. |
| `wandb:on` + `--wandb.project` | off | `WANDB_PROJECT` env var is also honored. |
| `--experiment-name` | "debug" | Run name on wandb and for the output directory. |
| `--skip-weight-loading` | off | Smoke test: build the architecture, skip reading `--base-model-path`'s weights. The processor still loads from the base model and the VLM backbone still loads its own pretrained weights, so the data and construction paths run for real. Use a fresh `--output-dir` — training resumes from any checkpoint already there, which would resume into a model that was never given its weights. |

### Checkpoint overrides

These flags default to None, which keeps the base checkpoint's value, or
the model default when training from scratch. `--random-crop-fraction`
and `--random-rotation-angle` belong here too; see
[Image pipeline flags](#image-pipeline-flags).

| Flag | What it overrides |
|---|---|
| `--backbone-path` | Backbone VLM, a HF hub id or a local checkpoint. From scratch the default is `Qwen/Qwen3-VL-8B-Instruct`. |
| `--backbone-select-layer` | The LLM layer the backbone is truncated at; layers above it are dropped. |
| `--max-num-embodiments` | Projector width, i.e. how many embodiment slots the model carries. A value other than the model default is written to `config.json`, so evaluation rebuilds the same width. |
| `--use-percentiles` | State/action normalisation: `True` = q01/q99, `False` = min/max. |

## Tuning which parameters are trained

Defaults reflect the refactor decision: backbone frozen below, action
head fully tuned.

| Flag | Default | What it controls |
|---|---|---|
| `--tune-llm` | False | Train the entire backbone VLM. |
| `--tune-visual` | False | Train the vision tower. |
| `--tune-top-llm-layers` | 4 | When `--tune-llm` is off, train just the top N LM layers. |
| `--tune-projector` | True | Per-embodiment state/action encoders and action decoder in the action head. Keep on for `general_embodiment` / `new_embodiment` unless the base checkpoint already trained this robot under that tag. |
| `--tune-diffusion-model` | True | MSAT action model. |
| `--freeze-cog-tokens` | False | Freeze the learnable cognition-token embeddings. |
| `--state-dropout-prob` | 0.3 | Stochastic state dropout for regularisation. |
| `--state-dropout-prob-final` | None | If set, state dropout is annealed from `--state-dropout-prob` to this value over `--max-steps`. None keeps it constant. |
| `--state-dropout-schedule` | cosine | Annealing curve, `cosine` or `linear`. Ignored without `--state-dropout-prob-final`. |
| `--attn-dropout` | 0.0 | Attention-map dropout in the MSAT action model, applied in training only. |
| `--general-embodiment-train-ratio` | 0 | Probability that a training sample is routed through the `general_embodiment` projector instead of its own tag's, so that projector keeps receiving gradient. 0 disables it. |

### LoRA fine-tuning

For memory-constrained fine-tunes you can replace full-parameter
training of either the action model (MSAT) or the backbone VLM
with [PEFT](https://github.com/huggingface/peft) LoRA adapters.
Both surfaces are independent; you can enable one or both.

```bash
# Action-model LoRA — replaces full MSAT tuning
action-model-lora:on \
--action-model-lora.rank 16 \
--action-model-lora.alpha 32

# Backbone LoRA — replaces top-N backbone VLM layer tuning
backbone-lora:on \
--backbone-lora.rank 16 \
--backbone-lora.alpha 32 \
--backbone-lora.num-layers -1   # -1 = all layers, N>0 = top-N suffix
```

| Flag | Default | What it controls |
|---|---|---|
| `action-model-lora:on` | off | Inject LoRA into MSAT QKV / projection / FFN linears. Only the adapters train in MSAT, so `--tune-diffusion-model` is not given with it. |
| `--action-model-lora.rank` | 16 | Adapter rank `r`. |
| `--action-model-lora.alpha` | 32 | Scaling factor `α`. |
| `--action-model-lora.dropout` | 0.0 | Must stay 0: the frozen MSAT the adapters sit in runs in eval mode. |
| `backbone-lora:on` | off | Inject LoRA into the backbone VLM's attention + MLP projections. The backbone is frozen and only the adapters train, so `--tune-llm`, `--tune-visual`, `--tune-top-llm-layers` and `--freeze-cog-tokens` are not given with it. |
| `--backbone-lora.rank` | 16 | Adapter rank. |
| `--backbone-lora.alpha` | 32 | Scaling factor. |
| `--backbone-lora.num-layers` | -1 | `-1` = inject into all backbone VLM layers; `N > 0` = inject into the top `N` layers only. |
| `--backbone-lora.dropout` | 0.0 | Must stay 0: the frozen backbone the adapters sit in runs in eval mode. |

The wrapped MSAT linears are `ActionModelLoraConfig.target_modules` in
`rldx/configs/model/rldx.py`; targets that don't exist in the current MSAT (e.g. `p_qkv` without physics) are filtered before the PEFT call.

The launcher rejects a LoRA combined with a tune flag of the part it
covers (`ValueError`), so a stale config file does not silently produce a
no-op run.

## Real-Time Chunking (training-time)

Training-time RTC (the `t_tok=1` trick from
[Black et al. (Training-Time Action Conditioning for Efficient Real-Time Chunking)](https://arxiv.org/abs/2512.05964)) bakes
clean-prefix conditioning directly into the model. At inference time
the prefix is just inpainted at `τ=1` instead of running an extra
guidance pass, so the chunk-boundary stitching incurs no latency
overhead.

```bash
--rtc-training-max-delay 4
```

When `--rtc-training-max-delay > 0` each training step samples a
per-sample prefix length `d ~ U{0, …, max_delay}`; those positions are
conditioned on ground-truth clean actions and are excluded from the
loss. A checkpoint trained this way enables `rtc:trained`
on the server. A checkpoint
trained with `--rtc-training-max-delay 0` (the default) can still serve
RTC by using `rtc:guided` at inference time, but the
guided path is slower.

See [`inference_server.md`](inference_server.md#real-time-chunking-rtc)
for the corresponding inference-time flags and the
mode/compile compatibility matrix.

## Validation during training

`--validation spec.json` scores the policy's predicted action chunks
against the dataset's actions every `every` optimizer steps. Several specs go
after one flag (`--validation a.json b.json`). A minimal spec:

```json
{
  "name": "val",
  "every": 1000,
  "memory": "reset",
  "space": "normalized",
  "datasets": {
    "general_embodiment": {"path": "/data/my_robot_lerobot"}
  },
  "samples": [
    {"tag": "general_embodiment", "episode_id": 12, "episode_length": 180,
     "frame_starts": [0, 16, 32, 48]}
  ],
  "metrics": {"all": {}}
}
```

| Field | Meaning |
|---|---|
| `name` | Names the outputs; must be unique across the run's specs. |
| `every` | Cadence in optimizer steps. |
| `memory` | `reset` predicts every chunk with fresh memory; `episode` carries memory across an episode's chunks, which then must be exactly `action_horizon` frames apart. |
| `space` | `normalized` compares in the training representation; `raw` decodes to robot units, so each metric may cover one embodiment only. |
| `datasets` | Keyed by embodiment tag value (`general_embodiment`, not `GENERAL_EMBODIMENT`). `path` is the LeRobot dataset; the optional `metadata_sha256` maps `meta/` file names to digests, and the run refuses to start if they changed. |
| `samples` | Optional. The episodes to score and the frames each chunk starts at (`task` optional). |
| `metrics` | Name → optional `tags` (datasets to average over; default all) and `keys` (action modality keys; default all). |
| `provenance` | Optional free-form record of how the samples were chosen. |

With `samples`, the dataset must be one the run trains on, and the listed
episodes are held out of training. Without `samples`, every episode is
scored in back-to-back chunks of `action_horizon` frames, so the dataset
must not be one the run trains on. A spec that breaks either rule, names a tag without a modality
config or a metric key the embodiment lacks is refused before training
starts. A run whose checkpoints held out a different set of episodes
refuses to resume.

[`rldx/experiment/validation/selection.py`](../rldx/experiment/validation/selection.py)'s
`select_samples` picks a task-balanced set of samples from a dataset's
metadata alone.

Each evaluation writes
`<output_dir>/validation/<name>/step_<NNNNNNNN>/metrics.json` and one
`.npz` of predictions and ground truth per episode, and logs
`<name>/<metric>_L1` and `<name>/<metric>_MSE` to wandb.

## Checkpoint format

After every `save_steps`, the `CheckpointFormatCallback` writes the
trainer output plus:

```
{output_dir}/{experiment_name}/checkpoint-{step}/
├── config.json                   # model_type="RLDX-1", see rldx/configs/model/rldx.py
├── model-00001-of-00003.safetensors
├── ...
├── experiment_cfg/               # conf.yaml
├── processor/                    # <-- subdir layout required for inference
│   ├── processor_config.json
│   ├── embodiment_id.json
│   └── statistics.json
├── wandb_config.json
└── train.log
```

Load a checkpoint for serving with `rldx.serving.policy.load_policy`. The processor
takes its settings from the checkpoint's `config.json`, so it loads together with
it: `RLDXProcessor.from_pretrained(ckpt, rldx_config=RLDXConfig.from_dict(...))`.

Alongside those files the trainer also writes the optimizer, scheduler and RNG
state (and, under DeepSpeed, a `global_step*/` directory) — that is what makes
a checkpoint resumable, and it dominates the on-disk size. `--save-only-model`
drops all of it; see the cheat-sheet above.

## Troubleshooting

### "RuntimeError: CUDA out of memory"
- Lower `--global-batch-size` and raise `--gradient-accumulation-steps` by the same factor; the effective batch stays the same.
- Drop `--tune-top-llm-layers` to 0 to freeze more of the backbone.

### "Can't find a valid checkpoint at {output_dir}/checkpoint-N"
A DeepSpeed resume found the checkpoint directory but not the engine state
inside it — the checkpoint was written with `--save-only-model`. Those
checkpoints cannot be resumed from; that is the whole trade. Start fresh with
`--no-resume-from-checkpoint`, or drop `--save-only-model` for runs you intend
to resume. Launching with both set is rejected up front by `warn_configs`.

### "[!] jemalloc not found: ..."
The training script did not find jemalloc and is training with glibc malloc.
The run is unaffected, but on long runs host RSS can creep up as DataLoader
workers fragment the heap. Install `libjemalloc2` or set `JEMALLOC_PATH`; see
[`installation.md`](installation.md#prerequisites).

### "[i] Memory enabled: length=…" is not what you expected
`memory:on` defaults to `--memory.length 4` (`--memory.n-cog-tokens` and
`--memory.concat` default to 16 and on); pass `--memory.length` to change it.
Double-check that the base model actually supports the memory path (all current RLDX-1 pre-training runs do).

## Where to next

- [`installation.md`](installation.md) — environment setup
- [`evaluation.md`](evaluation.md) — run the trained checkpoint on benchmarks
- [`inference_server.md`](inference_server.md) — serve checkpoints over ZeroMQ
- [`architecture.md`](architecture.md) — what the config flags are actually wiring up
