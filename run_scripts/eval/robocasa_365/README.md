# RoboCasa365

Large-scale household manipulation benchmark extending RoboCasa Kitchen
to a broader distribution of everyday tasks. RLDX-1 scores **36.0 %** on the
[RoboCasa365 leaderboard](https://github.com/robocasa-benchmark/leaderboard):
Atomic-Seen 67.6 (18 tasks), Composite-Seen 27.9 (16), Composite-Unseen 8.5 (16).
The score is the mean over all 50 target tasks; because the splits differ in
size, it is not the plain mean of the three split scores.

The RLDX-1 technical report lists 32.1 % (67.3 / 19.0 / 5.6). It predates
RoboCasa 1.0.1, which lengthened every task horizon by 1.5×; the defaults below
follow 1.0.1 and the leaderboard.

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1.1-FT-RC365`](https://huggingface.co/RLWRLD/RLDX-1.1-FT-RC365) |
| Reported success rate | 40.2 % (target50 average) |
| Simulator venv | `run_scripts/eval/robocasa_365/robocasa365_uv/.venv` |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/robocasa_365/setup_robocasa365.sh
```

Creates the standalone simulator venv and downloads RoboCasa365 assets.

## 2. Fine-tune from RLDX-1.1-PT

```bash
DATA_ROOT=/path/to/robocasa365/v1.0/pretrain NODE_RANK=0 \
bash run_scripts/train/benchmarks/finetune_robocasa365.sh
```

The recipe runs on 2 nodes × 8 GPUs: start the same command on each node,
with `NODE_RANK=0` on the master and `NODE_RANK=1` on the other. See the
script header for the rendezvous and NCCL variables.

Override `BASE_MODEL_PATH=...` to start from a different base.

## 3. Run evaluation

```bash
bash run_scripts/eval/robocasa_365/eval_robocasa365.sh \
    --model-path <RC365 checkpoint path or HF repo> \
    --task-set target50 \
    --split pretrain
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--model-path` | required | HF repo or local checkpoint dir |
| `--task-set` | `target50` | `atomic_seen` / `composite_seen` / `composite_unseen` / `target50` |
| `--split` | `pretrain` | `pretrain` (training-distribution kitchens and objects) or `target` (held-out) |
| `--n-episodes` | 50 | Episodes per task |
| `--n-envs` | 5 | Parallel sim environments |
| `--n-action-steps` | 8 | Execution horizon |
| `--num-shards` / `--shard-index` | 1 / 0 | Optional sharding for parallel runs |

Outputs land in `output/robocasa365_eval/<exp_name>/`.

The defaults match RoboCasa's
[multi-task learning benchmark](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html),
which the leaderboard uses: models trained on the pretraining datasets only,
such as `RLDX-1.1-FT-RC365`, are evaluated with `--split pretrain`. Models
fine-tuned on the target datasets follow the
[foundation model benchmark](https://robocasa.ai/docs/build/html/benchmarking/foundation_model_learning.html)
and are evaluated with `--split target`.

Per-task step horizons are defined in
[`task_sets.yaml`](task_sets.yaml) and match RoboCasa 1.0.1. Sections:

- `atomic_seen` — atomic tasks included during pre-training
- `composite_seen` — composite tasks included during pre-training
- `composite_unseen` — composite tasks **not** in pre-training (OOD)
- `target50` — union of the three above (full benchmark)
