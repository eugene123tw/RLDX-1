# Embodiment Tags

Every dataset and every inference call goes through an `EmbodimentTag`.
The tag selects a slot in the
[category-specific MLP encoder/decoder](architecture.md#stage-5--action-encoder--decoder)
— a per-embodiment pair of small MLPs that maps the robot's raw state
and action vectors into the model's shared latent space.

The naming convention and the list itself originate from
[NVIDIA Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release);
RLDX-1.1 preserves and extends it. See
[`rldx/data/embodiment_tags.py`](../rldx/data/embodiment_tags.py)
for the canonical enum.

## Picking a tag for a new robot

| Your case | Tag to use |
|---|---|
| Custom robot, fine-tuning on top of `RLDX-1.1-PT` | `GENERAL_EMBODIMENT` |
| Fine-tuning on Fractal (Google robot) / Bridge (WidowX) data | `OXE_FRACTAL` / `OXE_BRIDGE_ORIG` |
| Serving `RLDX-1-MT-DROID`, or fine-tuning on DROID | `OXE_DROID` |
| Any other listed simulation benchmark | `GENERAL_EMBODIMENT` |
| Loading a checkpoint that stores its slot as `NEW_EMBODIMENT` | `NEW_EMBODIMENT` |

`GENERAL_EMBODIMENT` is always the safe default. It is the slot that
`RLDX-1.1-PT` reserves for downstream fine-tuning, so any custom robot's
state/action MLP heads land there.

## Tag categories

| Tag | When it applies |
|---|---|
| `GENERAL_EMBODIMENT` | Downstream fine-tuning and every benchmark but those below |
| `NEW_EMBODIMENT` | Fine-tuning a new robot with the embodiment projector trained from scratch |
| `OXE_FRACTAL` | The slot `RLDX-1.1-PT` was pre-trained on Fractal (Google robot) |
| `OXE_BRIDGE_ORIG` | The slot `RLDX-1.1-PT` was pre-trained on Bridge (WidowX) |
| `OXE_DROID` | The slot `RLDX-1-MT-DROID` was trained on |

`OXE_FRACTAL` and `OXE_BRIDGE_ORIG` have no built-in modality config;
pass one with `--modality-config-path`.

The enum and the dataset-key string each tag maps to are in
[`rldx/data/embodiment_tags.py`](../rldx/data/embodiment_tags.py).

## How the tag is consumed

At training time the tag determines:

1. Which row of the embodiment-conditioned MLP gets gradient updates
   (each tag has its own input/output projection pair).
2. Which `MODALITY_CONFIGS` entry is loaded (in
   [`rldx/configs/data/embodiment_configs.py`](../rldx/configs/data/embodiment_configs.py))
   — this dictates the modality keys the data loader and processor use.
3. Which `embodiment_id` integer the processor passes to select that
   projection pair.

At inference time the loaded checkpoint already has trained weights
for the tag you trained with. The server resolves the tag's
`modality_keys` to slice incoming observations and to build the
action chunk it returns.

## Common pitfall: tag mismatch

A wrong `--embodiment-tag` rarely surfaces as a shape error. Instead
the lookup `state_action_processor.modality_configs[tag]["action"]`
silently returns the wrong embodiment's modality keys, and the
incoming observation dict is sliced under the wrong joint convention.
Symptoms include:

- `KeyError` on a modality key that the model expects but the dataset
  does not provide (or vice versa).
- The server appears to run, but the action chunk is garbage / nonsense
  joint values.

If the loaded checkpoint was fine-tuned on `GENERAL_EMBODIMENT`,
you must pass `--embodiment-tag GENERAL_EMBODIMENT` at inference time.
Each released checkpoint's tag is in README.md's evaluation table.

## Where to next

- [`training.md`](training.md) — register a custom modality config
  alongside a tag
- [`inference_server.md`](inference_server.md) — `--embodiment-tag`
  on the server CLI
- [GR00T N1.7 README](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release)
  — upstream concept and the original tag introduction
