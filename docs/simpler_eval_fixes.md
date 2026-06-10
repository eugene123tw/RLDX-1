# SIMPLER Eval Fixes (`RLDX-1-FT-SIMPLER-GOOGLE` on SimplerEnv)

Summary of the bugs that blocked end-to-end SIMPLER evaluation of the released
`RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE` checkpoint and the corresponding fixes. Each
item below is independent and can be cherry-picked into an upstream PR.

Affected files:

- [rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh](rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh)
- [rldx/eval/sim/SimplerEnv/simpler_env.py](rldx/eval/sim/SimplerEnv/simpler_env.py)
- [rldx/eval/sim/LIBERO/libero_env.py](rldx/eval/sim/LIBERO/libero_env.py)
- [rldx/eval/sim/LIBERO_PLUS/libero_plus_env.py](rldx/eval/sim/LIBERO_PLUS/libero_plus_env.py)
- [rldx/policy/rldx_policy.py](rldx/policy/rldx_policy.py)
- [rldx/model/core/processing_rldx.py](rldx/model/core/processing_rldx.py)
- [rldx/eval/rollout_policy.py](rldx/eval/rollout_policy.py)
- [run_scripts/eval/simpler/eval_simpler.sh](run_scripts/eval/simpler/eval_simpler.sh)

---

## 1. SimplerEnv venv: missing rldx import-time dependencies

**Issue.** The rollout-side venv created by
[rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh](rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh)
installed only `tianshou pydantic av zmq torchvision transformers`. Importing
`rldx.eval.rollout_policy` pulls in `rldx.policy.*`, which transitively imports
`tyro`, `einops`, `omegaconf`, `peft`, `diffusers`, `accelerate`,
`huggingface_hub`, `wandb`, `msgpack`, `dm-tree`, `lmdb`, `datasets`,
`albumentations`, etc. The script failed at every rollout launch with
`ModuleNotFoundError`.

Additionally, `transformers==4.51.3` was pinned but
`rldx.model.modules.backbone.modeling_qwen3_vl` imports
`transformers.masking_utils`, which was added in `transformers >= 4.55`.

**Fix.** Install the rldx import-time deps explicitly (so SimplerEnv's pinned
`numpy / opencv / gymnasium` are preserved) and bump transformers to `4.57.0`
to match the rldx pin.

---

## 2. `run_rldx_server.py` no longer accepts `--num-inference-timesteps`

**Issue.** [run_scripts/eval/simpler/eval_simpler.sh](run_scripts/eval/simpler/eval_simpler.sh)
forwarded `--num-inference-timesteps "$DENOISE_STEP"` to
`rldx/eval/run_rldx_server.py`. The flag was removed from the server CLI;
launching the script aborted before the server could come up.

**Fix.** Drop the unsupported flag from the server invocation.

---

## 3. `OXE_BRIDGE_ORIG` embodiment was not routed to the SIMPLER env factory

**Issue.** [rldx/eval/rollout_policy.py](rldx/eval/rollout_policy.py)
selected `get_simpler_env_fn` only for `OXE_FRACTAL`. WidowX evaluations
(embodiment tag `OXE_BRIDGE_ORIG`) fell into the `else` branch and raised
`ValueError: Invalid environment name`.

**Fix.** Add `OXE_BRIDGE_ORIG` to the SIMPLER routing branch alongside
`OXE_FRACTAL`.

---

## 4. State observation key schema mismatch (per-axis vs packed)

**Issue.** SIMPLER's `GoogleFractalEnv` / `WidowXBridgeEnv` emit per-axis state
keys (`state.x`, `state.y`, `state.z`, `state.rx/ry/rz/rw`, `state.gripper`),
but the released checkpoint's `processor_config.json` declares the OXE training
schema with **packed** keys
(`state.end_effector_position`, `state.end_effector_rotation`,
`state.gripper_position`). The processor raised
`KeyError: 'state.end_effector_position'` during observation validation.

**Fix.** In `RLDXSimPolicyWrapper.check_observation`
([rldx/policy/rldx_policy.py](rldx/policy/rldx_policy.py)), after the LIBERO
shim, pack SIMPLER per-axis keys into the OXE schema before validation:

- `state.x | state.y | state.z` → `state.end_effector_position` (shape `(B, T, 3)`)
- `state.rx | state.ry | state.rz | state.rw` → `state.end_effector_rotation`
  (shape `(B, T, 4)`, xyzw order)
- `state.gripper` → `state.gripper_position`

The original per-axis keys are left in place so downstream code (e.g. the
`is_libero` discriminator and env step adapter) is unaffected.

---

## 5. Action emitter misclassified SIMPLER as LIBERO

**Issue.** `_get_action` in [rldx/policy/rldx_policy.py](rldx/policy/rldx_policy.py)
used `is_libero = "state.x" in observation or "video.image" in observation` to
decide whether to emit LIBERO-style per-axis action keys
(`action.x, action.y, …, action.gripper`). After fix #4, SIMPLER observations
contain `state.x`, so SIMPLER actions were emitted with the wrong key schema
and the env step adapter never received the keys it expected.

**Fix.** Tighten the discriminator to a LIBERO-only marker:
`is_libero = "state.roll" in observation`. SIMPLER (which has `state.rx/ry/rz/rw`,
not Euler `roll/pitch/yaw`) now falls into the generic packed-key branch and
emits `action.end_effector_position`, `action.end_effector_rotation`,
`action.gripper_close` — matching the released checkpoint's
`modality_configs["action"]`.

---

## 6. SIMPLER env `step()` consumed per-axis action keys

**Issue.** After fix #5 the policy emits packed action keys, but both
`GoogleFractalEnv.step` and `WidowXBridgeEnv.step` in
[rldx/eval/sim/SimplerEnv/simpler_env.py](rldx/eval/sim/SimplerEnv/simpler_env.py)
indexed `action["action.x"], action["action.y"], …, action["action.gripper"]`
and raised `KeyError`.

**Fix.** Update both `step()` methods to unpack the packed dict into the
underlying 7-D SAPIEN action vector:

```python
pos = action["action.end_effector_position"]   # (3,) delta xyz
rot = action["action.end_effector_rotation"]   # (3,) axis-angle / euler
gripper = self._postprocess_gripper(action["action.gripper_close"])  # (1,)
action_vector = np.concatenate(
    [pos[..., 0:1], pos[..., 1:2], pos[..., 2:3],
     rot[..., 0:1], rot[..., 1:2], rot[..., 2:3], gripper],
    axis=0,
)
```

---

## 7. `gym.spaces.Dict` action space still declared per-axis keys

**Issue.** `SyncVectorEnv._iterate_dict` iterates `env.action_space.spaces`
keys and pulls them from the action dict. Both `GoogleFractalEnv.__init__`
and `WidowXBridgeEnv.__init__` in
[rldx/eval/sim/SimplerEnv/simpler_env.py](rldx/eval/sim/SimplerEnv/simpler_env.py)
still declared the action space with per-axis keys, so even after fix #6 the
vector-env wrapper raised `KeyError: 'action.gripper'` before `step()` was
called.

**Fix.** Update both `__init__` methods to declare the packed schema:

```python
self.action_space = gym.spaces.Dict({
    "action.end_effector_position": gym.spaces.Box(
        low=action_low[0:3], high=action_high[0:3], shape=(3,)),
    "action.end_effector_rotation": gym.spaces.Box(
        low=action_low[3:6], high=action_high[3:6], shape=(3,)),
    "action.gripper_close": gym.spaces.Box(
        low=action_low[6], high=action_high[6], shape=(1,)),
})
```

---

## Verification

```bash
pkill -f run_rldx_server.py 2>/dev/null
pkill -f rollout_policy.py 2>/dev/null
bash run_scripts/eval/simpler/eval_simpler.sh google_vm
```

Server boots, rollouts step through SIMPLER without `KeyError`s, and the
policy receives the OXE-schema observations / emits the OXE-schema actions
that the released checkpoint was trained on.

---

## 8. Refactor — envs own their I/O schema (supersedes #4–#7)

**Issue.** Fixes #4–#7 patched the symptoms by accumulating per-embodiment
shim blocks inside `RLDXSimPolicyWrapper` (two duplicated SIMPLER/LIBERO
key-packing blocks in `check_observation` and `_get_action`, plus an
`is_libero = "state.roll" in observation` action sentinel). The wrapper had
to know per-embodiment quirks; envs were emitting per-axis state keys that
no consumer ever used.

The same approach also did **not** cover WidowX state rotation: the SIMPLER
mapping block only packed `state.end_effector_rotation` from quaternion keys
(`state.rx/ry/rz/rw`), so the Bridge env — which emits Euler
`state.roll/pitch/yaw` — still raised
`AssertionError: State key 'state.end_effector_rotation' must be in
observation` (the "WidowX state rotation" follow-up below).

**Fix.** Move all schema knowledge into the envs and make the wrapper
embodiment-agnostic:

- `GoogleFractalEnv` / `WidowXBridgeEnv`
  ([rldx/eval/sim/SimplerEnv/simpler_env.py](rldx/eval/sim/SimplerEnv/simpler_env.py))
  emit packed OXE keys directly from `_process_observation`:
  `state.end_effector_position` (3D), `state.end_effector_rotation`
  (4D quat xyzw for Google, 3D euler for WidowX), `state.gripper_position`.
  Per-axis / `state.pad` keys are removed. `gym.spaces.Dict` declares the
  same packed keys.
- `LiberoEnv` / `LiberoPlusEnv`
  ([rldx/eval/sim/LIBERO/libero_env.py](rldx/eval/sim/LIBERO/libero_env.py),
  [rldx/eval/sim/LIBERO_PLUS/libero_plus_env.py](rldx/eval/sim/LIBERO_PLUS/libero_plus_env.py))
  emit `video.front_view`, `video.left_wrist_view`, `state.eef_pos_absolute`
  (3D), `state.eef_rot_absolute` (3D), `state.gripper_close` (2D). `step()`
  reads `action.eef_pos_delta`, `action.eef_rot_delta`, `action.gripper_close`
  and applies the `1 - gripper_close` flip locally (previously done in the
  wrapper).
- `RLDXSimPolicyWrapper` ([rldx/policy/rldx_policy.py](rldx/policy/rldx_policy.py))
  loses both `===== LIBERO KEY MAPPING =====` blocks, the `===== SIMPLER KEY
  MAPPING =====` block, and the `is_libero` action-unpacking branch. The
  action transform is now unconditionally
  `flat_actions = {f"action.{key}": action[key] for key in action}`.
  The GR-1 / RoboCasa video-key fallbacks remain (those envs were not part of
  this refactor).

The wrapper now contains no per-embodiment logic; adding a new SIMPLER /
LIBERO variant no longer requires editing `rldx_policy.py`. Per-axis state
keys (`state.x/y/z/rx/ry/rz/rw/roll/pitch/yaw/pad/gripper`) and the LIBERO
per-axis action keys (`action.x/y/z/roll/pitch/yaw/gripper`) are no longer
emitted anywhere in the repo.

---

## 9. Defensive default for `image_max_area=None` in checkpoints

**Issue.** `RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX`'s `processor_config.json`
persists `"image_max_area": null` (and `"image_resize_m": 32`). At inference,
the WidowX env feeds 256×256 frames into the image pipeline, which lands in
`resize_preserve_aspect_area_then_crop` and crashes with:

```
TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'
```

`image_max_area` is annotated as `int` everywhere and has no `is None`
fallback — only `random_crop_fraction` / `random_rotation_angle` /
`color_jitter_params` are honestly optional. The `null` slipped into the
JSON via an older training recipe that never tripped the `sqrt(area / hw)`
path (`max_area=65536` is a no-op for 256×256 inputs, so training silently
"worked").

**Fix.** In `RLDXProcessor.__init__`
([rldx/model/core/processing_rldx.py](rldx/model/core/processing_rldx.py)),
coerce `None → default` for both `image_max_area` (65536) and
`image_resize_m` (32) with a rank-zero warning. Recovered defaults match the
in-repo training defaults
([rldx/configs/train_config.py](rldx/configs/train_config.py); the sibling
[run_scripts/train/benchmarks/finetune_rldx1_simpler_google.sh](run_scripts/train/benchmarks/finetune_rldx1_simpler_google.sh)
passes `--image-max-area 65536` explicitly, and the WidowX script relies on
the same dataclass default). For 256×256 inputs both values are no-ops, so
inference now reproduces training behavior bit-exactly.

The next bad checkpoint will print:

```
[!] processor: image_max_area=None in checkpoint; falling back to default 65536. Set it explicitly to silence.
```

instead of crashing inside albumentations.

## Known follow-ups (not blockers)

- **Quaternion ordering — verified xyzw.** The allenzren `ManiSkill2_real2sim`
  fork returns `eef_pos` proprio as `[x, y, z, qw, qx, qy, qz, gripper]`
  (wxyz). The Google adapter `_process_observation` converts to xyzw via
  `np.roll(proprio[3:7], -1)` and emits it as the 4D
  `state.end_effector_rotation`, matching the OXE Fractal training
  convention. Do not swap to wxyz unless rollouts demonstrably show wrong
  rotations.
- ~~**WidowX state rotation.**~~ **Resolved by #8.** Bridge emits Euler
  `state.roll/pitch/yaw`; the WidowX env now packs them directly into the
  3D `state.end_effector_rotation` expected by the `simpler_widowx` modality
  config, no wrapper-side shim needed.
