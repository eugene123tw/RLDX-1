# SIMPLER Eval Fixes (`RLDX-1-FT-SIMPLER-GOOGLE` on SimplerEnv)

Summary of the bugs that blocked end-to-end SIMPLER evaluation of the released
`RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE` checkpoint and the corresponding fixes. Each
item below is independent and can be cherry-picked into an upstream PR.

Affected files:

- [rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh](rldx/eval/sim/SimplerEnv/setup_SimplerEnv.sh)
- [rldx/eval/sim/SimplerEnv/simpler_env.py](rldx/eval/sim/SimplerEnv/simpler_env.py)
- [rldx/policy/rldx_policy.py](rldx/policy/rldx_policy.py)
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

## Known follow-ups (not blockers)

- **Quaternion ordering — verified xyzw.** The allenzren `ManiSkill2_real2sim`
  fork returns `eef_pos` proprio as `[x, y, z, qw, qx, qy, qz, gripper]`
  (wxyz). The SimplerEnv adapter `_process_observation` already converts to
  xyzw via `np.roll(proprio[3:7], -1)` and stores it under `state.rx/ry/rz/rw`
  with `rw` as the scalar. Fix #4 concatenates `[rx, ry, rz, rw]` in that
  same xyzw order, matching the OXE Fractal training convention. Do not swap
  to wxyz unless rollouts demonstrably show wrong rotations.
- **WidowX state rotation.** Bridge emits Euler `state.roll/pitch/yaw`, not a
  quaternion. A separate euler→quaternion (or euler→packed-rotation) shim is
  needed for `OXE_BRIDGE_ORIG` and is not part of this changeset.
