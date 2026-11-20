# SPDX-License-Identifier: Apache-2.0
"""Export RLDX checkpoints to OpenVINO straight from the upstream model, and check parity.

The export traces the upstream eager ``RLDX.get_action`` -- no re-implementation of the
model -- for one fixed input layout, the same contract the forward CUDA graphs rely on:
every tensor leaf of the planned batch becomes a graph input and every non-tensor leaf
(host layout, plan sizes) is baked in. A request whose non-tensor leaves or tensor
shapes differ from the exported ones is rejected instead of silently run.

Subcommands (run from the repo root):

    # 1. Serve the CUDA reference for a benchmark and record what the simulator sends.
    uv run python run_scripts/export/rldx_openvino.py serve \
        --model-path RLWRLD/RLDX-1.1-FT-LIBERO --record-dir fixtures/libero --port 20000

    # 2. Export (fp32 CPU trace, FP16 weights, f16-safe action head).
    uv run --with openvino==2026.4.1 python run_scripts/export/rldx_openvino.py export \
        --model-path RLWRLD/RLDX-1.1-FT-LIBERO --observation fixtures/libero/000000 \
        --output exports/rldx11_libero_ov

    # 3. Open-loop parity: upstream torch vs OpenVINO with shared noise.
    uv run --with openvino==2026.4.1 python run_scripts/export/rldx_openvino.py parity \
        --export-dir exports/rldx11_libero_ov --fixtures fixtures/libero --ov-devices CPU GPU

    # 4. Closed loop: LIBERO eval against the OpenVINO server (batch-1 export -> N_ENVS=1).
    bash run_scripts/export/eval_libero_openvino.sh rldx11_ov exports/rldx11_libero_ov GPU.0
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterator
import warnings

# Tracing needs a pure-PyTorch attention path; must be set before rldx is imported.
os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")

import numpy as np
import optree
import torch
from torch import nn
import torch.nn.functional as F

from rldx.configs.execution import (
    PRESETS,
    InferenceRuntimeProfile,
    QuantizationRecipe,
    ServingLimits,
)
from rldx.configs.model.inference import InferenceConfig
from rldx.configs.model.rldx import RLDXConfig
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.model.contracts import ActionOutput
from rldx.model.core.collator import PromptEncoder, RLDXDataCollator
from rldx.model.core.processor import RLDXProcessor
from rldx.model.export import RLDXExport, graph_inputs
from rldx.model.inference_noise import initial_noise, noise_source
from rldx.model.modules.norms import RMSNorm
from rldx.planner import BatchPlanner, tree
from rldx.planner.input.qwen3vl import Qwen3VLInputPlanner
from rldx.planner.input.spec import Qwen3VLInputSpec
from rldx.planner.input.tensor_image import TensorImageSpec
from rldx.runtime.inference import InferenceCall, InferenceSpec
from rldx.serving import loader
from rldx.serving.policy import RLDXPolicy, load_policy
from rldx.serving.server import serve


META_FILE = "rldx_openvino.json"
XML_FILE = "rldx.xml"
F16_MAX = 65504.0
# Upstream's recipe (MSAT.rescale_vl_rows): GR1's 3.85e5 VL-row peak drops to ~6e3.
F16_SAFE_VL_ROW_SCALE = 1 / 64
TRANSFORMERS_KWARGS = {"trust_remote_code": False, "use_fast": True}


# --------------------------------------------------------------------------- #
# Policy assembly                                                              #
# --------------------------------------------------------------------------- #


def build_policy(
    model_path: str,
    tag: EmbodimentTag,
    *,
    dtype: torch.dtype,
    text_bucket: int | None,
    device: str = "cpu",
    runtime: Any = None,
    input_spec: Qwen3VLInputSpec | None = None,
    hidden_alignment: int = 1,
    preset: str | None = None,
) -> tuple[RLDXPolicy, Any]:
    """``RLDXPolicy.from_checkpoint`` with a chosen dtype, eager kernels and a fixed text bucket.

    With ``runtime`` given, no torch model is loaded: the planner takes ``input_spec``.
    Returns ``(policy, model_or_None)``.
    """
    model_dir = loader.resolve_model_dir(model_path)
    inference_config = InferenceConfig()
    saved = RLDXConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
    if saved.memory is not None or saved.physics is not None or saved.rtc_training_max_delay:
        raise ValueError("export covers checkpoints without memory, physics or trained RTC")
    model_config = inference_config.model_config(saved)
    modality_configs = inference_config.modality_configs(
        RLDXProcessor.load_config(model_dir).modality_configs[tag.value], model_config
    )
    processor = RLDXProcessor.from_pretrained(
        model_dir,
        rldx_config=inference_config.processor_config(model_config, modality_configs),
        modality_configs={tag.value: modality_configs},
        transformers_loading_kwargs=dict(TRANSFORMERS_KWARGS),
    )
    processor.eval()
    model = None
    if runtime is None:
        if preset:
            # Portable reference kernels: same math as _scaled_mm, and they run below sm_89.
            from rldx.model.quantization import linear as quant_linear

            quant_linear.CUDA_KERNEL_MODULES = {}
        model = loader.load_model(model_dir, model_config, device, inference_config, dtype=dtype)
        # from_pretrained leaves some submodules in the checkpoint's bf16; make the dtype uniform.
        model.to(dtype)
        runtime = loader.build_inference_runtime(
            model,
            False,
            dtype=dtype,
            # 1 keeps canonical SwiGLU / vision-MLP widths; 64 zero-pads them like the any:* presets.
            profile=InferenceRuntimeProfile(
                backend="eager",
                hidden_alignment=PRESETS[preset].hidden_alignment if preset else hidden_alignment,
                quantization=PRESETS[preset].quantization if preset else QuantizationRecipe(),
            ),
            tensor_image_path=processor.eval_image_spec,
        )
        runtime.prepare()
        input_spec = model.backbone.input_spec
    encoder = PromptEncoder(
        processor.collator.encoder.processor,
        image_max_area=model_config.image.max_area,
        image_first=model_config.conversation_image_first,
        instruction_limit=None,
        text_bucket=text_bucket,
    )
    collate_fn = BatchPlanner(
        collator=RLDXDataCollator(encoder),
        input_planner=Qwen3VLInputPlanner(
            dtype=dtype,
            spec=input_spec,
            text_attention=True,
            memo_size=loader.INFERENCE_PLANNER_MEMO_SIZE,
        ),
    )
    policy = RLDXPolicy(
        processor=processor,
        inference=runtime,
        modality_configs=modality_configs,
        embodiment_tag=tag,
        collate_fn=collate_fn,
        physics_keys=[],
        rtc_timing=None,
    )
    return policy, model


def model_inputs(call: InferenceCall) -> dict:
    """The mapping ``InferenceRuntime`` hands to ``get_action`` (as forward CUDA graphs see it)."""
    inputs = {**call.planned_batch["inputs"], **call.controls}
    host = inputs.get("host_layout")
    if host is not None:
        # Mirrors InferenceRuntime: token_layout is consumed by routing and never reaches the model.
        inputs["host_layout"] = {k: v for k, v in host.items() if k != "token_layout"}
    return inputs


def planned_call(policy: RLDXPolicy, observation: dict) -> InferenceCall:
    """Run the policy's parse / collate / plan steps on ``observation`` without the model."""
    request = policy.parse(observation)
    _, collated = policy._prepare_inputs(request)
    return policy._inference_call(collated)


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #


@contextmanager
def activation_peaks(root: nn.Module, msat: nn.Module) -> Iterator[dict]:
    """Max |x| per module output, plus the MSAT single-stream residual split into VL / SA rows."""
    peaks: dict[str, float] = {}
    n_vl = {"rows": 0}

    def record(name: str, value: torch.Tensor) -> None:
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel():
            peaks[name] = max(peaks.get(name, 0.0), float(value.detach().abs().amax()))

    def single_stream(_m, args) -> None:
        record("<msat.single.vl_rows>", args[0][:, : n_vl["rows"]])
        record("<msat.single.sa_rows>", args[0][:, n_vl["rows"] :])

    handles = [
        m.register_forward_hook(lambda _m, _a, out, n=n: record(n, out))
        for n, m in root.named_modules()
        if n
    ]
    handles.append(
        msat.vl_proj_to_sa.register_forward_hook(
            lambda _m, _a, out: n_vl.__setitem__("rows", out.shape[1])
        )
    )
    handles += [blk.pre_norm.register_forward_pre_hook(single_stream) for blk in msat.single_blocks]
    try:
        yield peaks
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- #
# OpenVINO runtime (drop-in for InferenceRuntime inside RLDXPolicy)            #
# --------------------------------------------------------------------------- #


class OpenVINORuntime:
    """Runs an exported IR behind the ``InferenceRuntime`` interface ``RLDXPolicy`` calls.

    Pre/post-processing (tokenization, planning, action decoding) stays in torch on the host;
    only ``get_action`` runs in OpenVINO. Requests whose layout differs from the export fail.
    """

    inference_mode = True

    def __init__(
        self,
        export_dir: Path,
        device: str,
        precision: str | None = None,
        extra_config: dict[str, str] | None = None,
    ):
        import openvino as ov

        self.meta = json.loads((export_dir / META_FILE).read_text())
        config = {} if precision is None else {"INFERENCE_PRECISION_HINT": precision}
        config.update(extra_config or {})
        self.compiled = ov.Core().compile_model(str(export_dir / XML_FILE), device, config)
        self.request = self.compiled.create_infer_request()
        self.spec = InferenceSpec(
            action_dim=self.meta["action_dim"], memory_shape=None, dtype=torch.float32
        )

    def prepare(self) -> None:
        pass

    def warm_up(self, calls) -> None:
        for call in calls:
            self.execute(call)

    def execute(self, call: InferenceCall) -> ActionOutput:
        tensors = graph_inputs(model_inputs(call), self.meta["layout"])
        # initial_noise honours noise_source(), so parity can share noise with torch.
        noise = initial_noise(tuple(self.meta["noise_shape"]), torch.float32, torch.device("cpu"))
        feed = [t.detach().cpu().numpy() for t in tensors] + [noise.numpy()]
        result = self.request.infer(feed)
        return ActionOutput(torch.from_numpy(np.array(result[0])))


# --------------------------------------------------------------------------- #
# Observation fixtures                                                         #
# --------------------------------------------------------------------------- #


def save_observation(path: Path, observation: dict) -> None:
    """Store an observation as ``arrays.npz`` (numpy leaves) plus ``other.json`` (the rest)."""
    path.mkdir(parents=True, exist_ok=True)
    arrays, other = {}, {}
    for group, values in observation.items():
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                arrays[f"{group}/{key}"] = value
            else:
                other.setdefault(group, {})[key] = value
    np.savez(path / "arrays.npz", **arrays)
    (path / "other.json").write_text(json.dumps(other))


def load_observation(path: Path) -> dict:
    """Inverse of ``save_observation``; pickles are refused."""
    observation: dict[str, dict] = json.loads((path / "other.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        for name in arrays.files:
            group, key = name.split("/", 1)
            observation.setdefault(group, {})[key] = arrays[name]
    return observation


def fixture_dirs(root: Path) -> list[Path]:
    """``root`` itself if it is one recorded observation, else its observation subdirectories."""
    if (root / "arrays.npz").exists():
        return [root]
    return sorted(p for p in root.iterdir() if (p / "arrays.npz").exists())


def blank_observation(policy: RLDXPolicy, instruction: str) -> dict:
    """Batch-1 observation with blank frames and ``instruction``, to fix a layout without data."""
    observation = policy._blank_observation(1, "go")
    for key in observation["language"]:
        observation["language"][key] = [[instruction] * len(observation["language"][key][0])]
    return observation


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #


def _tag(name: str) -> EmbodimentTag:
    return EmbodimentTag[name]


def cmd_export(args: argparse.Namespace) -> int:
    """Trace ``get_action`` for the layout of one observation and save the IR plus metadata.

    Steps: build the fp32 CPU policy, plan the observation, report fp32 activation peaks,
    check that text-bucket padding is exact, apply the f16-safe patches (checked exact in
    fp32), convert and save the IR, then optionally run it once on ``--check-device``.
    """
    import openvino as ov

    torch.manual_seed(0)
    tag = _tag(args.embodiment_tag)
    text_bucket = args.text_bucket or None
    # fp32: OpenVINO does not lower a bf16-traced graph faithfully.
    policy, model = build_policy(
        args.model_path, tag, dtype=torch.float32, text_bucket=text_bucket, device="cpu",
        hidden_alignment=args.hidden_alignment,
    )
    if args.observation is not None:
        observation = load_observation(args.observation)
    else:
        observation = blank_observation(policy, args.instruction)
    inputs = model_inputs(planned_call(policy, observation))
    image_spec = policy.processor.eval_image_spec
    export = RLDXExport(model, inputs, image_spec).eval()
    msat = model.action_model.model
    noise = torch.randn(export.noise_shape, generator=torch.Generator().manual_seed(0))
    sample = export.example_inputs(noise)

    print(f"[export] {len(sample) - 1} tensor inputs, noise {export.noise_shape}")
    for name, shape, dtype in export.layout["tensors"]:
        print(f"    {name}: {shape} {dtype}")
    with torch.no_grad(), activation_peaks(export, msat) as peaks:
        reference = export(*sample)
    top = sorted(peaks.items(), key=lambda kv: -kv[1])[:12]
    print("[export] largest fp32 activations (|x| > f16 max overflows on OpenVINO GPU f16):")
    for name, value in top:
        print(f"    {value:12.1f}  {name}{'  <-- > f16 max' if value > F16_MAX else ''}")
    residual_peaks = {k: v for k, v in peaks.items() if k.startswith("<msat.single.")}
    print(f"[export] MSAT single-stream residual peaks {residual_peaks}")

    if text_bucket is not None:
        # Padding is handled by the planned text attention layout: it must match no padding.
        encoder = policy.collate_fn.collator.encoder
        encoder.text_bucket = None
        # Tokens are memoized per prompt, padding included.
        encoder.__dict__.pop("_tensor_text_cache", None)
        unpadded = RLDXExport(model, model_inputs(planned_call(policy, observation)), image_spec)
        encoder.text_bucket = text_bucket
        encoder.__dict__.pop("_tensor_text_cache", None)
        with torch.no_grad():
            exact = unpadded(*unpadded.example_inputs(noise))
        err = float((exact - reference).abs().max())
        print(
            f"[export] padded (bucket {text_bucket}, input_ids {tuple(inputs['input_ids'].shape)}) vs "
            f"unpadded prompt: max|diff| = {err:.3e}"
        )

    if args.f16_safe:
        # Both patches leave fp32 outputs unchanged (up to the VL-row norm eps).
        msat.rescale_vl_rows(args.vl_row_scale)
        export.untie_first_denoising_step()
        with torch.no_grad(), activation_peaks(export, msat) as patched_peaks:
            patched = export(*sample)
        err = float((patched - reference).abs().max())
        print(f"[export] f16-safe rescale exactness (fp32): max|diff| = {err:.3e}")
        if err > args.rescale_tol:
            raise RuntimeError(f"f16-safe rescale changed outputs by {err:.3e}")
        residual_peaks = {k: v for k, v in patched_peaks.items() if k.startswith("<msat.single.")}
        print(f"[export] single-stream residual peaks after rescale {residual_peaks}")
        over = {n: v for n, v in patched_peaks.items() if v > F16_MAX}
        if over:
            print(f"[export] WARNING: still above f16 max after rescale: {over}")
    with torch.no_grad(), warnings.catch_warnings():
        # A flood of TracerWarnings can fill the pipe OpenVINO captures them through and hang.
        warnings.simplefilter("ignore")
        # Static shapes: some MOC fusions throw on the dynamic ones the tracer produces.
        ov_model = ov.convert_model(
            export, example_input=sample, input=[list(t.shape) for t in sample]
        )
    # Stable input/output names so runtimes can feed by name, not just position.
    for port, name in zip(ov_model.inputs, export.input_names, strict=True):
        port.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"action_pred"})
    args.output.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(args.output / XML_FILE), compress_to_fp16=args.compress_to_fp16)
    meta = {
        "model_path": args.model_path,
        "embodiment_tag": tag.name,
        "text_bucket": text_bucket,
        "hidden_alignment": args.hidden_alignment,
        "action_dim": model.action_model.action_dim,
        "noise_shape": list(export.noise_shape),
        "input_spec": asdict(model.backbone.input_spec),
        "layout": export.layout,
        "f16_safe": args.f16_safe,
        "vl_row_scale": args.vl_row_scale if args.f16_safe else 1.0,
        "compress_to_fp16": args.compress_to_fp16,
        "fp32_activation_peaks_top": top,
        "single_stream_residual_peaks": residual_peaks,
    }
    (args.output / META_FILE).write_text(json.dumps(meta, indent=2))
    print(f"[export] saved {args.output / XML_FILE}")

    if args.check_device:
        del ov_model
        gc.collect()
        runtime = OpenVINORuntime(args.output, args.check_device, "f32")
        result = runtime.request.infer([t.numpy() for t in sample])
        out = torch.from_numpy(np.array(result[0]))
        err = float((out - reference).abs().max())
        print(f"[export] OpenVINO {args.check_device} f32 vs torch fp32: max|diff| = {err:.3e}")
    return 0


def _run(policy: RLDXPolicy, observation: dict, noise: torch.Tensor) -> tuple[np.ndarray, dict]:
    """Normalized action chunk and decoded actions for one observation with fixed noise."""
    captured = {}
    execute = policy.inference.execute

    def capture(call):
        out = execute(call)
        captured["normalized"] = out.action_pred.float().cpu().numpy()
        return out

    policy.inference.execute = capture
    try:
        with noise_source(lambda shape, dtype, device: noise.to(dtype=dtype, device=device)):
            action, _ = policy.get_action(observation)
    finally:
        policy.inference.execute = execute
    return captured["normalized"], {k: np.asarray(v, np.float32) for k, v in action.items()}


def _compare(ref: np.ndarray, other: np.ndarray) -> dict:
    """Max / mean absolute difference, cosine similarity and NaN flag of ``other`` vs ``ref``."""
    diff = np.abs(ref - other)
    a, b = ref.ravel(), other.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"max": float(diff.max()), "mean": float(diff.mean()), "cos": cos,
            "nan": bool(np.isnan(other).any())}


def cmd_parity(args: argparse.Namespace) -> int:
    """Open-loop parity: torch reference vs each OpenVINO device on recorded observations.

    Each observation gets its own seeded noise, shared by torch and OpenVINO, so differences
    come from the backends only. Fails if any normalized chunk exceeds ``--atol`` or has NaN.
    """
    meta = json.loads((args.export_dir / META_FILE).read_text())
    tag = _tag(meta["embodiment_tag"])
    model_path = args.model_path or meta["model_path"]
    fixtures = fixture_dirs(args.fixtures)[: args.max_fixtures]
    if not fixtures:
        raise FileNotFoundError(f"no recorded observations under {args.fixtures}")
    observations = [load_observation(p) for p in fixtures]
    noises = [
        torch.randn(meta["noise_shape"], generator=torch.Generator().manual_seed(args.seed + i))
        for i in range(len(observations))
    ]

    ref_bucket = meta["text_bucket"] if args.ref_text_bucket is None else (args.ref_text_bucket or None)
    ref_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.ref_dtype]
    print(f"[parity] reference: torch {args.ref_dtype} on {args.ref_device}, text_bucket={ref_bucket}")
    if args.ref_cache is not None and args.ref_cache.exists():
        # Only valid for the same fixtures, seed, checkpoint and reference settings.
        with np.load(args.ref_cache, allow_pickle=False) as cached:
            keys = sorted({k.split("/", 2)[2] for k in cached.files if k.count("/") == 2})
            reference = [
                (cached[f"{i}/norm"], {k: cached[f"{i}/dec/{k}"] for k in keys})
                for i in range(len(observations))
            ]
        print(f"[parity] reference loaded from {args.ref_cache}")
    else:
        policy, model = build_policy(
            model_path, tag, dtype=ref_dtype, text_bucket=ref_bucket, device=args.ref_device
        )
        reference = [_run(policy, o, n) for o, n in zip(observations, noises, strict=True)]
        del policy, model
        gc.collect()
        torch.cuda.empty_cache()
        if args.ref_cache is not None:
            arrays = {}
            for i, (norm, dec) in enumerate(reference):
                arrays[f"{i}/norm"] = norm
                arrays.update({f"{i}/dec/{k}": v for k, v in dec.items()})
            np.savez(args.ref_cache, **arrays)

    spec = Qwen3VLInputSpec(**meta["input_spec"])
    report: dict[str, Any] = {"fixtures": [str(p) for p in fixtures], "reference": args.ref_dtype}
    ok = True
    # Torch candidates ("cuda:bf16") measure the drift upstream itself ships, to calibrate atol.
    for candidate in [*args.torch_candidates, *args.ov_devices]:
        model = None
        if candidate in args.torch_candidates:
            cand_device, cand_dtype, *preset = candidate.split(":", 2)
            preset = preset[0] if preset else None
            device, precision = f"torch-{cand_device}", f"{cand_dtype}{':' + preset if preset else ''}"
            policy, model = build_policy(
                model_path, tag, dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[cand_dtype],
                text_bucket=ref_bucket, device=cand_device, preset=preset,
            )
        else:
            device = candidate
            precision = args.ov_precision or ("f16" if device.startswith("GPU") else "f32")
            runtime = OpenVINORuntime(args.export_dir, device, precision, args.ov_config)
            policy, _ = build_policy(
                model_path, tag, dtype=torch.float32, text_bucket=meta["text_bucket"],
                runtime=runtime, input_spec=spec,
            )
            del runtime
        rows = []
        for (ref_norm, ref_dec), obs, noise in zip(reference, observations, noises, strict=True):
            norm, dec = _run(policy, obs, noise)
            row = {"normalized": _compare(ref_norm, norm)}
            row["decoded"] = {k: _compare(ref_dec[k], dec[k]) for k in ref_dec}
            rows.append(row)
        worst = max(r["normalized"]["max"] for r in rows)
        any_nan = any(r["normalized"]["nan"] for r in rows)
        mean = float(np.mean([r["normalized"]["mean"] for r in rows]))
        cos = min(r["normalized"]["cos"] for r in rows)
        passed = not any_nan and worst <= args.atol
        ok &= passed
        print(
            f"[parity] {device}/{precision}: normalized max|diff| {worst:.3e}, "
            f"mean {mean:.3e}, min cos {cos:.5f}, nan={any_nan} -> {'PASS' if passed else 'FAIL'}"
        )
        for key in rows[0]["decoded"]:
            print(f"    decoded {key}: max|diff| {max(r['decoded'][key]['max'] for r in rows):.3e}")
        report[f"{device}/{precision}"] = rows
        del policy, model
        gc.collect()
        torch.cuda.empty_cache()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Upstream policy server on a torch preset or an OpenVINO export.

    With ``--record-dir``, every ``--record-every``-th observation (up to ``--record-max``) is
    saved as a fixture for ``export --observation`` and ``parity``.
    """
    tag = _tag(args.embodiment_tag)
    if args.backend == "openvino":
        if args.export_dir is None:
            raise ValueError("--backend openvino needs --export-dir")
        meta = json.loads((args.export_dir / META_FILE).read_text())
        runtime = OpenVINORuntime(args.export_dir, args.ov_device, args.ov_precision, args.ov_config)
        policy, _ = build_policy(
            args.model_path or meta["model_path"], _tag(meta["embodiment_tag"]),
            dtype=torch.float32, text_bucket=meta["text_bucket"],
            runtime=runtime, input_spec=Qwen3VLInputSpec(**meta["input_spec"]),
        )
    else:
        policy = load_policy(
            embodiment_tag=tag, model_path=args.model_path, device=args.device,
            runtime_profile=PRESETS[args.runtime_profile],
        )
    if args.record_dir is not None:
        get_action, count = policy.get_action, [0]

        def recording(observation, options=None):
            if count[0] < args.record_max * args.record_every and count[0] % args.record_every == 0:
                save_observation(args.record_dir / f"{count[0]:06d}", observation)
            count[0] += 1
            return get_action(observation, options)

        policy.get_action = recording
    serve(policy, args.host, args.port)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Median ``get_action`` latency, measured like docs/optimization.md (RTC off)."""
    if args.backend == "openvino":
        if args.export_dir is None:
            raise ValueError("--backend openvino needs --export-dir")
        meta = json.loads((args.export_dir / META_FILE).read_text())
        runtime = OpenVINORuntime(args.export_dir, args.ov_device, args.ov_precision, args.ov_config)
        policy, _ = build_policy(
            args.model_path or meta["model_path"], _tag(meta["embodiment_tag"]),
            dtype=torch.float32, text_bucket=meta["text_bucket"],
            runtime=runtime, input_spec=Qwen3VLInputSpec(**meta["input_spec"]),
        )
        label = (
            f"openvino {args.ov_device}/{args.ov_precision or 'default'} "
            f"{meta.get('weight_compression', {}).get('mode', 'fp16')} {args.ov_config or ''}"
        )
    else:
        graphed = args.runtime_profile not in ("compile", "eager")
        policy = load_policy(
            embodiment_tag=_tag(args.embodiment_tag), model_path=args.model_path,
            device=args.device, runtime_profile=PRESETS[args.runtime_profile],
            serving_limits=(
                ServingLimits(args.limits_batch, args.limits_instruction_tokens) if graphed else None
            ),
        )
        label = f"torch {args.runtime_profile} on {torch.cuda.get_device_name(args.device)}"

    if args.observation is not None:
        observation = load_observation(args.observation)
    else:
        observation = blank_observation(policy, args.instruction)

    model_ms: list[float] = []
    execute = policy.inference.execute

    def timed(call):
        start = time.perf_counter()
        out = execute(call)
        if args.backend == "torch":
            torch.cuda.synchronize(args.device)
        model_ms.append((time.perf_counter() - start) * 1e3)
        return out

    policy.inference.execute = timed
    for _ in range(args.warmup):
        policy.get_action(observation)
    medians, model_medians = [], []
    for _ in range(args.repeats):
        model_ms.clear()
        calls = []
        for _ in range(args.iters):
            start = time.perf_counter()
            policy.get_action(observation)
            calls.append((time.perf_counter() - start) * 1e3)
        medians.append(statistics.median(calls))
        model_medians.append(statistics.median(model_ms))
    result = {
        "label": label,
        "get_action_median_ms": statistics.median(medians),
        "model_median_ms": statistics.median(model_medians),
        "per_repeat_get_action_ms": medians,
    }
    print(
        f"[bench] {label}: get_action median {result['get_action_median_ms']:.1f} ms "
        f"(model {result['model_median_ms']:.1f} ms), repeats {[round(m, 1) for m in medians]}"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    """NNCF weight compression of an exported IR (data-free, per-channel unless --group-size)."""
    import nncf
    import openvino as ov

    meta = json.loads((args.export_dir / META_FILE).read_text())
    model = ov.Core().read_model(str(args.export_dir / XML_FILE))
    ignored = nncf.IgnoredScope(patterns=args.ignore, validate=False) if args.ignore else None
    compressed = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode(args.mode),
        group_size=args.group_size,
        ratio=args.ratio,
        all_layers=args.all_layers or None,
        ignored_scope=ignored,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    ov.save_model(compressed, str(args.output / XML_FILE))
    meta["weight_compression"] = {
        "mode": args.mode,
        "group_size": args.group_size,
        "ratio": args.ratio,
        "all_layers": args.all_layers,
        "ignore": args.ignore,
        "source": str(args.export_dir),
    }
    (args.output / META_FILE).write_text(json.dumps(meta, indent=2))
    print(f"[quantize] saved {args.output / XML_FILE} ({args.mode})")
    return 0


def _ov_config(items: list[str]) -> dict[str, str]:
    """Parse ``--ov-config KEY=VALUE ...`` into OpenVINO compile properties."""
    config = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--ov-config takes KEY=VALUE, got {item!r}")
        config[key] = value
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="trace the upstream model to an OpenVINO IR")
    p.add_argument("--model-path", required=True)
    p.add_argument("--embodiment-tag", default="GENERAL_EMBODIMENT")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observation", type=Path, help="recorded observation fixing the layout")
    p.add_argument("--instruction", default="pick up the object", help="used without --observation")
    p.add_argument("--text-bucket", type=int, default=64,
                   help="pad prompts to a multiple of this many tokens (0 = exact length)")
    p.add_argument("--no-f16-safe", dest="f16_safe", action="store_false",
                   help="skip the MSAT VL-row rescale and the GPU first-step untie")
    p.add_argument("--vl-row-scale", type=float, default=F16_SAFE_VL_ROW_SCALE)
    p.add_argument("--rescale-tol", type=float, default=1e-3)
    p.add_argument("--no-compress-to-fp16", dest="compress_to_fp16", action="store_false")
    p.add_argument("--hidden-alignment", type=int, default=1,
                   help="zero-pad SwiGLU/vision-MLP hidden widths to this multiple (64 for INT8 GPU)")
    p.add_argument("--check-device", default="CPU", help="OpenVINO device for a post-export check ('' to skip)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("parity", help="upstream torch vs OpenVINO on recorded observations")
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--model-path", help="defaults to the exported checkpoint")
    p.add_argument("--ov-devices", nargs="*", default=["CPU"])
    p.add_argument("--torch-candidates", nargs="*", default=[], metavar="DEVICE:DTYPE",
                   help="also compare upstream torch runs, e.g. cuda:bf16 or cuda:bf16:any:fp8 "
                   "(preset quantization on upstream's portable reference kernels)")
    p.add_argument("--ref-cache", type=Path, help="npz storing the reference outputs for reuse")
    p.add_argument("--ov-precision", default=None, help="f32 / f16 / bf16 (default: f32 CPU, f16 GPU)")
    p.add_argument("--ref-device", default="cuda")
    p.add_argument("--ref-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--ref-text-bucket", type=int, default=None,
                   help="reference prompt bucket (default: the export's; 0 = unpadded like run_rldx_server)")
    p.add_argument("--max-fixtures", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=5e-2, help="max |diff| on the normalized chunk")
    p.add_argument("--report", type=Path)
    p.set_defaults(func=cmd_parity)

    p = sub.add_parser("serve", help="policy server (torch or OpenVINO), optionally recording inputs")
    p.add_argument("--model-path")
    p.add_argument("--embodiment-tag", default="GENERAL_EMBODIMENT")
    p.add_argument("--backend", choices=["torch", "openvino"], default="torch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runtime-profile", default="compile", choices=sorted(PRESETS))
    p.add_argument("--export-dir", type=Path)
    p.add_argument("--ov-device", default="CPU")
    p.add_argument("--ov-precision", default=None)
    p.add_argument("--record-dir", type=Path)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--record-max", type=int, default=64)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=20000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("bench", help="median get_action latency (torch preset or OpenVINO)")
    p.add_argument("--model-path")
    p.add_argument("--embodiment-tag", default="GENERAL_EMBODIMENT")
    p.add_argument("--backend", choices=["torch", "openvino"], default="torch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runtime-profile", default="compile", choices=sorted(PRESETS))
    p.add_argument("--limits-batch", type=int, default=1, help="docs use 3; 1 loads faster")
    p.add_argument("--limits-instruction-tokens", type=int, default=64)
    p.add_argument("--export-dir", type=Path)
    p.add_argument("--ov-device", default="GPU")
    p.add_argument("--ov-precision", default=None)
    p.add_argument("--observation", type=Path, help="recorded observation (default: blank frames)")
    p.add_argument("--instruction", default="pick up the object")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--report", type=Path)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("quantize", help="NNCF weight compression of an exported IR")
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", default="int8_sym",
                   choices=["int8_sym", "int8_asym", "int4_sym", "int4_asym"])
    p.add_argument("--group-size", type=int, default=None, help="int4 needs one, e.g. 128")
    p.add_argument("--ratio", type=float, default=None, help="int4 share of layers (rest int8)")
    p.add_argument("--all-layers", action="store_true", help="also compress embeddings/last layer")
    p.add_argument("--ignore", nargs="*", default=[], help="node-name regexes kept in fp16")
    p.set_defaults(func=cmd_quantize)

    for name in ("parity", "serve", "bench"):
        sub.choices[name].add_argument(
            "--ov-config", nargs="*", default=[], metavar="KEY=VALUE",
            help="extra OpenVINO compile properties, e.g. DYNAMIC_QUANTIZATION_GROUP_SIZE=32",
        )

    args = parser.parse_args(argv)
    if hasattr(args, "ov_config"):
        args.ov_config = _ov_config(args.ov_config)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
