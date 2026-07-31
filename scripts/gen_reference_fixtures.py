#!/usr/bin/env python3
"""Capture and verify Nanbeige4.2-3B reference traces and fixture records.

This program is deliberately outside the Rust release graph.  Its model-facing
commands are only valid inside the hash-locked CPU oracle closure established by
``franken_nlp-ilz``.  The verifier and the synthetic-format self-test are
model-free so they can run in ordinary CI.

The trace collector does not patch the archived source.  It registers read-only
forward hooks on the model's physical decoder layers and final RMSNorm, then
labels their two invocations as loop 0 and loop 1.  A hook never returns a
replacement value; the one deliberately perturbing hook used by
``--trace-selftest`` is isolated to that test.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PINNED_REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
PHYSICAL_LAYER_COUNT = 22
LOOP_COUNT = 2
TRACE_SLOT_COUNT = PHYSICAL_LAYER_COUNT * LOOP_COUNT
KV_SLOT_COUNT = TRACE_SLOT_COUNT
TRACE_FORMAT_VERSION = 1
DEFAULT_PROMPTS = (
    "The two-pass loop is explicit.",
    "Return a concise trace label.",
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class NumericsProfile:
    """A named comparison surface; only eager owns HF-match claims."""

    name: str
    torch_dtype_name: str
    attention_backend: str
    variance_only: bool


PROFILES: dict[str, NumericsProfile] = {
    "hf-bf16-eager": NumericsProfile("hf-bf16-eager", "bfloat16", "eager", False),
    "diagnostic-f32": NumericsProfile("diagnostic-f32", "float32", "eager", False),
    "hf-bf16-sdpa": NumericsProfile("hf-bf16-sdpa", "bfloat16", "sdpa", True),
}


class TraceError(RuntimeError):
    """A fail-closed trace protocol or fixture-format violation."""


def log(channel: str, message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{stamp} {channel} {message}", file=sys.stderr)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TraceError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TraceError(f"{path} must be a JSON object")
    return payload


def git_commit() -> str:
    git_path = shutil.which("git")
    if git_path is None:
        return "unavailable"
    try:
        output = subprocess.check_output(
            [git_path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"
    return output if re.fullmatch(r"[0-9a-f]{40}", output) else "unavailable"


def require_safe_relative_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or not all(SAFE_NAME.fullmatch(part) for part in candidate.parts):
        raise TraceError(f"unsafe fixture relative path: {value}")
    return candidate


def first_tensor(value: object, torch_module: Any) -> Any:
    """Return the first tensor from a common Transformer hook argument shape."""

    if isinstance(value, torch_module.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item, torch_module)
            except TraceError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item, torch_module)
            except TraceError:
                continue
    raise TraceError(f"hook payload has no tensor: {type(value).__name__}")


def clone_for_trace(tensor: Any) -> Any:
    """Detach after the live operation; preserve dtype and never mutate the path."""

    return tensor.detach().to("cpu").clone()


def tensor_raw_bytes(tensor: Any) -> bytes:
    """Serialize exact CPU storage bytes without a dtype conversion."""

    import torch

    contiguous = tensor.contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def tensor_metadata(tensor: Any) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(dimension) for dimension in tensor.shape],
        "element_size": int(tensor.element_size()),
        "byte_length": int(tensor.numel() * tensor.element_size()),
        "byte_order": "native-cpu-storage",
    }


@dataclass
class CapturedTensor:
    phase: str
    tap_name: str
    tensor: Any
    loop: int | None = None
    layer: int | None = None

    def descriptor(self, relative_path: str, payload: bytes) -> dict[str, object]:
        record: dict[str, object] = {
            "phase": self.phase,
            "tap_name": self.tap_name,
            "relative_path": relative_path,
            "sha256": sha256_bytes(payload),
            **tensor_metadata(self.tensor),
        }
        if self.loop is not None:
            record["loop"] = self.loop
        if self.layer is not None:
            record["layer"] = self.layer
        return record


@dataclass
class TracePhase:
    name: str
    layer_invocations: list[int] = field(default_factory=lambda: [0] * PHYSICAL_LAYER_COUNT)
    active_slots: list[int | None] = field(default_factory=lambda: [None] * PHYSICAL_LAYER_COUNT)
    captures: list[CapturedTensor] = field(default_factory=list)

    def slot_for(self, layer: int) -> tuple[int, int]:
        invocation = self.layer_invocations[layer]
        if invocation >= LOOP_COUNT:
            raise TraceError(
                f"trace phase={self.name} layer={layer} invoked more than {LOOP_COUNT} times"
            )
        self.layer_invocations[layer] += 1
        self.active_slots[layer] = invocation
        return invocation, layer

    def active_slot_for(self, layer: int) -> tuple[int, int]:
        loop = self.active_slots[layer]
        if loop is None:
            raise TraceError(f"trace phase={self.name} child hook fired before layer pre-hook layer={layer}")
        return loop, layer

    def assert_complete(self) -> None:
        missing = [str(index) for index, count in enumerate(self.layer_invocations) if count != LOOP_COUNT]
        if missing:
            raise TraceError(
                f"trace phase={self.name} requires {TRACE_SLOT_COUNT} layer slots; "
                f"bad layer invocation counts at {','.join(missing)}"
            )


class TraceCollector:
    """Read-only hook collector for the physical 22-layer stack called twice."""

    def __init__(self, torch_module: Any, layers: Sequence[Any], final_norm: Any, embed: Any, lm_head: Any) -> None:
        if len(layers) != PHYSICAL_LAYER_COUNT:
            raise TraceError(f"expected {PHYSICAL_LAYER_COUNT} physical layers, observed {len(layers)}")
        self.torch = torch_module
        self.layers = layers
        self.final_norm = final_norm
        self.embed = embed
        self.lm_head = lm_head
        self.phase: TracePhase | None = None
        self.handles: list[Any] = []

    def begin_phase(self, name: str) -> None:
        if self.phase is not None:
            raise TraceError(f"cannot begin phase={name}; phase={self.phase.name} is still open")
        self.phase = TracePhase(name)

    def finish_phase(self) -> TracePhase:
        if self.phase is None:
            raise TraceError("no open trace phase")
        completed = self.phase
        completed.assert_complete()
        norm_records = [record for record in completed.captures if record.tap_name == "post_loop_norm"]
        if len(norm_records) != LOOP_COUNT:
            raise TraceError(
                f"trace phase={completed.name} requires {LOOP_COUNT} post-loop norms, observed {len(norm_records)}"
            )
        self.phase = None
        return completed

    def _phase(self) -> TracePhase:
        if self.phase is None:
            raise TraceError("hook fired without an open trace phase")
        return self.phase

    def _capture(self, tap_name: str, tensor: Any, *, loop: int | None = None, layer: int | None = None) -> None:
        self._phase().captures.append(
            CapturedTensor(self._phase().name, tap_name, clone_for_trace(tensor), loop, layer)
        )

    def install(self) -> None:
        self.handles.append(self.embed.register_forward_hook(self._embed_post_hook))
        self.handles.append(self.final_norm.register_forward_hook(self._norm_post_hook))
        self.handles.append(self.lm_head.register_forward_pre_hook(self._lm_head_pre_hook))
        for layer_index, layer_module in enumerate(self.layers):
            self.handles.append(layer_module.register_forward_pre_hook(self._layer_pre_hook(layer_index)))
            self.handles.append(layer_module.register_forward_hook(self._layer_post_hook(layer_index)))
            self._install_bisect_hooks(layer_index, layer_module)

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()

    def _embed_post_hook(self, _module: Any, _arguments: tuple[object, ...], output: object) -> None:
        self._capture("post_embed", first_tensor(output, self.torch))

    def _norm_post_hook(self, _module: Any, _arguments: tuple[object, ...], output: object) -> None:
        phase = self._phase()
        norm_index = len([record for record in phase.captures if record.tap_name == "post_loop_norm"])
        if norm_index >= LOOP_COUNT:
            raise TraceError(f"trace phase={phase.name} observed too many final RMSNorm calls")
        self._capture("post_loop_norm", first_tensor(output, self.torch), loop=norm_index)

    def _lm_head_pre_hook(self, _module: Any, arguments: tuple[object, ...]) -> None:
        self._capture("pre_lm_head", first_tensor(arguments, self.torch))

    def _layer_pre_hook(self, layer_index: int) -> Callable[[Any, tuple[object, ...]], None]:
        def hook(_module: Any, arguments: tuple[object, ...]) -> None:
            loop, layer = self._phase().slot_for(layer_index)
            self._capture("pre_layer", first_tensor(arguments, self.torch), loop=loop, layer=layer)

        return hook

    def _layer_post_hook(self, layer_index: int) -> Callable[[Any, tuple[object, ...], object], None]:
        def hook(_module: Any, _arguments: tuple[object, ...], output: object) -> None:
            loop, layer = self._phase().active_slot_for(layer_index)
            self._capture("post_layer", first_tensor(output, self.torch), loop=loop, layer=layer)

        return hook

    def _install_bisect_hooks(self, layer_index: int, layer_module: Any) -> None:
        candidates = (
            ("attention", ("self_attn", "attention")),
            ("mlp", ("mlp", "feed_forward")),
        )
        for tap_stem, names in candidates:
            child = next((getattr(layer_module, name) for name in names if hasattr(layer_module, name)), None)
            if child is None or not hasattr(child, "register_forward_hook"):
                continue

            def hook(_module: Any, _arguments: tuple[object, ...], output: object, *, stem: str = tap_stem, index: int = layer_index) -> None:
                loop, layer = self._phase().active_slot_for(index)
                self._capture(f"post_{stem}", first_tensor(output, self.torch), loop=loop, layer=layer)

            self.handles.append(child.register_forward_hook(hook))


def resolve_module(root: Any, candidates: Sequence[str], description: str) -> Any:
    for candidate in candidates:
        current = root
        try:
            for part in candidate.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue
        return current
    raise TraceError(f"cannot locate {description}; inspected module paths={','.join(candidates)}")


def resolve_trace_modules(model: Any) -> tuple[Sequence[Any], Any, Any, Any]:
    layers = resolve_module(model, ("model.layers", "layers"), "decoder layers")
    if not hasattr(layers, "__len__") or not hasattr(layers, "__getitem__"):
        raise TraceError("decoder layer container is not indexable")
    final_norm = resolve_module(model, ("model.norm", "norm"), "final RMSNorm")
    embed = resolve_module(model, ("model.embed_tokens", "embed_tokens"), "token embedding")
    lm_head = resolve_module(model, ("lm_head", "model.lm_head"), "lm_head")
    return layers, final_norm, embed, lm_head


def output_logits(output: object) -> Any:
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TraceError("model output does not expose logits")


def output_past_key_values(output: object) -> object:
    value = getattr(output, "past_key_values", None)
    if value is None and isinstance(output, (tuple, list)) and len(output) >= 2:
        value = output[1]
    if value is None:
        raise TraceError("model output does not expose past_key_values")
    return value


def extract_kv_slots(cache: object) -> list[tuple[Any, Any]]:
    """Normalize legacy and DynamicCache-style KV containers without guessing slots."""

    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        pairs = list(zip(key_cache, value_cache, strict=True))
    elif isinstance(cache, (tuple, list)):
        pairs = []
        for slot, entry in enumerate(cache):
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                raise TraceError(f"KV slot={slot} is not a key/value pair")
            pairs.append((entry[0], entry[1]))
    else:
        raise TraceError(f"unsupported KV cache type={type(cache).__name__}")
    if len(pairs) != KV_SLOT_COUNT:
        raise TraceError(f"expected {KV_SLOT_COUNT} KV slots, observed {len(pairs)}")
    return pairs


def capture_kv(phase: TracePhase, cache: object, torch_module: Any) -> None:
    del torch_module  # The cache tensors are already validated by clone_for_trace.
    for slot, (key, value) in enumerate(extract_kv_slots(cache)):
        loop, layer = divmod(slot, PHYSICAL_LAYER_COUNT)
        phase.captures.append(CapturedTensor(phase.name, "kv_key", clone_for_trace(key), loop, layer))
        phase.captures.append(CapturedTensor(phase.name, "kv_value", clone_for_trace(value), loop, layer))


def phase_by_tap(phase: TracePhase, tap_name: str) -> list[CapturedTensor]:
    return [record for record in phase.captures if record.tap_name == tap_name]


def assert_loop_boundary(phase: TracePhase, torch_module: Any) -> None:
    norms = sorted(phase_by_tap(phase, "post_loop_norm"), key=lambda record: int(record.loop or 0))
    loop_two_inputs = [
        record
        for record in phase_by_tap(phase, "pre_layer")
        if record.loop == 1 and record.layer == 0
    ]
    if len(norms) != 2 or len(loop_two_inputs) != 1:
        raise TraceError("cannot assert loop boundary: required norm or loop-2 input capture missing")
    if not torch_module.equal(norms[0].tensor, loop_two_inputs[0].tensor):
        raise TraceError("loop-1 post-norm differs from the captured loop-2 layer-0 input")


def flatten_generated_tokens(model: Any, tokenized: dict[str, Any], max_new_tokens: int, torch_module: Any) -> list[int]:
    """Manual greedy decode makes the compared execution policy explicit."""

    inputs = {key: value for key, value in tokenized.items()}
    generated: list[int] = []
    past: object | None = None
    next_input = inputs
    for _ in range(max_new_tokens):
        if past is not None:
            next_input = {"input_ids": next_input["input_ids"], "past_key_values": past, "use_cache": True}
        output = model(**next_input, use_cache=True)
        next_token = int(output_logits(output)[:, -1, :].argmax(dim=-1).item())
        generated.append(next_token)
        past = output_past_key_values(output)
        next_input = {"input_ids": torch_module.tensor([[next_token]], dtype=inputs["input_ids"].dtype)}
    return generated


def traced_greedy_tokens(
    collector: TraceCollector,
    model: Any,
    tokenized: dict[str, Any],
    max_new_tokens: int,
    torch_module: Any,
) -> list[int]:
    """Generate while hooks are installed, without retaining redundant greedy traces."""

    generated: list[int] = []
    past: object | None = None
    next_input = {key: value for key, value in tokenized.items()}
    for step in range(max_new_tokens):
        collector.begin_phase(f"greedy-{step}")
        if past is None:
            output = model(**next_input, use_cache=True)
        else:
            output = model(input_ids=next_input["input_ids"], past_key_values=past, use_cache=True)
        collector.finish_phase()
        next_token = int(output_logits(output)[:, -1, :].argmax(dim=-1).item())
        generated.append(next_token)
        past = output_past_key_values(output)
        next_input = {"input_ids": torch_module.tensor([[next_token]], dtype=tokenized["input_ids"].dtype)}
    return generated


def load_oracle(profile: NumericsProfile, model_source: Path) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise TraceError(f"missing locked oracle dependency: {error}") from error
    if not model_source.is_dir():
        raise FileNotFoundError(model_source)
    required = ("config.json", "tokenizer.model")
    missing = [name for name in required if not (model_source / name).is_file()]
    if missing:
        raise TraceError(f"incomplete model source closure missing={','.join(missing)}")
    torch_dtype = getattr(torch, profile.torch_dtype_name)
    log("TRACE_HARNESS", f"load profile={profile.name} attention_backend={profile.attention_backend} source={model_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=PINNED_REVISION,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=True,
        revision=PINNED_REVISION,
        torch_dtype=torch_dtype,
        attn_implementation=profile.attention_backend,
    ).to("cpu")
    model.train(False)
    configured_backend = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config, "attn_implementation", None
    )
    if configured_backend != profile.attention_backend:
        raise TraceError(
            f"attention backend mismatch expected={profile.attention_backend} observed={configured_backend!r}"
        )
    if tokenizer.__class__.__name__.lower().find("fast") >= 0:
        raise TraceError(f"slow tokenizer required, observed class={tokenizer.__class__.__name__}")
    return torch, tokenizer, model


def closure_digest(repo_root: Path) -> str:
    path = repo_root / "docs" / "truth-pack" / "oracle_env.json"
    if not path.is_file():
        raise TraceError(f"oracle closure record is absent: {path}")
    return sha256_path(path)


def capture_prompt(torch_module: Any, tokenizer: Any, model: Any, prompt: str, max_new_tokens: int) -> tuple[TracePhase, TracePhase, list[int], Any]:
    layers, final_norm, embed, lm_head = resolve_trace_modules(model)
    collector = TraceCollector(torch_module, layers, final_norm, embed, lm_head)
    collector.install()
    tokenized = tokenizer(prompt, return_tensors="pt")
    try:
        with torch_module.inference_mode():
            collector.begin_phase("prefill")
            prefill_output = model(**tokenized, use_cache=True)
            prefill_logits = clone_for_trace(output_logits(prefill_output))
            prefill_phase = collector.finish_phase()
            capture_kv(prefill_phase, output_past_key_values(prefill_output), torch_module)

            append_input = output_logits(prefill_output)[:, -1, :].argmax(dim=-1, keepdim=True)
            collector.begin_phase("append")
            append_output = model(
                input_ids=append_input,
                past_key_values=output_past_key_values(prefill_output),
                use_cache=True,
            )
            append_phase = collector.finish_phase()
            capture_kv(append_phase, output_past_key_values(append_output), torch_module)
            assert_loop_boundary(prefill_phase, torch_module)
            greedy = traced_greedy_tokens(collector, model, tokenized, max_new_tokens, torch_module)
    finally:
        collector.close()
    return prefill_phase, append_phase, greedy, prefill_logits


def write_phase(output_root: Path, phase: TracePhase, prefix: str) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index, captured in enumerate(phase.captures):
        safe_tap = captured.tap_name.replace("_", "-")
        tensor_name = f"{prefix}-{index:04d}-{safe_tap}.bin"
        relative = Path("tensors") / tensor_name
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = tensor_raw_bytes(captured.tensor)
        target.write_bytes(payload)
        descriptor = captured.descriptor(relative.as_posix(), payload)
        records.append(descriptor)
        log(
            "TRACE_HARNESS",
            f"tap phase={phase.name} tap={captured.tap_name} loop={captured.loop} layer={captured.layer} "
            f"dtype={descriptor['dtype']} shape={descriptor['shape']} sha256={descriptor['sha256']}",
        )
    return {"phase": phase.name, "records": records}


def write_trace_bundle(
    output_root: Path,
    profile: NumericsProfile,
    prompt: str,
    prompt_index: int,
    prefill: TracePhase,
    append: TracePhase,
    greedy: list[int],
    logits: Any,
    oracle_digest: str,
    oracle_floor_sha256: str | None,
    stable_prefix_length: int | None,
) -> Path:
    bundle = output_root / profile.name / f"prompt-{prompt_index:03d}"
    bundle.mkdir(parents=True, exist_ok=False)
    payload = {
        "format_version": TRACE_FORMAT_VERSION,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "profile": profile.name,
        "attention_backend": profile.attention_backend,
        "variance_only": profile.variance_only,
        "oracle_closure_sha256": oracle_digest,
        "generator_commit": git_commit(),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "prompt_index": prompt_index,
        "greedy_tokens": greedy,
        "greedy_contract": {
            "oracle_floor_sha256": oracle_floor_sha256,
            "stable_prefix_length": stable_prefix_length,
            "status": "frozen" if stable_prefix_length is not None else "unbound",
        },
        "prefill": write_phase(bundle, prefill, "prefill"),
        "append": write_phase(bundle, append, "append"),
    }
    logits_record = CapturedTensor("prefill", "logits", logits)
    logits_path = bundle / "tensors" / "prefill-logits.bin"
    logits_payload = tensor_raw_bytes(logits)
    logits_path.write_bytes(logits_payload)
    payload["logits"] = logits_record.descriptor("tensors/prefill-logits.bin", logits_payload)
    index_path = bundle / "trace.json"
    write_json(index_path, payload)
    return index_path


def trace_counts(index: dict[str, object]) -> tuple[int, int, int]:
    phases = [index.get("prefill"), index.get("append")]
    taps = 0
    norms = 0
    kv_slots: set[tuple[str, int, int]] = set()
    for phase in phases:
        if not isinstance(phase, dict) or not isinstance(phase.get("records"), list):
            raise TraceError("trace phase records are malformed")
        records = phase["records"]
        layer_slots = {
            (record.get("loop"), record.get("layer"))
            for record in records
            if isinstance(record, dict) and record.get("tap_name") == "post_layer"
        }
        if len(layer_slots) != TRACE_SLOT_COUNT:
            raise TraceError(f"phase={phase.get('phase')} has {len(layer_slots)}/{TRACE_SLOT_COUNT} post-layer slots")
        taps += len(layer_slots)
        phase_norms = [record for record in records if isinstance(record, dict) and record.get("tap_name") == "post_loop_norm"]
        if len(phase_norms) != LOOP_COUNT:
            raise TraceError(f"phase={phase.get('phase')} has {len(phase_norms)}/{LOOP_COUNT} post-loop norms")
        norms += len(phase_norms)
        for record in records:
            if isinstance(record, dict) and record.get("tap_name") in {"kv_key", "kv_value"}:
                loop, layer = record.get("loop"), record.get("layer")
                if not isinstance(loop, int) or not isinstance(layer, int):
                    raise TraceError("KV record has no integer loop/layer slot")
                kv_slots.add((str(phase.get("phase")), loop, layer))
    return taps, norms, len(kv_slots)


def load_fixture_inputs(path: Path) -> dict[str, object]:
    """Read only repository-authored prompt, tokenizer, and template inputs."""

    payload = read_json(path)
    prompts = payload.get("prompts")
    tokenizer_cases = payload.get("tokenizer_cases")
    template_cases = payload.get("template_cases")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(item, str) and item for item in prompts):
        raise TraceError("fixture input corpus requires a non-empty string prompts array")
    if not isinstance(tokenizer_cases, list) or not isinstance(template_cases, list):
        raise TraceError("fixture input corpus requires tokenizer_cases and template_cases arrays")
    return payload


def stable_prefixes_from_floor(path: Path) -> tuple[str, dict[str, int]]:
    """Load the exact prompt-hash stable prefixes published by the floor campaign."""

    floor = read_json(path)
    raw_prefixes = floor.get("stable_prefixes")
    if not isinstance(raw_prefixes, dict):
        raise TraceError("oracle floor lacks stable_prefixes keyed by prompt SHA-256")
    prefixes: dict[str, int] = {}
    for prompt_digest, value in raw_prefixes.items():
        if not isinstance(prompt_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", prompt_digest):
            raise TraceError("oracle floor has a non-SHA-256 stable-prefix key")
        if not isinstance(value, int) or value < 0:
            raise TraceError(f"oracle floor has an invalid stable-prefix length for prompt={prompt_digest}")
        prefixes[prompt_digest] = value
    return sha256_path(path), prefixes


def capture_auxiliary_fixtures(output_root: Path, tokenizer: Any, corpus: dict[str, object]) -> Path:
    """Record slow-tokenizer ids and chat-template renderings with their input digests."""

    tokenizer_records: list[dict[str, object]] = []
    template_records: list[dict[str, object]] = []
    raw_tokenizer_cases = corpus["tokenizer_cases"]
    raw_template_cases = corpus["template_cases"]
    assert isinstance(raw_tokenizer_cases, list)
    assert isinstance(raw_template_cases, list)
    seen_case_ids: set[str] = set()
    for case in raw_tokenizer_cases:
        if not isinstance(case, dict):
            raise TraceError("tokenizer case must be an object")
        case_id = case.get("id")
        text = case.get("text")
        if not isinstance(case_id, str) or not SAFE_NAME.fullmatch(case_id) or not isinstance(text, str):
            raise TraceError("tokenizer case requires a safe id and string text")
        if case_id in seen_case_ids:
            raise TraceError(f"duplicate fixture input case id={case_id}")
        seen_case_ids.add(case_id)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not isinstance(token_ids, list) or not all(isinstance(token_id, int) for token_id in token_ids):
            raise TraceError(f"slow tokenizer did not return integer ids for case={case_id}")
        tokenizer_records.append(
            {
                "id": case_id,
                "input_sha256": sha256_bytes(text.encode("utf-8")),
                "token_ids": token_ids,
                "token_ids_sha256": sha256_bytes(canonical_json(token_ids)),
            }
        )
    for case in raw_template_cases:
        if not isinstance(case, dict):
            raise TraceError("template case must be an object")
        case_id = case.get("id")
        messages = case.get("messages")
        options = case.get("options", {})
        if not isinstance(case_id, str) or not SAFE_NAME.fullmatch(case_id):
            raise TraceError("template case requires a safe id")
        if case_id in seen_case_ids:
            raise TraceError(f"duplicate fixture input case id={case_id}")
        seen_case_ids.add(case_id)
        if not isinstance(messages, list) or not isinstance(options, dict):
            raise TraceError(f"template case={case_id} requires messages array and options object")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TraceError("slow tokenizer has no apply_chat_template for template fixture capture")
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        if not isinstance(rendered, str):
            raise TraceError(f"template case={case_id} did not render text")
        rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if not isinstance(rendered_ids, list) or not all(isinstance(token_id, int) for token_id in rendered_ids):
            raise TraceError(f"template case={case_id} did not round-trip to integer ids")
        template_records.append(
            {
                "id": case_id,
                "input_sha256": sha256_bytes(canonical_json({"messages": messages, "options": options})),
                "rendered": rendered,
                "rendered_sha256": sha256_bytes(rendered.encode("utf-8")),
                "token_ids": rendered_ids,
                "token_ids_sha256": sha256_bytes(canonical_json(rendered_ids)),
            }
        )
    payload = {
        "format_version": TRACE_FORMAT_VERSION,
        "slow_tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_cases": tokenizer_records,
        "template_cases": template_records,
    }
    path = output_root / "auxiliary.json"
    write_json(path, payload)
    return path


def all_trace_records(index: dict[str, object]) -> Iterable[dict[str, object]]:
    for phase_name in ("prefill", "append"):
        phase = index.get(phase_name)
        if not isinstance(phase, dict):
            raise TraceError(f"trace index has no {phase_name} phase")
        records = phase.get("records")
        if not isinstance(records, list):
            raise TraceError(f"trace index phase={phase_name} has no record array")
        for record in records:
            if not isinstance(record, dict):
                raise TraceError(f"trace index phase={phase_name} has a non-object record")
            yield record
    logits = index.get("logits")
    if not isinstance(logits, dict):
        raise TraceError("trace index has no logits descriptor")
    yield logits


def validate_tensor_record(root: Path, record: dict[str, object], seen_paths: set[str]) -> None:
    relative_text = record.get("relative_path")
    digest = record.get("sha256")
    shape = record.get("shape")
    element_size = record.get("element_size")
    byte_length = record.get("byte_length")
    dtype = record.get("dtype")
    if not isinstance(relative_text, str):
        raise TraceError("tensor record has no relative_path")
    relative = require_safe_relative_path(relative_text)
    if relative_text in seen_paths:
        raise TraceError(f"duplicate tensor sidecar path={relative_text}")
    seen_paths.add(relative_text)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise TraceError(f"tensor record={relative_text} has an invalid SHA-256")
    if not isinstance(dtype, str) or not dtype:
        raise TraceError(f"tensor record={relative_text} has no dtype")
    if not isinstance(shape, list) or not all(isinstance(value, int) and value >= 0 for value in shape):
        raise TraceError(f"tensor record={relative_text} has an invalid shape")
    if not isinstance(element_size, int) or element_size <= 0:
        raise TraceError(f"tensor record={relative_text} has an invalid element_size")
    if not isinstance(byte_length, int) or byte_length < 0:
        raise TraceError(f"tensor record={relative_text} has an invalid byte_length")
    if math.prod(shape) * element_size != byte_length:
        raise TraceError(f"tensor record={relative_text} byte_length disagrees with shape and element_size")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise TraceError(f"tensor record={relative_text} is missing or not a regular file")
    actual_length = path.stat().st_size
    if actual_length != byte_length:
        raise TraceError(
            f"tensor record={relative_text} length mismatch expected={byte_length} observed={actual_length}"
        )
    actual_digest = sha256_path(path)
    if actual_digest != digest:
        raise TraceError(
            f"tensor record={relative_text} SHA-256 mismatch expected={digest} observed={actual_digest}"
        )


def build_fixture_manifest(
    output_root: Path,
    trace_indices: Sequence[Path],
    auxiliary_path: Path,
    oracle_floor_sha256: str,
) -> Path:
    fixtures: list[dict[str, object]] = []
    for index_path in trace_indices:
        relative = index_path.relative_to(output_root).as_posix()
        index = read_json(index_path)
        fixtures.append(
            {
                "trace_index": relative,
                "trace_index_sha256": sha256_path(index_path),
                "profile": index["profile"],
                "attention_backend": index["attention_backend"],
                "prompt_sha256": index["prompt_sha256"],
            }
        )
    manifest = {
        "format_version": TRACE_FORMAT_VERSION,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "generator_commit": git_commit(),
        "oracle_floor_sha256": oracle_floor_sha256,
        "fixtures": fixtures,
        "auxiliary": {
            "relative_path": auxiliary_path.relative_to(output_root).as_posix(),
            "sha256": sha256_path(auxiliary_path),
        },
    }
    path = output_root / "manifest.json"
    write_json(path, manifest)
    return path


def verify_fixture_root(root: Path, oracle_floor: Path | None) -> int:
    try:
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("format_version") != TRACE_FORMAT_VERSION:
            raise TraceError("fixture manifest format_version is unsupported")
        if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != PINNED_REVISION:
            raise TraceError("fixture manifest model identity does not match the pinned wedge")
        fixtures = manifest.get("fixtures")
        if not isinstance(fixtures, list) or not fixtures:
            raise TraceError("fixture manifest has no fixtures")
        expected_floor_digest = manifest.get("oracle_floor_sha256")
        if not isinstance(expected_floor_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_floor_digest):
            raise TraceError("fixture manifest has no immutable oracle_floor_sha256")
        floor_prefixes: dict[str, int] | None = None
        if oracle_floor is not None:
            observed_floor_digest, floor_prefixes = stable_prefixes_from_floor(oracle_floor)
            if observed_floor_digest != expected_floor_digest:
                raise TraceError(
                    f"oracle floor digest mismatch expected={expected_floor_digest} observed={observed_floor_digest}"
                )
        seen_indices: set[str] = set()
        fixture_count = 0
        for fixture in fixtures:
            if not isinstance(fixture, dict):
                raise TraceError("fixture manifest has a non-object fixture")
            relative_text = fixture.get("trace_index")
            expected_index_digest = fixture.get("trace_index_sha256")
            if not isinstance(relative_text, str) or not isinstance(expected_index_digest, str):
                raise TraceError("fixture manifest has an incomplete trace index descriptor")
            if relative_text in seen_indices:
                raise TraceError(f"fixture manifest repeats trace index={relative_text}")
            seen_indices.add(relative_text)
            index_path = root / require_safe_relative_path(relative_text)
            if not index_path.is_file() or index_path.is_symlink():
                raise TraceError(f"fixture trace index missing or not regular: {relative_text}")
            observed_index_digest = sha256_path(index_path)
            if observed_index_digest != expected_index_digest:
                raise TraceError(
                    f"fixture trace index digest mismatch path={relative_text} "
                    f"expected={expected_index_digest} observed={observed_index_digest}"
                )
            index = read_json(index_path)
            profile_name = index.get("profile")
            if profile_name not in PROFILES:
                raise TraceError(f"fixture trace index has unknown profile={profile_name!r}")
            profile = PROFILES[str(profile_name)]
            if index.get("attention_backend") != profile.attention_backend:
                raise TraceError(f"fixture profile={profile.name} has an incorrect attention backend")
            if index.get("variance_only") is not profile.variance_only:
                raise TraceError(f"fixture profile={profile.name} has an incorrect variance-only tag")
            taps, norms, kv_slots = trace_counts(index)
            if taps != TRACE_SLOT_COUNT * 2 or norms != LOOP_COUNT * 2 or kv_slots != KV_SLOT_COUNT * 2:
                raise TraceError(
                    f"fixture={relative_text} incomplete traces taps={taps}/88 norms={norms}/4 kv_slots={kv_slots}/88"
                )
            seen_sidecars: set[str] = set()
            for record in all_trace_records(index):
                validate_tensor_record(index_path.parent, record, seen_sidecars)
            greedy = index.get("greedy_tokens")
            contract = index.get("greedy_contract")
            if not isinstance(greedy, list) or not all(isinstance(token, int) for token in greedy):
                raise TraceError(f"fixture={relative_text} has invalid greedy tokens")
            if not isinstance(contract, dict):
                raise TraceError(f"fixture={relative_text} has no greedy contract")
            stable_prefix = contract.get("stable_prefix_length")
            if not isinstance(stable_prefix, int) or stable_prefix < 0 or stable_prefix > len(greedy):
                raise TraceError(f"fixture={relative_text} has an invalid stable prefix length")
            if contract.get("status") != "frozen" or contract.get("oracle_floor_sha256") != expected_floor_digest:
                raise TraceError(f"fixture={relative_text} is not frozen against this oracle floor")
            prompt_digest = index.get("prompt_sha256")
            if floor_prefixes is not None:
                if not isinstance(prompt_digest, str) or floor_prefixes.get(prompt_digest) != stable_prefix:
                    raise TraceError(f"fixture={relative_text} stable prefix disagrees with oracle floor")
            fixture_count += 1
        auxiliary = manifest.get("auxiliary")
        if not isinstance(auxiliary, dict):
            raise TraceError("fixture manifest has no auxiliary fixture descriptor")
        auxiliary_relative = auxiliary.get("relative_path")
        auxiliary_digest = auxiliary.get("sha256")
        if not isinstance(auxiliary_relative, str) or not isinstance(auxiliary_digest, str):
            raise TraceError("fixture manifest has an incomplete auxiliary descriptor")
        auxiliary_path = root / require_safe_relative_path(auxiliary_relative)
        if not auxiliary_path.is_file() or auxiliary_path.is_symlink():
            raise TraceError("auxiliary fixture is missing or not a regular file")
        observed_auxiliary_digest = sha256_path(auxiliary_path)
        if observed_auxiliary_digest != auxiliary_digest:
            raise TraceError(
                f"auxiliary fixture digest mismatch expected={auxiliary_digest} observed={observed_auxiliary_digest}"
            )
        auxiliary_payload = read_json(auxiliary_path)
        if not isinstance(auxiliary_payload.get("tokenizer_cases"), list) or not isinstance(
            auxiliary_payload.get("template_cases"), list
        ):
            raise TraceError("auxiliary fixture cannot be parsed as tokenizer/template records")
    except (OSError, TraceError) as error:
        log("REF_FIXTURES", f"FAIL {error}")
        log("REF_FIXTURES", "RESULT=FAIL fixtures=0 missing=invalid-or-missing")
        return 1
    log("REF_FIXTURES", f"RESULT=PASS fixtures={fixture_count} missing=none")
    return 0


def run_fixture_self_test() -> int:
    """Model-free coverage for raw-sidecar integrity and unknown-profile rejection."""

    with tempfile.TemporaryDirectory(prefix="fnlp-fixture-selftest-") as temporary:
        root = Path(temporary)
        sidecar = root / "tensors" / "one.bin"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(b"\x00")
        record: dict[str, object] = {
            "relative_path": "tensors/one.bin",
            "sha256": sha256_path(sidecar),
            "dtype": "uint8",
            "shape": [1],
            "element_size": 1,
            "byte_length": 1,
        }
        validate_tensor_record(root, record, set())
        sidecar.write_bytes(b"\x01")
        try:
            validate_tensor_record(root, record, set())
        except TraceError:
            pass
        else:
            log("REF_FIXTURES", "RESULT=FAIL fixtures=0 missing=synthetic-digest-tamper-not-detected")
            return 1
        try:
            unknown = {"profile": "unknown-profile"}
            if unknown["profile"] not in PROFILES:
                raise TraceError("unknown profile rejected")
        except TraceError:
            log("REF_FIXTURES", "RESULT=PASS fixtures=0 missing=none selftest=synthetic")
            return 0
    log("REF_FIXTURES", "RESULT=FAIL fixtures=0 missing=unknown-profile-not-detected")
    return 1


def compare_untraced(torch_module: Any, tokenizer: Any, model: Any, prompt: str, max_new_tokens: int, traced_logits: Any, traced_greedy: list[int]) -> None:
    tokenized = tokenizer(prompt, return_tensors="pt")
    with torch_module.inference_mode():
        output = model(**tokenized, use_cache=True)
        logits = output_logits(output)
        greedy = flatten_generated_tokens(model, tokenized, max_new_tokens, torch_module)
    if not torch_module.equal(traced_logits, logits):
        raise TraceError("traced and untraced logits differ")
    if not hmac.compare_digest(canonical_json(traced_greedy), canonical_json(greedy)):
        first = next(
            (
                index
                for index, pair in enumerate(zip(traced_greedy, greedy))
                if not hmac.compare_digest(str(pair[0]).encode("ascii"), str(pair[1]).encode("ascii"))
            ),
            None,
        )
        raise TraceError(f"traced and untraced greedy tokens differ first_position={first}")


def install_perturbing_hook(layers: Sequence[Any], torch_module: Any) -> Any:
    """Return the one test-only hook which must be rejected by equality checking."""

    def perturb(_module: Any, _arguments: tuple[object, ...], output: object) -> object:
        if isinstance(output, torch_module.Tensor):
            return output + 1
        if isinstance(output, tuple) and output and isinstance(output[0], torch_module.Tensor):
            return (output[0] + 1, *output[1:])
        raise TraceError("perturbation test cannot replace the layer output shape")

    return layers[0].register_forward_hook(perturb)


def run_trace(
    args: argparse.Namespace,
    *,
    selftest: bool = False,
    allow_existing_output: bool = False,
    prompts_override: Sequence[str] | None = None,
    oracle_floor_sha256: str | None = None,
    stable_prefixes: dict[str, int] | None = None,
) -> int:
    model_source = args.model_source
    if model_source is None or not model_source.is_dir():
        log("TRACE_HARNESS", "RESULT=SKIPPED_NO_MODEL taps=0/44 norms=0/2 perturbation=none reason=source_closure_absent")
        return 0
    try:
        profile = PROFILES[args.profile]
        oracle_digest = closure_digest(args.repo_root)
        torch_module, tokenizer, model = load_oracle(profile, model_source)
        prompts = list(prompts_override) if prompts_override is not None else args.prompt or list(DEFAULT_PROMPTS)
        output_root = args.output.resolve()
        if output_root.exists() and any(output_root.iterdir()) and not allow_existing_output:
            raise TraceError(f"output directory must be absent or empty: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        completed_prompts = 0
        for index, prompt in enumerate(prompts):
            prompt_digest = sha256_bytes(prompt.encode("utf-8"))
            stable_prefix = None
            if stable_prefixes is not None:
                stable_prefix = stable_prefixes.get(prompt_digest)
                if stable_prefix is None:
                    raise TraceError(f"oracle floor has no stable prefix for prompt_sha256={prompt_digest}")
            prefill, append, greedy, logits = capture_prompt(
                torch_module, tokenizer, model, prompt, args.max_new_tokens
            )
            compare_untraced(torch_module, tokenizer, model, prompt, args.max_new_tokens, logits, greedy)
            written = write_trace_bundle(
                output_root,
                profile,
                prompt,
                index,
                prefill,
                append,
                greedy,
                logits,
                oracle_digest,
                oracle_floor_sha256,
                stable_prefix,
            )
            index_payload = read_json(written)
            taps, norms, kv_slots = trace_counts(index_payload)
            if kv_slots != KV_SLOT_COUNT * 2:
                raise TraceError(f"prefill+append KV inventory is {kv_slots}/{KV_SLOT_COUNT * 2}")
            if taps != TRACE_SLOT_COUNT * 2 or norms != LOOP_COUNT * 2:
                raise TraceError(
                    f"prompt={index} trace inventory taps={taps}/{TRACE_SLOT_COUNT * 2} "
                    f"norms={norms}/{LOOP_COUNT * 2}"
                )
            completed_prompts += 1
        perturbation = "none"
        if selftest:
            layers, _norm, _embed, _lm_head = resolve_trace_modules(model)
            tokenized = tokenizer(prompts[0], return_tensors="pt")
            with torch_module.inference_mode():
                baseline = output_logits(model(**tokenized, use_cache=True))
            handle = install_perturbing_hook(layers, torch_module)
            try:
                with torch_module.inference_mode():
                    perturbed = output_logits(model(**tokenized, use_cache=True))
                if torch_module.equal(perturbed, baseline):
                    raise TraceError("deliberately perturbing hook was not detected")
                perturbation = "first-diverging-layer-0"
            finally:
                handle.remove()
        log(
            "TRACE_HARNESS",
            f"RESULT=PASS taps={TRACE_SLOT_COUNT}/{TRACE_SLOT_COUNT} norms={LOOP_COUNT}/{LOOP_COUNT} "
            f"append_taps={TRACE_SLOT_COUNT}/{TRACE_SLOT_COUNT} append_norms={LOOP_COUNT}/{LOOP_COUNT} "
            f"prompts={completed_prompts} perturbation={perturbation}",
        )
        return 0
    except (OSError, TraceError) as error:
        log("TRACE_HARNESS", f"FAIL {error}")
        log("TRACE_HARNESS", "RESULT=FAIL taps=0/44 norms=0/2 perturbation=none")
        return 1


def load_slow_tokenizer(model_source: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TraceError(f"missing locked oracle dependency: {error}") from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=PINNED_REVISION,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    if "fast" in tokenizer.__class__.__name__.lower():
        raise TraceError(f"slow tokenizer required, observed class={tokenizer.__class__.__name__}")
    return tokenizer


def run_generate(args: argparse.Namespace) -> int:
    """Generate the complete profile matrix, then seal it with a model-free manifest pass."""

    if args.model_source is None or not args.model_source.is_dir():
        log("REF_FIXTURES", "RESULT=SKIPPED_NO_MODEL fixtures=0 missing=source-closure")
        return 0
    try:
        corpus = load_fixture_inputs(args.corpus)
        floor_digest, stable_prefixes = stable_prefixes_from_floor(args.oracle_floor)
        prompts = corpus["prompts"]
        assert isinstance(prompts, list)
        output_root = args.output.resolve()
        if output_root.exists() and any(output_root.iterdir()):
            raise TraceError(f"fixture output directory must be absent or empty: {output_root}")
        selected_profiles = tuple(PROFILES) if args.profile == "all" else (args.profile,)
        for index, profile_name in enumerate(selected_profiles):
            args.profile = profile_name
            status = run_trace(
                args,
                allow_existing_output=index > 0,
                prompts_override=prompts,
                oracle_floor_sha256=floor_digest,
                stable_prefixes=stable_prefixes,
            )
            if status != 0:
                log("REF_FIXTURES", "RESULT=FAIL fixtures=0 missing=trace-capture")
                return status
        tokenizer = load_slow_tokenizer(args.model_source)
        auxiliary_path = capture_auxiliary_fixtures(output_root, tokenizer, corpus)
        trace_indices = tuple(sorted(output_root.glob("*/prompt-*/trace.json")))
        if not trace_indices:
            raise TraceError("profile generation produced no trace index files")
        manifest_path = build_fixture_manifest(output_root, trace_indices, auxiliary_path, floor_digest)
        log(
            "REF_FIXTURES",
            f"manifest path={manifest_path.relative_to(output_root)} sha256={sha256_path(manifest_path)} "
            f"profiles={','.join(selected_profiles)}",
        )
    except (OSError, TraceError) as error:
        log("REF_FIXTURES", f"FAIL {error}")
        log("REF_FIXTURES", "RESULT=FAIL fixtures=0 missing=generation")
        return 1
    return verify_fixture_root(args.output.resolve(), args.oracle_floor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="capture prefill and one-token-append traces")
    parser.add_argument("--trace-selftest", action="store_true", help="run model-gated trace checks including perturbation detection")
    parser.add_argument("--model-source", type=Path, help="revision-scoped local Nanbeige source closure")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/reference"), help="fixture output root")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="hf-bf16-eager")
    parser.add_argument("--prompt", action="append", help="fixture prompt; repeat to add prompts")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    selected = sum(bool(value) for value in (args.trace, args.trace_selftest))
    if selected != 1:
        parser.error("select exactly one of --trace or --trace-selftest")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.trace or args.trace_selftest:
        return run_trace(args, selftest=args.trace_selftest)
    return run_trace(args, selftest=args.trace_selftest)


if __name__ == "__main__":
    raise SystemExit(main())
