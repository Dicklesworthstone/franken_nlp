#!/usr/bin/env python3
"""Generate and check the Nanbeige4.2-3B tensor census.

The ordinary drift guard regenerates the committed artifact from the frozen
one-model arithmetic and requires byte identity.  The separate real-index
replay verifies the pinned ``config.json`` first, derives the same arithmetic
from it, and then checks the authenticated safetensors index names.  The index
does not contain tensor shapes, so the census retains those config-derived
records explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MODEL_REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
MODEL_NAME = "Nanbeige4.2-3B"
INDEX_SHA256 = "30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1"
INDEX_BYTES = 16_519
CONFIG_SHA256 = "f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19"
CONFIG_BYTES = 1_019

HIDDEN_SIZE = 3_072
INTERMEDIATE_SIZE = 10_752
NUM_LAYERS = 22
NUM_LOOPS = 2
NUM_ATTENTION_HEADS = 48
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 166_144
BF16_BYTES = 2

PER_LAYER_ATTENTION_PARAMS = 44_040_192
PER_LAYER_MLP_PARAMS = 99_090_432
PER_LAYER_RMSNORM_PARAMS = 6_144
PER_LAYER_PARAMS = 143_136_768
LAYER_PARAMS = 3_149_008_896
FINAL_NORM_PARAMS = 3_072
NON_EMBEDDING_PARAMS = 3_149_011_968
EMBEDDING_PARAMS = 510_394_368
TOTAL_PARAMS = 4_169_800_704
PAYLOAD_BYTES = 8_339_601_408
SHARD_BYTES = 8_339_624_720
CONTAINER_OVERHEAD_BYTES = 23_312
KV_BF16_BYTES_PER_TOKEN = 180_224
KV_INT8_BYTES_PER_TOKEN = 90_112

OQ1_FORBIDDEN_FAMILIES = ("mhc", "ngram", "depth", "loopsplit")


class CensusError(RuntimeError):
    """A census invariant, index, or committed-artifact failure."""


class DesignAssumptionError(CensusError):
    """The index contains a tensor family excluded by OQ-1."""


@dataclass(frozen=True)
class Architecture:
    """The pinned config fields that determine every tensor shape."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_layers: int
    num_loops: int
    head_dim: int
    vocab_size: int


PINNED_ARCHITECTURE = Architecture(
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_attention_heads=NUM_ATTENTION_HEADS,
    num_key_value_heads=NUM_KEY_VALUE_HEADS,
    num_layers=NUM_LAYERS,
    num_loops=NUM_LOOPS,
    head_dim=HEAD_DIM,
    vocab_size=VOCAB_SIZE,
)


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str = "BF16"

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def payload_bytes(self) -> int:
        return self.elements * BF16_BYTES

    def to_json(self) -> dict[str, Any]:
        return {
            "bytes": self.payload_bytes,
            "dtype": self.dtype,
            "name": self.name,
            "shape": list(self.shape),
        }


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{timestamp()} TENSOR_CENSUS {message}", file=sys.stderr)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def expected_tensors(architecture: Architecture) -> list[TensorRecord]:
    records = [
        TensorRecord("lm_head.weight", (architecture.vocab_size, architecture.hidden_size)),
        TensorRecord(
            "model.embed_tokens.weight",
            (architecture.vocab_size, architecture.hidden_size),
        ),
    ]
    for layer in range(architecture.num_layers):
        prefix = f"model.layers.{layer}"
        records.extend(
            (
                TensorRecord(f"{prefix}.input_layernorm.weight", (architecture.hidden_size,)),
                TensorRecord(
                    f"{prefix}.mlp.down_proj.weight",
                    (architecture.hidden_size, architecture.intermediate_size),
                ),
                TensorRecord(
                    f"{prefix}.mlp.gate_proj.weight",
                    (architecture.intermediate_size, architecture.hidden_size),
                ),
                TensorRecord(
                    f"{prefix}.mlp.up_proj.weight",
                    (architecture.intermediate_size, architecture.hidden_size),
                ),
                TensorRecord(
                    f"{prefix}.post_attention_layernorm.weight",
                    (architecture.hidden_size,),
                ),
                TensorRecord(
                    f"{prefix}.self_attn.k_proj.weight",
                    (architecture.num_key_value_heads * architecture.head_dim, architecture.hidden_size),
                ),
                TensorRecord(
                    f"{prefix}.self_attn.o_proj.weight",
                    (architecture.hidden_size, architecture.num_attention_heads * architecture.head_dim),
                ),
                TensorRecord(
                    f"{prefix}.self_attn.q_proj.weight",
                    (architecture.num_attention_heads * architecture.head_dim, architecture.hidden_size),
                ),
                TensorRecord(
                    f"{prefix}.self_attn.v_proj.weight",
                    (architecture.num_key_value_heads * architecture.head_dim, architecture.hidden_size),
                ),
            )
        )
    records.append(TensorRecord("model.norm.weight", (architecture.hidden_size,)))
    return sorted(records, key=lambda record: record.name)


def assert_inline_arithmetic(records: list[TensorRecord], architecture: Architecture) -> None:
    q_params = architecture.hidden_size * (architecture.num_attention_heads * architecture.head_dim)
    kv_params = architecture.hidden_size * (architecture.num_key_value_heads * architecture.head_dim)
    o_params = (architecture.num_attention_heads * architecture.head_dim) * architecture.hidden_size
    attention = q_params + kv_params + kv_params + o_params
    mlp_projection = architecture.hidden_size * architecture.intermediate_size
    mlp = mlp_projection * 3
    rmsnorm = architecture.hidden_size * 2

    checks = {
        "q params": (q_params, 18_874_368),
        "k/v params": (kv_params, 3_145_728),
        "o params": (o_params, 18_874_368),
        "attention params": (attention, PER_LAYER_ATTENTION_PARAMS),
        "MLP projection params": (mlp_projection, 33_030_144),
        "MLP params": (mlp, PER_LAYER_MLP_PARAMS),
        "RMSNorm params": (rmsnorm, PER_LAYER_RMSNORM_PARAMS),
        "per-layer params": (attention + mlp + rmsnorm, PER_LAYER_PARAMS),
        "layer params": (PER_LAYER_PARAMS * architecture.num_layers, LAYER_PARAMS),
        "non-embedding params": (LAYER_PARAMS + FINAL_NORM_PARAMS, NON_EMBEDDING_PARAMS),
        "embedding params": (architecture.vocab_size * architecture.hidden_size, EMBEDDING_PARAMS),
        "total params": (NON_EMBEDDING_PARAMS + 2 * EMBEDDING_PARAMS, TOTAL_PARAMS),
        "bf16 payload bytes": (TOTAL_PARAMS * BF16_BYTES, PAYLOAD_BYTES),
        "container overhead": (SHARD_BYTES - PAYLOAD_BYTES, CONTAINER_OVERHEAD_BYTES),
        "KV bf16 bytes/token": (
            architecture.num_layers
            * architecture.num_loops
            * 2
            * architecture.num_key_value_heads
            * architecture.head_dim
            * BF16_BYTES,
            KV_BF16_BYTES_PER_TOKEN,
        ),
        "KV int8 bytes/token": (
            architecture.num_layers
            * architecture.num_loops
            * 2
            * architecture.num_key_value_heads
            * architecture.head_dim,
            KV_INT8_BYTES_PER_TOKEN,
        ),
        "tensor count": (len(records), 201),
        "record payload bytes": (sum(record.payload_bytes for record in records), PAYLOAD_BYTES),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise CensusError(f"arithmetic mismatch for {name}: expected={expected} observed={observed}")


def census_document(architecture: Architecture) -> dict[str, Any]:
    records = expected_tensors(architecture)
    assert_inline_arithmetic(records, architecture)
    return {
        "model": {
            "name": "Nanbeige4.2-3B",
            "revision": MODEL_REVISION,
        },
        "schema_version": 1,
        "summary": {
            "architecture": {
                "head_dim": architecture.head_dim,
                "hidden_size": architecture.hidden_size,
                "intermediate_size": architecture.intermediate_size,
                "num_attention_heads": architecture.num_attention_heads,
                "num_key_value_heads": architecture.num_key_value_heads,
                "num_layers": architecture.num_layers,
                "num_loops": architecture.num_loops,
                "vocab_size": architecture.vocab_size,
            },
            "bf16_payload_bytes": PAYLOAD_BYTES,
            "context_buffers": {
                "bf16_kv_bytes_per_token": KV_BF16_BYTES_PER_TOKEN,
                "bf16_kv_kib_per_token": KV_BF16_BYTES_PER_TOKEN // 1024,
                "int8_kv_bytes_per_token": KV_INT8_BYTES_PER_TOKEN,
                "int8_kv_kib_per_token": KV_INT8_BYTES_PER_TOKEN // 1024,
                "positions": {
                    "4096": {"bf16_kv_bytes": KV_BF16_BYTES_PER_TOKEN * 4096},
                    "8192": {"bf16_kv_bytes": KV_BF16_BYTES_PER_TOKEN * 8192},
                    "32768": {"bf16_kv_bytes": KV_BF16_BYTES_PER_TOKEN * 32768},
                    "262144": {"bf16_kv_bytes": KV_BF16_BYTES_PER_TOKEN * 262144},
                },
            },
            "index": {
                "expected_bytes": INDEX_BYTES,
                "expected_sha256": INDEX_SHA256,
            },
            "parameters": {
                "container_overhead_bytes": CONTAINER_OVERHEAD_BYTES,
                "embedding": EMBEDDING_PARAMS,
                "final_norm": FINAL_NORM_PARAMS,
                "layers": LAYER_PARAMS,
                "non_embedding": NON_EMBEDDING_PARAMS,
                "per_layer": {
                    "attention": PER_LAYER_ATTENTION_PARAMS,
                    "mlp": PER_LAYER_MLP_PARAMS,
                    "rmsnorm": PER_LAYER_RMSNORM_PARAMS,
                    "total": PER_LAYER_PARAMS,
                },
                "shard_bytes": SHARD_BYTES,
                "total": TOTAL_PARAMS,
                "untied_lm_head": EMBEDDING_PARAMS,
            },
            "score_bucket_single_token_verification_inputs": {
                "head_dim": architecture.head_dim,
                "vocab_size": architecture.vocab_size,
            },
            "tensor_count": len(records),
        },
        "tensors": [record.to_json() for record in records],
    }


def forbidden_extra_names(names: set[str]) -> list[str]:
    return sorted(
        name for name in names if any(family in name.lower() for family in OQ1_FORBIDDEN_FAMILIES)
    )


def diff_names(expected: set[str], observed: set[str]) -> tuple[list[str], list[str]]:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return missing, extra


def diff_tensor_records(expected: list[dict[str, Any]], observed: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    expected_by_name = {record["name"]: record for record in expected}
    observed_by_name = {record["name"]: record for record in observed}
    missing, extra = diff_names(set(expected_by_name), set(observed_by_name))
    mismatched = [
        name
        for name in sorted(set(expected_by_name) & set(observed_by_name))
        if expected_by_name[name] != observed_by_name[name]
    ]
    return missing, mismatched, extra


def load_index(path: Path) -> set[str]:
    raw = path.read_bytes()
    observed_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != INDEX_BYTES:
        raise CensusError(f"index size mismatch: expected={INDEX_BYTES} observed={len(raw)} path={path}")
    if observed_sha != INDEX_SHA256:
        raise CensusError(
            f"index digest mismatch: expected={INDEX_SHA256} observed={observed_sha} path={path}"
        )
    try:
        document = json.loads(raw)
        weight_map = document["weight_map"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CensusError(f"invalid safetensors index at {path}: {exc}") from exc
    if not isinstance(weight_map, dict) or not all(isinstance(name, str) for name in weight_map):
        raise CensusError(f"invalid weight_map at {path}")
    log(
        "index "
        f"path={path} bytes_expected={INDEX_BYTES} bytes_observed={len(raw)} "
        f"sha256_expected={INDEX_SHA256} sha256_observed={observed_sha}"
    )
    return set(weight_map)


def load_pinned_architecture(path: Path) -> Architecture:
    """Authenticate the pinned config and expose its shape-owning integers."""
    raw = path.read_bytes()
    observed_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTES:
        raise CensusError(f"config size mismatch: expected={CONFIG_BYTES} observed={len(raw)} path={path}")
    if observed_sha != CONFIG_SHA256:
        raise CensusError(
            f"config digest mismatch: expected={CONFIG_SHA256} observed={observed_sha} path={path}"
        )
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CensusError(f"invalid pinned config at {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CensusError(f"invalid pinned config root at {path}")

    config_fields = {
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "num_layers": "num_hidden_layers",
        "num_loops": "num_loops",
        "head_dim": "head_dim",
        "vocab_size": "vocab_size",
    }
    values: dict[str, int] = {}
    for architecture_field, config_field in config_fields.items():
        value = document.get(config_field)
        if type(value) is not int or value <= 0:
            raise CensusError(
                f"pinned config field {config_field} must be a positive integer, observed={value!r}"
            )
        values[architecture_field] = value
    if document.get("torch_dtype") != "bfloat16":
        raise CensusError(f"pinned config torch_dtype must be bfloat16, observed={document.get('torch_dtype')!r}")
    if document.get("tie_word_embeddings") is not False:
        raise CensusError("pinned config must retain the untied lm_head")
    if document.get("skip_loop_final_norm") is not False:
        raise CensusError("pinned config must retain final RMSNorm after each loop")
    architecture = Architecture(**values)
    log(
        "config "
        f"path={path} bytes_expected={CONFIG_BYTES} bytes_observed={len(raw)} "
        f"sha256_expected={CONFIG_SHA256} sha256_observed={observed_sha}"
    )
    return architecture


def default_index_path() -> Path:
    """Return the approved fetch-script cache location for the pinned index."""
    source_dir = os.environ.get("FNLP_SOURCE_DIR")
    if source_dir:
        return Path(source_dir) / "model.safetensors.index.json"
    return (
        Path.home()
        / ".cache"
        / "franken_nlp"
        / "source"
        / MODEL_NAME
        / MODEL_REVISION
        / "model.safetensors.index.json"
    )


def assert_committed_byte_identity(path: Path, generated: dict[str, Any]) -> None:
    """Require the retained artifact to equal the generated canonical bytes."""
    observed = path.read_bytes()
    expected = canonical_json_bytes(generated)
    observed_sha = hashlib.sha256(observed).hexdigest()
    expected_sha = hashlib.sha256(expected).hexdigest()
    if observed != expected:
        raise CensusError(
            "committed census byte identity mismatch: "
            f"expected_bytes={len(expected)} observed_bytes={len(observed)} "
            f"expected_sha256={expected_sha} observed_sha256={observed_sha}"
        )
    log(
        "artifact "
        f"path={path} bytes_expected={len(expected)} bytes_observed={len(observed)} "
        f"sha256_expected={expected_sha} sha256_observed={observed_sha}"
    )


def compare_committed_artifact(path: Path, generated: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    try:
        committed = json.loads(path.read_bytes())
        committed_tensors = committed["tensors"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CensusError(f"unable to read committed census {path}: {exc}") from exc
    missing, mismatched, extra = diff_tensor_records(generated["tensors"], committed_tensors)
    if canonical_json_bytes(committed) != canonical_json_bytes(generated):
        if not (missing or mismatched or extra):
            mismatched.append("summary-or-schema")
    return missing, mismatched, extra


def report_and_raise(missing: list[str], mismatched: list[str], extra: list[str]) -> None:
    for name in missing:
        log(f"MISSING name={name}")
    for name in mismatched:
        log(f"SHAPE-MISMATCH name={name}")
    for name in extra:
        log(f"EXTRA name={name}")
    forbidden = forbidden_extra_names(set(extra))
    if forbidden:
        raise DesignAssumptionError(
            "OQ-1 design-assumption error: unexpected disabled tensor family: "
            + ", ".join(forbidden)
        )
    if missing or mismatched or extra:
        raise CensusError(
            f"census diff missing={len(missing)} mismatched={len(mismatched)} extra={len(extra)}"
        )


def run_self_test() -> None:
    generated = census_document(PINNED_ARCHITECTURE)
    expected = generated["tensors"]
    missing, mismatched, extra = diff_tensor_records(expected, expected)
    if (missing, mismatched, extra) != ([], [], []):
        raise CensusError("identical tensor records did not compare equal")

    renamed = [dict(record) for record in expected]
    renamed[0]["name"] = "renamed.tensor"
    missing, mismatched, extra = diff_tensor_records(expected, renamed)
    if len(missing) != 1 or len(extra) != 1 or mismatched:
        raise CensusError("renamed-tensor negative fixture did not produce MISSING plus EXTRA")

    wrong_shape = [dict(record) for record in expected]
    wrong_shape[0]["shape"] = [1]
    missing, mismatched, extra = diff_tensor_records(expected, wrong_shape)
    if missing or extra or len(mismatched) != 1:
        raise CensusError("wrong-shape negative fixture did not produce SHAPE-MISMATCH")

    try:
        report_and_raise([], [], ["mhc_probe.weight"])
    except DesignAssumptionError:
        pass
    else:
        raise CensusError("mHC negative fixture did not trigger the OQ-1 abort")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true", help="write a canonical census after index-name validation")
    mode.add_argument(
        "--check",
        action="store_true",
        help="require the pinned config/index replay and diff the committed census",
    )
    mode.add_argument(
        "--check-artifact",
        action="store_true",
        help="hermetically regenerate and require byte identity with the committed census",
    )
    mode.add_argument("--self-test", action="store_true", help="run hermetic arithmetic and negative-fixture tests")
    parser.add_argument(
        "--index",
        type=Path,
        help="verified model.safetensors.index.json (defaults to the fetch-script cache for --check)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="verified config.json beside --index by default; required for real-index generation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/truth-pack/tensor_census.json"),
        help="census artifact path (default: docs/truth-pack/tensor_census.json)",
    )
    args = parser.parse_args(argv)
    if args.generate and args.index is None:
        parser.error("--generate requires --index")
    if args.config is not None and args.index is None:
        parser.error("--config requires --index")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    missing: list[str] = []
    mismatched: list[str] = []
    extra: list[str] = []
    try:
        generator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        log(f"generator_sha256={generator_sha}")
        if args.self_test:
            log("phase=self-test start")
            run_self_test()
        elif args.check_artifact:
            log("phase=artifact-drift-guard start")
            generated = census_document(PINNED_ARCHITECTURE)
            missing, mismatched, extra = compare_committed_artifact(args.output, generated)
            report_and_raise(missing, mismatched, extra)
            assert_committed_byte_identity(args.output, generated)
        else:
            index_path = args.index or default_index_path()
            if not index_path.is_file():
                raise CensusError(
                    f"pinned index is absent at {index_path}; "
                    "use --check-artifact for the hermetic ordinary drift guard"
                )
            log("phase=regenerate start")
            config_path = args.config or index_path.with_name("config.json")
            architecture = load_pinned_architecture(config_path)
            generated = census_document(architecture)
            observed_names = load_index(index_path)
            expected_names = {record["name"] for record in generated["tensors"]}
            missing, extra = diff_names(expected_names, observed_names)
            report_and_raise(missing, mismatched, extra)
            if args.generate:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(canonical_json_bytes(generated))
                log(f"phase=generate output={args.output}")
            else:
                missing, mismatched, extra = compare_committed_artifact(args.output, generated)
                report_and_raise(missing, mismatched, extra)
                assert_committed_byte_identity(args.output, generated)
        log("RESULT=PASS missing=0 mismatched=0 extra=0")
        return 0
    except (CensusError, OSError) as exc:
        log(f"error={exc}")
        log(f"RESULT=FAIL missing={len(missing)} mismatched={len(mismatched)} extra={len(extra)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
