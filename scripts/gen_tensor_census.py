#!/usr/bin/env python3
"""Generate and check the Nanbeige4.2-3B tensor census.

The expected tensor inventory is intentionally derived from the pinned model
shape constants in this file.  The safetensors index is a second authority for
the *names* that are actually shipped; it does not contain tensor shapes, so
the census carries those config-derived records explicitly.
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


def expected_tensors() -> list[TensorRecord]:
    records = [
        TensorRecord("lm_head.weight", (VOCAB_SIZE, HIDDEN_SIZE)),
        TensorRecord("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN_SIZE)),
    ]
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        records.extend(
            (
                TensorRecord(f"{prefix}.input_layernorm.weight", (HIDDEN_SIZE,)),
                TensorRecord(f"{prefix}.mlp.down_proj.weight", (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
                TensorRecord(f"{prefix}.mlp.gate_proj.weight", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
                TensorRecord(f"{prefix}.mlp.up_proj.weight", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
                TensorRecord(f"{prefix}.post_attention_layernorm.weight", (HIDDEN_SIZE,)),
                TensorRecord(f"{prefix}.self_attn.k_proj.weight", (NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE)),
                TensorRecord(f"{prefix}.self_attn.o_proj.weight", (HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM)),
                TensorRecord(f"{prefix}.self_attn.q_proj.weight", (NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE)),
                TensorRecord(f"{prefix}.self_attn.v_proj.weight", (NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE)),
            )
        )
    records.append(TensorRecord("model.norm.weight", (HIDDEN_SIZE,)))
    return sorted(records, key=lambda record: record.name)


def assert_inline_arithmetic(records: list[TensorRecord]) -> None:
    q_params = HIDDEN_SIZE * (NUM_ATTENTION_HEADS * HEAD_DIM)
    kv_params = HIDDEN_SIZE * (NUM_KEY_VALUE_HEADS * HEAD_DIM)
    o_params = (NUM_ATTENTION_HEADS * HEAD_DIM) * HIDDEN_SIZE
    attention = q_params + kv_params + kv_params + o_params
    mlp_projection = HIDDEN_SIZE * INTERMEDIATE_SIZE
    mlp = mlp_projection * 3
    rmsnorm = HIDDEN_SIZE * 2

    checks = {
        "q params": (q_params, 18_874_368),
        "k/v params": (kv_params, 3_145_728),
        "o params": (o_params, 18_874_368),
        "attention params": (attention, PER_LAYER_ATTENTION_PARAMS),
        "MLP projection params": (mlp_projection, 33_030_144),
        "MLP params": (mlp, PER_LAYER_MLP_PARAMS),
        "RMSNorm params": (rmsnorm, PER_LAYER_RMSNORM_PARAMS),
        "per-layer params": (attention + mlp + rmsnorm, PER_LAYER_PARAMS),
        "layer params": (PER_LAYER_PARAMS * NUM_LAYERS, LAYER_PARAMS),
        "non-embedding params": (LAYER_PARAMS + FINAL_NORM_PARAMS, NON_EMBEDDING_PARAMS),
        "embedding params": (VOCAB_SIZE * HIDDEN_SIZE, EMBEDDING_PARAMS),
        "total params": (NON_EMBEDDING_PARAMS + 2 * EMBEDDING_PARAMS, TOTAL_PARAMS),
        "bf16 payload bytes": (TOTAL_PARAMS * BF16_BYTES, PAYLOAD_BYTES),
        "container overhead": (SHARD_BYTES - PAYLOAD_BYTES, CONTAINER_OVERHEAD_BYTES),
        "KV bf16 bytes/token": (
            NUM_LAYERS * NUM_LOOPS * 2 * NUM_KEY_VALUE_HEADS * HEAD_DIM * BF16_BYTES,
            KV_BF16_BYTES_PER_TOKEN,
        ),
        "KV int8 bytes/token": (
            NUM_LAYERS * NUM_LOOPS * 2 * NUM_KEY_VALUE_HEADS * HEAD_DIM,
            KV_INT8_BYTES_PER_TOKEN,
        ),
        "tensor count": (len(records), 201),
        "record payload bytes": (sum(record.payload_bytes for record in records), PAYLOAD_BYTES),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise CensusError(f"arithmetic mismatch for {name}: expected={expected} observed={observed}")


def census_document() -> dict[str, Any]:
    records = expected_tensors()
    assert_inline_arithmetic(records)
    return {
        "model": {
            "name": "Nanbeige4.2-3B",
            "revision": MODEL_REVISION,
        },
        "schema_version": 1,
        "summary": {
            "architecture": {
                "head_dim": HEAD_DIM,
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_attention_heads": NUM_ATTENTION_HEADS,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "num_layers": NUM_LAYERS,
                "num_loops": NUM_LOOPS,
                "vocab_size": VOCAB_SIZE,
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
                "head_dim": HEAD_DIM,
                "vocab_size": VOCAB_SIZE,
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
    return set(weight_map)


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
    generated = census_document()
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
    mode.add_argument("--check", action="store_true", help="diff the real index and committed census against regenerated expectations")
    mode.add_argument("--self-test", action="store_true", help="run hermetic arithmetic and negative-fixture tests")
    parser.add_argument(
        "--index",
        type=Path,
        help="verified model.safetensors.index.json (defaults to the fetch-script cache for --check)",
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
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    missing: list[str] = []
    mismatched: list[str] = []
    extra: list[str] = []
    try:
        if args.self_test:
            log("phase=self-test start")
            run_self_test()
        else:
            index_path = args.index or default_index_path()
            if args.check and not index_path.is_file():
                log(f"phase=check source={index_path}")
                log("RESULT=SKIPPED_NO_MODEL missing=0 mismatched=0 extra=0")
                return 0
            log("phase=regenerate start")
            generated = census_document()
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
        log("RESULT=PASS missing=0 mismatched=0 extra=0")
        return 0
    except (CensusError, OSError) as exc:
        log(f"error={exc}")
        log(f"RESULT=FAIL missing={len(missing)} mismatched={len(mismatched)} extra={len(extra)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
