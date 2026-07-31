"""Generate/check the pinned Nanbeige materialized configuration truth-pack record.

The released config.json is intentionally not treated as complete: this tool
loads its exact local remote-code configuration class with trust_remote_code,
captures ``to_dict()``, and records the raw/default/index provenance of every
materialized field.  It never downloads model code or weights itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
RAW_CONFIG_BYTES = 1_019
DEFAULT_OUTPUT = Path("docs/truth-pack/materialized_config.json")
FEATURE_TOKENS = {
    "depth_attention": ("depth",),
    "loop_split": ("loopsplit", "loop_split"),
    "mhc": ("mhc",),
    "ngram": ("ngram", "n_gram"),
}


class MaterializedConfigError(RuntimeError):
    """A source-closure, serialization, or semantic-config failure."""


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{timestamp()} MATERIALIZED_CONFIG {message}", file=sys.stderr)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def normalize_json(value: Any) -> Any:
    """Reject values that cannot be represented canonically in the truth pack."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MaterializedConfigError("materialized config contains NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise MaterializedConfigError("materialized config contains a non-string key")
        return {key: normalize_json(item) for key, item in sorted(value.items())}
    raise MaterializedConfigError(f"materialized config has unsupported value type {type(value)!r}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializedConfigError(f"unable to parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializedConfigError(f"expected JSON object in {path}")
    return normalize_json(value)


def field_entry(name: str, raw: dict[str, Any], materialized: dict[str, Any]) -> dict[str, Any]:
    raw_present = name in raw
    raw_value = raw.get(name)
    materialized_value = materialized[name]
    return {
        "changed_by_materialization": raw_present and raw_value != materialized_value,
        "materialized_value": materialized_value,
        "name": name,
        "raw_present": raw_present,
        "raw_value": raw_value if raw_present else None,
        "source": "both" if raw_present else "default",
        "value_provenance": "serialized" if raw_present else "class_default",
    }


def index_matches(index_names: set[str], tokens: tuple[str, ...]) -> list[str]:
    return sorted(
        name for name in index_names if any(token in name.lower() for token in tokens)
    )


def inactive_finding(
    materialized: dict[str, Any], index_names: set[str], tokens: tuple[str, ...]
) -> dict[str, Any]:
    config_fields = sorted(
        name
        for name in materialized
        if any(token in name.lower() for token in tokens)
    )
    values = {name: materialized[name] for name in config_fields}
    tensor_matches = index_matches(index_names, tokens)
    config_is_false_or_null = bool(config_fields) and all(
        value is None or (isinstance(value, bool) and not value) for value in values.values()
    )
    return {
        "config_fields": values,
        "config_is_false_or_null": config_is_false_or_null,
        "inactive": config_is_false_or_null and not tensor_matches,
        "tensor_matches": tensor_matches,
    }


def build_document(
    raw: dict[str, Any],
    materialized: dict[str, Any],
    index_names: set[str],
    module_inventory: list[dict[str, str]],
    raw_bytes: bytes,
    config_class: str,
) -> dict[str, Any]:
    fields = [field_entry(name, raw, materialized) for name in sorted(materialized)]
    bias_tensors = index_matches(index_names, (".bias",))
    qk_norm_tensors = index_matches(index_names, ("q_norm", "k_norm", "qk_layernorm"))
    features = {
        family: inactive_finding(materialized, index_names, tokens)
        for family, tokens in FEATURE_TOKENS.items()
    }
    return {
        "config_class": config_class,
        "fields": fields,
        "index_implications": {
            "projection_biases": {
                "no_biases_in_hot_path": not bias_tensors,
                "tensor_matches": bias_tensors,
            },
            "qk_norm": {
                "tensor_matches": qk_norm_tensors,
                "tensor_family_absent": not qk_norm_tensors,
            },
            "disabled_feature_families": features,
        },
        "instantiated_module_inventory": module_inventory,
        "model": {
            "name": "Nanbeige4.2-3B",
            "revision": MODEL_REVISION,
        },
        "raw_config": {
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
        "schema_version": 1,
    }


def fields_by_name(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in document["fields"]}


def require_value(entries: dict[str, dict[str, Any]], name: str, expected: Any) -> None:
    entry = entries.get(name)
    if entry is None:
        raise MaterializedConfigError(f"required materialized field missing: {name}")
    observed = entry["materialized_value"]
    if observed != expected:
        raise MaterializedConfigError(
            f"required materialized field mismatch: {name} expected={expected!r} observed={observed!r}"
        )


def validate_named_findings(document: dict[str, Any]) -> None:
    entries = fields_by_name(document)
    required = {
        "attention_bias": False,
        "head_dim": 128,
        "num_loops": 2,
        "rope_scaling": None,
        "rope_theta": 70_000_000,
        "skip_loop_final_norm": False,
        "vocab_size": 166_144,
    }
    for name, expected in required.items():
        require_value(entries, name, expected)
    require_value(entries, "loop_loss_weights", [])
    require_value(entries, "mlp_bias", False)
    if entries["mlp_bias"]["raw_present"]:
        raise MaterializedConfigError("mlp_bias must be omitted from raw config and supplied by class default")
    qk_entry = entries.get("qk_layernorm")
    if qk_entry is not None and qk_entry["materialized_value"] is not False:
        raise MaterializedConfigError("qk_layernorm must be false when materialized")
    if document["index_implications"]["projection_biases"]["tensor_matches"]:
        raise MaterializedConfigError("projection-bias tensor present in released index")
    if document["index_implications"]["qk_norm"]["tensor_matches"]:
        raise MaterializedConfigError("q/k norm tensor present in released index")
    for family, finding in document["index_implications"]["disabled_feature_families"].items():
        if not finding["inactive"]:
            raise MaterializedConfigError(
                f"{family} is not inactive: config={finding['config_fields']} "
                f"tensor_matches={finding['tensor_matches']}"
            )


def load_pinned_configuration(source: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], bytes, str, set[str]]:
    raw_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not raw_path.is_file() or not index_path.is_file():
        raise MaterializedConfigError(
            f"source closure missing config.json or model.safetensors.index.json: {source}"
        )
    raw_bytes = raw_path.read_bytes()
    if len(raw_bytes) != RAW_CONFIG_BYTES:
        raise MaterializedConfigError(
            f"raw config length mismatch: expected={RAW_CONFIG_BYTES} observed={len(raw_bytes)}"
        )
    raw = load_json(raw_path)
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(isinstance(name, str) for name in weight_map):
        raise MaterializedConfigError("safetensors index weight_map is missing or malformed")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise MaterializedConfigError("SKIPPED_NO_MODEL: pinned oracle dependencies are unavailable") from exc
    try:
        config = AutoConfig.from_pretrained(
            source,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
        )
        materialized = normalize_json(config.to_dict())
        config_class = f"{type(config).__module__}.{type(config).__qualname__}"

        import torch

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        inventory = [
            {
                "class": f"{type(module).__module__}.{type(module).__qualname__}",
                "name": name or "<root>",
            }
            for name, module in model.named_modules()
        ]
    except Exception as exc:  # Remote-code construction failures must be named verbatim.
        raise MaterializedConfigError(
            f"unable to instantiate pinned configuration/module inventory: {type(exc).__name__}: {exc}"
        ) from exc
    return raw, materialized, inventory, raw_bytes, config_class, set(weight_map)


def log_fields(document: dict[str, Any]) -> None:
    for entry in document["fields"]:
        log(
            "field="
            + entry["name"]
            + f" source={entry['source']} raw={entry['raw_value']!r} "
            + f"materialized={entry['materialized_value']!r}"
        )


def run_self_test() -> None:
    raw = {
        "attention_bias": False,
        "head_dim": 128,
        "num_loops": 2,
        "rope_theta": 70_000_000,
        "skip_loop_final_norm": False,
        "vocab_size": 166_144,
    }
    materialized = {
        **raw,
        "loop_loss_weights": [],
        "mlp_bias": False,
        "qk_layernorm": False,
        "rope_scaling": None,
        "use_mhc": False,
        "use_depth_attention": False,
        "use_loop_split": False,
        "use_ngram": False,
    }
    source_bytes = canonical_json_bytes(raw)
    document = build_document(
        raw,
        materialized,
        {"model.layers.0.self_attn.q_proj.weight"},
        [{"class": "test.Model", "name": "<root>"}],
        source_bytes,
        "test.NanbeigeConfig",
    )
    validate_named_findings(document)
    mlp_bias = fields_by_name(document)["mlp_bias"]
    if mlp_bias["source"] != "default" or mlp_bias["raw_present"]:
        raise MaterializedConfigError("mlp_bias default provenance self-test failed")

    changed_raw = dict(raw)
    changed_raw["num_loops"] = 3
    changed_document = build_document(
        changed_raw,
        materialized,
        {"model.layers.0.self_attn.q_proj.weight"},
        [{"class": "test.Model", "name": "<root>"}],
        canonical_json_bytes(changed_raw),
        "test.NanbeigeConfig",
    )
    changed = fields_by_name(changed_document)["num_loops"]
    if not changed["changed_by_materialization"]:
        raise MaterializedConfigError("changed raw-config negative fixture was silent")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true", help="generate the canonical materialized config record")
    mode.add_argument("--check", action="store_true", help="regenerate and byte-compare the committed record")
    mode.add_argument("--self-test", action="store_true", help="run hermetic finding and changed-config tests")
    parser.add_argument("--source", type=Path, help="verified local pinned source closure")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"output path (default: {DEFAULT_OUTPUT})")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    fields = 0
    try:
        if args.self_test:
            log("phase=self-test start")
            run_self_test()
        elif args.source is None:
            log("RESULT=SKIPPED_NO_MODEL fields=0 detail=--source pinned closure not supplied")
            return 0
        else:
            log(f"phase=load source={args.source}")
            raw, materialized, inventory, raw_bytes, config_class, index_names = load_pinned_configuration(args.source)
            document = build_document(
                raw,
                materialized,
                index_names,
                inventory,
                raw_bytes,
                config_class,
            )
            fields = len(document["fields"])
            validate_named_findings(document)
            log_fields(document)
            generated = canonical_json_bytes(document)
            if args.generate:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(generated)
                log(f"phase=generate output={args.output}")
            else:
                if not args.output.is_file():
                    raise MaterializedConfigError(f"committed materialized config missing: {args.output}")
                observed = args.output.read_bytes()
                if observed != generated:
                    raise MaterializedConfigError(
                        f"byte-identical regeneration failed: expected_sha256={hashlib.sha256(generated).hexdigest()} "
                        f"observed_sha256={hashlib.sha256(observed).hexdigest()}"
                    )
        log(f"RESULT=PASS fields={fields}")
        return 0
    except (MaterializedConfigError, OSError) as exc:
        detail = str(exc)
        if detail.startswith("SKIPPED_NO_MODEL:"):
            log(f"RESULT=SKIPPED_NO_MODEL fields={fields} detail={detail.removeprefix('SKIPPED_NO_MODEL: ')}")
            return 0
        log(f"error={detail}")
        log(f"RESULT=FAIL fields={fields}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
