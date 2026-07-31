#!/usr/bin/env python3
"""Generate/check the OQ-12 GGUF-prior audit.

This is deliberately a source-rule audit, not a converter or a quantization
decision.  It only becomes observed GGUF evidence after a retained conversion
records the exact output digest and its tensor inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "docs" / "truth-pack"
BASELINE = TRUTH / "llamacpp_baseline.json"
MAPPING = TRUTH / "gguf_prior_mapping.json"
TIERS = TRUTH / "gguf_quant_tiers.json"
HYPOTHESES = TRUTH / "gguf_int4_hypotheses.json"
LINEAGE = TRUTH / "gguf_prior_lineage.md"

MODEL_REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
INDEX_SHA256 = "30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1"
OFFICIAL_REVISION = "000547513f1530346ecd163db8b3e13962949961"
SUPPORT_REVISION = "b77d646751d01c0962bc203b6809e9d94f7d50b7"
FORK_BRANCH_TIP = "c6640a1c0cf7b38df342b67021a3900b04d092e7"
SOURCE_PREFIX = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/"
    f"{OFFICIAL_REVISION}/"
)
SOURCE_RULE_STATUS = "SOURCE_RULE_PREDICTION_BLOCKED_NO_GGUF_DIGEST"


class AuditError(RuntimeError):
    """The committed OQ-12 prior is malformed or internally inconsistent."""


def log(message: str) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{now} GGUF_PRIOR {message}", file=sys.stderr)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_pairs)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot parse {path.relative_to(ROOT)}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{path.relative_to(ROOT)} root must be an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


def tensor_names() -> list[str]:
    names = ["lm_head.weight", "model.embed_tokens.weight"]
    for layer in range(22):
        prefix = f"model.layers.{layer}"
        names.extend((
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.mlp.down_proj.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ))
    names.append("model.norm.weight")
    return sorted(names)


def mapping_for(name: str) -> tuple[str, str, str, int | None]:
    if name == "lm_head.weight":
        return "output.weight", "lm_head", "identity", None
    if name == "model.embed_tokens.weight":
        return "token_embd.weight", "embed", "identity", None
    if name == "model.norm.weight":
        return "output_norm.weight", "norm", "identity", None
    _, _, layer, rest = name.split(".", 3)
    index = int(layer)
    table = {
        "input_layernorm.weight": ("attn_norm.weight", "norm", "identity"),
        "post_attention_layernorm.weight": ("ffn_norm.weight", "norm", "identity"),
        "self_attn.q_proj.weight": ("attn_q.weight", "q", "rope_permute"),
        "self_attn.k_proj.weight": ("attn_k.weight", "k", "rope_permute"),
        "self_attn.v_proj.weight": ("attn_v.weight", "v", "identity"),
        "self_attn.o_proj.weight": ("attn_output.weight", "o", "identity"),
        "mlp.gate_proj.weight": ("ffn_gate.weight", "gate", "identity"),
        "mlp.up_proj.weight": ("ffn_up.weight", "up", "identity"),
        "mlp.down_proj.weight": ("ffn_down.weight", "down", "identity"),
    }
    suffix, tensor_class, transform = table[rest]
    return f"blk.{index}.{suffix}", tensor_class, transform, index


def more_bits(layer: int) -> bool:
    return layer < 22 // 8 or layer >= (7 * 22) // 8 or (layer - 22 // 8) % 3 == 2


def records() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for hf_name in tensor_names():
        gguf_name, tensor_class, transform, layer = mapping_for(hf_name)
        eligible = tensor_class != "norm"
        q4 = "UNQUANTIZED" if not eligible else "Q4_K"
        if tensor_class == "lm_head":
            q4 = "Q6_K"
        elif tensor_class in {"v", "down"} and layer is not None and more_bits(layer):
            q4 = "Q6_K"
        output.append({
            "class": tensor_class,
            "eligible_for_weight_quantization": eligible,
            "gguf_name": gguf_name,
            "hf_name": hf_name,
            "layer": layer,
            "merge": "none",
            "q4_k_m_source_rule": q4,
            "q8_0_source_rule": "Q8_0" if eligible else "UNQUANTIZED",
            "transform": transform,
        })
    return output


def source_evidence() -> dict[str, str]:
    return {
        "base_mapping": SOURCE_PREFIX + "conversion/base.py",
        "llama_mapping_and_permute": SOURCE_PREFIX + "conversion/llama.py",
        "nanbeige_model_registration": SOURCE_PREFIX + "conversion/nanbeige.py",
        "quant_mixture_rules": SOURCE_PREFIX + "src/llama-quant.cpp",
    }


def payloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    all_records = records()
    high_layers = sorted(record["layer"] for record in all_records if record["class"] == "v" and record["q4_k_m_source_rule"] == "Q6_K")
    if high_layers != [0, 1, 4, 7, 10, 13, 16, 19, 20, 21]:
        raise AuditError(f"Q4_K_M V override invariant failed: {high_layers}")
    common = {
        "schema_version": 1,
        "evidence_status": SOURCE_RULE_STATUS,
        "model": "Nanbeige4.2-3B",
        "model_revision": MODEL_REVISION,
        "official_llamacpp_revision": OFFICIAL_REVISION,
        "conversion_command": "python3 llama.cpp/convert_hf_to_gguf.py <pinned-source-dir> --outfile <nanbeige4.2-3b-f56ec5a-q8_0.gguf> --outtype q8_0",
        "gguf_digests": [],
        "non_authority": "These are llama.cpp source-rule predictions and search seeds. They are not a forward-parity result, artifact evidence, or an approved FrankenNLP quantization recipe.",
        "source_evidence": source_evidence(),
    }
    mapping = {
        **common,
        "census_contract": {
            "expected_index_sha256": INDEX_SHA256,
            "expected_tensor_count": 201,
            "live_census_artifact": "docs/truth-pack/tensor_census.json",
            "reconciliation": "PENDING_CENSUS_ARTIFACT",
        },
        "records": all_records,
        "unmatched_findings": [
            "No local pinned source closure or generated GGUF exists; an observed converter inventory and GGUF SHA-256 are required before this mapping may be promoted."
        ],
    }
    tiers = {
        **common,
        "q4_k_m_rule": {
            "base": "Q4_K for eligible rank-2 weights",
            "overrides": {
                "down": {"layers": high_layers, "tier": "Q6_K"},
                "lm_head": {"tier": "Q6_K"},
                "v": {"layers": high_layers, "tier": "Q6_K"},
            },
            "ordering_assumption": "Layer index is the source-rule ordinal; retain actual GGUF tensor order before treating it as observed output.",
        },
        "records": all_records,
    }
    hypotheses = {
        **common,
        "hypotheses": [
            {"id": "OQ12-H1", "candidate": "lm_head", "candidate_tier": "Q6_K_or_higher", "evidence_pointer": "llama-quant.cpp Q4_K_M output override", "status": "SEARCH_SEED_ONLY"},
            {"id": "OQ12-H2", "candidate": "attention value projections at layers 0,1,4,7,10,13,16,19,20,21", "candidate_tier": "Q6_K_or_higher", "evidence_pointer": "llama-quant.cpp use_more_bits rule", "status": "SEARCH_SEED_ONLY"},
            {"id": "OQ12-H3", "candidate": "FFN down projections at layers 0,1,4,7,10,13,16,19,20,21", "candidate_tier": "Q6_K_or_higher", "evidence_pointer": "llama-quant.cpp use_more_bits rule", "status": "SEARCH_SEED_ONLY"},
        ],
        "decision_gate": "A candidate becomes a recipe decision only after retained full-artifact conversion plus preregistered held-out parity and task metrics.",
    }
    lineage = """# OQ-12 GGUF-prior lineage\n\n+## Official baseline\n\n+The conversion prior is tied to official `ggml-org/llama.cpp` commit\n+`000547513f1530346ecd163db8b3e13962949961`, selected by the baseline record.\n+It is 57 commits after the minimum Nanbeige-support commit\n+`b77d646751d01c0962bc203b6809e9d94f7d50b7`.  The mapping and tier rules\n+are source observations at that immutable commit: [Nanbeige conversion]\n+(https://raw.githubusercontent.com/ggml-org/llama.cpp/000547513f1530346ecd163db8b3e13962949961/conversion/nanbeige.py),\n+[Llama mapping]\n+(https://raw.githubusercontent.com/ggml-org/llama.cpp/000547513f1530346ecd163db8b3e13962949961/conversion/llama.py), and\n+[quant tiers]\n+(https://raw.githubusercontent.com/ggml-org/llama.cpp/000547513f1530346ecd163db8b3e13962949961/src/llama-quant.cpp).\n\n+## Authors' fork\n\n+`Nanbeige/llama.cpp` branch `nanbeige42` was observed at\n+`c6640a1c0cf7b38df342b67021a3900b04d092e7` on 2026-07-31 via\n+`git ls-remote https://github.com/Nanbeige/llama.cpp.git refs/heads/nanbeige42`.\n+The branch tip is a reported observation, not a pin in the baseline record.\n+No merge-base or source-diff evidence is retained here, so its divergence from\n+the official baseline is **inconclusive**.  It is historical lineage only and\n+never an independent oracle or a deciding vote.\n\n+## Authority boundary\n\n+No local GGUF exists for this audit, and no GGUF digest or converter tensor\n+inventory is claimed.  The JSON tables are source-rule predictions/search\n+seeds only.  Full candidate artifacts and held-out metrics decide any int4\n+recipe; nearest GGUF formats are peers, never a "match" claim.\n+"""
    return mapping, tiers, hypotheses, lineage


def generate() -> None:
    mapping, tiers, hypotheses, lineage = payloads()
    TRUTH.mkdir(parents=True, exist_ok=True)
    write_json(MAPPING, mapping)
    write_json(TIERS, tiers)
    write_json(HYPOTHESES, hypotheses)
    LINEAGE.write_text(lineage, encoding="utf-8")
    log("RESULT=PASS action=generate tensors=201 evidence=SOURCE_RULE_PREDICTION")


def verify_census(records_to_check: list[dict[str, Any]]) -> str:
    census_path = TRUTH / "tensor_census.json"
    if not census_path.exists():
        return "SKIPPED_NO_CENSUS"
    census = load_json(census_path)
    names = {item["name"] for item in census.get("tensors", []) if isinstance(item, dict) and isinstance(item.get("name"), str)}
    audit_names = {item["hf_name"] for item in records_to_check}
    if names != audit_names:
        only_census = sorted(names - audit_names)
        only_audit = sorted(audit_names - names)
        raise AuditError(f"census mismatch only_census={only_census} only_audit={only_audit}")
    return "PASS"


def check() -> None:
    baseline = load_json(BASELINE)
    if baseline.get("model_revision") != MODEL_REVISION or baseline.get("selected_official_llamacpp_revision") != OFFICIAL_REVISION:
        raise AuditError("llamacpp baseline revision drift")
    expected_mapping, expected_tiers, expected_hypotheses, expected_lineage = payloads()
    for path, expected in ((MAPPING, expected_mapping), (TIERS, expected_tiers), (HYPOTHESES, expected_hypotheses)):
        actual = load_json(path)
        if actual != expected:
            raise AuditError(f"stale generated artifact: {path.relative_to(ROOT)}")
    if LINEAGE.read_text(encoding="utf-8") != expected_lineage:
        raise AuditError(f"stale generated artifact: {LINEAGE.relative_to(ROOT)}")
    audit_records = expected_mapping["records"]
    if len(audit_records) != 201:
        raise AuditError(f"expected 201 records, got {len(audit_records)}")
    if len({item["hf_name"] for item in audit_records}) != 201 or len({item["gguf_name"] for item in audit_records}) != 201:
        raise AuditError("mapping has duplicate source or GGUF names")
    for record in audit_records:
        log(
            "MAP "
            f"hf={record['hf_name']} gguf={record['gguf_name']} "
            f"transform={record['transform']} merge={record['merge']} "
            f"q8={record['q8_0_source_rule']} q4km={record['q4_k_m_source_rule']}"
        )
    census_result = verify_census(audit_records)
    digest = hashlib.sha256(canonical(expected_mapping)).hexdigest()
    log(f"RESULT=PASS tensors=201 census={census_result} mapping_sha256={digest}")


def self_test() -> None:
    generated = records()
    if len(generated) != 201 or generated[0]["hf_name"] != "lm_head.weight":
        raise AuditError("generated mapping cardinality/order invariant failed")
    q6_v = sorted(item["layer"] for item in generated if item["class"] == "v" and item["q4_k_m_source_rule"] == "Q6_K")
    if q6_v != [0, 1, 4, 7, 10, 13, 16, 19, 20, 21]:
        raise AuditError(f"Q4_K_M V override invariant failed: {q6_v}")
    if any(item["q8_0_source_rule"] != "UNQUANTIZED" for item in generated if item["class"] == "norm"):
        raise AuditError("norm quantization invariant failed")
    try:
        json.loads('{"x": 1, "x": 2}', object_pairs_hook=no_duplicate_pairs)
    except AuditError:
        pass
    else:
        raise AuditError("duplicate-key rejection invariant failed")
    log("RESULT=PASS mode=self-test tensors=201 duplicate_keys=rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.generate:
            generate()
        elif args.check:
            check()
        else:
            self_test()
    except AuditError as error:
        log(f"RESULT=FAIL detail={error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
