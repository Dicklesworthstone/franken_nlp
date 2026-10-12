//! Exact schema for the namespaced selftest NDJSON records, not task events.
use serde_json::{Value, json};
use crate::native_engine::{dispatch::KernelTier, selftest::{CASES_PER_TIER,
    SELFTEST_SCHEMA_VERSION, SELFTEST_SUITE_VERSION}};
use super::PROTOCOL;

fn object(properties: Value) -> Value {
    // All call sites below supply literal objects; external JSON is never read.
    let required: Vec<String> = properties.as_object().expect("literal schema object").keys().cloned().collect();
    json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false})
}
fn nullable(schema: Value) -> Value { json!({"anyOf": [schema, {"type": "null"}]}) }
fn record(name: &str, data: Value) -> Value {
    object(json!({
        "schema_version": {"const": SELFTEST_SCHEMA_VERSION},
        "protocol": {"const": PROTOCOL}, "event": {"const": name}, "data": data,
    }))
}

pub(super) fn document() -> Value {
    let tier = json!({"enum": KernelTier::ALL});
    let count = json!({"type": "integer", "minimum": 0, "maximum": KernelTier::ALL.len() * CASES_PER_TIER});
    let digest = json!({"type": "string", "pattern": "^[0-9a-f]{64}$"});
    let kernel = json!({"type": "string", "minLength": 1});
    let features = object(json!({
        "architecture": {"enum": ["aarch64", "x86_64", "other"]},
        "aarch64_dotprod": {"type": "boolean"}, "aarch64_i8mm": {"type": "boolean"},
        "x86_avx2": {"type": "boolean"}, "x86_avx512f": {"type": "boolean"},
        "x86_avx512vl": {"type": "boolean"}, "x86_avx512vnni": {"type": "boolean"},
        "x86_avxvnni": {"type": "boolean"},
    }));
    let start = record("selftest_start", object(json!({
        "suite_version": {"const": SELFTEST_SUITE_VERSION},
        "requested_tier": nullable(tier.clone()),
        "cases_per_tier": {"const": CASES_PER_TIER}, "detected_features": features,
    })));
    let mismatch = object(json!({
        "row": {"type": "integer", "minimum": 0, "maximum": 4},
        "column": {"type": "integer", "minimum": 0, "maximum": 2},
        "expected": {"type": "integer", "minimum": -2147483648_i64, "maximum": 2147483647_i64},
        "actual": {"type": "integer", "minimum": -2147483648_i64, "maximum": 2147483647_i64},
    }));
    let case = record("selftest_case", object(json!({
        "tier": tier.clone(), "operation": {"enum": ["gemm", "gemv", "cohort"]},
        "pattern": {"enum": ["extremes", "mixed"]},
        "m": {"type": "integer", "minimum": 1, "maximum": 5},
        "k": {"type": "integer", "minimum": 0, "maximum": 10752},
        "n": {"type": "integer", "minimum": 1, "maximum": 3},
        "requested_kernel": kernel.clone(), "observed_kernel": nullable(kernel),
        "expected_sha256": digest.clone(), "actual_sha256": nullable(digest),
        "failure": nullable(json!({"enum": ["dispatch", "kernel", "wrong_kernel", "output_shape", "integer_mismatch"]})),
        "first_mismatch": nullable(mismatch),
    })));
    let tier_summary = record("selftest_tier", object(json!({
        "tier": tier.clone(), "status": {"enum": ["pass", "fail", "skipped_unsupported_isa", "not_implemented"]},
        "cases_run": {"type": "integer", "minimum": 0, "maximum": CASES_PER_TIER},
        "cases_passed": {"type": "integer", "minimum": 0, "maximum": CASES_PER_TIER},
    })));
    let complete = record("selftest_complete", object(json!({
        "suite_version": {"const": SELFTEST_SUITE_VERSION},
        "scope": {"const": "finite_synthetic_integer_cases_only_not_model_or_performance_qualification"},
        "requested_tier": nullable(tier), "verdict": {"enum": ["pass", "fail", "unsupported"]},
        "tiers_considered": {"type": "integer", "minimum": 1, "maximum": KernelTier::ALL.len()},
        "tiers_skipped": {"type": "integer", "minimum": 0, "maximum": KernelTier::ALL.len()},
        "cases_run": count.clone(), "cases_passed": count,
        "cases_per_tier": {"const": CASES_PER_TIER},
    })));
    let error = record("selftest_error", object(json!({
        "code": {"enum": ["allocation_refused", "oracle_overflow"]},
    })));
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:franken-nlp:cpu-selftest:1",
        "title": "FrankenNLP finite CPU selftest event protocol v1",
        "description": "A pass covers only the executed synthetic integer cases on this binary and CPU; not model parity, speed, or release qualification.",
        "oneOf": [start, case, tier_summary, complete, error],
    })
}
