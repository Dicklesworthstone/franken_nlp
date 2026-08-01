#!/bin/sh
# Model-free robot-plan contract exercise for franken_nlp-17h.
#
# Usage: scripts/e2e_robot_plan_admission.sh /absolute/path/to/fnlp
#
# This checks the machine-readable term/rejection contract only. It does not
# claim allocator instrumentation: that requires the separately retained
# central allocation-count harness for the named binary.
set -eu

if [ "$#" -ne 1 ]; then
    printf '%s\n' "usage: $0 /absolute/path/to/fnlp" >&2
    exit 64
fi

fnlp_bin=$1

run_case() {
    case_name=$1
    expected_rejection=$2
    shift 2
    output=$("$fnlp_bin" robot plan "$@")
    printf '%s\n' "$output" | python3 -c '
import json
import sys

document = json.load(sys.stdin)
case_name, expected_rejection = sys.argv[1:3]
required_terms = (
    "memory_budget_total_bytes", "memory_reserve_os_bytes", "fixed_residency",
    "elastic_cache_bytes", "replicated_weight_residency", "kv_payload_bytes",
    "kv_scale_bytes", "kv_page_metadata_bytes", "activation_bytes",
    "full_logit_bytes", "grammar_state_bytes", "source_state_bytes",
    "queue_bytes", "output_buffer_bytes", "unmodeled_emergency_reserve_bytes",
    "safety_margin_bytes", "committed_bytes", "peak_bytes",
    "aggregate_available_ledger_bytes",
)
assert document["kind"] == "robot_plan", (case_name, document)
assert document["allocations"] == "none", (case_name, document)
assert document["schema_version"] == 3, (case_name, document)
assert set(required_terms).issubset(document["terms"]), (case_name, document)
rejection = document.get("rejection")
assert rejection is not None, (case_name, document)
assert rejection["code"] == expected_rejection, (case_name, document)
rejection_code = rejection["code"]
kv_payload_bytes = document["terms"]["kv_payload_bytes"]
full_logit_bytes = document["terms"]["full_logit_bytes"]
print(
    "ROBOT_PLAN_E2E"
    f" case={case_name}"
    f" result=PASS"
    f" rejection={rejection_code}"
    f" kv_payload_bytes={kv_payload_bytes}"
    f" full_logit_bytes={full_logit_bytes}"
)
' "$case_name" "$expected_rejection"
}

# The 8192-token bf16 row contains exactly 1,476,395,008 bytes of 44-slot KV.
# The deliberately one-byte-short budget demonstrates the first violated term
# after fixed residency and the complete K/V term have been counted.
run_case \
    bf16_8192 \
    local_budget_exceeded \
    --ctx 8192 \
    --batch 1 \
    --quant bf16 \
    --memory-budget 1477059584 \
    --fixed-mapped-bytes 1 \
    --fixed-resident-bytes 1 \
    --kv-page-metadata-per-token 0 \
    --unmodeled-emergency-reserve-bytes 0 \
    --safety-margin-bytes 0

# Int8 retains its scales in the certificate and does not round the 64-row
# batch down to a guessed width. The small local budget forces an exact,
# non-allocating refusal while every term remains visible.
run_case \
    int8_8192_batch64 \
    local_budget_exceeded \
    --ctx 8192 \
    --batch 64 \
    --quant int8 \
    --memory-budget 1 \
    --fixed-mapped-bytes 1 \
    --fixed-resident-bytes 1 \
    --kv-page-metadata-per-token 16 \
    --unmodeled-emergency-reserve-bytes 0 \
    --safety-margin-bytes 0

# Checked arithmetic rejects a hostile request before it can wrap into a small
# capacity claim or create a reservation.
run_case \
    overflow \
    arithmetic_overflow \
    --ctx 18446744073709551615 \
    --batch 2 \
    --quant bf16
