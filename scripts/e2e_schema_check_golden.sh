#!/bin/sh
# Model-free CLI golden runner for the bounded v1 schema compiler.
#
# Set FNLP_BIN to an already built fnlp executable to avoid Cargo.  Without it,
# this runnable uses `cargo run --bin fnlp` for a local developer invocation.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FIXTURE_ROOT="$REPO_ROOT/tests/fixtures/schema"
CASE_COUNT=0
FAILED_CASES=0

run_fnlp() {
    if [ -n "${FNLP_BIN:-}" ]; then
        "$FNLP_BIN" "$@"
    else
        cargo run --quiet --bin fnlp -- "$@"
    fi
}

record_pass() {
    fixture=$1
    expected=$2
    output=$3
    CASE_COUNT=$((CASE_COUNT + 1))
    printf 'SCHEMA_GOLDEN fixture=%s expected=%s result=PASS output=%s\n' \
        "$fixture" "$expected" "$(printf '%s' "$output" | tr '\n' ' ')"
}

record_fail() {
    fixture=$1
    expected=$2
    output=$3
    CASE_COUNT=$((CASE_COUNT + 1))
    FAILED_CASES=$((FAILED_CASES + 1))
    printf 'SCHEMA_GOLDEN fixture=%s expected=%s result=FAIL output=%s\n' \
        "$fixture" "$expected" "$(printf '%s' "$output" | tr '\n' ' ')" >&2
}

expect_accept() {
    fixture=$1
    check_output=$(run_fnlp schema check "$FIXTURE_ROOT/$fixture" 2>&1) || {
        record_fail "$fixture" accept "$check_output"
        return
    }
    printf '%s\n' "$check_output" | grep -F 'SCHEMA mode=check RESULT=PASS' >/dev/null || {
        record_fail "$fixture" accept "$check_output"
        return
    }
    sample_output=$(run_fnlp schema sample "$FIXTURE_ROOT/$fixture" 2>&1) || {
        record_fail "$fixture" sample "$sample_output"
        return
    }
    printf '%s\n' "$sample_output" | grep -F 'SCHEMA mode=sample RESULT=PASS' >/dev/null || {
        record_fail "$fixture" sample "$sample_output"
        return
    }
    record_pass "$fixture" accept "$check_output"
}

expect_reject() {
    fixture=$1
    keyword=$2
    if output=$(run_fnlp schema check "$FIXTURE_ROOT/$fixture" 2>&1); then
        record_fail "$fixture" "reject:$keyword" "$output"
        return
    fi
    printf '%s\n' "$output" | grep -F "SCHEMA mode=check RESULT=FAIL" >/dev/null \
        && printf '%s\n' "$output" | grep -F "keyword=$keyword" >/dev/null || {
        record_fail "$fixture" "reject:$keyword" "$output"
        return
    }
    record_pass "$fixture" "reject:$keyword" "$output"
}

expect_accept accept-object.schema.json
expect_reject reject-ref.schema.json '$ref'
expect_reject reject-duplicate-number-enum.schema.json none

if [ "$FAILED_CASES" -ne 0 ]; then
    printf 'SCHEMA_GOLDEN RESULT=FAIL cases=%s failures=%s\n' "$CASE_COUNT" "$FAILED_CASES" >&2
    exit 1
fi
printf 'SCHEMA_GOLDEN RESULT=PASS cases=%s failures=0\n' "$CASE_COUNT"
