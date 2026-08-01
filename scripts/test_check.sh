#!/usr/bin/env bash
# Hermetic behavior tests for scripts/check.sh.  Scratch trees are preserved so
# a failure can be inspected without deleting any operator or test evidence.
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CHECK_SOURCE="${REPO_ROOT}/scripts/check.sh"
readonly TENSOR_CENSUS_GENERATOR_SOURCE="${REPO_ROOT}/scripts/gen_tensor_census.py"
readonly TENSOR_CENSUS_ARTIFACT_SOURCE="${REPO_ROOT}/docs/truth-pack/tensor_census.json"
readonly SYSTEM_PATH="/usr/bin:/bin"

CASE_COUNT=0
FAILED_CASES=()

fail_case() {
    local case_name="$1"
    FAILED_CASES+=("${case_name}")
    printf 'CHECK_SELFTEST case=%s result=FAIL\n' "${case_name}" >&2
}

make_fixture_tree() {
    local case_name="$1"
    local fixture_root
    fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/franken-nlp-check-${case_name}.XXXXXX")"
    mkdir -p "${fixture_root}/scripts" "${fixture_root}/bin" "${fixture_root}/docs/truth-pack"
    cp "${CHECK_SOURCE}" "${fixture_root}/scripts/check.sh"
    cp "${TENSOR_CENSUS_GENERATOR_SOURCE}" "${fixture_root}/scripts/gen_tensor_census.py"
    cp "${TENSOR_CENSUS_ARTIFACT_SOURCE}" "${fixture_root}/docs/truth-pack/tensor_census.json"
    chmod +x "${fixture_root}/scripts/check.sh"
    cat >"${fixture_root}/bin/cargo" <<'CARGO'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "fmt" && "${FAKE_CARGO_FAIL_FMT:-0}" == "1" ]]; then
    printf '%s\n' 'formatting violation' >&2
    exit 1
fi
if [[ "${1:-}" == "check" && "${FAKE_CARGO_DUPLICATE_TARGET_WARNING:-0}" == "1" ]]; then
    printf '%s\n' 'warning: file is present in multiple build targets'
fi
exit 0
CARGO
    chmod +x "${fixture_root}/bin/cargo"
    printf '%s\n' '# Fixture' >"${fixture_root}/README.md"
    printf '%s\n' '# Fixture docs' >"${fixture_root}/docs/fixture.md"
    printf '%s' "${fixture_root}"
}

run_fixture_check() {
    local fixture_root="$1"
    local output_path="$2"
    shift 2
    if PATH="${fixture_root}/bin:${SYSTEM_PATH}" CHECK_LOG_DIR="${fixture_root}/logs" "$@" \
        "${fixture_root}/scripts/check.sh" >"${output_path}" 2>&1; then
        return 0
    fi
    return $?
}

assert_contains() {
    local path="$1"
    local needle="$2"
    grep -Fq "${needle}" "${path}"
}

case_clean_scaffold() {
    local fixture_root output_path
    fixture_root="$(make_fixture_tree clean)"
    output_path="${fixture_root}/clean.log"
    if ! run_fixture_check "${fixture_root}" "${output_path}" env; then
        fail_case clean-scaffold
        return
    fi
    for section in fmt tensor-census cargo-check clippy test doc-links; do
        if ! assert_contains "${output_path}" "CHECK section=${section} result=PASS"; then
            fail_case clean-scaffold
            return
        fi
    done
    if ! assert_contains "${output_path}" 'CHECK section=ubs result=SKIPPED_NO_UBS'; then
        fail_case clean-scaffold
        return
    fi
    printf 'CHECK_SELFTEST case=clean-scaffold result=PASS scratch=%s\n' "${fixture_root}"
}

case_tensor_census_drift_fails_before_cargo() {
    local fixture_root output_path
    fixture_root="$(make_fixture_tree tensor-census-drift)"
    output_path="${fixture_root}/tensor-census-drift.log"
    printf '\n' >>"${fixture_root}/docs/truth-pack/tensor_census.json"
    if run_fixture_check "${fixture_root}" "${output_path}" env; then
        fail_case tensor-census-drift
        return
    fi
    if ! assert_contains "${output_path}" 'CHECK section=tensor-census result=FAIL' \
        || assert_contains "${output_path}" 'CHECK section=cargo-check result='; then
        fail_case tensor-census-drift
        return
    fi
    printf 'CHECK_SELFTEST case=tensor-census-drift result=PASS scratch=%s\n' "${fixture_root}"
}

case_fmt_failure_is_fail_fast() {
    local fixture_root output_path
    fixture_root="$(make_fixture_tree fmt-fail)"
    output_path="${fixture_root}/fmt-fail.log"
    if run_fixture_check "${fixture_root}" "${output_path}" env FAKE_CARGO_FAIL_FMT=1; then
        fail_case fmt-fail-fast
        return
    fi
    if ! assert_contains "${output_path}" 'CHECK section=fmt result=FAIL' \
        || assert_contains "${output_path}" 'CHECK section=cargo-check result='; then
        fail_case fmt-fail-fast
        return
    fi
    printf 'CHECK_SELFTEST case=fmt-fail-fast result=PASS scratch=%s\n' "${fixture_root}"
}

case_dead_doc_link_fails() {
    local fixture_root output_path
    fixture_root="$(make_fixture_tree doc-link-fail)"
    output_path="${fixture_root}/doc-link-fail.log"
    printf '%s\n' '[broken](docs/missing.md)' >>"${fixture_root}/README.md"
    if run_fixture_check "${fixture_root}" "${output_path}" env; then
        fail_case doc-link-fail
        return
    fi
    if ! assert_contains "${output_path}" 'README.md:2: dead markdown link: docs/missing.md' \
        || ! assert_contains "${output_path}" 'CHECK section=doc-links result=FAIL'; then
        fail_case doc-link-fail
        return
    fi
    printf 'CHECK_SELFTEST case=doc-link-fail result=PASS scratch=%s\n' "${fixture_root}"
}

case_duplicate_target_warning_fails() {
    local fixture_root output_path
    fixture_root="$(make_fixture_tree warning-fail)"
    output_path="${fixture_root}/warning-fail.log"
    if run_fixture_check "${fixture_root}" "${output_path}" env FAKE_CARGO_DUPLICATE_TARGET_WARNING=1; then
        fail_case warning-grep
        return
    fi
    if ! assert_contains "${output_path}" 'CHECK section=cargo-check result=FAIL' \
        || ! assert_contains "${output_path}" 'forbidden multiple-build-targets warning'; then
        fail_case warning-grep
        return
    fi
    printf 'CHECK_SELFTEST case=warning-grep result=PASS scratch=%s\n' "${fixture_root}"
}

main() {
    for case_function in \
        case_clean_scaffold \
        case_tensor_census_drift_fails_before_cargo \
        case_fmt_failure_is_fail_fast \
        case_dead_doc_link_fails \
        case_duplicate_target_warning_fails; do
        CASE_COUNT=$((CASE_COUNT + 1))
        "${case_function}"
    done

    if [[ "${#FAILED_CASES[@]}" -ne 0 ]]; then
        printf 'CHECK_SELFTEST RESULT=FAIL cases=%s failed=%s\n' \
            "${CASE_COUNT}" "$(IFS=,; printf '%s' "${FAILED_CASES[*]}")" >&2
        return 1
    fi
    printf 'CHECK_SELFTEST RESULT=PASS cases=%s\n' "${CASE_COUNT}"
}

main "$@"
