#!/usr/bin/env bash
# The sole CI and central batch-verification entrypoint for this repository.
#
# CHECK_UBS_TIMEOUT_S bounds `ubs --diff` (default: 180 seconds).  UBS is
# intentionally optional only when its binary is absent; every other gate is
# mandatory once the uxw scaffold has landed.
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"
readonly CHECK_LOG_DIR="${CHECK_LOG_DIR:-${CARGO_TARGET_DIR}/check-logs}"
readonly CHECK_RUN_ID="${CHECK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
readonly CHECK_UBS_TIMEOUT_S="${CHECK_UBS_TIMEOUT_S:-180}"
export CARGO_TARGET_DIR

CHECK_SECTIONS=0
CHECK_SKIPPED=0

usage_error() {
    printf 'CHECK section=bootstrap result=FAIL time_s=0 lines=1 reason=%s\n' "$1" >&2
    exit 2
}

case "${CHECK_UBS_TIMEOUT_S}" in
    ''|*[!0-9]*) usage_error 'CHECK_UBS_TIMEOUT_S must be a positive integer' ;;
    0) usage_error 'CHECK_UBS_TIMEOUT_S must be greater than zero' ;;
esac

mkdir -p "${CHECK_LOG_DIR}"
cd "${REPO_ROOT}"

elapsed_seconds() {
    local started_at="$1"
    local finished_at
    finished_at="$(date +%s)"
    printf '%s' "$((finished_at - started_at))"
}

emit_section() {
    local section="$1"
    local result="$2"
    local time_s="$3"
    local lines="$4"
    printf 'CHECK section=%s result=%s time_s=%s lines=%s\n' \
        "${section}" "${result}" "${time_s}" "${lines}"
}

run_section() {
    local section="$1"
    shift
    local started_at
    local time_s
    local status
    local result
    local log_path
    local lines

    started_at="$(date +%s)"
    log_path="${CHECK_LOG_DIR}/${CHECK_RUN_ID}-${section}.log"
    printf 'CHECK section=%s result=RUNNING time_s=0 lines=0\n' "${section}"

    if "$@" >"${log_path}" 2>&1; then
        status=0
    else
        status=$?
    fi
    cat "${log_path}"
    lines="$(wc -l < "${log_path}" | tr -d '[:space:]')"
    time_s="$(elapsed_seconds "${started_at}")"
    CHECK_SECTIONS=$((CHECK_SECTIONS + 1))

    if [[ "${status}" -eq 0 ]]; then
        result=PASS
    elif [[ "${status}" -eq 77 ]]; then
        result=SKIPPED_NO_UBS
        CHECK_SKIPPED=$((CHECK_SKIPPED + 1))
    else
        result=FAIL
    fi
    emit_section "${section}" "${result}" "${time_s}" "${lines}"

    if [[ "${status}" -ne 0 && "${status}" -ne 77 ]]; then
        printf 'CHECK RESULT=FAIL failed_section=%s exit_code=%s\n' "${section}" "${status}" >&2
        return "${status}"
    fi
}

cargo_check_without_duplicate_target_warning() {
    local output_path="${CHECK_LOG_DIR}/${CHECK_RUN_ID}-cargo-check-output.log"
    local status

    if cargo check --locked --all-targets >"${output_path}" 2>&1; then
        status=0
    else
        status=$?
    fi
    cat "${output_path}"
    if [[ "${status}" -ne 0 ]]; then
        return "${status}"
    fi
    if grep -Fq 'present in multiple build targets' "${output_path}"; then
        printf '%s\n' 'cargo check emitted the forbidden multiple-build-targets warning' >&2
        return 1
    fi
}

validate_doc_links() {
    command -v python3 >/dev/null 2>&1 || {
        printf '%s\n' 'python3 is required for the documentation-link validator' >&2
        return 1
    }
    python3 - "${REPO_ROOT}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
markdown_files = sorted(
    path for path in root.rglob("*.md")
    if ".git" not in path.parts and "target" not in path.parts
)
markdown_link = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
plain_cross_reference = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:docs/[A-Za-z0-9_./-]+|README|AGENTS|COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP)\.md)(?![A-Za-z0-9_./-])"
)


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9 _-]", "", value)
    return re.sub(r"[ _]+", "-", value).strip("-")


def anchors(path: Path) -> set[str]:
    return {
        slug(match.group(1))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line))
    }


def ignored(target: str) -> bool:
    return (
        not target
        or target.startswith(("#", "http://", "https://", "mailto:", "data:"))
        or "://" in target
    )


def check_target(source: Path, line_number: int, target: str) -> list[str]:
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split(maxsplit=1)[0]
    if ignored(target):
        return []
    path_text, separator, anchor = target.partition("#")
    destination = (source.parent / path_text).resolve() if path_text else source
    try:
        destination.relative_to(root)
    except ValueError:
        return [f"{source.relative_to(root)}:{line_number}: link escapes repository: {target}"]
    if not destination.exists():
        return [f"{source.relative_to(root)}:{line_number}: dead markdown link: {target}"]
    if anchor and destination.is_file() and destination.suffix.lower() == ".md":
        if slug(anchor) not in anchors(destination):
            return [f"{source.relative_to(root)}:{line_number}: dead markdown anchor: {target}"]
    return []


errors: list[str] = []
for source in markdown_files:
    fenced = False
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for match in markdown_link.finditer(line):
            errors.extend(check_target(source, line_number, match.group(1)))
        for match in plain_cross_reference.finditer(line):
            errors.extend(check_target(source, line_number, match.group(1)))

if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"DOC_LINKS RESULT=PASS files={len(markdown_files)}")
PY
}

validate_platform_surfaces() {
    command -v python3 >/dev/null 2>&1 || {
        printf '%s\n' 'python3 is required for the platform-surface validator' >&2
        return 1
    }
    python3 "${REPO_ROOT}/scripts/validate_platform_surfaces.py"
}

run_bounded_ubs() {
    if ! command -v ubs >/dev/null 2>&1; then
        printf 'UBS RESULT=SKIPPED_NO_UBS reason=binary-absent\n' >&2
        return 77
    fi
    if command -v timeout >/dev/null 2>&1; then
        timeout "${CHECK_UBS_TIMEOUT_S}" ubs --diff
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "${CHECK_UBS_TIMEOUT_S}" ubs --diff
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "${CHECK_UBS_TIMEOUT_S}" <<'PY'
import subprocess
import sys

try:
    completed = subprocess.run(["ubs", "--diff"], check=False, timeout=int(sys.argv[1]))
except subprocess.TimeoutExpired:
    print(f"UBS RESULT=FAIL reason=timeout limit_s={sys.argv[1]}", file=sys.stderr)
    raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
    else
        printf 'UBS RESULT=FAIL reason=no-timeout-runner limit_s=%s\n' "${CHECK_UBS_TIMEOUT_S}" >&2
        return 1
    fi
}

run_optional_policy_section() {
    local section="$1"
    local script_path="$2"
    if [[ -f "${script_path}" ]]; then
        if [[ ! -x "${script_path}" ]]; then
            printf 'CHECK section=%s result=FAIL time_s=0 lines=1 reason=not-executable\n' "${section}" >&2
            return 1
        fi
        run_section "${section}" "${script_path}"
        return
    fi
    CHECK_SECTIONS=$((CHECK_SECTIONS + 1))
    CHECK_SKIPPED=$((CHECK_SKIPPED + 1))
    printf 'CHECK section=%s result=SKIPPED_NOT_LANDED time_s=0 lines=0\n' "${section}"
}

main() {
    run_section fmt cargo fmt --check
    run_section cargo-check cargo_check_without_duplicate_target_warning
    run_section clippy cargo clippy --locked --all-targets -- -D warnings
    run_section test cargo test --locked
    run_section doc-links validate_doc_links
    run_section adr-registry python3 scripts/validate_adrs.py
    run_section platform-surfaces validate_platform_surfaces
    run_section ubs run_bounded_ubs

    # These dedicated policy scripts become mandatory as their sibling beads land.
    run_optional_policy_section lint-policy "${REPO_ROOT}/scripts/check_lint_policy.sh"
    run_optional_policy_section claims "${REPO_ROOT}/scripts/check_claims.sh"
    run_optional_policy_section behavior-notes "${REPO_ROOT}/scripts/lint_behavior_notes.sh"
    run_optional_policy_section dependency-policy "${REPO_ROOT}/scripts/check_dependency_policy.sh"
    run_optional_policy_section suite-lock "${REPO_ROOT}/scripts/check_suite_lock.sh"
    run_optional_policy_section toolchain-policy "${REPO_ROOT}/scripts/check_toolchain.sh"

    printf 'CHECK RESULT=PASS sections=%s skipped=%s\n' "${CHECK_SECTIONS}" "${CHECK_SKIPPED}"
}

main "$@"
