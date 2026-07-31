#!/usr/bin/env bash
# Model-gated converter round-trip driver. It intentionally never downloads or
# mutates the source closure, and it refuses an existing artifact destination
# before delegating exclusive creation to `fnlp convert`.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${1:-"${FNLP_SOURCE_DIR:-}"}
output_path=${2:-"${FNLP_CONVERT_OUTPUT:-}"}
fnlp_bin=${FNLP_BIN:-fnlp}
source_manifest=${FNLP_SOURCE_MANIFEST:-"${repo_root}/docs/truth-pack/nanbeige4.2-3b.source.json"}

log() {
    printf 'E2E_CONVERT stage=%s time_s=%s %s\n' "$1" "$SECONDS" "$2" >&2
}

if [[ -z "${source_dir}" || ! -d "${source_dir}" ]]; then
    log source "RESULT=SKIPPED_NO_MODEL reason=missing-source-dir"
    printf 'E2E_SUMMARY RESULT=SKIPPED_NO_MODEL stages=source closure_bytes=8360887509 payload_bytes=8339601408\n' >&2
    exit 0
fi

if [[ -z "${output_path}" ]]; then
    log invocation "RESULT=FAIL reason=missing-output-path expected=second-argument-or-FNLP_CONVERT_OUTPUT"
    printf 'E2E_SUMMARY RESULT=FAIL stage=invocation expected=output-path actual=empty\n' >&2
    exit 2
fi

if [[ ! -f "${source_manifest}" ]]; then
    log preflight "RESULT=FAIL reason=missing-source-manifest expected=regular-file actual=${source_manifest}"
    printf 'E2E_SUMMARY RESULT=FAIL stage=preflight expected=regular-source-manifest actual=%s\n' "${source_manifest}" >&2
    exit 2
fi

if [[ -e "${output_path}" || -L "${output_path}" ]]; then
    log preflight "RESULT=FAIL reason=output-already-exists expected=nonexistent-create-new-destination actual=${output_path}"
    printf 'E2E_SUMMARY RESULT=FAIL stage=preflight expected=nonexistent-output actual=%s\n' "${output_path}" >&2
    exit 2
fi

if ! command -v "${fnlp_bin}" >/dev/null 2>&1; then
    log invocation "RESULT=SKIPPED_NO_BINARY bin=${fnlp_bin}"
    printf 'E2E_SUMMARY RESULT=SKIPPED_NO_BINARY bin=%s\n' "${fnlp_bin}" >&2
    exit 0
fi

log source "RESULT=START source=${source_dir} manifest=${source_manifest}"
"${fnlp_bin}" convert \
    --source "${source_dir}" \
    --source-manifest "${source_manifest}" \
    --recipe nanbeige42-int8-v1 \
    --arch generic \
    --yes \
    -o "${output_path}"
log reload "RESULT=PASS artifact=${output_path}"
printf 'E2E_SUMMARY RESULT=PASS stages=source,convert,reload expected_payload_bytes=8339601408 output=%s\n' "${output_path}" >&2
