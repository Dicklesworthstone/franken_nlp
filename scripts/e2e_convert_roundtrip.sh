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
set +e
"${fnlp_bin}" convert \
    --source "${source_dir}" \
    --source-manifest "${source_manifest}" \
    --recipe nanbeige42-int8-v1 \
    --arch generic \
    --yes \
    -o "${output_path}"
convert_status=$?
set -e

if [[ ${convert_status} -ne 0 ]]; then
    log convert "RESULT=INCOMPLETE exit=${convert_status} reason=converter-command-failed"
    printf 'E2E_SUMMARY RESULT=INCOMPLETE stage=convert exit=%s expected=successful-conversion-receipt actual=converter-nonzero\n' "${convert_status}" >&2
    exit "${convert_status}"
fi

if [[ ! -f "${output_path}" || -L "${output_path}" || ! -s "${output_path}" ]]; then
    log artifact "RESULT=FAIL reason=missing-or-invalid-created-artifact expected=nonempty-regular-file actual=${output_path}"
    printf 'E2E_SUMMARY RESULT=FAIL stage=artifact expected=nonempty-regular-output actual=%s\n' "${output_path}" >&2
    exit 3
fi

artifact_bytes=$(wc -c < "${output_path}" | tr -d '[:space:]')
if command -v sha256sum >/dev/null 2>&1; then
    artifact_sha256=$(sha256sum "${output_path}" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    artifact_sha256=$(shasum -a 256 "${output_path}" | awk '{print $1}')
else
    log artifact "RESULT=INCOMPLETE reason=missing-sha256-utility"
    printf 'E2E_SUMMARY RESULT=INCOMPLETE stage=artifact expected=sha256sum-or-shasum actual=unavailable\n' >&2
    exit 3
fi

# A successful process exit and a newly created file do not demonstrate that
# the staged artifact was re-opened, census-checked, or round-tripped.  This
# driver must stay fail-closed until `fnlp convert` exposes a machine-readable
# reload-verification receipt that the driver can inspect.
log artifact "RESULT=CREATED artifact=${output_path} bytes=${artifact_bytes} sha256=${artifact_sha256}"
log reload "RESULT=BLOCKED reason=no-machine-readable-reload-verification-receipt"
printf 'E2E_SUMMARY RESULT=INCOMPLETE stages=source,convert,artifact,reload expected_payload_bytes=8339601408 output=%s output_bytes=%s output_sha256=%s next=add-and-verify-reload-receipt\n' "${output_path}" "${artifact_bytes}" "${artifact_sha256}" >&2
exit 3
