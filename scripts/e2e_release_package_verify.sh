#!/bin/sh
# Model-package closure verification and tamper-detection transcript driver.
#
# This script intentionally requires a prebuilt package and never runs Cargo.
# It operates on a fresh temporary copy for the tamper leg and leaves that copy
# behind on failure for inspection.
set -eu

if [ "$#" -ne 2 ] || [ "$1" != "--package-dir" ]; then
    echo "usage: $0 --package-dir /absolute/path/release-staging" >&2
    exit 2
fi

package_dir=$2
fnlp_bin=${FNLP_BIN:-fnlp}
case $package_dir in
    /*) ;;
    *) echo "E2E_SUMMARY result=FAIL reason=package-dir-must-be-absolute" >&2; exit 2 ;;
esac

start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "RELEASE_E2E stage=verify-start timestamp=$start package_dir=$package_dir" >&2
"$fnlp_bin" release verify-model-package --staging-dir "$package_dir"
echo "RELEASE_E2E stage=verify-pass timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2

first_part=$(awk -F'"' '/"name": ".*\.part[0-9][0-9]"/{print $4; exit}' "$package_dir/MODEL_ASSET_RECEIPT.json")
if [ -z "$first_part" ]; then
    echo "RELEASE_E2E stage=tamper-skip reason=zero-length-artifact" >&2
    echo "E2E_SUMMARY result=PASS parts=0 tamper=SKIPPED_ZERO_ARTIFACT" >&2
    exit 0
fi

tampered_dir=$(mktemp -d "${TMPDIR:-/tmp}/fnlp-release-package-tamper.XXXXXXXX")
cp -R "$package_dir/." "$tampered_dir/"
printf '\001' >> "$tampered_dir/$first_part"
echo "RELEASE_E2E stage=tamper part=$first_part expected=verification-failure" >&2
if "$fnlp_bin" release verify-model-package --staging-dir "$tampered_dir"; then
    echo "E2E_SUMMARY result=FAIL part=$first_part reason=tamper-was-accepted" >&2
    exit 1
fi
echo "RELEASE_E2E stage=tamper-rejected part=$first_part timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
echo "E2E_SUMMARY result=PASS part=$first_part tamper=REJECTED" >&2
