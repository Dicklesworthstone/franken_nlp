#!/usr/bin/env bash
# Compare the running compiler identity and target defaults against xym's pin.
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EXPECTED_PATH="${REPO_ROOT}/ci/toolchain.expected.json"

if [[ ! -f "${EXPECTED_PATH}" ]]; then
    printf 'TOOLCHAIN RESULT=FAIL drift=missing_expectation\n' >&2
    exit 1
fi

python3 - "${EXPECTED_PATH}" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

expected_path = Path(sys.argv[1])
expected = json.loads(expected_path.read_text(encoding="utf-8"))
compiler = expected["compiler_identity"]


def run(*command: str) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


rustc_vv = run("rustc", "-Vv")
cfg = run("rustc", "--print", "cfg")
catalogue = run("rustc", "--print", "target-features")
print("TOOLCHAIN rustc_vv_begin")
print(rustc_vv, end="")
print("TOOLCHAIN enabled_target_features_begin")
print("\n".join(line for line in cfg.splitlines() if line.startswith("target_feature=")))
print("TOOLCHAIN target_feature_catalogue_begin")
print(catalogue, end="")

observed_vv: dict[str, str] = {}
for line in rustc_vv.splitlines():
    if ": " in line:
        key, value = line.split(": ", 1)
        observed_vv[key] = value

drifts: list[tuple[str, str, str]] = []
for key in ("release", "commit-hash", "commit-date", "LLVM version"):
    expected_key = {
        "commit-hash": "commit_hash",
        "commit-date": "commit_date",
        "LLVM version": "llvm_version",
    }.get(key, key)
    observed = observed_vv.get(key, "<missing>")
    wanted = compiler[expected_key]
    if observed != wanted:
        drifts.append((key, wanted, observed))

host = observed_vv.get("host", "<missing>")
targets = {target["triple"]: target for target in expected["release_targets"]}
target = targets.get(host)
if target is None:
    drifts.append(("target_triple", ",".join(sorted(targets)), host))
else:
    observed_features = sorted(
        line.split("=", 1)[1].strip().strip('"')
        for line in cfg.splitlines()
        if line.startswith("target_feature=")
    )
    wanted_features = sorted(target["enabled_target_features"])
    if observed_features != wanted_features:
        drifts.append(("enabled_target_features", ",".join(wanted_features), ",".join(observed_features)))
    for feature in target["required_compiler_features"]:
        if f"    {feature}" not in catalogue:
            drifts.append((f"compiler_feature:{feature}", "present", "missing"))

if drifts:
    for field, wanted, observed in drifts:
        print(f"TOOLCHAIN drift field={field} expected={wanted} observed={observed}", file=sys.stderr)
    print(f"TOOLCHAIN RESULT=FAIL drift={drifts[0][0]}", file=sys.stderr)
    raise SystemExit(1)

print("TOOLCHAIN RESULT=PASS drift=none")
PY
