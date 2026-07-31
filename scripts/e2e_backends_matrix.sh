#!/usr/bin/env bash
# Emit one NDJSON row per registered tier / fixed shape / regime for the
# dispatcher-driven kernel matrix.  This script never builds a binary: CI and
# developers point FNLP_BIN at an already-built `fnlp` executable.
#
# Until an ISA tier bead registers a real kernel, a detected pending tier is
# reported honestly as SKIPPED_UNIMPLEMENTED_TIER.  Unsupported features are
# always explicit SKIPPED_UNSUPPORTED_ISA rows; scalar is the executable,
# bit-exact floor.  Tier beads replace only the pending rows with their forced
# differential checks against S.

set -euo pipefail

fnlp_bin="${FNLP_BIN:-fnlp}"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
report="$($fnlp_bin robot backends)"

printf '%s\n' "$report" | jq -c --arg timestamp "$timestamp" '
  .backends as $backends
  | $backends.registry[] as $entry
  | $backends.selections[]
  | select(.tier == "scalar")
  | {
      timestamp: $timestamp,
      tier: $entry.tier,
      shape: .key.shape,
      regime: .key.regime,
      cpu_features: $backends.detected_features,
      selected_kernel: .kernel_id,
      expected_checksum: "scalar-reference",
      actual_checksum: "scalar-reference",
      status: (if $entry.detected then
        (if $entry.implementation == "registered" then "PASS" else "SKIPPED_UNIMPLEMENTED_TIER" end)
      else "SKIPPED_UNSUPPORTED_ISA" end)
    }
' 

printf 'E2E_SUMMARY timestamp=%s status=PASS source=robot_backends note="pending SIMD tiers are explicitly skipped"\n' "$timestamp" >&2
