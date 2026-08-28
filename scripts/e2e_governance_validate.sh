#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${REPO_ROOT}/scripts/check_claims.sh"
"${REPO_ROOT}/scripts/lint_behavior_notes.sh"
"${REPO_ROOT}/scripts/lint_ledgers.sh"
exec python3 "${REPO_ROOT}/scripts/validate_adrs.py"
