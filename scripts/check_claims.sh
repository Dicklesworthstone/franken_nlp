#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${REPO_ROOT}/scripts/check_claims.py" --root "${REPO_ROOT}" --check
