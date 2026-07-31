#!/usr/bin/env bash
# Validate the intentional-behavior and numeric-discrepancy ledgers.
#
# This intentionally uses only Python's standard library because it is
# repository tooling, not product or inference code. `--self-test` exercises
# committed malformed fixtures and in-memory field mutations without creating
# or deleting temporary files.
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "${REPO_ROOT}" "$@" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(sys.argv[1])
ARGS = sys.argv[2:]
BEHAVIOR = ROOT / "docs" / "BEHAVIOR_NOTES.md"
DISCREPANCIES = ROOT / "docs" / "DISCREPANCIES.md"
SOURCE_MANIFEST = ROOT / "docs" / "truth-pack" / "nanbeige4.2-3b.source.json"
FIXTURES = ROOT / "tests" / "fixtures" / "behavior_notes"
REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
GENERATION_CONFIG_SHA256 = "68c690ce23efb6caae30c006ff3c1efd826297ff1df4338c04f7ac6f685d8746"

BEHAVIOR_FIELDS = (
    "Source pin",
    "Minimized fixture",
    "Decision",
    "Rationale",
    "Compatibility impact",
    "Revisit condition",
)
DISCREPANCY_FIELDS = (
    "Reference behavior",
    "Our behavior",
    "Profile",
    "Affected operators/surfaces",
    "Measured impact",
    "Review date",
    "Rollback mechanism",
)
FIELD = re.compile(r"^- ([A-Za-z][A-Za-z /-]*):\s*(.*)$")
ENTRY = re.compile(r"^## ([A-Z][A-Z0-9-]+)\s*$")
SOURCE_PIN = re.compile(
    r"`(?P<name>[A-Za-z0-9_.-]+)@(?P<revision>[0-9a-f]{40}):"
    r"(?P<start>[1-9][0-9]*)-(?P<end>[1-9][0-9]*); "
    r"sha256:(?P<digest>[0-9a-f]{64})`"
)
DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
PROFILE = re.compile(r"^(hf-bf16-eager|diagnostic-f32|strict-quantized-v[0-9]+|fast-v[0-9]+)$")
ROLLBACK_PREFIXES = (
    "kernel-selector:",
    "cli-or-builder-option:",
    "prior-immutable-artifact:",
)


def log(message: str) -> None:
    print(f"BEHAVIOR_NOTES {message}", file=sys.stderr)


def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in source manifest: {key}")
        result[key] = value
    return result


def source_manifest_generation_digest() -> str:
    try:
        value = json.loads(
            SOURCE_MANIFEST.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot parse source manifest: {error}") from error
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, list):
        raise ValueError("source manifest has no files array")
    matching = [entry for entry in files if isinstance(entry, dict) and entry.get("name") == "generation_config.json"]
    if len(matching) != 1 or not isinstance(matching[0].get("sha256"), str):
        raise ValueError("source manifest must contain exactly one generation_config.json digest")
    return matching[0]["sha256"]


def entries(path: Path, prefix: str) -> tuple[list[tuple[str, int, dict[str, str]]], list[str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], [f"{path.relative_to(ROOT)}: cannot read: {error}"]
    found: list[tuple[str, int, dict[str, str]]] = []
    failures: list[str] = []
    current_id: str | None = None
    current_line = 0
    fields: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        heading = ENTRY.match(line)
        if heading:
            if current_id is not None:
                found.append((current_id, current_line, fields))
            current_id = heading.group(1)
            current_line = line_number
            fields = {}
            continue
        if current_id is None:
            continue
        field = FIELD.match(line)
        if field:
            name, value = field.groups()
            if name in fields:
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: {current_id}: duplicate field {name}")
            fields[name] = value
    if current_id is not None:
        found.append((current_id, current_line, fields))

    seen: set[str] = set()
    for identifier, line_number, _ in found:
        if not identifier.startswith(prefix):
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: entry id {identifier} must start with {prefix}")
        if identifier in seen:
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: duplicate entry id {identifier}")
        seen.add(identifier)
    return found, failures


def validate_behavior(path: Path) -> tuple[int, list[str]]:
    found, failures = entries(path, "BN-")
    for identifier, line_number, fields in found:
        missing = [field for field in BEHAVIOR_FIELDS if not fields.get(field)]
        for field in missing:
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: missing {field}")
        source = fields.get("Source pin", "")
        match = SOURCE_PIN.fullmatch(source)
        if source and match is None:
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: invalid Source pin")
            continue
        if match and int(match["start"]) > int(match["end"]):
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: Source pin has inverted line span")
        if identifier == "BN-GEN-DEFAULT-001" and match:
            if match["name"] != "generation_config.json" or match["revision"] != REVISION:
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: source pin must name the pinned generation_config.json")
            if match["digest"] != GENERATION_CONFIG_SHA256:
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: generation_config digest mismatch")
            decision = fields.get("Decision", "")
            required = ("greedy", "temperature=0.6", "top_k=20", "top_p=0.95", "--sample --preset nanbeige")
            for token in required:
                if token not in decision:
                    failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: Decision must state {token}")
    if path == BEHAVIOR and not any(identifier == "BN-GEN-DEFAULT-001" for identifier, _, _ in found):
        failures.append(f"{path.relative_to(ROOT)}: missing required seed BN-GEN-DEFAULT-001")
    return len(found), failures


def validate_discrepancies(path: Path) -> tuple[int, list[str]]:
    found, failures = entries(path, "DISC-")
    for identifier, line_number, fields in found:
        missing = [field for field in DISCREPANCY_FIELDS if not fields.get(field)]
        for field in missing:
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: missing {field}")
        profile = fields.get("Profile", "")
        if profile and PROFILE.fullmatch(profile) is None:
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: invalid Profile {profile}")
        impact = fields.get("Measured impact", "")
        if impact and ("fixture=" not in impact or re.search(r"sha256:[0-9a-f]{64}", impact) is None):
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: Measured impact requires fixture= and sha256")
        review_date = fields.get("Review date", "")
        if review_date and DATE.fullmatch(review_date) is None:
            failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: Review date must be YYYY-MM-DD")
        rollback = fields.get("Rollback mechanism", "")
        if rollback:
            if "env" in rollback.lower() or not rollback.startswith(ROLLBACK_PREFIXES):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: {identifier}: invalid rollback mechanism")
    return len(found), failures


def validate_real_docs() -> tuple[int, int, list[str]]:
    failures: list[str] = []
    try:
        manifest_digest = source_manifest_generation_digest()
    except ValueError as error:
        failures.append(str(error))
    else:
        if manifest_digest != GENERATION_CONFIG_SHA256:
            failures.append("source manifest generation_config.json digest drift")
    behavior_count, behavior_failures = validate_behavior(BEHAVIOR)
    discrepancy_count, discrepancy_failures = validate_discrepancies(DISCREPANCIES)
    failures.extend(behavior_failures)
    failures.extend(discrepancy_failures)
    return behavior_count, discrepancy_count, failures


def self_test() -> tuple[int, list[str]]:
    cases = 0
    failures: list[str] = []
    for fixture, expected in (
        (FIXTURES / "behavior_notes_missing_source.md", "missing Source pin"),
        (FIXTURES / "discrepancies_missing_rollback.md", "missing Rollback mechanism"),
        (FIXTURES / "discrepancies_env_rollback.md", "invalid rollback mechanism"),
    ):
        if fixture.name.startswith("behavior"):
            _, observed = validate_behavior(fixture)
        else:
            _, observed = validate_discrepancies(fixture)
        cases += 1
        if not any(expected in failure for failure in observed):
            failures.append(f"{fixture.relative_to(ROOT)}: expected rejection containing {expected!r}, got {observed!r}")

    source = BEHAVIOR.read_text(encoding="utf-8")
    for field in BEHAVIOR_FIELDS:
        mutated = source.replace(f"- {field}:", f"- Removed {field}:", 1)
        temporary = FIXTURES / "in_memory_behavior.md"
        original_read = Path.read_text
        try:
            Path.read_text = lambda self, encoding="utf-8": mutated if self == temporary else original_read(self, encoding=encoding)  # type: ignore[method-assign]
            _, observed = validate_behavior(temporary)
        finally:
            Path.read_text = original_read  # type: ignore[method-assign]
        cases += 1
        if not any(f"missing {field}" in failure for failure in observed):
            failures.append(f"in-memory mutation did not reject missing {field}")
    return cases, failures


def main() -> int:
    if ARGS not in ([], ["--self-test"]):
        log("RESULT=FAIL reason=usage expected=--self-test-or-empty")
        return 2
    behavior_count, discrepancy_count, failures = validate_real_docs()
    fixture_cases = 0
    if ARGS == ["--self-test"]:
        fixture_cases, self_test_failures = self_test()
        failures.extend(self_test_failures)
    if failures:
        for failure in failures:
            log(f"FAIL {failure}")
        log(f"RESULT=FAIL behavior_entries={behavior_count} discrepancy_entries={discrepancy_count} fixture_cases={fixture_cases}")
        return 1
    log(f"RESULT=PASS behavior_entries={behavior_count} discrepancy_entries={discrepancy_count} fixture_cases={fixture_cases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
