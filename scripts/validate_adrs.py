"""Validate G0 ADR metadata, evidence integrity, and registry reconciliation.

This pre-write deliberately contains no committed registry or test fixtures;
those arrive with the Phase-0 scaffold. The validator is complete enough to
validate that later tree without adding a release dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STATUS_VALUES = frozenset({"RATIFIED", "ABSENT-WITH-FALLBACK", "BLOCKED"})
PRIMARY_PROBES = frozenset(range(1, 12))
ADR_FILENAME = re.compile(r"ADR-G0-(?P<probe>0[1-9]|1[01])-[a-z0-9]+(?:-[a-z0-9]+)*\.md\Z")
ADR_ID = re.compile(r"G0-(?:0[1-9]|1[01])(?:-[a-z0-9]+(?:-[a-z0-9]+)*)?\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
METADATA_BLOCK = re.compile(
    r"(?ms)^```adr-metadata[ \t]*\n(?P<body>.*?)^```[ \t]*$"
)


class AdrValidationError(ValueError):
    """A precise G0 ADR contract violation."""


@dataclass(frozen=True)
class Issue:
    path: Path
    field: str
    detail: str

    def render(self) -> str:
        return f"file={self.path} field={self.field} detail={self.detail}"


@dataclass(frozen=True)
class AdrRecord:
    path: Path
    metadata: dict[str, Any]

    @property
    def adr_id(self) -> str:
        return self.metadata["adr_id"]


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{stamp} ADR_REGISTRY {message}", file=sys.stderr)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdrValidationError(f"cannot parse JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdrValidationError(f"JSON document is not an object: {path}")
    return value


def require_string(metadata: dict[str, Any], field: str, issues: list[Issue], path: Path) -> str | None:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        issues.append(Issue(path, field, "must be a non-empty string"))
        return None
    return value


def validate_named_records(
    value: Any, field: str, required_keys: frozenset[str], issues: list[Issue], path: Path
) -> None:
    if not isinstance(value, list) or not value:
        issues.append(Issue(path, field, "must be a non-empty array"))
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            issues.append(Issue(path, f"{field}[{index}]", "must be an object"))
            continue
        for key in required_keys:
            if not isinstance(item.get(key), str) or not item[key].strip():
                issues.append(Issue(path, f"{field}[{index}].{key}", "must be a non-empty string"))


def parse_metadata(path: Path) -> AdrRecord:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AdrValidationError(f"cannot read ADR {path}: {error}") from error
    matches = list(METADATA_BLOCK.finditer(text))
    if len(matches) != 1:
        raise AdrValidationError(f"ADR {path} must contain exactly one adr-metadata block; observed={len(matches)}")
    payload = matches[0].group("body")
    try:
        metadata = json.loads(payload)
    except json.JSONDecodeError as error:
        raise AdrValidationError(f"ADR {path} metadata is not JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise AdrValidationError(f"ADR {path} metadata must be a JSON object")
    if canonical_json_bytes(metadata).decode("utf-8") != payload:
        raise AdrValidationError(f"ADR {path} metadata block is not canonical sorted JSON")
    return AdrRecord(path=path, metadata=metadata)


def validate_metadata(record: AdrRecord) -> list[Issue]:
    metadata = record.metadata
    path = record.path
    issues: list[Issue] = []
    allowed = {
        "adr_id",
        "blocked_surface",
        "decision",
        "evidence",
        "exact_commands",
        "fallback",
        "g0_item",
        "host_pin",
        "killed_alternatives",
        "source_pin",
        "status",
    }
    for field in metadata:
        if field not in allowed and not field.startswith("x_"):
            issues.append(Issue(path, field, "unknown field; extensions must begin with x_"))

    adr_id = require_string(metadata, "adr_id", issues, path)
    if adr_id is not None and ADR_ID.fullmatch(adr_id) is None:
        issues.append(Issue(path, "adr_id", "must match G0-<probe##>[-sub-probe]"))
    status = require_string(metadata, "status", issues, path)
    if status is not None and status not in STATUS_VALUES:
        issues.append(Issue(path, "status", f"unknown status {status!r}"))
    decision = require_string(metadata, "decision", issues, path)
    if decision is not None and "\n" in decision:
        issues.append(Issue(path, "decision", "must be one paragraph without newline characters"))

    g0_item = metadata.get("g0_item")
    if not isinstance(g0_item, dict):
        issues.append(Issue(path, "g0_item", "must be an object"))
    else:
        probe = g0_item.get("probe")
        if not isinstance(probe, int) or isinstance(probe, bool) or probe not in PRIMARY_PROBES:
            issues.append(Issue(path, "g0_item.probe", "must be an integer in 1..11"))
        if not isinstance(g0_item.get("name"), str) or not g0_item["name"].strip():
            issues.append(Issue(path, "g0_item.name", "must be a non-empty string"))

    for field in ("source_pin", "host_pin"):
        value = metadata.get(field)
        if not isinstance(value, dict) or not value:
            issues.append(Issue(path, field, "must be a non-empty object"))
            continue
        if any(not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip() for key, item in value.items()):
            issues.append(Issue(path, field, "must map non-empty strings to non-empty strings"))

    commands = metadata.get("exact_commands")
    if not isinstance(commands, list) or not commands or any(not isinstance(command, str) or not command.strip() for command in commands):
        issues.append(Issue(path, "exact_commands", "must be a non-empty array of literal command strings"))
    validate_named_records(metadata.get("killed_alternatives"), "killed_alternatives", frozenset({"name", "reason"}), issues, path)

    fallback = metadata.get("fallback")
    if fallback is not None and (not isinstance(fallback, str) or not fallback.strip()):
        issues.append(Issue(path, "fallback", "must be null or a non-empty string"))
    blocked_surface = metadata.get("blocked_surface")
    if not isinstance(blocked_surface, list) or any(not isinstance(surface, str) or not surface.strip() for surface in blocked_surface):
        issues.append(Issue(path, "blocked_surface", "must be an array of non-empty strings"))
        blocked_surface = []
    if status == "ABSENT-WITH-FALLBACK" and not isinstance(fallback, str):
        issues.append(Issue(path, "fallback", "is mandatory for ABSENT-WITH-FALLBACK"))
    if status == "BLOCKED" and not blocked_surface:
        issues.append(Issue(path, "blocked_surface", "is mandatory for BLOCKED"))

    evidence = metadata.get("evidence")
    if not isinstance(evidence, list):
        issues.append(Issue(path, "evidence", "must be an array"))
    else:
        for index, item in enumerate(evidence):
            prefix = f"evidence[{index}]"
            if not isinstance(item, dict):
                issues.append(Issue(path, prefix, "must be an object"))
                continue
            if not isinstance(item.get("path"), str) or not item["path"].strip():
                issues.append(Issue(path, f"{prefix}.path", "must be a non-empty relative path"))
            if not isinstance(item.get("bytes"), int) or isinstance(item["bytes"], bool) or item["bytes"] < 0:
                issues.append(Issue(path, f"{prefix}.bytes", "must be a non-negative integer"))
            digest = item.get("sha256")
            if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
                issues.append(Issue(path, f"{prefix}.sha256", "must be lowercase SHA-256 hex"))
    return issues


def safe_evidence_path(adr_root: Path, adr_id: str, value: str) -> Path | None:
    candidate = (adr_root / value).resolve()
    expected_root = (adr_root / "evidence" / adr_id).resolve()
    try:
        candidate.relative_to(expected_root)
    except ValueError:
        return None
    return candidate


def validate_evidence(record: AdrRecord, adr_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    evidence = record.metadata.get("evidence")
    if not isinstance(evidence, list):
        return issues
    adr_id = record.metadata.get("adr_id")
    if not isinstance(adr_id, str):
        return issues
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        path = safe_evidence_path(adr_root, adr_id, item["path"])
        prefix = f"evidence[{index}]"
        if path is None:
            issues.append(Issue(record.path, f"{prefix}.path", "must stay under evidence/<adr_id>/"))
            continue
        if not path.is_file():
            issues.append(Issue(record.path, f"{prefix}.path", f"missing regular evidence file {item['path']}"))
            continue
        observed_bytes = path.stat().st_size
        observed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_bytes != item.get("bytes"):
            issues.append(Issue(record.path, f"{prefix}.bytes", f"expected={item.get('bytes')} observed={observed_bytes}"))
        if observed_digest != item.get("sha256"):
            issues.append(Issue(record.path, f"{prefix}.sha256", f"expected={item.get('sha256')} observed={observed_digest}"))
    return issues


def validate_registry(registry: dict[str, Any], records: list[AdrRecord], adr_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    registry_path = adr_root / "g0_registry.json"
    if registry.get("schema_version") != 1:
        issues.append(Issue(registry_path, "schema_version", "must equal 1"))
    rows = registry.get("items")
    if not isinstance(rows, list):
        return [*issues, Issue(registry_path, "items", "must be an array")]
    record_by_id = {record.adr_id: record for record in records if isinstance(record.metadata.get("adr_id"), str)}
    ids_seen: set[str] = set()
    paths_seen: set[str] = set()
    probes_seen: set[int] = set()
    registry_ids: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"items[{index}]"
        if not isinstance(row, dict):
            issues.append(Issue(registry_path, prefix, "must be an object"))
            continue
        adr_id = row.get("adr_id")
        adr_path = row.get("adr_path")
        owner_bead = row.get("owner_bead")
        status = row.get("status")
        g0_item = row.get("g0_item")
        if not isinstance(adr_id, str) or ADR_ID.fullmatch(adr_id) is None:
            issues.append(Issue(registry_path, f"{prefix}.adr_id", "must name a valid G0 ADR"))
            continue
        registry_ids.add(adr_id)
        if adr_id in ids_seen:
            issues.append(Issue(registry_path, f"{prefix}.adr_id", f"duplicate adr_id {adr_id}"))
        ids_seen.add(adr_id)
        if not isinstance(adr_path, str) or ADR_FILENAME.fullmatch(adr_path) is None:
            issues.append(Issue(registry_path, f"{prefix}.adr_path", "must be an ADR-G0 filename relative to docs/adr"))
        else:
            if adr_path in paths_seen:
                issues.append(Issue(registry_path, f"{prefix}.adr_path", f"duplicate ADR path {adr_path}"))
            paths_seen.add(adr_path)
        if not isinstance(owner_bead, str) or not owner_bead.strip():
            issues.append(Issue(registry_path, f"{prefix}.owner_bead", "must be a non-empty bead id"))
        if status not in STATUS_VALUES:
            issues.append(Issue(registry_path, f"{prefix}.status", "must be a legal ADR status"))
        if not isinstance(g0_item, dict) or not isinstance(g0_item.get("probe"), int) or g0_item["probe"] not in PRIMARY_PROBES:
            issues.append(Issue(registry_path, f"{prefix}.g0_item", "must contain a primary probe 1..11"))
        else:
            probes_seen.add(g0_item["probe"])
        record = record_by_id.get(adr_id)
        if record is None:
            issues.append(Issue(registry_path, f"{prefix}.adr_id", "registry row has no matching ADR file"))
            continue
        if isinstance(adr_path, str) and record.path.name != adr_path:
            issues.append(Issue(registry_path, f"{prefix}.adr_path", f"expected={record.path.name} observed={adr_path}"))
        for key in ("g0_item", "status"):
            if row.get(key) != record.metadata.get(key):
                issues.append(Issue(registry_path, f"{prefix}.{key}", "must match ADR metadata"))
    for adr_id, record in record_by_id.items():
        if adr_id not in registry_ids:
            issues.append(Issue(record.path, "adr_id", "ADR file has no registry row"))
    missing_probes = sorted(PRIMARY_PROBES - probes_seen)
    if missing_probes:
        issues.append(Issue(registry_path, "items", f"missing primary probes {missing_probes}"))
    return issues


def validate_tree(adr_root: Path, registry_path: Path) -> tuple[list[AdrRecord], list[Issue]]:
    records: list[AdrRecord] = []
    issues: list[Issue] = []
    for path in sorted(adr_root.glob("ADR-G0-*.md")):
        if ADR_FILENAME.fullmatch(path.name) is None:
            issues.append(Issue(path, "filename", "must match ADR-G0-<probe##>-<slug>.md"))
            continue
        try:
            record = parse_metadata(path)
        except AdrValidationError as error:
            issues.append(Issue(path, "metadata", str(error)))
            continue
        records.append(record)
        issues.extend(validate_metadata(record))
        issues.extend(validate_evidence(record, adr_root))
    ids = [record.adr_id for record in records]
    for adr_id in sorted({value for value in ids if ids.count(value) > 1}):
        issues.append(Issue(adr_root, "adr_id", f"duplicate ADR metadata id {adr_id}"))
    try:
        registry = read_json_object(registry_path)
    except AdrValidationError as error:
        issues.append(Issue(registry_path, "registry", str(error)))
        return records, issues
    issues.extend(validate_registry(registry, records, adr_root))
    return records, issues


def run_self_test() -> None:
    metadata = {
        "adr_id": "G0-01",
        "blocked_surface": [],
        "decision": "A bounded decision for an in-memory parser check.",
        "evidence": [],
        "exact_commands": ["echo fixture"],
        "fallback": None,
        "g0_item": {"name": "fixture", "probe": 1},
        "host_pin": {"applicability": "not-host-sensitive"},
        "killed_alternatives": [{"name": "none", "reason": "fixture"}],
        "source_pin": {"fixture": "not-applicable"},
        "status": "RATIFIED",
    }
    record = AdrRecord(path=Path("ADR-G0-01-fixture.md"), metadata=metadata)
    issues = validate_metadata(record)
    if issues:
        raise AdrValidationError("valid metadata fixture rejected: " + "; ".join(issue.render() for issue in issues))
    absent = {**metadata, "status": "ABSENT-WITH-FALLBACK"}
    absent_issues = validate_metadata(AdrRecord(record.path, absent))
    if not any(issue.field == "fallback" for issue in absent_issues):
        raise AdrValidationError("ABSENT-WITH-FALLBACK fixture without fallback was accepted")
    blocked = {**metadata, "status": "BLOCKED"}
    blocked_issues = validate_metadata(AdrRecord(record.path, blocked))
    if not any(issue.field == "blocked_surface" for issue in blocked_issues):
        raise AdrValidationError("BLOCKED fixture without blocked_surface was accepted")
    unknown = {**metadata, "status": "UNKNOWN"}
    unknown_issues = validate_metadata(AdrRecord(record.path, unknown))
    if not any(issue.field == "status" for issue in unknown_issues):
        raise AdrValidationError("unknown-status fixture was accepted")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adr-root", type=Path, default=Path("docs/adr"), help="G0 ADR directory")
    parser.add_argument("--registry", type=Path, help="registry path; defaults to <adr-root>/g0_registry.json")
    parser.add_argument("--self-test", action="store_true", help="run in-memory metadata status tests")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        try:
            run_self_test()
        except AdrValidationError as error:
            log(f"RESULT=FAIL items=0/11 blocked=none detail={error}")
            return 1
        log("RESULT=PASS items=0/11 blocked=none mode=self-test")
        return 0
    registry_path = args.registry or args.adr_root / "g0_registry.json"
    records, issues = validate_tree(args.adr_root, registry_path)
    blocked = sorted(record.adr_id for record in records if record.metadata.get("status") == "BLOCKED")
    for record in sorted(records, key=lambda item: (item.metadata.get("g0_item", {}).get("probe", 0), item.adr_id)):
        g0_item = record.metadata.get("g0_item", {})
        print(f"probe={g0_item.get('probe')} adr_id={record.adr_id} status={record.metadata.get('status')} path={record.path}")
    for issue in issues:
        log(f"FAIL {issue.render()}")
    result = "PASS" if not issues else "FAIL"
    blocked_value = ",".join(blocked) if blocked else "none"
    log(f"RESULT={result} items={len(records)}/11 blocked={blocked_value}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
