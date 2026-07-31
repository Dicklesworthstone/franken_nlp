#!/usr/bin/env bash
# Validate the negative-evidence, performance, and feature-parity ledgers.
#
# The validator deliberately uses only Python's standard library: it is
# repository tooling, not the FrankenNLP product or inference graph. Its
# self-test uses committed malformed fixtures and deterministic in-memory
# mutations, so it creates and deletes no temporary files.
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "${REPO_ROOT}" "$@" <<'PY'
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(sys.argv[1])
ARGS = sys.argv[2:]
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
ENTRY = re.compile(r"^## ([A-Z][A-Z0-9-]+)\s*$")
FIELD = re.compile(r"^- ([A-Za-z][A-Za-z0-9 +/()_-]*):\s*(.*)$")
DISPOSITIONS = {"won", "rejected", "prior", "deferred"}
PARITY_STATES = {"present", "partial", "missing", "n/a"}
PERF_REGIMES = {
    "R0-cold-warm-startup",
    "R1-latency-generate",
    "R2-corpus-scoring",
    "R3-corpus-generation",
    "R4-long-context",
}
BANDWIDTH_DENOMINATORS = {
    "logical-tensor-bytes",
    "packed-payload-bytes",
    "measured-dram-bytes",
}
COMMON_FIELDS = (
    "Claim ID",
    "Evidence",
    "Fixture hashes",
    "CPU feature string",
    "Command + environment",
    "Disposition",
)
NEGATIVE_FIELDS = COMMON_FIELDS + (
    "Hypothesis",
    "Five-pass loop",
    "Loss basis",
    "Revert proof",
    "Re-evaluation conditions",
)
PERF_FIELDS = COMMON_FIELDS + (
    "Regime",
    "Host fingerprint",
    "Artifact recipe + packing + kernel table + load mode",
    "p50/p95/p99",
    "Fairness controls",
    "Bandwidth denominator",
)
PARITY_FIELDS = COMMON_FIELDS + (
    "Surface",
    "Reference counterpart",
    "State",
    "Missing behavior",
    "Gate",
)
CLAIM_ID = re.compile(r"[a-z][a-z0-9-]{1,79}\Z")
KNOWN_CLAIMS: set[str] | None = None


@dataclass(frozen=True)
class Ledger:
    path: Path
    prefix: str
    fields: tuple[str, ...]


LEDGERS = (
    Ledger(ROOT / "docs" / "NEGATIVE_EVIDENCE.md", "NE-", NEGATIVE_FIELDS),
    Ledger(ROOT / "docs" / "PERF_LEDGER.md", "PERF-", PERF_FIELDS),
    Ledger(ROOT / "docs" / "FEATURE_PARITY.md", "PARITY-", PARITY_FIELDS),
)
FIXTURES = ROOT / "tests" / "fixtures" / "ledgers"


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def failure(path: Path, line: int, entry_id: str, kind: str, expected: str, actual: str) -> str:
    return (
        f"{display(path)}:{line}: {entry_id}: class={kind} "
        f"expected={expected} actual={actual}"
    )


def parse(path: Path, text: str, ledger: Ledger) -> tuple[list[tuple[str, int, dict[str, str]]], list[str]]:
    records: list[tuple[str, int, dict[str, str]]] = []
    errors: list[str] = []
    entry_id: str | None = None
    entry_line = 0
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        heading = ENTRY.match(line)
        if heading:
            if entry_id is not None:
                records.append((entry_id, entry_line, fields))
            entry_id = heading.group(1)
            entry_line = line_number
            fields = {}
            continue
        if entry_id is None:
            continue
        field = FIELD.match(line)
        if field:
            key, value = field.groups()
            if key in fields:
                errors.append(failure(path, line_number, entry_id, "duplicate_field", "one field", key))
            fields[key] = value
    if entry_id is not None:
        records.append((entry_id, entry_line, fields))

    seen: set[str] = set()
    for identifier, line_number, _ in records:
        if not identifier.startswith(ledger.prefix):
            errors.append(failure(path, line_number, identifier, "entry_prefix", ledger.prefix, identifier))
        if identifier in seen:
            errors.append(failure(path, line_number, identifier, "duplicate_entry", "unique entry id", identifier))
        seen.add(identifier)
    return records, errors


def validate_fixture_hashes(value: str, disposition: str) -> str | None:
    if value.startswith("pending:"):
        if disposition == "deferred":
            return None
        return "pending fixture hashes require disposition=deferred"
    hashes = [item.strip() for item in value.split(",") if item.strip()]
    if not hashes or any(SHA256.fullmatch(item) is None for item in hashes):
        return "comma-separated sha256:<64-lowercase-hex> values"
    return None


def validate_record(path: Path, line: int, entry_id: str, fields: dict[str, str], ledger: Ledger) -> list[str]:
    errors: list[str] = []
    for name in ledger.fields:
        if not fields.get(name):
            errors.append(failure(path, line, entry_id, "missing_field", name, "empty"))
    disposition = fields.get("Disposition", "")
    if disposition and disposition not in DISPOSITIONS:
        errors.append(failure(path, line, entry_id, "disposition", "won|rejected|prior|deferred", disposition))
    fixture_hashes = fields.get("Fixture hashes", "")
    if fixture_hashes and disposition:
        reason = validate_fixture_hashes(fixture_hashes, disposition)
        if reason:
            errors.append(failure(path, line, entry_id, "fixture_hashes", reason, fixture_hashes))
    cpu = fields.get("CPU feature string", "")
    if cpu and not re.search(r"[A-Za-z0-9]", cpu):
        errors.append(failure(path, line, entry_id, "cpu_features", "named CPU feature string", cpu))
    command = fields.get("Command + environment", "")
    if command and not ("=" in command or command.startswith("n/a:")):
        errors.append(failure(path, line, entry_id, "command_environment", "command with environment key=value or n/a scope", command))
    claim_id = fields.get("Claim ID", "")
    if claim_id and claim_id != "none":
        if CLAIM_ID.fullmatch(claim_id) is None:
            errors.append(failure(path, line, entry_id, "claim_id", "lowercase registered claim-id or none", claim_id))
        elif KNOWN_CLAIMS is not None and claim_id not in KNOWN_CLAIMS:
            errors.append(failure(path, line, entry_id, "claim_id", "id present in docs/CLAIMS.json", claim_id))

    if ledger.prefix == "NE-":
        revert = fields.get("Revert proof", "")
        if revert and "no source landed" not in revert.lower():
            errors.append(failure(path, line, entry_id, "revert_proof", "explicit 'no source landed' proof", revert))
        loop = fields.get("Five-pass loop", "")
        if loop and not re.search(r"(?:five|5).*(?:pass|loop)|(?:pass|loop).*(?:five|5)", loop, re.I):
            errors.append(failure(path, line, entry_id, "five_pass_loop", "named five-pass loop", loop))
    elif ledger.prefix == "PERF-":
        regime = fields.get("Regime", "")
        if regime and regime not in PERF_REGIMES:
            errors.append(failure(path, line, entry_id, "regime", "|".join(sorted(PERF_REGIMES)), regime))
        percentiles = fields.get("p50/p95/p99", "")
        if percentiles and not all(token in percentiles for token in ("p50", "p95", "p99")):
            errors.append(failure(path, line, entry_id, "percentiles", "named p50, p95, and p99", percentiles))
        denominator = fields.get("Bandwidth denominator", "")
        if denominator and denominator not in BANDWIDTH_DENOMINATORS:
            errors.append(failure(path, line, entry_id, "bandwidth_denominator", "|".join(sorted(BANDWIDTH_DENOMINATORS)), denominator))
    else:
        state = fields.get("State", "")
        missing = fields.get("Missing behavior", "")
        gate = fields.get("Gate", "")
        if state and state not in PARITY_STATES:
            errors.append(failure(path, line, entry_id, "parity_state", "present|partial|missing|n/a", state))
        if state in {"present", "n/a"}:
            if missing and missing != "none":
                errors.append(failure(path, line, entry_id, "parity_round_up", "Missing behavior: none for present/n/a", missing))
            if gate and gate != "none":
                errors.append(failure(path, line, entry_id, "parity_round_up", "Gate: none for present/n/a", gate))
        if state == "partial":
            if missing == "none":
                errors.append(failure(path, line, entry_id, "partial_missing_behavior", "concrete missing behavior", missing))
            if gate == "none":
                errors.append(failure(path, line, entry_id, "partial_gate", "gating bead or phase", gate))
    return errors


def validate_text(path: Path, text: str, ledger: Ledger) -> tuple[int, list[str]]:
    records, errors = parse(path, text, ledger)
    for identifier, line, fields in records:
        errors.extend(validate_record(path, line, identifier, fields, ledger))
    return len(records), errors


def validate_file(ledger: Ledger) -> tuple[int, list[str]]:
    try:
        text = ledger.path.read_text(encoding="utf-8")
    except OSError as error:
        return 0, [f"{display(ledger.path)}:0: ledger: class=read expected=readable file actual={error}"]
    return validate_text(ledger.path, text, ledger)


def fixture_for(prefix: str, name: str) -> Ledger:
    return next(ledger for ledger in LEDGERS if ledger.prefix == prefix)


def load_claim_ids() -> tuple[set[str] | None, list[str]]:
    path = ROOT / "docs" / "CLAIMS.json"
    if not path.exists():
        return None, []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"{display(path)}:0: claims: class=claim_registry expected=readable JSON actual={error}"]
    claims = value.get("claims") if isinstance(value, dict) else None
    if not isinstance(claims, list):
        return None, [f"{display(path)}:0: claims: class=claim_registry expected=object with claims array actual={type(claims).__name__}"]
    ids = {entry.get("id") for entry in claims if isinstance(entry, dict) and isinstance(entry.get("id"), str)}
    if len(ids) != len(claims):
        return None, [f"{display(path)}:0: claims: class=claim_registry expected=unique string id for every claim actual=invalid-or-duplicate"]
    return ids, []


def self_test() -> tuple[int, list[str]]:
    cases = 0
    errors: list[str] = []
    for filename, prefix, expected_class in (
        ("negative_missing_cpu_feature.md", "NE-", "missing_field"),
        ("perf_missing_fairness_controls.md", "PERF-", "missing_field"),
        ("parity_partial_present.md", "PARITY-", "parity_round_up"),
    ):
        path = FIXTURES / filename
        count, observed = validate_text(path, path.read_text(encoding="utf-8"), fixture_for(prefix, filename))
        cases += 1
        if count != 1 or not any(f"class={expected_class}" in error for error in observed):
            errors.append(f"{display(path)}: self-test expected class={expected_class}, got {observed!r}")

    valid_entries = (
        (fixture_for("NE-", ""), """## NE-VALID-001\n\n- Claim ID: none\n- Evidence: scalar comparison\n- Fixture hashes: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n- CPU feature string: scalar\n- Command + environment: command=bench; threads=1\n- Disposition: rejected\n- Hypothesis: test candidate\n- Five-pass loop: five-pass loop retained\n- Loss basis: slower\n- Revert proof: no source landed\n- Re-evaluation conditions: new proof\n"""),
        (fixture_for("PERF-", ""), """## PERF-VALID-001\n\n- Claim ID: none\n- Evidence: receipt\n- Fixture hashes: sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n- CPU feature string: avx2\n- Command + environment: command=bench; threads=1\n- Disposition: rejected\n- Regime: R1-latency-generate\n- Host fingerprint: fixture\n- Artifact recipe + packing + kernel table + load mode: recipe=r1\n- p50/p95/p99: p50=1; p95=2; p99=3\n- Fairness controls: threads=1 allocator=system precision=f32 warmup=10\n- Bandwidth denominator: logical-tensor-bytes\n"""),
        (fixture_for("PARITY-", ""), """## PARITY-VALID-001\n\n- Claim ID: none\n- Evidence: inventory\n- Fixture hashes: sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n- CPU feature string: n/a: no CPU kernel selected\n- Command + environment: n/a: static surface audit\n- Disposition: deferred\n- Surface: output grammar\n- Reference counterpart: pinned reference\n- State: partial\n- Missing behavior: bounded decoder\n- Gate: franken_nlp-4pe\n"""),
    )
    for ledger, source in valid_entries:
        _, baseline = validate_text(Path("in-memory.md"), source, ledger)
        cases += 1
        if baseline:
            errors.append(f"in-memory valid {ledger.prefix} entry rejected: {baseline!r}")
        for field in ledger.fields:
            mutated = source.replace(f"- {field}:", f"- Removed {field}:", 1)
            _, observed = validate_text(Path("in-memory.md"), mutated, ledger)
            cases += 1
            if not any("class=missing_field" in error for error in observed):
                errors.append(f"in-memory {ledger.prefix} missing {field} did not produce typed error")

    generator = random.Random(0)
    alphabet = "#-: abcXYZ012\n\t\u2603"
    for _ in range(64):
        arbitrary = "".join(generator.choice(alphabet) for _ in range(generator.randrange(0, 256)))
        for ledger in LEDGERS:
            try:
                validate_text(Path("in-memory-fuzz.md"), arbitrary, ledger)
            except Exception as error:  # pragma: no cover - a failure is reported below.
                errors.append(f"arbitrary-markdown parser panic for {ledger.prefix}: {error!r}")
        cases += 1
    return cases, errors


def main() -> int:
    if ARGS not in ([], ["--self-test"]):
        print("LEDGERS RESULT=FAIL reason=usage expected=--self-test-or-empty", file=sys.stderr)
        return 2
    global KNOWN_CLAIMS
    KNOWN_CLAIMS, registry_errors = load_claim_ids()
    cross_registry = "checked" if KNOWN_CLAIMS is not None else "soft-pending"
    counts: dict[str, int] = {}
    errors: list[str] = list(registry_errors)
    for ledger, label in zip(LEDGERS, ("negative", "perf", "parity"), strict=True):
        counts[label], observed = validate_file(ledger)
        errors.extend(observed)
    fixture_cases = 0
    if ARGS == ["--self-test"]:
        fixture_cases, observed = self_test()
        errors.extend(observed)
    if errors:
        classes = Counter(re.search(r"class=([^ ]+)", error).group(1) if "class=" in error else "self_test" for error in errors)
        for error in errors:
            print(f"LEDGERS FAIL {error}", file=sys.stderr)
        print(
            "LEDGERS RESULT=FAIL "
            f"negative_entries={counts['negative']} perf_entries={counts['perf']} parity_entries={counts['parity']} "
            f"fixture_cases={fixture_cases} cross_registry={cross_registry} failures={len(errors)} failure_classes={dict(sorted(classes.items()))}",
            file=sys.stderr,
        )
        return 1
    print(
        "LEDGERS RESULT=PASS "
        f"negative_entries={counts['negative']} perf_entries={counts['perf']} parity_entries={counts['parity']} "
        f"fixture_cases={fixture_cases} cross_registry={cross_registry} failures=0",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
