#!/usr/bin/env python3
"""Validate truth-pack benchmark conflicts and OQ promotion evidence.

Expected inputs beneath ``docs/truth-pack`` are ``benchmark_conflicts.json``
and ``promotions.json``.  Both are deliberately data-only records.  A
promotion record has this shape::

    {
      "oq_id": "OQ-1",
      "claim": "...",
      "old_state": "OBSERVED@pin",
      "new_state": "EVIDENCED",
      "replay_fixture": "tests/fixtures/...",
      "evidence": [{
        "path": "archive/source.txt",
        "sha256": "<64 lowercase hex>",
        "line_start": 1,
        "line_end": 2,
        "expected_text": "first line\\nsecond line"
      }]
    }

``expected_text_sha256`` may replace ``expected_text`` for a line-bounded
canonical document whose one line is too large to duplicate in the promotion
record.  The validator still compares that precise span and logs a bounded
observed context if it drifts.

Partial or open records additionally carry a non-empty ``open_remainders``
list whose objects name the blocking phase.  A benchmark row uses
``thinking_mode`` with ``enabled: true`` and ``preserve_thinking: true``, plus
source entries with a truth-pack-relative path, SHA-256, and reported numeric
value.

The validator intentionally fails while the two dependency beads have not yet
materialized their evidence inputs.  It never downloads sources or fabricates a
promotion from source prose.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_OQS = frozenset(
    {
        "OQ-1",
        "OQ-2",
        "OQ-3",
        "OQ-4",
        "OQ-5",
        "OQ-6",
        "OQ-7",
        "OQ-8",
        "OQ-9",
        "OQ-16",
        "OQ-25",
    }
)
EXPECTED_BENCHMARKS: dict[str, Decimal] = {
    "HMMT-Feb-2026": Decimal("82.8"),
    "GPQA-Diamond": Decimal("87.4"),
    "SWE-Bench Verified": Decimal("63.6"),
    "Terminal-Bench 2.0": Decimal("44.1"),
    "GDPval-rubrics": Decimal("74.3"),
    "Agent-IF-Oneday": Decimal("67.5"),
    "Pinch-Bench-V2": Decimal("74.7"),
    "Claw-Gym": Decimal("65.0"),
    "GDPval": Decimal("68.8"),
    "DeepResearch Bench II": Decimal("33.4"),
    "ResearchRubrics": Decimal("44.8"),
}
HMMT_CONFLICT_VALUES = frozenset({Decimal("82.8"), Decimal("82.1")})
LEGAL_PROMOTION_TRANSITIONS = frozenset(
    {
        ("OBSERVED@pin", "EVIDENCED"),
        ("OBSERVED", "EVIDENCED"),
        ("PARTIAL", "EVIDENCED"),
        ("PARTIAL", "PARTIAL"),
        ("OPEN", "PARTIAL"),
        ("OPEN", "OPEN"),
    }
)


class PromotionError(ValueError):
    """A named truth-pack contract violation."""


@dataclass
class Reporter:
    failures: list[str] = field(default_factory=list)
    drifted: set[str] = field(default_factory=set)

    def log(self, message: str) -> None:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        print(f"{stamp} PROMOTIONS {message}", file=sys.stderr)

    def fail(self, message: str) -> None:
        self.failures.append(message)
        self.log(f"FAIL {message}")

    def mark_drifted(self, record_id: str, evidence_path: str) -> None:
        self.drifted.add(f"{record_id}:{evidence_path}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_nonfinite(token: str) -> None:
    raise PromotionError(f"non-finite JSON number {token!r}")


def load_object(path: Path, reporter: Reporter, label: str) -> dict[str, Any] | None:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PromotionError) as error:
        reporter.fail(f"{label} cannot parse {path}: {error}")
        return None
    if not isinstance(value, dict):
        reporter.fail(f"{label} root must be a JSON object: {path}")
        return None
    return value


def require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PromotionError(f"{context}.{key} must be a non-empty string")
    return value


def require_list(mapping: dict[str, Any], key: str, context: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise PromotionError(f"{context}.{key} must be an array")
    return value


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromotionError(f"{context} must be an object")
    return value


def require_positive_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PromotionError(f"{context}.{key} must be a positive integer")
    return value


def require_digest(mapping: dict[str, Any], key: str, context: str) -> str:
    digest = require_string(mapping, key, context)
    if not SHA256_RE.fullmatch(digest):
        raise PromotionError(f"{context}.{key} must be a lowercase SHA-256 digest")
    return digest


def safe_relative_path(value: str, context: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or "\\" in value or any(part in {"", ".", ".."} for part in path.parts):
        raise PromotionError(f"{context} must be a portable relative path")
    return path


def resolve_under(root: Path, relative: Path, context: str) -> Path:
    candidate = root / relative
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise PromotionError(f"{context} escapes {root}") from error
    if candidate.is_symlink():
        raise PromotionError(f"{context} must not be a symlink")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decimal_value(value: Any, context: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PromotionError(f"{context} must be a JSON number or decimal string")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise PromotionError(f"{context} is not a decimal number") from error
    if not result.is_finite():
        raise PromotionError(f"{context} must be finite")
    return result


def verify_source_digest(
    root: Path,
    source: dict[str, Any],
    context: str,
    reporter: Reporter,
) -> Decimal:
    path_text = require_string(source, "path", context)
    relative = safe_relative_path(path_text, f"{context}.path")
    expected_digest = require_digest(source, "sha256", context)
    candidate = resolve_under(root, relative, f"{context}.path")
    if not candidate.is_file():
        raise PromotionError(f"{context}.path missing: {path_text}")
    observed_digest = sha256_file(candidate)
    reporter.log(
        f"source={path_text} sha256_expected={expected_digest} sha256_observed={observed_digest}"
    )
    if not hmac.compare_digest(expected_digest, observed_digest):
        raise PromotionError(
            f"{context}.sha256 mismatch for {path_text}: expected {expected_digest}, observed {observed_digest}"
        )
    return decimal_value(source.get("value"), f"{context}.value")


def validate_benchmarks(root: Path, record_path: Path, reporter: Reporter) -> None:
    record = load_object(record_path, reporter, "benchmark_conflicts")
    if record is None:
        return
    try:
        rows = require_list(record, "benchmarks", "benchmark_conflicts")
    except PromotionError as error:
        reporter.fail(str(error))
        return

    by_name: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        context = f"benchmark_conflicts.benchmarks[{index}]"
        try:
            row = require_mapping(raw_row, context)
            name = require_string(row, "benchmark", context)
        except PromotionError as error:
            reporter.fail(str(error))
            continue
        if name in by_name:
            reporter.fail(f"duplicate benchmark row: {name}")
            continue
        by_name[name] = row

    observed_names = set(by_name)
    missing = sorted(set(EXPECTED_BENCHMARKS) - observed_names)
    unexpected = sorted(observed_names - set(EXPECTED_BENCHMARKS))
    for name in missing:
        reporter.fail(f"benchmark missing required row: {name}")
    for name in unexpected:
        reporter.fail(f"benchmark has unexpected row: {name}")

    for name in sorted(set(EXPECTED_BENCHMARKS) & observed_names):
        row = by_name[name]
        context = f"benchmark[{name}]"
        try:
            status = require_string(row, "state", context)
            thinking = require_mapping(row.get("thinking_mode"), f"{context}.thinking_mode")
            if thinking.get("enabled") is not True or thinking.get("preserve_thinking") is not True:
                raise PromotionError(
                    f"{context}.thinking_mode must disclose enabled=true and preserve_thinking=true"
                )
            sources = require_list(row, "sources", context)
            if not sources:
                raise PromotionError(f"{context}.sources must not be empty")
            values = {
                verify_source_digest(root, require_mapping(source, f"{context}.sources[{index}]"),
                                     f"{context}.sources[{index}]", reporter)
                for index, source in enumerate(sources)
            }
            if name == "HMMT-Feb-2026":
                if status != "PARTIAL":
                    raise PromotionError(f"{context}.state must be PARTIAL")
                if values != HMMT_CONFLICT_VALUES:
                    raise PromotionError(
                        f"{context} must retain exactly both conflict values 82.8 and 82.1, observed {sorted(values)}"
                    )
            else:
                if status != "REPORTED":
                    raise PromotionError(f"{context}.state must be REPORTED")
                expected = EXPECTED_BENCHMARKS[name]
                if values != {expected}:
                    raise PromotionError(
                        f"{context} values mismatch: expected only {expected}, observed {sorted(values)}"
                    )
            reporter.log(f"benchmark={name} state={status} values={','.join(map(str, sorted(values)))}")
        except PromotionError as error:
            reporter.fail(str(error))


def verify_evidence(
    root: Path,
    record_id: str,
    evidence: dict[str, Any],
    context: str,
    reporter: Reporter,
) -> None:
    path_text = require_string(evidence, "path", context)
    relative = safe_relative_path(path_text, f"{context}.path")
    expected_digest = require_digest(evidence, "sha256", context)
    line_start = require_positive_int(evidence, "line_start", context)
    line_end = require_positive_int(evidence, "line_end", context)
    expected_text = evidence.get("expected_text")
    expected_span_digest = evidence.get("expected_text_sha256")
    if expected_text is not None:
        if not isinstance(expected_text, str) or not expected_text:
            raise PromotionError(f"{context}.expected_text must be a non-empty string")
        if expected_span_digest is not None:
            raise PromotionError(
                f"{context} must provide expected_text or expected_text_sha256, not both"
            )
    elif expected_span_digest is not None:
        if not isinstance(expected_span_digest, str) or not SHA256_RE.fullmatch(expected_span_digest):
            raise PromotionError(
                f"{context}.expected_text_sha256 must be a lowercase SHA-256 digest"
            )
    else:
        raise PromotionError(f"{context} must provide expected_text or expected_text_sha256")
    if line_end < line_start:
        raise PromotionError(f"{context}.line_end must be at least line_start")
    candidate = resolve_under(root, relative, f"{context}.path")
    if not candidate.is_file():
        reporter.mark_drifted(record_id, path_text)
        raise PromotionError(f"{context}.path missing: {path_text}")
    raw = candidate.read_bytes()
    observed_digest = hashlib.sha256(raw).hexdigest()
    reporter.log(
        f"oq={record_id} evidence={path_text} lines={line_start}-{line_end} "
        f"sha256_expected={expected_digest} sha256_observed={observed_digest}"
    )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        reporter.mark_drifted(record_id, path_text)
        raise PromotionError(f"{context}.path is not UTF-8 text: {path_text}") from error
    if line_end > len(lines):
        reporter.mark_drifted(record_id, path_text)
        raise PromotionError(
            f"{context} line span {line_start}-{line_end} exceeds {path_text} line count {len(lines)}"
        )
    observed_text = "\n".join(lines[line_start - 1 : line_end])
    observed_span_digest = hashlib.sha256(observed_text.encode("utf-8")).hexdigest()
    text_matches = (
        expected_text == observed_text
        if expected_text is not None
        else hmac.compare_digest(expected_span_digest, observed_span_digest)
    )
    if not hmac.compare_digest(expected_digest, observed_digest) or not text_matches:
        reporter.mark_drifted(record_id, path_text)
        reporter.log(
            f"DRIFT oq={record_id} evidence={path_text} "
            f"expected_text={expected_text!r} expected_text_sha256={expected_span_digest} "
            f"observed_text={observed_text[:240]!r} "
            f"observed_text_sha256={observed_span_digest}"
        )
        raise PromotionError(
            f"{context} drift for {path_text}: expected digest/text differs from observed span"
        )


def validate_promotions(root: Path, repo_root: Path, record_path: Path, reporter: Reporter) -> int:
    record = load_object(record_path, reporter, "promotions")
    if record is None:
        return 0
    try:
        promotions = require_list(record, "promotions", "promotions")
    except PromotionError as error:
        reporter.fail(str(error))
        return 0

    by_oq: dict[str, dict[str, Any]] = {}
    for index, raw_promotion in enumerate(promotions):
        context = f"promotions.promotions[{index}]"
        try:
            promotion = require_mapping(raw_promotion, context)
            oq_id = require_string(promotion, "oq_id", context)
        except PromotionError as error:
            reporter.fail(str(error))
            continue
        if oq_id in by_oq:
            reporter.fail(f"duplicate promotion record: {oq_id}")
            continue
        by_oq[oq_id] = promotion

    observed_oqs = set(by_oq)
    for oq_id in sorted(EXPECTED_OQS - observed_oqs):
        reporter.fail(f"promotion missing required record: {oq_id}")
    for oq_id in sorted(observed_oqs - EXPECTED_OQS):
        reporter.fail(f"promotion has unexpected OQ record: {oq_id}")

    for oq_id in sorted(EXPECTED_OQS & observed_oqs):
        promotion = by_oq[oq_id]
        context = f"promotion[{oq_id}]"
        evidence_paths: list[str] = []
        try:
            require_string(promotion, "claim", context)
            old_state = require_string(promotion, "old_state", context)
            new_state = require_string(promotion, "new_state", context)
            if (old_state, new_state) not in LEGAL_PROMOTION_TRANSITIONS:
                raise PromotionError(
                    f"{context} has illegal transition {old_state}->{new_state}"
                )
            fixture_text = require_string(promotion, "replay_fixture", context)
            fixture = resolve_under(
                repo_root,
                safe_relative_path(fixture_text, f"{context}.replay_fixture"),
                f"{context}.replay_fixture",
            )
            if not fixture.is_file():
                raise PromotionError(f"{context}.replay_fixture missing: {fixture_text}")
            if new_state in {"PARTIAL", "OPEN"}:
                remainders = require_list(promotion, "open_remainders", context)
                if not remainders:
                    raise PromotionError(
                        f"{context}.open_remainders must not be empty for {new_state}"
                    )
                for remainder_index, remainder in enumerate(remainders):
                    remainder_context = f"{context}.open_remainders[{remainder_index}]"
                    require_string(require_mapping(remainder, remainder_context), "blocking_phase", remainder_context)
            evidence_entries = require_list(promotion, "evidence", context)
            if not evidence_entries:
                raise PromotionError(f"{context}.evidence must not be empty")
            for evidence_index, raw_evidence in enumerate(evidence_entries):
                evidence_context = f"{context}.evidence[{evidence_index}]"
                evidence = require_mapping(raw_evidence, evidence_context)
                evidence_paths.append(require_string(evidence, "path", evidence_context))
                verify_evidence(root, oq_id, evidence, evidence_context, reporter)
            reporter.log(
                f"oq={oq_id} old_state={old_state} new_state={new_state} "
                f"evidence={','.join(evidence_paths)} fixture={fixture_text}"
            )
        except PromotionError as error:
            reporter.fail(
                f"{error}; oq={oq_id} evidence={','.join(evidence_paths) if evidence_paths else '<unread>'}"
            )
    return len(by_oq)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_truth_pack = Path(__file__).resolve().parents[1] / "docs" / "truth-pack"
    parser.add_argument("--truth-pack", type=Path, default=default_truth_pack)
    parser.add_argument("--benchmark-conflicts", type=Path)
    parser.add_argument("--promotions", type=Path)
    args = parser.parse_args()

    truth_pack = args.truth_pack.resolve()
    reporter = Reporter()
    benchmark_path = args.benchmark_conflicts or truth_pack / "benchmark_conflicts.json"
    promotions_path = args.promotions or truth_pack / "promotions.json"
    repo_root = truth_pack.parents[1]

    validate_benchmarks(truth_pack, benchmark_path, reporter)
    record_count = validate_promotions(truth_pack, repo_root, promotions_path, reporter)
    status = "PASS" if not reporter.failures else "FAIL"
    drifted = ",".join(sorted(reporter.drifted)) if reporter.drifted else "none"
    reporter.log(
        f"RESULT={status} records={record_count} drifted={drifted}"
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
