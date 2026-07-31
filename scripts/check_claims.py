#!/usr/bin/env python3
"""Validate the public claim registry and mechanically annotated wording."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
STATES = ("targeted", "observed", "evidenced", "withdrawn")
STATE_RANK = {state: index for index, state in enumerate(STATES[:3])}
CLAIM_ID_RE = re.compile(r"[a-z][a-z0-9-]{1,79}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPIRY_RE = re.compile(r"(?P<kind>on_change|on_date|before_release): (?P<detail>.+)\Z")
ANNOTATION_RE = re.compile(
    r"fnlp-claim:\s*(?P<claim_id>[a-z][a-z0-9-]{1,79})\s*;\s*wording=(?P<wording>targeted|observed|evidenced|withdrawn)\b"
)
R4_CONTEXT_ANNOTATION_RE = re.compile(
    r"fnlp-r4-context:\s*ledger=(?P<ledger>PERF-[A-Z0-9-]+)\b"
)
PRACTICALITY_WORD_RE = re.compile(
    r"\b(?:usable|support(?:s|ed|ing)|handle(?:s|d|ing)|admission)\b",
    re.IGNORECASE,
)
CONTEXT_LENGTH_CLAUSE_RE = re.compile(
    r"\b(?:context|ctx)(?:\s+(?:length|window|cap))?\s*(?:(?:is|of|at|up\s+to)\s*)?:?\s*"
    r"(?P<value>\d{1,3}(?:,\d{3})+|\d+)\b",
    re.IGNORECASE,
)
CONTEXT_AMOUNT_RE = re.compile(
    r"\b(?P<value>\d{1,3}(?:,\d{3})+|\d+)(?:\s*(?P<unit>[kK]|tokens?|tok(?:ens)?|positions?))?\b",
    re.IGNORECASE,
)
OBSERVED_MODEL_LIMIT_CLAUSE_RE = re.compile(
    r"\bobserved\s+model(?:-card)?\s+limit\b(?:\s+(?:is|of|at))?\s*:?\s*(?:up\s+to\s+)?"
    r"(?P<value>\d{1,3}(?:,\d{3})+|\d+)\s*(?:tokens?|positions?)\b",
    re.IGNORECASE,
)
PERF_ENTRY_RE = re.compile(r"^## (?P<entry>PERF-[A-Z0-9-]+)\s*$")
LEDGER_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
R4_EVIDENCE_RECEIPT_RE = re.compile(
    r"\b(?P<kind>r4-receipt|admission-receipt)="
    r"(?P<path>(?:docs/evidence|tests/fixtures/claims)/[A-Za-z0-9_./-]+)#sha256:(?P<digest>[0-9a-f]{64})\b"
)
R4_RECEIPT_SCHEMA_VERSION = 1
R4_COMMON_RECEIPT_KEYS = {
    "artifact",
    "claim_id",
    "context",
    "cpu_feature_string",
    "host_fingerprint",
    "kind",
    "ledger_entry",
    "schema_version",
    "validity_domain",
}
R4_ARTIFACT_KEYS = {"kernel_table_sha256", "load_mode", "packing_sha256", "recipe_id"}
R4_CONTEXT_KEYS = {"kv_dtype", "tokens"}
R4_MEASUREMENT_KEYS = {"decode_tokens_per_s", "kv_bytes", "peak_rss_bytes", "prefill_ms"}
R4_ADMISSION_KEYS = {"committed_bytes", "outcome", "peak_bytes"}
PERCENTILE_KEYS = {"p50", "p95", "p99"}
R4_REQUIRED_FIELDS = (
    "Host fingerprint",
    "Artifact recipe + packing + kernel table + load mode",
    "Context point",
    "p50/p95/p99",
    "R4 measurement summary",
    "Fairness controls",
    "Admission boundary outcomes",
)
DEFAULT_CONTEXT_CAP = 8_192
NUMERIC_RE = re.compile(r"(?<![A-Za-z_])\d+(?:[.,]\d+)?(?:\s*(?:%|×|x|tok/s|KiB|MiB|GiB|GB|B))?", re.IGNORECASE)
SUPERLATIVE_RE = re.compile(
    r"\b(?:best|fastest|faster|smallest|largest|lowest|highest|only|exact|verified|proven|guaranteed|deterministic|secure|safe)\b",
    re.IGNORECASE,
)
TOP_LEVEL_KEYS = {"claims", "schema_version"}
CLAIM_KEYS = {
    "evidence_artifact_digests",
    "expiry_revalidation_trigger",
    "id",
    "probability_space",
    "public_surfaces",
    "state",
    "transition_history",
    "validity_domain",
    "wording_scope",
}
REQUIRED_CLAIM_KEYS = CLAIM_KEYS - {"probability_space"}
VALID_SURFACES = {"README", "--help", "release_notes", "robot_schema_descriptions"}
DOMAIN_KEYS = {"dataset", "host", "numerics_profile", "prompt_hash", "recipe_id", "thinking_mode"}
TRANSITION_KEYS = {"at", "from_state", "reason", "to_state"}
P7_REGISTER_ROWS = {
    "P7-METAL": "| P7-METAL | DEFERRED |",
    "P7-SERVE": "| P7-SERVE | DEFERRED |",
    "P7-TRANSLATION": "| P7-TRANSLATION | DEFERRED |",
}
P7_SCOPE_REQUIREMENTS = (
    "**Status:** DEFERRED — no implementation authority",
    "`BLOCKED_CPU_PARITY`",
    "`metal-prefill-v1`",
    "`fnlp serve` or any remote/routable listener",
    "Translation | DEFERRED and not a task surface.",
    "CPU-only build/host retains identical CPU behavior",
)
ALLOWED_TRANSITIONS = {
    "targeted": {"observed", "evidenced", "withdrawn"},
    "observed": {"evidenced", "withdrawn"},
    "evidenced": {"withdrawn"},
    "withdrawn": set(),
}


class ClaimsError(Exception):
    """A registry or public-wording contract failed."""


class DuplicateKeyError(ClaimsError):
    """A JSON object repeated a key before semantic validation."""


@dataclass(frozen=True)
class Annotation:
    claim_id: str
    wording: str


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{timestamp()} CLAIMS {message}", file=sys.stderr)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ClaimsError(f"registry unavailable file={path}: {error}") from error
    try:
        value = json.loads(raw, object_pairs_hook=no_duplicate_object)
    except (json.JSONDecodeError, DuplicateKeyError) as error:
        raise ClaimsError(f"registry is not duplicate-key-free JSON file={path}: {error}") from error
    if not isinstance(value, dict):
        raise ClaimsError(f"registry must be a JSON object file={path}")
    return value, raw


def require_string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimsError(f"expected non-empty string at {location}: observed={value!r}")
    return value


def validate_expiry_trigger(value: Any, *, location: str) -> None:
    trigger = require_string(value, location=location)
    match = EXPIRY_RE.fullmatch(trigger)
    if match is None:
        raise ClaimsError(f"expiry trigger grammar violation at {location}: {trigger!r}")
    if match["kind"] == "on_date":
        try:
            date.fromisoformat(match["detail"])
        except ValueError as error:
            raise ClaimsError(f"expiry date is invalid at {location}: {match['detail']!r}") from error


def validate_transition_history(history: Any, state: str, *, location: str) -> None:
    if not isinstance(history, list):
        raise ClaimsError(f"claim transition_history must be a list at {location}")
    previous_to: str | None = None
    for index, transition in enumerate(history):
        transition_location = f"{location}/{index}"
        if not isinstance(transition, dict) or set(transition) != TRANSITION_KEYS:
            raise ClaimsError(f"transition must have exactly {sorted(TRANSITION_KEYS)!r} at {transition_location}")
        from_state = transition["from_state"]
        to_state = transition["to_state"]
        if from_state not in STATES or to_state not in STATES:
            raise ClaimsError(f"transition state is invalid at {transition_location}")
        if to_state not in ALLOWED_TRANSITIONS[from_state]:
            raise ClaimsError(f"transition is not allowed at {transition_location}: {from_state}->{to_state}")
        if previous_to is not None and from_state != previous_to:
            raise ClaimsError(f"transition history is discontinuous at {transition_location}")
        try:
            date.fromisoformat(require_string(transition["at"], location=f"{transition_location}/at"))
        except ValueError as error:
            raise ClaimsError(f"transition date is invalid at {transition_location}") from error
        require_string(transition["reason"], location=f"{transition_location}/reason")
        previous_to = to_state
    if previous_to is not None and previous_to != state:
        raise ClaimsError(f"transition history does not end at current state={state!r} at {location}")


def validate_registry(registry: dict[str, Any], raw: bytes | None = None) -> dict[str, dict[str, Any]]:
    unknown_top_level = sorted(set(registry) - TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise ClaimsError(f"registry unknown key {unknown_top_level[0]!r}")
    if set(registry) != TOP_LEVEL_KEYS:
        missing = sorted(TOP_LEVEL_KEYS - set(registry))
        raise ClaimsError(f"registry missing key {missing[0]!r}")
    if registry["schema_version"] != SCHEMA_VERSION:
        raise ClaimsError(
            f"registry schema_version expected={SCHEMA_VERSION} observed={registry['schema_version']!r}"
        )
    claims = registry["claims"]
    if not isinstance(claims, list) or not claims:
        raise ClaimsError("registry claims must be a non-empty list")
    if raw is not None and raw != canonical_json_bytes(registry):
        raise ClaimsError("registry is not canonical JSON (sorted keys, two-space indentation, trailing newline)")

    by_id: dict[str, dict[str, Any]] = {}
    for index, claim in enumerate(claims):
        location = f"/claims/{index}"
        if not isinstance(claim, dict):
            raise ClaimsError(f"claim must be object at {location}")
        unknown = sorted(set(claim) - CLAIM_KEYS)
        if unknown:
            raise ClaimsError(f"claim unknown key at {location}: {unknown[0]!r}")
        missing = sorted(REQUIRED_CLAIM_KEYS - set(claim))
        if missing:
            raise ClaimsError(f"claim missing key at {location}: {missing[0]!r}")
        claim_id = require_string(claim["id"], location=f"{location}/id")
        if not CLAIM_ID_RE.fullmatch(claim_id):
            raise ClaimsError(f"claim id grammar violation at {location}/id: {claim_id!r}")
        if claim_id in by_id:
            raise ClaimsError(f"duplicate claim id {claim_id!r} at {location}/id")
        state = claim["state"]
        if state not in STATES:
            raise ClaimsError(f"claim invalid state at {location}/state: {state!r}")
        require_string(claim["wording_scope"], location=f"{location}/wording_scope")
        validate_expiry_trigger(claim["expiry_revalidation_trigger"], location=f"{location}/expiry_revalidation_trigger")

        domain = claim["validity_domain"]
        if not isinstance(domain, dict) or set(domain) != DOMAIN_KEYS:
            raise ClaimsError(f"claim validity_domain must have exactly {sorted(DOMAIN_KEYS)!r} at {location}")
        for key, value in domain.items():
            require_string(value, location=f"{location}/validity_domain/{key}")

        surfaces = claim["public_surfaces"]
        if not isinstance(surfaces, list) or not surfaces:
            raise ClaimsError(f"claim public_surfaces must be a non-empty list at {location}")
        if len(surfaces) != len(set(surfaces)):
            raise ClaimsError(f"claim public_surfaces contains a duplicate at {location}")
        for surface in surfaces:
            if surface not in VALID_SURFACES:
                raise ClaimsError(f"claim invalid public surface at {location}: {surface!r}")

        digests = claim["evidence_artifact_digests"]
        if not isinstance(digests, list):
            raise ClaimsError(f"claim evidence_artifact_digests must be a list at {location}")
        for digest in digests:
            if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
                raise ClaimsError(f"claim invalid evidence digest at {location}: {digest!r}")
        if state == "evidenced" and not digests:
            raise ClaimsError(f"evidenced claim requires an evidence digest at {location}")

        history = claim["transition_history"]
        validate_transition_history(history, state, location=f"{location}/transition_history")
        if state == "withdrawn" and not history:
            raise ClaimsError(f"withdrawn claim must retain transition history at {location}")
        if "probability_space" in claim:
            require_string(claim["probability_space"], location=f"{location}/probability_space")
        by_id[claim_id] = claim
    return by_id


def tier_is_permitted(annotation: Annotation, claim: dict[str, Any]) -> bool:
    state = claim["state"]
    if state == "withdrawn":
        return annotation.wording == "withdrawn"
    return annotation.wording in STATE_RANK and STATE_RANK[annotation.wording] <= STATE_RANK[state]


def is_claim_bearing(line: str) -> bool:
    return bool(NUMERIC_RE.search(line) or SUPERLATIVE_RE.search(line))


def field_is_measured(value: str | None) -> bool:
    """Reject a placeholder where an R4 receipt requires a measured value."""

    if value is None or not value.strip():
        return False
    return re.search(r"\b(?:pending|blocked|deferred|n/?a|none|not\s+(?:measured|available))\b", value, re.IGNORECASE) is None


def retained_regular_file(root: Path, relative_path: str) -> Path | None:
    """Resolve a repository-relative retained artifact without following links."""

    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    current = root
    for part in candidate.parts:
        current = current / part
        try:
            if current.is_symlink():
                return None
        except OSError:
            return None
    try:
        if not current.is_file():
            return None
    except OSError:
        return None
    return current


def require_positive_int(value: Any, *, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ClaimsError(f"expected positive integer at {location}: observed={value!r}")
    return value


def require_positive_number(value: Any, *, location: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ClaimsError(f"expected positive number at {location}: observed={value!r}")
    return value


def receipt_number(value: int | float) -> str:
    """Render a JSON number for the fixed R4 ledger binding grammar."""

    return format(value, ".17g")


def r4_artifact_binding(artifact: dict[str, Any]) -> str:
    return (
        f"recipe_id={artifact['recipe_id']}; packing_sha256={artifact['packing_sha256']}; "
        f"kernel_table_sha256={artifact['kernel_table_sha256']}; load_mode={artifact['load_mode']}"
    )


def r4_context_binding(context: dict[str, Any]) -> str:
    return f"tokens={context['tokens']}; kv_dtype={context['kv_dtype']}"


def r4_percentile_binding(measurement: dict[str, Any]) -> str:
    parts: list[str] = []
    for metric in ("prefill_ms", "decode_tokens_per_s"):
        percentiles = measurement[metric]
        rendered = ",".join(f"{key}={receipt_number(percentiles[key])}" for key in ("p50", "p95", "p99"))
        parts.append(f"{metric}({rendered})")
    return "; ".join(parts)


def r4_measurement_summary(measurement: dict[str, Any]) -> str:
    return f"kv_bytes={measurement['kv_bytes']}; peak_rss_bytes={measurement['peak_rss_bytes']}"


def r4_admission_binding(admission: dict[str, Any]) -> str:
    return (
        f"outcome={admission['outcome']}; committed_bytes={admission['committed_bytes']}; "
        f"peak_bytes={admission['peak_bytes']}"
    )


def load_r4_receipt(path: Path, expected_digest: str, expected_kind: str) -> dict[str, Any]:
    """Load a canonical typed R4 receipt and verify its content address."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ClaimsError(f"R4 receipt unavailable file={path}: {error}") from error
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise ClaimsError(f"R4 receipt digest mismatch file={path} expected={expected_digest} observed={actual_digest}")
    try:
        receipt = json.loads(raw, object_pairs_hook=no_duplicate_object)
    except (json.JSONDecodeError, DuplicateKeyError) as error:
        raise ClaimsError(f"R4 receipt is not duplicate-key-free JSON file={path}: {error}") from error
    if not isinstance(receipt, dict) or raw != canonical_json_bytes(receipt):
        raise ClaimsError(f"R4 receipt is not canonical JSON file={path}")
    if receipt.get("schema_version") != R4_RECEIPT_SCHEMA_VERSION:
        raise ClaimsError(f"R4 receipt schema version is invalid file={path}")
    if receipt.get("kind") != expected_kind:
        raise ClaimsError(f"R4 receipt kind is invalid file={path} expected={expected_kind}")

    expected_keys = set(R4_COMMON_RECEIPT_KEYS)
    expected_keys.add("measurement" if expected_kind == "r4-measurement" else "admission")
    if set(receipt) != expected_keys:
        raise ClaimsError(f"R4 receipt keys are invalid file={path}")
    for key in ("ledger_entry", "claim_id", "cpu_feature_string", "host_fingerprint"):
        require_string(receipt[key], location=f"R4 receipt {path}/{key}")
    if not CLAIM_ID_RE.fullmatch(receipt["claim_id"]):
        raise ClaimsError(f"R4 receipt claim id grammar violation file={path}")

    domain = receipt["validity_domain"]
    if not isinstance(domain, dict) or set(domain) != DOMAIN_KEYS:
        raise ClaimsError(f"R4 receipt validity domain keys are invalid file={path}")
    for key, value in domain.items():
        require_string(value, location=f"R4 receipt {path}/validity_domain/{key}")

    artifact = receipt["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != R4_ARTIFACT_KEYS:
        raise ClaimsError(f"R4 receipt artifact keys are invalid file={path}")
    require_string(artifact["recipe_id"], location=f"R4 receipt {path}/artifact/recipe_id")
    require_string(artifact["load_mode"], location=f"R4 receipt {path}/artifact/load_mode")
    for key in ("packing_sha256", "kernel_table_sha256"):
        if not isinstance(artifact[key], str) or not DIGEST_RE.fullmatch(artifact[key]):
            raise ClaimsError(f"R4 receipt artifact digest is invalid file={path} key={key}")
    if artifact["recipe_id"] != domain["recipe_id"]:
        raise ClaimsError(f"R4 receipt artifact recipe does not match validity domain file={path}")
    if receipt["host_fingerprint"] != domain["host"]:
        raise ClaimsError(f"R4 receipt host does not match validity domain file={path}")

    context = receipt["context"]
    if not isinstance(context, dict) or set(context) != R4_CONTEXT_KEYS:
        raise ClaimsError(f"R4 receipt context keys are invalid file={path}")
    if require_positive_int(context["tokens"], location=f"R4 receipt {path}/context/tokens") <= DEFAULT_CONTEXT_CAP:
        raise ClaimsError(f"R4 receipt context must exceed default cap file={path}")
    require_string(context["kv_dtype"], location=f"R4 receipt {path}/context/kv_dtype")

    if expected_kind == "r4-measurement":
        measurement = receipt["measurement"]
        if not isinstance(measurement, dict) or set(measurement) != R4_MEASUREMENT_KEYS:
            raise ClaimsError(f"R4 measurement keys are invalid file={path}")
        for metric in ("prefill_ms", "decode_tokens_per_s"):
            distribution = measurement[metric]
            if not isinstance(distribution, dict) or set(distribution) != PERCENTILE_KEYS:
                raise ClaimsError(f"R4 measurement percentile keys are invalid file={path} metric={metric}")
            for percentile, value in distribution.items():
                require_positive_number(value, location=f"R4 receipt {path}/{metric}/{percentile}")
        for metric in ("kv_bytes", "peak_rss_bytes"):
            require_positive_int(measurement[metric], location=f"R4 receipt {path}/{metric}")
    else:
        admission = receipt["admission"]
        if not isinstance(admission, dict) or set(admission) != R4_ADMISSION_KEYS:
            raise ClaimsError(f"R4 admission keys are invalid file={path}")
        if admission["outcome"] != "admitted":
            raise ClaimsError(f"R4 admission outcome must be admitted file={path}")
        for key in ("committed_bytes", "peak_bytes"):
            require_positive_int(admission[key], location=f"R4 receipt {path}/admission/{key}")
    return receipt


def r4_evidence_is_retained(
    root: Path,
    entry_id: str,
    fields: dict[str, str],
    fixture_hashes: set[str],
    claims: Mapping[str, dict[str, Any]],
    *,
    allow_fixture_receipts: bool,
) -> bool:
    """Require typed receipts whose contents exactly bind the R4 ledger row."""

    matches = list(R4_EVIDENCE_RECEIPT_RE.finditer(fields.get("Evidence", "")))
    if len(matches) != 2:
        return False
    receipt_paths: dict[str, tuple[str, str]] = {}
    for match in matches:
        if match["kind"] in receipt_paths:
            return False
        receipt_paths[match["kind"]] = (match["path"], match["digest"])
    if set(receipt_paths) != {"r4-receipt", "admission-receipt"}:
        return False
    if receipt_paths["r4-receipt"][0] == receipt_paths["admission-receipt"][0]:
        return False

    receipts: dict[str, dict[str, Any]] = {}
    expected_kinds = {"r4-receipt": "r4-measurement", "admission-receipt": "r4-admission"}
    for evidence_kind, (path_text, digest) in receipt_paths.items():
        if not path_text.startswith("docs/evidence/") and not (
            allow_fixture_receipts and path_text.startswith("tests/fixtures/claims/")
        ):
            return False
        if f"sha256:{digest}" not in fixture_hashes:
            return False
        path = retained_regular_file(root, path_text)
        if path is None:
            return False
        try:
            receipts[evidence_kind] = load_r4_receipt(path, digest, expected_kinds[evidence_kind])
        except ClaimsError:
            return False

    measurement = receipts["r4-receipt"]
    admission = receipts["admission-receipt"]
    claim_id = fields.get("Claim ID")
    claim = claims.get(claim_id or "")
    if claim is None or claim["state"] != "evidenced":
        return False
    for key in ("ledger_entry", "claim_id", "validity_domain", "host_fingerprint", "cpu_feature_string", "artifact", "context"):
        if measurement[key] != admission[key]:
            return False
    if measurement["ledger_entry"] != entry_id or measurement["claim_id"] != claim_id:
        return False
    if measurement["validity_domain"] != claim["validity_domain"]:
        return False
    if fields.get("Host fingerprint") != measurement["host_fingerprint"]:
        return False
    if fields.get("CPU feature string") != measurement["cpu_feature_string"]:
        return False
    if fields.get("Artifact recipe + packing + kernel table + load mode") != r4_artifact_binding(measurement["artifact"]):
        return False
    if fields.get("Context point") != r4_context_binding(measurement["context"]):
        return False
    if fields.get("p50/p95/p99") != r4_percentile_binding(measurement["measurement"]):
        return False
    if fields.get("R4 measurement summary") != r4_measurement_summary(measurement["measurement"]):
        return False
    if fields.get("Admission boundary outcomes") != r4_admission_binding(admission["admission"]):
        return False
    return True


def eligible_r4_ledger_entries(
    root: Path,
    claims: Mapping[str, dict[str, Any]],
    *,
    ledger_path: Path | None = None,
) -> dict[str, str]:
    """Return measured R4 ledger rows eligible to support public context claims."""

    path = ledger_path if ledger_path is not None else root / "docs" / "PERF_LEDGER.md"
    allow_fixture_receipts = ledger_path is not None and path.is_relative_to(root / "tests" / "fixtures")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ClaimsError(f"R4 ledger unavailable file={path}: {error}") from error

    entries: dict[str, dict[str, str]] = {}
    entry_id: str | None = None
    fields: dict[str, str] = {}
    for line in lines:
        match = PERF_ENTRY_RE.fullmatch(line)
        if match:
            if entry_id is not None:
                entries[entry_id] = fields
            entry_id = match["entry"]
            fields = {}
            continue
        if entry_id is None or not line.startswith("- ") or ": " not in line:
            continue
        name, value = line[2:].split(": ", maxsplit=1)
        fields[name] = value
    if entry_id is not None:
        entries[entry_id] = fields

    eligible: dict[str, str] = {}
    for candidate, fields in entries.items():
        fixture_hashes = [item.strip() for item in fields.get("Fixture hashes", "").split(",")]
        fixture_hash_set = set(fixture_hashes)
        claim_id = fields.get("Claim ID", "")
        if (
            fields.get("Regime") == "R4-long-context"
            and fields.get("Disposition") == "won"
            and fixture_hashes
            and all(LEDGER_SHA256_RE.fullmatch(item) for item in fixture_hashes)
            and all(field_is_measured(fields.get(field)) for field in R4_REQUIRED_FIELDS)
            and all(percentile in fields["p50/p95/p99"].lower() for percentile in ("p50", "p95", "p99"))
            and r4_evidence_is_retained(
                root,
                candidate,
                fields,
                fixture_hash_set,
                claims,
                allow_fixture_receipts=allow_fixture_receipts,
            )
        ):
            eligible[candidate] = claim_id
    return eligible


def context_amount_exceeds_default(line: str) -> bool:
    """Recognize explicit context quantities above the 8K product default."""

    observed_limit_values = {
        match.span("value") for match in OBSERVED_MODEL_LIMIT_CLAUSE_RE.finditer(line)
    }
    direct_context_values = {
        match.span("value") for match in CONTEXT_LENGTH_CLAUSE_RE.finditer(line)
    }
    practicality_language = PRACTICALITY_WORD_RE.search(line) is not None
    for match in CONTEXT_AMOUNT_RE.finditer(line):
        value = int(match["value"].replace(",", ""))
        unit = (match["unit"] or "").lower()
        if unit == "k":
            value *= 1024
        nearby_text = line[max(0, match.start() - 48) : min(len(line), match.end() + 48)]
        observed_limit_value = match.span("value") in observed_limit_values
        if (
            match.span("value") not in direct_context_values
            and PRACTICALITY_WORD_RE.search(nearby_text) is None
            and not (observed_limit_value and practicality_language)
        ):
            continue
        if observed_limit_value and not practicality_language:
            continue
        if value > DEFAULT_CONTEXT_CAP:
            return True
    return False


def validate_r4_context_claim(
    *,
    path: Path,
    line_number: int,
    active: Annotation | None,
    r4_ledger: str | None,
    claims: dict[str, dict[str, Any]],
    eligible_ledgers: Mapping[str, str],
) -> None:
    if active is None:
        raise ClaimsError(f"file={path}:{line_number} R4 context claim missing fnlp-claim annotation")
    claim = claims[active.claim_id]
    if active.wording != "evidenced" or claim["state"] != "evidenced":
        raise ClaimsError(
            f"file={path}:{line_number} R4 context claim requires an evidenced claim id={active.claim_id}"
        )
    if r4_ledger is None:
        raise ClaimsError(f"file={path}:{line_number} R4 context claim missing fnlp-r4-context ledger annotation")
    if r4_ledger not in eligible_ledgers:
        raise ClaimsError(
            f"file={path}:{line_number} R4 context ledger={r4_ledger} lacks a typed, bound won R4 receipt"
        )
    if eligible_ledgers[r4_ledger] != active.claim_id:
        raise ClaimsError(
            f"file={path}:{line_number} R4 context ledger={r4_ledger} is bound to claim "
            f"id={eligible_ledgers[r4_ledger]}, not active id={active.claim_id}"
        )


def scan_surface(
    path: Path,
    surface: str,
    claims: dict[str, dict[str, Any]],
    eligible_r4_ledgers: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ClaimsError(f"public surface is not UTF-8 file={path}: {error}") from error
    active: Annotation | None = None
    active_r4: tuple[str, int] | None = None
    annotations = 0
    claim_lines = 0
    r4_context_lines = 0
    if eligible_r4_ledgers is None:
        eligible_r4_ledgers = {}
    for line_number, line in enumerate(lines, start=1):
        consumed_r4 = False
        if active_r4 is not None:
            if not context_amount_exceeds_default(line):
                ledger, annotation_line = active_r4
                raise ClaimsError(
                    f"file={path}:{annotation_line} fnlp-r4-context ledger={ledger} must immediately precede "
                    "one >8K context claim"
                )
            ledger, _ = active_r4
            validate_r4_context_claim(
                path=path,
                line_number=line_number,
                active=active,
                r4_ledger=ledger,
                claims=claims,
                eligible_ledgers=eligible_r4_ledgers,
            )
            active_r4 = None
            r4_context_lines += 1
            consumed_r4 = True
        match = ANNOTATION_RE.search(line)
        if match:
            active = Annotation(match.group("claim_id"), match.group("wording"))
            annotations += 1
            claim = claims.get(active.claim_id)
            if claim is None:
                raise ClaimsError(f"file={path}:{line_number} unknown claim id={active.claim_id}")
            if surface not in claim["public_surfaces"]:
                raise ClaimsError(
                    f"file={path}:{line_number} claim id={active.claim_id} does not allow surface={surface}"
                )
            if not tier_is_permitted(active, claim):
                raise ClaimsError(
                    f"file={path}:{line_number} claim id={active.claim_id} registry_state={claim['state']} "
                    f"offending_wording={active.wording}"
                )
            continue
        r4_match = R4_CONTEXT_ANNOTATION_RE.search(line)
        if r4_match:
            active_r4 = (r4_match["ledger"], line_number)
            continue
        if not consumed_r4 and context_amount_exceeds_default(line):
            validate_r4_context_claim(
                path=path,
                line_number=line_number,
                active=active,
                r4_ledger=None,
                claims=claims,
                eligible_ledgers=eligible_r4_ledgers,
            )
        if not is_claim_bearing(line):
            continue
        claim_lines += 1
        if active is None:
            raise ClaimsError(f"file={path}:{line_number} missing claim annotation for numeric/superlative wording")
        claim = claims[active.claim_id]
        if surface not in claim["public_surfaces"]:
            raise ClaimsError(
                f"file={path}:{line_number} claim id={active.claim_id} does not allow surface={surface}"
            )
        if not tier_is_permitted(active, claim):
            raise ClaimsError(
                f"file={path}:{line_number} claim id={active.claim_id} registry_state={claim['state']} "
                f"offending_wording={active.wording}"
            )
    if active_r4 is not None:
        ledger, annotation_line = active_r4
        raise ClaimsError(
            f"file={path}:{annotation_line} unused fnlp-r4-context ledger={ledger}; it must immediately precede "
            "one >8K context claim"
        )
    log(
        f"SCAN file={path} surface={surface} annotations={annotations} claim_lines={claim_lines} "
        f"r4_context_lines={r4_context_lines}"
    )
    return annotations, claim_lines


def existing_public_surfaces(root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    fixed = {
        "README": (root / "README.md",),
        "--help": (root / "src" / "cli.rs",),
        "robot_schema_descriptions": (root / "src" / "robot.rs",),
        "release_notes": (root / "CHANGELOG.md",),
    }
    for surface, paths in fixed.items():
        result.extend((surface, path) for path in paths if path.is_file())
    for path in sorted((root / ".github").glob("release*.md")) if (root / ".github").is_dir() else []:
        result.append(("release_notes", path))
    for path in sorted((root / "docs").glob("release*.md")) if (root / "docs").is_dir() else []:
        result.append(("release_notes", path))
    return result


def validate_links(root: Path) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "[docs/CLAIMS.json](docs/CLAIMS.json)" not in readme:
        raise ClaimsError("README must link to docs/CLAIMS.json with the canonical relative link")
    annotation_doc = (root / "docs" / "CLAIMS_ANNOTATIONS.md").read_text(encoding="utf-8")
    if "[CLAIMS.json](CLAIMS.json)" not in annotation_doc:
        raise ClaimsError("CLAIMS_ANNOTATIONS.md must link to CLAIMS.json")


def validate_deferred_p7_scope(root: Path) -> None:
    """Keep the deferred Metal/serve/translation boundary mechanically visible.

    This is deliberately a scope check rather than a parity claim: while the
    P7 decision remains deferred, the release manifest must not acquire the
    Metal surface merely because its future profile name appears in documents.
    A later ratified promotion changes this validator together with the
    decision record and its named L-ladder evidence.
    """

    register_path = root / "docs" / "RESEARCH_DECISION_REGISTER.md"
    scope_path = root / "docs" / "adr" / "drafts" / "p7-metal-prefill-scope.md"
    manifest_path = root / "Cargo.toml"
    try:
        register = register_path.read_text(encoding="utf-8")
        scope = scope_path.read_text(encoding="utf-8")
        manifest = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ClaimsError(f"P7 scope input unavailable: {error}") from error

    for decision_id, row in P7_REGISTER_ROWS.items():
        if row not in register:
            raise ClaimsError(f"P7 decision register missing deferred row id={decision_id}")
    normalized_scope = " ".join(scope.split())
    for requirement in P7_SCOPE_REQUIREMENTS:
        if " ".join(requirement.split()) not in normalized_scope:
            raise ClaimsError(f"P7 scope gate missing requirement={requirement!r}")
    if "ft-kernel-metal" in manifest:
        raise ClaimsError("P7 scope violation: deferred ft-kernel-metal entered Cargo.toml")
    log("P7_SCOPE RESULT=PASS decisions=3 metal_release_dependency=absent")


def check_fixture(path: Path, claims: dict[str, dict[str, Any]], expected_fragment: str | None) -> None:
    try:
        scan_surface(path, "README", claims)
    except ClaimsError as error:
        if expected_fragment is None:
            raise ClaimsError(f"fixture unexpectedly rejected file={path}: {error}") from error
        if expected_fragment not in str(error):
            raise ClaimsError(
                f"fixture rejected with wrong detail file={path} expected={expected_fragment!r} observed={error}"
            ) from error
        log(f"FIXTURE file={path} expected_rejection={expected_fragment}")
        return
    if expected_fragment is not None:
        raise ClaimsError(f"fixture unexpectedly passed file={path} expected_rejection={expected_fragment}")
    log(f"FIXTURE file={path} status=PASS")


def run_self_test(root: Path, claims: dict[str, dict[str, Any]]) -> None:
    fixtures = root / "tests" / "fixtures" / "claims"
    check_fixture(fixtures / "compliant.md", claims, None)
    check_fixture(fixtures / "stronger_than_state.md", claims, "registry_state=targeted")
    check_fixture(fixtures / "missing_id_numeric.md", claims, "missing claim annotation")
    check_fixture(fixtures / "superlative_without_id.md", claims, "missing claim annotation")

    baseline = {"claims": list(claims.values()), "schema_version": SCHEMA_VERSION}

    def assert_rejected(name: str, expected: str, mutate: Any) -> None:
        mutated = copy.deepcopy(baseline)
        mutate(mutated)
        try:
            validate_registry(mutated)
        except ClaimsError as error:
            if expected not in str(error):
                raise ClaimsError(
                    f"self-test {name} rejected with wrong detail expected={expected!r} observed={error}"
                ) from error
            return
        raise ClaimsError(f"self-test {name} was accepted")

    assert_rejected(
        "duplicate-id",
        "duplicate claim id",
        lambda mutated: mutated["claims"].append(copy.deepcopy(mutated["claims"][0])),
    )
    assert_rejected(
        "evidenced-without-digest",
        "requires an evidence digest",
        lambda mutated: mutated["claims"][0].update({"state": "evidenced"}),
    )
    assert_rejected(
        "id-grammar",
        "id grammar violation",
        lambda mutated: mutated["claims"][0].update({"id": "INVALID_ID"}),
    )
    assert_rejected(
        "expiry-grammar",
        "expiry trigger grammar violation",
        lambda mutated: mutated["claims"][0].update({"expiry_revalidation_trigger": "eventually"}),
    )
    assert_rejected(
        "transition-regression",
        "transition is not allowed",
        lambda mutated: mutated["claims"][0].update(
            {
                "state": "targeted",
                "transition_history": [
                    {
                        "at": "2026-07-31",
                        "from_state": "evidenced",
                        "reason": "invalid regression fixture",
                        "to_state": "targeted",
                    }
                ],
            }
        ),
    )

    r4_claim = copy.deepcopy(next(iter(claims.values())))
    r4_claim["id"] = "r4-fixture-claim"
    r4_claim["state"] = "evidenced"
    r4_claim["evidence_artifact_digests"] = ["a" * 64]
    r4_claim["validity_domain"] = {
        "dataset": "r4-fixture-dataset",
        "host": "fixture-host-v1",
        "numerics_profile": "hf-bf16-eager",
        "prompt_hash": "r4-fixture-prompt",
        "recipe_id": "r4-fixture-recipe",
        "thinking_mode": "r4-fixture-thinking",
    }
    r4_claims = {r4_claim["id"]: r4_claim}
    r4_annotation = Annotation(r4_claim["id"], "evidenced")
    validate_r4_context_claim(
        path=Path("in-memory.md"),
        line_number=1,
        active=r4_annotation,
        r4_ledger="PERF-R4-VALID-001",
        claims=r4_claims,
        eligible_ledgers={"PERF-R4-VALID-001": r4_claim["id"]},
    )
    try:
        validate_r4_context_claim(
            path=Path("in-memory.md"),
            line_number=1,
            active=r4_annotation,
            r4_ledger=None,
            claims=r4_claims,
            eligible_ledgers={"PERF-R4-VALID-001": r4_claim["id"]},
        )
    except ClaimsError as error:
        if "missing fnlp-r4-context" not in str(error):
            raise
    else:
        raise ClaimsError("self-test R4 context claim without ledger annotation was accepted")
    if not context_amount_exceeds_default("Measured context window reaches 16K tokens"):
        raise ClaimsError("self-test R4 context quantity above default was not recognized")
    if context_amount_exceeds_default("Observed model limit: 262,144 positions"):
        raise ClaimsError("self-test observed model limit was treated as a practicality claim")

    hostile_cases, _ = load_json(fixtures / "r4_context_hostile.json")
    for case in hostile_cases["cases"]:
        observed = context_amount_exceeds_default(case["claim"])
        if observed != case["is_practicality_claim"]:
            raise ClaimsError(
                f"self-test hostile R4 claim mismatch claim={case['claim']!r} expected="
                f"{case['is_practicality_claim']} observed={observed}"
            )
    fabricated = eligible_r4_ledger_entries(
        root,
        r4_claims,
        ledger_path=fixtures / "r4_fabricated_perf_ledger.md",
    )
    if fabricated:
        raise ClaimsError("self-test fabricated R4 ledger row was eligible without retained receipts")
    typed = eligible_r4_ledger_entries(
        root,
        r4_claims,
        ledger_path=fixtures / "r4_typed_ledger.md",
    )
    if typed != {"PERF-R4-TYPED-001": r4_claim["id"]}:
        raise ClaimsError("self-test typed R4 ledger row was not eligible")
    for fixture_name in ("r4_retained_garbage_ledger.md", "r4_duplicate_evidence_ledger.md"):
        hostile = eligible_r4_ledger_entries(
            root,
            r4_claims,
            ledger_path=fixtures / fixture_name,
        )
        if hostile:
            raise ClaimsError(f"self-test hostile R4 ledger row was eligible fixture={fixture_name}")
    try:
        validate_r4_context_claim(
            path=Path("in-memory.md"),
            line_number=1,
            active=r4_annotation,
            r4_ledger="PERF-R4-TYPED-001",
            claims=r4_claims,
            eligible_ledgers={"PERF-R4-TYPED-001": "another-claim"},
        )
    except ClaimsError as error:
        if "is bound to claim" not in str(error):
            raise
    else:
        raise ClaimsError("self-test R4 ledger claim binding mismatch was accepted")
    floating_claim = copy.deepcopy(claims["hf-bf16-eager-fidelity"])
    floating_claim["state"] = "evidenced"
    floating_claim["evidence_artifact_digests"] = ["a" * 64]
    try:
        scan_surface(
            fixtures / "r4_floating_annotation.md",
            "README",
            {floating_claim["id"]: floating_claim},
            {"PERF-R4-VALID-001": floating_claim["id"]},
        )
    except ClaimsError as error:
        if "must immediately precede" not in str(error):
            raise
    else:
        raise ClaimsError("self-test floating fnlp-r4-context annotation was accepted")

    with tempfile.TemporaryDirectory(prefix="fnlp-claims-") as temporary:
        malformed = Path(temporary) / "duplicate.json"
        malformed.write_text('{"schema_version":1,"schema_version":1,"claims":[]}', encoding="utf-8")
        try:
            load_json(malformed)
        except ClaimsError as error:
            if "duplicate JSON key" not in str(error):
                raise
        else:
            raise ClaimsError("self-test duplicate JSON key was accepted")
    log("SELF_TEST registry_mutations=PASS fixtures=PASS")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate registry, fixtures, links, and public surfaces")
    mode.add_argument("--self-test", action="store_true", help="run hermetic registry and fixture negatives")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    scanned = 0
    try:
        registry, raw = load_json(root / "docs" / "CLAIMS.json")
        claims = validate_registry(registry, raw)
        log(
            f"REGISTRY schema_version={registry['schema_version']} claims={len(claims)} "
            f"sha256={hashlib.sha256(raw).hexdigest()}"
        )
        run_self_test(root, claims)
        if arguments.check:
            validate_links(root)
            validate_deferred_p7_scope(root)
            r4_ledgers = eligible_r4_ledger_entries(root, claims)
            log(f"R4_CONTEXT_GATE eligible_ledgers={len(r4_ledgers)}")
            for surface, path in existing_public_surfaces(root):
                scan_surface(path, surface, claims, r4_ledgers)
                scanned += 1
    except ClaimsError as error:
        log(f"REJECT detail={error}")
        log(f"RESULT=FAIL files={scanned} claims=0")
        return 1
    log(f"RESULT=PASS files={scanned} claims={len(claims)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
