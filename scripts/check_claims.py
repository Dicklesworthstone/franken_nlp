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
from typing import Any


SCHEMA_VERSION = 1
STATES = ("targeted", "observed", "evidenced", "withdrawn")
STATE_RANK = {state: index for index, state in enumerate(STATES[:3])}
CLAIM_ID_RE = re.compile(r"[a-z][a-z0-9-]{1,79}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPIRY_RE = re.compile(r"(?P<kind>on_change|on_date|before_release): (?P<detail>.+)\Z")
ANNOTATION_RE = re.compile(
    r"fnlp-claim:\s*(?P<claim_id>[a-z][a-z0-9-]{1,79})\s*;\s*wording=(?P<wording>targeted|observed|evidenced|withdrawn)\b"
)
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


def scan_surface(path: Path, surface: str, claims: dict[str, dict[str, Any]]) -> tuple[int, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ClaimsError(f"public surface is not UTF-8 file={path}: {error}") from error
    active: Annotation | None = None
    annotations = 0
    claim_lines = 0
    for line_number, line in enumerate(lines, start=1):
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
    log(f"SCAN file={path} surface={surface} annotations={annotations} claim_lines={claim_lines}")
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
            for surface, path in existing_public_surfaces(root):
                scan_surface(path, surface, claims)
                scanned += 1
    except ClaimsError as error:
        log(f"REJECT detail={error}")
        log(f"RESULT=FAIL files={scanned} claims=0")
        return 1
    log(f"RESULT=PASS files={scanned} claims={len(claims)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
