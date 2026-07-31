#!/usr/bin/env python3
"""Replay the Nanbeige research truth-pack manifest without making it a product input.

The manifest describes research evidence only.  This validator neither downloads
sources nor invokes ``fnlp convert``; it only checks the immutable archived copies
and, when explicitly supplied, a local checkout of the pinned upstream repository.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn


PINNED_MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
PINNED_REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
MANIFEST_NAME = "nanbeige4.2-3b.research.json"
CONVERSION_MANIFEST_NAME = "nanbeige4.2-3b.source.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_RELATIVE_PATH_RE = re.compile(r"^[^/\\\x00]+(?:/[^/\\\x00]+)*$")
LEGAL_BASENAMES = frozenset(
    {"license", "license.md", "license.txt", "notice", "notice.md", "notice.txt"}
)
REQUIRED_ARCHIVE_KINDS = frozenset(
    {
        "modeling_nanbeige_py",
        "configuration_nanbeige_py",
        "model_card",
        "hf_api_metadata",
        "technical_report",
        "evaluation_metadata",
    }
)


class ManifestError(ValueError):
    """A named rejection that becomes one deterministic validator mismatch."""


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{timestamp()} RESEARCH_MANIFEST {message}", file=sys.stderr)


def fail(message: str) -> NoReturn:
    raise ManifestError(message)


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> NoReturn:
    fail(f"non-finite JSON number: {value}")


def canonical_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        fail(f"cannot read manifest {path}: {error}")
    if payload.startswith(b"\xef\xbb\xbf"):
        fail("manifest must be UTF-8 without a byte-order mark")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"manifest is not UTF-8: {error}")
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=no_duplicate_object,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        fail(f"manifest JSON parse error line={error.lineno} column={error.colno}: {error.msg}")
    if not isinstance(value, dict):
        fail("manifest root must be a JSON object")
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if decoded != expected:
        fail("manifest is not canonical JSON (sorted UTF-8 keys, indent=2, trailing newline required)")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        fail(f"cannot hash {path}: {error}")
    return digest.hexdigest()


def required_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        fail(f"{context}.{key} must be a non-empty string")
    return value


def required_length(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{context}.{key} must be a non-negative integer")
    return value


def required_digest(mapping: dict[str, Any], key: str, context: str) -> str:
    value = required_string(mapping, key, context)
    if not SHA256_RE.fullmatch(value):
        fail(f"{context}.{key} must be a lowercase SHA-256 hex digest")
    return value


def relative_path(value: str, context: str) -> Path:
    if not SAFE_RELATIVE_PATH_RE.fullmatch(value):
        fail(f"{context} must be a non-empty portable relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        fail(f"{context} must not escape its root")
    return path


def under_root(root: Path, relative: Path, context: str) -> Path:
    candidate = root / relative
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        fail(f"{context} escapes archive root")
    return candidate


def expect_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    return value


def expect_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{context} must be an array")
    return value


def validate_top_level(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        fail("schema_version must equal 1")
    if manifest.get("kind") != "nanbeige4.2-3b-research-truth-pack":
        fail("kind must equal nanbeige4.2-3b-research-truth-pack")
    model = expect_mapping(manifest.get("model"), "model")
    if model.get("id") != PINNED_MODEL_ID:
        fail(f"model.id must equal {PINNED_MODEL_ID}")
    if model.get("revision") != PINNED_REVISION:
        fail(f"model.revision must equal pinned revision {PINNED_REVISION}")
    boundary = expect_mapping(manifest.get("role_boundary"), "role_boundary")
    if boundary.get("research_only") is not True:
        fail("role_boundary.research_only must be true")
    if boundary.get("converter_reads_manifest") is not False:
        fail("role_boundary.converter_reads_manifest must be false")
    if boundary.get("runtime_reads_manifest") is not False:
        fail("role_boundary.runtime_reads_manifest must be false")


def validate_archived_files(manifest: dict[str, Any], archive_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    entries = expect_list(manifest.get("archived_files"), "archived_files")
    if not entries:
        fail("archived_files must not be empty")
    typed_entries: list[dict[str, Any]] = []
    kinds: set[str] = set()
    archive_paths: list[str] = []
    source_paths: set[str] = set()
    mismatches: list[str] = []
    for index, raw_entry in enumerate(entries):
        context = f"archived_files[{index}]"
        entry = expect_mapping(raw_entry, context)
        kind = required_string(entry, "kind", context)
        archive_rel = relative_path(required_string(entry, "archive_path", context), f"{context}.archive_path")
        source_path = relative_path(required_string(entry, "source_path", context), f"{context}.source_path")
        expected_length = required_length(entry, "length", context)
        expected_digest = required_digest(entry, "sha256", context)
        if kind in kinds:
            fail(f"duplicate archived_files kind: {kind}")
        kinds.add(kind)
        archive_text = archive_rel.as_posix()
        source_text = source_path.as_posix()
        if source_text in source_paths:
            fail(f"duplicate archived_files source_path: {source_text}")
        source_paths.add(source_text)
        archive_paths.append(archive_text)
        path = under_root(archive_root, archive_rel, context)
        observed_length = -1
        observed_digest = "MISSING"
        if path.is_symlink():
            mismatches.append(f"archive_symlink:{archive_text}")
        elif not path.is_file():
            mismatches.append(f"archive_missing:{archive_text}")
        else:
            observed_length = path.stat().st_size
            observed_digest = sha256_file(path)
            if observed_length != expected_length:
                mismatches.append(f"length:{archive_text}")
            if not hmac.compare_digest(observed_digest, expected_digest):
                mismatches.append(f"digest:{archive_text}")
        log(
            "file="
            f"{archive_text} source={source_path.as_posix()} "
            f"bytes_expected={expected_length} bytes_observed={observed_length} "
            f"sha256_expected={expected_digest} sha256_observed={observed_digest}"
        )
        typed_entries.append(entry)
    if archive_paths != sorted(archive_paths):
        fail("archived_files must be sorted by archive_path")
    missing_kinds = sorted(REQUIRED_ARCHIVE_KINDS - kinds)
    if missing_kinds:
        fail(f"archived_files missing required evidence kind: {missing_kinds[0]}")
    if not any(
        entry["kind"] == "modeling_nanbeige_py"
        and entry["source_path"].endswith("modeling_nanbeige.py")
        for entry in typed_entries
    ):
        fail("modeling_nanbeige_py entry must archive modeling_nanbeige.py")
    if not any(
        entry["kind"] == "configuration_nanbeige_py"
        and entry["source_path"].endswith("configuration_nanbeige.py")
        for entry in typed_entries
    ):
        fail("configuration_nanbeige_py entry must archive configuration_nanbeige.py")
    if not any(
        entry["kind"] == "technical_report"
        and entry["source_path"].endswith("Nanbeige42_report.pdf")
        for entry in typed_entries
    ):
        fail("technical_report entry must archive Nanbeige42_report.pdf")
    if not any(
        entry["kind"] == "evaluation_metadata"
        and entry["source_path"].startswith(".eval_results/")
        for entry in typed_entries
    ):
        fail("evaluation_metadata entry must originate below .eval_results/")
    return typed_entries, mismatches


def validate_repository_census(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    census = expect_mapping(manifest.get("repository_census"), "repository_census")
    if census.get("revision") != PINNED_REVISION:
        fail("repository_census.revision must equal the pinned revision")
    for null_field in ("upstream_license_file", "upstream_notice_file"):
        if null_field not in census:
            fail(f"repository_census missing required null field: {null_field}")
        if census[null_field] is not None:
            fail(f"repository_census.{null_field} must be null")
    entries = expect_list(census.get("entries"), "repository_census.entries")
    declared_count = required_length(census, "file_count", "repository_census")
    if declared_count != len(entries):
        fail(
            "repository_census.file_count does not equal entries length "
            f"expected={declared_count} observed={len(entries)}"
        )
    paths: list[str] = []
    typed_entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        context = f"repository_census.entries[{index}]"
        entry = expect_mapping(raw_entry, context)
        path = relative_path(required_string(entry, "path", context), f"{context}.path")
        required_length(entry, "length", context)
        required_digest(entry, "sha256", context)
        basename = path.name.lower()
        if basename in LEGAL_BASENAMES:
            fail(
                "repository census contradicts null LICENSE/NOTICE fact: "
                f"found {path.as_posix()}"
            )
        paths.append(path.as_posix())
        typed_entries.append(entry)
    if not typed_entries:
        fail("repository_census.entries must not be empty")
    if paths != sorted(paths):
        fail("repository_census.entries must be sorted by path")
    if len(paths) != len(set(paths)):
        duplicate = next(path for path in paths if paths.count(path) > 1)
        fail(f"repository_census has duplicate path: {duplicate}")
    return typed_entries, []


def validate_source_replay(census_entries: list[dict[str, Any]], source_repo: Path) -> list[str]:
    if not source_repo.is_dir():
        fail(f"source repository does not exist: {source_repo}")
    expected = {entry["path"]: entry for entry in census_entries}
    observed_paths: dict[str, Path] = {}
    for path in source_repo.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source_repo).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        observed_paths[relative] = path
    mismatches: list[str] = []
    for relative in sorted(set(expected) | set(observed_paths)):
        if relative not in expected:
            mismatches.append(f"census_extra:{relative}")
            log(f"census path={relative} verdict=EXTRA")
            continue
        if relative not in observed_paths:
            mismatches.append(f"census_missing:{relative}")
            log(f"census path={relative} verdict=MISSING")
            continue
        entry = expected[relative]
        path = observed_paths[relative]
        observed_length = path.stat().st_size
        observed_digest = sha256_file(path)
        expected_length = entry["length"]
        expected_digest = entry["sha256"]
        verdict = "PASS"
        if observed_length != expected_length:
            mismatches.append(f"census_length:{relative}")
            verdict = "LENGTH_MISMATCH"
        if not hmac.compare_digest(observed_digest, expected_digest):
            mismatches.append(f"census_digest:{relative}")
            verdict = "DIGEST_MISMATCH"
        log(
            f"census path={relative} bytes_expected={expected_length} bytes_observed={observed_length} "
            f"sha256_expected={expected_digest} sha256_observed={observed_digest} verdict={verdict}"
        )
    return mismatches


def conversion_paths(manifest: dict[str, Any]) -> set[str]:
    for key in ("source_files", "files", "conversion_files"):
        value = manifest.get(key)
        if not isinstance(value, list):
            continue
        paths: set[str] = set()
        for index, raw_entry in enumerate(value):
            entry = expect_mapping(raw_entry, f"conversion_manifest.{key}[{index}]")
            raw_path = entry.get("path", entry.get("name"))
            if not isinstance(raw_path, str):
                fail(f"conversion_manifest.{key}[{index}] needs path or name")
            paths.add(relative_path(raw_path, f"conversion_manifest.{key}[{index}]").as_posix())
        return paths
    fail("conversion manifest has no recognized file-entry array")


def validate_separation(manifest: dict[str, Any], conversion_manifest: Path | None) -> list[str]:
    if conversion_manifest is None:
        log("separation verdict=SKIPPED_NO_CONVERSION_MANIFEST")
        return []
    other = canonical_json(conversion_manifest)
    other_model = expect_mapping(other.get("model"), "conversion_manifest.model")
    if other_model.get("revision") != PINNED_REVISION:
        fail("conversion manifest revision differs from research manifest revision")
    research_paths = {
        entry["source_path"]
        for entry in expect_list(manifest.get("archived_files"), "archived_files")
        if isinstance(entry, dict) and isinstance(entry.get("source_path"), str)
    }
    overlap = sorted(research_paths & conversion_paths(other))
    if overlap:
        return [f"role_overlap:{path}" for path in overlap]
    log("separation verdict=PASS shared_revision=true file_overlap=0")
    return []


def validate(manifest_path: Path, archive_root: Path, source_repo: Path | None, conversion_manifest: Path | None) -> tuple[int, list[str]]:
    manifest = canonical_json(manifest_path)
    validate_top_level(manifest)
    archived, mismatches = validate_archived_files(manifest, archive_root)
    census, census_mismatches = validate_repository_census(manifest)
    mismatches.extend(census_mismatches)
    if source_repo is not None:
        mismatches.extend(validate_source_replay(census, source_repo))
    mismatches.extend(validate_separation(manifest, conversion_manifest))
    return len(archived), sorted(set(mismatches))


def expect_self_test_failure(callback: Any, expected: str) -> None:
    """Assert that a deliberately malformed in-memory manifest is rejected."""
    try:
        callback()
    except ManifestError as error:
        if expected not in str(error):
            fail(
                "self-test rejection differed "
                f"expected_substring={expected!r} observed={str(error)!r}"
            )
        return
    fail(f"self-test expected rejection: {expected}")


def self_test() -> int:
    """Exercise fail-closed manifest invariants without creating mutable evidence files."""
    valid_census = {
        "entries": [
            {
                "length": 0,
                "path": "README.md",
                "sha256": "0" * 64,
            }
        ],
        "file_count": 1,
        "revision": PINNED_REVISION,
        "upstream_license_file": None,
        "upstream_notice_file": None,
    }
    validate_top_level(
        {
            "kind": "nanbeige4.2-3b-research-truth-pack",
            "model": {"id": PINNED_MODEL_ID, "revision": PINNED_REVISION},
            "role_boundary": {
                "converter_reads_manifest": False,
                "research_only": True,
                "runtime_reads_manifest": False,
            },
            "schema_version": 1,
        }
    )
    census, mismatches = validate_repository_census(valid_census)
    if len(census) != 1 or mismatches:
        fail("self-test valid census did not validate")
    expect_self_test_failure(
        lambda: validate_repository_census(
            {key: value for key, value in valid_census.items() if key != "upstream_license_file"}
        ),
        "repository_census missing required null field: upstream_license_file",
    )
    expect_self_test_failure(
        lambda: relative_path("../escape", "self-test path"),
        "self-test path must not escape its root",
    )
    expect_self_test_failure(
        lambda: required_digest({"sha256": "not-a-digest"}, "sha256", "self-test digest"),
        "self-test digest.sha256 must be a lowercase SHA-256 hex digest",
    )
    log("self_test verdict=PASS checks=4")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise in-memory validator invariants without reading or writing evidence",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "truth-pack" / MANIFEST_NAME,
        help="canonical research manifest",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "truth-pack",
        help="root that owns the immutable archived evidence copies",
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="optional pinned upstream checkout for complete census replay",
    )
    parser.add_argument(
        "--conversion-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "truth-pack"
        / CONVERSION_MANIFEST_NAME,
        help="conversion-source manifest for the required no-role-overlap check",
    )
    args = parser.parse_args()
    if args.self_test:
        try:
            files = self_test()
            mismatches: list[str] = []
        except ManifestError as error:
            files = 0
            mismatches = [str(error)]
        if mismatches:
            for mismatch in mismatches:
                log(f"FAIL {mismatch}")
            print(
                f"RESEARCH_MANIFEST RESULT=FAIL files={files} mismatches={','.join(mismatches)}",
                file=sys.stderr,
            )
            return 1
        print(
            f"RESEARCH_MANIFEST RESULT=PASS files={files} mismatches=",
            file=sys.stderr,
        )
        return 0
    try:
        files, mismatches = validate(
            args.manifest,
            args.archive_root,
            args.source_repo,
            args.conversion_manifest,
        )
    except ManifestError as error:
        files = 0
        mismatches = [str(error)]
    if mismatches:
        for mismatch in mismatches:
            log(f"FAIL {mismatch}")
        print(
            f"RESEARCH_MANIFEST RESULT=FAIL files={files} mismatches={','.join(mismatches)}",
            file=sys.stderr,
        )
        return 1
    print("RESEARCH_MANIFEST RESULT=PASS files=" f"{files} mismatches=", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
