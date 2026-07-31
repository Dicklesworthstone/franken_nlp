#!/usr/bin/env python3
"""Validate the pinned Nanbeige4.2-3B conversion-source manifest.

The static table below is intentionally duplicated from the reviewed source
manifest contract.  A manifest that merely self-consistently sums is not
authority: each name, byte length, and SHA-256 must match this pinned table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn


REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
CLOSURE_TOTAL_BYTES = 8_360_887_509
LOGICAL_SAFETENSORS_PAYLOAD_BYTES = 8_339_601_408
SAFETENSORS_CONTAINER_HEADER_BYTES = 23_312
SHA256_HEX_LENGTH = 64
HASH_CHUNK_BYTES = 1024 * 1024

# The sequence is sorted byte-lexicographically by filename to make the
# manifest's array order deterministic as well as its object-key order.
EXPECTED_FILES: tuple[tuple[str, int, str], ...] = (
    (
        "added_tokens.json",
        174,
        "9e3b127a27647df2c353cc1e5500826f7cdbe8bd15a458e368bba8422e9719cf",
    ),
    (
        "config.json",
        1_019,
        "f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19",
    ),
    (
        "generation_config.json",
        187,
        "68c690ce23efb6caae30c006ff3c1efd826297ff1df4338c04f7ac6f685d8746",
    ),
    (
        "model-00001-of-00002.safetensors",
        4_973_547_960,
        "09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94",
    ),
    (
        "model-00002-of-00002.safetensors",
        3_366_076_760,
        "31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389",
    ),
    (
        "model.safetensors.index.json",
        16_519,
        "30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1",
    ),
    (
        "special_tokens_map.json",
        623,
        "b718fce2b7a8940ffeddc1e67f3b092cc0d13ac885c63a021528786f8c4cf6c0",
    ),
    (
        "tokenizer.json",
        18_450_979,
        "1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f",
    ),
    (
        "tokenizer.model",
        2_782_298,
        "fb41d04798b714520a9b075727b0226538b7330254299062742c50ec8374bc36",
    ),
    (
        "tokenizer_config.json",
        10_990,
        "3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518",
    ),
)

EXPECTED_ROOT_KEYS = {
    "closure_total_bytes",
    "files",
    "logical_safetensors_payload_bytes",
    "model",
    "observed_state",
    "revision",
    "safetensors_container_header_bytes",
    "schema_version",
}
EXPECTED_FILE_KEYS = {"bytes", "name", "sha256"}


class ManifestError(ValueError):
    """A manifest violates the immutable conversion-source contract."""


def log(message: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{timestamp} {message}", file=sys.stderr)


def fail(message: str) -> NoReturn:
    raise ManifestError(message)


def duplicate_key_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            fail(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        fail(f"cannot read manifest {path}: {error}")

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"manifest is not UTF-8: {error}")

    try:
        decoded_value = json.loads(
            decoded,
            object_pairs_hook=duplicate_key_rejecting_object,
            parse_constant=lambda token: fail(f"non-finite JSON constant: {token}"),
        )
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: line={error.lineno} column={error.colno}: {error.msg}")

    if not isinstance(decoded_value, dict):
        fail("manifest root must be a JSON object")

    canonical = json.dumps(
        decoded_value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if raw != canonical:
        fail("manifest is not canonical JSON (sorted keys, compact separators, final newline)")
    return decoded_value


def required_int(container: dict[str, object], key: str) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{key} must be an integer, observed {value!r}")
    return value


def required_string(container: dict[str, object], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str):
        fail(f"{key} must be a string, observed {value!r}")
    return value


def validate_manifest(manifest: dict[str, object]) -> list[dict[str, object]]:
    root_keys = set(manifest)
    if root_keys != EXPECTED_ROOT_KEYS:
        fail(
            "manifest root keys differ: "
            f"missing={sorted(EXPECTED_ROOT_KEYS - root_keys)!r} "
            f"extra={sorted(root_keys - EXPECTED_ROOT_KEYS)!r}"
        )

    if required_int(manifest, "schema_version") != 1:
        fail(f"schema_version must be 1, observed {manifest['schema_version']!r}")
    if required_string(manifest, "model") != "Nanbeige4.2-3B":
        fail(f"model must be 'Nanbeige4.2-3B', observed {manifest['model']!r}")
    if required_string(manifest, "revision") != REVISION:
        fail(f"revision mismatch: expected {REVISION}, observed {manifest['revision']!r}")
    if required_string(manifest, "observed_state") != "OBSERVED@pin":
        fail(
            "observed_state must remain 'OBSERVED@pin' until replay promotion, "
            f"observed {manifest['observed_state']!r}"
        )

    entries_value = manifest["files"]
    if not isinstance(entries_value, list):
        fail("files must be a JSON array")
    if len(entries_value) != len(EXPECTED_FILES):
        fail(f"expected exactly {len(EXPECTED_FILES)} file entries, observed {len(entries_value)}")

    entries: list[dict[str, object]] = []
    for index, entry_value in enumerate(entries_value):
        if not isinstance(entry_value, dict):
            fail(f"files[{index}] must be an object")
        entry_keys = set(entry_value)
        if entry_keys != EXPECTED_FILE_KEYS:
            fail(
                f"files[{index}] keys differ: "
                f"missing={sorted(EXPECTED_FILE_KEYS - entry_keys)!r} "
                f"extra={sorted(entry_keys - EXPECTED_FILE_KEYS)!r}"
            )
        entries.append(entry_value)

    observed_names = [required_string(entry, "name") for entry in entries]
    duplicate_names = sorted({name for name in observed_names if observed_names.count(name) > 1})
    if duplicate_names:
        fail(f"duplicate file name(s): {duplicate_names!r}")

    expected_names = [name for name, _, _ in EXPECTED_FILES]
    if observed_names != expected_names:
        fail(f"file names/order mismatch: expected {expected_names!r}, observed {observed_names!r}")

    expected_by_name = {name: (byte_count, digest) for name, byte_count, digest in EXPECTED_FILES}
    for entry in entries:
        name = required_string(entry, "name")
        observed_bytes = required_int(entry, "bytes")
        observed_digest = required_string(entry, "sha256")
        expected_bytes, expected_digest = expected_by_name[name]
        if observed_bytes != expected_bytes:
            fail(
                f"{name} byte length mismatch: expected {expected_bytes}, observed {observed_bytes}"
            )
        if len(observed_digest) != SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in observed_digest
        ):
            fail(f"{name} sha256 must be {SHA256_HEX_LENGTH} lowercase hex characters")
        if observed_digest != expected_digest:
            fail(
                f"{name} sha256 mismatch: expected {expected_digest}, observed {observed_digest}"
            )

    total_bytes = sum(required_int(entry, "bytes") for entry in entries)
    if total_bytes != CLOSURE_TOTAL_BYTES:
        fail(
            "closure byte total mismatch: "
            f"expected {CLOSURE_TOTAL_BYTES}, observed {total_bytes}"
        )
    if required_int(manifest, "closure_total_bytes") != CLOSURE_TOTAL_BYTES:
        fail(
            "closure_total_bytes field mismatch: "
            f"expected {CLOSURE_TOTAL_BYTES}, observed {manifest['closure_total_bytes']!r}"
        )

    shard_bytes = sum(
        required_int(entry, "bytes")
        for entry in entries
        if required_string(entry, "name").endswith(".safetensors")
    )
    if shard_bytes != LOGICAL_SAFETENSORS_PAYLOAD_BYTES + SAFETENSORS_CONTAINER_HEADER_BYTES:
        fail(
            "safetensors shard accounting mismatch: "
            f"expected {LOGICAL_SAFETENSORS_PAYLOAD_BYTES} + "
            f"{SAFETENSORS_CONTAINER_HEADER_BYTES}, observed {shard_bytes}"
        )
    if (
        required_int(manifest, "logical_safetensors_payload_bytes")
        != LOGICAL_SAFETENSORS_PAYLOAD_BYTES
    ):
        fail("logical_safetensors_payload_bytes does not match the pinned payload total")
    if (
        required_int(manifest, "safetensors_container_header_bytes")
        != SAFETENSORS_CONTAINER_HEADER_BYTES
    ):
        fail("safetensors_container_header_bytes does not match the pinned header total")

    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_directory(entries: list[dict[str, object]], source_dir: Path) -> None:
    if not source_dir.is_dir():
        fail(f"source directory is unavailable: {source_dir}")

    for entry in entries:
        name = required_string(entry, "name")
        expected_bytes = required_int(entry, "bytes")
        expected_digest = required_string(entry, "sha256")
        candidate = source_dir / name
        if candidate.is_symlink() or not candidate.is_file():
            fail(f"{name} must be a regular non-symlink file under {source_dir}")
        observed_bytes = candidate.stat().st_size
        observed_digest = sha256_file(candidate)
        log(
            f"file={name} expected_bytes={expected_bytes} observed_bytes={observed_bytes} "
            f"expected_sha256={expected_digest} observed_sha256={observed_digest}"
        )
        if observed_bytes != expected_bytes:
            fail(f"{name} byte length mismatch: expected {expected_bytes}, observed {observed_bytes}")
        if observed_digest != expected_digest:
            fail(
                f"{name} sha256 mismatch: expected {expected_digest}, observed {observed_digest}"
            )


def expected_manifest() -> dict[str, object]:
    return {
        "closure_total_bytes": CLOSURE_TOTAL_BYTES,
        "files": [
            {"bytes": byte_count, "name": name, "sha256": digest}
            for name, byte_count, digest in EXPECTED_FILES
        ],
        "logical_safetensors_payload_bytes": LOGICAL_SAFETENSORS_PAYLOAD_BYTES,
        "model": "Nanbeige4.2-3B",
        "observed_state": "OBSERVED@pin",
        "revision": REVISION,
        "safetensors_container_header_bytes": SAFETENSORS_CONTAINER_HEADER_BYTES,
        "schema_version": 1,
    }


def require_rejected(name: str, manifest: dict[str, object], expected_fragment: str) -> None:
    try:
        validate_manifest(manifest)
    except ManifestError as error:
        if expected_fragment not in str(error):
            fail(f"self-test {name} failed with unexpected error: {error}")
        log(f"self_test={name} RESULT=PASS rejection={error}")
        return
    fail(f"self-test {name} unexpectedly accepted an invalid manifest")


def run_self_tests() -> None:
    eleven_files = expected_manifest()
    eleven_files["files"] = list(eleven_files["files"]) + [
        {"bytes": 0, "name": "unexpected.json", "sha256": "0" * SHA256_HEX_LENGTH}
    ]
    require_rejected("eleven-file", eleven_files, "expected exactly 10")

    wrong_digest = expected_manifest()
    wrong_digest_entries = list(wrong_digest["files"])
    wrong_digest_entries[0] = {**wrong_digest_entries[0], "sha256": "0" * SHA256_HEX_LENGTH}
    wrong_digest["files"] = wrong_digest_entries
    require_rejected("wrong-digest", wrong_digest, "sha256 mismatch")

    wrong_length = expected_manifest()
    wrong_length_entries = list(wrong_length["files"])
    wrong_length_entries[0] = {**wrong_length_entries[0], "bytes": 175}
    wrong_length["files"] = wrong_length_entries
    require_rejected("wrong-length", wrong_length, "byte length mismatch")

    duplicate_name = expected_manifest()
    duplicate_name_entries = list(duplicate_name["files"])
    duplicate_name_entries[1] = {**duplicate_name_entries[1], "name": "added_tokens.json"}
    duplicate_name["files"] = duplicate_name_entries
    require_rejected("duplicate-name", duplicate_name, "duplicate file name")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "docs/truth-pack/nanbeige4.2-3b.source.json",
        help="canonical source manifest to validate",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="rehash a downloaded conversion-source closure after static validation",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise the named invalid-manifest cases without model weights",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = read_json(args.manifest)
        entries = validate_manifest(manifest)
        for entry in entries:
            log(
                f"file={entry['name']} expected_bytes={entry['bytes']} "
                f"expected_sha256={entry['sha256']} observed=manifest-contract"
            )
        if args.self_test:
            run_self_tests()
        if args.source_dir is not None:
            validate_source_directory(entries, args.source_dir)
    except ManifestError as error:
        log(f"SOURCE_MANIFEST RESULT=FAIL detail={error}")
        return 1

    log(f"SOURCE_MANIFEST RESULT=PASS files={len(EXPECTED_FILES)}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
