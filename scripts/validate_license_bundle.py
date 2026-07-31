"""Validate the immutable Nanbeige Apache-2.0 provenance bundle."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PINNED_REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
CANONICAL_APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
EXPECTED_CARD_SHA256 = "01cbce423668845435003dc0828033ab7ad07e88f076ed96ec66d4b014f4654f"
EXPECTED_ATTRIBUTION = (
    "model: Nanbeige4.2-3B, revision "
    "f56ec5a9650268aa098496734743c25ea778bd2d, "
    "author attribution per the pinned card: Nanbeige Team\n"
)
CONTENT_FILES = (
    "APACHE-2.0.txt",
    "ATTRIBUTION.txt",
    "MODIFICATION_NOTICE_TEMPLATE.txt",
    "model_card_license_snapshot.json",
    "upstream_license_notice_absence.json",
)


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{stamp} LICENSE_BUNDLE {message}", file=sys.stderr)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return data


def bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for filename in sorted(CONTENT_FILES):
        payload = (root / filename).read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_bytes(payload).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate(root: Path) -> list[str]:
    failures: list[str] = []
    manifest_path = root / "bundle_manifest.json"
    required_paths = [root / filename for filename in (*CONTENT_FILES, "bundle_manifest.json")]
    for path in required_paths:
        if not path.is_file():
            failures.append(f"missing required bundle file: {path}")
    if failures:
        return failures

    manifest = load_json(manifest_path)
    if manifest.get("bundle_content_files") != list(CONTENT_FILES):
        failures.append("bundle_manifest.json content-file order differs from the canonical bundle")
    if manifest.get("canonical_apache_license_sha256") != CANONICAL_APACHE_SHA256:
        failures.append("bundle_manifest.json canonical Apache SHA-256 is incorrect")

    apache_digest = sha256_bytes((root / "APACHE-2.0.txt").read_bytes())
    log(f"apache file=APACHE-2.0.txt sha256={apache_digest}")
    if not hmac.compare_digest(apache_digest, CANONICAL_APACHE_SHA256):
        failures.append(
            "APACHE-2.0.txt does not byte-match the canonical Apache-2.0 text "
            f"(expected={CANONICAL_APACHE_SHA256} observed={apache_digest})"
        )

    attribution = (root / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    if attribution != EXPECTED_ATTRIBUTION:
        failures.append("ATTRIBUTION.txt differs from the required factual attribution string")

    card = load_json(root / "model_card_license_snapshot.json")
    if card.get("revision") != PINNED_REVISION:
        failures.append("model-card snapshot revision is not the pinned model revision")
    if card.get("spdx_license") != "apache-2.0":
        failures.append("model-card snapshot does not declare SPDX apache-2.0")
    if card.get("readme_sha256") != EXPECTED_CARD_SHA256:
        failures.append("model-card snapshot README SHA-256 does not match the archived observation")
    front_matter = card.get("front_matter_snapshot")
    if not isinstance(front_matter, str) or "license: apache-2.0\n" not in front_matter:
        failures.append("model-card front-matter snapshot lacks the exact Apache-2.0 declaration")

    absence = load_json(root / "upstream_license_notice_absence.json")
    observed_paths = absence.get("observed_paths")
    if not isinstance(observed_paths, list):
        failures.append("upstream absence record has no observed path inventory")
    else:
        upstream_legal_names = {Path(str(path)).name.lower() for path in observed_paths}
        if any(name in upstream_legal_names for name in ("license", "license.md", "license.txt", "notice", "notice.md", "notice.txt")):
            failures.append("upstream absence record contradicts itself: a LICENSE/NOTICE object is inventoried")
    if absence.get("research_manifest_cross_reference") != "../nanbeige4.2-3b.source.json":
        failures.append("upstream absence record lacks the required research-manifest cross-reference")

    upstream_copyright = re.compile(r"(?im)^.*copyright(?:\s*\(c\))?.*nanbeige")
    for filename in CONTENT_FILES:
        text = (root / filename).read_text(encoding="utf-8")
        if upstream_copyright.search(text):
            failures.append(f"{filename} synthesizes an upstream Nanbeige copyright line")
        digest = sha256_bytes(text.encode("utf-8"))
        log(f"inventory file={filename} bytes={len(text.encode('utf-8'))} sha256={digest}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "truth-pack" / "license",
        help="license-bundle directory",
    )
    args = parser.parse_args()
    try:
        failures = validate(args.root)
        digest = bundle_digest(args.root) if not failures else "unavailable"
    except (OSError, UnicodeDecodeError, ValueError) as error:
        failures = [str(error)]
        digest = "unavailable"
    if failures:
        for failure in failures:
            log(f"FAIL {failure}")
        log(f"RESULT=FAIL license_bundle_sha256={digest}")
        return 1
    log(f"RESULT=PASS license_bundle_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
