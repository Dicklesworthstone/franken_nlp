#!/usr/bin/env python3
"""Validate the evaluation registry and emit digest-bound scorecards.

This is truth-pack/evaluation tooling, never product inference code.  It keeps
dataset permission, split locking, scorecard identity, and deterministic metric
calculation in a small standard-library-only surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "docs" / "eval" / "DATASETS.md"
DEFAULT_SCHEMA = ROOT / "tests" / "fixtures" / "eval" / "scorecard.schema.json"
REQUIRED_DATASET_FIELDS = {
    "acquisition",
    "allowed_use",
    "contamination_risks",
    "dataset_digest_sha256",
    "dataset_id",
    "dataset_version",
    "fixture_path",
    "license_name",
    "license_text",
    "preprocessing",
    "preprocessing_digest_sha256",
    "redistribution",
    "source",
    "splits",
    "unit_of_analysis",
}
SPLITS = ("development", "calibration", "test")
SCORECARD_REQUIRED = {
    "dataset",
    "dataset_digest_sha256",
    "dataset_id",
    "dataset_version",
    "input_digest_sha256",
    "metric",
    "model_recipe_id",
    "prompt_hash",
    "schema_version",
    "split",
    "status",
    "thinking_mode",
}
Z_95 = 1.959963984540054
MIN_CI_SAMPLE = 5


class EvalError(RuntimeError):
    """A registry, acquisition, split, or scorecard contract violation."""


@dataclass(frozen=True)
class Dataset:
    record: dict[str, Any]

    @property
    def id(self) -> str:
        return self.record["dataset_id"]


def log(message: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{timestamp} EVAL_CORE {message}", file=sys.stderr)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvalError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(value: bytes, context: str) -> Any:
    try:
        return json.loads(value, object_pairs_hook=no_duplicate_pairs)
    except json.JSONDecodeError as error:
        raise EvalError(f"invalid JSON {context}: {error}") from error


def parse_registry(path: Path) -> list[Dataset]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvalError(f"cannot read registry {path}: {error}") from error
    blocks = re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL)
    if not blocks:
        raise EvalError(f"registry {path} contains no JSON dataset records")
    records: list[Dataset] = []
    for index, block in enumerate(blocks, start=1):
        value = parse_json(block.encode("utf-8"), f"registry block={index}")
        if not isinstance(value, dict):
            raise EvalError(f"registry block={index} must be a JSON object")
        records.append(Dataset(value))
    return records


def require_sha256(value: Any, field: str, dataset_id: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise EvalError(f"dataset={dataset_id} field={field} must be a lowercase SHA-256")


def validate_dataset(dataset: Dataset, *, verify_fixture: bool) -> None:
    record = dataset.record
    missing = sorted(REQUIRED_DATASET_FIELDS - set(record))
    if missing:
        raise EvalError(f"dataset={dataset.id if 'dataset_id' in record else '<unknown>'} missing fields={missing}")
    if not isinstance(dataset.id, str) or not dataset.id:
        raise EvalError("dataset_id must be a non-empty string")
    for field in ("dataset_digest_sha256", "preprocessing_digest_sha256"):
        require_sha256(record[field], field, dataset.id)
    for field in ("source", "license_name", "license_text", "allowed_use", "redistribution", "preprocessing", "unit_of_analysis"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise EvalError(f"dataset={dataset.id} field={field} must be non-empty text")
    if "#sha256=" not in record["source"]:
        raise EvalError(f"dataset={dataset.id} source must contain immutable SHA-256 fragment")
    if not isinstance(record["contamination_risks"], list) or not record["contamination_risks"]:
        raise EvalError(f"dataset={dataset.id} contamination_risks must be a non-empty list")
    acquisition = record["acquisition"]
    if not isinstance(acquisition, dict) or not isinstance(acquisition.get("automated_access_allowed"), bool):
        raise EvalError(f"dataset={dataset.id} acquisition must state automated_access_allowed")
    splits = record["splits"]
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise EvalError(f"dataset={dataset.id} splits must contain exactly {list(SPLITS)}")
    seen: dict[str, str] = {}
    for split in SPLITS:
        ids = splits[split]
        if not isinstance(ids, list) or not ids or not all(isinstance(item, str) and item for item in ids):
            raise EvalError(f"dataset={dataset.id} split={split} must be a non-empty string-id list")
        for item_id in ids:
            prior = seen.get(item_id)
            if prior is not None:
                raise EvalError(f"dataset={dataset.id} split-overlap id={item_id} first={prior} second={split}")
            seen[item_id] = split
    if verify_fixture:
        fixture = ROOT / record["fixture_path"]
        try:
            raw = fixture.read_bytes()
        except OSError as error:
            raise EvalError(f"dataset={dataset.id} fixture missing path={fixture}") from error
        observed = sha256_bytes(raw)
        if observed != record["dataset_digest_sha256"]:
            raise EvalError(
                f"dataset={dataset.id} digest mismatch expected={record['dataset_digest_sha256']} observed={observed}"
            )
        rows = load_ndjson(fixture)
        fixture_ids = {row.get("id") for row in rows if isinstance(row.get("id"), str)}
        if fixture_ids != set(seen):
            raise EvalError(
                f"dataset={dataset.id} fixture/split ids differ missing={sorted(set(seen) - fixture_ids)} extra={sorted(fixture_ids - set(seen))}"
            )
        for row in rows:
            if row.get("split") != seen.get(row.get("id")):
                raise EvalError(f"dataset={dataset.id} fixture split mismatch id={row.get('id')}")
    sizes = ",".join(f"{split}={len(splits[split])}" for split in SPLITS)
    log(f"DATASET id={dataset.id} version={record['dataset_version']} digest={record['dataset_digest_sha256']} splits={sizes}")


def validate_registry(path: Path, *, verify_fixture: bool = True) -> dict[str, Dataset]:
    datasets = parse_registry(path)
    by_id: dict[str, Dataset] = {}
    for dataset in datasets:
        if dataset.id in by_id:
            raise EvalError(f"duplicate dataset_id={dataset.id}")
        validate_dataset(dataset, verify_fixture=verify_fixture)
        by_id[dataset.id] = dataset
    return by_id


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvalError(f"cannot read NDJSON {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = parse_json(line.encode("utf-8"), f"path={path} line={number}")
        if not isinstance(value, dict):
            raise EvalError(f"NDJSON object required path={path} line={number}")
        rows.append(value)
    return rows


def verify_user_file(dataset: Dataset, path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvalError(f"dataset={dataset.id} user-supplied file unavailable path={path}") from error
    observed = sha256_bytes(raw)
    expected = dataset.record["dataset_digest_sha256"]
    if observed != expected:
        raise EvalError(f"dataset={dataset.id} typed refusal digest expected={expected} observed={observed}")
    log(f"ACQUIRE RESULT=PASS dataset={dataset.id} path={path} digest={observed}")
    return observed


def acquisition_path(dataset: Dataset, user_file: Path | None) -> Path:
    """Select an authorized acquisition path without treating a hash as consent."""
    if user_file is not None:
        return user_file
    acquisition = dataset.record["acquisition"]
    if not acquisition["automated_access_allowed"]:
        raise EvalError(f"dataset={dataset.id} user-supplied file required: automated access is not permitted")
    if acquisition.get("mode") != "repo-authored-fixture":
        raise EvalError(
            f"dataset={dataset.id} automated acquisition is permitted but no downloader is implemented for mode={acquisition.get('mode')!r}"
        )
    return ROOT / dataset.record["fixture_path"]


def wilson_95(correct: int, total: int) -> tuple[float, float]:
    proportion = correct / total
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    radius = (Z_95 / denominator) * math.sqrt((proportion * (1 - proportion) + z2 / (4 * total)) / total)
    return center - radius, center + radius


def scorecard(
    dataset: Dataset,
    split: str,
    rows: Iterable[dict[str, Any]],
    *,
    input_digest: str,
    recipe_id: str,
    prompt_hash: str,
    thinking_mode: str,
    model_gated_skip: bool,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise EvalError(f"dataset={dataset.id} unknown split={split}")
    if not recipe_id or not thinking_mode:
        raise EvalError("scorecard recipe_id and thinking_mode must be non-empty")
    require_sha256(prompt_hash, "prompt_hash", dataset.id)
    expected_ids = set(dataset.record["splits"][split])
    excluded: list[dict[str, str]] = []
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str):
            raise EvalError("score input row missing string id")
        if row_id not in expected_ids:
            excluded.append({"id": row_id, "reason": f"not-in-locked-{split}-split"})
            log(f"EXCLUDED id={row_id} reason=not-in-locked-{split}-split")
            continue
        if row_id in selected:
            raise EvalError(f"duplicate score input id={row_id}")
        prediction, target = row.get("prediction"), row.get("target")
        if not isinstance(prediction, str) or not isinstance(target, str):
            raise EvalError(f"score input id={row_id} requires string prediction and target")
        selected[row_id] = row
    missing = sorted(expected_ids - set(selected))
    if missing and not model_gated_skip:
        raise EvalError(f"locked split incomplete split={split} missing_ids={missing}")
    common = {
        "dataset": {"id": dataset.id, "version": dataset.record["dataset_version"]},
        "dataset_digest_sha256": dataset.record["dataset_digest_sha256"],
        "dataset_id": dataset.id,
        "dataset_version": dataset.record["dataset_version"],
        "excluded_rows": excluded,
        "input_digest_sha256": input_digest,
        "model_recipe_id": recipe_id,
        "prompt_hash": prompt_hash,
        "schema_version": 1,
        "split": split,
        "thinking_mode": thinking_mode,
    }
    if model_gated_skip:
        return {**common, "metric": {"reason": "model execution unavailable"}, "status": "SKIPPED_NO_MODEL"}
    ordered = [selected[item_id] for item_id in sorted(selected)]
    total = len(ordered)
    correct = sum(row["prediction"] == row["target"] for row in ordered)
    if total < MIN_CI_SAMPLE:
        metric: dict[str, Any] = {
            "confidence_interval_95": None,
            "correct": correct,
            "method": "INSUFFICIENT_DATA total below preregistered minimum",
            "point_estimate": None,
            "total": total,
        }
        status = "INSUFFICIENT_DATA"
    else:
        low, high = wilson_95(correct, total)
        metric = {
            "confidence_interval_95": {"lower": low, "method": "Wilson z=1.959963984540054", "upper": high},
            "correct": correct,
            "method": "accuracy with preregistered Wilson 95% interval",
            "point_estimate": correct / total,
            "total": total,
        }
        status = "OK"
    return {**common, "metric": metric, "status": status}


def validate_scorecard(value: dict[str, Any], schema_path: Path) -> None:
    try:
        schema = parse_json(schema_path.read_bytes(), f"schema={schema_path}")
    except OSError as error:
        raise EvalError(f"scorecard schema unavailable: {schema_path}") from error
    if not isinstance(schema, dict) or schema.get("schema_version") != 1:
        raise EvalError("scorecard schema_version must equal 1")
    required = set(schema.get("required", []))
    if required != SCORECARD_REQUIRED:
        raise EvalError("frozen scorecard schema required-field drift")
    missing = sorted(required - set(value))
    if missing:
        raise EvalError(f"scorecard missing mandatory identity fields={missing}")
    if value.get("schema_version") != 1:
        raise EvalError("scorecard schema_version must equal 1")
    if value.get("status") not in schema.get("status_values", []):
        raise EvalError(f"unknown scorecard status={value.get('status')}")
    require_sha256(value.get("dataset_digest_sha256"), "dataset_digest_sha256", str(value.get("dataset_id")))
    require_sha256(value.get("input_digest_sha256"), "input_digest_sha256", str(value.get("dataset_id")))
    require_sha256(value.get("prompt_hash"), "prompt_hash", str(value.get("dataset_id")))
    if not isinstance(value.get("dataset"), dict) or value["dataset"].get("id") != value["dataset_id"]:
        raise EvalError("scorecard dataset identity object is incomplete")


def run_self_test() -> None:
    datasets = validate_registry(DEFAULT_REGISTRY)
    dataset = datasets["fnlp-repo-authored-binary-v1"]
    fixture = ROOT / dataset.record["fixture_path"]
    verify_user_file(dataset, fixture)
    with tempfile.TemporaryDirectory(prefix="fnlp-eval-core-") as temporary:
        directory = Path(temporary)
        bad = directory / "bad.ndjson"
        bad.write_text("tampered\n", encoding="utf-8")
        try:
            verify_user_file(dataset, bad)
        except EvalError as error:
            if "typed refusal digest" not in str(error):
                raise
        else:
            raise EvalError("digest-mismatch negative was accepted")
        overlap = dict(dataset.record)
        overlap["splits"] = {key: list(value) for key, value in dataset.record["splits"].items()}
        overlap["splits"]["calibration"].append("dev-001")
        try:
            validate_dataset(Dataset(overlap), verify_fixture=False)
        except EvalError as error:
            if "split-overlap" not in str(error):
                raise
        else:
            raise EvalError("split-overlap negative was accepted")
        restricted = Dataset({
            **dataset.record,
            "acquisition": {**dataset.record["acquisition"], "automated_access_allowed": False},
        })
        try:
            acquisition_path(restricted, None)
        except EvalError as error:
            if "user-supplied file required" not in str(error):
                raise
        else:
            raise EvalError("license-gated acquisition negative was accepted")
        rows = load_ndjson(ROOT / "tests" / "fixtures" / "eval" / "stub_binary_inputs.ndjson")
        digest = sha256_bytes(canonical_json(rows))
        first = scorecard(dataset, "test", rows, input_digest=digest, recipe_id="fixture-r1", prompt_hash="a" * 64, thinking_mode="disabled", model_gated_skip=False)
        second = scorecard(dataset, "test", list(reversed(rows)), input_digest=digest, recipe_id="fixture-r1", prompt_hash="a" * 64, thinking_mode="disabled", model_gated_skip=False)
        if canonical_json(first) != canonical_json(second):
            raise EvalError("metric order-invariance property failed")
        validate_scorecard(first, DEFAULT_SCHEMA)
        tiny = scorecard(dataset, "test", rows[:1], input_digest=digest, recipe_id="fixture-r1", prompt_hash="a" * 64, thinking_mode="disabled", model_gated_skip=True)
        if tiny["status"] != "SKIPPED_NO_MODEL":
            raise EvalError("model-gated skip did not remain typed")
        incomplete_dataset = Dataset({**dataset.record, "splits": {**dataset.record["splits"], "test": ["test-001"]}})
        tiny_rows = [row for row in rows if row["id"] == "test-001"]
        tiny_metric = scorecard(incomplete_dataset, "test", tiny_rows, input_digest=digest, recipe_id="fixture-r1", prompt_hash="a" * 64, thinking_mode="disabled", model_gated_skip=False)
        if tiny_metric["status"] != "INSUFFICIENT_DATA" or tiny_metric["metric"]["point_estimate"] is not None:
            raise EvalError("tiny-sample handling claimed precision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-registry", action="store_true")
    action.add_argument("--acquire", metavar="DATASET_ID")
    action.add_argument("--score", metavar="DATASET_ID")
    action.add_argument("--check-scorecard", type=Path)
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--user-file", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--recipe-id")
    parser.add_argument("--prompt-hash")
    parser.add_argument("--thinking-mode")
    parser.add_argument("--model-gated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
            log("RESULT=PASS action=self-test checks=6 failures=none")
            return 0
        datasets = validate_registry(args.registry)
        if args.validate_registry:
            log(f"RESULT=PASS action=validate-registry datasets={len(datasets)} failures=none")
            return 0
        if args.acquire:
            dataset = datasets.get(args.acquire)
            if dataset is None:
                raise EvalError(f"unknown dataset_id={args.acquire}")
            source = acquisition_path(dataset, args.user_file)
            verify_user_file(dataset, source)
            return 0
        if args.check_scorecard:
            value = parse_json(args.check_scorecard.read_bytes(), f"scorecard={args.check_scorecard}")
            if not isinstance(value, dict):
                raise EvalError("scorecard root must be an object")
            validate_scorecard(value, args.schema)
            log("RESULT=PASS action=check-scorecard failures=none")
            return 0
        if args.score:
            dataset = datasets.get(args.score)
            if dataset is None:
                raise EvalError(f"unknown dataset_id={args.score}")
            if args.input is None or args.output is None:
                raise EvalError("--score requires --input and --output")
            if args.recipe_id is None or args.prompt_hash is None or args.thinking_mode is None:
                raise EvalError("--score requires --recipe-id --prompt-hash --thinking-mode")
            raw = args.input.read_bytes()
            value = scorecard(
                dataset,
                args.split,
                load_ndjson(args.input),
                input_digest=sha256_bytes(raw),
                recipe_id=args.recipe_id,
                prompt_hash=args.prompt_hash,
                thinking_mode=args.thinking_mode,
                model_gated_skip=args.model_gated,
            )
            validate_scorecard(value, args.schema)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical_json(value))
            log(f"RESULT=PASS action=score dataset={dataset.id} status={value['status']} split={args.split}")
            return 0
        raise EvalError("no action selected")
    except (EvalError, OSError) as error:
        log(f"RESULT=FAIL action=error detail={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
