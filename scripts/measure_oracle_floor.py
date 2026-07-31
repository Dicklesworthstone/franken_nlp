#!/usr/bin/env python3
"""Analyze the pinned CPU oracle's independently-launched-run floor.

This pre-write deliberately implements only the model-free portions of
franken_nlp-snp: `--analyze-only` and `--self-test`.  Recording a live oracle
run is blocked on franken_nlp-ilz's immutable, smoke-proven closure.  No mode
in this script imports torch, downloads a model, creates a venv, or overwrites
an existing artifact.

Expected recorded-campaign layout (to be produced only after ilz is complete):

  <runs-dir>/preregistration.json
  <runs-dir>/runs/<fresh-process-run>.json
  <runs-dir>/runs/<optional raw metric sidecars>

Each run JSON is bound to preregistration.json by its SHA-256 and carries a
unique launch_id, a thread count, fresh_interpreter=true, the full environment
record, and a list of prompt records.  Each prompt provides greedy_token_ids,
one logits metric, and zero or more named taps.  Metrics are either an inline
finite-number `values` array or a digest-checked raw f32/f64/bf16 sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
from collections.abc import Iterable
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
RUN_SCHEMA_VERSION = 1
PREREGISTRATION_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 1
PROFILE = "hf-bf16-eager"
MINIMUM_PROCESSES = 5
MAX_METRIC_BYTES = 512 * 1024 * 1024
MAX_INLINE_VALUES = 2_000_000
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}")
SHA256 = re.compile(r"[0-9a-f]{64}")


class FloorError(RuntimeError):
    """A deterministic input or analysis contract failure."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(event: str, **fields: object) -> None:
    rendered = " ".join(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in fields.items())
    print(f"{utc_now()} ORACLE_FLOOR {event}" + (f" {rendered}" if rendered else ""), file=sys.stderr)


def emit_result(status: str, *, runs: int, thread_counts: Iterable[int], **fields: object) -> int:
    rendered_counts = json.dumps(sorted(thread_counts))
    extras = " ".join(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in fields.items())
    print(
        f"ORACLE_FLOOR RESULT={status} runs={runs} thread_counts={rendered_counts}"
        + (f" {extras}" if extras else ""),
        file=sys.stderr,
    )
    return 0 if status in {"PASS", "SKIPPED_NO_MODEL"} else 1


def fail(message: str) -> NoReturn:
    raise FloorError(message)


def is_uint(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def require_object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{location} must be an object")
    return value


def require_list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{location} must be an array")
    return value


def require_identifier(value: object, location: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        fail(f"{location} must be an ASCII identifier")
    return value


def require_sha256(value: object, location: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        fail(f"{location} must be lowercase SHA-256 hex")
    return value


def require_timestamp(value: object, location: str) -> datetime:
    if not isinstance(value, str):
        fail(f"{location} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FloorError(f"{location} is not RFC3339: {value}") from exc
    if parsed.tzinfo is None:
        fail(f"{location} must include a UTC offset")
    return parsed.astimezone(UTC)


def validate_run_timing(
    started_at: datetime,
    ended_at: datetime,
    preregistration_created_at: datetime,
    filename: str,
) -> None:
    if ended_at < started_at:
        fail(f"run record {filename}.ended_at precedes started_at")
    if started_at < preregistration_created_at:
        fail(f"preregistration ordering failure: {filename} starts before preregistration.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rejecting_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FloorError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} must be a regular file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FloorError(f"cannot read {label}: {path}: {exc}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=rejecting_object_pairs)
    except UnicodeDecodeError as exc:
        raise FloorError(f"{label} is not UTF-8: {path}") from exc
    except (json.JSONDecodeError, FloorError) as exc:
        raise FloorError(f"invalid {label}: {path}: {exc}") from exc
    return require_object(parsed, label), hashlib.sha256(raw).hexdigest()


def checked_product(values: Iterable[int], location: str) -> int:
    result = 1
    for value in values:
        if not is_uint(value) or value == 0:
            fail(f"{location} dimensions must be positive integers")
        result *= value
        if result > (1 << 63) - 1:
            fail(f"{location} element count exceeds supported limit")
    return result


def load_preregistration(path: Path) -> tuple[dict[str, Any], str]:
    record, digest = read_json(path, "preregistration")
    if record.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        fail("preregistration.schema_version is unsupported")
    if record.get("profile") != PROFILE:
        fail(f"preregistration.profile must be {PROFILE}")
    created_at = require_timestamp(record.get("created_at"), "preregistration.created_at")
    counts = require_list(record.get("thread_counts"), "preregistration.thread_counts")
    if len(counts) < 2:
        fail("preregistration.thread_counts must contain at least two counts")
    parsed_counts: list[int] = []
    for index, count in enumerate(counts):
        if not is_uint(count) or count < 1:
            fail(f"preregistration.thread_counts[{index}] must be a positive integer")
        parsed_counts.append(count)
    if len(set(parsed_counts)) != len(parsed_counts):
        fail("preregistration.thread_counts contains a duplicate count")
    minimum = record.get("minimum_processes")
    if minimum != MINIMUM_PROCESSES:
        fail(f"preregistration.minimum_processes must be exactly {MINIMUM_PROCESSES}")
    additional = record.get("additional_processes_per_thread")
    if not is_uint(additional) or additional < 1:
        fail("preregistration.additional_processes_per_thread must be a positive integer")
    confidence = record.get("confidence_level")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not (0.0 < float(confidence) < 1.0):
        fail("preregistration.confidence_level must be between zero and one")
    rule = record.get("convergence_rule")
    if not isinstance(rule, str) or not rule:
        fail("preregistration.convergence_rule must be a non-empty string")
    return {
        "created_at": created_at,
        "thread_counts": sorted(parsed_counts),
        "minimum_processes": minimum,
        "additional_processes_per_thread": additional,
        "confidence_level": float(confidence),
        "convergence_rule": rule,
    }, digest


def resolve_sidecar(run_path: Path, relative: object, location: str) -> Path:
    if not isinstance(relative, str):
        fail(f"{location}.path must be a relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail(f"{location}.path must stay below the run file directory")
    path = run_path.parent / candidate
    if not path.is_file() or path.is_symlink():
        fail(f"{location}.path must be a regular sidecar: {path}")
    try:
        path.resolve().relative_to(run_path.parent.resolve())
    except ValueError as exc:
        raise FloorError(f"{location}.path escapes the run file directory") from exc
    return path


def finite_values(values: object, location: str) -> list[float]:
    raw_values = require_list(values, f"{location}.values")
    if not raw_values:
        fail(f"{location}.values must not be empty")
    if len(raw_values) > MAX_INLINE_VALUES:
        fail(f"{location}.values exceeds inline cap; use a digest-checked sidecar")
    parsed: list[float] = []
    for index, value in enumerate(raw_values):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            fail(f"{location}.values[{index}] must be a number")
        converted = float(value)
        if not math.isfinite(converted):
            fail(f"{location}.values[{index}] must be finite")
        parsed.append(converted)
    return parsed


def decode_sidecar(path: Path, descriptor: dict[str, Any], location: str) -> list[float]:
    dtype = descriptor.get("dtype")
    widths = {"f32": 4, "f64": 8, "bf16": 2}
    if dtype not in widths:
        fail(f"{location}.dtype must be one of f32, f64, bf16")
    shape = require_list(descriptor.get("shape"), f"{location}.shape")
    expected_elements = checked_product(shape, f"{location}.shape")
    expected_sha = require_sha256(descriptor.get("sha256"), f"{location}.sha256")
    sidecar = resolve_sidecar(path, descriptor.get("path"), location)
    byte_length = expected_elements * widths[dtype]
    if byte_length > MAX_METRIC_BYTES:
        fail(f"{location} exceeds metric byte cap")
    observed_size = sidecar.stat().st_size
    if observed_size != byte_length:
        fail(f"{location} sidecar length expected={byte_length} observed={observed_size}")
    observed_sha = sha256_file(sidecar)
    if observed_sha != expected_sha:
        fail(f"{location} sidecar digest expected={expected_sha} observed={observed_sha}")
    payload = sidecar.read_bytes()
    if dtype == "f32":
        decoded = [item[0] for item in struct.iter_unpack("<f", payload)]
    elif dtype == "f64":
        decoded = [item[0] for item in struct.iter_unpack("<d", payload)]
    else:
        decoded = [struct.unpack("<f", struct.pack("<I", item[0] << 16))[0] for item in struct.iter_unpack("<H", payload)]
    if any(not math.isfinite(value) for value in decoded):
        fail(f"{location} contains a non-finite floating value")
    return decoded


def metric_values(run_path: Path, descriptor: object, location: str) -> list[float]:
    record = require_object(descriptor, location)
    if "values" in record:
        if set(record) != {"values"}:
            fail(f"{location} inline descriptor may contain only values")
        return finite_values(record["values"], location)
    expected_keys = {"path", "dtype", "shape", "sha256"}
    if set(record) != expected_keys:
        fail(f"{location} sidecar descriptor must contain exactly {sorted(expected_keys)}")
    return decode_sidecar(run_path, record, location)


def load_prompt(run_path: Path, value: object, location: str) -> dict[str, Any]:
    prompt = require_object(value, location)
    required = {"id", "greedy_token_ids", "logits", "taps"}
    if set(prompt) != required:
        fail(f"{location} must contain exactly {sorted(required)}")
    prompt_id = require_identifier(prompt["id"], f"{location}.id")
    tokens = require_list(prompt["greedy_token_ids"], f"{location}.greedy_token_ids")
    parsed_tokens: list[int] = []
    for token_index, token in enumerate(tokens):
        if not is_uint(token) or token > 0xFFFFFFFF:
            fail(f"{location}.greedy_token_ids[{token_index}] must be a u32")
        parsed_tokens.append(token)
    taps = require_object(prompt["taps"], f"{location}.taps")
    parsed_taps: dict[str, list[float]] = {}
    for name, metric in taps.items():
        require_identifier(name, f"{location}.taps key")
        parsed_taps[name] = metric_values(run_path, metric, f"{location}.taps.{name}")
    return {
        "id": prompt_id,
        "tokens": parsed_tokens,
        "metrics": {"logits": metric_values(run_path, prompt["logits"], f"{location}.logits"), **{f"tap:{name}": values for name, values in parsed_taps.items()}},
    }


def load_run(path: Path, preregistration_digest: str, preregistration: dict[str, Any]) -> dict[str, Any]:
    record, record_digest = read_json(path, "run record")
    required = {
        "schema_version",
        "run_id",
        "launch_id",
        "started_at",
        "ended_at",
        "profile",
        "thread_count",
        "fresh_interpreter",
        "preregistration_sha256",
        "environment",
        "prompts",
    }
    if set(record) != required:
        fail(f"run record {path.name} must contain exactly {sorted(required)}")
    if record["schema_version"] != RUN_SCHEMA_VERSION:
        fail(f"run record {path.name} has unsupported schema_version")
    run_id = require_identifier(record["run_id"], f"run record {path.name}.run_id")
    launch_id = require_identifier(record["launch_id"], f"run record {path.name}.launch_id")
    started_at = require_timestamp(record["started_at"], f"run record {path.name}.started_at")
    ended_at = require_timestamp(record["ended_at"], f"run record {path.name}.ended_at")
    validate_run_timing(started_at, ended_at, preregistration["created_at"], path.name)
    if record["profile"] != PROFILE:
        fail(f"run record {path.name}.profile must be {PROFILE}")
    thread_count = record["thread_count"]
    if not is_uint(thread_count) or thread_count not in preregistration["thread_counts"]:
        fail(f"run record {path.name}.thread_count is not preregistered")
    if record["fresh_interpreter"] is not True:
        fail(f"run record {path.name}.fresh_interpreter must be true")
    if record["preregistration_sha256"] != preregistration_digest:
        fail(f"run record {path.name} preregistration digest mismatch")
    require_object(record["environment"], f"run record {path.name}.environment")
    prompts = require_list(record["prompts"], f"run record {path.name}.prompts")
    if not prompts:
        fail(f"run record {path.name}.prompts must not be empty")
    parsed_prompts = [load_prompt(path, prompt, f"run record {path.name}.prompts[{index}]") for index, prompt in enumerate(prompts)]
    prompt_ids = [prompt["id"] for prompt in parsed_prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        fail(f"run record {path.name} has a duplicate prompt id")
    log(
        "run_loaded",
        run_file=str(path),
        run_id=run_id,
        launch_id=launch_id,
        thread_count=thread_count,
        sha256=record_digest,
        prompt_count=len(parsed_prompts),
    )
    return {
        "path": path,
        "run_id": run_id,
        "launch_id": launch_id,
        "started_at": started_at,
        "thread_count": thread_count,
        "record_sha256": record_digest,
        "prompts": {prompt["id"]: prompt for prompt in parsed_prompts},
    }


def load_campaign(runs_dir: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    if not runs_dir.is_dir() or runs_dir.is_symlink():
        fail(f"--analyze-only must name a real, non-symlink directory: {runs_dir}")
    preregistration, preregistration_digest = load_preregistration(runs_dir / "preregistration.json")
    run_dir = runs_dir / "runs"
    if not run_dir.is_dir() or run_dir.is_symlink():
        fail(f"missing regular runs directory: {run_dir}")
    run_paths = sorted(path for path in run_dir.glob("*.json") if path.is_file() and not path.is_symlink())
    if not run_paths:
        fail(f"no run JSON records found under {run_dir}")
    records = [load_run(path, preregistration_digest, preregistration) for path in run_paths]
    return preregistration, preregistration_digest, records


def ensure_same_prompt_shape(records: list[dict[str, Any]], thread_count: int) -> list[str]:
    reference = records[0]["prompts"]
    expected_ids = sorted(reference)
    for record in records[1:]:
        observed_ids = sorted(record["prompts"])
        if observed_ids != expected_ids:
            fail(
                f"thread_count={thread_count} run={record['run_id']} prompt set differs "
                f"expected={expected_ids} observed={observed_ids}"
            )
        for prompt_id in expected_ids:
            expected_metric_names = sorted(reference[prompt_id]["metrics"])
            observed_metric_names = sorted(record["prompts"][prompt_id]["metrics"])
            if observed_metric_names != expected_metric_names:
                fail(
                    f"thread_count={thread_count} prompt={prompt_id} run={record['run_id']} metric set differs "
                    f"expected={expected_metric_names} observed={observed_metric_names}"
                )
    return expected_ids


def bootstrap_mean_interval(values: list[float], confidence_level: float) -> dict[str, float | int | str]:
    if not values:
        fail("cannot compute a confidence interval without values")
    if len(values) == 1:
        return {"method": "degenerate-single-observation", "samples": 1, "lower": values[0], "upper": values[0]}
    encoded = ",".join(format(value, ".17g") for value in values).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    rng = random.Random(seed)
    resample_count = 2000
    size = len(values)
    means = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(resample_count))
    tail = (1.0 - confidence_level) / 2.0
    lower_index = max(0, math.floor(tail * (resample_count - 1)))
    upper_index = min(resample_count - 1, math.ceil((1.0 - tail) * (resample_count - 1)))
    return {
        "method": "deterministic-bootstrap-percentile-mean-per-run-max-abs",
        "samples": resample_count,
        "lower": means[lower_index],
        "upper": means[upper_index],
    }


def analyze_metric(metric_name: str, vectors: list[list[float]], confidence_level: float) -> dict[str, Any]:
    expected_length = len(vectors[0])
    if expected_length == 0:
        fail(f"metric {metric_name} is empty")
    for index, vector in enumerate(vectors[1:], start=1):
        if len(vector) != expected_length:
            fail(f"metric {metric_name} length differs in run index {index}: expected={expected_length} observed={len(vector)}")
    count = len(vectors)
    means = [0.0] * expected_length
    m2 = [0.0] * expected_length
    baseline = vectors[0]
    per_run_max_abs: list[float] = []
    max_abs_spread = 0.0
    for run_index, vector in enumerate(vectors):
        run_max = 0.0
        for element_index, value in enumerate(vector):
            if not math.isfinite(value):
                fail(f"metric {metric_name} has non-finite value run={run_index} element={element_index}")
            delta = value - means[element_index]
            means[element_index] += delta / (run_index + 1)
            m2[element_index] += delta * (value - means[element_index])
            spread = abs(value - baseline[element_index])
            run_max = max(run_max, spread)
            max_abs_spread = max(max_abs_spread, spread)
        per_run_max_abs.append(run_max)
    variances = [value / (count - 1) for value in m2] if count > 1 else [0.0] * expected_length
    zero_divergence_upper_rate = 1.0 - (1.0 - confidence_level) ** (1.0 / count)
    return {
        "metric": metric_name,
        "elements": expected_length,
        "run_count": count,
        "max_abs_spread": max_abs_spread,
        "mean_elementwise_variance": sum(variances) / len(variances),
        "max_elementwise_variance": max(variances),
        "per_run_max_abs": per_run_max_abs,
        "mean_per_run_max_abs_confidence_interval": bootstrap_mean_interval(per_run_max_abs, confidence_level),
        "zero_observed_divergence_upper_rate": zero_divergence_upper_rate if max_abs_spread == 0.0 else None,
    }


def common_prefix(token_streams: list[list[int]]) -> tuple[int, int | None]:
    if not token_streams:
        fail("cannot find a stable prefix without token streams")
    limit = min(len(tokens) for tokens in token_streams)
    for index in range(limit):
        token = token_streams[0][index]
        if any(tokens[index] != token for tokens in token_streams[1:]):
            return index, index
    if any(len(tokens) != limit for tokens in token_streams):
        return limit, limit
    return limit, None


def analyze_thread_group(records: list[dict[str, Any]], preregistration: dict[str, Any], thread_count: int) -> dict[str, Any]:
    if len(records) < preregistration["minimum_processes"]:
        fail(
            f"thread_count={thread_count} has only {len(records)} independently launched processes; "
            f"minimum is {preregistration['minimum_processes']} (two observations are never sufficient)"
        )
    launch_ids = [record["launch_id"] for record in records]
    if len(set(launch_ids)) != len(launch_ids):
        fail(f"thread_count={thread_count} reuses a launch_id; fresh interpreter evidence is not independent")
    prompt_ids = ensure_same_prompt_shape(records, thread_count)
    prompt_results: dict[str, Any] = {}
    all_metric_results: list[dict[str, Any]] = []
    variation_found = False
    for prompt_id in prompt_ids:
        token_streams = [record["prompts"][prompt_id]["tokens"] for record in records]
        stable_count, first_divergence = common_prefix(token_streams)
        token_varies = first_divergence is not None
        metrics: list[dict[str, Any]] = []
        for metric_name in sorted(records[0]["prompts"][prompt_id]["metrics"]):
            metric = analyze_metric(
                f"prompt:{prompt_id}:{metric_name}",
                [record["prompts"][prompt_id]["metrics"][metric_name] for record in records],
                preregistration["confidence_level"],
            )
            metrics.append(metric)
            all_metric_results.append(metric)
            variation_found = variation_found or metric["max_abs_spread"] != 0.0
        variation_found = variation_found or token_varies
        prompt_results[prompt_id] = {
            "stable_prefix_token_count": stable_count,
            "first_divergence_token_position": first_divergence,
            "token_stream_lengths": [len(tokens) for tokens in token_streams],
            "metrics": metrics,
        }
    return {
        "thread_count": thread_count,
        "run_count": len(records),
        "run_record_sha256s": [record["record_sha256"] for record in records],
        "launch_ids": launch_ids,
        "variation_detected": variation_found,
        "prompts": prompt_results,
        "derived_tolerance_vector": [
            {
                "metric": metric["metric"],
                "max_abs": metric["max_abs_spread"],
                "mean_elementwise_variance": metric["mean_elementwise_variance"],
                "run_count": metric["run_count"],
            }
            for metric in all_metric_results
        ],
    }


def analyze_records(preregistration: dict[str, Any], preregistration_digest: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["thread_count"]].append(record)
    observed_counts = sorted(groups)
    if observed_counts != preregistration["thread_counts"]:
        fail(
            f"recorded thread counts differ from preregistration "
            f"expected={preregistration['thread_counts']} observed={observed_counts}"
        )
    results = [analyze_thread_group(groups[count], preregistration, count) for count in observed_counts]
    any_variation = any(result["variation_detected"] for result in results)
    extra_required = preregistration["minimum_processes"] + preregistration["additional_processes_per_thread"]
    if any_variation:
        under_sampled = [result["thread_count"] for result in results if result["run_count"] < extra_required]
        if under_sampled:
            fail(
                "preregistered additional-repetition rule not satisfied after variation: "
                f"thread_counts={under_sampled} require_at_least={extra_required} fresh processes each"
            )
    stable_prefixes = {
        str(result["thread_count"]): {
            prompt_id: value["stable_prefix_token_count"] for prompt_id, value in result["prompts"].items()
        }
        for result in results
    }
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "profile": PROFILE,
        "preregistration_sha256": preregistration_digest,
        "preregistration": {
            "thread_counts": preregistration["thread_counts"],
            "minimum_processes": preregistration["minimum_processes"],
            "additional_processes_per_thread": preregistration["additional_processes_per_thread"],
            "confidence_level": preregistration["confidence_level"],
            "convergence_rule": preregistration["convergence_rule"],
        },
        "total_run_count": len(records),
        "variation_detected": any_variation,
        "oracle_reproducible_prefixes": stable_prefixes,
        "per_thread_count": results,
        "analysis_generated_at": utc_now(),
        "execution_status": "measured_from_recorded_runs",
    }


def write_new_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite existing output: {path}")
    if not path.parent.is_dir():
        fail(f"output parent does not exist: {path.parent}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with path.open("xb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as exc:
        raise FloorError(f"refusing to overwrite existing output: {path}") from exc
    digest = hashlib.sha256(encoded).hexdigest()
    log("analysis_written", path=str(path), sha256=digest)
    return digest


def synthetic_preregistration() -> dict[str, Any]:
    return {
        "created_at": datetime(2026, 7, 31, tzinfo=UTC),
        "thread_counts": [1, 4],
        "minimum_processes": MINIMUM_PROCESSES,
        "additional_processes_per_thread": 5,
        "confidence_level": 0.95,
        "convergence_rule": "After any observed variation, add five fresh processes at every preregistered thread count before deriving tolerances.",
    }


def synthetic_run(thread_count: int, index: int, *, divergent: bool = False, started_at: datetime | None = None) -> dict[str, Any]:
    tokens = [101, 102, 103, 104]
    logits = [0.25, -0.5, 1.0]
    tap = [0.125, 0.5]
    if divergent and index == 0:
        tokens[2] = 999
        logits[1] = -0.25
        tap[0] = 0.25
    started = started_at or datetime(2026, 8, 1, tzinfo=UTC)
    return {
        "path": Path(f"synthetic-{thread_count}-{index}.json"),
        "run_id": f"run-{thread_count}-{index}",
        "launch_id": f"launch-{thread_count}-{index}",
        "started_at": started,
        "thread_count": thread_count,
        "record_sha256": hashlib.sha256(f"{thread_count}:{index}:{divergent}".encode("ascii")).hexdigest(),
        "prompts": {
            "fixture": {
                "id": "fixture",
                "tokens": tokens,
                "metrics": {"logits": logits, "tap:loop0.layer0": tap},
            }
        },
    }


def self_test() -> None:
    preregistration = synthetic_preregistration()
    identical = [synthetic_run(count, index) for count in (1, 4) for index in range(MINIMUM_PROCESSES)]
    identical_result = analyze_records(preregistration, "0" * 64, identical)
    if identical_result["variation_detected"]:
        fail("self-test identical-runs case unexpectedly detected variation")
    for group in identical_result["per_thread_count"]:
        if group["prompts"]["fixture"]["stable_prefix_token_count"] != 4:
            fail("self-test identical-runs case did not preserve the complete stable prefix")
        if any(metric["max_abs_spread"] != 0.0 for metric in group["prompts"]["fixture"]["metrics"]):
            fail("self-test identical-runs case did not produce a zero numeric floor")
    log("self_test_case_pass", case="identical_runs_zero_floor")

    divergent = [synthetic_run(count, index, divergent=(count == 4)) for count in (1, 4) for index in range(10)]
    divergent_result = analyze_records(preregistration, "1" * 64, divergent)
    target = next(group for group in divergent_result["per_thread_count"] if group["thread_count"] == 4)
    prompt = target["prompts"]["fixture"]
    if prompt["stable_prefix_token_count"] != 2 or prompt["first_divergence_token_position"] != 2:
        fail("self-test divergent-token case did not exclude the unstable suffix at token index 2")
    if not any(metric["max_abs_spread"] > 0.0 for metric in prompt["metrics"]):
        fail("self-test divergent-numeric case did not report a measured spread")
    log("self_test_case_pass", case="divergence_stable_prefix_and_numeric_spread")

    try:
        analyze_records(preregistration, "2" * 64, [synthetic_run(1, 0), synthetic_run(1, 1), synthetic_run(4, 0), synthetic_run(4, 1)])
    except FloorError as exc:
        if "two observations are never sufficient" not in str(exc):
            raise
    else:
        fail("self-test two-observation refusal did not fail")
    log("self_test_case_pass", case="two_observation_refusal")

    out_of_order = [synthetic_run(count, index) for count in (1, 4) for index in range(MINIMUM_PROCESSES)]
    out_of_order[0]["started_at"] = datetime(2026, 7, 30, tzinfo=UTC)
    try:
        validate_run_timing(
            out_of_order[0]["started_at"],
            out_of_order[0]["started_at"],
            preregistration["created_at"],
            "synthetic-1-0.json",
        )
    except FloorError as exc:
        if "preregistration ordering failure" not in str(exc):
            raise
    else:
        fail("self-test preregistration ordering refusal did not fail")
    log("self_test_case_pass", case="preregistration_ordering_refusal")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analyze-only", type=Path, metavar="RUNS_DIR", help="analyze a previously recorded campaign; never loads a model")
    group.add_argument("--self-test", action="store_true", help="run model-free synthetic analysis checks")
    parser.add_argument("--output", type=Path, help="new JSON destination for --analyze-only; existing files are refused")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        if args.output is not None:
            fail("--output is valid only with --analyze-only")
        self_test()
        return emit_result("PASS", runs=0, thread_counts=[], mode="self_test", cases=4)

    preregistration, preregistration_digest, records = load_campaign(args.analyze_only)
    analysis = analyze_records(preregistration, preregistration_digest, records)
    if args.output is None:
        json.dump(analysis, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        log("analysis_emitted_stdout", sha256=hashlib.sha256(json.dumps(analysis, sort_keys=True).encode("utf-8")).hexdigest())
    else:
        write_new_json(args.output, analysis)
    return emit_result(
        "PASS",
        runs=analysis["total_run_count"],
        thread_counts=preregistration["thread_counts"],
        variation_detected=analysis["variation_detected"],
        profile=PROFILE,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except FloorError as exc:
        log("failure", detail=str(exc))
        raise SystemExit(emit_result("FAIL", runs=0, thread_counts=[], detail=str(exc)))
