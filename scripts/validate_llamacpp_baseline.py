"""Validate the official llama.cpp baseline record and optional local ancestry proof."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

MINIMUM_REVISION = "b77d646751d01c0962bc203b6809e9d94f7d50b7"
RECORD_NAME = "llamacpp_baseline.json"


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{stamp} LLAMACPP_BASELINE {message}", file=sys.stderr)


def load_record(path: Path) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read baseline record {path}: {error}") from error
    if not isinstance(record, dict):
        raise TypeError("baseline record must be a JSON object")
    return record


def valid_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def validate_record(record: dict[str, object]) -> list[str]:
    failures: list[str] = []
    selected = record.get("selected_official_llamacpp_revision")
    minimum = record.get("minimum_nanbeige_support_revision")
    if not valid_revision(selected):
        failures.append("selected_official_llamacpp_revision must be a lowercase 40-hex commit")
    if minimum != MINIMUM_REVISION:
        failures.append("minimum_nanbeige_support_revision is not the mandated b77d646 support commit")

    evidence = record.get("selection_evidence")
    if not isinstance(evidence, dict):
        failures.append("selection_evidence must be an object")
    else:
        if evidence.get("remote_compare_status") not in {"ahead", "identical"}:
            failures.append("recorded remote comparison does not prove the selected revision is at/after support")
        if evidence.get("remote_compare_merge_base") != MINIMUM_REVISION:
            failures.append("recorded remote comparison merge base is not the support commit")
        if evidence.get("remote_compare_behind_by") != 0:
            failures.append("recorded remote comparison says the selected revision is behind support")
        if selected and evidence.get("head_observation_output") != f"{selected}\tHEAD":
            failures.append("HEAD observation transcript does not bind the selected revision")

    boundaries = record.get("role_boundaries")
    if not isinstance(boundaries, dict):
        failures.append("role_boundaries must be an object")
    else:
        official = str(boundaries.get("official_ggml_org_llamacpp", "")).lower()
        authors = str(boundaries.get("authors_nanbeige_llamacpp_nanbeige42", "")).lower()
        rlx = str(boundaries.get("gpl3_rlx_nanbeige", "")).lower()
        if "secondary cpu/gguf differential" not in official or "performance baseline" not in official:
            failures.append("official llama.cpp role boundary is incomplete")
        if "historical lineage only" not in authors or "not an independent oracle" not in authors:
            failures.append("authors' fork must be historical lineage only, not an independent oracle")
        if "out-of-tree black-box differential only" not in rlx or "not an independent oracle" not in rlx:
            failures.append("RLX must be out-of-tree only, not an independent oracle")
        if "no code or dependency crosses" not in rlx:
            failures.append("RLX role boundary must prohibit code and dependency ingress")

    recipe = record.get("cpu_only_build_recipe")
    if not isinstance(recipe, dict):
        failures.append("cpu_only_build_recipe must be an object")
    else:
        configure = str(recipe.get("configure", ""))
        required_cpu_flags = ("-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF")
        for flag in required_cpu_flags:
            if flag not in configure:
                failures.append(f"CPU-only build recipe is missing {flag}")
        for key in ("build", "convert", "decode"):
            if not isinstance(recipe.get(key), str) or not recipe[key]:
                failures.append(f"CPU-only build recipe is missing its {key} command")

    smoke = record.get("smoke_transcript")
    if not isinstance(smoke, dict):
        failures.append("smoke_transcript must be an object")
    else:
        if smoke.get("status") not in {"PASS", "SKIPPED_NO_MODEL"}:
            failures.append("smoke transcript status must be PASS or SKIPPED_NO_MODEL")
        replay_order = smoke.get("armed_replay_order")
        if not isinstance(replay_order, list) or len(replay_order) != 4:
            failures.append("smoke transcript must retain configure/build/convert/decode replay commands")
    return failures


def verify_local_ancestry(llamacpp_dir: Path, minimum: str, selected: str) -> str | None:
    if not llamacpp_dir.is_dir():
        return f"llama.cpp directory does not exist: {llamacpp_dir}"
    command = [
        "git",
        "-C",
        str(llamacpp_dir.resolve()),
        "merge-base",
        "--is-ancestor",
        minimum,
        selected,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    except subprocess.TimeoutExpired:
        return f"local git ancestry check timed out after 15 seconds for {minimum} -> {selected}"
    log(f"ancestry command={' '.join(command)} exit={result.returncode}")
    if result.returncode != 0:
        return f"local git ancestry check failed for {minimum} -> {selected}: {result.stderr.strip()}"
    return None


def run_model_gated_smoke(record: dict[str, object], model_source: Path | None) -> str:
    smoke = record.get("smoke_transcript")
    if not isinstance(smoke, dict):
        log("smoke status=FAIL reason=baseline record has no smoke transcript")
        return "FAIL"
    commands = smoke.get("armed_replay_order")
    if not isinstance(commands, list) or not all(isinstance(command, str) for command in commands):
        log("smoke status=FAIL reason=baseline record has no valid replay-command list")
        return "FAIL"
    for index, command in enumerate(commands, start=1):
        log(f"smoke replay_step={index} command={command}")
    if model_source is None or not model_source.is_dir():
        log("smoke status=SKIPPED_NO_MODEL reason=pinned source closure is absent")
        return "SKIPPED_NO_MODEL"
    required_source_files = ("config.json", "model.safetensors.index.json", "tokenizer.model")
    missing = [name for name in required_source_files if not (model_source / name).is_file()]
    if missing:
        log(f"smoke status=SKIPPED_NO_MODEL reason=incomplete source closure missing={','.join(missing)}")
        return "SKIPPED_NO_MODEL"
    log("smoke status=FAIL reason=armed model execution requires an explicit retained build transcript; this validator never launches a build implicitly")
    return "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "truth-pack" / RECORD_NAME,
        help="baseline record to validate",
    )
    parser.add_argument("--llamacpp-dir", type=Path, help="local official llama.cpp checkout for a live ancestry check")
    parser.add_argument("--smoke", action="store_true", help="emit the exact model-gated replay commands and report PASS/FAIL/SKIPPED_NO_MODEL")
    parser.add_argument("--model-source", type=Path, help="revision-scoped pinned HF source closure used by --smoke")
    args = parser.parse_args()

    try:
        record = load_record(args.record)
        failures = validate_record(record)
    except ValueError as error:
        failures = [str(error)]
        record = {}
    selected = str(record.get("selected_official_llamacpp_revision", ""))
    if not failures and args.llamacpp_dir is not None:
        local_failure = verify_local_ancestry(args.llamacpp_dir, MINIMUM_REVISION, selected)
        if local_failure:
            failures.append(local_failure)
    if failures:
        for failure in failures:
            log(f"FAIL {failure}")
        log("RESULT=FAIL")
        return 1
    if args.smoke:
        result = run_model_gated_smoke(record, args.model_source)
        log(f"RESULT={result}")
        return 0 if result in {"PASS", "SKIPPED_NO_MODEL"} else 1
    log(f"record selected_revision={selected} minimum_revision={MINIMUM_REVISION}")
    log("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
