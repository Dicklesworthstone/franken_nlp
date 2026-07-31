#!/usr/bin/env python3
"""Verify the pinned, CPU-only Nanbeige4.2-3B reference-oracle closure.

This tool deliberately has no dependency outside the closure it verifies. It never
downloads model weights, uses CUDA, removes a directory, or rewrites a result:
callers must supply a new venv/work/output path for every mutating operation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import venv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "docs/truth-pack/oracle_env.json"
LOCK_PATH = ROOT / "docs/truth-pack/oracle_requirements.lock"
MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
CANONICAL_NAME = re.compile(r"[-_.]+")


class OracleFailure(RuntimeError):
    """A deterministic contract failure."""


class NoModel(RuntimeError):
    """The model-gated half could not start because the local snapshot is absent."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(event: str, **fields: object) -> None:
    rendered = " ".join(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in fields.items())
    print(f"{utc_now()} ORACLE_ENV {event}" + (f" {rendered}" if rendered else ""), file=sys.stderr)


def result(status: str, **fields: object) -> int:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    extras = " ".join(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in fields.items())
    print(f"ORACLE_ENV RESULT={status} python={version}" + (f" {extras}" if extras else ""), file=sys.stderr)
    return 0 if status in {"PASS", "SKIPPED_NO_MODEL"} else 1


def canonical_name(name: str) -> str:
    return CANONICAL_NAME.sub("-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record() -> dict[str, Any]:
    try:
        value = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OracleFailure(f"missing closure record: {RECORD_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise OracleFailure(f"invalid closure JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise OracleFailure("closure record root must be an object")
    return value


def expected_packages(record: dict[str, Any]) -> dict[str, str]:
    packages = record.get("closure", {}).get("packages")
    if not isinstance(packages, list) or not packages:
        raise OracleFailure("closure.packages must be a non-empty list")
    result_map: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise OracleFailure("closure.packages entries must be objects")
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise OracleFailure("closure package needs string name and version")
        canonical = canonical_name(name)
        if canonical in result_map:
            raise OracleFailure(f"duplicate closure package: {canonical}")
        result_map[canonical] = version
    return result_map


def freeze_digest(packages: dict[str, str]) -> str:
    payload = "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def installed_packages(python: Path | None = None) -> dict[str, str]:
    if python is None:
        distributions = importlib.metadata.distributions()
        return {canonical_name(dist.metadata["Name"]): dist.version for dist in distributions}
    completed = run([str(python), "-m", "pip", "freeze", "--all"], "pip_freeze")
    if completed.returncode != 0:
        raise OracleFailure("pip freeze failed")
    package_map: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        package_map[canonical_name(name)] = version
    return package_map


def first_difference(expected: dict[str, str], actual: dict[str, str]) -> str | None:
    for name in sorted(set(expected) | set(actual)):
        if expected.get(name) != actual.get(name):
            return f"{name}==expected:{expected.get(name, '<missing>')} actual:{actual.get(name, '<missing>')}"
    return None


def verify_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != 1:
        raise OracleFailure("unsupported oracle_env schema_version")
    source = record.get("source_identity")
    if not isinstance(source, dict) or source.get("model_id") != MODEL_ID or source.get("revision") != REVISION:
        raise OracleFailure("closure record model_id/revision differs from the pinned oracle")
    closure = record.get("closure")
    if not isinstance(closure, dict):
        raise OracleFailure("closure must be an object")
    if closure.get("lock_path") != "docs/truth-pack/oracle_requirements.lock":
        raise OracleFailure("closure lock_path is not the committed oracle lock")
    if not LOCK_PATH.is_file():
        raise OracleFailure(f"missing closure lock: {LOCK_PATH}")
    observed_lock_sha = sha256_file(LOCK_PATH)
    if closure.get("lock_sha256") != observed_lock_sha:
        raise OracleFailure(
            f"lock digest mismatch expected={closure.get('lock_sha256')} observed={observed_lock_sha}"
        )
    packages = expected_packages(record)
    observed_freeze_sha = freeze_digest(packages)
    if closure.get("freeze_sha256") != observed_freeze_sha:
        raise OracleFailure(
            f"closure package digest mismatch expected={closure.get('freeze_sha256')} observed={observed_freeze_sha}"
        )
    required_names = {"torch", "transformers", "sentencepiece"}
    if not required_names <= set(packages):
        raise OracleFailure("closure must pin torch, transformers, and sentencepiece")
    if closure.get("platform") != "macos-arm64" or closure.get("python") != "CPython 3.11":
        raise OracleFailure("the selected closure must state its CPython 3.11 macOS-arm64 target")
    if not isinstance(record.get("environment_capture"), dict):
        raise OracleFailure("environment_capture contract is missing")
    log("record_valid", lock_sha256=observed_lock_sha, freeze_sha256=observed_freeze_sha, package_count=len(packages))


def run(command: list[str], event: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    log("command_start", step=event, command=command, cwd=str(cwd) if cwd else None)
    completed = subprocess.run(command, text=True, capture_output=True, cwd=cwd, check=False)
    log(
        "command_complete",
        step=event,
        returncode=completed.returncode,
        stdout_tail=completed.stdout[-2000:],
        stderr_tail=completed.stderr[-2000:],
    )
    return completed


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OracleFailure(f"refusing to overwrite existing output: {path}")
    if not path.parent.is_dir():
        raise OracleFailure(f"output parent does not exist: {path.parent}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as exc:
        raise OracleFailure(f"refusing to overwrite existing output: {path}") from exc
    log("record_written", path=str(path), sha256=sha256_file(path))


def venv_python(venv_path: Path) -> Path:
    return venv_path / "bin" / "python"


def recreate(record: dict[str, Any], target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise OracleFailure(f"--venv must name a new, non-symlink path: {target}")
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        raise OracleFailure("the committed closure is macOS arm64 only; refusing incompatible wheel installation")
    if sys.version_info[:2] != (3, 11):
        raise OracleFailure("--recreate must run under CPython 3.11 for this cp311 closure")
    if not target.parent.is_dir():
        raise OracleFailure(f"venv parent does not exist: {target.parent}")
    log("recreate_start", venv=str(target), lock=str(LOCK_PATH))
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(target)
    python = venv_python(target)
    install = run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--require-hashes",
            "--no-deps",
            "--requirement",
            str(LOCK_PATH),
        ],
        "recreate_install",
    )
    if install.returncode != 0:
        raise OracleFailure("hash-addressed closure installation failed")
    expected = expected_packages(record)
    actual = installed_packages(python)
    difference = first_difference(expected, {name: actual[name] for name in expected if name in actual})
    if difference:
        raise OracleFailure(f"recreation mismatch first differing package==version: {difference}")
    observed = freeze_digest({name: actual[name] for name in expected})
    if observed != record["closure"]["freeze_sha256"]:
        raise OracleFailure(
            f"recreation closure hash mismatch expected={record['closure']['freeze_sha256']} observed={observed}"
        )
    log("recreate_pass", venv=str(target), freeze_sha256=observed, package_count=len(expected))


def source_files(record: dict[str, Any], source: Path, *, need_weights: bool) -> None:
    if not source.is_dir():
        raise NoModel(f"source snapshot absent: {source}")
    required = record["source_identity"].get("required_files", {})
    if not isinstance(required, dict):
        raise OracleFailure("source_identity.required_files must be an object")
    for name, expected_sha in required.items():
        path = source / name
        if not path.is_file():
            raise NoModel(f"source snapshot missing required file: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise OracleFailure(f"source hash mismatch file={name} expected={expected_sha} observed={observed_sha}")
    if need_weights and not list(source.glob("model-*.safetensors")):
        raise NoModel(f"source snapshot has no model-*.safetensors weights: {source}")


def capture_environment() -> dict[str, Any]:
    import torch

    prefixes = ("OMP_", "MKL_", "KMP_", "OPENBLAS_", "VECLIB_", "BLIS_", "NUMEXPR_", "TORCH", "PYTORCH", "ATEN", "CUBLAS_", "CUDA_", "HIP_", "MPS_")
    explicit = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "KMP_SETTINGS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "TORCH_DETERMINISTIC",
        "PYTORCH_DETERMINISTIC",
    )
    environment = {name: os.environ.get(name) for name in explicit}
    environment.update({name: value for name, value in os.environ.items() if name.startswith(prefixes)})
    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "sentencepiece", "tokenizers", "safetensors")
    }
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": package_versions,
        "torch_version": torch.__version__,
        "torch_build_config": torch.__config__.show(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_mkldnn_available": torch.backends.mkldnn.is_available(),
        "environment": dict(sorted(environment.items())),
    }


def validate_installed_closure(record: dict[str, Any]) -> None:
    expected = expected_packages(record)
    actual = installed_packages()
    relevant_actual = {name: actual[name] for name in expected if name in actual}
    difference = first_difference(expected, relevant_actual)
    if difference:
        raise OracleFailure(f"installed closure mismatch first differing package==version: {difference}")
    observed = freeze_digest(relevant_actual)
    if observed != record["closure"]["freeze_sha256"]:
        raise OracleFailure(
            f"installed closure hash mismatch expected={record['closure']['freeze_sha256']} observed={observed}"
        )


def smoke(record: dict[str, Any], source: Path, output: Path | None) -> None:
    source_files(record, source, need_weights=True)
    validate_installed_closure(record)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("smoke_load_start", source=str(source), revision=REVISION, device="cpu", attention="eager")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False, local_files_only=True)
    if getattr(tokenizer, "is_fast", None) is not False:
        raise OracleFailure(f"slow tokenizer required; got {tokenizer.__class__.__name__}")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cpu")
    model.eval()
    actual_attention = getattr(model.config, "_attn_implementation", None)
    if actual_attention != "eager":
        raise OracleFailure(f"eager attention required; runtime config reported {actual_attention!r}")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise OracleFailure("CPU placement required; a model parameter is not on CPU")
    prompt = record["smoke"]["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        forward = model(**inputs, use_cache=True)
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=record["smoke"]["max_new_tokens"])
    prompt_length = int(inputs["input_ids"].shape[1])
    new_tokens = generated[0, prompt_length:].tolist()
    payload = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "runtime": capture_environment(),
        "execution": {
            "device": "cpu",
            "inference_mode": True,
            "attn_implementation": actual_attention,
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_is_fast": tokenizer.is_fast,
            "prompt": prompt,
            "prompt_token_ids": inputs["input_ids"][0].tolist(),
            "forward_logits_shape": list(forward.logits.shape),
            "first_next_token": int(torch.argmax(forward.logits[0, -1]).item()),
            "greedy_new_token_ids": new_tokens,
            "greedy_text": tokenizer.decode(new_tokens, skip_special_tokens=False),
        },
    }
    expected = record["smoke"].get("expected")
    if expected is not None and payload["execution"]["greedy_new_token_ids"] != expected.get("greedy_new_token_ids"):
        raise OracleFailure("smoke greedy tokens differ from the recorded oracle output")
    if output is not None:
        write_new_json(output, payload)
    log("smoke_pass", tokenizer=tokenizer.__class__.__name__, generated_tokens=len(new_tokens))


def matrix_import_command(source: Path) -> str:
    return (
        "from transformers import AutoConfig, AutoTokenizer; "
        "from transformers.dynamic_module_utils import get_class_from_dynamic_module; "
        f"p={str(source)!r}; "
        "c=AutoConfig.from_pretrained(p, trust_remote_code=True, local_files_only=True); "
        "r=c.auto_map['AutoModelForCausalLM']; "
        "get_class_from_dynamic_module(r, p, local_files_only=True); "
        "t=AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=False, local_files_only=True); "
        "assert t.is_fast is False; print(type(t).__name__)"
    )


def matrix(record: dict[str, Any], source: Path, work_dir: Path, output: Path | None) -> None:
    source_files(record, source, need_weights=False)
    if work_dir.exists() or work_dir.is_symlink() or not work_dir.parent.is_dir():
        raise OracleFailure("--matrix-work-dir must be a new path whose parent already exists")
    candidates = record.get("matrix", {}).get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise OracleFailure("matrix.candidates must be a non-empty list")
    work_dir.mkdir()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        identifier = candidate["id"]
        candidate_venv = work_dir / identifier
        log("matrix_candidate_start", candidate=identifier)
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(candidate_venv)
        python = venv_python(candidate_venv)
        if candidate.get("selected"):
            install_command = [
                str(python), "-m", "pip", "install", "--disable-pip-version-check", "--require-hashes", "--no-deps", "-r", str(LOCK_PATH)
            ]
        else:
            install_command = [
                str(python), "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
                "--extra-index-url", "https://download.pytorch.org/whl/cpu",
                f"torch=={candidate['torch']}", f"transformers=={candidate['transformers']}", "sentencepiece==0.2.0",
            ]
        installed = run(install_command, f"matrix_install_{identifier}")
        imported = run([str(python), "-c", matrix_import_command(source)], f"matrix_import_{identifier}") if installed.returncode == 0 else None
        verdict = "PASS" if imported is not None and imported.returncode == 0 else "REJECTED"
        row = {
            "id": identifier,
            "transformers": candidate["transformers"],
            "torch": candidate["torch"],
            "verdict": verdict,
            "install_returncode": installed.returncode,
            "import_returncode": imported.returncode if imported is not None else None,
            "error": (imported.stderr if imported is not None else installed.stderr)[-4000:],
            "venv": str(candidate_venv),
        }
        rows.append(row)
        log("matrix_candidate_complete", candidate=identifier, verdict=verdict)
    payload = {"schema_version": 1, "source": str(source), "rows": rows, "captured_at": utc_now()}
    if output is not None:
        write_new_json(output, payload)
    if not any(row["verdict"] == "PASS" and row["id"] == "tf451_torch260" for row in rows):
        raise OracleFailure("selected candidate tf451_torch260 did not import the pinned remote code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true", help="validate the committed record and hash-addressed lock")
    action.add_argument("--recreate", action="store_true", help="create and verify a new closure venv")
    action.add_argument("--smoke", action="store_true", help="run CPU eager forward plus 32-token greedy decode")
    action.add_argument("--matrix", action="store_true", help="run the bounded Transformers/PyTorch source-import matrix")
    parser.add_argument("--venv", type=Path, help="new venv path required by --recreate")
    parser.add_argument("--source", type=Path, help="pinned local Hugging Face snapshot required by --smoke/--matrix")
    parser.add_argument("--smoke-output", type=Path, help="new JSON path for a smoke transcript")
    parser.add_argument("--matrix-work-dir", type=Path, help="new persistent directory for matrix candidate venvs")
    parser.add_argument("--matrix-output", type=Path, help="new JSON path for matrix results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = read_record()
        verify_record(record)
        if args.validate:
            return result("PASS", check="record_and_lock")
        if args.recreate:
            if args.venv is None:
                raise OracleFailure("--recreate requires --venv NEW_PATH")
            recreate(record, args.venv)
            return result("PASS", check="recreate")
        if args.smoke:
            if args.source is None:
                raise NoModel("--smoke requires --source PINNED_SNAPSHOT")
            smoke(record, args.source, args.smoke_output)
            return result("PASS", check="cpu_eager_smoke")
        if args.matrix:
            if args.source is None or args.matrix_work_dir is None:
                raise NoModel("--matrix requires --source PINNED_SNAPSHOT and --matrix-work-dir NEW_PATH")
            matrix(record, args.source, args.matrix_work_dir, args.matrix_output)
            return result("PASS", check="bounded_matrix")
        raise AssertionError("argparse action dispatch is exhaustive")
    except NoModel as exc:
        log("skipped_no_model", reason=str(exc))
        return result("SKIPPED_NO_MODEL", reason=str(exc))
    except OracleFailure as exc:
        log("failure", reason=str(exc))
        return result("FAIL", reason=str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
