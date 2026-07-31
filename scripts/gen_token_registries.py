#!/usr/bin/env python3
"""Generate or check the pinned Nanbeige token-control truth-pack registries.

The registry data is always extracted from the locally verified source closure.
In particular, ``special=true`` is not the untrusted-document containment
authority: the complete TemplateControlIds registry is that authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


MODEL_NAME = "Nanbeige4.2-3B"
MODEL_REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
VOCAB_SIZE = 166_144
DEFAULT_SOURCE = (
    Path.home()
    / ".cache"
    / "franken_nlp"
    / "source"
    / MODEL_NAME
    / MODEL_REVISION
)
DEFAULT_OUTPUT_DIR = Path("docs/truth-pack")
DEFAULT_DOCS_ROOT = Path(".")

TOKENIZER_SPECIAL_OUTPUT = "tokenizer_special_ids.json"
TEMPLATE_CONTROL_OUTPUT = "template_control_ids.json"
TOKENIZER_CONFIG = "tokenizer_config.json"
SPECIAL_TOKENS_MAP = "special_tokens_map.json"
ADDED_TOKENS = "added_tokens.json"

PINNED_SOURCES = {
    TOKENIZER_CONFIG: {
        "bytes": 10_990,
        "sha256": "3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518",
    },
    SPECIAL_TOKENS_MAP: {
        "bytes": 623,
        "sha256": "b718fce2b7a8940ffeddc1e67f3b092cc0d13ac885c63a021528786f8c4cf6c0",
    },
    ADDED_TOKENS: {
        "bytes": 174,
        "sha256": "9e3b127a27647df2c353cc1e5500826f7cdbe8bd15a458e368bba8422e9719cf",
    },
}
TEMPLATE_ONLY_SURFACES = ("<think>", "</think>", "<tool_call>", "</tool_call>")
REQUIRED_SPECIAL_SURFACES = ("<|im_start|>", "<|im_end|>")
TOKEN_ID_CONTRACT = {"bos": 166_100, "eos": 166_101, "pad": 0}
FORBIDDEN_SET_STATEMENT = (
    "The forbidden set for untrusted-document encoding is the complete "
    "TemplateControlIds census, never the tokenizer special flag. An untrusted "
    "segment must decode to the original bytes while excluding every "
    "TemplateControlIds member, or reject."
)


class RegistryError(RuntimeError):
    """A pinned-source, registry, or policy invariant failed."""


@dataclass(frozen=True)
class TokenMetadata:
    token_id: int
    surface: str
    special: bool
    config_path: str


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{timestamp()} TOKEN_REGISTRIES {message}", file=sys.stderr)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_pointer_escape(component: str) -> str:
    return component.replace("~", "~0").replace("/", "~1")


def json_path(*components: str) -> str:
    return "/" + "/".join(json_pointer_escape(component) for component in components)


def load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RegistryError(f"source file unavailable: {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"source file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"source file must be a JSON object: {path}")
    return value, raw


def verified_source_files(source: Path) -> dict[str, dict[str, Any]]:
    """Load the three pinned source files only after length/digest verification."""

    loaded: dict[str, dict[str, Any]] = {}
    for name, expected in PINNED_SOURCES.items():
        path = source / name
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RegistryError(f"pinned source missing: {name} path={path}: {exc}") from exc
        observed_digest = sha256_bytes(raw)
        if len(raw) != expected["bytes"]:
            raise RegistryError(
                f"pinned source length mismatch file={name} expected={expected['bytes']} "
                f"observed={len(raw)} path={path}"
            )
        if observed_digest != expected["sha256"]:
            raise RegistryError(
                f"pinned source digest mismatch file={name} expected={expected['sha256']} "
                f"observed={observed_digest} path={path}"
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"pinned source is not valid JSON file={name}: {exc}") from exc
        if not isinstance(value, dict):
            raise RegistryError(f"pinned source must be a JSON object file={name}")
        loaded[name] = {"document": value, "raw": raw}
        log(f"SOURCE file={name} bytes={len(raw)} sha256={observed_digest}")
    return loaded


def unverified_source_files(source: Path) -> dict[str, dict[str, Any]]:
    """Load a synthetic source fixture for the hermetic self-test only."""

    loaded: dict[str, dict[str, Any]] = {}
    for name in PINNED_SOURCES:
        document, raw = load_json_object(source / name)
        loaded[name] = {"document": document, "raw": raw}
    return loaded


def require_int(value: Any, *, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RegistryError(f"expected integer >= {minimum} at {location}: observed={value!r}")
    return value


def require_string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"expected non-empty string at {location}: observed={value!r}")
    return value


def extract_added_token_metadata(tokenizer_config: dict[str, Any]) -> dict[str, TokenMetadata]:
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise RegistryError("missing object tokenizer_config.json:/added_tokens_decoder")

    by_surface: dict[str, TokenMetadata] = {}
    ids: set[int] = set()
    for encoded_id, entry in decoder.items():
        location = json_path("added_tokens_decoder", str(encoded_id))
        try:
            token_id = int(encoded_id)
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"non-integer added-token id at {location}: {encoded_id!r}") from exc
        if token_id < 0 or token_id >= VOCAB_SIZE:
            raise RegistryError(f"added-token id outside vocab at {location}: {token_id}")
        if token_id in ids:
            raise RegistryError(f"duplicate added-token id at {location}: {token_id}")
        ids.add(token_id)
        if not isinstance(entry, dict):
            raise RegistryError(f"added-token metadata must be object at {location}")
        surface = require_string(entry.get("content"), location=f"{location}/content")
        special = entry.get("special")
        if not isinstance(special, bool):
            raise RegistryError(f"added-token special flag must be boolean at {location}/special")
        if surface in by_surface:
            raise RegistryError(
                f"duplicate added-token surface surface={surface!r} "
                f"first={by_surface[surface].config_path} second={location}"
            )
        by_surface[surface] = TokenMetadata(
            token_id=token_id,
            surface=surface,
            special=special,
            config_path=location,
        )
    return by_surface


def extract_added_token_mapping(added_tokens: dict[str, Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for surface, token_id_value in added_tokens.items():
        surface_text = require_string(surface, location="added_tokens.json object key")
        token_id = require_int(
            token_id_value,
            location=json_path(surface_text),
            minimum=0,
        )
        if token_id >= VOCAB_SIZE:
            raise RegistryError(f"added_tokens.json id outside vocab at {json_path(surface_text)}: {token_id}")
        mapping[surface_text] = token_id
    return mapping


def token_surface(value: Any, *, location: str) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    if value is None:
        return None
    raise RegistryError(f"token surface must be string or AddedToken object at {location}")


def matching_special_map_paths(value: Any, surface: str, path: str = "") -> list[str]:
    """Return JSON-pointer evidence locations which name a token surface."""

    matches: list[str] = []
    if isinstance(value, str):
        if value == surface:
            matches.append(path or "/")
    elif isinstance(value, dict):
        content = value.get("content")
        if content == surface:
            matches.append((path or "") + "/content")
        for key, child in value.items():
            matches.extend(matching_special_map_paths(child, surface, (path or "") + "/" + json_pointer_escape(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(matching_special_map_paths(child, surface, (path or "") + f"/{index}"))
    return sorted(set(matches))


def source_file_records(loaded: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bytes": len(loaded[name]["raw"]),
            "name": name,
            "sha256": sha256_bytes(loaded[name]["raw"]),
        }
        for name in sorted(loaded)
    ]


def evidence_for_token(
    metadata: TokenMetadata,
    added_mapping: dict[str, int],
    special_tokens_map: dict[str, Any],
    *,
    template_offset: int | None = None,
) -> list[dict[str, Any]]:
    evidence = [
        {
            "file": TOKENIZER_CONFIG,
            "json_path": metadata.config_path + "/content",
            "observed": metadata.surface,
        },
        {
            "file": TOKENIZER_CONFIG,
            "json_path": metadata.config_path + "/special",
            "observed": metadata.special,
        },
    ]
    mapped_id = added_mapping.get(metadata.surface)
    if mapped_id is not None:
        if mapped_id != metadata.token_id:
            raise RegistryError(
                f"added-token id disagreement surface={metadata.surface!r} "
                f"tokenizer_config={metadata.token_id} added_tokens={mapped_id} "
                f"path={json_path(metadata.surface)}"
            )
        evidence.append(
            {
                "file": ADDED_TOKENS,
                "json_path": json_path(metadata.surface),
                "observed": mapped_id,
            }
        )
    for path in matching_special_map_paths(special_tokens_map, metadata.surface):
        evidence.append(
            {
                "file": SPECIAL_TOKENS_MAP,
                "json_path": path,
                "observed": metadata.surface,
            }
        )
    if template_offset is not None:
        evidence.append(
            {
                "character_offset": template_offset,
                "file": TOKENIZER_CONFIG,
                "json_path": "/chat_template",
                "observed": metadata.surface,
            }
        )
    return sorted(evidence, key=lambda item: (item["file"], item["json_path"], str(item.get("observed"))))


def resolve_role_id(
    role: str,
    tokenizer_config: dict[str, Any],
    special_tokens_map: dict[str, Any],
    metadata_by_surface: dict[str, TokenMetadata],
) -> dict[str, Any]:
    expected_id = TOKEN_ID_CONTRACT[role]
    config_key = f"{role}_token_id"
    if config_key in tokenizer_config:
        observed_id = require_int(tokenizer_config[config_key], location="/" + config_key)
        if observed_id != expected_id:
            raise RegistryError(
                f"pinned {role} id mismatch path=/{config_key} expected={expected_id} observed={observed_id}"
            )
        return {
            "id": observed_id,
            "role": role,
            "source_evidence": {"file": TOKENIZER_CONFIG, "json_path": "/" + config_key},
        }

    token_key = f"{role}_token"
    candidates: list[tuple[str, str]] = []
    if token_key in tokenizer_config:
        surface = token_surface(tokenizer_config[token_key], location="/" + token_key)
        if surface is not None:
            candidates.append((surface, TOKENIZER_CONFIG + ":/" + token_key))
    if token_key in special_tokens_map:
        surface = token_surface(special_tokens_map[token_key], location="/" + token_key)
        if surface is not None:
            candidates.append((surface, SPECIAL_TOKENS_MAP + ":/" + token_key))
    for surface, location in candidates:
        metadata = metadata_by_surface.get(surface)
        if metadata is not None and metadata.token_id == expected_id:
            file_name, pointer = location.split(":", maxsplit=1)
            return {
                "id": expected_id,
                "role": role,
                "source_evidence": {"file": file_name, "json_path": pointer},
            }
    metadata = next((entry for entry in metadata_by_surface.values() if entry.token_id == expected_id), None)
    if metadata is None:
        raise RegistryError(
            f"unable to prove pinned {role} id={expected_id} from tokenizer metadata or special-token map"
        )
    return {
        "id": expected_id,
        "role": role,
        "source_evidence": {"file": TOKENIZER_CONFIG, "json_path": metadata.config_path + "/content"},
    }


def registry_entry(
    metadata: TokenMetadata,
    added_mapping: dict[str, int],
    special_tokens_map: dict[str, Any],
    *,
    template_offset: int | None = None,
) -> dict[str, Any]:
    return {
        "id": metadata.token_id,
        "source_evidence": evidence_for_token(
            metadata,
            added_mapping,
            special_tokens_map,
            template_offset=template_offset,
        ),
        "special": metadata.special,
        "surface": metadata.surface,
    }


def build_registries(loaded: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    tokenizer_config = loaded[TOKENIZER_CONFIG]["document"]
    special_tokens_map = loaded[SPECIAL_TOKENS_MAP]["document"]
    added_tokens = loaded[ADDED_TOKENS]["document"]
    metadata_by_surface = extract_added_token_metadata(tokenizer_config)
    added_mapping = extract_added_token_mapping(added_tokens)

    for surface, token_id in added_mapping.items():
        metadata = metadata_by_surface.get(surface)
        if metadata is None:
            raise RegistryError(
                f"added_tokens entry has no tokenizer_config metadata surface={surface!r} "
                f"path={json_path(surface)}"
            )
        if metadata.token_id != token_id:
            raise RegistryError(
                f"added_tokens id mismatch surface={surface!r} expected={metadata.token_id} observed={token_id}"
            )

    special_entries = [
        registry_entry(metadata, added_mapping, special_tokens_map)
        for metadata in metadata_by_surface.values()
        if metadata.special
    ]
    special_entries.sort(key=lambda entry: (entry["id"], entry["surface"]))
    special_surfaces = {entry["surface"] for entry in special_entries}
    for surface in REQUIRED_SPECIAL_SURFACES:
        if surface not in special_surfaces:
            raise RegistryError(f"missing TokenizerSpecialIds entry surface={surface!r} source=tokenizer_config.json")

    chat_template = require_string(tokenizer_config.get("chat_template"), location="/chat_template")
    template_entries = {entry["surface"]: entry for entry in special_entries}
    for surface in TEMPLATE_ONLY_SURFACES:
        metadata = metadata_by_surface.get(surface)
        if metadata is None:
            raise RegistryError(
                f"missing TemplateControlIds entry surface={surface!r} "
                "source=tokenizer_config.json:/added_tokens_decoder"
            )
        if metadata.special:
            raise RegistryError(
                f"template control must remain special=false at pin surface={surface!r} "
                f"path={metadata.config_path}/special observed=true"
            )
        template_offset = chat_template.find(surface)
        if template_offset < 0:
            raise RegistryError(
                f"template control lacks chat-template evidence surface={surface!r} "
                "path=/chat_template"
            )
        template_entries[surface] = registry_entry(
            metadata,
            added_mapping,
            special_tokens_map,
            template_offset=template_offset,
        )

    template_control_entries = sorted(template_entries.values(), key=lambda entry: (entry["id"], entry["surface"]))
    role_ids = [
        resolve_role_id(role, tokenizer_config, special_tokens_map, metadata_by_surface)
        for role in ("bos", "eos", "pad")
    ]
    common = {
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION, "vocab_size": VOCAB_SIZE},
        "normative_untrusted_document_rule": FORBIDDEN_SET_STATEMENT,
        "schema_version": 1,
        "source_files": source_file_records(loaded),
        "token_ids": role_ids,
    }
    specials = {**common, "entries": special_entries, "registry": "TokenizerSpecialIds"}
    controls = {**common, "entries": template_control_entries, "registry": "TemplateControlIds"}
    validate_registry_pair(specials, controls)
    return specials, controls


def entries_by_surface(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise RegistryError(f"registry missing entries list registry={registry.get('registry')!r}")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("surface"), str):
            raise RegistryError(f"registry has malformed entry registry={registry.get('registry')!r}")
        surface = entry["surface"]
        if surface in result:
            raise RegistryError(f"registry duplicate surface registry={registry['registry']} surface={surface!r}")
        result[surface] = entry
    return result


def validate_registry_pair(specials: dict[str, Any], controls: dict[str, Any]) -> None:
    if specials.get("registry") != "TokenizerSpecialIds":
        raise RegistryError("TokenizerSpecialIds registry name is missing or incorrect")
    if controls.get("registry") != "TemplateControlIds":
        raise RegistryError("TemplateControlIds registry name is missing or incorrect")
    special_by_surface = entries_by_surface(specials)
    control_by_surface = entries_by_surface(controls)
    missing_controls = sorted(set(special_by_surface) - set(control_by_surface))
    if missing_controls:
        raise RegistryError("TokenizerSpecialIds is not a TemplateControlIds subset missing=" + repr(missing_controls[0]))
    for surface in REQUIRED_SPECIAL_SURFACES:
        if surface not in special_by_surface or surface not in control_by_surface:
            raise RegistryError(f"required im control missing from both registries surface={surface!r}")
    for surface in TEMPLATE_ONLY_SURFACES:
        entry = control_by_surface.get(surface)
        if entry is None:
            raise RegistryError(f"template control missing surface={surface!r}")
        if entry.get("special") is not False:
            raise RegistryError(f"template control special flag changed surface={surface!r} observed={entry.get('special')!r}")
        evidence = entry.get("source_evidence")
        if not isinstance(evidence, list) or not any(item.get("json_path") == "/chat_template" for item in evidence if isinstance(item, dict)):
            raise RegistryError(f"template control lacks template evidence surface={surface!r}")
    token_ids = {entry.get("role"): entry.get("id") for entry in controls.get("token_ids", []) if isinstance(entry, dict)}
    if token_ids != TOKEN_ID_CONTRACT:
        raise RegistryError(f"pinned token id contract mismatch expected={TOKEN_ID_CONTRACT!r} observed={token_ids!r}")
    if specials.get("normative_untrusted_document_rule") != FORBIDDEN_SET_STATEMENT:
        raise RegistryError("TokenizerSpecialIds lacks the normative forbidden-set statement")
    if controls.get("normative_untrusted_document_rule") != FORBIDDEN_SET_STATEMENT:
        raise RegistryError("TemplateControlIds lacks the normative forbidden-set statement")


def describe_registry_diff(expected: dict[str, Any], observed_bytes: bytes, *, path: Path) -> str:
    expected_bytes = canonical_json_bytes(expected)
    if observed_bytes == expected_bytes:
        return ""
    try:
        observed = json.loads(observed_bytes)
    except json.JSONDecodeError as exc:
        return f"file={path} is not JSON: {exc}"
    if not isinstance(observed, dict):
        return f"file={path} is not a JSON object"
    try:
        expected_entries = entries_by_surface(expected)
        observed_entries = entries_by_surface(observed)
    except RegistryError as exc:
        return f"file={path} has malformed entries: {exc}"
    missing = sorted(set(expected_entries) - set(observed_entries))
    extra = sorted(set(observed_entries) - set(expected_entries))
    if missing:
        entry = expected_entries[missing[0]]
        evidence = entry["source_evidence"][0]
        return (
            f"file={path} missing id={entry['id']} surface={entry['surface']!r} "
            f"source={evidence['file']}:{evidence['json_path']}"
        )
    if extra:
        entry = observed_entries[extra[0]]
        return f"file={path} extra id={entry.get('id')!r} surface={extra[0]!r}"
    for surface in sorted(expected_entries):
        if expected_entries[surface] != observed_entries[surface]:
            return f"file={path} mismatched id={expected_entries[surface]['id']} surface={surface!r}"
    return f"file={path} differs outside entries or is noncanonical"


def check_committed_artifacts(output_dir: Path, specials: dict[str, Any], controls: dict[str, Any]) -> None:
    for name, generated in ((TOKENIZER_SPECIAL_OUTPUT, specials), (TEMPLATE_CONTROL_OUTPUT, controls)):
        path = output_dir / name
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise RegistryError(f"committed registry unavailable file={path}: {exc}") from exc
        detail = describe_registry_diff(generated, observed, path=path)
        if detail:
            raise RegistryError(f"registry drift {detail}")
        log(f"CHECK file={path} status=byte-identical sha256={sha256_bytes(observed)}")


def write_artifacts(output_dir: Path, specials: dict[str, Any], controls: dict[str, Any]) -> None:
    """Write only absent artifacts; a changed artifact must be review-created."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, generated in ((TOKENIZER_SPECIAL_OUTPUT, specials), (TEMPLATE_CONTROL_OUTPUT, controls)):
        path = output_dir / name
        rendered = canonical_json_bytes(generated)
        if path.exists():
            try:
                observed = path.read_bytes()
            except OSError as exc:
                raise RegistryError(f"cannot read existing registry file={path}: {exc}") from exc
            detail = describe_registry_diff(generated, observed, path=path)
            if detail:
                raise RegistryError(
                    f"refusing to overwrite differing registry; review the drift first: {detail}"
                )
            log(f"WRITE file={path} status=already-identical sha256={sha256_bytes(rendered)}")
            continue
        try:
            path.write_bytes(rendered)
        except OSError as exc:
            raise RegistryError(f"cannot write registry file={path}: {exc}") from exc
        log(f"WRITE file={path} bytes={len(rendered)} sha256={sha256_bytes(rendered)}")


def iter_policy_documents(root: Path) -> Iterable[Path]:
    excluded = {".git", ".beads", "target", "node_modules", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.suffix.lower() not in {".md", ".rst", ".txt"}:
            continue
        yield path


def lint_skip_special_tokens_boundary_claims(docs_root: Path) -> None:
    patterns = (
        re.compile(r"skip_special_tokens\s+(?:is|acts as|serves as|provides)\s+(?:the\s+)?boundary authority", re.IGNORECASE),
        re.compile(r"boundary authority\s+(?:is|uses|relies on)\s+skip_special_tokens", re.IGNORECASE),
    )
    for path in sorted(iter_policy_documents(docs_root)):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RegistryError(f"documentation policy input is not UTF-8 file={path}: {exc}") from exc
        normalized = re.sub(r"\s+", " ", text)
        if any(pattern.search(normalized) for pattern in patterns):
            raise RegistryError(
                f"docs policy violation file={path}: skip_special_tokens cannot be the boundary authority"
            )


def synthetic_source_fixture(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    config = {
        "added_tokens_decoder": {
            "0": {"content": "<pad>", "special": True},
            "166100": {"content": "<|im_start|>", "special": True},
            "166101": {"content": "<|im_end|>", "special": True},
            "166102": {"content": "<think>", "special": False},
            "166103": {"content": "</think>", "special": False},
            "166104": {"content": "<tool_call>", "special": False},
            "166105": {"content": "</tool_call>", "special": False},
        },
        "bos_token_id": 166100,
        "chat_template": "<|im_start|>{% if think %}<think>{% endif %}{% if tool %}<tool_call>{% endif %}</tool_call></think><|im_end|>",
        "eos_token_id": 166101,
        "pad_token_id": 0,
    }
    added = {entry["content"]: int(token_id) for token_id, entry in config["added_tokens_decoder"].items()}
    special_map = {"bos_token": "<|im_start|>", "eos_token": "<|im_end|>", "pad_token": "<pad>"}
    for name, document in (
        (TOKENIZER_CONFIG, config),
        (ADDED_TOKENS, added),
        (SPECIAL_TOKENS_MAP, special_map),
    ):
        (source / name).write_bytes(canonical_json_bytes(document))
    return source


def run_self_test() -> tuple[int, int]:
    with tempfile.TemporaryDirectory(prefix="fnlp-token-registries-") as temporary:
        root = Path(temporary)
        source = synthetic_source_fixture(root)
        loaded = unverified_source_files(source)
        specials, controls = build_registries(loaded)
        validate_registry_pair(specials, controls)
        if canonical_json_bytes(specials) != canonical_json_bytes(build_registries(loaded)[0]):
            raise RegistryError("self-test canonical special-registry replay was not byte-identical")
        output = root / "out"
        write_artifacts(output, specials, controls)
        check_committed_artifacts(output, specials, controls)

        config_path = source / TOKENIZER_CONFIG
        config, _ = load_json_object(config_path)
        config["added_tokens_decoder"]["166106"] = {"content": "<new_special>", "special": True}
        config_path.write_bytes(canonical_json_bytes(config))
        changed_specials, changed_controls = build_registries(unverified_source_files(source))
        try:
            check_committed_artifacts(output, changed_specials, changed_controls)
        except RegistryError as exc:
            if "missing id=166106 surface='<new_special>'" not in str(exc):
                raise RegistryError(
                    "self-test new-added-token negative did not report exact drift: " + str(exc)
                ) from exc
        else:
            raise RegistryError("self-test new-added-token negative was accepted by --check")

        docs = root / "docs"
        docs.mkdir()
        (docs / "safe.md").write_text("skip_special_tokens is never the boundary authority.\n", encoding="utf-8")
        lint_skip_special_tokens_boundary_claims(docs)
        (docs / "unsafe.md").write_text("skip_special_tokens is the boundary authority.\n", encoding="utf-8")
        try:
            lint_skip_special_tokens_boundary_claims(docs)
        except RegistryError as exc:
            if "docs policy violation" not in str(exc):
                raise
        else:
            raise RegistryError("self-test docs policy negative was accepted")
        return len(specials["entries"]), len(controls["entries"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    action = result.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify pinned source and byte-compare committed registries")
    action.add_argument("--write", action="store_true", help="write absent canonical registries from verified pinned source")
    action.add_argument("--self-test", action="store_true", help="run hermetic registry, drift, and docs-policy assertions")
    result.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="verified pinned source closure")
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="truth-pack registry directory")
    result.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT, help="repository root for docs policy scan")
    return result


def run(arguments: argparse.Namespace) -> tuple[int, int]:
    if arguments.self_test:
        return run_self_test()
    loaded = verified_source_files(arguments.source)
    specials, controls = build_registries(loaded)
    lint_skip_special_tokens_boundary_claims(arguments.docs_root)
    for registry in (specials, controls):
        for entry in registry["entries"]:
            log(
                f"ENTRY registry={registry['registry']} id={entry['id']} surface={entry['surface']!r} "
                f"special={entry['special']} evidence={entry['source_evidence']}"
            )
    if arguments.check:
        check_committed_artifacts(arguments.output_dir, specials, controls)
    else:
        write_artifacts(arguments.output_dir, specials, controls)
    return len(specials["entries"]), len(controls["entries"])


def main() -> int:
    arguments = parser().parse_args()
    specials = 0
    controls = 0
    drift = "none"
    try:
        specials, controls = run(arguments)
    except RegistryError as exc:
        message = str(exc)
        log(f"ERROR detail={message}")
        if "file=" in message:
            drift = message.split("file=", maxsplit=1)[1].split()[0]
        log(f"RESULT=FAIL specials={specials} controls={controls} drift={drift}")
        return 1
    log(f"RESULT=PASS specials={specials} controls={controls} drift={drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
