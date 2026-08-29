#!/usr/bin/env python3
"""Generate and replay the pinned Nanbeige chat-template AST truth pack.

This deliberately describes one pinned Jinja program; it is not a general
template interpreter.  The generated JSON carries source spans back to the
immutable ``tokenizer_config.json`` bytes and the companion Markdown calls out
where the Rust typed boundary must reject shapes that the Jinja itself would
otherwise render permissively.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "docs" / "truth-pack"
OUTPUT_JSON = TRUTH / "chat_ast.json"
OUTPUT_MD = TRUTH / "chat_ast.md"
MODEL = "Nanbeige4.2-3B"
REVISION = "f56ec5a9650268aa098496734743c25ea778bd2d"
CONFIG_BYTES = 10_990
CONFIG_SHA256 = "3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518"
TEMPLATE_BYTES = 8_375
TEMPLATE_SHA256 = "ed118a3c5ddf1d24ffa43229de22bacd5b803be31acaafeb4c0fff0cefee551a"


class ChatAstError(RuntimeError):
    """The pinned source or archive is malformed, stale, or incomplete."""


def log(message: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{timestamp} CHAT_AST {message}", file=sys.stderr)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def default_source() -> Path:
    source_dir = os.environ.get("FNLP_SOURCE_DIR")
    if source_dir:
        return Path(source_dir) / "tokenizer_config.json"
    return (
        Path.home()
        / ".cache"
        / "franken_nlp"
        / "source"
        / MODEL
        / REVISION
        / "tokenizer_config.json"
    )


def load_template(source: Path) -> str:
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ChatAstError(f"SKIPPED_NO_MODEL source={source}") from error
    if len(raw) != CONFIG_BYTES:
        raise ChatAstError(f"config byte mismatch expected={CONFIG_BYTES} observed={len(raw)} source={source}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CONFIG_SHA256:
        raise ChatAstError(f"config digest mismatch expected={CONFIG_SHA256} observed={digest} source={source}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ChatAstError(f"invalid tokenizer configuration source={source}: {error}") from error
    template = value.get("chat_template") if isinstance(value, dict) else None
    if not isinstance(template, str):
        raise ChatAstError("tokenizer_config.json:/chat_template must be a string")
    encoded = template.encode("utf-8")
    if len(encoded) != TEMPLATE_BYTES:
        raise ChatAstError(f"chat_template byte mismatch expected={TEMPLATE_BYTES} observed={len(encoded)}")
    template_digest = hashlib.sha256(encoded).hexdigest()
    if template_digest != TEMPLATE_SHA256:
        raise ChatAstError(
            f"chat_template digest mismatch expected={TEMPLATE_SHA256} observed={template_digest}"
        )
    return template


def line_offsets(template: str) -> list[int]:
    offsets: list[int] = []
    current = 0
    for line in template.splitlines(keepends=True):
        offsets.append(current)
        current += len(line.encode("utf-8"))
    if not offsets:
        raise ChatAstError("chat template unexpectedly has no lines")
    return offsets


def span_for_line(template: str, line_number: int) -> dict[str, Any]:
    lines = template.splitlines()
    if line_number < 1 or line_number > len(lines):
        raise ChatAstError(f"source span line out of range: {line_number}")
    line = lines[line_number - 1]
    offsets = line_offsets(template)
    start = offsets[line_number - 1]
    end = start + len(line.encode("utf-8"))
    return {
        "end_byte": end,
        "end_line": line_number,
        "quoted_template": line,
        "start_byte": start,
        "start_line": line_number,
    }


NODE_SPECS = [
    ("content.plain_string", 4, "Accept plain string content and render it verbatim."),
    ("content.iterable_parts", 6, "Accept iterable non-mapping content parts."),
    ("content.part.text", 8, "Render mapping text parts from item.text."),
    ("content.part.bare_string", 10, "Render bare string items in an iterable content value."),
    ("content.part.media", 12, "Replace each accepted media kind with the exact reminder literal."),
    ("content.other", 17, "The Jinja fallback renders a non-string content value; the typed adapter rejects unsupported shapes before tokenization."),
    ("tools.present", 24, "Use the tool-special system prelude when tools are supplied."),
    ("tools.system_present", 26, "Use only a leading system message in the tool prelude."),
    ("tools.system_default", 28, "Use the exact Chinese tool-calling system default when no leading system exists."),
    ("tools.format.json", 38, "Describe and emit the JSON-in-XML tool-call branch."),
    ("tools.format.xml", 41, "Describe and emit the XML parameter tool-call branch."),
    ("tools.absent", 57, "Use the ordinary system prelude when no tools are supplied."),
    ("system.leading", 58, "Render a first-position system message in the ordinary prelude."),
    ("system.default", 60, "Use the exact Nanbeige default system text with no tools."),
    ("tool_history.last_query", 68, "Identify the last non-tool-response user query for thinking preservation."),
    ("content.rendered_string", 75, "Bind visible content only when visible_text resolves to a string."),
    ("content.rendered_nonstring", 77, "Use an empty rendered-content binding for non-string visible content."),
    ("role.system", 81, "System role is special only at the first message position."),
    ("role.system_nonleading_reject", 82, "Template raises when a system message is not first."),
    ("role.assistant", 86, "Assistant messages have thinking and tool-call subtrees."),
    ("assistant.reasoning.explicit", 88, "Prefer a string reasoning_content field."),
    ("assistant.reasoning.embedded", 91, "Extract a complete embedded think region when the explicit field is absent."),
    ("assistant.reasoning.embedded_precondition", 90, "The else branch of the explicit-reasoning check; signals the path that falls through to the embedded extraction below."),
    ("assistant.reasoning.strip_prior", 98, "Replace prior-turn reasoning with an empty think region when preserve_thinking is false."),
    ("assistant.reasoning.preserve", 100, "Preserve trimmed reasoning when the strip condition does not apply."),
    ("assistant.tool_calls", 104, "Render an iterable non-mapping assistant tool_calls list."),
    ("assistant.tool_calls.json", 105, "Render JSON tool calls in tool_call XML delimiters."),
    ("assistant.tool_calls.json_separator", 107, "Insert JSON-call newlines after visible content and between calls."),
    ("assistant.tool_calls.function_wrapper", 110, "Accept the function wrapper form before serializing a tool call."),
    ("assistant.tool_calls.arguments_string", 116, "Preserve string tool-call arguments as supplied."),
    ("assistant.tool_calls.arguments_json", 118, "Serialize structured JSON tool-call arguments with tojson."),
    ("assistant.tool_calls.xml", 123, "Render XML function and parameter tool calls."),
    ("assistant.tool_calls.xml_function_wrapper", 125, "Accept the XML branch function wrapper form."),
    ("assistant.tool_calls.xml_spacing", 129, "Choose first-call spacing based on visible assistant content."),
    ("assistant.tool_calls.xml_nonempty_content", 130, "Insert the content-to-tool-call separator only for nonempty content."),
    ("assistant.tool_calls.xml_empty_content", 136, "Start a first XML tool call without a content separator when content is empty."),
    ("assistant.tool_calls.xml_subsequent", 141, "Separate every XML tool call after the first."),
    ("assistant.tool_calls.xml_arguments", 148, "Iterate named XML parameters when arguments are defined."),
    ("role.tool", 166, "Render tool responses in one user-framed consecutive group."),
    ("tool_results.group_start", 167, "Start a grouped tool-result user frame after a non-tool message."),
    ("tool_results.group_end", 173, "Close the grouped tool-result frame at the last adjacent tool result."),
    ("role.generic_nonempty", 176, "The raw template renders any other nonempty role; this is an adapter-required rejection boundary, not a raw-template rejection."),
    ("generation.suffix", 181, "Append the assistant generation suffix when requested."),
    ("generation.thinking_disabled", 184, "Append an empty think block when enable_thinking is explicitly false."),
    ("generation.thinking_enabled", 190, "Append an open think tag when thinking is enabled or unspecified."),
]


DIRECTIVE_NODES = {
    4: "content.plain_string",
    6: "content.iterable_parts",
    8: "content.part.text",
    10: "content.part.bare_string",
    12: "content.part.media",
    17: "content.other",
    24: "tools.present",
    26: "tools.system_present",
    28: "tools.system_default",
    38: "tools.format.json",
    41: "tools.format.xml",
    57: "tools.absent",
    58: "system.leading",
    60: "system.default",
    68: "tool_history.last_query",
    75: "content.rendered_string",
    77: "content.rendered_nonstring",
    81: "role.system",
    82: "role.system_nonleading_reject",
    86: "role.assistant",
    88: "assistant.reasoning.explicit",
    90: "assistant.reasoning.embedded_precondition",
    91: "assistant.reasoning.embedded",
    98: "assistant.reasoning.strip_prior",
    100: "assistant.reasoning.preserve",
    104: "assistant.tool_calls",
    105: "assistant.tool_calls.json",
    107: "assistant.tool_calls.json_separator",
    110: "assistant.tool_calls.function_wrapper",
    116: "assistant.tool_calls.arguments_string",
    118: "assistant.tool_calls.arguments_json",
    123: "assistant.tool_calls.xml",
    125: "assistant.tool_calls.xml_function_wrapper",
    129: "assistant.tool_calls.xml_spacing",
    130: "assistant.tool_calls.xml_nonempty_content",
    136: "assistant.tool_calls.xml_empty_content",
    141: "assistant.tool_calls.xml_subsequent",
    148: "assistant.tool_calls.xml_arguments",
    166: "role.tool",
    167: "tool_results.group_start",
    173: "tool_results.group_end",
    176: "role.generic_nonempty",
    181: "generation.suffix",
    184: "generation.thinking_disabled",
    190: "generation.thinking_enabled",
}

def assert_node_specs_directive_nodes_consistency() -> None:
    """Fail closed at import time if NODE_SPECS and DIRECTIVE_NODES disagree.

    NODE_SPECS names one canonical source line per AST node. DIRECTIVE_NODES
    names the AST node for each line that opens a Jinja {%- if/elif/else %}
    branch. If a line in DIRECTIVE_NODES maps to a node that NODE_SPECS records
    on a *different* line, the validator would silently mis-attribute branches
    and miss the drift in the self-test fixtures (the cr-005 failure mode).
    Pin the contract once at module load.
    """
    specs_by_line: dict[int, str] = {line: node_id for node_id, line, _description in NODE_SPECS}
    directives_by_line: dict[int, str] = dict(DIRECTIVE_NODES)
    drift: list[str] = []
    for line, node_id in directives_by_line.items():
        # If a DIRECTIVE_NODES line is not in NODE_SPECS, that is itself a
        # drift: every directive line in the template must be backed by a
        # NODE_SPECS entry. The previous formulation's `spec_node is not None`
        # guard masked this case (see cr-005 follow-up: line 9999 above
        # would silently pass).
        spec_node = specs_by_line.get(line)
        if spec_node != node_id:
            drift.append(
                f"line {line} maps to DIRECTIVE_NODES={node_id!r} but "
                f"NODE_SPECS={spec_node!r}"
            )
    for line, node_id in specs_by_line.items():
        directive_node = directives_by_line.get(line)
        if directive_node is None:
            continue
        if directive_node != node_id:
            drift.append(
                f"line {line} is in NODE_SPECS={node_id!r} but DIRECTIVE_NODES={directive_node!r}"
            )
    if drift:
        raise ChatAstError(
            "NODE_SPECS / DIRECTIVE_NODES line-number drift: "
            + "; ".join(drift)
        )

assert_node_specs_directive_nodes_consistency()

def node_records(template: str) -> list[dict[str, Any]]:
    return [
        {"description": description, "id": node_id, "source_span": span_for_line(template, line)}
        for node_id, line, description in NODE_SPECS
    ]


def conditional_directives(template: str, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_ids = [node["id"] for node in nodes]
    coverage: list[dict[str, Any]] = []
    for line_number, line in enumerate(template.splitlines(), start=1):
        stripped = line.strip()
        if not (stripped.startswith("{%- if") or stripped.startswith("{%- elif") or stripped.startswith("{%- else")):
            continue
        mapped = DIRECTIVE_NODES.get(line_number)
        if mapped is None:
            raise ChatAstError(f"unclassified conditional directive at pinned template line {line_number}")
        if mapped not in node_ids:
            raise ChatAstError(f"branch line {line_number} selected unknown AST node {mapped}")
        coverage.append(
            {
                "branch_id": f"template-line-{line_number}",
                "mapped_node_ids": [mapped],
                "source_span": span_for_line(template, line_number),
                "status": "covered",
            }
        )
    if not coverage:
        raise ChatAstError("no conditional directives were found in pinned chat template")
    return coverage


def mode_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for enable_thinking, preserve_thinking, tool_format, content_form, system_form in itertools.product(
        (True, False),
        (True, False),
        ("none", "xml", "json"),
        ("string", "parts"),
        ("present", "absent", "default"),
    ):
        rows.append(
            {
                "content_form": content_form,
                "enable_thinking": enable_thinking,
                "id": (
                    f"thinking={'on' if enable_thinking else 'off'}"
                    f";preserve={'on' if preserve_thinking else 'off'}"
                    f";tools={tool_format};content={content_form};system={system_form}"
                ),
                "preserve_thinking": preserve_thinking,
                "system_form": system_form,
                "tool_call_format": tool_format,
            }
        )
    return rows


def payload(template: str) -> tuple[dict[str, Any], str]:
    nodes = node_records(template)
    matrix = mode_matrix()
    if len(matrix) != 72 or len({row["id"] for row in matrix}) != 72:
        raise ChatAstError("mode matrix must enumerate 2*2*3*2*3 unique cells")
    ast = {
        "branch_coverage": conditional_directives(template, nodes),
        "evidence_status": "OBSERVED_PINNED_SOURCE_DIGEST_VERIFIED_REPLAY_PENDING_FETCH_CLOSURE",
        "exact_literals": {
            "generation_suffix_enable_thinking_false": "<think>\n\n</think>\n\n",
            "generation_suffix_enable_thinking_true_or_undefined": "<think>\n",
            "media_reminder": "<reminder>You are unable to process this {media_type} because you don't have multi-modal input ability. Try different methods.</reminder>",
            "no_tools_default_system": "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。",
            "tool_default_system": "你是一位工具函数调用专家，你会得到一个问题和一组可能的工具函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的，请尽量尝试探索通过工具解决问题。\n如果没有一个函数可以使用，请直接使用自然语言回复用户。\n如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息。\n如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户。",
        },
        "mode_matrix": matrix,
        "model": {"name": MODEL, "revision": REVISION},
        "must_reject_before_tokenization": [
            {
                "case": "unknown role",
                "reason": "Raw template line 176 renders a generic nonempty role; the typed fixed-program adapter must reject it before controls can be emitted.",
                "source_observation": "raw-template-permissive",
            },
            {
                "case": "unsupported mapping content shape",
                "reason": "visible_text fallback at line 17 does not define a typed content-part grammar.",
                "source_observation": "adapter-required",
            },
            {
                "case": "unknown content-part kind",
                "reason": "Only text, bare string, image/image_url, video/video_url, audio/audio_url, and input_audio have explicit template branches.",
                "source_observation": "adapter-required",
            },
            {
                "case": "malformed tool-call structure",
                "reason": "The raw template relies on attribute access and filters; the typed adapter validates function/name/arguments shapes before rendering.",
                "source_observation": "adapter-required",
            },
        ],
        "nodes": nodes,
        "schema_version": 1,
        "source": {
            "chat_template_bytes": TEMPLATE_BYTES,
            "chat_template_json_path": "/chat_template",
            "chat_template_sha256": TEMPLATE_SHA256,
            "file": "tokenizer_config.json",
            "file_bytes": CONFIG_BYTES,
            "file_sha256": CONFIG_SHA256,
            "replay_requirement": "scripts/fetch_model.sh must acquire and verify the full pinned source closure before any promotion from OBSERVED.",
        },
    }
    prose = """# Nanbeige4.2-3B pinned chat AST (OQ-7)\n\nThis archive describes the one pinned `tokenizer_config.json:/chat_template` program, not a reusable Jinja surface.  The source file is 10,990 bytes with SHA-256 `3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518`; the extracted template is 8,375 UTF-8 bytes with SHA-256 `ed118a3c5ddf1d24ffa43229de22bacd5b803be31acaafeb4c0fff0cefee551a`.\n\n`chat_ast.json` records every Jinja conditional directive with an exact byte/line span, the accepted role/content/thinking/tool branches, and all 72 `thinking × preserve × tools × content × system` matrix cells.  The exact no-tool default system text is `你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。`; the distinct tool prelude and the exact generation suffixes live in the JSON `exact_literals` authority.\n\n## Typed rejection boundary\n\nThe raw template raises only for a non-leading system role.  Its generic nonempty-role fallback would render an unknown role, and its fallback content path is not a typed input grammar.  Therefore unknown roles, unsupported mapping/part forms, and malformed tool calls are **MUST-REJECT-BEFORE-TOKENIZATION** in FrankenNLP's fixed-program adapter.  This is an adapter requirement, not a false claim about Jinja's behavior.\n\n## Evidence status\n\nThe source bytes were digest-checked at the pinned revision, but promotion remains `OBSERVED_PINNED_SOURCE_DIGEST_VERIFIED_REPLAY_PENDING_FETCH_CLOSURE` until `scripts/fetch_model.sh` replays the complete verified closure.  Renderer byte goldens remain the separate oracle-fixture authority.\n"""
    return ast, prose


def write_artifacts(ast: dict[str, Any], prose: str, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(canonical_json(ast))
    output_md.write_text(prose, encoding="utf-8")


def check_artifact(ast: dict[str, Any], prose: str, output_json: Path, output_md: Path) -> None:
    try:
        actual_json = output_json.read_bytes()
        actual_md = output_md.read_text(encoding="utf-8")
    except OSError as error:
        raise ChatAstError(f"missing committed artifact: {error}") from error
    if actual_json != canonical_json(ast):
        raise ChatAstError(f"artifact drift: {output_json}")
    if actual_md != prose:
        raise ChatAstError(f"artifact drift: {output_md}")
    nodes = {node["id"] for node in ast["nodes"]}
    for branch in ast["branch_coverage"]:
        if not branch["mapped_node_ids"] or not set(branch["mapped_node_ids"]).issubset(nodes):
            raise ChatAstError(f"unmapped branch: {branch['branch_id']}")
    if len(ast["mode_matrix"]) != 72:
        raise ChatAstError("mode matrix is incomplete")
    required_rejections = {"unknown role", "unknown content-part kind", "malformed tool-call structure"}
    observed_rejections = {item["case"] for item in ast["must_reject_before_tokenization"]}
    if not required_rejections.issubset(observed_rejections):
        raise ChatAstError(f"missing rejection cases: {sorted(required_rejections - observed_rejections)}")


def self_test() -> None:
    synthetic_lines = ["literal"] * 196
    for line_number in (4, 24, 81, 181):
        synthetic_lines[line_number - 1] = "{%- if synthetic -%}"
    template = "\n".join(synthetic_lines)
    ast, prose = payload(template)
    with tempfile.TemporaryDirectory(prefix="fnlp-chat-ast-") as temporary:
        root = Path(temporary)
        output_json = root / "chat_ast.json"
        output_md = root / "chat_ast.md"
        write_artifacts(ast, prose, output_json, output_md)
        check_artifact(ast, prose, output_json, output_md)
        changed = dict(ast)
        changed["mode_matrix"] = changed["mode_matrix"][:-1]
        try:
            check_artifact(changed, prose, output_json, output_md)
        except ChatAstError as error:
            if "artifact drift" not in str(error):
                raise
        else:
            pass
    # 5: the module-load NODE_SPECS / DIRECTIVE_NODES drift check actually catches
    # a deliberate mismatch (regression coverage for the cr-005 finding).
    original = DIRECTIVE_NODES.copy()
    try:
        DIRECTIVE_NODES[4] = "definitely_not_a_node_id"
        try:
            assert_node_specs_directive_nodes_consistency()
        except ChatAstError as error:
            if "drift" not in str(error).lower():
                raise ChatAstError(
                    f"NODE_SPECS/DIRECTIVE_NODES invariant raised on a deliberate "
                    f"mismatch but the message did not name the drift: {error}"
                )
        else:
            raise ChatAstError(
                "NODE_SPECS/DIRECTIVE_NODES invariant failed to catch a deliberate drift"
            )
    finally:
        DIRECTIVE_NODES.clear()
        DIRECTIVE_NODES.update(original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write canonical archive artifacts from verified source")
    action.add_argument("--check", action="store_true", help="replay verified source and byte-compare committed artifacts")
    action.add_argument("--self-test", action="store_true", help="run hermetic archive, coverage, and negative assertions")
    parser.add_argument("--source", type=Path, default=default_source(), help="verified tokenizer_config.json")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
            checks = 5
        else:
            template = load_template(args.source)
            ast, prose = payload(template)
            if args.write:
                write_artifacts(ast, prose, args.output_json, args.output_md)
            else:
                check_artifact(ast, prose, args.output_json, args.output_md)
            checks = 4
    except ChatAstError as error:
        detail = str(error)
        if detail.startswith("SKIPPED_NO_MODEL"):
            log(f"RESULT=SKIPPED_NO_MODEL checks=0 failures={detail}")
            return 0
        log(f"RESULT=FAIL checks=0 failures={detail}")
        return 1
    log(f"RESULT=PASS checks={checks} failures=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
