# Nanbeige4.2-3B pinned chat AST (OQ-7)

This archive describes the one pinned `tokenizer_config.json:/chat_template` program, not a reusable Jinja surface.  The source file is 10,990 bytes with SHA-256 `3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518`; the extracted template is 8,375 UTF-8 bytes with SHA-256 `ed118a3c5ddf1d24ffa43229de22bacd5b803be31acaafeb4c0fff0cefee551a`.

`chat_ast.json` records every Jinja conditional directive with an exact byte/line span, the accepted role/content/thinking/tool branches, and all 72 `thinking × preserve × tools × content × system` matrix cells.  The exact no-tool default system text is `你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。`; the distinct tool prelude and the exact generation suffixes live in the JSON `exact_literals` authority.

## Typed rejection boundary

The raw template raises only for a non-leading system role.  Its generic nonempty-role fallback would render an unknown role, and its fallback content path is not a typed input grammar.  Therefore unknown roles, unsupported mapping/part forms, and malformed tool calls are **MUST-REJECT-BEFORE-TOKENIZATION** in FrankenNLP's fixed-program adapter.  This is an adapter requirement, not a false claim about Jinja's behavior.

## Evidence status

The source bytes were digest-checked at the pinned revision, but promotion remains `OBSERVED_PINNED_SOURCE_DIGEST_VERIFIED_REPLAY_PENDING_FETCH_CLOSURE` until `scripts/fetch_model.sh` replays the complete verified closure.  Renderer byte goldens remain the separate oracle-fixture authority.
