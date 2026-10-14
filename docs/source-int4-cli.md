# Source-candidate INT4 text and chat

`fnlp candidate source-int4 generate|chat` connects the pinned local source
loader, charged resident INT4 model, pinned text planner, complete native INT4
engine and bounded JSON publication. It uses the existing `NlpEngine` runtime
and requires the `asupersync-runtime` feature in the selected build.

This is a **code-first, non-certified source-candidate surface**. It does not
create, read, activate or qualify an INT4 `.fnlpq`, alter the ordinary INT8
commands, download sources, or persist converted weights. Source verification
and in-memory conversion repeat for each command. Canonical INT4 artifact
persistence, model-quality gates and controller validation remain separate.

## Commands

Use the explicit local pinned source directory and its pinned ten-file
conversion-source manifest. A `.fnlpq` path is not a substitute for either.
The existing loader checks source closure, tensor census and tokenizer assets.

```sh
fnlp candidate source-int4 generate prompt.txt \
  --source-dir /models/nanbeige-source \
  --source-manifest /models/pinned-source-manifest.json \
  --memory-mib 8192 --group-width 128 \
  --context-tokens 2048 --max-new-tokens 128

fnlp candidate source-int4 chat conversation.json \
  --source-dir /models/nanbeige-source \
  --source-manifest /models/pinned-source-manifest.json \
  --memory-mib 8192 --group-width 128
```

Omitting the input path, or supplying `-`, reads stdin. `generate` interprets
input as UTF-8 text inside the pinned user-message template, not raw control
tokens. `chat` takes a JSON array of `{ "role": ..., "content": ... }` objects:
optional first system, then alternating user/assistant, ending with user.
Unknown fields, duplicate JSON keys, invalid roles and nonalternating history
are refused. The compiler retains all admitted history; it never repairs or
silently truncates it. Caller control-marker spellings remain ordinary data.

The default is greedy scalar execution with sequential prefill. `--seed`
accepts exactly 64 lowercase hexadecimal digits and enables the existing
addressed seeded sampler. `--temperature-milli`, `--top-k`, `--top-p-ppm`,
`--logprobs`, `--min-new-tokens`, repeated `--stop`, `--ban-token`, repetition,
presence and frequency penalties, and repeated `--logit-bias` are supported.
Stops retain matching bytes. The pinned EOS cannot be banned or replaced.
INT8-only `--prefill-rows` is rejected, never ignored or implemented by a
fallback. This route has no streaming, batch, durable-job or tool-execution mode.

## Admission and delivery

`--memory-mib` is a process ledger ceiling, not a promised OS RSS maximum.
Preflight accounts for the complete group-aware model, source/conversion
staging, tokenizer/request preparation, all allocated KV capacity, native
scratch, sampler, allocator allowances and guarded output. Loading and execution
stages are priced separately because their scratch does not coexist. The host
independently admits the complete runtime-domain aggregate.

`--max-load-mib` bounds the whole weight-plus-conversion-scratch payload;
`--source-preparation-mib` prices retained source metadata/hash workspace;
`--preparation-mib` prices input, pinned tokenizer/planning and serialization.
These are modeled allowances, not measured peak RSS or per-tensor loopholes.
`--load-checkpoints` and `--max-checkpoints` are separate finite stage budgets.
`--timeout-seconds` is a cooperative deadline across input, source loading,
planning and execution. Blocking file reads, hashing and output writes cannot
be preempted; the remaining deadline is checked between stages.

Finite options and raw input/JSON/role structure are checked before loading.
Exact token planning follows conversion because the actual INT4 model content
fingerprint does not exist before then. No synthetic pre-load model identity is
used to claim planning or activation authority.

Successful delivery is one JSON object plus newline with `schema_version: 1`,
`scope: "real-source-current-candidate-int4"`, `evidence: "non_authoritative"`,
source-root identity, `model_content_sha256`, candidate format, quantization
recipe/group width and the complete `Int4ChatResult` under `output`. The model
content fingerprint is explicitly not a canonical artifact digest. Output
retains full group-dot/MAC/attention accounting and native finish semantics.
The model is bound independently again by hosted execution.

Nothing is written to stdout before native completion and task validation.
The preparation and hosted result charges remain alive through serialization,
write and flush. An oversized response or broken pipe is an error, not a
successful partial task. Diagnostics do not interpolate source paths, prompts
or nested arbitrary model/parser errors.

## Validation status

Added tests cover CLI dispatch, required sources, finite and aggregate memory
limits, group-dependent host reservations, unsupported options, pinned EOS,
seed/policy propagation, exact transcript preservation, duplicate/invalid JSON,
feature-disabled refusal before IO and distinct response framing. An explicitly
ignored end-to-end generate/chat test requires `FNLP_INT4_TEST_SOURCE_DIR`,
`FNLP_INT4_TEST_SOURCE_MANIFEST` and `FNLP_INT4_TEST_MEMORY_MIB` plus controller
admission. No Rust compilation/tests, full-model execution, DSR, numerical
qualification or performance measurements were run during this implementation.
