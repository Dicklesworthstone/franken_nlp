# Source-candidate INT4 text and chat

`fnlp candidate source-int4 generate|chat` connects the pinned local source
loader, charged resident INT4 model, pinned text planner, complete native INT4
engine and bounded JSON publication. It uses the existing `NlpEngine` runtime
and requires the `asupersync-runtime` feature in the selected build.

This is a **code-first, non-certified source-candidate surface**. It does not
create, read, activate or qualify an INT4 `.fnlpq`, alter the ordinary INT8
commands or download sources. Explicit local candidate snapshots can preserve
converted weights for later requests without the BF16 sources. They are not
canonical FNLPQ, release activation or publisher authentication. Model-quality
gates, the disputed canonical envelope and controller validation remain separate.

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

`--max-load-mib` bounds the whole weight-plus-load/save-scratch payload;
`--source-preparation-mib` prices retained source metadata/hash workspace during
conversion only (that reserve is not taken during snapshot reload);
`--preparation-mib` prices input, pinned tokenizer/planning and serialization.
These are modeled allowances, not measured peak RSS or per-tensor loopholes.
`--load-checkpoints`, `--snapshot-checkpoints` and `--max-checkpoints` are
separate finite source-conversion, snapshot-save-or-reload, and execution
budgets. Neither persistence nor loading replenishes execution.
`--timeout-seconds` is one cooperative deadline across input, model loading,
planning, optional persistence and execution. Blocking file reads, hashing and output writes cannot
be preempted; the remaining deadline is checked between stages.

Finite options and raw input/JSON/role structure are checked before loading.
Exact token planning follows conversion or verified restoration of the actual
model. No synthetic pre-load model identity or unchecked header field is used
to claim planning or activation authority.

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

## Save once, reload without source conversion

Opt in to saving a converted model by adding `--save-snapshot NEW_FILE`:

```sh
fnlp candidate source-int4 generate prompt.txt \
  --source-dir /models/nanbeige-source \
  --source-manifest /models/pinned-source-manifest.json \
  --save-snapshot /models/nanbeige-g128.int4snap \
  --memory-mib 8192 --group-width 128
```

Retain the successful response's `model_content_sha256` independently, together
with its `group_width`. It is a MODEL-content pin, not a whole-file hash. Use
that value from the trusted conversion on subsequent commands:

```sh
fnlp candidate source-int4 chat conversation.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128
```

Source-directory/manifest mode and snapshot/expected-digest mode are mutually
exclusive. The pin is exactly 64 lowercase hex digits; the group width must
match the original conversion. No original source directory, manifest, network
or conversion is accessed by reload. Missing, corrupted, truncated, appended,
wrong-group or wrong-pin files fail instead of converting a fallback source.
Never obtain the trusted pin solely from the same untrusted file: neither the
header nor a self-consistent checksum authenticates its publisher.

Save happens after successful exact prompt planning and before generation.
The host uses `create_new`, flush and `sync_all`, retaining the resident model
and codec memory charges until physical completion. Existing files and symlinks
are never overwritten. The destination directory must already exist, and the
caller must provision disk space. No file is deleted, automatically retried,
renamed or activated. This is deliberately NOT atomic publication or a directory
crash-durability guarantee. A failed/cancelled save may leave partial or complete
bytes, preserved for inspection; it does not emit a success receipt. A successful
save also remains intact if generation later fails. To save again, choose a new
destination rather than relying on automatic replacement.

When persistence is used, the completed JSON also includes `snapshot` with
`operation: "saved"` or `"loaded"`, `format: "fnlp-int4-local-snapshot-v1"`,
and exact `file_bytes`. No private path is included. Without persistence, this
field is omitted and the existing result framing remains unchanged. The outer
`model_content_sha256` is always derived from verified resident weights.

### Snapshot format and embedding API

`native_engine::strict_int4::weights::snapshot` supplies `Int4SnapshotPin`,
`Int4SnapshotRequirements`, `Int4SnapshotBudget`, and typed errors. Owned model
weights expose `write_snapshot` and `read_snapshot` over bounded `Write`/`Read`.
`NlpEngine::save_int4_snapshot` and `load_int4_snapshot` add the existing charged
resident lifetime, physical-completion host and explicit file operations. Loaded
models use the same `Int4ChatPlanner::for_model` and `execute_int4_chat` as source
conversion. No weights are cloned into a second model and no runtime is created.

The 128-byte header is fixed: bytes 0..8 are `FNLPI4S\0`; 8..12 contain u32le
version 1; 12..16 are zero; 16..24 contain u64le group width; 24..32 contain
u64le complete weight-payload length; 32..64 contain source-root SHA-256 bytes;
64..96 contain model-content SHA-256 bytes; 96..128 are zero. The body has no
padding or file-controlled directories: BF16 embeddings, BF16 final norm,
packed INT4 lm-head with f32le scales, then each physical layer's two BF16 norms
and q/k/v/o/gate/up/down packed matrices with f32le scales. Shapes and ordering
are fixed by the compiled 201-tensor census, not by untrusted dimensions.

Every restored BF16 value must be finite, every group scale positive/finite,
and every nibble within the symmetric7 producer's domain. The raw kernel's
support for -8 does not permit that value under this producer recipe. Exact
EOF and the recomputed existing source/role/shape/recipe-bound model digest must
both pass before any executable weights are returned. Changing persistence
mode does not change the model digest, numerical profile or generation policy.

Memory admission includes the complete compact payload plus one 65,536-byte
codec buffer (allocator, metadata and runtime allowances remain separately
priced). Read IO is bounded by exact file length plus one byte for the EOF
probe. Short and interrupted IO polls cancellation between attempts; individual
blocking calls are not preemptible. At group width 128, the format's computed
file length is 2,965,051,520 bytes, not a measured RSS or performance claim.

## Validation status

Added tests cover CLI dispatch, required sources, finite and aggregate memory
limits, group-dependent host reservations, unsupported options, pinned EOS,
seed/policy propagation, exact transcript preservation, duplicate/invalid JSON,
feature-disabled refusal before IO and distinct response framing. An explicitly
ignored end-to-end generate/chat test requires `FNLP_INT4_TEST_SOURCE_DIR`,
`FNLP_INT4_TEST_SOURCE_MANIFEST` and `FNLP_INT4_TEST_MEMORY_MIB` plus controller
admission. No Rust compilation/tests, full-model execution, DSR, numerical
qualification or performance measurements were run during this implementation.

Snapshot regression sources additionally cover all group partitions, exact
BF16/nibble domains, compact matrix roundtrips, header mutation, truncation,
trailing bytes, external pin mismatch, short/interrupted IO, cancellation,
charged-host failure cleanup and exclusive CLI input modes. An ignored CLI
source-save/reload/generation comparison also requires `FNLP_INT4_TEST_SNAPSHOT`
(an explicit new path preserved after testing). These Rust tests and full-model
runs remain unexecuted; independent Python byte/layout checks and source hash
checks do not constitute Rust, task-quality or performance validation.
