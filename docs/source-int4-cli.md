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
Text tasks and one-off `classify`, `sentiment`, `judge`, `extract`, `ner`,
`keyphrases`, `summarize` and `answer` can opt in to grouped-INT4 layer-major
prompt execution with `--prefill-rows 1..=64`, with separately admitted extra
workspace; see [INT4 prefill](int4-prefill.md). This changes prompt scheduling
while preserving each task's original model identity, work allowance, decoding
policy and completed response. Structured tasks retain their greedy grammar
decoder and independent source finalizers. Composed batch, map, retained-job,
redaction and semantic workflows keep their existing prompt schedules.

Whole-document maps have separate native scheduling choices:
`--cohort-chunks N` groups independent discovery chunks, and
`--cohort-hierarchy-groups N` groups independent evidence passes inside an
explicit summary or answer hierarchy. The latter also applies to their
complete-document jobs, with an authenticated schedule on resume. It requires
the matching synthesis and hierarchy flags and never enables prompt prefill.
See [structured INT4 cohorts](int4-structured-cohorts.md#native-hierarchy-group-cohorts)
for allowed widths, complete memory admission and examples.

### Persistent CPU teams for complete source documents

`ner-map`, `keyphrases-map`, `summarize-map` and `answer-map` accept
`--compute-threads N` for `N` from 1 through 64, including explicit neural
summary/answer synthesis and their hierarchies. The default of 1 retains serial
execution. Larger widths form one fixed numerical team spanning all discovery
chunks, the exact reducer or answer collection, and every dependent neural
evidence pass selected by the command. Each
projection divides complete output rows; the reduction order within each row
stays unchanged. Each chunk retains
its own causal context, grammar and original-source evidence.

```sh
fnlp candidate source-int4 summarize-map report.txt \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 8192 --compute-threads 4

fnlp candidate source-int4 answer-map report.txt --question question.txt \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 8192 --compute-threads 4 --input-encoding safe-bpe-v1
```

The compute width counts the coordinator and its row workers. The runtime
thread is additional. Worker scratch is priced once before private input or
model IO and remains owned through physical worker join. Native work, masks,
source verification, final top-k limits and the whole-document deadline keep
their existing bounds. Both byte fallback and compact Safe-BPE input use the
same team path. Successful threaded results add a separate `compute` receipt
with the actual completed projections and rows; the semantic result and its
execution identity do not absorb worker width. The guarded result remains
owned through serialization, write and flush.

QA keeps the same private question across every nonblank chunk, preserves
interior whitespace ranges, and retains all independently validated answers
and abstentions. Selecting a worker width does not itself enable synthesis.
The existing `--synthesize-answer` or `--synthesize-summary` switch controls
that additional native pass; the hierarchy switches explicitly enable bounded
dependent quote-selection levels. Those modes reuse one team from discovery
through the last evidence pass. They retain their original complete work,
mask, context, evidence and maximum-pass bounds without renewed allowances or
per-pass worker scratch. Generated prose still cannot become source evidence.
Neural synthesis remains byte-only; compact independent QA retains one
cumulative encoding allowance across question sizing, source probes, CLI
preflight and hosted replanning.

Library callers use `NlpEngine::map_int4_source_with_threads`,
`summarize_int4_source_with_threads`,
`reduce_int4_keyphrases_document_with_threads`, or
`answer_int4_document_with_threads` on the same resource host as
their resident model. Neural callers use
`synthesize_int4_source_summary_with_threads`,
`synthesize_int4_document_answer_with_threads`, or their
`_hierarchy_with_threads` counterparts. Each returns
`HostedInt4TeamResult<T>`, retaining the
ordinary result owner alongside the physical compute receipt. The serial and
threaded routes share admission, complete-map preparation, source/result seals
and reducer validation. The installed process envelope must admit the requested
worker width; it never silently creates a larger host or falls back to another
width.

These numerical teams currently require serial chunk scheduling. Combining
them with `--cohort-chunks`, `--cohort-hierarchy-groups`, or packed prompt
prefill is refused. Corpus and durable-job commands retain their existing
schedules. Regression definitions cover admission, unsupported combinations,
cancellation, charge release and serial/team model parity. Executable and
model-present checks remain pending the controller's DSR gates; no measured
speedup or parity result is claimed here.

### Compact input for complete source maps

`ner-map`, `keyphrases-map`, plain cited `summarize-map` and independent
`answer-map` accept explicit `--input-encoding safe-bpe-v1`. This uses the pinned byte-preserving encoder for
every real chunk, including exact keyphrase/summary reducers and
`--cohort-chunks`. Actual encoded token counts determine chunk capacity,
compiled prompts and native work. Source evidence continues to use the exact
original UTF-8 bytes and their original byte/scalar coordinates. The default
remains byte fallback; compact input is a distinct candidate input policy and
does not establish tokenizer-oracle parity or task-quality equivalence.

```sh
fnlp candidate source-int4 ner-map report.txt \
  --source-dir /models/nanbeige-source \
  --source-manifest /models/pinned-source-manifest.json \
  --memory-mib 8192 --input-encoding safe-bpe-v1 \
  --max-chunk-bytes 8192 --cohort-chunks 2
```

One nonrenewable tokenizer ledger covers every chunk-size probe, discarded
probe, actual plan and hosted repeat preflight. A failed or cancelled encoding
keeps its incurred work. Input/output totals default to eight times
`--max-input-bytes`; `--max-total-tokenizer-input-bytes` and
`--max-total-tokenizer-output-tokens` select explicit ceilings. The complete
invocation defaults to 268,435,456 pair checks and 4,294,967,296 examined pair
bytes, selected by `--max-total-tokenizer-pair-checks` and
`--max-total-tokenizer-pair-bytes`. Existing `--max-tokenizer-pair-checks` and
`--max-tokenizer-pair-bytes` additionally cap each encoding. Exceeding any
ceiling refuses the complete document. No retry renews those allowances.

The per-call tokenizer workspace is included in preparation admission before
private input or model IO. Policy and every configured ceiling bind the
template identity; successful compact responses add an `input_encoding`
object containing the policy, complete-operation limits and total work used.
Byte-default response JSON keeps its existing shape. Internal source-map
commitments additionally bind the exact source, options, all map limits and
ordered native children. Completion seals reject substituted or changed
payloads; exact reducer boundaries also bind every selected aggregation and
final top-k limit. These process-private bindings are absent from JSON and
default diagnostics.

Library callers use `SourceTaskPlanner::with_map_document_encoding(policy,
complete_limits)` once for the complete operation, retain that same planner
through preflight/replanning, and admit
`map_document_encoding_workspace_bytes()`. The one-request
`with_document_encoding` constructor alone does not authorize repeated map
probes. Neural summary/answer synthesis and hierarchies continue to require
their existing byte-only policy. Plain source corpora and complete-document
jobs use explicitly configured population factories with per-document ledgers,
as described below.

#### Compact complete-document QA

`answer-map` retains every independent answer, abstention and whitespace
interval across the complete source. Its compact policy encodes the exact
question, canonical passage manifest and original chunk source under one
per-call allowance. That allowance is bounded by `--max-input-bytes` and the
encoder's one-MiB ceiling; it is separate from the complete document's byte
limit. Capacity reserves the question and passage metadata before partitioning
source bytes, and uses actual encoded token counts for context and native work.
A question may therefore contain more UTF-8 bytes than the token ceiling while
still fitting its actual compact prompt. Oversized questions or source
segments are refused instead of truncated.

```sh
fnlp candidate source-int4 answer-map report.txt --question question.txt \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 16384 --input-encoding safe-bpe-v1 \
  --max-chunk-bytes 8192 --cohort-chunks 2
```

One cumulative ledger covers repeated question/manifest encodings, every
discarded chunk-size probe, complete CLI preflight and hosted recompilation.
Serial, chunk-cohort and persistent-team QA share that same planner; neither a
new chunk nor a host boundary resets its allowance. The selected per-call policy and all total
ceilings enter execution identity, and compact output reports consumed work.
The default policy preserves the existing response shape.

Question-answer completion checks now privately bind the exact source,
question, answer options, model/task authority, planning and map limits, and
source partition. A second seal covers the completed public payload. Equal
lengths, equal work estimates or identical source geometry cannot substitute
for the admitted request, and modifying a completed answer or its counters
invalidates its seal. These commitments never enter serialized responses or
Debug output. Bounded streaming hashing avoids allocating a second
document-sized serialization and checks cancellation during long writes.
Hosted and CLI completion checks also use the invocation's existing control
while rehashing the result and traversing its accounting and source coordinates.

The independent source-citation verifier continues to use original UTF-8 byte
and Unicode-scalar coordinates. Encoding literal role/tool marker text never
grants it template-control authority. Structural citation validity does not
certify semantic entailment or answer quality. Compact QA currently supports
independent passage answers, native chunk cohorts and persistent numerical
teams. Combining compact encoding with neural synthesis or hierarchy is
refused before private IO. Independent QA also supports fixed-question corpora,
per-record-question corpora and retained complete-document jobs through the
population policy below. Numerical teams and chunk cohorts cannot be combined
in one call.

#### Compact source corpora and retained complete-document jobs

The same explicit `--input-encoding safe-bpe-v1` policy is available on
`ner-corpus`, `keyphrases-corpus`, `summarize-corpus`, and
`source-corpus ner|keyphrases|summarize|answer`. The fixed-question
`source-corpus answer` route keeps its existing question file; `answer-corpus`
keeps each record's private question. Plain independent maps, exact keyphrase
and cited-summary reducers, and their existing native chunk cohorts are
supported. Neural synthesis and hierarchies remain byte-only.

```sh
fnlp candidate source-int4 ner-corpus documents.ndjson \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 16384 --input-encoding safe-bpe-v1 \
  --max-chunk-bytes 8192 --cohort-chunks 2
```

Each complete document operation owns one fresh tokenizer ledger. The
prepared record retains that exact planner through execution reconstruction,
so rejected probes, complete preflight and final compilation all consume the
same allowance. Immutable tokenizer and control-token assets are shared.
A chunk or host boundary cannot renew the document's budget; a new document
gets its own explicitly admitted scope. The four `--max-total-tokenizer-*`
ceilings apply to that scope. Existing stream record, input, native-work,
mask and verification limits continue to bound the complete stream across
documents and flush controls. Encoder workspace is included in preparation
admission before population IO and reused by successive documents.

`document-job start|resume ner|keyphrases|summarize|answer` supports the same
plain modes, including raw `summarize --map-only`.
`answer-document-job start|resume` supports independent answers with per-record
questions. The private retained recipe and execution identity authenticate
the exact encoding policy and every per-document ceiling. Repeat the selected
flags on resume; changing them is refused before results can be mixed.

Tokenizer usage is **not** persisted as a lifetime counter. An explicit resume
can create a fresh document-operation ledger under the same authenticated
limits. Preparation occurs before the existing durable native-work/attempt
debit, so failed preparation does not consume a persisted attempt. This policy
does not replace or renew the existing durable native-work, mask or persisted
attempt accounting. Neither chunks nor partially prepared documents become
durable results.

Compact population provenance adds `input_encoding` with the policy,
`complete_document_limits`, and
`budget_scope: "complete_document_operation"`. It does not report the unused
factory ledger as consumed work. Byte-default frames and stored semantic
document shapes remain unchanged. Library callers construct the existing
byte-default source/question batch or job factory, then call
`with_map_document_encoding(policy, complete_limits)` before selecting its
native cohort schedule. The factory validates supported task modes and
creates a private planner for each original document.

Added regression definitions cover actual compact prompt work, Unicode and
literal-control preservation, per-call and cumulative exhaustion, cancellation,
host workspace admission, unchanged default output, request/payload substitution
and every exact reducer limit. Executable Rust regressions, native-model
quality/parity and performance remain pending the controller's DSR checkpoint
and model-present gates.

Complete-document `sentiment-map` also accepts `--input-encoding safe-bpe-v1`
for serial, chunk-cohort and persistent-team execution. Actual encoded source
tokens determine chunk boundaries. One shared ledger for input bytes, output
tokens, pair checks and pair bytes covers rejected probes and final plans;
each chunk's encoding is reused across sentiment axes.
`--max-tokenizer-input-bytes` and `--max-tokenizer-output-tokens` cap cumulative
encoder work separately from source and native-work limits. The byte-default
response stays unchanged, and sentiment corpora and durable jobs remain
byte-only. Compact execution accepts pinned sources or a digest-pinned snapshot
reload; combining it with `--save-snapshot` is refused.

Entity `resolve` and resident `resolve-corpus` have independent native schedules:
`--cohort-pairs 1..=32` keeps both orders of each candidate pair together, while
`--cohort-chunks 1..=64` batches NER discovery with `--discover-entities`.
Discovery windows can cross original document boundaries, with or without
`--chunked`. These schedules preserve whole-snapshot and whole-invocation work
ceilings and use shared host admission for the complete phase peak. See
[native INT4 entity cohorts](native-entity-cohorts-int4.md) for examples and
semantic versus physical output fields.

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
