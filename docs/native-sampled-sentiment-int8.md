# Native INT8 sampled sentiment

`fnlp candidate sentiment-sampled` generates independent score-and-explanation
samples on an explicitly selected local current-candidate INT8 artifact. It
implements the independent generation part of comprehensive plan §7.5, Mode B,
for valence, arousal, dominance and approach. Every sample produces an integer
score in `[-100,100]` and a bounded, nonempty explanation.

This candidate implementation requires `asupersync-runtime`. Compilation,
executable regressions, real-model replay, task quality and release qualification
remain separate pending controller gates.

## Request and invocation

Use a UTF-8 JSON file such as `sentiment-request.json`:

```json
{
  "document": "The support agent fixed the problem quickly, but the repeated failures were frustrating.",
  "axes": ["valence", "arousal"]
}
```

Omit `axes` to request all four dimensions. Empty documents, empty or repeated
axes, duplicate JSON keys and unknown fields are rejected. The positional input
`-` reads stdin.

```sh
fnlp candidate sentiment-sampled sentiment-request.json \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 12288 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 \
  --temperature-milli 1000 \
  --top-p-ppm 950000 \
  --max-new-tokens 256
```

The command does not discover or download model weights. Its held model source
supplies authenticated artifact facts for planning and the subsequent weight
load. The memory value is a declared process ledger; actual requirements depend
on artifact, context, preparation and output limits.

The seed must contain exactly 64 lowercase hexadecimal digits. Every axis and
sample has an independent semantic address. Reusing the same model, input
encoding, source, task policy, native profile and seed preserves those addresses.
Explicit `--top-k N` limits the legal token distribution before nucleus
selection; omitting it retains the complete legal set for nucleus processing.
All logits are validated before the exact grammar/control mask, temperature,
top-k and top-p policy selects a token. Terminal EOS is scored and can be
selected only after a complete accepting JSON value.

## Input and prompt scheduling

Byte-preserving control-excluding encoding is the default. For compact input,
add `--input-encoding safe-bpe-v1`; the existing finite tokenizer work options
also apply. This policy preserves original UTF-8 bytes and excludes template
control tokens. It changes the token sequence and can change model output.

The source, axis prompt and generated sample must fit one admitted native
context. This command rejects an over-context request instead of truncating
or chunking the source. By default each sample starts with fresh logical KV.

Add `--prefill-rows 8` for bounded layer-major prompt execution. Widths from
1 through 64 are accepted, with the declared extra prompt workspace charged
before input or model IO. Prompt width changes scheduling and does not enter the
semantic draw address. Model-present output parity and performance still require
qualification.

## Reusing prompts across independent samples

Add `--reuse-prefill` to `sentiment-sampled` or
`sentiment-sampled-batch` to avoid repeating the complete prompt forward for
every sample of an axis. The first sample prefills normally. Before each later
sample, the engine retains the same session's KV prefix through the penultimate
prompt token, discards the previous continuation, invalidates the final hidden
state, and recomputes the final prompt token. Every layer still performs that
causal final-token forward and the complete vocabulary projection.

```sh
fnlp candidate sentiment-sampled sentiment-request.json \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 12288 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 --reuse-prefill --compute-threads 4 --prefill-rows 8
```

The option composes with bounded layer-major prefill and the persistent
numerical team. It preserves the original seed, semantic sample addresses,
prompt identity, grammar, EOS and sampling policy. Reuse stays inside the
current record and axis; completing or abandoning that session clears its
logical KV. No prompt state is copied into a cross-request cache or persisted.

Admission still reserves the complete cold work ceiling and full native KV.
A separately priced bounded array holds the independent sample addresses.
The result uses a distinct execution label and records the actual work for
each sample and for the complete bundle. A later sample therefore reports
only its replayed final prompt token and continuation work. Early completion
does not replenish admitted work or grammar-mask allowances.

Hosted callers use
`execute_int8_sampled_sentiment_reusing_prefill` or
`execute_int8_sampled_sentiment_reusing_prefill_with_threads`, both with an
optional `Int8PrefillLimits`. Resident single-context populations use
`batch_int8_sampled_sentiment_reusing_prefill` and its
`_with_threads` variant. Lower-level prepared callers can quote
`reuse_sampler_requirement_bytes()` and invoke
`execute_reusing_prefill_with_control`.

An explicitly armed model gate compares scores, explanations, descriptive
aggregates and token counts against cold execution across serial, team and
prefill schedules, while separately checking reduced native forwards and
complete resource drain. That gate and the deterministic regression definitions
remain unrun pending controller validation; there is no measured speed or
numerical-parity claim.

## Complete-result semantics and limits

One successful response contains every requested sample and, for each axis,
its mean, median, minimum, maximum and population standard deviation. Individual
scores, explanations, generated-token counts and native work remain available.
The response names its execution profile, sampler, aggregation recipe, public
prompt-template hash and current-candidate artifact provenance. It does not
publish the effective seed or private prompt/content commitments.

The aggregates describe variation among generated scores. They do not establish
calibrated confidence, psychological measurement or human-label accuracy.
Explanations are unverified model output from an untrusted source document.

All axes and samples are admitted together before model weights load. One
nonrenewable set of limits covers the complete bundle:

| Limit | Scope |
| --- | --- |
| `--max-new-tokens` | Each sample, including terminal EOS |
| `--max-justification-chars` | Unicode scalar count in each explanation |
| `--max-sample-json-bytes` | One generated score/explanation JSON value |
| `--max-result-bytes` | Complete task result, including every sample and aggregate |
| `--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs` | Complete native bundle |
| `--max-dot-products`, `--max-multiply-accumulates` | Complete native projection work |
| `--mask-step-node-visits` | One complete grammar-mask traversal |
| `--max-mask-node-visits` | All samples and scored EOS steps |
| `--timeout-seconds` | Preparation, model load, execution and final delivery |
| `--max-checkpoints` | Shared native execution control across all axes and samples |

Admission includes full resident KV, sampler storage, caller-owned preparation,
transferred plans/vocabulary, retained output and optional prompt workspace.
Independent samples reuse one native context while their execution work is
summed. Failed attempts do not produce neutral scores or partial successful
bundles. A failed final output write or flush remains a failure; callers must
also check the exit status because arbitrary stdout cannot retract delivered
bytes.

## Library entrypoints

Use `tasks::sentiment::sampled::quantized::Int8SampledSentimentPlanner::for_model`
or `for_model_with_encoding` with the authenticated `ArtifactIdentity`. Planning
seals the actual model/profile, exact prompts, grammar, policy, complete work and
output bounds into `PreparedInt8SampledSentiment`.

`NlpEngine::execute_int8_sampled_sentiment` accepts the resident `&ResidentInt8`,
that prepared value, a pinned `Arc<SampledSentimentVocabulary>`,
`Int8SampledSentimentHostLimits` and cancellation.
The `execute_int8_sampled_sentiment_layer_major` variant additionally takes
`Int8PrefillLimits`. The hosted path checks the resident resource domain, keeps
native and preparation charges through physical completion and returns a guarded
`Int8SampledSentimentRun` whose output charge survives delivery.

Lower-level callers can use `decode_json_int8_sampled` and
`decode_json_int8_sampled_layer_major` from
`native_engine::constrained_int8`. These retain the same exact JSON/source
grammar driver, native work reconciliation, explicit EOS and abort behavior
as the greedy decoder.

## Ordered resident corpora

Use `fnlp candidate sentiment-sampled-batch` to keep the authenticated INT8
model, vocabulary, compiler and native buffers resident across records. Each
text fits a single context. For example, save the following as `reviews.ndjson`.

```jsonl
{"id":"review-42","text":"The replacement arrived early.","task_args":{"axes":["valence"]}}
{"id":"review-43","text":"Still waiting for a response."}
{"flush":true}
```

```sh
fnlp candidate sentiment-sampled-batch reviews.ndjson \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 12288 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 2 --max-requests 100 --max-input-mib 16 --max-output-mib 32
```

Omitting task arguments requests all four axes. Each stable ID participates in
the private sampling address along with the model and request. Reordering
distinct items or inserting a flush preserves their addresses. An optional
complete `task_args.budget` can tighten the task limits for that item.

The five native work limits and the mask-node limit cover the entire run,
including admitted attempts that fail. A flush starts a fresh ID epoch. It
does not replenish work. Compact-tokenizer limits apply to each record.
A successful item includes every requested sample. Inspect the final
`run_complete` and process status to distinguish full success from a stream
containing rejected records or an interrupted run.

Library callers use `NlpEngine::batch_int8_sampled_sentiment` with the
model-bound planner, pinned vocabulary, `SampledSentimentCorpusConfig`,
`CorpusLimits`, optional `Int8PrefillLimits` and owned input/output streams.

## Durable sampled populations

Use `fnlp candidate sentiment-sampled-job start` and `resume` to retain
complete sampled bundles and continue interrupted work. This route requires
`asupersync-runtime` and `metadata-store` on the existing Linux x86-64 or
AArch64 owned-storage profile. Each input document must fit one native context.
The separate complete-document map command below remains the route for a single
long source.

Input is a complete NDJSON population with globally unique stable IDs:

```jsonl
{"id":"review-42","text":"The replacement arrived early.","task_args":{"axes":["valence"]}}
{"id":"review-43","text":"Still waiting for a response."}
```

Omitted axes select all four dimensions. Per-record arguments may tighten the
task budget. Live-stream flush controls are not accepted. Ordinary resume needs
the complete original ordered population, including the exact envelopes, even
when some items have already committed.

### Start and resume

Prepare a fresh protected directory, a random job ID, a protected file containing
exactly 32 raw secret bytes, and a finite `JobLimits` JSON file as described in
[owned-job setup](candidate-owned-jobs-cli.md#explicit-retention-and-setup).
The examples use those existing paths and `JOB_ID`. The key authenticates
stored state; it does not encrypt the private result spool. Neither the key nor
the job ID is generated implicitly by this command.

```sh
fnlp candidate sentiment-sampled-job start reviews.ndjson \
  --model ./model.fnlpq --memory-mib 16384 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --max-input-bytes 1048576 \
  --samples 2 --compute-threads 4 --prefill-rows 8 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key \
  --limits ./job-limits.json
```

After an interruption, supply the same semantic options, protected job, key,
limits and population with `resume`:

```sh
fnlp candidate sentiment-sampled-job resume reviews.ndjson \
  --model ./model.fnlpq --memory-mib 16384 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --max-input-bytes 1048576 \
  --samples 2 --compute-threads 2 --prefill-rows 8 \
  --store-results --job-dir ./protected-job \
  --job-id "$JOB_ID" --key-file ./protected.key \
  --limits ./job-limits.json --materialize
```

Already committed bundles are not generated again. Each pending attempt reserves
its complete native and grammar-mask work in the authenticated journal before
inference. Failed or interrupted attempts retain their charges across resume.
Choose retry and work headroom in the original limits; resume cannot enlarge
the lifetime contract.

The private recipe binds the actual model and execution identity, effective
seed, sample count, temperature, top-k/top-p, axis defaults, prompt and grammar
versions, input encoding, compiler and task limits, work ceilings and prefill
policy. Changes are refused before pending inference. Worker count can change
because it is a physical resource choice. Keep `--prefill-rows` unchanged for
a retained job, even though it does not change semantic draw addresses.

Stdout contains a bounded metadata report. Private scores, explanations and
generated tokens stay in the protected result spool. Explicit `--materialize`
publishes the verified ordered `materialized.ndjson` only after the population
completes; existing model-free `fnlp job status`, `verify` and `materialize`
commands remain available. Failure to deliver the final metadata report does
not undo already committed results.

### Explicit input retention

To store the exact private input envelopes, add
`--spool-input --max-retained-input-mib N` to `start`. Without that option,
keep the original population yourself. To resume from a previously authenticated
input spool, use `resume --use-retained-input --max-retained-input-mib N` and
omit the positional population argument. This mode never reads a replacement
file or stdin, and it does not fall back to an external population if retained
inputs are missing or corrupt.

Both resume modes authenticate the entire snapshot, original recipe and committed
result frames. An uncommitted tail is refused by default. The separate explicit
`resume --discard-uncommitted-tail` option permits cleanup of uncommitted storage
only after that authentication succeeds.

### Durable limits and memory

The five `--max-*` native-work options and `--max-mask-node-visits` bound one
sampled item on this command. `JobLimits.max_work` is the independent lifetime
allowance for all records and failed attempts. Its mask allowance must be
positive, and its result limit must cover the configured complete sampled bundle.
The limits file also bounds item count, attempts, snapshot bytes, spool,
materialization and journal storage.

`--max-stream-mib` and `--max-input-lines` bound input transport for an invocation.
`--journal-memory-mib` and `--serialization-memory-mib` reserve database and
serialization memory separately from disk ceilings. Memory admission includes
the actual sampler, full resident KV, both preparation owners, population or
retained-record workspace, guarded complete output and optional workers/prefill
buffers before model weights or population IO.

With `--input-encoding safe-bpe-v1`, one shared tokenizer-work allowance covers
the population and failed preparation attempts in an invocation. The existing
`--max-total-tokenizer-input-bytes`, `--max-total-tokenizer-output-tokens`,
`--max-total-tokenizer-pair-checks` and `--max-total-tokenizer-pair-bytes`
options set those totals. Per-record input-encoding ceilings still apply.
The total tokenizer policy is part of the authenticated recipe.

Library callers create
`jobs::runner::sampled_sentiment::quantized::Int8SampledSentimentJobPlanner`
from the actual model-bound compiler, pinned vocabulary, native/mask limits and
`JobLimits`. Its explicit document-encoding constructor accepts the shared
tokenizer-work limits, and `with_layer_major_prefill` freezes prompt scheduling.
Hosted callers pass the immutable factory to
`NlpEngine::job_int8_sampled_sentiment` or
`NlpEngine::job_int8_sampled_sentiment_with_threads`.
The public `int8_sampled_sentiment_job_requirements` quote exposes the admitted
memory envelope before IO. Planner, vocabulary, input and native claims stay
owned through physical worker join; guarded output stays charged through
durable publication.

These additions include regression definitions for recipe authentication,
resource limits, cancellation, retained replay and committed-result reuse.
Compilation, executable recovery tests and real-model replay remain pending
the repository's controller DSR gates.

## Complete-document sampled maps

Use `fnlp candidate sentiment-sampled-map` when the complete source exceeds one
context. It accepts the same strict `{document,axes?}` JSON request as the
single-context command. It partitions the original source into contiguous,
nonoverlapping chunks that each fit every requested axis prompt, then generates
the complete independent sample bundle for every chunk.

```sh
fnlp candidate sentiment-sampled-map sentiment-request.json \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 12288 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 --max-chunks 64 \
  --compute-threads 4 --prefill-rows 8
```

The memory value is an example declared cap. The command checks its complete
memory quote and refuses an insufficient cap. Worker and prefill settings are
optional; a serial document uses the same sample addressing and reduction
recipe. One persistent numerical team can serve all chunks, axes and samples.

Partitioning preserves original UTF-8 bytes, CRLF sequences and source offsets.
The response retains every chunk's original byte and Unicode scalar spans,
every score and explanation, the encoding policy and cumulative work. Equal
chunk text at different source positions receives distinct sample addresses.
Replaying the same model, document, seed and task policy preserves those
addresses. This establishes reproducible draw addressing; executable numerical
parity across schedules remains a separate model-present gate.

### Document replicates and aggregates

For each axis and sample ordinal, the reducer pairs that ordinal across all
chunks and weights the chunk scores by their original Unicode scalar counts.
It retains the checked exact integer numerator and complete source-scalar
denominator for each document replicate, alongside the resulting score.
Mean, median, minimum, maximum and population standard deviation summarize
these complete document replicates.

The reducer shares its recipe with the existing INT4 sampled document path.
These statistics describe the sampled chunk-based procedure. They do not
establish calibrated confidence or equivalence to asking the model about the
entire document in one context. Every explanation remains attributable to its
own source chunk.

### Whole-document limits and completion

The planner preflights every chunk, axis and sample before the first model
forward. No successful document omits a source tail, failed chunk or incomplete
sample. One end-to-end cooperative deadline covers input, model load, planning,
execution, reduction and publication; blocking IO itself is not preempted.

| Option | Document behavior |
| --- | --- |
| `--max-input-bytes` | Complete input envelope; defaults to 1 MiB |
| `--max-chunks` | Maximum complete source partition; defaults to 64, maximum 256 |
| `--max-chunk-bytes` | Optional additional source-byte ceiling for each chunk |
| `--max-tokenizer-calls` | Shared source-sizing probes, including rejected candidates; defaults to 8,192 |
| `--max-new-tokens` | Each sample, including scored terminal EOS |
| `--max-result-bytes` | Complete sample bundle for one chunk |
| `--max-retained-result-bytes` | Cumulative canonical bytes of all retained chunk bundles; defaults to 64 MiB |
| `--max-map-result-bytes` | Complete map JSON, including chunks and aggregates; defaults to 16 MiB |
| Native work and mask options listed above | One nonrenewable allowance across every chunk, axis and sample |
| `--timeout-seconds`, `--max-checkpoints` | Shared complete-document deadline and cooperative control |

Byte fallback remains the default. With explicit
`--input-encoding safe-bpe-v1`, one complete-document tokenizer-work ledger
covers accepted and rejected sizing probes and the final chunk plans. No
tokenizer, native, mask or retained-output allowance renews at a chunk boundary.

The CLI emits one completed response only after full result verification.
Typed cancellation, work exhaustion, partition failure or invalid model output
prevents complete-document success. A transport failure during final stdout
publication can still leave partial bytes, so callers must check process status.

### Complete-document library APIs

`tasks::sentiment::sampled::quantized::long` defines
`Int8SampledSentimentMapLimits`, `PreparedInt8SampledSentimentMap` and
`Int8SampledSentimentMapRun`. Use
`Int8SampledSentimentPlanner::plan_document_with_control` to seal a complete
document plan. The explicit encoding-work and stable-item variants support
callers that manage tokenizer work or document identities themselves.

`PreparedInt8SampledSentimentMap::execute_scheduled_with_control` accepts the
strict INT8 engine, pinned sample vocabulary, complete `Int8JsonBudget`, sampler
allowance, optional `Int8PrefillLimits` and caller control. The result identifies
the execution as `portable-int8-complete-sampled-sentiment-map-v1`.

Hosted callers use `Int8SampledSentimentMapHostConfig` with:

- `NlpEngine::map_int8_sampled_sentiment` for byte-preserving serial execution.
- `NlpEngine::map_int8_sampled_sentiment_with_encoding` for an explicit encoding.
- `NlpEngine::map_int8_sampled_sentiment_with_threads` for one persistent team.
- `NlpEngine::map_int8_sampled_sentiment_with_encoding_and_threads` to combine
  explicit encoding and a persistent team.

The config's `requirements` and `requirements_with_encoding` methods provide
pure memory quotes before model or source IO. Preparation, vocabulary, plans,
KV, scratch and guarded output retain their charges through physical worker
drain. Optional prefill workspace is charged once; the dispatcher separately
charges the numerical workers.

This new route has source-reviewed task, hosted and CLI regression definitions,
including an explicitly armed real-model comparison gate. Compilation, executed
regressions, numerical parity, quality and performance remain pending controller
validation. The map command processes one complete document per invocation.

## Complete-document resident populations

Use `fnlp candidate sentiment-sampled-corpus` to process ordered NDJSON
documents while keeping one INT8 model, compiler, vocabulary and native context
resident. It uses the same lossless complete-document estimator as
`sentiment-sampled-map`. Each successful record contains every chunk,
sample and explanation with original source spans.

```jsonl
{"id":"document-42","text":"The complete first document...","task_args":{"axes":["valence","arousal"]}}
{"id":"document-43","text":"The complete second document..."}
{"flush":true}
```

```sh
fnlp candidate sentiment-sampled-corpus documents.ndjson \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 65536 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 --max-chunks 64 \
  --max-forward-positions 1048576 \
  --max-document-forward-positions 131072 \
  --max-requests 100 --max-input-mib 64 --max-output-mib 128 \
  --compute-threads 4 --prefill-rows 8
```

The memory value is an example ceiling; the independent CLI and hosted quotes
must both fit it before input or model IO. Source length, chunk count, retained
results, output envelopes and physical scheduling all affect that quote.

Stable IDs participate in sample addressing along with the actual source,
chunk coordinates and task policy. Reordering records or adding flush controls
preserves those addresses. A flush begins a new duplicate-ID epoch without
renewing work. Empty or malformed records and failed documents remain visible
in the ordered protocol; any failed item makes the process exit nonzero.
The terminal completion frame is published only after physical worker join.

The scopes of the limits are explicit:

| Options | Resident population scope |
| --- | --- |
| Five `--max-*` native work flags | One nonrenewable allowance across all records and admitted failures |
| `--max-mask-node-visits` | One nonrenewable grammar-mask allowance across the population |
| `--max-document-forward-positions`, `--max-document-projected-logits`, `--max-document-attention-pairs` | Optional complete-document limits, each at most its population allowance |
| `--max-document-dot-products`, `--max-document-multiply-accumulates`, `--max-document-mask-node-visits` | Optional complete-document projection and mask limits |
| `--max-input-bytes`, chunk and map-result flags | Each complete original document |
| `--max-requests`, `--max-input-mib`, `--max-output-mib`, `--max-line-bytes` | Input framing and complete candidate-framed output |
| Compact tokenizer work flags | One live invocation ledger across all records, accepted/rejected probes and failed preparations |
| Deadline and checkpoint flags | One end-to-end population invocation |

If a document limit is omitted, it equals that dimension's population ceiling.
An optional complete `task_args.budget` can further tighten a record.
Native work is conservatively charged before execution and is not refunded
after early EOS, output failure or unsuccessful admission. A failed chunk never
produces a successful partial-document result.

Library users have `Int8SampledSentimentDocumentBatchHostConfig`, its pure
`requirements()` and `requirements_with_threads(width)` quotes, and
`NlpEngine::batch_int8_sampled_sentiment_documents` with its
`_with_threads` variant. Width one uses the serial quote; the threaded
quote accepts widths two through 64. The lower-level sealed planner and
processor live in `batch::sentiment::sampled::quantized::long`.

## Durable complete-document start and resume

`fnlp candidate sentiment-sampled-document-job start|resume` applies
the same complete-document estimator to an authenticated, retained population.
It requires `metadata-store` and `asupersync-runtime` on Linux
x86-64 or AArch64. The explicit storage contract uses an existing protected job
directory, expected job ID, protected file containing a 32-byte raw secret,
and immutable `JobLimits` JSON.

```sh
fnlp candidate sentiment-sampled-document-job start documents.ndjson \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 65536 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --store-results --job-dir /path/to/protected-job \
  --job-id 0123456789abcdef0123456789abcdef \
  --key-file /path/to/raw-32-byte.key \
  --limits /path/to/immutable-job-limits.json \
  --samples 4 --max-chunks 64 --prefill-rows 8 --compute-threads 4
```

For ordinary resume, repeat those semantic and storage arguments with
`resume` and the complete original NDJSON population. Durable
populations accept complete records and do not use flush controls. Committed
results are authenticated and reused; pending work resumes under the remaining
lifetime allowances.

The private recipe binds the actual model identity, seed and sampling policy,
prompt/grammar/compiler policy, exact source population, complete-document
mapping and output limits, encoding policy, and prefill schedule. Changed
recipes or populations refuse resume before pending inference or tail repair.
Worker count is physical scheduling metadata and may change on resume.
Optional `--discard-uncommitted-tail` authorizes repair only after
authentication.

The five native counters and mask visits in the immutable job limits are
copied exactly as lifetime ceilings. They are never multiplied by records,
chunks, sample counts or attempts, and do not renew on restart. CLI native and
mask flags bound one complete document and must fit those lifetime ceilings.
Failed-attempt debits remain spent. Compact tokenizer work is cumulative within
each invocation; its counters are not added to the journal's six persisted
native/mask dimensions.

Retaining input requires an independent opt-in:

- At start, add `--spool-input --max-retained-input-mib N`.
- On retained resume, omit the population positional argument and use
  `--use-retained-input --max-retained-input-mib N` with the original cap.
- `--store-results` remains required. `--materialize` publishes the
  verified result file only after all items commit.

Stdout contains verified job metadata rather than private generated document
results. The guarded complete-document result stays charged through durable
commit, and final metadata publication preserves the original cancellation
cause even if its write or flush fails.

Library users have
`jobs::runner::sampled_sentiment::quantized::long::Int8SampledSentimentDocumentJobPlanner`,
`int8_sampled_sentiment_document_job_requirements`, and the
`NlpEngine::job_int8_sampled_sentiment_documents` serial/team APIs.
The planner derives its immutable recipe from the actual model-bound planner;
callers cannot replace that authority with a deserialized result envelope.

The resident and durable adapters, command routes, memory and accounting
regression definitions have source review. Compilation, executed regressions,
native-model parity/recovery, throughput and task-quality evidence remain
pending controller validation.
