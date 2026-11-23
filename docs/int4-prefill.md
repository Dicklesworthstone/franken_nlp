# Explicit grouped-INT4 prompt prefill

`fnlp candidate source-int4 generate`, `chat`, `classify`, `sentiment`,
`judge`, `extract`, `ner`, `keyphrases`, `summarize` and `answer` accept
`--prefill-rows N` for bounded layer-major prompt execution. Each row is one
prompt position within a single task; scored tasks group each independent
head's prompt. The value is 1 through
64. Without this flag the existing sequential path is unchanged. This is an
explicit non-certified candidate, not automatic dispatch, a benchmark result,
a numerical qualification, or an INT8 fallback.

## CLI

```sh
fnlp candidate source-int4 generate prompt.txt \
  --source-dir /models/nanbeige-source \
  --source-manifest /models/pinned-source-manifest.json \
  --memory-mib 8192 --group-width 128 --prefill-rows 4

fnlp candidate source-int4 chat conversation.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128 --prefill-rows 4 \
  --kernel x3b_avx2_widened_i16
```

These commands require the `asupersync-runtime` feature. The second example
requires the named AVX2 implementation on the actual host. Kernel selection
and prompt scheduling are independent; scalar remains the kernel default and
unsupported explicit kernels fail rather than fall back. The example memory
ceiling is a ledger allowance, not measured RSS.

### Buffered and streamed text with persistent workers

Buffered `generate` and `chat`, and live `stream generate|chat`, can use the
same persistent output-row team for prompt prefill and decode. Add
`--compute-threads 2..=64` to either example:

```sh
fnlp candidate source-int4 generate prompt.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --prefill-rows 4 --compute-threads 4 \
  --input-encoding safe-bpe-v1
```

The coordinator and three workers form once before inference. Each shard
prepares a packed group once and applies it across the prompt morsel, preserving
each row's activation scale, increasing-group accumulation and final BF16 cast.
The same buffers and workers then execute single-token decode. Causal
attention, KV, sampling and the complete native work ledger remain with the
coordinator. Selected output rows retain their exact ordered selection.

Admission reserves both the ordinary prefill workspace and a fixed batch
input/output capacity for every worker. No projection grows its worker buffers,
creates another team or renews the request work/checkpoint allowance. Larger
declared prompt widths therefore increase the team scratch quote even when a
particular prompt uses a shorter final morsel. Cancellation and failure drain
the team before releasing its charged storage or publishing completed output.

Library callers use `NlpEngine::execute_int4_chat_layer_major_with_threads` or
`NlpEngine::execute_int4_chat_stream_layer_major_with_threads`, with
`int4_chat_prefill_team_workspace_bytes` and a compatible resource host. The
completed response includes the existing `compute` receipt; `completed_rows` counts
completed output scalars across all prompt rows, while `completed_projections`
counts successfully completed batch or single-row operations. These counters
do not establish speed or model quality. Resident `score-batch` also composes
prompt morsels with one persistent CPU team across all classification,
sentiment or judgment heads, records and flushes. Its hosted entry is
`NlpEngine::batch_int4_scored_layer_major_with_threads`; the terminal footer
uses the original run control after every worker joins. See the
[scoring stream contract](candidate-scored-batch-cli.md#grouped-int4-scoring-streams)
for combined memory admission and per-frame schedule metadata. The retained-job
sections below describe which durable routes support combined schedules.

Live streams retain directly backpressured provisional token events. Their
`run_complete` remains unavailable until prompt/decode execution, independent
task validation, native cleanup, all worker joins and the final cancellation
decision succeed. A writer or permit failure does not append terminal success.
The same team handles each short prompt morsel and each decoded token without
reallocating its admitted capacity or renewing any work allowance. See the
[token-stream contract](candidate-streaming.md#grouped-int4-streams-with-packed-prompts-and-workers).

### One-shot scored and structured tasks

`classify`, `sentiment`, `judge`, `extract`, `ner`, `keyphrases`, `summarize`
and `answer` can combine `--prefill-rows 1..=64` with
`--compute-threads 2..=64`. Add both flags to the existing one-shot command;
the task's input format, source or snapshot pin and other limits still apply.
One fixed team serves every prompt morsel and continuation across the complete
task. For sentiment this includes every axis; for judgment it includes both
pairwise orders, all rubric criteria or all source/evidence heads.

Library callers use `NlpEngine::execute_int4_classify_layer_major_with_threads`,
`execute_int4_sentiment_layer_major_with_threads`,
`execute_int4_judge_layer_major_with_threads`,
`execute_int4_extract_layer_major_with_threads` or
`execute_int4_source_task_layer_major_with_threads`.
The public `int4_chat_prefill_team_workspace_bytes` quote also covers these
tasks' shared worker layout; ordinary native reservations and coordinator
prompt workspace remain separate.
The shared `HostedInt4TeamResult<T>` retains ownership through publication.

Combined scheduling supports each task's existing compact-input and exact
selected-row options. It preserves full-vocabulary scoring for scored tasks,
the complete legal row set for constrained decoding, and the original aggregate
work allowance. It cannot combine with `--cohort-heads`. Invalid widths,
insufficient memory and stale worker quotations refuse before private IO.
Cancellation drains the team before its model and scratch owners can release.
The `compute` receipt records actual work; model parity and speed still require
controller-admitted verification.

### Other task routes

Source verification, independent snapshot pins, optional create-new snapshot
saving, tokenizer/template binding, input validation and the completed JSON
envelope remain unchanged. A snapshot failure never starts source conversion.
One-off scored and structured tasks use the same explicit scheduling option
with complete task and native work admission. The five structured batch commands
`extract-batch`, `ner-batch`, `keyphrases-batch`, `summarize-batch` and
`answer-batch` accept it for independent records as described below. When
`--cohort-documents N` is explicit, `P` instead bounds total prompt/feedback
token positions across the live window,
and the CLI and host reserve the extra scratch once. See
[structured cohort packing](int4-structured-cohorts.md#packed-prompt-and-feedback-passes)
for the schedule, memory quote and ordered delivery contract.

Maps, complete-document retained jobs, resolution, redaction and
semantic extraction keep their existing prompt schedules and reject this option,
including programmatically constructed requests. Their independent
document/head/chunk cohort options remain separate. Generation/chat retain
their existing packed batch and corpus interfaces.

### Independent extraction and source streams

All five structured batch commands accept `--prefill-rows 1..=64` with serial
compute or `--compute-threads 2..=64`. Each complete record executes as it
arrives; this schedule needs no document cohort or producer-side fill/flush
coordination. For example:

```sh
fnlp candidate source-int4 extract-batch documents.ndjson \
  --schema extraction.schema.json --source-membership \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --prefill-rows 4 --compute-threads 4 \
  --input-encoding safe-bpe-v1 --max-selected-rows 4096
```

`ner-batch`, `keyphrases-batch`, `summarize-batch` and `answer-batch` accept
the same scheduling flags with their existing typed input formats. The complete
legal-row set must fit an explicit selected-row ceiling; the command does not
prune that set to make a smaller ceiling succeed. Source conversion, pinned
snapshots and optional snapshot saving retain their existing rules.

The host admits one coordinator prompt workspace plus fixed worker row buffers
for the declared width, beside the complete input/transport envelope, compact
encoder workspace, native KV and scratch. One native processor and team span
every record, constrained token and flush. Native and mask work remain
whole-stream allowances; neither scheduling width increases them.

Each event reports `prefill_schedule` and, when selected, `compute_schedule`.
The original schema/source finalizers, source coordinates, per-record encoding
policy and result identities remain in force. Charged input and committed native
reservations survive physical worker join on both success and ordinary stream
errors. Only after joining may the existing deferred `run_complete` be published,
under the original deadline and checkpoint control. Whole-run success still
requires that footer and process exit zero.

Library callers use `NlpEngine::batch_int4_extract_layer_major` and
`NlpEngine::batch_int4_source_layer_major`, or their
`_layer_major_with_threads` variants. Document cohorts retain a separate packed
schedule and cannot combine with worker teams. Regression definitions cover
combined admission, exact memory boundaries, all task routes and completion
ownership; controller-only model fixtures cover full canonical stream results
across schedules. These definitions remain unexecuted and do not establish
numerical parity, task quality or a performance gain.

### Retained generation and chat

`fnlp candidate source-int4 text-job start|resume generate|chat` also accepts
`--prefill-rows N`. Add it to the existing retained-job command, for example:

```sh
fnlp candidate source-int4 text-job resume generate prompts.ndjson \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --context-tokens 512 --max-new-tokens 32 \
  --max-output-bytes 4096 --prefill-rows 4 --compute-threads 4 \
  --store-results --job-dir /private/text-job \
  --job-id "$JOB_ID" --key-file /private/text-job.key \
  --limits text-job-limits.json --materialize
```

The job must already exist with the same model, inputs, task settings, key and
immutable lifetime limits. See the [retained generation/chat contract](../README.md#implemented-candidate-int4-generation-and-chat-jobs)
for creation and an example limits file. These jobs additionally require
`metadata-store` and the supported native storage platform.

The row declaration is a physical invocation schedule. It may change on resume,
including switching to or from sequential prefill, without changing the
authenticated recipe or original addressed samples. Every pending item still
reserves its full cold native work before execution; earlier attempts and
committed results never renew that allowance. Committed items are skipped, and
pending items restart from their own empty logical KV. No KV is persisted.

`--compute-threads 2..64` combines prompt morsels with one persistent team across
all pending prompts and decoded tokens. The coordinator's prompt workspace and
the team's fixed batch buffers are admitted together with the full job envelope
before input access, including source/snapshot/save, compact BPE, and retained-input
replay routes. Both physical widths may change on authenticated resume without
renewing attempts or lifetime native work. The completion report adds an
optional `prefill_schedule` containing `grouped-int4-layer-major-causal-prefill-v1`,
the maximum row count and admitted workspace bytes, plus `compute_schedule`
when a worker team is selected. Successful metadata follows physical worker
drain; private stored results keep their existing schema.

The [sampled-sentiment commands](native-sampled-sentiment-int4.md) have their own
explicit prompt schedule. Their durable recipe freezes that choice; the
generation/chat rule allowing physical row changes does not alter it.

### Retained extraction and source tasks

The `extract-job`, `ner-job`, `keyphrases-job`, `summarize-job` and `answer-job`
commands accept `--prefill-rows N` on both `start` and `resume`, with the same
range of 1 through 64. Each retained item uses bounded prompt morsels within
its own exclusive session. This option also works with `--input-encoding
safe-bpe-v1` and the existing exact selected-row projection for extraction and
source tasks. The complete legal-row cap remains authenticated on resume.

Use serial compute or `--compute-threads 2..64` with this option. One fixed
batch-capable team serves all prompt morsels and constrained-decoding steps
across pending items, durable commits and materialization. The CLI validates
the full memory envelope before reading private inputs. Hosted execution
independently admits coordinator prompt storage, fixed worker batch buffers,
compact-encoder workspace, native allocation and retained-input buffers before
dispatch. These reservations remain owned through physical worker join.

Prompt width is a physical schedule and may change on authenticated resume,
including switching between serial and layer-major execution. It does not
change the job recipe, result schema, input policy, lifetime native/mask-work
limits or tokenizer allowance. Committed items remain skipped. Pending items
retain their original cumulative debits and restart with empty logical KV.
Completion reports add the same optional `prefill_schedule` object described
above and `compute_schedule` when workers are selected. Default reports stay
unchanged.

Library callers can opt into `NativeInt4ExtractionBatch::new_layer_major`,
`NativeInt4SourceBatch::new_layer_major`, and the corresponding
`Int4ExtractionJobProcessor` and `Int4SourceJobProcessor` constructors. The
hosted entry points are `NlpEngine::job_int4_extract_layer_major` and
`NlpEngine::job_int4_source_layer_major`; the corresponding combined entries
are `job_int4_extract_layer_major_with_threads` and
`job_int4_source_layer_major_with_threads`. Existing constructors keep serial
execution. The ignored pinned-model gate
`complete_structured_jobs_match_serial_results_and_resume_without_reexecution`
covers complete output/work equivalence and retained replay across schedules;
it must pass under the repository's authorized build controller before these
paths can claim numerical qualification or model-tested parity.

The separate [scoring-cohort mode](int4-scoring-cohorts.md) groups independent
heads with `--cohort-heads N`. It cannot be combined with `--prefill-rows` and
prices all simultaneously retained KV, prompts and cursors.

For an existing digest-pinned INT4 snapshot and a typed classification request:

```sh
fnlp candidate source-int4 classify request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128 --prefill-rows 4
```

The `sentiment` and `judge` commands accept the same option with their existing
typed request files. Their complete multi-head preparation remains independently
charged; every head must pass admission before the first model forward.

## Native execution and numerical contract

The native `Int4Session::append_layer_major` method validates the entire exact
prompt, remaining context, work ceilings and extra-workspace limit before
mutating state. Long prompts are divided into bounded morsels. Each morsel
runs the existing `LoopRunner` through both 22-layer passes and both final
norms, preserving all 44 independent logical KV slots. A complete prefix may
be extended without reinterpreting its position coordinates.

Grouped projection visits retain low-nibble-first signed INT4 storage,
per-row activation scales, each matrix's group partition and unpadded tails.
Up to four activation rows share each packed-group integer reduction. Every
lane dequantizes and accumulates groups in the original order, then casts to
BF16 once. Weights are not expanded into a dense INT8 or floating matrix.

Attention appends one token's key/value row and immediately attends to that
exact prefix. It does not append a whole morsel before attention; earlier rows
cannot see future tokens. Post-prompt decoding remains token-by-token. Only
the final prompt position runs the language-model head, never intermediate
morsel boundaries. `prefill_layer_major` also supports the existing selected
head-row contract for native callers.

The generation adapter reuses the existing cursor, sampler addressing, stop
and penalty policy. Prompt positions are reconciled without rerunning the
model or consuming random draws. Chat reuses the pinned tokenizer and the same
independent result finalizer. Model fingerprints, semantic execution identity,
profile and completed result fields do not change with prompt scheduling.
Algorithmic group-dot/MAC/attention counts still describe the same work, not a
claim about retired instructions, memory traffic or speed.

Finite-candidate scoring groups only the initial prompt of each head. The shared
candidate scorer still owns every continuation, terminal EOS edge, probability
denominator and trie branch. Backtracking retains one live KV branch, and heads
clear their KV state before the next head. Classifications, independent sentiment
axes, both pairwise judgment orders, rubric criteria and faithfulness evidence
retain their original finalizers and exact whole-bundle work allowances. The
same completed result contract applies to serial and grouped execution.

Structured tasks replace only prompt appending. The shared constrained decoder
continues to own full-vocabulary greedy selection, source-language and grammar
masks, scored terminal EOS and exact native/mask work. Extraction independently
validates the same raw schema and source fields. NER, keyphrases, summary and QA
run their original semantic/source finalizers while the exclusive session still
owns KV; a failed finalizer or late cancellation aborts that same session.
Prompt grouping grants no additional output, verification or model-work budget.

For example, opt into the same schedule for a single extraction or source task:

```sh
fnlp candidate source-int4 extract document.txt \
  --schema extraction.schema.json --source-membership \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128 --prefill-rows 4

fnlp candidate source-int4 ner document.txt \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128 --prefill-rows 4
```

`keyphrases` and `summarize` accept the same document input and schedule.
`answer` retains its existing JSON request containing the exact question and
passages. Typed options, original schema bytes, source evidence, complete result
envelopes and private-input handling retain their existing contracts. One-off
structured commands remain greedy; prompt grouping does not enable text
sampling or repair controls.

## Memory and cancellation

`Int4PrefillLimits::required_extra_scratch_bytes(rows)` is the authoritative
additional payload bound: `rows * (32 * intermediate_size * sizeof(f32) + 4096)`.
It prices the finite prompt-row rails, activations and descriptors separately
from base native scratch, KV capacity, RoPE, sampler and allocator allowances.
The engine's one existing group-unpacking buffer is not duplicated per row.
The bound is conservative, not a peak-RSS measurement.

The CLI adds this bound only to the execution stage, then takes the maximum of
loading, optional saving and execution storage. It does not charge all three
stages as though they coexist. The host independently reserves the additional
payload in `ActivationScratch` before any native allocation. That reservation,
the model and all other native guards remain owned until physical completion.
The hosted result and CLI preparation guards remain live during publication.

The whole prompt is preflighted, and complete projection work is charged before
its first output write. Cooperative cancellation is polled between bounded
native operations; failed work is not refunded or automatically retried.
Failures after mutation poison the normal session; RAII cleanup clears every
KV slot. No token event is emitted by the new generation adapter until prompt
prefill and its final head complete. Library stream events remain provisional
until the full task finalizer succeeds. Buffered commands emit one completed
JSON object; explicit `stream` commands expose provisional token events and
require the independently validated terminal frame.

## Embedding APIs

The existing sequential APIs remain available with their original signatures.
The explicit alternatives are:

- `Int4Session::append_layer_major` / `prefill_layer_major`.
- `Int4GenerationPlan::execute_layer_major` / `execute_layer_major_with_sink`.
- `PreparedInt4Chat::execute_layer_major` / `execute_layer_major_with_sink`.
- `NlpEngine::execute_int4_chat_layer_major` for charged hosted execution.
- `Int4GenerationJobProcessor::new_layer_major` for a durable generation/chat
  adapter with separately admitted prompt workspace.
- `NlpEngine::job_int4_generation_layer_major` and
  `job_int4_generation_layer_major_with_threads` for charged retained execution,
  authenticated original or retained inputs, and complete-item commits.
- `Int4CandidatePlan::execute_layer_major` for exact finite-language scoring.
- `PreparedInt4Classification`, `PreparedInt4Sentiment` and `PreparedInt4Judge`
  expose `execute_layer_major_with_control`.
- `NlpEngine::execute_int4_classify_layer_major`,
  `execute_int4_sentiment_layer_major` and `execute_int4_judge_layer_major`
  reserve the extra scratch before entering the existing blocking invocation.
- `constrained_int4::decode_json_int4_layer_major` and
  `Int4ExtractPlan::execute_layer_major` share the constrained decoder and
  independent extraction finalizer.
- `PreparedInt4SourceTask::execute_layer_major_with_control` retains the sealed
  NER, keyphrase, summary or answer finalizer inside the native session.
- `NlpEngine::execute_int4_extract_layer_major` and
  `execute_int4_source_task_layer_major` admit the full declared extra prompt
  scratch with the original native allocation and guarded semantic output.

Lower-level limits describe payload ceilings; they are not a substitute for a
resource-domain reservation. The hosted method shares one implementation with
sequential execution, including actual model/domain checks, native work bounds,
output admission and physical cleanup. It cannot import uncharged weights or
mint a successful task result from synthetic model output.

## Validation status

Regression source covers packed row/scalar equivalence across group widths,
odd reductions, tail groups and available explicit kernels; causal attention
against token-at-a-time replay; work and memory admission; cancellation and
cleanup; cursor output/score/policy reconciliation; and CLI source/snapshot
modes and stale scheduling admission. Synthetic logits exercise the real
cursor, not model quality or full-model numerical equivalence.

The ignored test
`model_present_snapshot_text_prefill_matches_sequential_complete_results`
compares the complete CLI response for sequential and 1/4/64-row prefill,
separately for generation and multi-turn chat, using seeded decoding and
log-probabilities. It requires controller admission and these explicit inputs:
`FNLP_INT4_PREFILL_SNAPSHOT` (an existing group-width-128 snapshot),
`FNLP_INT4_PREFILL_MODEL_SHA256` (its independently retained model-content pin),
`FNLP_INT4_TEST_MEMORY_MIB`, and `FNLP_INT4_TEST_KERNEL`. It only reads that
snapshot; it never saves, replaces, deletes or downloads model files.

The ignored
`model_present_structured_prefill_matches_every_complete_serial_cli_result`
uses those same explicit model, kernel and memory inputs and additionally
requires `FNLP_INT4_PREFILL_SCHEMA`, an existing local file containing
`{"type":"boolean"}`. It compares the complete canonical CLI response for
extraction and all four source tasks at serial execution and rows 1, 4 and 64,
including exact JSON/token transcripts, original-source evidence and work.
Model-free regression source additionally covers ragged prompts, numeric
continuation, UTF-8 fragments, source masks, semantic refusal and cancellation,
static scratch admission, all model-loading routes and unsupported workflow
boundaries. These are source fixtures, not executed proof.

Retained generation/chat additions cover unchanged recipes and native lifetime
work, exact coordinator and worker workspace for complete and retained inputs,
invalid schedules before IO, cancellation and physical drain, all model-input routes,
and optional metadata without private content. These regression definitions
remain unexecuted. The explicit pinned-model gate compares complete greedy and
seeded generation/chat results and lifetime work across serial, layer-major and
combined worker schedules, then authenticates retained-input replay after both
physical widths change. That durable real-model gate remains pending.

Compilation, Rust tests, full-model parity and performance remain unrun pending
the repository's controller-selected immutable-SHA DSR checkpoint. Source review,
whitespace checks and exact Git tree verification do not establish numerical
qualification or measured performance. The path remains explicit and opt-in.
