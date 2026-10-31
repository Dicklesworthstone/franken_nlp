# Explicit INT4 scoring cohorts

The one-off `candidate source-int4 classify`, `sentiment` and `judge` commands
accept `--cohort-heads N`. This executes every scoring head in one native
shared-weight cohort, with independent causal KV and probability normalization.
`N` is an explicit ceiling from 1 through 64. A task with more heads refuses;
it is never split into smaller cohorts or silently retried serially.

`candidate source-int4 score-batch` also accepts the option. It gathers bounded
windows of independent documents and executes their complete head bundles in
one shared native cohort. The stream retains one resident model across windows.

The option is an unqualified candidate execution path. No speedup, native model
parity or task-quality result is implied by the implementation or grouping
counters. Omitting the option preserves existing serial execution.

## Usage

For an existing independently pinned INT4 snapshot:

```sh
fnlp candidate source-int4 sentiment request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128 --context-tokens 1024 \
  --cohort-heads 4
```

The usual `--source-dir` and `--source-manifest` pair is also supported, with
optional create-new `--save-snapshot`. The request format and scoring policies
are unchanged. Exclusive classification has one head; multilabel classification
has one independent head per label. Sentiment has one head per requested axis.
Pairwise judgment has two presentation-order heads, rubric judgment has one
head per criterion, and faithfulness retains its existing complete-source and
evidence-window head construction.

`--cohort-heads` cannot be combined with `--prefill-rows`. Prompt prefill groups
successive positions within one head; a scoring cohort groups one next position
from each active head. They have different simultaneous storage requirements.
Scored jobs, resolution, extraction and source-task commands reject the cohort
option. Generation/chat retain their separate text cohort interfaces.

## Cross-document scored streams

For sentiment's default four axes, an eight-head ceiling can execute two
documents together:

```sh
fnlp candidate source-int4 score-batch reviews.ndjson --task sentiment \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 32768 --group-width 128 --context-tokens 1024 \
  --cohort-heads 8 --cohort-documents 2
```

The same route supports classification and judgment, using their existing
`--defaults` and per-record `task_args` contracts. Both head count and document
count are bounded. `--cohort-documents D` accepts 1 through `N`, requires
`--cohort-heads`, and counts buffered error records as well as valid documents.
If omitted, `D` defaults to the smaller of `N` and the number of maximum-size
inner result frames that fit the native 64 MiB complete-cohort output cap.
An explicitly supplied window that exceeds this cap rejects before model or
document IO. It is never silently narrowed.

Heads from a complete document stay together. A new document that would exceed
the remaining head capacity starts the next cohort after the current cohort
drains. A document whose own head count exceeds `N` produces an item error;
there is no split-document or serial retry. Head count, per-head context,
probability space, full execution identity and work come from sealed task plans,
not from NDJSON-supplied scheduling metadata. Each cohort receives its exact
number of native sequence contexts, with no dummy heads.

The input remains bounded and ordered. A window runs when its capacity is
reached, on an explicit `{"flush":true}`, or at EOF. Interactive producers must
send enough records or a flush before waiting for responses; the blocking
reader has no partial-window timer. A flush acknowledges delivery only after
earlier results have been written and flushed, and then resets the epoch's
duplicate-ID set. Malformed records consume their input and sequence positions
and cannot shift later results onto another document. Work limits apply to the
whole invocation across every cohort and flush, including all five native
counters. Increasing a window or request count adds no compute allowance.

Each completed semantic task result keeps the existing event format. The
candidate provenance adds `cohort_schedule` with
`execution="cross_document_scoring_cohorts"`, `declared_max_heads`, and
`declared_max_documents`. These are admission ceilings. The CLI does not report
them as executed group counts or a performance result. Its terminal record and
`run_complete` plus exit-zero success requirement stay unchanged. The library
return additionally contains actual aggregate cohort counters. Inner events
identify the transport as `execution="ordered-cohort-no-retry-v1"`; their
semantic task result and original task execution identifier are unchanged.

Memory admission covers all simultaneous head KV, native activation and RoPE
storage, every buffered prepared document, one lookahead plan, transport/ID
staging, grouped cursor and result workspace, and all complete output guards.
The CLI and host share the same checked quote. Plan preparation is charged for
`D + 1` live plans, in addition to outer CLI preparation; choosing a smaller
document window can therefore substantially reduce the required process
ceiling without reducing the head ceiling of a single document. The declared
per-document preparation estimate remains the caller's responsibility in the
library. All charges remain owned through physical completion and sink flush.

Failed cohort execution yields no partial completed document bundle. Earlier
fully delivered cohorts remain visible. Output failure poisons delivery and
does not append a second terminal record. This route retains the candidate
numerics and evidence status; model parity and host throughput remain separate
qualification gates.

## Execution and output for one-off tasks

All heads and their complete schedules preflight before the first forward.
Each head then advances through its own exact candidate trie. Prefix rewinds
discard only that head's completed KV suffix and invalidate its hidden state;
they never refund completed work. Ragged prompts and finished heads are not
padded or replayed. Only identical projection row sets share an lm-head call,
so full-vocabulary and candidate-conditional probability spaces remain distinct.

The original classification, sentiment and judgment finalizers run inside the
same exclusive native lifetime. Every expected head must be consumed exactly
once. A failed head, output ceiling, cancellation or finalizer error returns no
partial completed task. The existing output reservation remains held through
CLI serialization, writing and flushing.

The existing semantic result stays in the response's `output` field. An
additional sibling `cohort` object reports:

```json
{
  "schema_version": 1,
  "execution": "grouped-int4-independent-candidate-cohort-v1",
  "group_steps": 120,
  "projection_groups": 9
}
```

Those numbers are illustrative. `group_steps` counts shared decoder calls;
`projection_groups` counts compatible lm-head calls. The semantic result still
reports the complete per-token/model work. These counters are neither elapsed
time nor a throughput measurement. Serial and prompt-prefill responses omit
the new `cohort` field.

## Memory and limits for one-off tasks

The CLI admits the declared ceiling before document or model IO. Its execution
peak includes `N` full `--context-tokens` KV capacities, grouped native rails,
retained host preparation, cursor/prompt/result workspace, allocator reserve
and complete output. The shared workspace quote is:

`64 × max_forward_positions + 8 × max_result_bytes + 65,536 × N` bytes.

This execution peak is compared with source loading and snapshot saving;
storage used in separate phases is not automatically summed. The process's
resident model, CLI preparation and cleanup reserve remain independently
charged. The host subsequently derives exact per-head contexts from the sealed
plan and reserves their sum. A ceiling larger than the task's actual head count
can therefore require more CLI-admitted memory than the eventual native
allocation. Choose a ceiling appropriate to the request.

Work ceilings cover the complete task across all heads, never a fresh allowance
per head. Cohorts retain the same whole-task output ceiling, including the
small native grouping wrapper. The memory ledger is a declared payload and
ownership limit, not an operating-system RSS guarantee.

## Library entry points

- `score_int4_cohort` executes exact-token plans under per-head and aggregate
  limits on an `Int4CohortEngine`.
- Prepared classification, sentiment and judge plans expose `cohort_contexts`
  and `execute_cohort_with_control`; `execute_document_cohort_with_control`
  composes independently admitted documents in one native session.
- `NlpEngine::execute_int4_classify_cohort`,
  `execute_int4_sentiment_cohort` and `execute_int4_judge_cohort` provide charged
  process-host execution on one resident model.
- `NlpEngine::batch_int4_scored_cohorts` provides bounded ordered cross-document
  cohorts for the three sealed scored-task planners. Its
  `Int4ScoredCohortHostLimits` declares document and head ceilings; the return
  separates the existing transport summary from aggregate native cohort stats.
- `batch::scored_int4::memory::Int4ScoredCohortMemoryLimits::requirements`
  quotes the complete grouped stream execution peak without loading a model or
  enabling the runtime feature.

Lower-level callers must reserve the quoted transient workspace in addition to
native memory and retained preparation. Hosted methods perform those
reservations before building context vectors or flattening prompts and retain
them through physical completion.

Regression source covers independent score-space replay, ragged execution,
branch work, cancellation, complete semantic finalization, simultaneous memory,
CLI refusal boundaries and separately ignored full-model parity matrices.
The complete-stream fixture compares classification and sentiment with mixed
head counts and prompt lengths, a reused ID after flush, bounded head-induced
window splits and an EOF tail. It freezes identical TaskIR budgets and input
bytes, then compares every event after checking and removing only the declared
scheduling fields. It requires the same explicit `FNLP_INT4_PREFILL_SNAPSHOT`,
`FNLP_INT4_PREFILL_MODEL_SHA256`, `FNLP_INT4_TEST_MEMORY_MIB` and
`FNLP_INT4_TEST_KERNEL` settings as the existing model fixtures, and isolates
the first process-resource installation in its own test process.
Compilation, Rust tests and native model/performance qualification remain
pending controller-selected DSR validation.
