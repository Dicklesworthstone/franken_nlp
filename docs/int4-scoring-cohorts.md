# Explicit simultaneous INT4 scoring heads

The one-off `candidate source-int4 classify`, `sentiment` and `judge` commands
accept `--cohort-heads N`. This executes every scoring head in one native
shared-weight cohort, with independent causal KV and probability normalization.
`N` is an explicit ceiling from 1 through 64. A task with more heads refuses;
it is never split into smaller cohorts or silently retried serially.

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
Scored batch/job, resolution, extraction and source-task commands reject the
cohort option. Generation/chat retain their separate text cohort interfaces.

## Execution and output

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

## Memory and limits

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
  and `execute_cohort_with_control`.
- `NlpEngine::execute_int4_classify_cohort`,
  `execute_int4_sentiment_cohort` and `execute_int4_judge_cohort` provide charged
  process-host execution on one resident model.

Lower-level callers must reserve the quoted transient workspace in addition to
native memory and retained preparation. Hosted methods perform those
reservations before building context vectors or flattening prompts and retain
them through physical completion.

Regression source covers independent score-space replay, ragged execution,
branch work, cancellation, complete semantic finalization, simultaneous memory,
CLI refusal boundaries and a separately ignored full-model parity matrix.
Compilation, Rust tests and native model/performance qualification remain
pending controller-selected DSR validation.
