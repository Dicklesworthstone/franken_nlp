# Resident INT8 source-task corpora

`batch::source::quantized` connects the existing raw `SourceBatchArgs` to the
native NER, keyphrase, cited-summary and passage-QA implementations. One actual
`StrictInt8Engine` and `ExtractionVocabulary` serve the ordered NDJSON stream.
This is sequential resident-engine execution, not parallel/GEMM batching or
prefix-cache reuse, and does not activate the public model-backed CLI.

## Fixed task, exact evidence

Construct `Int8SourceBatchPlanner::new` from a borrowed pinned
`SourceTaskPlanner`, an explicit strict-INT8 identity, task ceiling, source
planning limits and optional default arguments. The identity fixes one task
for the run. A record cannot change tasks, raise the ceiling, supply tokens or
replace model identity. Defaults are bounded before per-record cloning;
record overrides affect only that record. The additional serialized argument
cap is `MAX_SOURCE_ARGUMENT_BYTES` (1 MiB), independent of transport limits.

The existing `SourceBatchArgs` protocol is unchanged:

- `ner`, `keyphrases`, `summarize`: document `text` is the exact source.
- `answer`: document `text` is the question; evidence passages are separately
  typed in `task_args.passages` (or the explicitly supplied defaults).

Preparation uses `SourceTaskPlanner::plan_int8_with_control` directly, never
an eager plan conversion. Its source exclusion, Unicode coordinates, repeated
occurrences, first-proposal keyphrase ranking, cited-bullet requirements and
original-passage citation boundaries remain with the existing finalizers.
Valid QA abstention is a successful typed result. Missing passages, failed
execution or invalid citations are failures, never rewritten to abstention.
Exact source membership is not a proof of semantic support or task quality.

## Native execution and ownership

`NativeInt8SourceBatch::new` takes the prepared compiler, a borrowed real native
engine, a borrowed vocabulary, genuine admission authority and
`Int8SourceBatchLimits`. These limits and admission types are aliases of the
existing INT8 extraction contracts: source tasks need the same full native
work, grammar-mask, complete resident KV and output reservations. An existing
`Int8ExtractionBatchAdmission` implementation therefore works unchanged.

Use the adapter with `batch::run_ndjson`. The new backward-compatible
`BatchProcessor::prepare_with_control` hook lends the original run control to
planning. Source planning checks that same deadline/checkpoint allowance around
bounded tokenizer/grammar calls; it never creates a fresh per-document budget.
The existing calls are not made internally preemptible by this hook. Legacy
processors retain their original `prepare` implementation by default.

This new adapter intentionally refuses its control-free `prepare` and
context-free `execute` methods. Standalone users prepare through the planner's
explicit control-taking method and execute the resulting native source plan.
The standard runner supplies both control and engine-assigned delivery context.

Every attempt atomically reserves all five native counters and grammar-mask
work before admission. Reservations are checked for overflow and never
refunded; flush epochs renew neither. The outer runner independently enforces
its forward/logit, input/output, framing and record limits. Complete result
work is compared to the native source plan, not to the eager projection count.

The task performs source verification, semantic finalization, complete-envelope
bounds and late cancellation inside its native session. Only a completed typed
result reaches the writer. Its admission guard stays owned through canonical
serialization, write and flush. Failed native state, an unwind, substituted
identity or invalid delivery coordinates closes the adapter. Cancellation keeps
its cause even if the engine is also poisoned. No retries or native resets are
performed; the outer runner stops on broken output before reading another item.

## Evidence boundary

This implements the source-corpus portion of `franken_nlp-5pz.5` and the resident
NDJSON execution goal in `franken_nlp-bdn.1`; it does not close those beads or
promote artifact/CLI/model-quality gates. Tests use real pinned planning and a
private synthetic driver for admission, cancellation and output-lifetime fault
injection. The public constructor cannot accept that test driver. Rust
compilation, Rust test execution and real-model validation remain pending the
controller-owned checkpoint.
