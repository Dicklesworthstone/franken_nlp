# Native INT8 source task portfolio

`SourceTaskPlanner::plan_int8_with_control` accepts the existing `SourceTaskRequest`
for NER, ranked keyphrases, cited summaries and supplied-passage answers. The result
is `tasks::source_planning::quantized::PreparedInt8SourceTask`; it executes against
an already admitted `StrictInt8Engine` and shared `ExtractionVocabulary` through
`execute_with_control`, returning `Int8SourceTaskRun`.

The context must already specify strict quantized v1, the exact INT8 backend,
BF16 KV, disabled thinking, and no tools. The compiler checks the pinned template
and tokenizer, request/task ceilings and complete context. It does not create an
eager executable and rewrite its identity. The shipped finalizer version extends
the execution policy during compilation; that same sealed identity flows through
execution unchanged. The eager planner and eager answer finalizer remain separate.

The implementation uses the existing trusted prompt fragments, byte-preserving
source encoder, schemas, source grammar and typed finalizers. It creates no second
tokenizer, model loader, task interpreter, scheduler or weight copy. Passage QA
moves the original passage metadata into its finalizer and releases the temporary
question/source encoding after grammar compilation, rather than retaining another
source copy merely for offset projection.

All four modes run the real strict-INT8 constrained decoder. Full model work and
mask budgets remain enforced, along with whole allocated KV capacity. Exact source
occurrences, task semantics, complete output serialization bounds and the final
cancellation checkpoint are checked inside the same native session. Semantic or
output failure poisons/drains that session instead of leaving a reusable engine.
There is no free-generation parse retry or hidden BF16 fallback.

NER retains all exact occurrences and explicit ambiguity. Keyphrases retain model
order, first-exact deduplication and all occurrences. Every nonempty summary bullet
requires a nonempty source quote. Passage answers distinguish successful model-
declared abstention from failure; citations must fit wholly inside original
passages, and question/manifest text cannot become evidence. Citation membership
is structural, not entailment, calibrated confidence, model quality or coverage.

Preparation and finite grammar operations have before/after checkpoints, not
thread preemption or guaranteed cancellation latency. Callers still own actual
model, temporary preparation and output admission. This does not activate catalog
discovery, public model-backed CLI commands or production artifact qualification.

New regression sources exercise the real compiler and semantic finalizers using
explicit synthetic decoder results. They do not constitute model inference,
quantized quality, BF16 parity, throughput or numerical qualification. Rust
compilation/tests were not run in the coding-agent session; repository policy
reserves those for its authorized validation environment.

## One hosted persistent team for a source request

`NlpEngine::execute_int8_source_with_threads` accepts the prepared source task,
shared vocabulary, ordinary `SourceLimits`, checked `Int8Kernel`, admitted
compute width 2–64 and cancellation token. One sealed team spans prompt prefill,
both decoder passes, constrained generation and source finalization. The native
session charges each complete projection once; workers receive the original
quantized activation and compute independent full output rows. The prepared
identity, selected-row head policy, exact source data and mask budgets do not
change with worker width.

The existing process host must admit the requested child envelope. Input and
vocabulary preparation, KV, scratch, output and worker reservations survive
physical join on success, error and panic. A checkpoint after join uses the
original shared cancellation authority. Successful
`HostedInt8TeamResult<Int8SourceTaskRun>` retains guarded output and exposes
separate physical compute counters. The existing default and layer-major APIs
retain their serial contracts; this single-row team does not silently compose
with packed prefill. Candidate users select it with `--compute-threads` on
`ner`, `keyphrases`, `summarize`, or `answer`; see the
[source command contract](candidate-source-cli.md#persistent-cpu-teams-for-source-tasks).

Added admission and real-artifact gate definitions cover the task/encoding/head
matrix, exact work ceilings, complete result equality between widths, typed
cancellation and output lifetime. They are unexecuted definitions and do not
establish whole-model parity or measured performance.
