# Process-hosted source-task corpora

With `asupersync-runtime`, `NlpEngine::batch_int8_source` executes NER,
keyphrases, cited summarization or passage QA over the existing ordered NDJSON
protocol. It extends `execute_int8_source` to a bounded corpus without requiring
a caller-written admission adapter or a separate runtime per document.

The method takes an already charged `ResidentInt8`, reusable
`Arc<SourceTaskPlanner>` and `Arc<ExtractionVocabulary>`, `SourceCorpusConfig`,
`CorpusLimits`, an owned reader/writer and a cancellation token. It remains a
synchronous library call. It does not discover/download a model, activate a
catalog candidate, claim artifact qualification, or enable a public CLI route.

## Configuration and data

`SourceCorpusConfig` fixes the execution identity, task ceiling, planning limits,
optional `SourceBatchArgs` defaults and whole-run native/mask work ceilings.
Those settings are host configuration, not deserializable executable authority.
The host verifies actual resident-model facts, pinned planner assets, strict
INT8 profile/backend, default-argument bounds and complete KV reservation
before native allocation or input reads. Record overrides cannot change tasks,
raise the admitted ceiling or replace the model identity.

The document envelope remains `{id,text,task_args?}`. For NER/keyphrases/summary,
`text` is the source. For QA it is the question, with evidence passages supplied
separately in the answer arguments. Source quotations remain byte-exact;
repeated occurrences and original-passage boundaries retain their existing task
contracts. Valid abstention is a successful QA result, not a way to hide failure.
Source membership does not establish semantic accuracy or entailment.

## One resource domain and physical lifetime

The complete run uses the existing process-owned blocking invocation and native
scope. One native engine retains its KV/scratch allocation across documents;
one pinned vocabulary is reused. Each request cold-prefills its own source and
clears logical KV on completion. There is no implicit prefix reuse, parallel
batching, per-item model reload, retry or repaired poisoned engine.

Preparation and owned I/O are captured in a storage-before-charge aggregate.
The preparation estimate must cover retained planner/vocabulary allocations,
source/grammar structures, passage metadata and task-argument copies. The I/O
estimate includes the actual supplied reader/writer buffers; a cursor over a
complete corpus must price that whole buffer. KV and scratch are separately
reserved once. These are modeled commitments, not allocator interception or
an observed/OS-enforced RSS bound.

The source adapter reuses the concrete extraction admission implementation.
It checks the already-charged KV capacity, mask contract, real model and result
limit, then retains a genuine process-ledger output reservation through native
execution, canonical serialization, write and flush. No no-op permit or guessed
post-allocation commit is synthesized. Native buffers, planner/vocabulary and
owned I/O drain before physical completion. Runtime-wrapper cancellation alone
does not release an active native invocation's charges.

## Failure and cancellation

The same `RunControl` reaches framing, source preparation, native execution and
final delivery checkpoints. Source planning does not get a renewed timeout or
checkpoint count for each item. Its existing tokenizer/grammar calls remain
bounded but noninterruptible internally; arbitrary caller I/O is not safely
preemptible. Completed structured results still require native source
verification, semantic finalization and full-envelope validation.

Model/attention/projection work and grammar-mask reservations do not refund
failed attempts or reset at flushes. An output failure stops the runner before
another record is read. A clean EOF returns `BatchSummary`; callers must inspect
`failed` rather than assume every document succeeded. Fatal failures retain
that summary in `HostedError::Batch`, including cancellation causes. Successful
abstentions count as successes; failed citations and decoder failures do not.

The code-first tests cover pinned configuration, real host admission arithmetic,
workspace limits, rejected defaults and `Send + 'static` ownership requirements.
The native adapter separately tests control propagation and delivery lifetimes
through the real runner with a private fault-injection driver. Compilation,
Rust test execution, model-present inference and quality/performance evidence
remain controller-owned and unexecuted here. No release/CLI gate is promoted.
