# Process-hosted current-candidate INT8 execution

With `asupersync-runtime`, `NlpEngine` now connects its real process resource
host to local candidate loading and native task execution:

- `load_current_candidate_int8(path, limits, cancellation)` creates a charged,
  shared `ResidentInt8` from the existing streamed candidate loader.
- `execute_int8_chat(model, prepared, request_seq, native_limits,
  max_sampler_bytes, cancellation)` executes a `PreparedInt8Chat`, covering
  both generation and multi-turn chat.
- `execute_int8_extract(model, prepared, vocabulary, native_limits,
  mask_limits, max_mask_node_visits, cancellation)` executes an
  `Int8ExtractPlan` with its existing schema/source finalizer.

These are real native library routes, not calls to a fake model or a second
runtime. They do not open a public model-backed CLI, install or activate a
catalog, ratify OQ-31, authenticate a publisher, or award numerical/task quality.
The current-candidate loader retains its explicit non-authoritative grade.
Prepared plans, their pinned tokenizers and extraction vocabularies are supplied
by the caller; existing preparation allocations are not secretly accounted as
new allocations made by the host. Their ownership remains explicit.

## Runtime and physical lifetime

Use an already installed `EngineResources` host with exactly one blocking
coordinator. This is the serial native INT8 baseline, not parallel inference.
The configured runtime can still have its separately counted scheduler workers;
no new runtime or ad-hoc thread pool is constructed. Each call enters one
`Cx::spawn_blocking` closure and one coordinator-only `Cx::scoped_cpu(0)` scope.
The native callback receives a checkpoint-only restricted context, and its
ambient context is restricted as well. Re-entry from a synchronous call or
native callback refuses before new reservations or nested runtime entry.

The async wrapper's join and the closure's physical completion are different
observations. A capacity-one, closure-owned handoff retains the native outcome.
On wrapper cancellation, the caller marks the registered closure as outstanding
and waits for that handoff before returning. Captured buffers/reservations drop
before an unstarted discarded closure signals completion. Native errors, scoped
errors, wrapper errors and cancellation attribution are retained separately;
panic payload strings are not copied into the public error's Display/Debug.
The process's configured panic hook is not replaced or suppressed.

This uses the existing runtime's request context/root task ownership. It does
not claim a separately certified per-request region or the full OQ-28/OQ-29
lifecycle certificate. No scope/tree gate is closed by adding these methods.

## Resource accounting

The loader reserves before constructing its tokenizer or materialized weights.
The caller explicitly supplies tokenizer/metadata and allocator headroom, while
weight and streaming payload caps come from `ArtifactLoadBudget`. A successful
resident model owns both storage and its committed ledger charge. `ResidentInt8`
clones share the same Arc and charge; copying or independently loading another
model pays separately. The model retains its resource-host lease after the
original loading `NlpEngine` is dropped.

Each task checks the actual resident model's name/revision/recipe/logical digest
against its sealed plan before native allocation. KV uses its own ledger class;
RoPE/reference scratch/sampler headroom and output use separate reservations.
Context-dependent payload requirements come from `Int8MemoryRequirement`.
The output reserve includes typed result/token payload and canonical staging.
Temporary storage drops before its committed charges. Returned `HostedOutput`
keeps the result charge until delivery/storage is finished; its serializer emits
only the result and there is no unguarded unwrap.

All claims are checked against the same process ledger. These are modeled
commitments, not an operating-system RSS limiter or a claim that caller-provided
allocator/preparation estimates were independently measured. Existing external
plans/vocabularies and other application allocations need the embedder's own
explicit reservation policy. Missing/zero overhead estimates and checked
arithmetic overflow refuse rather than constructing an unbounded private host.

## Cancellation and limits

`RunLimits` requires finite elapsed time, a finite checkpoint count and a
nonzero cleanup-memory reserve. Cleanup capacity is held separately through
physical drain; native compute cannot consume it. Time
includes the blocking-pool queue. Every native checkpoint and the final host
checkpoint consumes the same allowance; neither a task boundary nor an error
replenishes it. `CancellationToken` is explicit, cloneable caller state, with
first-cause-wins semantics. Native work still enforces its independent forward,
projection, attention, sampler and mask bounds.

Cancellation is cooperative. The existing file loader and native engine binding
have noninterruptible portions; a stopped load is observed before/after the
reader, not by abandoning its OS call or reporting cleanup before it finishes.
No p99 cancellation-latency bound is claimed. Finalization is checked again
before handing off success, and a cancelled wrapper cannot return a completed
native result as successful.

## Validation scope

The added unit cases exercise completion ownership, discarded queued captures,
output lifetime, cancellation-cause mapping, re-entry, identity comparisons and
memory arithmetic. A separate feature-gated integration target uses the actual
process runtime/blocking pool for missing-model failure, pre-cancellation,
reservation refusal and synchronous re-entry, checking drain and ledger balance.
It requires no model file and cannot produce an inference-success fixture.
Rust compilation/tests, controller DSR, real-model runs and performance
measurements were not executed in the implementation session.

## Whole-corpus methods with built-in admission

`NlpEngine::batch_int8_chat` accepts a shared `Int8ChatPlanner`, optional
`GenerationBatchArgs` defaults, `Int8BatchLimits`, `CorpusLimits`, and owned
reader/writer values. `NlpEngine::batch_int8_extract` takes an owned
`Int8ExtractionBatchPlanner`, shared `ExtractionVocabulary`, corresponding
`Int8ExtractionBatchLimits`, the same corpus envelope, and owned IO. Both take
a charged `ResidentInt8` and explicit `CancellationToken`.

Unlike the lower-level adapters, neither method asks the caller to implement
an admission trait. Both use a concrete admission provider backed by the same
`EngineLease` and process ledger. Each stream builds one native engine, then
retains its KV/workspace and the model/vocabulary across all documents. There
is one blocking crossing and one coordinator-only native scope for the entire
bounded corpus, not one per document. No neural batch-M, cache-sharing or
parallel-throughput claim is implied by resident stream execution.

`CorpusLimits` combines `NativeLimits`, the existing bounded `BatchLimits`,
an explicit preparation reserve and an explicit IO-buffer reserve. Both
reserves must be nonzero. A conservative checked staging model additionally
prices line/JSON copies, complete output-line staging and the epoch-ID tree.
Source/schema/TaskIR compiler memory and any whole-corpus reader or collecting
writer must be priced by the embedder; a cursor containing the whole input is
not assumed free. These remain modeled ledger commitments, not an allocator
interceptor or proof that unknown reader/writer/allocator behavior obeys a
measured RSS ceiling.

The stream's buffers and preparation reserve are owned together before the
pool admission. Per-document admission checks the actual resident model,
strict profile, full KV charge, sampler capacity and maximum result size. It
then reserves result storage in the real ledger. The output remains a live
reservation through allocation and final write/flush; because admission
precedes allocation, it does not falsely claim an allocation has committed.
The existing GuardedOutput drops result bytes before the reservation's explicit
abort/release. Corpus-wide KV and sampler storage are not charged again for
every item. Model and mask work retain their existing nonrefundable ledgers,
and epoch flush cannot renew wall-time/checkpoint limits or memory authority.

Readers and writers must be `Send + 'static`; they move into the real blocking
closure, so there is no unsafe borrowing across threads. They are dropped
before the completion signal. A discarded queued closure captures its complete
drop-ordered package rather than independently capturing buffer/guard fields.
The public API returns only the fixed-size `BatchSummary`, not an unaccounted
collecting writer. External sharing/retention performed by a caller-supplied
IO implementation remains that caller's responsibility.

A returned summary means the runner reached EOF and flushed its terminal
record; `summary.failed` still reports document refusals. `HostedError::Batch`
retains the actual `BatchRunError` and summary. Broken output is never retried
or followed by an invented error record. Runtime-wrapper failure or late
cancellation can still make the enclosing call fail after bytes were delivered;
those bytes are not retracted and no additional terminal record is appended.
Callers must check both host completion and per-document status. This is not
an exactly-once delivery, downstream acknowledgement or durable-job API.

Additional regression cases cover concrete admission inputs, profile/model
mismatch, complete KV/sampler accounting, output limits, checked corpus memory,
retained summaries, and discarded-closure capture order. The shared rejection
constructor now accepts either BatchCode or BatchFault, fixing the extraction
planner's incompatible call without discarding typed fault metadata. The
existing framing/runner body is otherwise byte-for-byte unchanged. These added
Rust cases also remain unexecuted in the implementation session.
