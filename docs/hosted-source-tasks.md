# Process-hosted INT8 source tasks

With `asupersync-runtime`, `NlpEngine::execute_int8_source` takes a charged
`ResidentInt8`, `PreparedInt8SourceTask`, `Arc<ExtractionVocabulary>`,
`hosted::SourceLimits`, and `CancellationToken`. It returns
`HostedOutput<Int8SourceTaskRun>`. The four task variants use the existing pinned
source planner and typed finalizers; source/manifest fields are never retokenized
or repaired during execution.

`SourceLimits` contains the existing `NativeLimits`, an explicit nonzero
`preparation_reserve_bytes`, `MaskWorkLimits`, and a whole-invocation
`max_mask_node_visits`. The preparation estimate must cover the transferred compiled
source/grammar/passage metadata and pinned vocabulary allocations retained through
Arcs. This is a caller-priced commitment, not measured RSS or allocator interception.

The host checks the resident resource domain and physical model identity, full
native context and full allocated KV against the task ceiling. It reserves
preparation, KV, scratch and output through the real process ledger. A single
whole captured `Charged<SourceInput>` keeps the input storage ahead of its memory
charge even when the invocation is cancelled before starting.

The existing process-owned runtime and blocking coordinator execute the task; no
new pool, artifact loader, model tensor copy or separate admission authority is
introduced. Output allocation is committed while its reservation is still live;
a commit failure drops the result before releasing the charge. Native storage and
preparation captures drain before physical completion. Only the owned output and
its committed memory guard escape.

`HostedError::Source` retains the original `Int8SourceError` and typed cancellation
cause while its Display/Debug omit arbitrary nested diagnostics. The existing
hosted preflight, non-reentrant execution, coordinator-only scope, queue-time run
limits and physical-drain rules remain in effect.

This adds a single-request host route. It does not add source NDJSON corpus
execution, public model-backed CLI activation, artifact qualification, numerical
parity or quality/performance evidence. The Rust source tests were not compiled
or run in this session; runtime validation belongs to the authorized controller.
