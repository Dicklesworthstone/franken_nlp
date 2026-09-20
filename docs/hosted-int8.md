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
