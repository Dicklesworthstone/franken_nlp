# Native grouped NDJSON generation

Status: source implementation. Rust compilation and Rust tests are UNRUN in the
implementation environment. A separate Python model of interval-to-slot
assignment passed 93,240 exhaustive small cases; that is algorithm-model evidence,
NOT execution of the Rust scheduler, native model parity, or a performance award.
The real-model NDJSON equivalence test is explicitly ignored/model-gated.

## The connected execution path

`batch::grouped::run_ndjson` now accepts a multi-row processor. With
`batch::grouped::generation::NativeGenerationGroups`, the complete path is:

```
bounded NDJSON frame and canonical parse
  -> existing GenerationBatchPlanner / pinned ChatPlanner
  -> bounded input window and no-refund aggregate work charge
  -> deterministic compatible cache-slot waves
  -> complete-window preflight and real host admission
  -> tasks::chat::batched::execute
  -> shared generation cursors + EagerBatchEngine::step
  -> independent pinned task finalization
  -> original-order doc / doc_error records, canonical write and flush
```

This no longer executes each document through the scalar generation adapter.
Each compatible wave uses the shared-weight layer-major engine, including mixed
prefill/decode positions and selection-only vocabulary projections. One immutable
model remains borrowed throughout. There is no new model loader, runtime, thread
pool, private admission broker, timer, retry or production activation bypass.
The existing `batch::run_ndjson` serial entrypoint remains unchanged. The frozen
robot CLI is NOT silently rewired. This is a callable native library path.

## Bounded input and output

`GroupLimits.max_records` bounds all nonempty document/error records in a window,
not just successfully prepared requests. Width defaults to one and is bounded
by 128 and the processor's declared maximum. Existing line, input-byte, request,
ID-epoch, duplicate-key/depth, output-line, output-byte and aggregate-work limits
still apply. Only empty LF/CRLF frames are ignored. Every nonempty frame retains
its own original sequence, line and byte offset even when errors are buffered
between successful requests. Invalid input is not re-echoed in error records.

Preparation stays model-free and item-bounded. The host must reserve the bounded
window's prepared plans plus one framing/planning scratch item; max_records alone
is NOT an RSS or process admission certificate. The native preflight separately
prices full arena payload, every sampler, retained results, native work and
canonical result limits before any model step.

The blocking reader may wait for a full window. Interactive producers must send
`{"flush":true}`, close input, or select width one; there is no concealed timer or
latency guarantee. A flush drains the prior window and writes/flushes its barrier
before the runner resets duplicate IDs or reads the next epoch. EOF drains the
last partial window before EOF-flush/run_complete. These are local writer
acknowledgements, not remote processing or disk-durability certificates.

Malformed/planning-rejected rows keep their output positions and never enter the
native cohort. Valid plans are charged BEFORE enqueue, without refunds after
early EOS, failure, cancellation, or a later input error. Fatal input/prepare/
control failure discards queued work rather than running inference after abort;
run_error is not a successful partial cohort. Output I/O failure poisons the
stream: no retries, further input, inference, or writes. One bounded window may
already have been read when delivery fails.

Events retain `fnlp-item-local-batch-v1` and its existing field schema, but the new
execution label is `ordered-cohort-no-retry-v1`; no grouped result claims serial
execution. The full returned routing is checked before any document result is
published. Per-row finalization no-results become fixed doc_error categories;
valid siblings remain deliverable. Native failures and identity/routing/work
corruption remain fatal.

## Heterogeneous slots and complete-window admission

A request's minimum cache capacity is its complete worst-case forward length.
Its maximum is its own TaskBudget K/V ceiling divided by the fixed 176-KiB/token
price. Slots are therefore matched to inclusive capacity intervals, not merely
chosen because they are big enough. A larger free slot can still violate a
small request's budget.

The allocator orders requests by upper capacity bound, then minimum and original
index, choosing the smallest fitting unused slot. A feasible single-wave matching
is preserved. Individually feasible leftover requests use later waves: multiple
long documents do not fail just because only one slot is large. Wave order and
slot reuse never alter the compiled semantic sampling address. Assignments are
restored to original row order for each native call and final transport delivery.
Only explicitly selected slots are used; other resident sequences are untouched.
No minimum-wave-count, DRAM reduction or throughput claim is made.

Every wave is priced before execution. The aggregate counts the complete shared
arena once (including unused slots), but conservatively SUMS all window sampler
and output payload bounds. Retained earlier-wave results therefore do not hide
behind a per-wave allowance. Work limits also apply to the entire window. The
sum of complete canonical native wave envelopes is capped by max_result_bytes,
in addition to independent task and final NDJSON limits.

`GenerationCohortHost::admit` has no default implementation: the embedder supplies
its existing real process/model/request admission. It receives every private
identity and delivery coordinate, deterministic wave/slot assignment, complete
slot reservation, sampler/result bounds, aggregate native payload and total work.
It returns every actually admitted identity in original request order plus one
real guard. ALL identities, including the last wave, are checked before the first
forward. No receipt or identity is fabricated by the production adapter.

`GroupedOutput` retains that guard until all returned rows have been serialized,
written and flushed, or discarded after failure/unwind. Result fields drop before
the guard. Native wave handles are already closed before the next wave can reuse
their slots; prior output storage remains covered until transport drain. The host
also owns borrowed weights, allocator/metadata overhead, decoder internals and
canonical/transport serialization staging.

## Embedding shape

The caller supplies an existing engine, its loaded identity, a pinned planner,
selected empty slots, configured budgets, I/O and its real cohort host:

```rust,ignore
use franken_nlp::batch::{BatchLimits, grouped::{self, GroupLimits}};
use franken_nlp::batch::generation::GenerationBatchPlanner;
use franken_nlp::batch::grouped::generation::NativeGenerationGroups;

let compiler = GenerationBatchPlanner::new(&planner, defaults)?;
let mut processor = NativeGenerationGroups::new(
    compiler, &mut engine, loaded_identity, selected_slots, admission_host,
    cohort_budget,
)?;
let summary = grouped::run_ndjson(
    &mut input, &mut output, &mut processor, BatchLimits::default(),
    GroupLimits { max_records: 8 }, &mut control,
)?;
```

The example deliberately does not invent a production host, load unchecked source
weights, or grant admission by cloning request identities. Native output remains
on `eager-addressed-batch-generation-v1`, with actual full-vocabulary projection
counts only at selected-token positions.

## Tests and remaining boundaries

Transport regression sources cover width-one/multi-row/partial windows, malformed
input ordering, duplicate epochs, flush barriers, no-refund work, result limits,
failed flush poisoning, fatal native errors, cancellation, corrupt routing and
result-before-guard destruction on return and unwind.

Native adapter sources cover exhaustive small interval matching, reusable long
slots, task-specific upper K/V bounds, invalid slots/shapes, aggregate prices and
overflow, delivery coordinates, private admitted-identity matching, and no-result
translation. The explicitly ignored real-model test
`real_model_grouped_ndjson_matches_width_one_without_cloning_weights` compares
complete canonical output between width one and width two on the same borrowed
weights. Its test-only synthetic host is not production admission. It loads the
authenticated pinned closure only in that test, using `FNLP_BATCH_SOURCE_DIR`, and
reports SKIPPED_NO_MODEL when the variable is absent. It is UNRUN.

This remains bounded cohort/wave scheduling, not continuous mid-wave slot refill,
wall-clock decode preemption, an ISA-tiled matrix kernel, a durable job, or a CLI
activation change. Real-model parity and physical throughput/memory measurements
are separate outstanding gates. No existing empirical bead is closed by source
or model-only algorithm checks.
