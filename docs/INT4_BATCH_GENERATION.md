# Ragged INT4 batched generation and chat

The explicit INT4 candidate can now execute 1..64 independently compiled
text/chat requests on one `Int4CohortEngine`. This connects the existing
layer-major decoder and shared packed-group projections to the existing
generation cursor, rather than running one complete single-sequence engine
per request. Scalar remains the default unless the embedding host supplies
an explicitly detected `PackedInt4Kernel`.

## Execution semantics

Each scheduling tick advances exactly one token in every unfinished row.
Short prompts may begin decoding while longer prompts are still prefilling.
Only a final-prompt or feedback position projects the full vocabulary head;
intermediate prompt positions do not. EOS, suffix completion, token-limit
completion and a byte-refused proposal retire their row without extra feedback.
Finished rows are not padded, forwarded or sampled while their neighbors run.
All rows begin together; membership is fixed for the lifetime of the cohort.

The shared `Cursor` still owns sampling, penalties, byte decoding, logprobs,
stop rules and two-phase token-event delivery. No second sampler was added.
Each request retains its own semantic key, seed, sample index, token index,
options and output limits. Physical row index, tick and delivery sequence do
not become sampling addresses. Compiled requests must share the actual model,
partition, tokenizer/template assets, numerical profile and backend identity;
their prompt/task/policy fields may differ. Delivery sequence IDs must be
nonzero and unique within the cohort.

One native session owns all sequence caches. Each sequence retains its own
44 KV slots, position and causal attention; both loop passes and both final
norms remain in the existing native engine. The driver validates the complete
returned tick's routing, dimensions and logits before delivering any token
from that tick. A native, decoder, sink or completed-work error aborts the
whole generation run without retries or a successful-prefix result. Session
Drop clears every row on completion/error/unwind; native execution failures
poison the engine. Already delivered streaming events cannot be retracted and
remain provisional until a complete successful task result exists.

## Library surfaces

`native_engine::generation::grouped_int4::batch` exposes
`requirements`, `preflight`, `execute` and `execute_with_sink` over
`Int4BatchGenerationRequest` and `Int4BatchGenerationBudget`. Public execution
accepts only the real native cohort engine; the synthetic driver seam is
private to the implementation/tests. Completed runs are returned in input
order, not completion order, with per-row work and aggregate work/tick counts.
The outer execution label is `grouped-int4-ragged-generation-v1`; individual
runs preserve the existing INT4 semantic generation label and identity.

`tasks::chat::grouped_int4::batch` accepts independently prepared
`Int4ChatBatchRequest` values. It uses the pinned tokenizer and calls the
existing single-request finalizer for every result. Token re-decoding, UTF-8,
control exclusion, stop/sample semantics, complete work and per-task wrapper
bounds are not replaced with batch-only checks. The finalizer additionally
checks input/result routing, aggregate work and group-step count, then bounds
the complete batch wrapper. No successful result is returned if a later
member fails finalization.

The charged embedding entrypoint is `NlpEngine::execute_int4_chat_batch`.
Member/limit types live in `hosted::grouped_int4::batch`. With an already
loaded `ResidentInt4`, prepared requests, explicit context capacities and
finite host-selected limits, an embedding can use:

```rust
use franken_nlp::hosted::grouped_int4::batch::Int4ChatBatchMember;

let members = vec![
    Int4ChatBatchMember { prepared: first, request_seq: 1, context_tokens: first_capacity },
    Int4ChatBatchMember { prepared: second, request_seq: 2, context_tokens: second_capacity },
];
let quote = limits.requirements(&members)?;
let output = host.execute_int4_chat_batch(&resident, members, limits, kernel, cancellation)?;
```

The snippet assumes those existing host/model/preparation variables; it does
not load a model, select unbounded budgets or introduce a CLI command.
`requirements` quotes storage/work but does not reserve it. The execution
entrypoint checks actual resident-model identity and the shared resource
domain, reserves every component, then enters the existing supervised blocking
host once. Prepared prompts, the shared model and memory guards remain owned
until physical completion. Native buffers drain before the guarded aggregate
output is published. The hosted API returns a complete result; provisional
streaming is available at the native/task library layers.

## Admission and resource boundaries

All requests are checked before a native session opens. Each row has its own
nonrenewable compiled forward/attention/group-dot/MAC ceiling, allocated KV
capacity and sampler allowance. Aggregate limits cannot cover an underfunded
member. Native accounts are tightened to the compiled bounds, not replenished
per tick. Each completed row is reconciled against its native projection
ledger and independent causal work, then the aggregate is reconciled too.

The memory quote covers all allocated ragged KV capacities, one shared RoPE
table, the existing grouped native scratch/head allowance, all sampler state
and bounded generation output/descriptor storage. Attention triangles are
summed per sequence; lengths are never concatenated into a cross-document
attention context. Model weights remain shared. Explicit allocator reserves,
caller-owned preparation and host/runtime reserves keep their existing roles.
These are modeled payload commitments, not measured peak RSS guarantees.

`Int4ChatBatchLimits::max_result_bytes` is an aggregate bound and must be at
least the sum of every task's output-byte ceiling plus 16,384 bytes of wrapper
allowance. Each task's KV bound still limits its own allocated capacity even
when the supplied native budget is larger. The hosted path separately charges
raw generation payload and final typed/serialized output because both can
coexist during finalization. Arithmetic overflow or an undersized component
refuses before native allocation or execution.

## Scope and validation status

This is explicit fixed-cohort generation and chat, not automatic daemon
batching, dynamic refill, a heterogeneous grammar scheduler or a new public
model release. There is no new CLI batching flag. The native single-sequence
and explicit prompt-prefill APIs remain available. No ISA body, unsafe policy,
weight recipe, dependency, thread-spawning implementation or automatic dispatch
preference changed.

Fifteen new Rust regression functions cover ragged prompt/decode interleaving,
no extra feedback, 64-row scheduling, independent seeds/stops, comparison with
the existing serial cursor driver under synthetic logits, late malformed
routing/logits, sink/cancellation/work failure, per-row and aggregate admission,
real pinned chat finalization, task KV/output bounds and host accounting.
These are test definitions, not passing native/model receipts.

An independent Python scheduling model passed 1,280 cases and checked 165,791
head-selection placements. It checked exact prompt/feedback coverage, compact
head routing, finished-row omission, maximum-row-length tick counts, independent
attention accounting and permutation behavior. This did not execute Rust,
the real sampler, an ISA kernel or a model. New source bytes were checked for
UTF-8/whitespace and matched to their uploaded Git blob hashes.

Rust compilation/tests, real cohort-versus-sequential model parity, cancellation
interleavings on the production host and performance measurements remain
unexecuted. Controller-owned DSR and task/model-specific qualification remain
separate. No numerical qualification or speedup is asserted.
