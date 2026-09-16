# Shared-weight layer-major eager batches

Status: source implementation. Rust compilation, regression execution, real-model
parity, physical memory measurements and throughput measurements are UNRUN.
This does not activate a CLI/model loader or claim a qualified production
scheduler. The existing ordered NDJSON processor remains whole-item serial.

## Concrete native path

`native_engine::batchsched::EagerBatchEngine` borrows ONE immutable
`HfBf16EagerWeights`. It neither clones a model per sequence nor runs the scalar
engine once per entire document. A finite arena owns independently sized bf16
K/V caches, each with all 44 logical slots. No threads, runtime, private process
broker, model download or activation path is created.

`LoopRunner::run_group` consumes the same schedule walker as `run_token` and
`run_token_structural`: 22 shared physical layers, shared final norm, the same
22 layers again, then the same final norm. There is no second embedding lookup
or loop-boundary projection. Its group callback intentionally has no scalar
PositionContext that could be mistakenly broadcast across ragged rows.

At each logical layer the concrete executor performs grouped Q/K/V projections,
per-row RoPE/KV append/eager 48:8 GQA, grouped O projection, attention residual
and norm, grouped gate/up/down projections and MLP residual. Every linear group
visits each output-weight row across all active sequences before advancing to
the next weight row. Each sequence uses the same per-row iterator sum expression
and named bf16 cast sites as the existing scalar matrix reference. GQA reads
only that sequence's cache; context length is not a whole-layer grouping key.

This is an output-row-major scalar reference, not an ISA-specific tiled GEMM.
It does not prove fewer physical DRAM reads, lower RSS, a strict-profile token
parity award or higher throughput. Integer/native SIMD work and empirical
crossover selection remain separate requirements.

## Sequence ownership and failure

An arena-scoped generation-checked BatchSequence identifies a cache slot, not
caller text, request_seq or a sampler address. Handles from another arena or a
prior slot occupant are refused. A step rejects duplicate handles, invalid token
IDs, full context or divergent logical slot lengths before modifying any row.
A successful step advances every participating cache slot exactly once. Rows may
be omitted/reordered freely while maintaining their own cache/RoPE positions.

After computation starts, a scope guard retires ALL participants on an error,
cancellation or unwind. This discards their partial request prefixes instead of
pretending a partially executed 44-slot step is reusable. Untouched siblings
remain valid. Logical clear does not zero backing memory. Reopening a slot
changes its generation and cannot authorize an old handle.

Checkpoints occur between stages, before each projected weight row and before
each row's bounded attention call. Attention retains the existing eager
primitive's granularity; there is no assertion of tile-level interruption inside
that primitive. The embedding host owns physical worker supervision and must
retain its real admission resources until synchronous work and delivery drain.

## Payload and work accounting

BatchEnvelope validates all capacities before allocating cache/table payload.
The first reference route caps each sequence at 8192 positions and the arena at
128 rows; these are finite API bounds, not throughput recommendations. Admission
still depends on the complete requested payload, not the row count alone.

The estimate reports the sum of every row's complete bf16 K/V capacity, one
shared maximum-cap RoPE table, a conservative sum of stage/primitive scratch
payloads, and full f32 logit capacity for every arena row. This is NOT an aggregate
process certificate or RSS estimate. The embedding host additionally owns the
borrowed weights, metadata/allocator overhead, safety reserve, decoder-internal
work, serialized envelopes and retained prior outputs. Buffers still allocate;
no allocation-free hot-loop claim is made.

Intermediate prefill inputs can explicitly request no lm-head projection. Full
logits and greedy selection are absent only for those rows. Step work counts
forward positions, logical layer/linear groups, and full-vocabulary rows. Logical
groups are not mislabeled physical traffic or hardware measurements.

## Batched generation

`native_engine::generation::batched::{preflight, execute, execute_with_sink}`
connect the engine to the same private Cursor now used by scalar GenerationPlan.
There is one implementation of addressed sampling, processors, byte decoding,
EOS/minimum length, exact retained stop suffixes, output limits, raw full-model
logprobs and reserve/permit token delivery. The compiled semantic key is not
changed merely because execution uses a batch or another physical slot.

Every cohort request carries its exact admitted identity, empty arena slot and
unique nonzero delivery sequence. Preflight checks every request plus the common
loaded model/tokenizer/backend identity before opening any slot or constructing
sampler state. The host supplies the identity of the actual loaded model; the
module does not manufacture or attest model identity. Different prompts, tasks,
policies, seeds and sample indexes remain row-local. The common template digest
is conservatively retained because pinned chat binds tokenizer assets there.

Each scheduler tick advances one token per unfinished sequence. A short prompt
can begin decoding while longer prompts continue prefilling. Intermediate prompt
positions omit lm-head work; final prompt and feedback positions compute a full
vocabulary row. Finished rows are closed independently and never fed back again.
Final outputs retain input order. Every token event contains its own request_seq
and row-local token_index, not a global batch token counter.

The result execution label is `eager-addressed-batch-generation-v1`. Its actual
projected_logits counts selection forwards only. The scalar label remains
`eager-addressed-generation-v1`, counting every prompt/feedback projection.
Both label their raw token logprob space identically because every selected token
still has its actual full-vocabulary normalizer. Planned batch work counts worst
case forwards and max_new_tokens full projections per row; early EOS/byte limits
do not grant another row a hidden enlarged allowance.

Streaming is token-committed, not atomic across a whole cohort. A later error or
cancellation can follow already delivered tokens, but returns an error rather
than a successful partial cohort, retracting bytes or retrying delivery. Cohort
scope cleanup closes every handle it opened on success/error/unwind, including
rows outside the final failed native group. The caller retains actual output
admission and supervises the sink's truthful reserve/permit contract.

Regression sources exercise per-row scalar/batched projection equality, ragged
state/routing, shared loop boundaries, handle reuse, partial-step retirement,
seeded scalar/batch/reorder equivalence, mixed prefill/decode ticks, independent
EOS/byte stops, exact selected-logit work and typed cancellation. These are
synthetic mechanism tests and remain UNRUN, not evidence of model qualification.
