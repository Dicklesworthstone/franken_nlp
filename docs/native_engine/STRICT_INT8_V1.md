# Executable portable int8 model candidate

This increment implements the missing integer-weight model forward, not another
artifact format or model loader. It is **code-first library implementation** for
the `strict-quantized-v1` candidate, identified more specifically by
`portable-int8-bf16-rails-eager-gqa-v1`. It does not ratify OQ-30, resolve OQ-31,
close Bead `7p1s`, or establish HF parity, quantized quality, native performance,
production admission, or release readiness.

## Data path

`Int8WeightView::from_materialized` borrows the existing bridge's
`ArtifactWeightSet`. It requires every one of the 201 one-model tensors: the
BF16 embedding, 45 BF16 norm vectors, 154 decoder int8 matrices, and the untied
int8 head. Missing, extra, transposed or wrongly typed tensors refuse. Each
int8 view checks all positive finite scales and recomputes every semantic row
sum before projection. No matrix is cloned or expanded to floating weights.
The materializing bridge is still synthetic/non-authoritative. Its real-artifact
opener continues to refuse; this route neither authenticates the supplied
identity nor bypasses the production loader.

`StrictInt8Engine` allocates one 44-slot BF16 KV cache, admitted-cap RoPE tables,
and a reusable activation-quantization rail. An exclusive `Int8Session` executes
exact input IDs through the existing `LoopRunner`: 22 physical layers, final
norm, the same 22 physical layers, final norm. Logical KV slots are distinct
across passes. Both passes use the same token, causal and rotary positions.

Every projection dynamically quantizes its input using a signed symmetric
scale, nearest-even rounding and zero point zero. Q/K/V share one quantized
activation; gate/up share another. Integer dots and scale application use the
existing canonical scalar functions. Decoder projection outputs narrow to
BF16. Embedding, norm, residual, SwiGLU, split-half RoPE and eager GQA use the
existing BF16-rail reference primitives. The int8 lm-head exports its f32
fixed-order dequantization directly, without pretending to be HF logits.
This explicit numerical choice is a candidate requiring independent fidelity
budgets, not an inherited HF-match claim.

`append` advances exactly one token without a head projection. `logits` projects
the full head or an ascending unique row subset from the last completed hidden
state. `prefill` preflights the whole nonempty input and projects only the final
position. No intermediate full-vocabulary prompt rows or all-layer trace copies
are retained. Selection is real row slicing, not a full projection followed by
gathering. The caller may feed generated tokens back through `append`.

## Bounds and lifetime

The host must already admit the source-owned weights and must retain its actual
resource guards. `Int8MemoryBudget` separately caps KV, RoPE and conservative
reference scratch payload before engine allocations. Payload arithmetic excludes
allocator slack, metadata, source weights and externally retained result vectors;
it is not an RSS certificate. Context is deliberately capped at the existing
8192-position baseline rather than the source configuration's larger maximum.

`Int8Work::for_sequence` prices every integer decoder projection, requested
head rows, and causal attention pairs across all 44 executions. It includes the
existing prefix length, so appending to a long prefix is not priced like an
empty-context decode. `Int8RunBudget` limits forward positions, attention pairs,
and both projection rows and integer multiply-accumulates. Whole-sequence
preflight happens before execution; each native projection also reserves its
own complete slice before work. Failed or cancelled attempts receive no refund.

The session clears every logical KV slot on drop and retains allocated capacity.
A native error, cancellation or unwind leaves the engine poisoned; it is not
silently reusable. A forgotten session leaves it active and refuses readmission.
Input/shape/work refusals before native work preserve an existing valid prefix.
Projection polling occurs at bounded row groups, with additional checkpoints
at layer, norm and attention boundaries. The inherited eager-attention primitive
is not preempted inside one call. No runtime, worker, signal handler, retry,
network request, or new admission broker is created.

Reference norm and attention helpers still allocate bounded temporaries. This is
a semantic baseline; it does not meet the future allocator-free hot-loop or
SIMD-throughput gates merely because the integer projections avoid allocation.

## Library outline

The caller supplies the existing materialized weight set, exact token IDs and
one genuine cancellation controller. For a prefill with one complete head:

```rust,ignore
let weights = Int8WeightView::from_materialized(&materialized.weights)?;
let need = Int8MemoryRequirement::for_context(context_positions)?;
// The host must admit these payloads PLUS its actual overhead and weights.
let memory = Int8MemoryBudget {
    max_kv_bytes: need.kv_bytes,
    max_rope_bytes: need.rope_bytes,
    max_scratch_payload_bytes: need.scratch_payload_bound,
};
let mut engine = StrictInt8Engine::new(weights, context_positions, memory)?;
let work = Int8Work::for_sequence(0, prompt.len(), NANBEIGE_VOCAB_SIZE)?;
let mut session = engine.session(Int8RunBudget::exact(work), &mut control)?;
let logits = session.prefill(&prompt, LinearRows::All)?;
let completed = session.work();
drop(session);
```

## Evidence

The projection and model-core increments add 28 Rust regression tests covering
activation quantization, full-domain integer algebra, true row slicing, work
limits, cancellation, fixed tensor geometry, 44-slot cleanup, poisoned-session
behavior, variance overflow and aggregate decoder/attention pricing. They are
written, **not executed** here. The full model forward has not been run.
Independent Python specification checks passed 280 projection cases, 100000
ledger transitions and 10000 work-composition cases. Those checks are not Rust
compilation, DSR evidence, native execution, model parity or task-quality proof.
The repository's clean-SHA DSR and model-present gates remain independent.
