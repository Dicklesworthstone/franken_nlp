# Portable int8 autoregressive generation

This is a code-first **library path**, not production model activation or a new
CLI command. It connects the portable int8 model to the existing generation
compiler, addressed sampler, processors, stop rules and two-phase token stream.
Rust compilation and tests, full native model execution, DSR, quantized fidelity,
quality, performance and release qualification remain unproven by this increment.
The source bridge remains synthetic/non-authoritative and its production opener
still refuses. No artifact format authority, host admission or model authenticity
is invented by these APIs.

## Workflow

Use `native_engine::generation::quantized::Int8GenerationPlan::compile` with exact
already-tokenized prompt IDs, the existing `GenerationOptions`, a stable semantic
item ID, sample index, planning limits and the caller's complete execution
identity. The identity must already name `strict-quantized-v1`, BF16 KV, disabled
thinking/tools and backend `portable-int8-bf16-rails-eager-gqa-v1`. A BF16 identity
is rejected, never silently rewritten. Compilation binds options, prompt, item,
sample and the distinct generation strategy before admission.

The resulting plan exposes its exact private execution identity, complete
`Int8Work` ceiling and sampler workspace bytes. The host admits that identity
and those resources, including the existing engine's complete resident KV
capacity, source-owned weights, scratch and allocator overhead. Pass an existing
`StrictInt8Engine`, genuine admitted identity, byte decoder, limits and one
cancellation controller to `execute` or `execute_with_sink`.

The driver compares the plan to the actually borrowed model view's model ID,
source revision, quantization recipe and logical model digest. It cannot certify
packing or publisher authenticity because the current bridge carries no such
capability. Other host-owned identity fields remain exactly bound rather than
being repaired or manufactured. A separate `Int8GenerationPlan` type prevents
these requests from entering the public eager or grouped-eager driver APIs.

## Execution and output

A single exclusive native session feeds every prompt token through all 44 layer
executions. Only the final prompt token projects the untied head; intermediate
prompt tokens do not compute discarded full-vocabulary rows. Each subsequent
selection uses one real feedback forward and a complete head projection. The
last selected token, including terminal EOS, is never needlessly fed back.

The existing cursor remains the only sampler/stop/stream state machine. Greedy
selection and explicit seeded sampling retain the existing repetition, presence,
frequency, bias, temperature, top-k and top-p processor order. No global RNG,
new tokenizer, alternative template renderer, worker or runtime is introduced.
The existing eager compiler's identity and sampling-key framing remain unchanged.
Quantized execution has a distinct strategy identity and does not impersonate
HF logits or promise the same tokens as the BF16 oracle.

EOS is included in token IDs and optional raw full-vocabulary log probabilities,
but contributes no content bytes. Byte-stop suffixes remain in delivered content.
A byte-budget-refused proposal consumes actual forward/projection/sampling work
without becoming a committed token. Event deltas concatenate to returned content;
no bytes are retracted and uncertain sink delivery is never retried. The existing
`max_output_bytes` is a content-byte bound, not a serialized-envelope byte bound.

`Int8GenerationRun` returns the completed `GeneratedSequence` and full native
`model_work`: decoder/head dot products and integer multiply-accumulates, causal
attention pairs, forward positions and projected head rows. The driver compares
these native counters against the completed cursor schedule before returning a
successful result. Counts are algorithmic work, not measured hardware traffic,
latency, throughput, physical instructions or a quality score.

## Refusal, cancellation and cleanup

The whole prompt-plus-maximum-feedback closure is preflighted before native work.
Budgets price all seven decoder projections in every logical layer, requested
head rows, quadratic causal attention, resident KV and the existing sampler
workspace. The context cap remains the engine's explicit 8192-position baseline.
Early stopping does not refund a native attempt or create a renewed allowance.
The host owns aggregate admission across distinct calls and must not treat a
per-call budget as a process-wide reservation.

The sampler and native engine borrow the same controller sequentially; there is
no second deadline/cancellation object. Cancellation retains its typed cause.
A native, decoder, stream, allocation or work-consistency failure aborts the
session, rather than returning a success-shaped partial result. Logical KV is
cleared on drop. Errors after model work and unwinds in callbacks between native
calls poison the engine, so a host that catches an error cannot silently reuse
uncertain state. Stream delivery may already have published earlier tokens;
this path makes no rollback or exactly-once delivery claim.

The caller retains genuine host resource guards through the call and any later
result serialization/write. This library does not fabricate an admission guard,
catch and relabel panics, contact the network, load source weights, or promote
an unqualified artifact into production.

## Evidence scope

This increment adds 16 synthetic-logit regression tests through the real shared
compiler/cursor/driver plus a native-session unwind-cleanup regression. Together
with the projection/model-core increments there are 45 new Rust tests. They are
written, not executed in this session. Coverage includes identity substitution,
eager key compatibility, final-only prompt projection, EOS, byte refusal, cross-
token stops, seeded transport-order independence, every budget axis, cancellation,
native/stream failures, corrupted logits and mismatched work counters.

An independent Python schedule specification passed 30000 randomized generation
cases, checking forward/feedback positions, head counts, EOS/stop/byte behavior,
event concatenation and complete work bounds. This is not execution of the Rust
implementation or model, and cannot substitute for clean-SHA DSR or model-present
fidelity evidence. No Bead or profile/release gate is closed by this source delta.
