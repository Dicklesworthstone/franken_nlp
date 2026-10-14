# Full-model grouped INT4 candidate

`native_engine::strict_int4` connects the existing compact INT4 matrices to
full-model source conversion and the shared two-pass native loop runner. It is
an explicit code-first numerical candidate, not a canonical fnlpq artifact,
production activation path, INT8 alias, or BF16-fidelity award. The existing
INT8/eager paths and ordinary CLI behavior are unchanged.

## Source conversion

`Int4ModelWeights::from_prepared` consumes a borrowed `PreparedConversionInput`
created by the existing pinned ten-file source preparation path. It uses that
session's sealed retained shard handles and bounded row reads; it never reopens
a shard by path, materializes a whole BF16 model, or creates an output file.

The complete 201-tensor census is checked before payload reads. Embeddings and
all norms stay BF16; the seven projections of each of 22 physical layers and
the untied head become 155 `GroupedInt4Matrix` values. Each uses the existing
explicit `bf16-s4-symmetric7-group-f32-rne-v1` recipe with one selected group
width. There is no calibration-based recipe selection or mixed-dtype fallback.

`Int4ModelLoadRequirement` prices all retained tensor payloads together, one
raw source row plus the converter's BF16 row, source values, conversion groups,
and two retained-shard revalidation passes. An inadequate budget refuses before
rehash, reads, or weight allocation. Per-matrix budgets are internal subdivisions
of the admitted whole-model bound, not separately renewable allowances.

At group width 128, the formula yields 2,965,051,392 retained tensor bytes and
43,008 bytes for simultaneous conversion row buffers. These are computed
payload sizes, NOT measured process RSS. The host separately admits source
preparation/metadata, the existing source-session hash workspace, fixed stack
storage, allocator overhead, engine buffers, and its own retained outputs.
Smaller group widths may require substantially more scale storage.

The prepared source must not be modified concurrently. Retained handles prevent
pathname substitution; rehashing before and after traversal detects persistent
changes, not an adversarial mutate/restore race. Existing shard rehashes and
individual reads are synchronous and not preemptible. Cancellation is checked
around them, throughout row/group conversion, and in bounded content-hash chunks.
A failure returns no partial model.

The private-field `Int4ModelIdentity` binds the prepared source root, pinned
revision, candidate profile, group width, tensor roles, and actual converted
weight/scale and BF16 bits. Kernel choice is excluded. This candidate content
fingerprint is not a canonical fnlpq logical-model digest. The explicit
generation adapter below binds it into the central `ExecutionIdentity` only
under its separate source-candidate format and profile namespace.

## Execution

`StrictInt4Engine` borrows the immutable complete weight set. Its constructor
checks the context/KV/RoPE/scratch budgets before allocation. Scalar is default;
`with_kernel` accepts an already real-host-checked `PackedInt4Kernel`. There is
no implicit fastest-ISA selection, model loader, runtime, or worker spawning in
the engine. The host still owns request admission and execution supervision.

`Int4Session` supplies token append, token-ordered prefill, and full or selected
head-row projection. It executes the common `LoopRunner` with 22 layers, final
RMSNorm, another 22 layers, and final RMSNorm. All 44 logical KV slots must reach
the same next position. Q/K/V reuse one quantized activation, as do gate/up;
RoPE, eager grouped-query attention, residuals, norms, and SwiGLU use the existing
BF16 primitives. Each INT4 group is scaled before ordered F32 summation; BF16
rounding happens after the complete decoder projection. The head exports F32.

`Int4Work::for_sequence` counts actual group dots, unpadded MACs, all two-pass
projections, and position-dependent attention pairs. Preflight checks an entire
prompt/head closure before native mutation. Per-projection work is charged
through the existing no-refund ledger. A native failure, cancellation, or unwind
poisons the engine; session drop clears every logical slot while retaining
allocated capacity. Forgetting a session leaves the engine unavailable.

```rust
use franken_nlp::{artifact::converter::PreparedConversionInput, native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    portable_int8::LinearRows,
    strict_int4::{Int4ModelWeights, Int4ModelLoadRequirement, Int4ModelLoadBudget,
        StrictInt4Engine, Int4MemoryRequirement, Int4MemoryBudget, Int4Work, Int4RunBudget},
}};
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
// Exact-token numerical demonstration. Production hosts supply real admitted
// controls, actual memory allowances, and their owned blocking/runtime seam.
fn two_tokens(source: &PreparedConversionInput) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let need = Int4ModelLoadRequirement::for_group_width(128)?;
    let mut control = Continue;
    let weights = Int4ModelWeights::from_prepared(source, 128, Int4ModelLoadBudget {
        max_payload_bytes: need.peak_payload_bytes()?, max_source_values: need.source_values,
        max_groups: need.groups, max_shard_revalidation_bytes: need.shard_revalidation_bytes,
    }, &mut control)?;
    let memory = Int4MemoryRequirement::for_context(2, 128)?;
    let mut engine = StrictInt4Engine::new(&weights, 2, Int4MemoryBudget {
        max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
        max_scratch_payload_bytes: memory.scratch_payload_bound,
    })?;
    let work = Int4Work::for_sequence(128, 0, 2, 3)?;
    let mut session = engine.session(Int4RunBudget::exact(work), &mut control)?;
    Ok(session.prefill(&[1, 2], LinearRows::Selected(&[0, 42, 166143]))?)
}
```

## Remaining gates

Rust regressions cover aggregate model accounting, every group partition,
source-value decoding, identity separation, sequence-cost composition, memory
admission, cleanup/unwind, and the actual grouped projection adapter. An ignored
model-present regression compares prefill with append, sparse with full logits,
and fresh sessions using the full source-backed candidate. Its exact name is
`model_present_prefill_append_sparse_full_and_session_cleanup_agree`; the
controller must arm it on an admitted model-present DSR host with
`FNLP_INT4_TEST_SOURCE_DIR` and `FNLP_INT4_TEST_SOURCE_MANIFEST`. Missing variables
fail when that test is explicitly selected; an ignored test is not model proof.

## Identity-bound generation

`generation::grouped_int4::Int4GenerationPlan` connects the real engine to the
existing generation compiler, sampler, processors and streaming cursor. The
central numerical profile is now `strict-quantized-v2`; version 1 stays INT8.
This naming change also changes the model content fingerprint, which includes
the profile. It is a distinct candidate program, not a quality ratification.

The host explicitly chooses that profile, `STRICT_INT4_EXECUTION`, and BF16 KV,
then calls `bind_candidate_model` with the actual model identity before request
compilation. That call preserves host-owned tokenizer/template/task fields and
binds source revision, complete content, group-specific recipe, and packed
layout under `fnlp-int4-source-candidate-v1`, never `fnlpq-v1`. Compilation
requires the same group width. Execution checks every admitted identity field
and compares the bound content and partition with the actual borrowed engine.
Neither the eager nor INT8 plan type accepts this profile.

Greedy and explicitly seeded generation reuse the same EOS, penalties, top-k,
top-p, minimum length, suffix-stop, raw full-vocabulary logprob, byte budget,
and reserve/permit output code as existing generation. Only the final prompt
position projects the head. Each generation run prices all decoder/group/head
work, position-dependent attention, complete KV capacity and sampler storage
before opening its session. The result includes actual `Int4Work`, checked
against the cursor's completed forwards and projections. Source text and exact
prompt tokens are not exposed by the private plan.

Decoder/sink/native errors abort the session without retry; unwinds clear KV
and poison the engine through the existing guard. Earlier stream events may
already have been delivered, but a failed run never returns a success receipt.
The host owns tokenizer/decoder authenticity, runtime admission and publication.

Code landing does not establish compilation, native execution, whole-model
quality, performance, or platform evidence. Canonical artifact serialization,
ordinary CLI/NlpEngine activation, chat-template integration, structured NLP
tasks and grouped generation scheduling remain separate work. This module
does not add a `--int4` switch or promote a release.
