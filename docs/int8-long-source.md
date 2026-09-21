# Long-document INT8 source execution

`tasks::source_planning::quantized::long` connects the existing lossless source
partitioner and ordered map/reduce coordinator to the real native INT8 source
portfolio. `SourceTaskPlanner::plan_int8_map_with_control` prepares NER,
keyphrases, or cited summaries for every source chunk without loading weights.
`PreparedInt8SourceMap::execute_with_control` consumes that complete commitment
on an already admitted `StrictInt8Engine` and vocabulary.

## Meaning of the result

This is **map/merge**, not cross-chunk reasoning. The reducer preserves all native
chunk results in original-source order; it does not ask the model to synthesize a
global summary, normalize local keyphrase ranks, merge entity types, or infer
missing occurrences. Options such as maximum entities/bullets apply per chunk.
Entities or facts crossing chunk boundaries can be missed. The result declares
`independent-chunks-no-cross-chunk-reasoning-v1` and retains the coordinator's
`SingleContextEquivalenceNotEstablished` warning. Passage QA is excluded rather
than quietly changing the question/evidence contract.

Each `MappedSourceChunk.native` is the unchanged typed native result: its spans
remain chunk-local. `original_spans` provides original-document byte and Unicode
scalar coordinates, keyed by typed entity/keyphrase/citation locations. Every
lift checks exact source bytes, UTF-8 boundaries and the original local scalar
coordinates. Repeated occurrences are retained; no preferred occurrence is
invented. Summary citations retain structural source-membership assurance, not
semantic-entailment certification.

## Planning and execution

Supply a `SourceMapTask`, a per-chunk `TaskBudget`, a pinned `PlanContext`, the
existing `SourcePlanningLimits`, and explicit `Int8SourceMapLimits`. Chunk
limits cover the whole source and partition/tokenizer work. They must leave room
for the trusted prompt and maximum output. Counts come from the real pinned
source encoder. Every complete task prompt is then compiled independently, so
underestimating the scaffold reserve fails before any forward, not by dropping
a document suffix. At most 256 chunks are admitted by this adapter.

The embedding host admits every identity exposed by `execution_identities()`.
The entire identity vector, actual model/profile, resident KV capacity, and each
chunk's KV limit are checked before the first native call. Execution uses one
resident engine with cold logical KV per chunk. There is no model reload, new
runtime, scalar-as-GEMM claim, free-generation fallback or retry.

The complete ceilings for forward positions, projected logits, attention pairs,
integer dot products and multiply-accumulates are summed with checked arithmetic
before execution. Grammar-mask reservations are multiplied across the complete
chunk count before execution. Neither reduction levels nor early EOS renew
these commitments. Every completed native result is checked against exact
prompt/output work geometry before becoming a map value.

The existing coordinator enforces map/reduce call, depth, value, live-value,
cumulative-value and final-envelope limits. The outer INT8 envelope is checked
again. Reductions share immutable results through `Arc`; native text and token
buffers are not deep-cloned at each tree level. Serialized-value accounting
remains conservative; it is not an allocator or RSS certificate. The host must
price retained prepared grammars/prompts, output storage, metadata and allocator
slack separately.

The same cooperative control reaches source planning, native decoding,
coordinate projection and every map/merge call. Bounded tokenizer and serializer
operations are checked at their boundaries, not preempted in the middle.
Any failure returns no partial document result. The host must discard a failed
native invocation and retain its resource/output ownership through actual
delivery, as for the existing single-source task.

## Evidence boundary

This is library implementation, with regression source for real pinned planning
and a private synthetic execution driver. It does not promote artifact
activation, production DSR acceptance, model-present parity, task-quality,
throughput, public neural CLI or durable-job claims. No Cargo, RCH, DSR or GitHub
Actions validation was run when this implementation was authored. Controller
validation remains pending under `WIRING.md`.
