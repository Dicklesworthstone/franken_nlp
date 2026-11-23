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

## Compact source encoding

A planner configured through
`SourceTaskPlanner::with_map_document_encoding(UntrustedEncoding::SafeBpeV1(...), SafeBpeWorkLimits { ... })`
supports compact NER, keyphrase, cited-summary and passage-QA maps, including
the existing exact keyphrase and summary reducers. QA also charges question
sizing and every repeated question/manifest segment in its native plans.
A one-request encoding policy without a
complete-operation ledger is refused by both map capacity and map planning.

All candidate partition probes, including candidates discarded while shrinking,
and every final chunk compilation debit the same cumulative input-byte,
output-token, pair-check and pair-byte limits. Reusing the planner for hosted
replanning preserves spent work. The policy and complete limits bind each
execution identity. Actual encoded counts determine context fit; byte/scalar
source coordinates and verified citation lifting retain the original document.

Use `map_document_encoding_work()` for cumulative usage and
`map_document_encoding_limits()` for the admitted totals. The hosted map and
exact reducers add `map_document_encoding_workspace_bytes()` to their input
reservation, alongside source capacity and the caller's full preparation reserve.

Exact-schema extraction uses the corresponding
`Int8ExtractionBatchPlanner::with_map_document_encoding` method and the same
four cumulative work ceilings. Schema sizing, discarded source probes, every
final schema/source prompt and hosted reconstruction all debit that planner's
single ledger. Each final plan retains the declared policy and identity even
as remaining work decreases. All per-call and complete-operation caps bind the
extended template identity; original schema bytes and decimal constants remain
exact. Concurrent planning on one ledger and record preparation through a map
planner refuse instead of silently restarting work.

`NlpEngine::extract_int8_document` adds the map encoder/ledger workspace beside
source and schema capacities and the entire retained preparation reserve.
`map_document_encoding_work`, `map_document_encoding_limits` and
`map_document_encoding_workspace_bytes` are available on the extraction planner
as well. Complete-document extraction corpora and retained extraction jobs
retain their separate byte-only admission. Chunk JSON remains independent;
compact encoding does not authorize a global object merge.

Neural answer and summary synthesis accept the same complete-operation compact
policy. Prepared synthesis retains the original encoder operation across the
discovery phase and final verified-evidence plan. Summary and question hierarchies
use that operation for every evidence-context fit probe, including rejected
groups, and every node plan. Question fit includes the original question and
canonical passage manifest at every level. Passing a fresh planner with an equivalent policy to execution
cannot replenish the retained ledger. All dynamic plans recheck their actual
token fit and identity; exhaustion refuses the complete operation without a
byte fallback. Empty evidence skips synthesis encoding and native execution.

The hosted synthesis and hierarchy entrypoints reserve encoder workspace beside
the retained plans and source/evidence buffers. Their private completion
commitments include the authenticated discovery result. Original source bytes
and coordinates remain citation authority through all phases. Long-redaction
compositions keep their separate byte-only admission. Compact tokens may change
native output; implementation and regression definitions establish no
model-quality, parity or measured-speed result.

`SourceTaskPlanner::plan_int8_question_hierarchy_with_control` explicitly admits
bounded quotation-selection levels when all collected QA evidence exceeds one
answer context. Every level pre-admits all its groups, retains compact witnesses
and reconstructs one grammar at a time. The hierarchy preserves original
occurrence sets, requires strict reduction in quote bytes, and retains every
native and lifted answer. Its private request/result commitments compose the
exact discovery owner. Use
`NlpEngine::synthesize_int8_document_answer_hierarchical` for one hosted
invocation with complete retained-workspace and output ownership. See
[question synthesis](candidate-question-synthesis.md) for the public contract.

## Exact completion association

`preflight_int8_map_with_control` returns an `Int8SourceMapPreflight` that binds
the exact source, task options, model identity, encoding policy, partition and
complete planning/work allowances. `PreparedInt8SourceMap::preflight_metadata`
returns the same contract after full planning. Sizing uses the actual shared
encoder without retaining native grammars, and its work remains charged.

`verify_completed_with_control` checks the full completed result against that
preflight, including native payloads and bookkeeping. Equal chunk geometry or
identical visible output cannot substitute a result from another source.
Exact summary and keyphrase reducers additionally bind their own aggregation,
verification and output limits. All three result types retain private request
and result commitments; these fields are omitted from JSON and diagnostics.
Final verification enforces the complete serialized byte ceiling and preserves
typed cancellation throughout the scan.

## Process-hosted entrypoint

`NlpEngine::map_int8_source` accepts an owned original `String`, the resident
`ResidentInt8`, pinned planner/vocabulary Arcs, a `hosted::SourceMapConfig`, and
one cancellation token. It performs partitioning and complete task planning
inside the existing process-owned blocking invocation, under the same deadline
and checkpoint budget as native inference and coordinate projection. It does
not call the single-request host once per chunk.

The host verifies the fixed task, strict backend, resident resource domain,
physical model identity, pinned planner assets, complete KV capacity and
explicit preparation/reduction headroom. Source `String` capacity is charged,
not only its visible byte length. Complete planning precedes native-engine
construction. Native KV, RoPE and scratch are reserved once; transient native
results, original-coordinate copies and retained reduction-frontier storage are
priced using the actual partition size. Retained prepared grammars/prompts and
metadata require the explicit preparation reservation; these modeled amounts
are not measured RSS or an allocator-level guarantee.

Every prepared identity is checked before the first forward. All preparation,
engine, admitted-identity and reduction scratch storage drains before physical
completion. The returned `HostedOutput<Int8SourceMapRun>` keeps the separate
output charge through the caller's serialization/delivery and offers no
unguarded ownership escape. Native/reduction failures use the typed
`HostedError::SourceMap`; default formatting does not expose source content.

## Evidence boundary

This is library implementation, with regression source for real pinned planning
and a private synthetic execution driver. It does not promote artifact
activation, production DSR acceptance, model-present parity, task-quality,
throughput, public neural CLI or durable-job claims. No Cargo, RCH, DSR or GitHub
Actions validation was run when this implementation was authored. Controller
validation remains pending under `WIRING.md`.
