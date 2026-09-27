# Automatic native INT8 entity extraction and resolution

`corpus::entities_int8::prepare_int8_entities` accepts original `EntityDocument`
values (`id`, `text`), fixed NER options, a pinned source planner, a pinned
resolution planner, both model-bound identities and explicit finite limits.
No caller-supplied mentions, scores, provisional clusters or input identities
are accepted as document fields. It produces an owned, consumed executable
`PreparedInt8EntityCorpus`, not a serializable instruction to skip preflight.

This closes the native INT8 composition gap between source-constrained NER and
cross-document entity resolution. The existing BF16 entity pipeline and the
explicit-mention resolution API keep their independent numerical profiles.
No eager result is renamed to pass an INT8 gate.

## Complete execution

Preparation canonicalizes document order, validates IDs and aggregate source
bytes, checks both stage identities and preflights every exact NER request.
It retains original documents and compact private plan witnesses, not one
large source grammar/index per document. Execution rebuilds only one source
plan at a time and checks its entire identity, prompt length and five work
counters against the witness before calling the actual strict INT8 engine.

After each NER result, an independent occurrence scan checks every byte/scalar
span and the anchored/ambiguous status. Every occurrence becomes a contextual
mention. Duplicated proposals are verified before exact-span deduplication;
a missing repeated occurrence, absent source surface, unselected type, wrong
profile or malformed work receipt rejects the snapshot. Occurrence verification
and expanded-mention byte/count limits are shared across all documents.
Documents with no proposed entities remain in completed document receipts.

The complete candidate graph is built only after all NER passes succeed. The
existing pinned resolver scores both presentation orders and requires all
cross-cluster comparisons to be explicitly same before merging. Missing,
conflicting and uncertain comparisons remain barriers. The pipeline reuses
one resident model and one native engine; it introduces no loader, runtime,
retry callback, lexical identity shortcut or public fake model adapter.

## Work and ownership

`Int8EntityConfig::max_model_work` is the ceiling for BOTH stages together:
forward positions, projected logits, attention pairs, dot products and
multiply-accumulates. All conservative NER work is reserved before execution.
Pair scoring can use only the remaining allowance, additionally intersected
with the configured scoring-stage cap. Early EOS does not renew unused NER
reservations. Mask work is finite per document and for the entire snapshot.
There is one caller-supplied cancellation/checkpoint control across both stages.

Native execution consumes the prepared value. Failed operations publish no
completed corpus and cannot be retried through that value. Embeddings must
retain actual source, planner, native, graph and output reservations through
physical completion and delivery, and discard an engine after a failed run.
`retained_input_bytes` reports actual raw-input String/Vec capacities; planner,
configuration, graph and allocator costs need separate modeled reservations.
These are not measured or operating-system-enforced RSS limits.

Only an empty raw document list can use `finalize_without_model`. A nonempty
list of apparently empty/name-free documents still requires NER. After real NER,
a graph with no candidate pairs can use the existing genuine zero-pair finalizer
without pretending that pair scoring occurred.

## Output and evidence

A completed `Int8EntityRun` includes document receipts, per-stage and aggregate
work, mask reservations/charges, verification work and the full resolution
result. Original document/prompt hashes and NER token transcripts stay private.
Resolved mention surfaces remain sensitive output; this is not redaction or
anonymization. Source membership proves location, not recognition recall or
correct entity typing. Pair scores are uncalibrated and cluster IDs are
snapshot-local. Existing candidate numerical/quality gates remain unchanged.

Tests cover actual pinned preparation, private-witness checks, all work axes,
empty-snapshot finalization, source/profile corruption, Unicode repetitions,
aggregate verification/mention limits, cancellation and capacity accounting.
Private synthetic receipts are corruption fixtures only, not evidence of
neural success. Rust compilation/tests, full-model inference, recognition and
resolution quality, benchmarks and controller DSR were not executed here.
