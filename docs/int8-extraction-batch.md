# Resident INT8 schema/source extraction

`batch::extract::quantized` connects raw `ExtractionBatchArgs` and document
text to the existing `Int8ExtractPlan` and actual `StrictInt8Engine` through
`batch::run_ndjson`. It does not introduce another grammar, decoder, sampler,
model loader or runtime. This is library integration, not public CLI/model
activation or a model-present fidelity/performance receipt.

## Planning and execution

Construct `Int8ExtractionBatchPlanner::pinned` with the pinned control registry,
EOS, explicit strict-quantized-v1 identity, host task ceiling, compiler/source
limits and optional default extraction arguments. The eager planner continues
to require BF16; neither public type exposes a conversion to the other. The
shared private compiler preserves the existing role scaffold, exact schema
bytes and separately encoded source. Defaults are bounded before per-item
cloning; per-document arguments replace them for that document only.

`PreparedInt8BatchExtraction` retains the exact source, TaskPlan and sealed
INT8 extraction plan. It exposes their borrowed views and complete planned
model work for admission. Schema input remains a JSON string, not a floating-
point serde Value: 38-digit constants and exact schema digests are retained.
Role/thinking-marker spellings in schema keys or source remain data tokens.
SourceMembership mode keeps its source-language and occurrence-evidence
contract; impossible required source fields and unsupported schemas refuse
instead of falling back to unconstrained or structural-only extraction.

Construct `NativeInt8ExtractionBatch::new` with that planner, a borrowed real
`StrictInt8Engine`, a reusable `ExtractionVocabulary`, the host's real admission
implementation and `Int8ExtractionBatchLimits`. Pass this processor to the
existing ordered NDJSON runner. The native model and vocabulary stay resident
across documents; each request cold-prefills its own prompt and drains all
logical KV before another request. This is sequential resident-engine corpus
execution, not layer-major parallel batching or prefix-cache reuse.

## Admission and complete work

`Int8ExtractionBatchAdmission::admit` receives the complete ExecutionIdentity,
all decoder/head work, attention pairs, grammar-mask reservation and checkpoint
limits, complete resident KV capacity and complete result-envelope byte cap.
It returns the identity actually admitted and a genuine owned resource guard.
There is no no-op/default admission implementation. The host separately owns
and admits resident weights, RoPE/activation/logit rails, vocabulary, private
source/grammar plans, allocator slack and transport-envelope staging.

The whole-run ledger reserves checked model and mask charges together before
admission. A rejection or early completion does not refund either allowance.
Numbered flush epochs reset neither. The outer runner additionally applies its
forward/logit, input/output, record and framing limits. INT8 head accounting is
selection-only: intermediate prompt positions still execute every native layer
but do not project the full language-model head. These are structural work
counts, not measured performance claims.

Execution verifies the admitted identity before forwarding, then the existing
INT8 task rechecks actual materialized model identity and vocabulary. Grammar
acceptance, exact JSON, source occurrences and the entire result envelope are
finalized inside the exclusively owned native session. Only the completed
`Int8ExtractRun` is handed to the batch writer, with its guard retained through
both write and flush. No partial structured result is published as successful.
Source membership is a byte-level guarantee, not semantic extraction accuracy.

## Failure and evidence boundaries

Invalid delivery coordinates, work overflow, identity substitution, poisoned
native state, nonempty KV after completion or an unwind permanently closes the
adapter. Cancellation retains its cause and cannot publish a success after
native completion. Normally recoverable errors become terminal when native
state is poisoned. Output failure is terminal to the NDJSON runner; no retries
or implicit engine resets occur. Deliberate host reconstruction is a separate
admission decision, not a hidden recovery path.

Regression tests exercise real pinned prompt/schema compilation and a private
synthetic driver for work, lifecycle and writer-failure injection. Public
construction accepts only the actual native engine, never that test driver.
The source-level checks retained during implementation preserve the shared
prompt compiler and existing eager execution body byte-for-byte. Rust
compilation/tests, controller DSR and model-present validation were not run;
no artifact, release, numerics or task-quality gate is promoted by this change.
