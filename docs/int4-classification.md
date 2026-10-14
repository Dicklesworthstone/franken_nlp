# Grouped-INT4 finite-label classification

The candidate INT4 path now connects the existing pinned raw-text classification
compiler to real grouped-INT4 execution and the existing charged `NlpEngine`
host. This is a library/embedding surface, not a new CLI subcommand, release
qualification, calibrated classifier or canonical FNLPQ format.

## Task and host API

`tasks::classify::grouped_int4::Int4ClassificationPlanner::for_model` builds a
pinned planner from an actual `Int4ModelIdentity`. Obtain that identity from a
source-converted `ResidentInt4` or an independently digest-pinned snapshot-loaded
one. The factory owns the pinned tokenizer/control/template setup; callers do
not supply synthetic model, tokenizer or packing fingerprints.

Call `plan` or `plan_with_control` with an ordinary `ClassificationRequest`.
The result is a private `PreparedInt4Classification`, not an eager/INT8 plan.
Then call:

```rust,ignore
// The host and resident model already exist in the same resource domain.
// Hold an admitted preparation reservation before constructing the planner
// and request, and through serialization/write/flush of the guarded result.
use franken_nlp::tasks::classify::grouped_int4::Int4ClassificationPlanner;
use franken_nlp::native_engine::int4::PackedInt4Kernel;

let planner = Int4ClassificationPlanner::for_model(
    model.model_identity(), task_ceiling, classification_limits,
)?;
let prepared = planner.plan_with_control(&request, &mut preparation_control)?;
let output = host.execute_int4_classify(
    &model, prepared, native_limits, PackedInt4Kernel::scalar(), cancellation,
)?;
// Borrow output.result() and retain output until delivery has completed.
```

The host requires `asupersync-runtime`, just like the existing hosted INT4
text/chat surface. `NativeLimits.context_tokens` must fit the longest live
head plus its candidate continuation. The entire allocated KV capacity must
fit the request's `max_kv_bytes`. There is no sampler and no automatic ISA
selection. The existing checked kernel handle can be supplied explicitly.

Embedders already managing an execution context can instead use
`ClassificationPlanner::plan_int4_with_control`. It requires an actual model
identity and verifies that the supplied context already matches that model and
explicitly selects INT4. It refuses an INT8/BF16 identity rather than repairing
one. Execution independently checks the prepared identity against the loaded
model again, including source revision, candidate format, group recipe, content
fingerprint and packed-layout identity.

## Classification semantics

Exclusive classification scores the complete opaque-code language, restores
original UTF-8 label IDs and applies the existing candidate-weight/margin
abstention policy. Multi-label classification runs independent yes/no heads;
several labels may be included, and their positive weights are not normalized
against one another. Label IDs, descriptions and documents still use the shared
untrusted-data compiler, not privileged template/control tokenization.

Both modes use full-vocabulary denominators at every scored prefix and append
and score exactly one EOS per candidate. These are complete sequence scores,
not sampled responses or selected-token logits mislabeled as full probabilities.
All heads and the existing semantic finalizer must succeed before a result is
returned. A failed head is not a successful abstention or a partial label list.
Weights and thresholds remain explicitly **uncalibrated**; this addition makes
no task-quality or correctness-confidence claim.

`Int4ClassificationRun` contains the complete semantic result, head count, group
width, `strict-quantized-v2` profile, the distinct INT4 classification execution
version, exact model-work counters and rewound positions. Existing BF16/INT8
classification entry points remain separate, and INT4 policy identity binds the
new scoring execution version instead of reusing the INT8 scoring identity.

## Native finite-language engine

`native_engine::strict_int4::scoring::Int4CandidatePlan` is also available as an
exact-token leaf. It supports the existing full-vocabulary, trie-conditional and
three sequence-score modes; these score spaces are distinct, not interchangeable
optimizations. This leaf requires the embedding task to bind prompt, language
and model identity and to admit preparation storage. Raw-text classification
supplies that composition and intentionally uses only full-vocabulary scoring.

The shared `CandidateScorer` remains the probability, EOS and traversal
implementation. The native adapter prefills once, forwards every unique
nonempty continuation prefix once, and backtracks by rewinding all 44 completed
KV slots. Rewinds invalidate descendant hidden state; they never retain an
unpriced hidden-state table, fork KV pages or replenish work allowances.

The whole finite language is preflighted before its first forward. Work counts
include actual branch-depth attention, every projection's group-dot count and
logical unpadded MACs. Group tails use `ceil(K / group_width)` for each individual
matrix. Serial classification heads reuse one allocation, but their compute
costs sum: each head receives an exact slice, not another copy of the bundle's
full allowance. Counters describe algorithmic work, not measured instructions,
traffic, runtime or a demonstrated speedup.

## Lifetime, cancellation and output

The existing resource host verifies the resident-model domain and reserves KV,
workspace, allocator allowance and complete result storage before dispatch.
The resident model and all native guards remain owned by the blocking closure
until physical completion. Cancellation is cooperative; it does not pretend to
preempt a native operation or release storage while that operation still runs.
Typed cancellation causes survive native and semantic boundaries.

A successful output guard retains result memory for the caller's delivery.
The host's allocation helper drops a result before releasing its reservation
when commitment fails. Each completed head leaves all native KV slots empty;
partial native execution poisons the session, and its existing RAII cleanup
clears the cache. Native accounting, semantic completeness, output-size and
final cancellation checks all precede success.

The host does not reserve the caller's input/tokenizer/planner memory on its
behalf. Admit and retain that preparation storage separately, including any
publication buffers. These are modeled process-ledger obligations, not OS RSS
limits or evidence that a particular machine can execute the complete model.

## Implementation and validation status

Added regression sources exercise shared-scorer differential replay with
synthetic logits, group-aware schedules, candidate/EOS completeness, label-ID
restoration, independent binary decisions, profile separation, aggregate
budgets, late cancellation and guarded-host admission arithmetic. An explicitly
ignored source-model test requires controller admission and the pinned local
source closure. Synthetic tests are not full-model inference evidence.

Rust compilation, tests, full-model execution, DSR, numerical qualification,
quality evaluation and performance measurements were **not run** during this
implementation. The new API is code-first candidate functionality. It neither
changes snapshot trust requirements nor activates or certifies an artifact.
