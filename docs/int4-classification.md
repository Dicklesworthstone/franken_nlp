# Grouped-INT4 finite-label classification

The candidate INT4 path connects the existing pinned raw-text classification
compiler to grouped-INT4 execution, the charged `NlpEngine` host and the
`fnlp candidate source-int4 classify` command. It remains a non-certified
candidate, not a calibrated classifier or canonical FNLPQ format.

## Command-line execution

A classification request uses the same JSON contract as `fnlp candidate classify`:

```json
{
  "document": "Please refund the duplicate invoice.",
  "labels": [
    {"id": "Billing issue", "description": "Invoices and payment problems"},
    {"id": "Refund", "description": "Requests to return a payment"}
  ],
  "mode": "multi_label",
  "policy": {
    "minimum_candidate_weight_ppm": 600000,
    "minimum_margin_ppm": 200000
  }
}
```

`mode` defaults to `exclusive` and `policy` defaults to zero thresholds. Labels
and descriptions are data, not privileged instructions. Request JSON cannot
supply a model identity, budget, EOS, scoring mode, work receipt or sampling
policy. Duplicate keys, unknown fields, duplicate labels and malformed requests
are refused before model loading. Omitting the path or passing `-` reads stdin.

```sh
fnlp candidate source-int4 classify request.json \
  --source-dir /models/nanbeige-source \
  --source-manifest /models/pinned-source-manifest.json \
  --memory-mib 8192 --group-width 128 --context-tokens 2048

fnlp candidate source-int4 classify request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib 8192 --group-width 128
```

These use the existing source/snapshot loader and require `asupersync-runtime`.
The memory values are example ledger ceilings, not measured RSS requirements.
Source mode can add `--save-snapshot NEW_FILE`; snapshot mode requires an
independent content pin and cannot fall back to source conversion. Saving occurs
only after complete planning and all work-axis checks. Existing destinations and
failed partial files are preserved, and a successful save survives a later task
failure. The snapshot format and trust contract are unchanged.

Classification-specific flags are `--max-candidate-tokens` (default 16,
INCLUDING scored EOS), `--max-result-bytes` (default 1048576), and whole-request
ceilings `--max-forward-positions`, `--max-projected-logits`,
`--max-attention-pairs`, `--max-dot-products` and
`--max-multiply-accumulates`. The native group-dot ceiling counts each matrix's
actual groups, including tail groups; it is not an INT8 row count. All heads must
fit, not just the first head. Native context holds only the largest live head,
while compute allowances cover the complete bundle.

`--preparation-mib` defaults to 512 for this task. Its floor includes repeated
prompt/IR storage bounded by aggregate forward positions, not just input bytes.
Increasing that forward ceiling can require increasing preparation and process
memory. The host charges the complete allocated KV capacity and output, but no
sampler. Generation controls such as `--seed`, `--stop`, `--logprobs`,
`--max-new-tokens`, `--max-output-bytes` and `--prefill-rows` are rejected, even
when explicitly supplied with default values. Use the scored-task flags instead.

The command emits one completed source-candidate JSON envelope. Its `output`
is the complete `Int4ClassificationRun`; provenance retains the existing
`real-source-current-candidate-int4` scope and `non_authoritative` evidence
label. Model content and group width derive from verified resident weights.
No source paths, private prompt digests or arbitrary loader diagnostics are
published. No label result is published before every head and finalizer succeeds.

## Task and host API

`tasks::classify::grouped_int4::Int4ClassificationPlanner::for_model` builds a
pinned planner from an actual `Int4ModelIdentity`. Obtain it from a source-
converted or independently pinned snapshot-loaded `ResidentInt4`. The factory
owns the pinned tokenizer/control/template setup, not a second model loader.

```rust,ignore
use franken_nlp::tasks::classify::grouped_int4::Int4ClassificationPlanner;
use franken_nlp::native_engine::int4::PackedInt4Kernel;

// The existing host and resident model share one resource domain. Admit
// preparation storage before constructing the planner/request and retain it
// through serialization, write and flush of the guarded result.
let planner = Int4ClassificationPlanner::for_model(
    model.model_identity(), task_ceiling, classification_limits,
)?;
let prepared = planner.plan_with_control(&request, &mut preparation_control)?;
let output = host.execute_int4_classify(
    &model, prepared, native_limits, PackedInt4Kernel::scalar(), cancellation,
)?;
// Borrow output.result() and retain output through delivery.
```

Embedders managing their own execution context can use
`ClassificationPlanner::plan_int4_with_control`. The context must already match
an actual model and explicitly select INT4. The compiler never repairs a BF16
or INT8 identity. Execution rechecks source revision, candidate format, group
recipe, content fingerprint and packed-layout identity against resident weights.
The prepared type cannot be coerced to an eager/INT8 plan or deserialized from
caller-provided native scores.

## Classification and finite-language semantics

Exclusive classification scores the entire opaque-code language, restores
original UTF-8 label IDs and applies candidate-weight/margin abstention policy.
Multi-label classification runs independent yes/no heads: several labels may be
included, and positive weights are not normalized against other labels. Both
modes use full-vocabulary denominators at every scored prefix and append and
score exactly one EOS per candidate. These are complete sequence scores, not
sampled responses or selected-token logits mislabeled as full probabilities.
A failed head is not a successful abstention or partial label list. Weights and
thresholds remain explicitly **uncalibrated**; no accuracy claim is implied.

The exact-token `native_engine::strict_int4::scoring::Int4CandidatePlan` also
supports trie-conditional and three sequence-score modes. These are different
score spaces, not interchangeable optimizations. That leaf requires its host to
bind the prompt/language/model identity; raw-text classification supplies the
composition and intentionally uses only full-vocabulary scoring.

The shared `CandidateScorer` owns probability, EOS and traversal semantics.
Native execution prefills once, forwards every unique continuation prefix once,
and backtracks by rewinding all 44 completed KV slots. Rewinds invalidate
stale descendant hidden state; they never retain an unpriced hidden/KV table or
replenish compute allowances. Attention is priced at actual branch depths.
Group dots use `ceil(K / group_width)` per matrix and MACs use logical, unpadded
K. Sequential heads reuse one context but sum all computation before execution.
Counters describe algorithmic work, not measured instructions or speedups.

## Lifetime, cancellation and output

The existing resource host checks the resident-model domain and reserves KV,
workspace, allocator allowance and complete result storage before dispatch.
The model and native guards remain in the blocking closure through physical
completion. Cancellation does not pretend to preempt a native operation or
release memory while it runs. CLI input, loading, planning, persistence and
execution share one cooperative deadline; classification planning checkpoints
also observe that deadline. Loading, persistence and native execution have
separate explicit checkpoint allowances, never refreshed per head.

A successful output guard retains result storage through delivery. The CLI's
typed completion wrapper borrows that guarded result for serialization instead
of making an unguarded JSON copy. Preparation and result reservations remain
live through write and flush. Native accounting, semantic completeness,
output-size and final cancellation checks precede success. An oversized result
or failed output write is an error, not a partial classification success.

Embedders must separately admit tokenizer, input, planner and publication
storage. These modeled process-ledger obligations are not OS RSS limits. Each
completed native head clears its KV context; incomplete native work poisons the
session and the existing RAII cleanup clears the cache.

## Validation status

Regression sources cover synthetic shared-scorer differential replay, group-
aware schedules, candidate/EOS completeness, independent labels, profile and
identity separation, all compute axes, late cancellation, CLI source/snapshot
modes, wire refusal, unsupported controls and process-memory arithmetic.
Model-present tests require the pinned local source closure and controller
admission. Synthetic tests are not full-model inference evidence.

The CLI addition was source-reviewed with independent memory-arithmetic checks.
Rust compilation, Rust tests, full-model execution, DSR, numerical qualification,
task quality and performance measurements were **not run**. No snapshot trust
requirement, canonical artifact gate or release qualification is weakened.
