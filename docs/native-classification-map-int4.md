# Native INT4 classification over complete documents

`fnlp candidate source-int4 classify-map` classifies every chunk of one complete
document and returns the original chunk results together with a named
document-level aggregate. It supports exclusive classification and independent
multi-label decisions through the existing native classifier.

This is the complete-document classification route from
[plan §§7.4 and 7.10](../COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md). The aggregate
is a source-scalar-weighted mixture of chunk candidate weights. It is distinct
from the full-vocabulary sequence scores retained for each chunk and does not
claim equivalence to one model pass over the complete document.

## Input and invocation

The input uses the same strict JSON object as native `classify`:

```json
{
  "document": "The complete original document, including every section...",
  "labels": [
    {"id": "support", "description": "A request for product help"},
    {"id": "billing", "description": "An invoice, payment or subscription question"}
  ],
  "mode": "exclusive",
  "policy": {
    "minimum_candidate_weight_ppm": 0,
    "minimum_margin_ppm": 0
  }
}
```

Use `"multi_label"` to score each label independently with its own yes/no head.
Exclusive mode requires at least two labels. Multi-label mode accepts one or
more; its labels never compete in a shared softmax. Label identifiers are
unique and exact. Descriptions are optional. Omitted mode selects exclusive
classification, and omitted policy uses the zero thresholds shown above.

For an independently pinned local candidate snapshot:

```sh
fnlp candidate source-int4 classify-map request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib "$MEMORY_MIB" \
  --group-width 128 --context-tokens 2048 \
  --max-chunks 64 \
  --minimum-classified-coverage-ppm 1000000
```

Use `-` or omit the input path to read the JSON object from stdin. The input is
one document, not NDJSON. The complete request, including JSON framing and
labels, must fit `--max-input-bytes`. The usual explicit `--source-dir` and
`--source-manifest` pair can replace the snapshot options. Both acquisition
routes use the same planner and result contract.

Duplicate keys, unknown fields, invalid policies, duplicate labels and empty
source refuse before model loading. Request JSON cannot supply task budgets,
chunks, scores, prompts, model identity, EOS, score-space selection or
aggregation authority. Literal control-marker text stays untrusted source
data under the existing byte-preserving encoder.

An explicit `--save-snapshot NEW_FILE` requires complete document preflight
before creating the requested file. The existing create-new policy applies;
this does not activate a model or authenticate a publisher. A saved snapshot
can remain if later inference or publication fails.

## Serial and native cohort execution

Serial execution remains the default and reuses one native cache. Add
`--cohort-chunks N` to score bounded windows of original chunks through the
native classification cohort engine. For example, the snapshot invocation
above can include `--cohort-chunks 4` to admit windows of up to four chunks.
This is an explicit execution choice with its own simultaneous memory quote.

The requested width must be between 1 and 64, must not exceed `--max-chunks`,
and must satisfy `N × heads_per_chunk <= 64`. Exclusive classification has
one head per chunk, irrespective of the number of candidate labels.
Multi-label classification has one independent yes/no head for each label.
Eight labels therefore admit at most eight chunks per window; nine labels
admit at most seven. The command checks the actual request's label mode and
head count before model loading. It refuses an invalid declared shape even
if the document would happen to produce a shorter window. It never silently
shrinks the requested width or switches execution modes.

Even `--cohort-chunks 1` uses simultaneous heads within each multi-label
chunk, so that schedule permits at most 64 labels. Omit the option to retain
the serial route's support for 128 sequential independent label heads.

Every chunk still has its own original source span, complete label codebook,
prompt context and exact native work. All windows and identities are checked
before the first native forward, including the final short window. The same
whole-document work ceilings, retained-result limit and complete-result limit
cover all windows. Optional snapshot saving first invokes the same exact
all-window admission checks used by execution. Each chunk's task budget must
cover its simultaneous heads; the whole window separately admits its combined native memory and
scoring workspace. The CLI keeps logical task authority fixed across serial
and cohort schedules while pricing their different physical allocations.

The candidate JSON keeps the semantic map at `output` for either schedule.
Cohort mode adds a separate top-level `cohort` object containing its schema,
execution profile, requested `max_chunks`, completed `chunk_cohorts`,
`group_steps` and `projection_groups`. These describe physical scheduling.
They do not change the aggregate's meaning or establish a throughput result.
The implementation uses the existing native chunk scorer and the same serial
collector and aggregation code. A dedicated native fixture compares both
modes' semantic maps; that fixture remains pending the required execution
checkpoint.

Each native window has an additional complete-output ceiling of 64 MiB,
besides the per-chunk, retained and final document limits. The declaration
`N * --max-result-bytes + 4096` must fit that ceiling; an oversized declared
window refuses before input IO. At a width of 64, reduce the default 1 MiB
per-chunk cap explicitly, for example with `--max-result-bytes 524288`.
The CLI memory quote reserves the largest valid unknown chunk/head geometry
before input IO, so cohort mode can require substantially more memory. Hosted
callers can quote their declared head count directly. No window result is
published independently: a later failure or cancellation yields no completed
document result. See the [native chunk-cohort controls](int4-classification.md#native-chunk-cohorts)
for exact limits and the embedding API.

## Complete source and scoring semantics

The planner creates a disjoint, lossless partition of the original UTF-8
document. Each chunk retains its identifier, original byte range, original
Unicode scalar range and source-token count. There is no normalization,
overlap, truncation or sentence-boundary guarantee. The pinned source encoder,
exact trusted scaffold, label codebook and every required classification head
determine the available chunk capacity. `--max-chunk-bytes` can impose a
smaller bound.

Every chunk and head is prepared before native execution begins. Complete
document work is the checked sum of five native axes: forward positions,
projected logits, attention pairs, dot products and multiply-accumulates.
The ceilings do not renew for each chunk or each label. A later invalid chunk
or exceeded whole-document allowance refuses before the first model forward.

Each successful chunk preserves its exact completed candidate sequence
scores, including its scored terminal EOS and full-vocabulary denominator at
each scored position. Its candidate weights, ranking, original policy,
classified/abstained decision and native work remain visible. A native failure
is not converted into an abstention or omitted from the result.

## Document-level aggregation

The aggregation profile is `classification-source-scalar-mixture-v1`. A
chunk contributes in proportion to its number of original Unicode scalars,
including whitespace. Multibyte UTF-8 characters therefore do not receive
extra weight because their encodings are longer. All chunks contribute to
the mixture, including chunks whose local decision abstained.

For exclusive classification, matching label weights are combined, then
ranked by aggregate weight with exact identifier order breaking ties. The
request's minimum candidate weight and margin policy is applied to this
separate mixture. For multi-label classification, the same operation applies
to each label's independent yes/no head. One label's classified coverage does
not authorize another label's decision.

| Output | Meaning |
|---|---|
| `chunks` | Every complete native chunk result and original source coordinates. |
| `aggregate.result` | Exclusive mixture or independent per-label mixtures. |
| `candidate_weights` | Scalar-weighted mixture over the complete candidate set. |
| `classified_scalars`, `classified_coverage` | Original source coverage from chunks whose relevant head classified successfully. |
| `selected_id` / `selected_ids` | Decisions satisfying both the mixture policy and coverage requirement. |
| `abstained_ids` | Multi-label heads whose aggregate cannot authorize a decision under those policies. |
| `single_context_equivalence` | Always `false`; the estimator uses independent chunk context. |

`--minimum-classified-coverage-ppm` defaults to `1000000`, requiring all
original scalars to belong to locally classified chunks for the relevant
head. Its range is 0 through 1000000. Lowering it relaxes the decision's
coverage requirement without dropping abstained chunks or changing their
mixture weights. Zero removes that coverage requirement; the candidate-weight
and margin policy still applies.

All scores remain explicitly uncalibrated. Mixture weights are not a
probability that the classification is correct. Changing chunk geometry can
change decisions because the model sees different local context.

## Budgets and publication

The command admits its declared memory requirements before input or model IO:
input parsing, retained chunk plans, label heads, native state, complete chunk
results, aggregation, final JSON, model loading/saving and runtime cleanup.
The same pure quote governs hosted execution. The resource ledger models
allocation ownership; it is not an operating-system RSS cap.

The main controls are `--max-input-bytes`, `--context-tokens`, `--max-chunks`,
`--max-chunk-bytes`, `--max-tokenizer-calls`, the five `--max-*` native work
ceilings, `--max-result-bytes` for each complete chunk,
`--max-retained-result-bytes` for retained chunk records, and
`--max-map-result-bytes` for the complete semantic document result. Final
candidate framing and optional cohort metadata have separate fixed
allowances. Tighten these bounds to the intended workload when choosing
`--memory-mib`.

`--timeout-seconds` covers reading, model acquisition, optional snapshot
preflight, execution and publication. `--max-checkpoints` bounds cooperative
native planning/execution checkpoints. Blocking IO observes the deadline
between operations. Publication retains the result's resource guards through
write and flush, and checks the deadline after a successful flush.

Planning, execution or aggregation failure publishes no partial inference
result. A writer or deadline failure during publication can leave complete
or truncated bytes and still returns failure. A complete operation therefore
requires exit 0 as well as a complete JSON object.

## Validation scope

Execution requires `asupersync-runtime`; help and argument validation remain
available without it. The route preserves the explicit candidate model
provenance and does not promote build, model-parity, task-quality or release
qualification.

Regression source covers strict dispatch and requests, rejection before IO,
whole-document budget and memory admission, source/snapshot symmetry, typed
deadline handling and complete publication. Cohort regression source also
covers ragged windows, per-chunk summed KV authority, every native work and
memory axis, late identity failures, malformed results and unchanged
aggregation of nonuniform chunk scores. Compilation and Rust tests remain
pending the repository-required controller-selected DSR checkpoint. Model
quality and performance require their separate retained evidence.
