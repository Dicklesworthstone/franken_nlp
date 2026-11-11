# Native INT4 classification over complete documents

`fnlp candidate source-int4 classify-map` classifies every chunk of one complete
document and returns the original chunk results together with a named
document-level aggregate. It supports exclusive classification and independent
multi-label decisions through the existing native classifier.

For a retained population with authenticated resume, use
[`classify-document-job start|resume`](native-classification-document-jobs-int4.md).
That route keeps each complete original document as one durable item and
preserves these same chunk results and aggregate semantics.

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
data under either explicitly selected byte-preserving encoder.

An explicit `--save-snapshot NEW_FILE` requires complete document preflight
before creating the requested file. The existing create-new policy applies;
this does not activate a model or authenticate a publisher. A saved snapshot
can remain if later inference or publication fails. This option currently
requires the default byte-fallback encoding; compact maps refuse it before
input or model IO.

## Explicit compact source encoding

The default remains one byte per source token. Add
`--input-encoding safe-bpe-v1` to use bounded compact encoding for both the
original document and the untrusted label records. The planner counts actual
compact tokens when selecting each chunk and reserves the exact metadata and
candidate-scoring context for every head. Ordinary prose can therefore fit
more source bytes in each context and need fewer native chunk evaluations.
The resulting chunk boundaries and model scores may change; this is an
explicit candidate tokenization policy with its own execution identity.

For example, add these options to the pinned snapshot invocation above:

```sh
  --input-encoding safe-bpe-v1 \
  --max-tokenizer-input-bytes 1048576 \
  --max-tokenizer-output-tokens 1048576 \
  --max-tokenizer-pair-checks 33554432 \
  --max-tokenizer-pair-bytes 268435456
```

Both pinned source inference and independently pinned snapshot reload support
compact maps, with serial, `--cohort-chunks` or `--compute-threads` execution.
Chunk cohorts and CPU teams are separate scheduling selections. The compact policy
does not normalize, truncate or rewrite source text. Token decoding must
reconstruct the original bytes, and source/label tokens cannot be template
control IDs. Literal marker text remains data. These guarantees concern the
token boundary; they do not establish resistance to instruction-shaped text
or equivalence to an ordinary tokenizer or another model's scores.

One tokenizer allowance covers the complete operation: encoding metadata to
find available source capacity, every attempted chunk count (including
oversized candidates that must shrink), and every final chunk plan and label
head. No probe, chunk, label or cohort receives a fresh allowance.

| Compact work option | Default | Maximum | Counts across the complete map |
|---|---:|---:|---|
| `--max-tokenizer-input-bytes` | 1,048,576 | 1,048,576 | All bytes submitted to compact encoding, including repeated probes and metadata. |
| `--max-tokenizer-output-tokens` | 1,048,576 | 1,048,576 | All tokens produced by those encodings. |
| `--max-tokenizer-pair-checks` | 4,194,304 | 67,108,864 | Pair comparisons during merge searches. |
| `--max-tokenizer-pair-bytes` | 67,108,864 | 536,870,912 | Pair bytes examined during those comparisons. |

These limits are separate from the original request's `--max-input-bytes`,
each context's token ceiling, `--max-tokenizer-calls` for chunk probes and the
five native work ceilings. A document fitting the request-byte ceiling can
still exhaust compact work, because planning may encode its bytes more than
once. `--max-chunk-bytes` can limit the size of attempted probes. Exhaustion
refuses the complete document before the first native forward; it never
silently switches encoding or publishes only the chunks planned so far.

Compact semantic output adds `input_encoding` with the exact `policy` and
consumed `work` counters. Each chunk's `source_tokens` is its actual token
count, while byte/scalar coordinates and aggregation weights still refer to
the original source. The complete result verifier binds those counts and
coordinates, the policy and its work receipt to the prepared map. The default
byte output omits this optional field and retains its original identities.

Completion verification also binds the exact original source, label taxonomy,
task and planning limits, encoding policy and ordered compiled model
identities. Equal source lengths, chunk geometry and native work cannot make
a result from another admitted request interchangeable. Returned typed
results carry private request and completed-payload seals; changing scores,
aggregate values or cohort counters invalidates the corresponding seal.
The cohort seal also binds its nested result's private authority, so replacing
the nested result cannot transplant another request's execution counters.

These checks apply to byte and compact maps. Hashing streams through the
existing finite input and result bounds without allocating a second
source-sized buffer. Request hashing checks cancellation at bounded byte
intervals. The seals are omitted from JSON and debug output, preserving the
serialized result contract without publishing source-derived fingerprints.

The memory quote adds one bounded tokenizer workspace beside all retained
plans. Hosted callers can use `requirements_with_encoding` or
`cohort_requirements_with_encoding` on `Int4ClassificationMapHostConfig`, then
`NlpEngine::map_int4_classify_with_encoding` or
`NlpEngine::map_int4_classify_cohorts_with_encoding`. The existing methods
continue to select byte fallback. Direct planner callers select
`Int4ClassificationPlanner::with_document_encoding` before
`plan_document_with_control` and add the encoder with the pure quote's
`with_input_encoding` method.

Compact `--save-snapshot` is refused because the current pre-save flow would
plan again during hosted inference and renew tokenizer work. Use an already
pinned snapshot or pinned source inference without saving on this route.
`classify-corpus` and `classify-document-job` retain their byte-only contracts;
compact encoding and its extra work flags are not accepted there.

## Persistent CPU teams

Add `--compute-threads 4` to `classify-map` to keep one coordinator and three
row-projection workers across the complete document, including every label
head and chunk. The runtime thread is additional. Worker scratch is admitted
once before input or model IO, and the fixed team joins before the complete
guarded result is returned. The equivalent library entry point is
`NlpEngine::map_int4_classify_with_threads`.

The team also accepts `--input-encoding safe-bpe-v1`. One compact-tokenizer
workspace and one worker workspace are admitted together; source and label
encoding still spend the complete operation's original tokenizer allowance.
Hosted callers select that combination with
`NlpEngine::map_int4_classify_with_encoding_and_threads`. The compact
snapshot-save restriction continues to apply.

The semantic `output`, request identity, chunk evidence, probability spaces,
aggregate and whole-document native work limits use the ordinary serial path.
A separate top-level `compute` receipt records the selected width and actual
completed projection/row counters. The default width of one retains the
existing output envelope. This scheduling option does not establish a measured
speedup or model-parity result; its regression tests await the required DSR
checkpoint and model-present comparison.

The width must be between 1 and 64. More than one compute thread cannot combine
with `--cohort-chunks`, `--cohort-heads` or `--prefill-rows`. The corpus and durable
job commands keep their existing serial/cohort schedules and do not expose this
option. Each independent head still has its own causal context and full
vocabulary denominator; worker count never multiplies its logical work budget.

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
cover all windows. Optional byte-mode snapshot saving first invokes the same exact
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
aggregation of nonuniform chunk scores. Compact regression source covers
lossless Unicode/control text, exact cumulative limits on all four tokenizer
axes, cancellation causes, actual source-token geometry, encoding/work receipt
substitution, byte-default compatibility, workspace admission and the
pre-IO snapshot-save refusal. The explicit model-present fixture includes
both encodings and both decision modes for serial/cohort comparisons.
Completion-binding regressions cover same-geometry source, model, taxonomy
and authority substitutions, public score/aggregate/counter changes, nested
result replacement, private-fingerprint omission and hashing cancellation.
Compilation and Rust tests remain
pending the repository-required controller-selected DSR checkpoint. Model
quality and performance require their separate retained evidence.
