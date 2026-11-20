# Native INT4 sampled sentiment over complete documents

The complete-document sampled-sentiment task combines the independent seeded
score-and-explanation mode in [plan §7.5](../COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)
with the lossless source partition in §7.10. Every source chunk receives the
complete requested axis-by-sample bundle. Results retain every chunk's scores,
bounded explanations and original byte and Unicode scalar spans, together
with explicitly defined document replicates.

This is an explicit grouped-INT4 candidate task. Its aggregate describes
separate chunk generations; it does not claim equivalence to one full-context
model call, calibrated confidence, human agreement or psychological
measurement. The existing one-context sampled task and distribution sentiment
keep their own versions and semantics.

## Complete source and independent draws

The planner accepts one `SentimentRequest` with its complete original document,
a nonempty unique subset of the four sentiment axes, and a per-chunk task
budget. Omitted axes at the CLI select all four. There is no normalization,
truncation, overlap or caller-supplied chunk score. Whitespace, CRLF and Unicode
source scalars all remain in the partition exactly once.

Every chunk must fit the longest actual trusted prompt scaffold among the
selected axes, its complete untrusted input and the requested generation
allowance. `ChunkLimits` independently caps source bytes, bytes and tokens per
chunk, context, chunk count and tokenizer probes. The maximum supported chunk
count is 256. An over-limit partition refuses the document.

Each chunk uses the existing independent sampled task: each requested sample
must produce a grammar-constrained integer score in `[-100,100]`, a bounded
nonempty justification and a valid native completion. All samples, including
EOS, must finish; there is no partial bundle to reduce. The complete document
fails if any chunk fails.

The private draw address binds the complete source, canonical axis order,
model and task recipe, declared limits, sampling options, and the chunk's
ordinal, original span and source-token count. Repeated equal-text chunks have
different addresses. Changing the surrounding document can change an otherwise
identical chunk's draws. Axis argument order is canonicalized. Worker count,
physical prefill width and same-axis prompt reuse do not alter this semantic
address. These private bindings and the effective seed are omitted from the
result schema.

## Versioned document estimator

Let chunk `c` contain `w[c]` original Unicode scalars and let
`s[c,a,j]` be its integer score for axis `a` and sample ordinal `j`. For each
axis and ordinal, the document replicate is:

```text
weighted_score_sum[a,j] = sum_c(w[c] * s[c,a,j])
score[a,j] = weighted_score_sum[a,j] / total_source_scalars
```

All weighted products and sums use checked exact integers. Conversion to
floating point occurs at the final division. Each replicate retains both its
`sample_index` and exact `weighted_score_sum`; the aggregate supplies the
common `source_scalars` denominator. Chunk weights count source scalars, not
UTF-8 bytes, encoded tokens or explanation lengths. For example, scores `100`
and `-100` on chunks containing one and three source scalars produce a document
replicate of `-50`.

An invocation with `N` samples per axis produces `N` document replicates per
axis, pairing the same sample ordinal across all chunks. It reports their
mean, median, minimum, maximum and population standard deviation. The standard
deviation is descriptive spread, not a standard error or confidence interval.
No extra quantiles, interval or generated document-level explanation are
invented. Every original chunk explanation remains in its complete chunk
bundle and is explicitly unverified model prose, not source evidence.

The reduction profile is
`source-scalar-ordinal-paired-sampled-replicates-v1`. The document execution
profile is `grouped-int4-complete-sampled-sentiment-map-v1`, or
`grouped-int4-complete-sampled-sentiment-map-reused-prefill-v1` with explicit
prompt reuse. `single_context_equivalence` is always `false`.

## Whole-document admission

`SampledSentimentMapLimits` declares the complete cold model-work ceiling,
cumulative retained chunk-result bytes and final document-result bytes, beside
the source partition limits. Every chunk plan, model binding and conservative
mask allowance is checked before the first native forward. Forward positions,
projected logits, attention pairs, projection dot products and multiply
accumulates are summed across the complete chunk-by-axis-by-sample schedule.
Neither early EOS nor saved prompt work authorizes extra chunks or samples.

Execution slices exact child native and mask allowances from that complete
admission. Results report planned cold model work and masks separately from
actual model work and mask charge. Each complete chunk retains its own work
and generated-token accounting. A document result must reconcile all sums,
the exact source partition and the deterministic reduction. Private request,
geometry and completion seals refuse altered or unrelated public payloads.

Byte fallback is the default control-excluding input encoder. Explicit
`safe-bpe-v1` uses actual compact counts for chunk boundaries and can change
prompt tokens and generated results. One caller-owned `SafeBpeWork` ledger
covers successful and rejected chunk probes and all actual chunk preparation.
It is never renewed at a chunk or axis boundary. Exhaustion refuses without
an implicit fallback. The optional `input_encoding` result records the compact
policy and this invocation's encoding work.

`SampledSentimentMapMemoryLimits::requirements` supplies a pure quote before
source or model IO. It reserves one shared compiler/vocabulary and encoder
workspace, all retained chunk prompts and grammar programs, source capacity,
retained explanations and result storage, one KV/native allocation and one
sampler workspace. It also reserves complete output tokens and bytes. Physical
prefill workspace and numerical worker storage are additional owned host
charges. These are conservative admission bounds, not measured process RSS.

## Task API

The task surface lives in `tasks::sentiment::sampled::long`:

- `SampledSentimentPlanner::plan_document_with_control` prepares every chunk
  under one fresh encoding ledger.
- `plan_document_with_encoding_work` shares a caller-owned ledger across
  preparation attempts; failed work retains its debit.
- `PreparedSampledSentimentMap::complete_mask_charge` validates the entire
  native and mask allowance before execution.
- `execute_scheduled_with_control` executes serial, layer-major prefill or
  same-axis prompt reuse on the supplied exclusive native engine. A host may
  attach an existing persistent projection team to that engine.
- `preflight_metadata().verify_completed_with_control` verifies the sealed
  complete result and its reduction before publication.

The native engine must enter with all 44 KV slots empty and unpoisoned. Every
child completes and cleans its logical state before the next chunk. Prefix
reuse remains inside the current chunk and axis; no KV crosses a chunk
boundary. Original typed cancellation causes propagate through partitioning,
hashing, generation, reduction and result verification.

## Owned runtime API

With `asupersync-runtime`, `NlpEngine` exposes:

- `map_int4_sampled_sentiment`
- `map_int4_sampled_sentiment_with_encoding`
- `map_int4_sampled_sentiment_with_threads`
- `map_int4_sampled_sentiment_with_encoding_and_threads`

Each method takes one owned request and `SampledSentimentMapHostConfig` beside
the resident candidate, checked integer kernel and cancellation token. The
config contains the native context/deadline, per-chunk planning and sampling
settings, complete map limits, preparation and sampler reserves, complete
mask ceiling, optional `Int4PrefillLimits`, and explicit `reuse_prefill`.
`requirements` and `requirements_with_encoding` quote the complete host
envelope without opening a model or source.

The host owns the complete request before constructing the pinned planner and
vocabulary. It prepares and admits every chunk, then allocates one native
engine for the whole document. Serial execution and layer-major prompts use
that same engine. A persistent numerical team instead remains attached across
every chunk, axis and sample. Team execution and layer-major prompt rows are
separate schedules and cannot be combined. Same-axis prompt reuse is supported
on either schedule, always isolated to the current chunk and axis.

Input, KV and workspace charges remain alive through physical dispatcher
completion and worker join. Only then does the host release those owners and
return a guarded complete output; its output reservation remains live with
the result. Every successful return verifies the complete private result seal
and all 44 empty KV slots. A cancellation or failure returns no partial
document, and the original cancellation cause survives the host boundary.

## Validation status

Regression definitions cover complete Unicode/CRLF source coverage, canonical
axes, repeated-text address separation, each complete native-work ceiling,
all-chunk mask admission, cumulative compact-tokenizer work, exact scalar
weighting, incomplete and inconsistent chunk bundles, all eleven cancellation
causes, memory bounds and private completion seals. Hosted regression sources
also cover excess retained request capacity, shared sampler/encoder/prefill
storage, exact complete output reservations, public API ownership, and an
ignored fresh-runtime fixture for cancellation and full physical drain under
six serial/team/prefill/reuse combinations.

These are authored test definitions and source review, not executed test or
native-model evidence. Compilation, controller production DSR validation,
pinned-model replay, task quality and measured memory/throughput remain
required. The repository's required named production feature graph and DSR
runner are unavailable in this session; no default-graph, RCH or Actions
substitute was run. The feature bead remains `in_progress` and no default
sentiment command or canonical model artifact is promoted.
