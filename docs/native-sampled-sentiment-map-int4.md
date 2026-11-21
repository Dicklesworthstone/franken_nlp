# Native INT4 sampled sentiment over complete documents

The complete-document sampled-sentiment task combines the independent seeded
score-and-explanation mode in [plan §7.5](../COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md)
with the lossless source partition in §7.10. Every source chunk receives the
complete requested axis-by-sample bundle. Results retain every chunk's scores,
bounded explanations and original byte and Unicode scalar spans, together
with explicitly defined document replicates.

The candidate command is `fnlp candidate source-int4 sentiment-sampled-map`.
This is an explicit grouped-INT4 candidate task. Its aggregate describes
separate chunk generations; it does not claim equivalence to one full-context
model call, calibrated confidence, human agreement or psychological
measurement. The existing one-context sampled task and distribution sentiment
keep their own versions and semantics.

## Request and invocation

Use the same strict JSON shape as one-context sampled sentiment:

```json
{
  "document": "The complete original document, including every section and its original whitespace...",
  "axes": ["valence", "arousal"]
}
```

Omitting `axes` requests valence, arousal, dominance and approach. Empty source,
empty or repeated axes, unknown fields, duplicate JSON keys and trailing JSON
objects refuse. This command consumes one complete document; it is not an
NDJSON population. The input cannot supply chunk boundaries, scores, prompts,
draw addresses or aggregation policy.

For an existing local candidate snapshot:

```sh
fnlp candidate source-int4 sentiment-sampled-map complete-request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib "$MEMORY_MIB" \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 --max-new-tokens 256 \
  --context-tokens 2048 --max-chunks 64 \
  --reuse-prefill
```

The model pin must come from an independently trusted conversion record.
Replace the snapshot and pin flags with `--source-dir /path/to/pinned-source`
and `--source-manifest /path/to/source-manifest.json` for local source
conversion. Both loading routes are supported. `--save-snapshot` is refused
before private IO on this complete sampled-map route. No weights are downloaded
and no model is activated by this command.

The explicit memory budget must cover the quoted complete peak. Declaring fewer
chunks or smaller retained/output limits reduces storage admission but also
reduces the accepted document envelope. Use the positional `-`, or omit it,
to read the complete JSON request from stdin. The seed must contain exactly 64
lowercase hexadecimal digits. Sampling temperature, top-k, top-p, per-sample
tokens and justification bounds follow the
[one-context sampled contract](native-sampled-sentiment-int4.md).

The command's map-specific limits are:

| Option | Default | Scope |
| --- | ---: | --- |
| `--max-input-bytes` | 1,048,576 | Complete JSON request, including source |
| `--max-chunks` | 64 | Complete nonoverlapping partition; range 1–256 |
| `--max-chunk-bytes` | Input byte limit | Additional source-byte bound per chunk; actual prompt must fit context |
| `--max-tokenizer-calls` | 8,192 | All chunk-count probes, including rejected candidates |
| `--max-result-bytes` | 1,048,576 | Complete sampled bundle per chunk |
| `--max-retained-result-bytes` | 67,108,864 | Sum of retained chunk envelopes |
| `--max-map-result-bytes` | 16,777,216 | Complete document payload, including all chunks and replicates |

Source size alone does not guarantee admission: actual partition, context,
generation, work and storage limits must all fit. The `--max-forward-positions`,
`--max-projected-logits`, `--max-attention-pairs`, `--max-dot-products`,
`--max-multiply-accumulates` and `--max-mask-node-visits` flags each apply once
to the entire map. The deadline also spans loading, all native work, reduction
and publication; `--max-checkpoints` bounds cooperative work in the owned
native invocation. There is no retry that renews a complete-document allowance.

### Compact input and physical execution

`--input-encoding safe-bpe-v1` enables compact input. Optional
`--max-tokenizer-input-bytes` and `--max-tokenizer-output-tokens` independently
bound cumulative work across probes and prepared chunks; each defaults to
1,048,576 and may be lower. The existing `--max-tokenizer-pair-checks` and
`--max-tokenizer-pair-bytes` also belong to this shared ledger. These flags
require compact mode. A byte scanned by several candidate probes is charged
several times, so the encoding-work ceiling is not a source-size promise.

`--compute-threads N` uses the coordinator plus persistent numerical workers,
with widths 1–64; width one is serial. Alternatively, `--prefill-rows N` uses
serial layer-major prompt blocks with rows 1–64. More than one compute thread
cannot combine with prefill rows. `--reuse-prefill` works with serial, team or
layer-major execution, but retains only the current chunk-and-axis prefix.
Worker and prefill scratch are charged before private input or model IO.
Physical widths do not change the sampling recipe or complete cold-work cap.
No measured speedup or executed schedule parity is claimed here.

`--max-selected-rows N` applies one fixed legal-row ceiling to every chunk,
axis and sample. It is also available on `sentiment-sampled-corpus` and
`sentiment-sampled-document-job start|resume`; retained jobs authenticate the
exact cap before replay. All legal rows, including accepting EOS, must fit
before top-k/top-p sampling, and an over-cap step refuses the complete document.
Original chunk coordinates, semantic draw addresses and full memory
reservations remain unchanged. See [selected output rows](native-sampled-sentiment-int4.md#selected-output-rows)
for the refusal contract, accounting, scheduling combinations and qualification
limits.

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

A map has its own address domain even if its partition contains only one
chunk. The same seed therefore does not promise the same samples as the
one-context command. Reproducible map replay requires the same source, model,
encoder, task/sampling recipe and declared map limits.

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

## Complete output and publication

The CLI response protocol is `fnlp-candidate-int4-sentiment-sampled-map-v1`.
The wrapper records candidate model provenance and includes the complete map
under `output`. Optional `snapshot`, `compute`, `prefill_schedule` and
`prefill_reuse` fields describe the selected physical route. Byte-default and
serial execution omit inapplicable fields.

`output.chunks` retains each chunk id, original `source_span`, source token
count and complete sampled `run`. `output.aggregate.dimensions` contains the
document replicates and descriptive moments. Source coverage, encoding work,
planned and actual model work, masks and generated tokens remain explicit.
The effective seed, document digest, prompt digest, private chunk addresses and
completion seals are not serialized. Generated explanations can still repeat
source content and should be treated as task output.

All reduction and verification must succeed before the bounded complete JSON
is staged. Deadline and cancellation are checked again during publication,
including after serialization and flush. Consumers require exit status zero
alongside the complete JSON. An output-device failure can leave written bytes;
it never turns a failed invocation into successful partial-document inference.

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
CLI regression sources cover strict parsing, unsupported-option refusal before
IO, source/snapshot admission, complete host/CLI quote equality, shared encoding
work, physical schedule ownership, typed failures, staging/flush cancellation,
and bounded optional response metadata.

These are authored test definitions and source review, not executed test or
native-model evidence. Compilation, controller production DSR validation,
pinned-model replay, task quality and measured memory/throughput remain
required. The repository's required named production feature graph and DSR
runner are unavailable in this session; no default-graph, RCH or Actions
substitute was run. The feature bead remains `in_progress` and no default
sentiment command or canonical model artifact is promoted.
