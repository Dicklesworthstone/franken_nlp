# Native INT4 sentiment over complete documents

`fnlp candidate source-int4 sentiment-map` scores every chunk of one complete
document and combines the independent dimension results under an explicit,
versioned rule. It uses the existing native sentiment scorer, deterministic
source partitioning and bounded resource host. It preserves each chunk's
complete result and original byte and Unicode scalar ranges.

This implements the score-over-long-documents route described in
[plan §7.10](../COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md). The aggregate is a
descriptive mixture of chunk scores. It does not establish equivalence to a
single model pass over the whole document, calibrated correctness confidence,
psychological measurement or task quality.

## Input and invocation

The input is the same strict JSON request used by native `sentiment`:

```json
{
  "document": "The first section welcomes the proposal. Later sections describe serious concerns...",
  "axes": ["valence", "arousal", "dominance", "approach"],
  "policy": {
    "minimum_peak_weight_ppm": 0,
    "maximum_normalized_entropy_ppm": 1000000
  }
}
```

The document string contains the complete source. JSON escaping is decoded
once; UTF-8 spelling, whitespace and line endings remain unchanged. Omitted
axes select all four. A nonempty unique subset is allowed. Omitted policy uses
the two values shown. Duplicate keys, unknown fields, repeated axes and invalid
policy integers refuse before model loading. The request cannot supply chunks,
scores, work receipts, prompts, model identity, EOS or aggregation policy.

For a trusted, independently pinned local candidate snapshot:

```sh
fnlp candidate source-int4 sentiment-map request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib "$MEMORY_MIB" \
  --group-width 128 --context-tokens 2048 \
  --max-chunks 64 \
  --minimum-estimated-coverage-ppm 1000000
```

Use `-` or omit the input path to read the JSON request from stdin. The default
request limit is 1 MiB, including the outer JSON. This is one document, not an
NDJSON stream. Existing `score-batch` remains the separate corpus surface.

The usual explicit `--source-dir` and `--source-manifest` pair is supported
instead of a snapshot. Optional `--save-snapshot NEW_FILE` preserves the
existing create-new policy. Complete document planning must succeed before
that file is created; the saved file remains if subsequent inference or output
fails. Snapshot creation does not activate a model or authenticate a publisher.

`--kernel` selects an existing hardware-checked native integer kernel. Scalar
is the default. Unsupported selections refuse without a fallback.

## Compact input across complete-document routes

`sentiment-map`, `sentiment-corpus` and `sentiment-document-job start|resume`
accept `--input-encoding safe-bpe-v1`. The encoder preserves source bytes,
excludes template-control tokens and chooses chunk boundaries using actual
token counts. The explicit input-byte, output-token, pair-check and pair-byte
ceilings are tokenizer work limits; they are separate from source size,
native context, model work and result size. Byte fallback remains the default.

One-shot maps share their allowance across every probe and final chunk plan.
Corpus and durable document attempts additionally carry that same ledger from
initial preparation into reconstruction for execution. Reconstruction and
axis/cohort boundaries cannot renew work. Exhaustion refuses the document
before native execution. The existing completion seal retains per-pass work;
the private attempt ledger charges all preparation. One encoder workspace
is admitted beside the retained chunk plans, independent of axis/cohort width.

Durable recipes authenticate the policy and all four ceilings, which must be
repeated unchanged on resume, including retained-input replay. Byte-mode recipe
bytes remain unchanged. Compact mode supports pinned source input and pinned
snapshot reload; `--save-snapshot` is refused before private IO. The complete
native work ledger retains its existing invocation or durable lifetime scope.

## Partition and native scoring

The one-document `sentiment-map` command accepts `--compute-threads N` to keep
one output-row team across every chunk and sentiment axis. The width includes
the coordinator and must be 1 through 64; the runtime thread is additional.
Worker storage is admitted once before input or model IO, and all workers join
before the complete guarded output is returned. Library callers use
`NlpEngine::map_int4_sentiment_with_threads` with the same host envelope.

The complete semantic result, task identity, source partition, score spaces,
aggregation and whole-document work ceilings use the existing serial path.
An explicit team adds a separate `compute` receipt with its width and completed
projection/row counters. Width one preserves the existing response. No measured
throughput or model parity is claimed by this integration; the new regressions
remain pending the required DSR and model-present checkpoints.

Teams with more than one compute thread cannot combine with `--cohort-chunks`,
`--cohort-heads` or `--prefill-rows`. Corpus and retained-job commands also
support a team spanning the complete invocation, as described below.
Cancellation and errors retain the same typed failure and publish no partial
document.

Chunks form a disjoint, complete partition in original source order. Every
boundary is a UTF-8 boundary, with both byte and Unicode scalar coordinates.
There is no overlap weighting, normalization, truncation or sentence-boundary
guarantee. The pinned source encoder and exact task scaffolds determine the
available source capacity across every selected axis; `--max-chunk-bytes` can
impose an additional bound.

The complete partition and every native candidate schedule preflight before
the first model forward. Each chunk receives the same requested axes and
sentiment policy. Each axis retains its existing five anchors: −1, −0.5, 0,
0.5 and 1. Their interpretations remain axis-specific: negative/positive,
calm/activated, powerless/in-control, and withdrawal/approach.

Every candidate includes its scored terminal EOS. The native scorer computes
the full-vocabulary denominator for each scored sequence step. Each chunk's
conditional candidate weights, sequence scores, distribution moments, entropy,
estimated/abstained decision and work remain in the result. Neither chunks nor
axes share a probability simplex.

Ordinary execution compiles the chunk plans once inside the charged host.
Explicit snapshot saving additionally performs the pre-save planning check
described above. Neither path recompiles a chunk between scoring its axes.

## Aggregate meaning

The reduction profile is `source-scalar-candidate-mixture-v1`. A chunk's weight
is its number of original Unicode scalars divided by the complete document's
scalar count. Thus a multibyte character does not count several times merely
because its UTF-8 encoding is longer. Whitespace belongs to the original
source and contributes to these counts.

For each axis, the reducer combines matching candidate weights in canonical
chunk order. Descriptive moments include **all** chunks, including chunks
whose own policy abstained. A failed native execution is never converted into
an abstention or omitted from a successful result.

| Aggregate field | Meaning |
|---|---|
| `mixture_weights` | Scalar-weighted mixture over the same named candidate anchors. |
| `distribution_mean` | Mean of that descriptive mixture on the existing −1 to +1 scale. |
| `within_chunk_variance` | Weighted mean of each chunk's conditional variance. |
| `between_chunk_variance` | Weighted variance of the chunk means. |
| `distribution_stddev` | Square root of within-chunk plus between-chunk variance. |
| `normalized_entropy` | Entropy of the descriptive mixture, normalized over its candidate set. |
| `estimated_scalars`, `abstained_scalars` | Original source coverage contributed by each chunk decision. |
| `estimated_coverage` | Estimated scalars divided by the complete original scalar count. |
| `coverage_passed` | Whether the explicit minimum coverage policy passed. |
| `mixture_policy_passed` | Whether the mixture passes the request's peak-weight and entropy policy. |
| `decision`, `estimate` | An estimate is present only when both checks pass; otherwise the aggregate abstains. |

Opposing decisive sections can have a mean near zero and substantial
between-chunk variance. That remains distinguishable from sections with
consistently central or diffuse distributions. These dispersions are
descriptive values, not confidence intervals.

`--minimum-estimated-coverage-ppm` defaults to `1000000`, requiring all original
scalars to belong to chunks whose axis result was estimated. The range is
0 through 1000000. Lowering it permits estimates with less estimated chunk
coverage, subject to the mixture policy; setting it to zero removes that
coverage requirement. It does not change the contributing chunks, their
weights, native scores or original decisions.

The result explicitly carries `single_context_equivalence: false`. Changing
chunk geometry can change the estimator because the model receives different
local context. It is not a harmless performance switch. Changing only the
native cohort schedule preserves the same chunk geometry and semantic policy.

## Persistent teams across corpora and retained jobs

`sentiment-corpus` and `sentiment-document-job start|resume` accept
`--compute-threads N`. One persistent numerical output-row team serves the
entire invocation, including every original document, chunk, selected axis,
flush and durable attempt. Width one preserves serial execution; widths 2..64
include the coordinator and admit their worker scratch once, with the runtime
thread additional. Invalid widths and any explicit chunk-cohort/team mixture
refuse before private IO.

This schedule reuses the existing sentiment planner, finite scorer and complete
map reducer. Selected axes and their order, peak/entropy thresholds, abstention,
coverage, source coordinates, per-chunk distributions and aggregate statistics
retain their original authority. Source documents remain sequential; workers
split complete output rows within native projections. Worker count does not
increase task, result, tokenizer or native work limits.

Byte fallback and explicit `safe-bpe-v1` are supported. Compact preparation and
execution reconstruction keep one shared allowance per original document
attempt, and admission retains one encoder workspace independently of team
width. The compact snapshot-save restriction continues to apply.

The corpus defers terminal `run_complete` until all workers physically join,
while charged IO owners remain alive through final delivery. A later failure
can leave earlier complete document frames, so successful whole-run completion
still requires the footer and exit zero. Durable invocations likewise join
workers before returning successful progress, and retain the original deadline
through metadata write and flush.

Population provenance adds optional `compute_schedule` with execution
`int4-output-row-team-v1` and `declared_compute_threads`. It describes selected
physical width without reporting unobserved projection counters. Width one
omits it; inner semantic results retain the serial format. Durable recipes
exclude numerical worker width, allowing explicit resume at another admitted
width without renewing attempts or lifetime work. The separately selected
chunk-cohort width remains authenticated.

Library callers use `NlpEngine::batch_int4_sentiment_documents_with_threads` or
`NlpEngine::job_int4_sentiment_document_with_threads` for widths 2..64; existing
methods retain serial execution. Added resource, cancellation, delivery and
recipe regressions are code-first and await the required DSR checkpoint.
This integration does not claim measured throughput or model-result parity.

## Native chunk cohorts

Use `--cohort-chunks N` to execute adjacent complete chunks through shared
native decoder layers:

```sh
fnlp candidate source-int4 sentiment-map request.json \
  --snapshot /models/nanbeige-g128.int4snap \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256_FROM_TRUSTED_CONVERSION" \
  --memory-mib "$MEMORY_MIB" --context-tokens 2048 \
  --max-chunks 64 --cohort-chunks 4
```

`N` must be 1–64 and cannot exceed `--max-chunks`. Each chunk contributes one
native head per requested axis, and `N × axis_count` must not exceed 64. For
example, four-axis requests can use at most 16 chunks per cohort; single-axis
requests can use 64. The actual combination is checked after strict input
parsing and before model loading. Invalid combinations refuse; the command
does not lower the requested ceiling or silently switch to serial execution.

Each native head owns independent causal KV and scoring state. The last window
uses its actual chunk count. Complete chunks keep their original task budgets,
candidate languages and semantic finalizers. A single failed head fails the
complete document result.

The outer response keeps the same semantic `output` object as serial mode and
adds a `cohort` sibling containing its execution version, requested maximum
chunks, actual chunk-cohort count, native group steps and projection groups.
These are execution counters, not elapsed time or measured speedup. Serial
mode omits the sibling. `--prefill-rows` and `--cohort-heads` are refused on this
route; they describe different scheduling contracts.

## Complete-operation budgets

| Option | Default and scope |
|---|---|
| `--max-input-bytes` | 1048576 for the complete JSON request; maximum 1 MiB. |
| `--max-chunks` | 64; range 1–256. |
| `--max-chunk-bytes` | No additional cap beyond the request and exact prompt capacity unless supplied. |
| `--max-tokenizer-calls` | 8192; range 1–1000000. |
| `--max-candidate-tokens` | 16 per candidate, including EOS; range 2–64. |
| `--max-result-bytes` | 1048576 for one complete native chunk result. |
| `--max-retained-result-bytes` | 67108864 across all retained complete chunk records. |
| `--max-map-result-bytes` | 16777216 for the complete semantic document result. |
| `--preparation-mib` | 1024 of explicit preparation reserve, with additional derived bounds. |
| `--max-forward-positions` | 262144 across all chunks and axes. |
| `--max-projected-logits` | 10000000000 across the complete document. |
| `--max-attention-pairs` | 1000000000000 across the complete document. |
| `--max-dot-products` | 10000000000000000 across the complete document. |
| `--max-multiply-accumulates` | 10000000000000000 across the complete document. |

Retained-result and semantic document limits have a 64 MiB maximum. Hosted
framing and CLI provenance/newline each receive an additional fixed 4096-byte
allowance. These allowances do not increase an individual chunk's native
result cap. Work limits never renew per chunk, axis, cohort or retry.

The pure memory quote runs before input or model IO and prices the complete
declared workload: original input, all retained plans and their axes, one
current native window, complete chunk results, aggregation, resident model,
loading/saving versus execution peaks, publication buffers and cleanup. Host
execution reuses this quote and retains its guards through physical completion;
the CLI retains output and publication ownership through write and flush.
The ledger is an allocation/ownership model, not an operating-system RSS cap.

Before reading the axes, the CLI quotes the largest valid native geometry
under the four-axis maximum and the 64-head ceiling. It still checks the actual
geometry afterward. A request with fewer axes can therefore use less memory
than the up-front quote. Per-chunk task KV authority is fixed for all four
possible axes in both serial and cohort modes; physical resident KV follows
the selected schedule. Scheduling does not silently change the task's budget
or identity.

Tighten chunk, context, retained-output and input limits to the intended
document when choosing the explicit memory ceiling. Failures during planning,
native execution or aggregation produce no partial inference result. A
successful complete operation requires exit 0. A writer, flush or deadline
failure during publication can leave complete or truncated bytes; it returns
an error even if complete JSON has already been written. The deadline is
checked again after successful physical flush, while output reservations remain
held. The end-to-end deadline spans reading, loading, optional snapshot
preparation, inference and publication; blocking IO is cooperative.

## Scope and validation

The command requires `asupersync-runtime`; without it, help remains available
and execution refuses before input or model IO. It retains the current
candidate model's non-authoritative evidence status. It adds neither focus
presets nor sampled justifications, a new sentiment default, semantic quality
certification or model activation.

Regression source covers dispatch, strict wire preservation, policy bounds,
complete memory peaks, schedule-invariant task authority, actual cohort
geometry, native failure typing and complete publication. An ignored
model-present CLI fixture compares serial and cohort semantic results over
multilingual source, verifies original coordinates and checks whole-document
work refusal without partial output. It uses `FNLP_INT4_PREFILL_SNAPSHOT`,
`FNLP_INT4_PREFILL_MODEL_SHA256`, `FNLP_INT4_TEST_MEMORY_MIB`,
`FNLP_INT4_TEST_KERNEL`, and optionally `FNLP_INT4_TEST_GROUP_WIDTH`, in an
isolated test process.

Compilation, tests and native model/performance qualification remain **UNRUN**
pending the controller-selected validation required by `AGENTS.md`. No measured
throughput, model parity or task-quality claim follows from these sources.
