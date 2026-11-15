# Native INT4 sampled sentiment

The explicit candidate command `fnlp candidate source-int4 sentiment-sampled`
generates independent scores and explanations on the local grouped-INT4
candidate. It implements the independent generation portion of plan §7.5,
Mode B. Each axis and sample starts from an empty logical KV cache and its
own semantic sample address. It does not sample the distribution-mode bins.

This surface requires the `asupersync-runtime` implementation. The repository's
controller production-build, native parity and task-quality gates still apply;
this change does not promote the no-flag public sentiment command.

## Request and command

Create a UTF-8 JSON input such as `sentiment-request.json`:

```json
{
  "document": "The support agent fixed the problem quickly, but the repeated failures were frustrating.",
  "axes": ["valence", "arousal"]
}
```

Omitting `axes` requests all four dimensions: valence, arousal, dominance and
approach. Empty or duplicate axes, unknown fields, duplicate JSON keys and an
empty document are rejected. Input carries document data and axis selection;
the command line supplies all execution settings.

```sh
fnlp candidate source-int4 sentiment-sampled sentiment-request.json \
  --source-dir /path/to/pinned-source \
  --source-manifest /path/to/source-manifest.json \
  --memory-mib 8192 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 \
  --temperature-milli 1000 \
  --top-p-ppm 950000 \
  --max-new-tokens 256
```

The seed must be exactly 64 lowercase hexadecimal digits. Reuse an existing
snapshot by replacing the two source flags with `--snapshot /path/to/model`
and `--expected-model-sha256` containing its authenticated model-content pin.
Source conversion also supports the existing explicit `--save-snapshot` route.
No command above downloads weights. Memory requirements depend on context,
group width and source/snapshot preparation; insufficient admission is refused.

The positional input `-` reads stdin. The complete original document bytes are preserved by
the selected control-excluding encoder. Byte fallback remains the default.
The source must fit a single native context
alongside the axis prompt and generated tokens. Increase the explicitly
admitted context for larger documents; this route does not chunk or truncate.
Prompt-injection resistance and explanation quality require empirical tests.

## Compact input and persistent CPU workers

The explicit options can be used together with either pinned source conversion
or a digest-pinned snapshot:

```sh
fnlp candidate source-int4 sentiment-sampled sentiment-request.json \
  --snapshot /path/to/local-int4-snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib 8192 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 \
  --input-encoding safe-bpe-v1 \
  --compute-threads 4
```

`MODEL_CONTENT_SHA256` must come from the trusted conversion record, independently
of the snapshot being read. These options select candidate execution; they do
not certify model quality or performance.

`--input-encoding safe-bpe-v1` uses the existing bounded BPE encoder for the
document. It preserves exact source bytes, including Unicode, whitespace and
marker-shaped text, while excluding every template-control token. The document
is encoded once and its token sequence is reused across all axes and samples.
Actual token counts govern context admission, and every independent sample
still pays for its complete model prompt. There is no truncation, retry with a
renewed tokenizer budget, or implicit byte fallback when encoding is exhausted.

The complete encoding policy, including its limits, is bound to the template
and sampled request identity. Compact tokenization can change the generated
scores and explanations. Choosing byte fallback again restores the original
byte-policy identity. The optional top-level `input_encoding` field records the
compact policy; byte-default responses retain their existing shape.

`--compute-threads N` selects the coordinator plus persistent numerical row
workers, with an allowed range of 1–64. One means serial execution. Larger
widths use one supervised team for the whole axis-by-sample bundle, including
all decoder projections and output heads. Each sample still starts with empty
logical KV and retains its original semantic seed address; worker count does
not become part of the semantic request identity. The optional top-level
`compute` receipt reports the actual physical schedule. The runtime has one
additional thread, and unsupported or over-budget widths are refused before
private input/model IO rather than silently narrowed.

The planner exposes `with_input_encoding`, `input_encoding` and
`encoding_work` for embedders; the hosted adapter exposes
`execute_int4_sampled_sentiment_with_threads`. Encoder workspace, worker
scratch, native state and complete result ownership are included in admission.
Same-recipe serial/team replay is an explicit model-present gate, not an
assumed result of source inspection.

## Sampling and complete results

For each axis the model produces a closed JSON object with an integer `score`
from -100 through 100 and a nonempty `justification`. The grammar constrains
the score and the Unicode scalar length. A separate final validator checks the
whole JSON value and rejects blank explanations. EOS must be selected from a
complete model logit vector after the JSON grammar reaches an accepting state.
There is no parse repair or retry with a different seed.

Grammar legality and control-token exclusions are applied before temperature,
top-k and top-p sampling. Temperature uses thousandths; nucleus probability
uses millionths. With no `--top-k`, nucleus sampling covers the complete legal
vocabulary. Explicit top-k accepts 1 through 20. Default temperature is 1000
and default top-p is 1000000.

The output protocol is `fnlp-candidate-int4-sentiment-sampled-v1`. Its `output`
contains the following fields:

| Field | Meaning |
| --- | --- |
| `dimensions` | Requested axes in canonical order, with their low/high anchors |
| `dimensions[].samples` | Every independent score and explanation, ordered by zero-based `sample_index` |
| `mean`, `median`, `minimum`, `maximum` | Descriptive statistics within each dimension |
| `population_stddev` | Square root of mean squared deviation, dividing by N |
| `samples_per_axis` | Requested N, identical for every included axis |
| `model_work`, `generated_tokens`, `mask_node_visit_charge` | Actual complete bundle accounting |
| `prompt_template_sha256`, `sampler_version`, `aggregation` | Versioned implementation identities |
| `interpretation` | `generated_score_variation_not_calibrated_confidence` |
| `justification_status` | `unverified_model_explanations_from_untrusted_document` |

The dimension statistics sit directly beside its `samples` array. A one-sample
dimension has population standard deviation zero. Scores are generated
assessments of expressed textual affect; they are not clinical measurements,
human labels, probabilities or guarantees that an explanation is supported.
No confidence interval is reported.

Private sampling addresses bind the authenticated model, prompt, actual grammar,
seed, policy and axis/sample ordinal. Axis ordering in the input does not change
canonical task planning. A successful replay requires identical admitted
settings and compatible numerical execution; no cross-kernel bitwise or
statistical independence claim has been established by model execution here.
The seed and document-dependent identity hashes are omitted from normal output.

## Admission and failure behavior

| Option | Default and scope |
| --- | --- |
| `--samples` | 4 per requested axis; allowed range 1–32 |
| `--max-new-tokens` | 256 per sample, including terminal EOS; range 1–1024 |
| `--max-justification-chars` | 512 Unicode scalars per explanation; range 1–4096 |
| `--max-sample-json-bytes` | 8192 bytes per generated JSON object; range 32–65536 |
| `--max-result-bytes` | 1048576 bytes for the complete task result |
| `--preparation-mib` | 1024; caller preparation and transferred host preparation are separately admitted |
| `--mask-step-node-visits` | 2000000 trie visits per generated step |
| `--max-mask-node-visits` | 32000000000 across the whole bundle |
| `--input-encoding` | `byte-fallback`; `safe-bpe-v1` is an explicit alternative |
| `--max-tokenizer-pair-checks` | 4194304 for one complete source encoding; compact mode only |
| `--max-tokenizer-pair-bytes` | 67108864 examined bytes for that same encoding; compact mode only |
| `--compute-threads` | 1; explicit coordinator-plus-worker width 1–64 |

All five native-work flags—`--max-forward-positions`,
`--max-projected-logits`, `--max-attention-pairs`, `--max-dot-products` and
`--max-multiply-accumulates`—bound the entire requested operation. Preflight
prices the complete maximum-token schedule before its first native forward;
completing a short sample does not renew a budget for another sample. The task
token budget is N × requested axes × per-sample token cap. One admitted native
allocation and sampler allowance serve sequential samples, with logical KV
cleared between them. Native buffers and result ownership remain charged
through physical completion and publication.

The invocation deadline spans input, model preparation, planning, native work
and publication checks. Blocking input/output remains cooperative. The complete
result is validated and serialized before its first output byte; a transport
failure during delivery can still leave partially written bytes and a nonzero
exit. Consumers must require successful completion.

Invalid CLI settings are refused before input or model access. Resource and
whole-schedule admission failures use exit 9; exhausted tokenizer/execution/output/mask
budgets use exit 5; cancellation uses the stable cancellation taxonomy. An
invalid completed sample or no legal completion uses structured no-result exit
10. Internal accounting failures remain exit 1. Any failed sample withholds the
entire semantic bundle, including earlier successful samples.

This route accepts the scalar or explicitly selected supported packed kernel,
with serial scheduling and byte input encoding as defaults. It rejects text-generation
processors, stops, token biases, logprobs and prompt chunk schedules.
Focus-area presets, sampled corpus/durable jobs, statistical intervals
and automatic default selection are outside this implementation slice.

## Verification status

The change includes source regressions for grammar legality and actual-program
binding, seeded replay addresses, existing greedy behavior, all cancellation
causes, complete bundle admission, Unicode explanation lengths, independent
sample validation, aggregate statistics, CLI rejection before IO and stable
failure categories. Source reviews, nonwriting Rust syntax parsing and diff
hygiene were performed. These are not executed test evidence.

Additional regressions cover compact Unicode/control-marker round trips, exact
tokenizer limits, complete repeated-prompt accounting, policy-bound identities,
workspace admission before IO, all cancellation causes inside encoding,
stable default response bytes, bounded physical receipts and publication
deadlines. An ignored controller-only native fixture compares complete seeded
results at serial, two-thread and four-thread widths within each encoding
recipe, then checks width/memory refusal, cancellation and physical drain.

Rust compilation, executable fixtures, production DSR qualification, real-model
replay/cancellation and human-label quality evaluation remain **UNRUN**. The
required production feature graph and DSR runner were unavailable in the
implementation environment; no default-feature or GitHub Actions fallback was
used. The related feature bead stays `in_progress` until those gates are met.
