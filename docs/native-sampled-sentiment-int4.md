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
the control-excluding byte encoder. The source must fit a single native context
alongside the axis prompt and generated tokens. Increase the explicitly
admitted context for larger documents; this route does not chunk or truncate.
Prompt-injection resistance and explanation quality require empirical tests.

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
whole-schedule admission failures use exit 9; exhausted execution/output/mask
budgets use exit 5; cancellation uses the stable cancellation taxonomy. An
invalid completed sample or no legal completion uses structured no-result exit
10. Internal accounting failures remain exit 1. Any failed sample withholds the
entire semantic bundle, including earlier successful samples.

This route accepts the scalar or explicitly selected supported packed kernel,
with serial scheduling and byte input encoding. It rejects text-generation
processors, stops, token biases, logprobs, prompt chunk schedules and compute
teams. Focus-area presets, sampled corpus/durable jobs, statistical intervals
and automatic default selection are outside this implementation slice.

## Verification status

The change includes source regressions for grammar legality and actual-program
binding, seeded replay addresses, existing greedy behavior, all cancellation
causes, complete bundle admission, Unicode explanation lengths, independent
sample validation, aggregate statistics, CLI rejection before IO and stable
failure categories. Source reviews, nonwriting Rust syntax parsing and diff
hygiene were performed. These are not executed test evidence.

Rust compilation, executable fixtures, production DSR qualification, real-model
replay/cancellation and human-label quality evaluation remain **UNRUN**. The
required production feature graph and DSR runner were unavailable in the
implementation environment; no default-feature or GitHub Actions fallback was
used. The related feature bead stays `in_progress` until those gates are met.
