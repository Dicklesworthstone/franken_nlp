# Native INT8 sampled sentiment

`fnlp candidate sentiment-sampled` generates independent score-and-explanation
samples on an explicitly selected local current-candidate INT8 artifact. It
implements the independent generation part of comprehensive plan §7.5, Mode B,
for valence, arousal, dominance and approach. Every sample produces an integer
score in `[-100,100]` and a bounded, nonempty explanation.

This candidate implementation requires `asupersync-runtime`. Compilation,
executable regressions, real-model replay, task quality and release qualification
remain separate pending controller gates.

## Request and invocation

Use a UTF-8 JSON file such as `sentiment-request.json`:

```json
{
  "document": "The support agent fixed the problem quickly, but the repeated failures were frustrating.",
  "axes": ["valence", "arousal"]
}
```

Omit `axes` to request all four dimensions. Empty documents, empty or repeated
axes, duplicate JSON keys and unknown fields are rejected. The positional input
`-` reads stdin.

```sh
fnlp candidate sentiment-sampled sentiment-request.json \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 12288 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 \
  --temperature-milli 1000 \
  --top-p-ppm 950000 \
  --max-new-tokens 256
```

The command does not discover or download model weights. Its held model source
supplies authenticated artifact facts for planning and the subsequent weight
load. The memory value is a declared process ledger; actual requirements depend
on artifact, context, preparation and output limits.

The seed must contain exactly 64 lowercase hexadecimal digits. Every axis and
sample has an independent semantic address. Reusing the same model, input
encoding, source, task policy, native profile and seed preserves those addresses.
Explicit `--top-k N` limits the legal token distribution before nucleus
selection; omitting it retains the complete legal set for nucleus processing.
All logits are validated before the exact grammar/control mask, temperature,
top-k and top-p policy selects a token. Terminal EOS is scored and can be
selected only after a complete accepting JSON value.

## Input and prompt scheduling

Byte-preserving control-excluding encoding is the default. For compact input,
add `--input-encoding safe-bpe-v1`; the existing finite tokenizer work options
also apply. This policy preserves original UTF-8 bytes and excludes template
control tokens. It changes the token sequence and can change model output.

The source, axis prompt and generated sample must fit one admitted native
context. This command rejects an over-context request instead of truncating
or chunking the source. Each sample starts with fresh logical KV.

Add `--prefill-rows 8` for bounded layer-major prompt execution. Widths from
1 through 64 are accepted, with the declared extra prompt workspace charged
before input or model IO. Prompt width changes scheduling and does not enter the
semantic draw address. Model-present output parity and performance still require
qualification.

## Complete-result semantics and limits

One successful response contains every requested sample and, for each axis,
its mean, median, minimum, maximum and population standard deviation. Individual
scores, explanations, generated-token counts and native work remain available.
The response names its execution profile, sampler, aggregation recipe, public
prompt-template hash and current-candidate artifact provenance. It does not
publish the effective seed or private prompt/content commitments.

The aggregates describe variation among generated scores. They do not establish
calibrated confidence, psychological measurement or human-label accuracy.
Explanations are unverified model output from an untrusted source document.

All axes and samples are admitted together before model weights load. One
nonrenewable set of limits covers the complete bundle:

| Limit | Scope |
| --- | --- |
| `--max-new-tokens` | Each sample, including terminal EOS |
| `--max-justification-chars` | Unicode scalar count in each explanation |
| `--max-sample-json-bytes` | One generated score/explanation JSON value |
| `--max-result-bytes` | Complete task result, including every sample and aggregate |
| `--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs` | Complete native bundle |
| `--max-dot-products`, `--max-multiply-accumulates` | Complete native projection work |
| `--mask-step-node-visits` | One complete grammar-mask traversal |
| `--max-mask-node-visits` | All samples and scored EOS steps |
| `--timeout-seconds` | Preparation, model load, execution and final delivery |
| `--max-checkpoints` | Shared native execution control across all axes and samples |

Admission includes full resident KV, sampler storage, caller-owned preparation,
transferred plans/vocabulary, retained output and optional prompt workspace.
Independent samples reuse one native context while their execution work is
summed. Failed attempts do not produce neutral scores or partial successful
bundles. A failed final output write or flush remains a failure; callers must
also check the exit status because arbitrary stdout cannot retract delivered
bytes.

## Library entrypoints

Use `tasks::sentiment::sampled::quantized::Int8SampledSentimentPlanner::for_model`
or `for_model_with_encoding` with the authenticated `ArtifactIdentity`. Planning
seals the actual model/profile, exact prompts, grammar, policy, complete work and
output bounds into `PreparedInt8SampledSentiment`.

`NlpEngine::execute_int8_sampled_sentiment` accepts the resident `&ResidentInt8`,
that prepared value, a pinned `Arc<SampledSentimentVocabulary>`,
`Int8SampledSentimentHostLimits` and cancellation.
The `execute_int8_sampled_sentiment_layer_major` variant additionally takes
`Int8PrefillLimits`. The hosted path checks the resident resource domain, keeps
native and preparation charges through physical completion and returns a guarded
`Int8SampledSentimentRun` whose output charge survives delivery.

Lower-level callers can use `decode_json_int8_sampled` and
`decode_json_int8_sampled_layer_major` from
`native_engine::constrained_int8`. These retain the same exact JSON/source
grammar driver, native work reconciliation, explicit EOS and abort behavior
as the greedy decoder.

## Ordered resident corpora

Use `fnlp candidate sentiment-sampled-batch` to keep the authenticated INT8
model, vocabulary, compiler and native buffers resident across records. Each
text fits a single context. For example, save the following as `reviews.ndjson`.

```jsonl
{"id":"review-42","text":"The replacement arrived early.","task_args":{"axes":["valence"]}}
{"id":"review-43","text":"Still waiting for a response."}
{"flush":true}
```

```sh
fnlp candidate sentiment-sampled-batch reviews.ndjson \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 12288 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 2 --max-requests 100 --max-input-mib 16 --max-output-mib 32
```

Omitting task arguments requests all four axes. Each stable ID participates in
the private sampling address along with the model and request. Reordering
distinct items or inserting a flush preserves their addresses. An optional
complete `task_args.budget` can tighten the task limits for that item.

The five native work limits and the mask-node limit cover the entire run,
including admitted attempts that fail. A flush starts a fresh ID epoch. It
does not replenish work. Compact-tokenizer limits apply to each record.
A successful item includes every requested sample. Inspect the final
`run_complete` and process status to distinguish full success from a stream
containing rejected records or an interrupted run.

Library callers use `NlpEngine::batch_int8_sampled_sentiment` with the
model-bound planner, pinned vocabulary, `SampledSentimentCorpusConfig`,
`CorpusLimits`, optional `Int8PrefillLimits` and owned input/output streams.
