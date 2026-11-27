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

## Complete-document sampled maps

Use `fnlp candidate sentiment-sampled-map` when the complete source exceeds one
context. It accepts the same strict `{document,axes?}` JSON request as the
single-context command. It partitions the original source into contiguous,
nonoverlapping chunks that each fit every requested axis prompt, then generates
the complete independent sample bundle for every chunk.

```sh
fnlp candidate sentiment-sampled-map sentiment-request.json \
  --model /path/to/local-current-candidate-int8-artifact \
  --memory-mib 12288 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 --max-chunks 64 \
  --compute-threads 4 --prefill-rows 8
```

The memory value is an example declared cap. The command checks its complete
memory quote and refuses an insufficient cap. Worker and prefill settings are
optional; a serial document uses the same sample addressing and reduction
recipe. One persistent numerical team can serve all chunks, axes and samples.

Partitioning preserves original UTF-8 bytes, CRLF sequences and source offsets.
The response retains every chunk's original byte and Unicode scalar spans,
every score and explanation, the encoding policy and cumulative work. Equal
chunk text at different source positions receives distinct sample addresses.
Replaying the same model, document, seed and task policy preserves those
addresses. This establishes reproducible draw addressing; executable numerical
parity across schedules remains a separate model-present gate.

### Document replicates and aggregates

For each axis and sample ordinal, the reducer pairs that ordinal across all
chunks and weights the chunk scores by their original Unicode scalar counts.
It retains the checked exact integer numerator and complete source-scalar
denominator for each document replicate, alongside the resulting score.
Mean, median, minimum, maximum and population standard deviation summarize
these complete document replicates.

The reducer shares its recipe with the existing INT4 sampled document path.
These statistics describe the sampled chunk-based procedure. They do not
establish calibrated confidence or equivalence to asking the model about the
entire document in one context. Every explanation remains attributable to its
own source chunk.

### Whole-document limits and completion

The planner preflights every chunk, axis and sample before the first model
forward. No successful document omits a source tail, failed chunk or incomplete
sample. One end-to-end cooperative deadline covers input, model load, planning,
execution, reduction and publication; blocking IO itself is not preempted.

| Option | Document behavior |
| --- | --- |
| `--max-input-bytes` | Complete input envelope; defaults to 1 MiB |
| `--max-chunks` | Maximum complete source partition; defaults to 64, maximum 256 |
| `--max-chunk-bytes` | Optional additional source-byte ceiling for each chunk |
| `--max-tokenizer-calls` | Shared source-sizing probes, including rejected candidates; defaults to 8,192 |
| `--max-new-tokens` | Each sample, including scored terminal EOS |
| `--max-result-bytes` | Complete sample bundle for one chunk |
| `--max-retained-result-bytes` | Cumulative canonical bytes of all retained chunk bundles; defaults to 64 MiB |
| `--max-map-result-bytes` | Complete map JSON, including chunks and aggregates; defaults to 16 MiB |
| Native work and mask options listed above | One nonrenewable allowance across every chunk, axis and sample |
| `--timeout-seconds`, `--max-checkpoints` | Shared complete-document deadline and cooperative control |

Byte fallback remains the default. With explicit
`--input-encoding safe-bpe-v1`, one complete-document tokenizer-work ledger
covers accepted and rejected sizing probes and the final chunk plans. No
tokenizer, native, mask or retained-output allowance renews at a chunk boundary.

The CLI emits one completed response only after full result verification.
Typed cancellation, work exhaustion, partition failure or invalid model output
prevents complete-document success. A transport failure during final stdout
publication can still leave partial bytes, so callers must check process status.

### Complete-document library APIs

`tasks::sentiment::sampled::quantized::long` defines
`Int8SampledSentimentMapLimits`, `PreparedInt8SampledSentimentMap` and
`Int8SampledSentimentMapRun`. Use
`Int8SampledSentimentPlanner::plan_document_with_control` to seal a complete
document plan. The explicit encoding-work and stable-item variants support
callers that manage tokenizer work or document identities themselves.

`PreparedInt8SampledSentimentMap::execute_scheduled_with_control` accepts the
strict INT8 engine, pinned sample vocabulary, complete `Int8JsonBudget`, sampler
allowance, optional `Int8PrefillLimits` and caller control. The result identifies
the execution as `portable-int8-complete-sampled-sentiment-map-v1`.

Hosted callers use `Int8SampledSentimentMapHostConfig` with:

- `NlpEngine::map_int8_sampled_sentiment` for byte-preserving serial execution.
- `NlpEngine::map_int8_sampled_sentiment_with_encoding` for an explicit encoding.
- `NlpEngine::map_int8_sampled_sentiment_with_threads` for one persistent team.
- `NlpEngine::map_int8_sampled_sentiment_with_encoding_and_threads` to combine
  explicit encoding and a persistent team.

The config's `requirements` and `requirements_with_encoding` methods provide
pure memory quotes before model or source IO. Preparation, vocabulary, plans,
KV, scratch and guarded output retain their charges through physical worker
drain. Optional prefill workspace is charged once; the dispatcher separately
charges the numerical workers.

This new route has source-reviewed task, hosted and CLI regression definitions,
including an explicitly armed real-model comparison gate. Compilation, executed
regressions, numerical parity, quality and performance remain pending controller
validation. The current command processes one complete document per invocation.
