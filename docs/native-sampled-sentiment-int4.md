# Native INT4 sampled sentiment

The explicit candidate commands `fnlp candidate source-int4 sentiment-sampled`,
`sentiment-sampled-batch` and `sentiment-sampled-job start|resume` generate
independent scores and explanations on the local grouped-INT4 candidate. They
accept one document, serve a bounded ordered NDJSON stream, or retain and resume
an authenticated population, respectively. Population routes use one resident
model. They implement the independent generation portion of plan §7.5,
Mode B. Each axis and sample starts from an empty logical KV cache and its
own semantic sample address. It does not sample the distribution-mode bins.

This surface requires the `asupersync-runtime` implementation. The repository's
controller production-build, native parity and task-quality gates still apply;
this change does not promote the no-flag public sentiment command.
Durable jobs additionally require `metadata-store` on Linux x86-64 or AArch64.

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

The planner exposes `with_input_encoding` and `input_encoding`; prepared plans
expose `encoding_work` for embedders. The hosted adapter exposes
`execute_int4_sampled_sentiment_with_threads`. Encoder workspace, worker
scratch, native state and complete result ownership are included in admission.
Same-recipe serial/team replay is an explicit model-present gate, not an
assumed result of source inspection.

## Layer-major prompt prefill

For both single-document and resident-population commands, `--prefill-rows N`
selects bounded layer-major prompt execution for every independent sample.
N is an explicit maximum morsel width from 1 through 64.
Use this option with serial compute (`--compute-threads 1`, the default); a
larger compute team and prompt morsels are mutually exclusive schedules.
Compact input can be combined with either schedule. The source conversion,
snapshot load and optional snapshot-save routes accept the same choices.

For example, replace `--compute-threads 4` in the compact-input command above
with `--prefill-rows 4`. The host reserves the full declared prompt workspace
once for the entire invocation, even if a prompt contains fewer than four rows.
In the corpus route, the reservation remains in place across every record and
flush; it does not multiply by document, axis or sample count.
Each sample still starts with empty logical KV, and each attention row sees
only its own causal prefix. The two model passes, final-prefill output head,
grammar mask, sample address and scored terminal EOS use the same contracts.
Generation after the prompt remains sequential.

The native API exposes `decode_json_int4_sampled_layer_major`; prepared tasks
expose `execute_layer_major_with_control`; the hosted entry point is
`execute_int4_sampled_sentiment_layer_major`. The sampled result shape and
semantic identity are unchanged. The corpus wrapper reports the selected
prompt schedule separately. Exact same-recipe result parity and performance
remain controller model gates; no speedup is claimed from source inspection.

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
| `--prefill-rows` | Omitted; explicit layer-major prompt morsels of 1–64 rows, with serial compute |

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
and publication checks. Blocking input/output remains cooperative. Before
returning a complete bundle, the shared finalizer rechecks the requested axis
and sample order, scores, explanation bounds, implementation metadata and
recomputed descriptive statistics. It reconstructs each sample's five native
work counts from the actual prompt and generated-token count, including EOS,
and reconciles every sample's charged mask allowance and the bundle totals.
Mask charges describe reserved accounting, not measured trie traversal.
The same checks protect serial, persistent-team, layer-major and corpus calls.

The complete result is validated and serialized before its first output byte;
a transport failure during delivery can still leave partially written bytes
and a nonzero exit. Consumers must require successful completion.

Invalid CLI settings are refused before input or model access. Resource and
whole-schedule admission failures use exit 9; exhausted tokenizer/execution/output/mask
budgets use exit 5; cancellation uses the stable cancellation taxonomy. An
invalid completed sample or no legal completion uses structured no-result exit
10. Internal accounting failures remain exit 1. Any failed sample withholds the
entire semantic bundle, including earlier successful samples.

This route accepts the scalar or explicitly selected supported packed kernel,
with serial scheduling and byte input encoding as defaults. It rejects text-generation
processors, stops, token biases and logprobs.
Focus-area presets, statistical intervals and automatic default selection are
outside this implementation slice.

## Resident NDJSON populations

`sentiment-sampled-batch` applies the same independently sampled task to a finite
ordered stream. Each invocation loads or converts the candidate weights once,
constructs one pinned vocabulary and task compiler, and keeps one native engine
resident through all documents. Documents and their independent samples execute
sequentially; the command does not collect the corpus in memory.

Create `sentiment-population.ndjson`:

```jsonl
{"id":"review-1","text":"The support agent fixed the problem quickly.","task_args":{"axes":["valence","arousal"]}}
{"id":"review-2","text":"The repeated failures were frustrating."}
{"flush":true}
{"id":"review-1","text":"The support agent fixed the problem quickly.","task_args":{"axes":["valence","arousal"]}}
```

```sh
fnlp candidate source-int4 sentiment-sampled-batch sentiment-population.ndjson \
  --snapshot /path/to/local-int4-snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib 8192 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 \
  --input-encoding safe-bpe-v1 \
  --compute-threads 4 \
  --max-requests 1000 \
  --max-total-input-bytes 67108864 \
  --max-total-output-bytes 67108864
```

Omit the positional input, or use `-`, for stdin. The existing pinned
`--source-dir` plus `--source-manifest` route and optional `--save-snapshot`
are also available. The authenticated snapshot pin, seed, sampler settings,
resource ceilings, input encoding and physical schedule stay fixed for the
invocation. Work defaults may exhaust before the request limit; the command
does not multiply compute authority by `--max-requests`.

For layer-major prompt execution, use `--compute-threads 1 --prefill-rows 4`
in the same command. This selects bounded causal prompt morsels for every
fresh sample in every document. Widths 1–64 are supported; the prompt schedule
cannot combine with a larger compute team. Compact input and source,
snapshot-load and snapshot-save routes remain available.

### Records and epochs

Every document is a closed `{id,text,task_args?}` object. `task_args` accepts
`axes` and an optional `budget`; omitting axes requests all four dimensions.
Selecting axes or a tighter budget on one record does not change the next
record's defaults. Empty or duplicate axes, unknown fields, duplicate JSON
keys and attempts to supply a seed, model identity or replacement document
are rejected. `text` is preserved exactly and must fit the chosen encoding
and single-context limits; there is no chunking or truncation.

A record may supply a complete `task_args.budget` object to impose stricter
limits than the invocation's task budget. All five fields are required,
positive, and no larger than their admitted invocation ceilings:

| Budget field | Record constraint |
| --- | --- |
| `max_input_tokens` | Maximum tokens in each complete prompted sample |
| `max_output_tokens` | Must admit the full requested axes × samples × per-sample token ceiling |
| `max_output_bytes` | Maximum bytes in the complete sampled result |
| `max_grammar_states` | Maximum states allowed by the compiled output grammar |
| `max_kv_bytes` | Must still cover the existing resident native KV allocation |

An omitted budget uses the invocation's task ceiling. A widened, zero,
incomplete or malformed budget rejects that record before native execution;
later valid records can continue. A tighter output-token cap that cannot
admit the full sample schedule rejects the record instead of reducing
samples or truncating generation. Tightening KV does not resize the resident
engine. Per-record limits never replenish whole-run native, mask, checkpoint,
transport or deadline allowances.

IDs are nonempty UTF-8 strings of at most 128 bytes with no control characters.
Each ID is unique within its numbered epoch, including IDs whose planning or
execution failed. The exact `{"flush":true}` control drains and acknowledges
all earlier output, advances the epoch, and then clears duplicate-ID storage.
It does not change model state, sampler policy, deadlines or resource ceilings.
An ordinary writer flush acknowledges local delivery, not remote consumption
or durable storage. EOF emits a final flush followed by the terminal event.

The caller's item ID is bound into its private sampling address along with
the model, prompt, compiled grammar, seed, policy and axis/sample ordinal.
Different IDs remain separately addressable even when their text is identical.
The same ID and request preserve their address across input ordering and flush
boundaries; sequence numbers and epoch counters do not enter the draw key.
Replays must retain the same semantic item context. The public output does not
reveal the seed or document-dependent identity digests.

### Whole-run budgets and bounded delivery

| Option | Default and scope in the batch command |
| --- | --- |
| `--max-requests` | 1000 nonempty input records, including malformed records and explicit flush controls |
| `--max-line-bytes` | 1048576 bytes before LF, including any CR, JSON syntax and task arguments |
| `--max-total-input-bytes` | 67108864 bytes for the entire stream, including blank lines and delimiters |
| `--max-total-output-bytes` | 67108864 bytes for all candidate wrappers, results, acknowledgements and terminal frames |
| `--max-input-bytes` | Complete source text allowance for each document |
| `--max-result-bytes` | Complete sampled task result per document; framing is separately reserved |
| Five native-work flags listed above | One cumulative ceiling for the entire invocation |
| `--max-mask-node-visits` | One cumulative mask allowance for every document, axis, sample and EOS |
| Tokenizer pair limits | One bounded complete source encoding per record, reused across all its axes and samples |
| `--prefill-rows` | Omitted by default; 1–64 causal prompt rows with one admitted workspace for the entire stream, using serial compute |

Before a document enters native execution, the processor charges its full
maximum-token schedule on all five native counters and its complete mask
allowance. Early completion, admission failure and later sample failure retain
that charge. A flush never restores it. Per-document preparation can reject a
record before native admission without consuming unused model work. The
transport summary reports conservative reserved forward/logit work; successful
task results separately report their actual complete five-axis model work,
generated tokens and mask accounting.

Every output line is a candidate provenance wrapper with protocol
`fnlp-candidate-int4-sentiment-sampled-batch-v1`, scope
`real-source-current-candidate-int4`, evidence `non_authoritative`, and public
actual-model provenance. With layer-major prompts, its optional
`prefill_schedule` records the native execution version, declared maximum rows
and admitted workspace bytes. The field is omitted by default; it reports the
selected schedule without claiming a measured speedup. Its `record` retains the canonical
`fnlp-item-local-batch-v1` event. Events are `run_start`, `doc`, `doc_error`,
`flush` and terminal `run_complete` or `run_error`. A successful `doc` contains
the entire validated sampled result under `record.result`, including every
requested sample and descriptive statistic. Document events carry the request
sequence, input line and byte offset, and echo a valid caller ID when available.
Error categories are fixed; they do not copy source snippets, parser details
or filesystem paths. Empty LF/CRLF records are ignored; whitespace-only lines
are malformed. The final input record need not end in LF.

One bounded frame is staged and checked before publication. Provenance and
terminal frames count against the output budget, and the result's resource
guard survives serialization, write and flush. No whole-population result array
is retained. A partial write or failed flush poisons output: the command does
not retry or append another record to an uncertain output prefix.

Require both `run_complete` and process exit 0. `run_complete` acknowledges EOF
and terminal delivery; inspect its summary because failed documents still
cause a nonzero CLI exit. Earlier complete frames remain valid when a later
document fails. Every sample must succeed before its document result is
published. Recoverable planning or clean task rejection may continue to the
next document. Cancellation, broken native state, host-admission failure and
transport failure stop the invocation. All eleven typed cancellation causes
are preserved, and cancellation never renews consumed work.

### Resident library and compute routes

The library exposes `SampledSentimentBatchPlanner`,
`SampledSentimentBatchLimits` and the sealed `NativeSampledSentimentBatch`
adapter. `sampled_sentiment_batch_preparation_bytes` quotes the largest retained
axis bundle, compiler/vocabulary allowance, result staging and one encoder
workspace before private IO. The task planner's `plan_for_item_with_control`
binds the caller ID while preserving the original one-shot planning methods.
The hosted methods
`NlpEngine::batch_int4_sampled_sentiment`,
`NlpEngine::batch_int4_sampled_sentiment_with_threads` and
`NlpEngine::batch_int4_sampled_sentiment_layer_major` own the reader, writer,
compiler and vocabulary through physical completion and delivery under one
blocking invocation and one run controller.

The CLI defaults to byte fallback and `--compute-threads 1`. Compact encoding
and explicit widths 2–64 use the same admitted source/snapshot routes as the
single-document command. A numerical team persists for the whole stream, and
all workers join before `run_complete` can be published under the original
cancellation control. Owned IO remains charged through that final publication.
The layer-major corpus route reserves its complete declared prompt workspace
before reading population data, including when actual prompts are shorter than
the selected width. Serial and layer-major routes drop their native allocation
before publishing `run_complete`, retaining charged IO and the original
cancellation control through that final write. Invalid rows, insufficient
workspace, or combined team/prefill schedules are refused before execution.
There is no implicit schedule fallback or renewal of native/mask work at
worker, item or epoch boundaries. Memory reservations are modeled commitments,
not measured RSS.

## Durable sampled populations

`fnlp candidate source-int4 sentiment-sampled-job start|resume` connects the
same complete sampled bundles to the authenticated job runner. It retains each
completed result, skips committed items on resume, and durably charges every
attempt against the original job's five native counters and grammar-mask limit.
The model, vocabulary and native engine are reused throughout an invocation.

Use unique `{id,text,task_args?}` records, such as:

```jsonl
{"id":"review-1","text":"The support agent fixed the problem quickly.","task_args":{"axes":["valence","arousal"]}}
{"id":"review-2","text":"The repeated failures were frustrating.","task_args":{"axes":["valence","arousal"]}}
```

IDs are unique across the entire frozen population. Flush controls are refused.
The same closed `task_args` contract as the batch route accepts axes and a
complete budget that can only tighten the invocation's task ceiling. Each item
must fit one native context. Its entire requested sample bundle must validate
before it can commit; a successful prefix of samples is never a successful item.

Prepare an existing owner-only job directory, a protected regular key file
containing exactly 32 random **raw bytes**, a random job ID represented by 32
lowercase hexadecimal characters, and an immutable `JobLimits` file. These are
the existing [owned job storage](owned-jobs.md) requirements. The command requires
`--store-results` and creates neither the directory nor the key.

For example, `sampled-job-limits.json` can declare these finite ceilings:

```json
{
  "max_items": 100,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 8192,
  "max_snapshot_bytes": 8388608,
  "max_result_bytes": 1048576,
  "max_spool_bytes": 268435456,
  "max_materialized_bytes": 268435456,
  "max_journal_bytes": 67108864,
  "max_attempts": 120,
  "max_work": {
    "model": {
      "forward_positions": 1310720,
      "projected_logits": 10000000000,
      "attention_pairs": 1000000000000,
      "projections": {
        "dot_products": 1000000000000,
        "multiply_accumulates": 10000000000000000
      }
    },
    "mask_node_visits": 320000000000
  }
}
```

These limits bound admitted work and storage; they do not predict how many
documents will fit. All six per-item work ceilings must fit their corresponding
lifetime ceiling. Unlike finite distribution scoring, sampled jobs require a
positive mask budget.

Start and explicitly retain both the completed results and the original inputs:

```sh
fnlp candidate source-int4 sentiment-sampled-job start sentiment-population.ndjson \
  --snapshot /path/to/local-int4-snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib 16384 --group-width 128 --max-input-bytes 8192 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 --input-encoding safe-bpe-v1 --compute-threads 4 \
  --store-results --job-dir /private/sampled-job \
  --job-id "$JOB_ID" --key-file /private/sampled-job.key \
  --limits sampled-job-limits.json \
  --spool-input --max-retained-input-mib 64 --materialize
```

Resume without opening the original population or reading population stdin:

```sh
fnlp candidate source-int4 sentiment-sampled-job resume \
  --snapshot /path/to/local-int4-snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib 16384 --group-width 128 --max-input-bytes 8192 \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --samples 4 --input-encoding safe-bpe-v1 --compute-threads 4 \
  --store-results --job-dir /private/sampled-job \
  --job-id "$JOB_ID" --key-file /private/sampled-job.key \
  --limits sampled-job-limits.json \
  --use-retained-input --max-retained-input-mib 64 --materialize
```

Use a population containing only unique document records, without the flush and
duplicate-ID examples from the streaming section. Source conversion using
`--source-dir` plus `--source-manifest`, with optional `--save-snapshot NEW_FILE`,
is also available. The declared memory ceiling must cover the selected model,
context, input mode and job buffers; there is no automatic admission fallback.

Input retention is separate from result retention. Without `--spool-input`,
supply the complete original population to every `resume`, including committed
records. Retained replay instead authenticates the saved population with one
record's payload in memory at a time. Its complete ordered identity index still
obeys the original item and snapshot ceilings. See
[retained input replay](retained-inputs.md) for exact storage and recovery rules.

### Immutable replay and complete work charges

Repeat the original effective seed, sample count, temperature, top-k/top-p,
sample token and explanation bounds, grammar limits, task ceilings, encoding
policy and all native/mask limits. The private recipe is derived from the actual
owned planner and model, not caller-supplied replacement identity fields. It
includes every sampled implementation version and the exact lifetime limits.
Changing the original record bytes, IDs, order, model/group or recipe refuses
resume before any permitted result-tail repair. Seed and recipe bytes are not
copied into public metadata.

The five native-work flags and `--max-mask-node-visits` bound **each item** on
this route. `JobLimits.max_work` is the sole **whole-job lifetime authority**.
Each prepared attempt reserves its complete maximum-token schedule, including
every axis, sample, repeated prompt and grammar-mask allowance, before native
execution. Early EOS, failed samples, interrupted attempts and later invocations
do not refund that work. Actual complete result accounting remains in the
private output; progress reports the durable reservation. Neither `max_items`
nor `max_attempts` multiplies work authority.

Results are synced to the spool before their journal pointer commits, with the
output's resource guard retained through both operations. Uncommitted tails or
output stages are refused by default. Explicit `resume --discard-uncommitted-tail`
selects the existing recovery policy only after authenticating the original
contract. Retained-input corruption is never repaired by that flag.

Serial execution is the default. `--compute-threads 2..64` retains one numerical
team for the invocation; alternatively, `--compute-threads 1 --prefill-rows 1..64`
uses one admitted layer-major prompt workspace across every fresh sample. Both
support compact input. The physical schedule can change on an authenticated
resume without changing sample addresses or replenishing work. All workers join
and native allocations drain before progress returns. Original cancellation and
the absolute deadline are rechecked even for an already-completed job and during
metadata publication. Source review alone establishes no cross-schedule parity
or performance result.

Stdout uses `fnlp-candidate-int4-sentiment-sampled-job-v1` and contains candidate
provenance, optional physical schedule metadata, and authenticated progress.
It omits documents, scores, explanations, raw seeds and private recipe bytes.
`--materialize` publishes ordered `materialized.ndjson` only after every item
commits. Existing `fnlp job status`, `verify` and `materialize` work without
loading the model. A failure can follow durable progress; it does not imply
rollback or authorize restarting as a fresh job.

Library callers build `SampledSentimentJobPlanner` from the real
`SampledSentimentBatchPlanner` and exact `JobLimits`, then use
`NlpEngine::job_int4_sampled_sentiment`,
`NlpEngine::job_int4_sampled_sentiment_with_threads`, or
`NlpEngine::job_int4_sampled_sentiment_layer_major` with `SourceJobRequest` and
`JobHostLimits`. Admission includes the selected input owner, caller and host
preparation, journal and serializer buffers, model, full native KV, one sampler,
physical workspace and complete guarded result. The existing job format and
authentication protocol are reused.

## Verification status

The change includes source regressions for grammar legality and actual-program
binding, seeded replay addresses, existing greedy behavior, all cancellation
causes, complete bundle admission, Unicode explanation lengths, independent
sample validation, aggregate statistics, CLI rejection before IO and stable
failure categories. The current follow-up has source review and text-integrity checks only;
its added regressions have not been executed.

Additional regressions cover compact Unicode/control-marker round trips, exact
tokenizer limits, complete repeated-prompt accounting, policy-bound identities,
workspace admission before IO, all cancellation causes inside encoding,
stable default response bytes, bounded physical receipts and publication
deadlines. An ignored controller-only native fixture compares complete seeded
results at serial, two-thread and four-thread widths within each encoding
recipe, plus one-row and four-row prompt morsels, then checks width/memory
refusal, cancellation and physical drain. Synthetic-driver prefill regressions
cover addressed samples across morsel widths, exact generated tokens and work,
numeric/source-bound/Unicode grammars, native and mask limits, all cancellation
causes and finalizer refusal. CLI admission fixtures cover the complete declared
prompt workspace, short inputs, source/snapshot/save paths and incompatible
physical schedules.

Resident batch regression definitions cover strict record arguments, all five
native counters plus mask debit, failed admission, clean and poisoned failure
paths, cross-factory plans, all cancellation causes, guard ownership through
delivery/unwind, actual NDJSON duplicate/flush handling, partial writes and
post-join footer cancellation. Host fixtures cover complete resident KV,
reused sampler/workspace, compact preparation and every admission axis.
Additional regression definitions cover per-record budget tightening without
native debit on rejected records, exact complete-bundle accounting, bounded
corpus prefill admission, unchanged stream work across prompt schedules,
source/snapshot/save CLI routes and public schedule provenance.
These are source definitions and independent review, not executed batch or
real-model evidence. Batch parity, replay, failure and physical-drain evidence
remain part of the controller's model-present gates.

Durable-job regression definitions additionally cover planner-derived seed and
policy authentication, exact lifetime limits, failed-attempt debits, committed
item skipping, authentication before repair, guarded spool/journal publication,
actual input-mode memory admission and CLI rejection before private IO.
An explicitly ignored independently pinned real-snapshot fixture compares
completed durable bundles, materialized bytes and reserved work across serial,
four-worker and four-row prefill schedules within each encoding policy. It also
checks retained replay without external input, completed-job cancellation,
changed-seed refusal and physical drain. CLI runtime definitions cover private
metadata exclusion and cancellation/deadline checks around final publication.
These additions have source review only; their test bodies have not run.

Rust compilation, executable fixtures, production DSR qualification, real-model
replay/cancellation and human-label quality evaluation remain **UNRUN**. The
required production feature graph and DSR runner were unavailable in the
implementation environment; no default-feature or GitHub Actions fallback was
used. The related feature bead stays `in_progress` until those gates are met.
