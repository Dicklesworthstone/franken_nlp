# Native INT4 sampled sentiment

The explicit candidate commands `fnlp candidate source-int4 sentiment-sampled`,
`sentiment-sampled-batch` and `sentiment-sampled-job start|resume` generate
independent scores and explanations on the local grouped-INT4 candidate. They
serve one document, a bounded ordered NDJSON stream, or an explicitly retained
authenticated population with durable resume, respectively. Each invocation
uses one resident model. They implement the independent generation portion of plan §7.5,
Mode B. Every sample has its own semantic address. By default each sample
starts from an empty logical KV cache; all three commands also offer
explicit document-and-axis-local prompt reuse with `--reuse-prefill`. It does not sample the
distribution-mode bins.

These surfaces require the `asupersync-runtime` implementation; durable jobs
additionally require `metadata-store` on Linux x86-64/AArch64. The repository's
controller production-build, native parity and task-quality gates still apply;
this change does not promote the no-flag public sentiment command.

For a complete source that exceeds one context, use the separate
[`sentiment-sampled-map` route](native-sampled-sentiment-map-int4.md). It retains
the full sample bundle for every lossless chunk and reduces ordinal-paired
scores using exact original-scalar weights. It has a distinct address domain
and an explicit document estimator; the single-context commands described
here retain their existing behavior.

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

## Selected output rows

All sampled command families accept `--max-selected-rows N`, with
`N` from 1 through 166144:

| Command | Fixed policy scope |
| --- | --- |
| `sentiment-sampled` | Every axis and independent sample in one request |
| `sentiment-sampled-map` | Every source chunk, axis and independent sample |
| `sentiment-sampled-batch` | Every record and flush in the resident single-context stream |
| `sentiment-sampled-corpus` | Every complete document in the resident stream |
| `sentiment-sampled-job` (`start` / `resume`) | The authenticated retained single-context population |
| `sentiment-sampled-document-job` (`start` / `resume`) | The authenticated retained complete-document population |

For example, the one-request route can select legal rows and reuse its private
axis prefixes with a persistent numerical team:

```sh
fnlp candidate source-int4 sentiment-sampled sentiment-request.json \
  --snapshot /path/to/local-int4-snapshot \
  --expected-model-sha256 "$MODEL_CONTENT_SHA256" \
  --memory-mib "$MEMORY_MIB" \
  --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --max-selected-rows 4096 --compute-threads 4 --reuse-prefill
```

Every grammar-legal row, including accepting EOS, is projected before the
existing temperature, top-k and top-p policy runs. If the complete legal set
exceeds the cap, the sample refuses before that projection. The strategy does
not prune the grammar, replace top-k, fall back to the full head or retry.
Explanation strings can have wide legal sets, so the example cap does not
guarantee that a request will complete. One failed sample still prevents a
complete bundle or document from committing.

The cap and selected execution version bind admission and private durable
recipes. Repeat the exact cap on resume: adding, removing or changing the
policy refuses before tail repair or pending execution. A full-vocabulary-sized
selected cap is still a distinct policy. Semantic draw addresses retain the
original request/item/chunk/axis/sample identity, independently of the selected
head cap, scheduling width and prefix reuse. This preserves draw addressing;
model-present output parity remains a separate qualification gate.

Planned projected rows use the cap for every allowed output token; actual work
counts every projected legal row. Transformer/attention work and complete mask
allowances remain charged. Full decoder, KV, sampler, output and host storage
stay admitted; selecting fewer rows grants no memory discount. Existing native
and mask ceilings remain scoped to the complete request, document, stream or
job lifetime. Prefix reuse retains conservative cold-work admission. Compact
input and each existing physical schedule remain available under their current
limits, including the exclusion between nonserial teams and layer-major prefill.

Public CLI metadata adds `max_selected_rows` only when requested; full-head
defaults retain their existing shape and recipes. Raw seeds, request bindings
and private draw digests remain absent. Library callers apply
`SampledSentimentPlanner::with_selected_rows` before planning or constructing
batch/job adapters. Whole-document hosted callers set
`SampledSentimentMapHostConfig.selected_rows`. No throughput, task-quality or
full-vocabulary probability-mass claim follows from this candidate integration.

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
Actual token counts govern context admission, and admission conservatively
reserves the complete model prompt for every sample, including when prompt
reuse is selected. There is no truncation, retry with a
renewed tokenizer budget, or implicit byte fallback when encoding is exhausted.

The complete encoding policy, including its limits, is bound to the template
and sampled request identity. Compact tokenization can change the generated
scores and explanations. Choosing byte fallback again restores the original
byte-policy identity. The optional top-level `input_encoding` field records the
compact policy; byte-default responses retain their existing shape.

`--compute-threads N` selects the coordinator plus persistent numerical row
workers, with an allowed range of 1–64. One means serial execution. Larger
widths use one supervised team for the whole axis-by-sample bundle, including
all decoder projections and output heads. Each sample retains its original
semantic seed address; worker count does
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
selects bounded layer-major prompt execution. By default this runs for every
independent sample; `--reuse-prefill` runs it once per document and axis.
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
Each attention row sees only its own causal prefix. The default starts every
sample with empty logical KV. The two model passes, final-prefill output head,
grammar mask, sample address and scored terminal EOS use the same contracts.
Generation after the prompt remains sequential.

The native API exposes `decode_json_int4_sampled_layer_major`; prepared tasks
expose `execute_layer_major_with_control`; the hosted entry point is
`execute_int4_sampled_sentiment_layer_major`. The sampled result shape and
semantic identity are unchanged. The corpus wrapper reports the selected
prompt schedule separately. Exact same-recipe result parity and performance
remain controller model gates; no speedup is claimed from source inspection.

## Reuse the prompt across independent samples

For any of the three commands, add `--reuse-prefill` to avoid repeatedly
forwarding the same axis prompt. This works with byte fallback or compact BPE,
source conversion or pinned snapshots, and either serial compute, one persistent
numerical team, or serial layer-major prefill. The existing restriction against
combining multiple compute threads with `--prefill-rows` still applies. Batch
execution fixes this policy for the invocation. Retained jobs authenticate it
at creation and require the same choice on every resume.

The first sample of each axis executes its complete prompt. Before a later
sample, the private native session rewinds to the position immediately before
the prompt's final token. It reappends that token, then generates the new
continuation. All generated continuation KV from the prior sample is removed
from the logical context. The session retains no previous sample's generated
tokens in the next sample's attention prefix. No KV copy or shared cache is
created; the prefix is dropped at the end of the axis and never crosses a
request boundary. The full prompt, model identity, grammar, seed and addressed
sample ordinal still determine the same private sampling address.

The result's `execution` field is
`grouped-int4-independent-sampled-sentiment-reused-prefill-v1` in reuse mode.
Per-sample and complete-bundle `model_work` report executed work. For a prompt
of P tokens and a completed sample of G generated tokens, including EOS, the
first sample forwards P + G − 1 positions; every later sample forwards G
positions starting at logical position P − 1. Attention accounting includes
the retained prompt prefix. Output projections and mask charges still occur
for every generated token, including the scored EOS. Projection uses the full
vocabulary by default or every legal row under the explicit selected-row cap.
For N samples this avoids (N − 1) × (P − 1) forward positions per axis. This is
work accounting, not a measured latency or quality claim.

Admission retains the existing complete cold-sample work ceiling and context,
sampler and output reservations. A failed sample still withholds the complete
bundle; cancellation retains its original cause and poisons and clears the
private session. Default execution and its result identity are unchanged.
Prepared-task embedders can use `execute_reusing_prefill_with_control`, and
hosted callers can use `execute_int4_sampled_sentiment_reusing_prefill` or
`execute_int4_sampled_sentiment_reusing_prefill_with_threads`.

For a resident corpus, use
`batch_int4_sampled_sentiment_reusing_prefill` (with optional layer-major
prefill) or `batch_int4_sampled_sentiment_reusing_prefill_with_threads`.
Every record finishes and clears its private native state before another
record is admitted. No prefix is shared between documents or preserved by a
flush. Every output frame, including empty-run and terminal frames, carries
an optional `prefill_reuse` execution version when selected. The default
wrappers omit this field.

For retained jobs, `SampledSentimentJobPlanner::with_reused_prefill` freezes
both sampled-task and native execution versions in the existing private
authenticated recipe. Changing the flag in either direction refuses resume
before uncommitted-tail repair or new inference. A pending item starts its own
fresh axis sessions after restart; model KV is never persisted. An already
committed item needs no new samples or work debit. Actual bundle work remains
separate from the complete cold attempt charge in the durable ledger.

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

The single-document output protocol is `fnlp-candidate-int4-sentiment-sampled-v1`. Its `output`
contains the following fields:

| Field | Meaning |
| --- | --- |
| `dimensions` | Requested axes in canonical order, with their low/high anchors |
| `dimensions[].samples` | Every independent score and explanation, ordered by zero-based `sample_index` |
| `mean`, `median`, `minimum`, `maximum` | Descriptive statistics within each dimension |
| `population_stddev` | Square root of mean squared deviation, dividing by N |
| `samples_per_axis` | Requested N, identical for every included axis |
| `execution` | Distinguishes default cold execution from explicit prompt reuse |
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
| `--reuse-prefill` | Off; document-and-axis-local prompt reuse in one-off, batch and durable modes, preserving conservative cold-work admission |

For the single-document command, all five native-work flags—`--max-forward-positions`,
`--max-projected-logits`, `--max-attention-pairs`, `--max-dot-products` and
`--max-multiply-accumulates`—bound the complete axis-by-sample bundle. The batch
and durable-job sections below specify their population and lifetime scopes. Preflight
prices the complete maximum-token schedule before its first native forward;
completing a short sample does not renew a budget for another sample. The task
token budget is N × requested axes × per-sample token cap. One admitted native
allocation and sampler allowance serve sequential samples. Default execution
clears logical KV between samples; explicit reuse clears generated continuations
between samples and clears the complete prefix at each axis boundary.
Native buffers and result ownership remain charged
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
Focus-area presets, statistical intervals and automatic default selection
remain outside this implementation slice. Explicit retained jobs are described
below; their lifetime work limits are separate from per-document CLI ceilings.

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
  --reuse-prefill \
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
in the same command. With `--reuse-prefill`, this selects bounded causal
prompt morsels for the first sample of each axis in every document. Omitting
reuse prefills every independent sample. Widths 1–64 are supported; the prompt schedule
cannot combine with a larger compute team. Compact input and source,
snapshot-load and snapshot-save routes remain available.

Adding `--reuse-prefill` applies the same exact prompt reuse within every
document and axis. It works with either prompt morsels or persistent numerical
workers. Every record starts and ends with empty logical KV, including repeated
IDs after a flush; there is no cross-document cache or reuse across attempts.
The first sample per axis still executes the complete prompt. Later samples
execute its final token and their own independently addressed continuation.
Lower actual work does not relax the full cold work/mask debit or admit extra
records under the same invocation budget.

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
| `--reuse-prefill` | Off by default; reuse only the current document/axis prompt, without refunding cold work reservations |

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
selected schedule without claiming a measured speedup. With `--reuse-prefill`,
the optional `prefill_reuse` field names
`grouped-int4-independent-sampled-sentiment-reused-prefill-v1` on every event,
including failures and flushes; it is absent by default. Its `record` retains the canonical
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

For explicit prompt reuse, construct the sealed adapter with
`NativeSampledSentimentBatch::new_reusing_prefill`, optionally supplying an
admitted layer-major prefill limit. Hosted callers can use
`NlpEngine::batch_int4_sampled_sentiment_reusing_prefill` or
`NlpEngine::batch_int4_sampled_sentiment_reusing_prefill_with_threads`. These
entry points share the existing admission and physical-drain paths. The
processor requires the selected schedule's execution marker before accepting a
result; task finalization independently checks complete sample work and values.

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

`fnlp candidate source-int4 sentiment-sampled-job start|resume` runs the same
independent sampled-sentiment task as a durable population job. One committed
item contains the complete result for one document: every requested dimension,
every independent sample, and the validated aggregate. A failed or interrupted
bundle produces no successful partial item.

This route requires `metadata-store` and `asupersync-runtime` on Linux
x86-64/AArch64. The model remains an explicitly selected source-converted or
independently digest-pinned candidate. This feature does not establish model
quality, performance, release readiness, or canonical artifact authority.

### Input and immutable configuration

Supply an NDJSON population with stable IDs unique across the entire frozen population:

```jsonl
{"id":"first","text":"The service was helpful. I feel calm.","task_args":{"axes":["valence","arousal"]}}
{"id":"second","text":"The parcel arrived broken.","task_args":{"axes":["valence","arousal"]}}
```

Each record accepts `id`, `text`, and optional `task_args`. Omitted or null
`task_args` selects all four affect dimensions. `task_args` may contain
`axes` and an optional complete `TaskBudget` whose fields only tighten the
invocation's task ceiling. It cannot replace the seed, sample count, model,
encoding, or sampling policy. Each document must fit one native context. Flush
controls are not accepted in durable jobs; do not reuse the flush or duplicate-ID
examples from the streaming section as a durable population.

Creation requires explicit `--store-results`, an existing protected
`--job-dir`, a protected `--key-file` containing exactly 32 raw secret bytes,
an explicit 32-character lowercase hexadecimal `--job-id`, and a
`--limits` file. The CLI does not generate the key or ID, create the job
directory, or silently adopt existing content. The sampling `--seed` is
separate from the job-authentication key and must be supplied as 64 lowercase
hexadecimal characters.

These are the existing [owned job storage](owned-jobs.md) requirements. The
job ID and authentication key should be independently generated random values.

The authenticated recipe includes the effective seed, sampling policy,
task/grammar/encoding ceilings, model and group identity, per-item work
ceilings, reuse execution policy, and the complete immutable job limits. Repeat them on resume.
Selecting `--prefill-rows` also freezes its row and workspace configuration;
changing that choice or `--reuse-prefill` requires a different job. A legal compute-team width is
a physical execution choice and does not replenish any durable work allowance.

### Lifetime work and storage limits

The five native work flags and `--max-mask-node-visits` cap **one complete
document**, across all requested dimensions and independent samples.
`JobLimits.max_work` supplies the separate lifetime allowances for the job,
including failed attempts. Each lifetime axis must be at least its per-document
ceiling. The complete planned native and mask charge is journaled before an
attempt executes. Cancellation, early EOS, a failed generation, and restart
do not refund it. Item count and `max_attempts` do not multiply the allowance
automatically.

For a two-record example with the default per-document work ceilings, this
`sampled-job-limits.json` allows up to ten complete maximum-sized attempts.
It is an explicit budget example, not a measured work requirement:

```json
{
  "max_items": 2,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 8192,
  "max_snapshot_bytes": 131072,
  "max_result_bytes": 1048576,
  "max_spool_bytes": 67108864,
  "max_materialized_bytes": 8388608,
  "max_journal_bytes": 16777216,
  "max_attempts": 10,
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

The immutable input cap covers each complete NDJSON envelope and must fit
`--max-input-bytes`; sampled job IDs are limited to 256 UTF-8 bytes.
`--max-result-bytes` bounds the complete sample bundle and must fit the
stored result limit. Disk ceilings do not stand in for RAM admission. The CLI
prices the actual original-population or retained-record owner, planner and
vocabulary, native KV/scratch, sampler, result staging, database/serialization
allowances, and optional team or prefill workspace before opening the key,
model, or population. `--memory-mib` remains a modeled process-ledger ceiling,
not an operating-system RSS limit.

### Start and resume

The following Bash example uses a previously created pinned snapshot and
pre-existing protected job credentials. Set `SNAPSHOT`, `MODEL_SHA256`,
`SAMPLING_SEED`, `JOB_DIR`, `JOB_KEY_FILE`, and `JOB_ID` to their actual
values. `MODEL_SHA256` must come from a trusted independent conversion or
receipt. Adjust the declared memory ceiling for the selected model and host.

```bash
sampled_job_args=(
  --snapshot "$SNAPSHOT"
  --expected-model-sha256 "$MODEL_SHA256"
  --memory-mib 8192
  --seed "$SAMPLING_SEED"
  --samples 2
  --reuse-prefill
  --max-input-bytes 8192
  --max-result-bytes 1048576
  --store-results
  --job-dir "$JOB_DIR"
  --key-file "$JOB_KEY_FILE"
  --job-id "$JOB_ID"
  --limits ./sampled-job-limits.json
)

fnlp candidate source-int4 sentiment-sampled-job start ./population.ndjson \
  "${sampled_job_args[@]}" \
  --spool-input --max-retained-input-mib 16 --materialize

fnlp candidate source-int4 sentiment-sampled-job resume \
  "${sampled_job_args[@]}" \
  --use-retained-input --max-retained-input-mib 16 --materialize
```

Input retention is separate consent from result retention.
`--spool-input` stages exact private input envelopes in the protected job,
validates the complete typed population, and authenticates it before inference.
`--use-retained-input` restores that population under the exclusive job lock;
omit the positional input argument. It does not open an original input file or
read stdin, and corrupt or missing retained bytes never trigger a fallback.
Replay keeps one record's payload at a time while retaining the complete ordered
identity index under the immutable item/snapshot ceilings. See
[retained input replay](retained-inputs.md) for the storage and recovery rules.

To retain results while continuing to supply the original population, omit
both input-retention flags and their disk cap. Pass the **complete original
NDJSON population** to both `start` and `resume`, including already committed
records. Resume authenticates it before skipping committed items. Reordering,
editing, or substituting records is refused.

Use `--source-dir` plus `--source-manifest` instead of the two snapshot flags
to convert explicitly selected pinned local sources. An explicit
`--save-snapshot NEW_FILE` can preserve that conversion for later invocations;
it never overwrites a path. Resume still requires the same model/group identity.
`--compute-threads 2..64` uses one persistent projection team for the whole
invocation. Alternatively, `--prefill-rows 1..64` selects bounded layer-major
prompts with serial compute; it cannot be combined with multiple compute
threads. `--input-encoding safe-bpe-v1` is supported, and its complete bounded
encoding policy is authenticated.

To select the compact, persistent-team route in the example, add
`--input-encoding safe-bpe-v1 --compute-threads 4` to both invocations. A legal
compute width can change on resume; the chosen encoding, prefill and reuse recipe
must remain identical. No cross-schedule parity or speedup follows from source
inspection.

### Library entry points

`SampledSentimentJobPlanner::new` accepts the actual model-bound task planner,
the pinned vocabulary, native per-item/lifetime ceilings, and immutable
`JobLimits`. `with_layer_major_prefill` freezes an optional prompt schedule.
`with_reused_prefill` selects private prompt reuse and may be combined with
that schedule in either builder order; both choices remain authenticated.
`sampled_sentiment_job_preparation_bytes` is available without the optional
storage backend, and `int4_sampled_sentiment_job_requirements` quotes the
complete hosted execution envelope. Pass the sealed planner and explicit
`SourceJobRequest` to `NlpEngine::job_int4_sampled_sentiment` or
`job_int4_sampled_sentiment_with_threads`, with explicit `JobHostLimits`.
`job_int4_sampled_sentiment_layer_major` also accepts and freezes an explicit
prompt workspace. The shared job runner owns
authentication, durable debit, recovery, result spooling and materialization.

### Completion and recovery

Stdout uses `fnlp-candidate-int4-sentiment-sampled-job-v1` and contains one
bounded candidate-provenance and progress report after the physical host drains;
it does not contain private document text, scores, explanations, or the
sampling seed. Successful results stay in the protected spool.
`--materialize` publishes verified `materialized.ndjson` only after every
record has committed.

Each completed result is synced to the spool before its journal pointer
commits, and its output guard survives both operations. The original
cancellation and remaining deadline are checked for completed-job replay and
before metadata serialization, before its first byte, and after its final flush.

An error can follow durable progress, including during acknowledgement or
metadata delivery. Do not infer rollback from a missing response. Use the
existing authenticated `fnlp job status`, `verify` and `materialize` commands
(which do not load the model), or explicitly
resume with the original configuration. Interrupted uncommitted output stages
or spool tails are refused by default; `resume --discard-uncommitted-tail`
explicitly permits their removal only after authentication. There is no
automatic retry or budget reset. That flag never repairs corrupt retained inputs.

The source includes recipe/population authentication, persistent failed-attempt
debit, complete-bundle publication, retained replay, host-lifetime, and CLI
regression definitions. Their presence is not an execution result; controller
validation and real-model qualification remain separate.

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

Prompt-reuse regression definitions cover exact first-versus-later-sample work,
retained-prefix attention accounting, unchanged generated-token and mask
charges, refusal of mislabeled or forged savings, single-token prompt bounds,
explicit CLI admission, and rejection before population/job IO. The existing
ignored model fixture also compares cold and reused results under serial,
threaded and layer-major schedules and checks physical resource drain. These
definitions have not been executed in this change.

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
Prompt-reuse additions cover cold/reused marker mismatches, unchanged sample
policy and full cold admission, lower actual work without refunds, fresh
record boundaries, poisoned/cancelled completion, and explicit provenance on
every stream event. These regression definitions have not been executed.
These are source definitions and independent review, not executed batch or
real-model evidence. Batch parity, replay, failure and physical-drain evidence
remain part of the controller's model-present gates.

Durable-job regression definitions cover exact seed/sampler/encoding/grammar
and prompt-schedule commitments, complete population authentication before
recovery, all six persisted work axes after failed attempts, retained-input
replay without reading the original stream, and guarded whole-bundle storage.
Host and CLI definitions cover complete memory quotes, queued cancellation,
reader/native failures, worker-join ownership, explicit storage consent,
source/snapshot schedules and bounded metadata-only publication. These have
source review and text-integrity checks; no execution result is claimed.

Rust compilation, executable fixtures, production DSR qualification, real-model
replay/cancellation and human-label quality evaluation remain **UNRUN**. The
required production feature graph and DSR runner were unavailable in the
implementation environment; no default-feature or GitHub Actions fallback was
used. The related feature beads stay `in_progress` until those gates are met.
