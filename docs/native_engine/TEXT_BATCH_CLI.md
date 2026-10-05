# Candidate text generation and chat corpora

`fnlp candidate text-batch` processes bounded NDJSON with one explicitly selected
local current-candidate INT8 artifact. The model is loaded lazily once and reused
for sequential hosted requests; this is not grouped-forward or parallel batching.
An empty input produces a completion frame without opening model metadata or weights.
This source increment has not been compiled or exercised with model weights.
It does not certify the candidate format, artifact, runtime, quality or performance.

## Generate

```sh
fnlp candidate text-batch prompts.ndjson \
  --task generate --model ./local.fnlpq --memory-mib 8192 \
  --max-records 1000 --max-new-tokens 128 \
  --repetition-penalty-milli 1200 --stop '</answer>'
```

Each line is an independent JSON object:

```json
{"id":"doc-001","prompt":"Summarize the following paragraph: ..."}
{"id":"doc-002","sample_index":3,"prompt":"Write a short answer: ..."}
```

IDs must be unique, nonblank, at most 256 UTF-8 bytes and free of control
characters. `sample_index` defaults to zero. Unknown fields, duplicate JSON
keys, malformed UTF-8 and blank records are rejected. No per-record budget,
model, seed or generation-policy overrides are accepted.

## Chat

```sh
fnlp candidate text-batch conversations.ndjson \
  --task chat --model ./local.fnlpq --memory-mib 8192
```

```json
{"id":"conversation-1","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is tokenization?"}]}
```

A transcript is an optional first system message followed by alternating user
and assistant messages, ending in a user message. Every record starts a new
conversation and native request; no previous record's KV state is inherited.

## Generation controls

Buffered `candidate generate`, `candidate chat`, their `candidate stream`
variants, and `candidate text-batch` share these switches:

- `--min-new-tokens`: suppress EOS and stop-suffix completion until the minimum.
  Hard token/byte/work limits still apply.
- Repeatable `--stop TEXT`: exact UTF-8 suffix matching after a complete token;
  matching bytes are retained. A stop can cross token boundaries, but a match
  strictly inside a token is not a suffix. No published bytes are retracted.
- Repeatable `--ban-token ID`, `--repetition-penalty-milli`,
  `--presence-penalty-milli` and `--frequency-penalty-milli`.
- Repeatable `--logit-bias TOKEN=MILLI`: canonical decimal token/value pairs,
  with duplicate token IDs rejected rather than silently overwritten.

Penalties include prompt tokens and committed generated tokens. Bias follows
penalties and cannot override bans. Greedy remains the default; seeded sampling
still requires the explicit 64-lowercase-hex `--seed` and uses the existing
native temperature/top-k/top-p processor order. Log-probabilities remain raw
full-vocabulary scores, not processed sampling probabilities or confidence.

For a fixed artifact, content, options, ID and sample index, the addressed seed
key does not depend on the physical NDJSON line number. The output request_seq
is the physical one-based record order; it is not a sampling address.

## Corpus limits and completion

`--max-input-bytes` bounds each encoded input record, including JSON framing and
line endings. `--max-total-input-bytes`, `--max-records`, and
`--max-total-output-bytes` bound the complete corpus. `--timeout-seconds` covers
input, preparation, loading, execution and delivery across the entire invocation;
blocking IO is not preempted. `--max-checkpoints` remains a per-record native
limit. Whole-corpus forward/logit limits conservatively charge each compiled
plan's worst-case work before execution, without refunding early EOS savings.
Increasing the record ceiling can require a larger preparation reserve because
the duplicate-ID set is explicitly included in the memory model.

Each completed record emits one `result` frame using protocol
`fnlp-candidate-text-batch-v1`, retaining its ID, physical request sequence,
model provenance and independently finalized native result. There are no
provisional token frames. A complete result is staged before stdout is touched,
and its hosted output guard survives write and flush.

A final `batch_complete` frame is emitted only after EOF and successful delivery
of every record. It reports record/input counts, preceding result-frame bytes,
and reserved native work (not measured work or a throughput claim). The footer
is included in the total output budget and space is reserved before each native
request. Whole-corpus success requires this footer AND successful process exit.

Invalid input, duplicate IDs, exhausted budgets, failed native finalization,
and output failures stop processing without retrying records or emitting a
success footer. Earlier completed results remain valid; the corpus is incomplete.
No content is written to a persistent spool, and this command is not a durable
job/resume protocol or the model-free `fnlp batch` command.
