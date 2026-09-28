# Native redaction corpus streams

`fnlp candidate redact --ndjson` runs a bounded ordered stream of documents with
one resident INT8 model, one native engine and one immutable detector/action/key
scope. Add `--chunked` when each document may exceed one NER context. The chunked
path scans rules over the complete document, merges independently checked native
NER spans, edits once, then optionally repartitions and rechecks the actual edited
text. Verification is enabled by default and the default action is `mask`.

```sh
fnlp candidate redact records.ndjson --ndjson --chunked \
  --model ./model.fnlpq --memory-mib 12288 \
  --max-new-tokens 128 --max-requests 100 \
  --max-total-mask-node-visits 128000000000
```

These are illustrative finite allowances, not measured memory requirements,
throughput results or a promise that every document will fit. A compatible local
current-candidate INT8 artifact and an `asupersync-runtime` build are required.
The command does not select, download or activate a release model. Its outputs
remain explicitly non-authoritative candidate results.

Without `--ndjson`, the existing single-document command and output remain
unchanged. With `--ndjson` but without `--chunked`, each document uses the existing
single-context native redaction adapter. The separate model-free
`fnlp redact --rules-only` command is unchanged. Corpus-only options require
`--ndjson`; long-document options still require `--chunked`.

## Input and policy

Input is one JSON object per line, for example:

```jsonl
{"id":"doc-001","text":"Alice: a@example.org"}
{"id":"doc-002","text":"Bob: b@example.org","task_args":{}}
```

`text` is the exact original UTF-8 document after JSON decoding, not normalized
or trimmed. Omit the path or use `-` to stream stdin, unless stdin is explicitly
reserved for a private pseudonym key. There is no whole-corpus collection. Each
record is bounded before native execution and its delivery finishes before the
next record is processed. Chunked mode refuses empty documents.

`task_args` may be omitted or an empty object. It cannot change the action,
selected rules, NER types, verification, chunking, key, namespace or budgets.
Unknown argument fields fail that record. One `--ner-options FILE` applies to all
documents and is parsed before weights; per-row model options are not accepted.
Use opaque document IDs: the batch protocol echoes IDs, and redaction does not
sanitize them. Input, output and optional coordinate maps remain sensitive.

Existing `--action mask|placeholder|pseudonymize`, `--rules`, `--include-map` and
`--no-verify` apply to the entire invocation. With verification enabled, residual
detections suppress that document's successful result, not just its suspect
fragments. Neither original nor intermediate NER text is emitted as a fallback.

## One pseudonym scope

For pseudonymization, provide an existing high-entropy binary key of 32 through
4096 bytes over inherited stdin, with a separate corpus file:

```sh
fnlp candidate redact records.ndjson --ndjson --chunked \
  --model ./model.fnlpq --memory-mib 12288 \
  --max-new-tokens 128 --max-requests 100 \
  --action pseudonymize --key-stdin \
  --key-id rotation-1 --namespace review-corpus \
  < private.key
```

There is no raw-key argv option. The key is read once and the supplied
`--expected-key-commitment`, when present, is checked before corpus input. One
full-256-bit HMAC context serves every record and flush epoch; equal values with
the same type, key and namespace use the existing deterministic pseudonym rule.
Changing an ID or flushing cannot change that context. The corpus and key cannot
both use stdin. No implicit key generation, truncated-HMAC preflight, persistent
secret storage or durable resume is introduced. Caller-held copies of an
embedding application's key remain that application's lifetime responsibility.

## Per-document and whole-corpus bounds

The existing source flags retain their per-document meaning. `--max-input-bytes`
bounds each decoded original document; `--max-result-bytes` bounds its complete
native redaction result, with additional bounded transport/provenance framing.
The larger transformed-source envelope does not enlarge original input admission.

In chunked mode, the existing long-document native and total-mask ceilings cover
BOTH stages of ONE document, not the entire corpus. `--max-ner-chunks` applies to
each original or verification partition. `--max-mask-node-visits` is per neural
chunk. Rules see the entire source and their work ceiling applies per scan.
Independent occurrence budgets are shared across chunks and both stages of a
document, then start afresh for the next independently admitted document.

The whole-corpus native ceiling is derived, with checked arithmetic, from the
complete per-document reservation multiplied by `--max-requests`. This scales
independent contexts, not a fictitious concatenated attention triangle. The
whole-corpus mask ceiling is derived the same way. Short mode reserves up to two
complete NER contexts per document when verification is enabled.

The following optional flags can tighten each derived whole-corpus ceiling:

| Flag | Accounted work |
| --- | --- |
| `--max-corpus-forward-positions` | Native forward positions |
| `--max-corpus-projected-logits` | Full-vocabulary projected logits |
| `--max-corpus-attention-pairs` | Native causal-attention pairs |
| `--max-corpus-dot-products` | Integer projection dot products |
| `--max-corpus-multiply-accumulates` | Integer projection multiply-accumulates |
| `--max-corpus-mask-node-visits` | Grammar-mask traversal charges |

An override never enlarges derived authority. Every axis must cover at least
one complete per-document reservation or setup refuses. Multiplication overflow
also refuses instead of saturating. Projected-logit reservations are rounded down
to complete vocabulary rows, the only projection granularity executed by the
pinned constrained driver; this excludes no executable native projection.

The complete per-document native/mask reservation is consumed before inference.
Success, early EOS, a failed document, a new epoch or an explicit flush never
refunds it. Actual work is checked against the reservation, but unused authority
is not recycled. Exhausting a whole-corpus axis stops the stream. A configuration
can therefore stop below its request-count limit, intentionally.

| Corpus transport flag | Default | Scope |
| --- | ---: | --- |
| `--max-requests` | 1000 | Nonempty records, including malformed records and flush commands; maximum 100000 |
| `--max-input-mib` | 1024 | All bytes read, including blank lines, oversized records and delimiters |
| `--max-output-mib` | 1024 | All emitted bytes, including candidate provenance and terminal frames |
| `--max-line-bytes` | 1048576 | NDJSON bytes before LF, including JSON syntax, escaping and trailing CR; maximum 4 MiB |

The line ceiling must cover JSON framing, not only decoded text. Fixed allowances
for start/request/EOF/terminal provenance are reserved from the aggregate output
budget. Both the native event sink and candidate wrapper independently enforce
their byte ceilings. A line/output/work refusal is not silently retried with a
smaller detector scope or disabled verification.

## Output, failure and ownership

Each event uses `protocol: fnlp-candidate-batch-v1`, `task: redact`,
`redaction_mode: single_context|chunked`, and `evidence: non_authoritative`.
The `record` member contains the existing ordered batch event unchanged. Numeric
fields are not parsed through a floating-point intermediate. Successful document
events contain only the completed native redaction result, with coordinate maps
only when requested. The long result additionally describes stage geometry,
work, detector scope and its explicit recall/boundary limitations.

The stream is transactional per document, not across the entire corpus. Earlier
completed events remain available if a later record fails. Any failed record or
terminal failure yields a nonzero process exit, even when a protocol
`run_complete` event was delivered. Local rejections can continue only after the
native engine is clean. Cancellation, admission failure, invalid completion or
poisoned native state stops further work. Residual coordinate vectors are dropped
under the live output reservation; transport errors contain fixed codes rather
than private model/parser/source diagnostics.

There is no retry, resume journal, token streaming, parallel/GEMM batching or
per-record model reload. A transport write/flush failure may leave partial external
bytes and is an error. No Drop-based writer flush publishes an incomplete event.

The CLI transfers owned reader/writer handles before taking stdio locks. A single
host invocation owns the resident native engine, deadline and run ledger. Memory
admission includes input/JSON/event staging, epoch IDs, retained IO, preparation,
full resident KV/scratch, edit headroom, every live native chunk result and the
map/reduction frontier. It is not sized only to final redacted text. Storage and
secret borrowers drain before their charges, and output guards survive actual
serialization, writing and flushing. These are modeled process-ledger reservations,
not measured or operating-system-enforced RSS bounds. Blocking IO and individual
bounded tokenizer/scanner operations are not preempted mid-operation.

Embedding applications use `NlpEngine::batch_int8_redact_document` with
`RedactionCorpusConfig<LongRedactionBatchConfig>`, an owned reader/writer, pinned
planner/vocabulary, optional `RedactionPseudonyms`, corpus limits and cancellation.
The default `RedactionCorpusConfig` and existing `batch_int8_redact` still select
the short-document path. Both reuse the same process-owned corpus infrastructure.

## Limits of detection and validation

NER chunks may split entities or lose context. Whole-source rules only cover
their declared patterns; source-coordinate correctness does not prove semantic
entity typing or exhaustive recall. A clean declared-detector rerun and stable
pseudonyms are not anonymization. Even a successfully delivered document can
retain PII that the selected model/rules did not detect.

This extension adds 24 model-free regression definitions across the corpus core,
host and CLI, including one feature-disabled test. They cover fixed policy,
per-stage/corpus accounting, counter overflow, identity drift, sequence replay,
error and output-guard lifetimes, completion corruption, memory arithmetic,
mode isolation, owned-IO dispatch, key-source exclusion and commitment rejection.
Private synthetic completions are fault-injection fixtures, not model-success or
quality evidence.

Source/API, ownership and targeted GitHub diff review were performed. Rust
compilation, Cargo tests, scripts, Actions, DSR and full-model execution were NOT
run; executable validation remains with the designated controller. No passing
build, redaction recall, residual quality, privacy certification, throughput or
performance result is claimed. Release/catalog evidence gates are unchanged.
