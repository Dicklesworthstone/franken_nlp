# Bounded text batch workflows

`fnlp batch` connects the ordered item-local NDJSON runner to four model-free
workflows: normalization, lossless splitting, exact tokenizer inspection, and
explicitly rules-only redaction. It does not load model weights, run NER,
perform generation, create a worker pool, or persist jobs. Its protocol is
`fnlp-item-local-batch-v1`, not an extension of `fnlp robot schema`.

These are source-level capabilities. This document is not a build, test,
benchmark, production-readiness or model-parity receipt.

## Input, output and failure handling

Each nonempty input line is a complete document:

```json
{"id":"doc-1","text":"Original UTF-8 text","task_args":null}
```

`task_args` may be omitted. When present, it replaces the command defaults for
that document; it does not merge with or mutate defaults for later documents.
Its `kind` must match the selected task. JSON duplicate keys, unknown fields,
invalid UTF-8 and unsupported task modes are rejected, not silently repaired.
`--task-args options.json` supplies bounded run defaults from a regular file;
`-` is not accepted for this configuration file because stdin owns documents.

Only empty LF/CRLF records are ignored. Whitespace-only records are malformed.
The final record may omit its trailing newline. IDs must be nonempty, contain
no control characters, and fit the runner's 128-byte default ID bound. Use opaque
IDs rather than private text: caller IDs are explicitly echoed in the protocol.

Each result is delivered and the writer flushed before the next document is
admitted. Output has delivery sequence, epoch and input coordinates. The inner
`result` contains `schema_version`, `task` and the task-specific `result` object.

Malformed or refused individual documents produce fixed error categories and
processing can continue. Cancellation, exhausted whole-run budgets and broken
output stop the run. The command exits nonzero if any document failed, even if
it reached EOF and emitted `run_complete`. Consumers must check BOTH the
terminal event and its failure count; a delivered prefix is not a complete run.
No output is retried or appended after a failed/partial write or failed flush.
A local flush is not a durability or remote-consumption acknowledgement.

Caller IDs cannot be reused within an epoch, including IDs of rejected tasks.
To acknowledge an epoch and allow reuse, send an explicit control record:

```json
{"flush":true}
```

Flush does not reset delivery sequence, input/output limits, or charged rule
work. Bounds on epoch IDs and their stored bytes keep duplicate detection
bounded; applications must insert flush controls rather than grow it forever.

## Normalization

```sh
printf '%s\n' '{"id":"n1","text":"a\r\nb"}' | fnlp batch --task normalize
```

CRLF/CR become LF. ASCII horizontal trimming and collapsing are opt-in:

```json
{"id":"n2","text":"  a\t b  ","task_args":{"kind":"normalize","trim_ascii_horizontal":true,"collapse_ascii_horizontal":true}}
```

Results retain change maps in original and normalized byte/scalar coordinates.
No NFC/NFKC, case folding or general Unicode whitespace normalization occurs.
Changed interiors and inverse deletion boundaries are not exact point mappings.

## Lossless splitting

```sh
fnlp batch --task split < requests.ndjson
```

```json
{"id":"s1","text":"Original text and all its whitespace","task_args":{"kind":"split","max_chunk_bytes":1024}}
```

Chunk text concatenates to the original bytes. Chunk boundaries preserve UTF-8
and CRLF pairs, and each chunk retains original byte and Unicode-scalar spans.
This is size/whitespace partitioning, not linguistic sentence segmentation.
Chunks are returned inside one bounded document result; an oversized result
fails rather than publishing a partial set of chunks.

## Exact tokenizer counts and IDs

```sh
printf '%s\n' '{"id":"t1","text":"Hello, world."}' | fnlp batch --task tokens
```

Default encoding is the pinned L0 SentencePiece BPE implementation, including
added-token recognition, with BOS enabled and EOS disabled. Reports carry
public tokenizer-asset digests, source byte/scalar lengths, and token count.
Token IDs are opt-in:

```json
{"id":"t2","text":"Hello","task_args":{"kind":"tokens","options":{"include_ids":true,"encoding":{"kind":"pinned_bpe","add_bos":false,"add_eos":true}}}}
```

The reference BPE input cap is **4096 bytes per document**, even when the general
`--max-document-bytes` limit is higher. Longer documents are refused, never
truncated, approximately counted or silently switched to another encoding.
This is tokenizer inspection, not the safe construction of an untrusted prompt;
literal added-token surfaces retain their L0 tokenizer behavior.

Byte-table encoding is a different, explicit mode without added-token
recognition or BOS/EOS insertion:

```json
{"id":"t3","text":"é\r\n上海","task_args":{"kind":"tokens","options":{"include_ids":true,"encoding":{"kind":"byte_fallback"}}}}
```

This mode checks exact byte decoding. Its counts are NOT interchangeable with
BPE counts and must not be used as an approximation of model-context length.
`--max-items` bounds encoded IDs even when IDs are omitted from output. The
pinned tokenizer and asset digests are initialized lazily once per batch run;
no document text or previous token IDs are retained for the next request.
Token IDs can reveal original text: treat explicitly requested IDs as sensitive
result content, not telemetry. Count reports export no source-content digest.

## Verified rules-only redaction

```sh
printf '%s\n' '{"id":"r1","text":"Contact a@example.org"}' | fnlp batch --task redact-rules
```

Default rules cover the existing ASCII email/phone shapes, HTTP(S) URLs, IP
literals and Luhn-valid card shapes. ISO calendar dates are opt-in. The default
action is a typed placeholder. To replace the rule scope and request masking
with an explicit coordinate map:

```json
{"id":"r2","text":"é a@example.org 2024-02-29","task_args":{"kind":"redact_rules","options":{"rules":{"enabled":["email","date"]},"action":"mask","include_map":true}}}
```

The detector union and transactional edit implementation are shared with
single-document redaction. Every result is rescanned against the same declared
rules before it is released. A residual or a scan-budget failure produces no
partially redacted document. Coordinate maps describe edits without including
original matched values; maps remain sensitive, explicitly requested output.

A clean declared scan does NOT establish that all PII is gone. Person,
organization and location NER are not run. Unicode obfuscations may be missed;
text outside selected detections remains unchanged. Empty scopes, NER-only
types, keyed pseudonymization, embedded secrets and verification opt-outs are
refused rather than quietly weakening the requested operation.

`--max-detections` bounds each scan; `--max-items` can impose a smaller ceiling.
`--max-rule-work` bounds EACH of the initial and residual rule scans.
`--max-total-rule-work` bounds the whole run: two scan ceilings are reserved
before an attempted document, including failures, and never refunded or reset
by flush. Exhaustion is a terminal `work_limit` failure. Results separately name
`rule_work_ceiling` and `cumulative_rule_work_reserved`; these are conservative
charges, not measured CPU operations. Native forward/logit work is zero.

## Resource and cancellation boundaries

Input framing, document bytes, total input bytes, request count, output line
bytes, total output bytes, result items, epoch ID count and retained ID bytes
all have separate bounds. `--max-output-line-bytes` covers the complete protocol
envelope and newline, not only transformed text. Terminal-record space is
reserved so ordinary output cannot consume the entire error-reporting budget.

Library callers may supply cooperative cancellation through the existing
`DecodeStepControl` boundary. Text work has checks around bounded operations;
blocking I/O and an in-progress rule scan or BPE call are not preempted. The CLI
does not install a signal handler or promise a terminal record on OS termination.
