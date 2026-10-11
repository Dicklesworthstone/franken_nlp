# Candidate complete-document extraction batches

`fnlp candidate batch --task extract --chunked --schema FILE` reads bounded NDJSON
records whose original text can exceed one native context. It uses one resident
candidate model, one native engine, one process-owned invocation and the existing
ordered batch transport. The default, without `--chunked`, remains single-context
extraction with the existing shared-default or per-record schema behavior.

```sh
fnlp candidate batch documents.ndjson \
  --task extract --chunked --schema schema.json --source-membership \
  --model local.fnlpq --memory-mib 8192 --preparation-mib 1024 \
  --max-chunks 64 --max-document-result-bytes 16777216
```

The model path must name an eligible local candidate artifact. These example
reservations are configuration inputs, not a measured resident-set requirement
or a claim that a particular machine/artifact has passed execution or quality
qualification. The candidate envelope remains `non_authoritative`.

## Fixed schema and original documents

Each input record is an object such as:

```json
{"id":"report-1","text":"The complete original UTF-8 document goes here."}
```

The shared schema is exact JSON text in a local file. In chunked mode, `--schema`
is required and `--defaults` is refused. Records may omit `task_args` or use an
empty object; they cannot override the schema, grounding, chunk geometry, model
profile or any work allowance. Other task families reject these chunked-extraction
switches rather than silently ignoring them.

The actual pinned extraction compiler determines the space left after the schema,
task scaffold and output reservation. Each chunk is encoded and checked separately;
UTF-8 scalar and CRLF boundaries are preserved. Exceeding a source, chunk, context,
work or output limit returns a failure, never a truncated document.

## Result meaning

One successful document event contains ordered independent native extraction
results for **every** source chunk. The complete result names
`portable-int8-exact-schema-document-map-v1` and
`independent-chunk-json-no-global-object-merge-v1`.

Each chunk retains its native JSON as a **string**, preserving exact numeric
literals rather than coercing them through floating point. With
`--source-membership`, native field evidence remains chunk-local and
`original_fields` supplies every verified occurrence in original-document byte
and Unicode-scalar coordinates. Without it, no source evidence is fabricated.

This is not a global merged JSON object, cross-chunk semantic reasoning or an
extraction-quality guarantee. Required fields must be satisfiable independently
in each chunk. In particular, a required verbatim constant absent from a chunk
cannot be repaired with a guessed null or a match in another chunk. Choose an
appropriate schema for independent extraction, such as a bounded array that can
legitimately be empty. Source membership establishes occurrence, not factual truth.

## Limits and lifetime

`--max-input-bytes` still bounds each **complete original document**;
`--max-line-bytes` bounds its entire incoming NDJSON frame.
`--max-chunks` defaults to 64 and accepts 1 through 256.
`--max-chunk-bytes` can impose an additional byte ceiling, but cannot waive actual
schema/context fit. `--max-result-bytes` remains the per-native-chunk ceiling;
`--max-document-result-bytes` defaults to 16 MiB for the complete document result.
Transport framing and candidate provenance are additionally reserved.
`--max-output-mib` bounds all emitted bytes across the stream.

Independent verification defaults to 4,096 fields, 16,384 occurrence spans and
67,108,864 scan steps **per complete document**, configurable with
`--max-extraction-fields`, `--max-extraction-evidence-spans` and
`--max-extraction-scan-steps`. The corpus allowance is the per-document allowance
times `--max-requests`, not a new allowance for each chunk. All five native-work
axes and mask work similarly have checked corpus ceilings derived from independent
chunk capacity, maximum chunks and maximum requests; they do not use the attention
triangle of one fictitious giant context.

Preparation prices all retained schema/prompt/grammar plans for a document.
The host additionally prices the maximum native-result/reduction frontier,
temporary coordinate scans, full four-coordinate spans, owned IO, KV and scratch.
`--reduction-reserve-mib` defaults to 64 MiB of additional modeled headroom.
Reservations are checked before native execution and remain live for the physical
lifetime of the allocations, including output serialization, write and flush.

Each admitted document's exact planned native work, reserved masks and complete
verification allowance are charged atomically and are not refunded. Explicit
flushes, epoch changes, small actual outputs and failed execution cannot renew
these ceilings. Every chunk identity is checked before the first forward.
Cancellation, a later invalid chunk, identity drift or execution failure discards
that document's result and closes the execution adapter. Earlier fully delivered
documents remain valid. A stream with any failed document returns a nonzero exit;
no extra success frame is appended by the CLI.

## Library entrypoints and validation status

Embedding hosts can use `NlpEngine::batch_int8_extract_document` with
`hosted::corpus::ExtractionDocumentCorpusConfig`, `CorpusLimits`, a resident model,
an immutable extraction planner, shared vocabulary, owned reader/writer and an
explicit cancellation token. The lower-level typed adapter is
`batch::extract::quantized::long::stream::NativeInt8DocumentExtractionBatch`.

The implementation includes model-free regression definitions for multi-document
engine reuse, exact child identities, nonrenewable work and verification ledgers,
late failure, cancellation, output guards, host admission, coordinate storage,
CLI flag restrictions, checked scaling and feature-disabled refusal before IO.
These definitions are not passing evidence: compilation, tests and native/model
execution were not run by the implementation worker under the controller-owned
validation policy. Artifact activation and task-quality qualification remain
separate gates.
