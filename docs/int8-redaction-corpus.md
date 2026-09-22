# Process-hosted INT8 redaction streams

`NlpEngine::batch_int8_redact` connects native INT8 NER-backed redaction to the
existing bounded, ordered NDJSON runner. The borrowed adapter is
`tasks::redact::batch::NativeInt8RedactionBatch`; it uses the existing extraction
admission trait and output guard rather than defining another permit system.

## Invocation and data contract

Supply a resident model, pinned `SourceTaskPlanner` and vocabulary Arcs,
`hosted::corpus::RedactionCorpusConfig`, optional `RedactionPseudonyms`, existing
`CorpusLimits`, owned `BufRead + Send + 'static` / `Write + Send + 'static`
streams, and one cancellation token. The configuration's `batch` member is an
`Int8RedactionBatchConfig`: fixed NER identity, detector configuration, redaction
request, and whole-stream native/mask ceilings. The NER seed is admitted exactly;
each native pass derives its own source-specific execution identity.

Input uses the existing library batch protocol, for example:

```jsonl
{"id":"document-1","text":"Alice can be reached at a@example.org"}
{"flush":true}
{"id":"document-2","text":"Alice works with Bob"}
```

Task arguments must be absent, null, or empty. Input cannot supply a different
key, namespace, detector type scope, action policy, or `verify=false` override.
`id` is caller-controlled transport metadata and is echoed by the existing
runner; use opaque IDs rather than private text. Source text remains exact
UTF-8. Successful records contain the complete `Int8RedactionRun`, not just
its edited text. Error records contain fixed batch categories, not source,
NER transcripts, prompt hashes, private error payloads or residual coordinates.

The original native redactor performs fresh NER plus rules on each document,
independently checks every source occurrence, applies transactional edits, and,
when enabled, reruns the declared union on the transformed document. The corpus
adapter never substitutes rules-only verification. A residual failure produces
no partial document. Optional verification is fixed by host configuration and
is reported as `NotRequested` when disabled, never as a clean pass.

## Budgets, ownership and failure

The detector's `max_model_work` is a **per-document ceiling for both passes**.
The batch configuration's `max_model_work` is a separate whole-stream ceiling.
All five native axes are reserved atomically before execution; mask reservations
cover all selected passes. The per-document model ceiling, not a prematurely
estimated second prompt, is also reported to the transport work limiter.
Consequently reservation can exceed actual work; native result receipts retain
their actual/reserved accounting. Early completion and failures do not refund
corpus reservations. Flush advances delivery/duplicate-ID scope, not model,
mask, deadline or cancellation budgets. Projected-logit ceilings must describe
whole vocabulary projections, as required by the existing corpus admission.

The hosted method keeps one model/native engine and KV/RoPE/scratch allocation
for the whole invocation. It does not call single-item hosted execution in a
loop, create workers, or reload weights. Existing `CorpusLimits` price transport,
preparation and the supplied IO buffers; `edit_reserve_bytes` additionally prices
rule/union/edit allocation headroom. Intermediate NER result/token staging and
retained namespace capacity are charged explicitly. These are modeled process
reservations, not measured RSS or an allocator interception guarantee.

Each completed output retains its real admission guard through canonical record
construction, write and flush. Output storage drops before its guard. Residual
vectors and native errors drain under that guard before conversion to fixed
batch failures. Cancellation, bad admission identity, broken native state and
unwinding prevent further admission. Safe local refusals can continue only with
clean native state, while retaining their consumed reservation. A sink failure
uses the existing poisoned-output path: no retry, appended terminal record, or
next-document execution after an unknown/partial delivery.

One caller-owned full-256-bit HMAC key/namespace scope serves the entire hosted
stream. Saved key commitments are checked before native allocation or stream IO;
key labels alone are not continuity evidence. No key is generated or persisted.
The borrowed adapter can use one already sealed job-wide 128-bit context, but
this host does not invent a per-document preflight or retain an unbounded value
dictionary. Caller-owned additional key Arcs remain caller-owned.

## Evidence and boundaries

Regression source covers immutable input policy, work/mask overflow and refunds,
sequence reuse, admission drift, cancellation, output/guard drop order, unwind,
key continuity, and the actual NDJSON runner's flush and sink-failure paths.
Private synthetic transaction fixtures exercise transport/accounting only; they
are not model inference, NER recall or numerical-parity evidence. Compilation,
Rust tests and real-model runs were not executed here, per `WIRING.md`.

This is ordered serial streaming, not continuous/GEMM batching, durable snapshots,
checkpoint/resume, corpus-wide atomic publication, or public neural CLI release.
A local IO flush does not prove remote receipt or durability. The controller's
quality/artifact/release gates remain unchanged. A clean declared detector union
still does not prove complete PII removal or anonymity.
