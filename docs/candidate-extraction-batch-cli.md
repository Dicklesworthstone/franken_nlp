# Resident candidate schema-extraction corpora

`fnlp candidate batch --task extract` connects the existing native INT8 schema
corpus implementation to the candidate CLI. One loaded model, native engine,
prompt compiler and vocabulary serve the entire bounded ordered NDJSON stream.
This is serial resident execution, not parallel neural batching or a durable
job. The `asupersync-runtime` feature and explicit local model remain required.

## Shared schema

```sh
fnlp candidate batch corpus.ndjson --task extract --schema schema.json \
  --model ./model.fnlpq --memory-mib 8192 --max-requests 1000
```

Use the same supported schema subset as `fnlp candidate extract`. A schema file
is bounded at 64 KiB and its exact UTF-8 bytes, whitespace and numeric constants
are retained. No remote schema resolution or floating-point schema conversion
is performed. A simple corpus with a shared schema is:

```jsonl
{"id":"a","text":"Acme opened an office in Paris."}
{"id":"b","text":"Example Corp opened an office in Berlin."}
{"flush":true}
```

`--source-membership` requires `--schema` and applies its
`x-fnlp-source=verbatim` fields to each record's own source. It is not semantic
verification. Neither schema option is accepted for the fixed NER/keyphrase/
summary/QA tasks. `--schema` and `--defaults` are mutually exclusive.

## Explicit per-record configuration

Alternatively, omit `--schema` and supply a complete `ExtractionBatchArgs` in
each record's `task_args`, or supply that same object in a local `--defaults`
file. Its required fields are `schema` (an exact JSON STRING), `grounding`
(`structural` or `source_membership`) and `budget`. For the default CLI context,
token, grammar and result limits, a complete defaults file is:

```json
{
  "schema": "{\"type\":\"string\",\"maxLength\":128}",
  "grounding": "structural",
  "budget": {
    "max_input_tokens": 1536,
    "max_output_tokens": 512,
    "max_output_bytes": 1048576,
    "max_grammar_states": 4096,
    "max_kv_bytes": 369098752
  }
}
```

The same object can be a record's `task_args`. Those numbers are tied to the
2048-token CLI context; changing host limits requires compatible budgets. Each
record override replaces the defaults for that record only. It may select a
different bounded schema, but cannot raise the host ceiling, change models or
backends, supply tokens/instructions, or enable sampling/thinking/tools. Missing
arguments without defaults are a document error, not an invented schema.

Defaults are capped at 1 MiB; duplicate and unknown configuration keys are
rejected. Every embedded schema uses the exact-number, duplicate-rejecting JSON
parser. Structural defaults are additionally compiled before loading weights.
Source-bound defaults receive syntax/budget checks at setup; full source-bound
schema compilation necessarily runs per document before its first forward.
No fabricated empty/example source stands in for a future record, and setup
success never certifies a schema's source membership. Per-record overrides
undergo the same actual native preparation and admission checks.

## Lifetime, budgets and output

The command transfers owned buffered IO into one existing hosted corpus call.
The CLI holds no stdin/stdout locks while waiting for that worker. Preparation,
IO, weights, KV, workspace and output remain charged through physical completion
and delivery. Preparation now checkpoints the same corpus controller around
bounded tokenizer/grammar work; it does not reset a deadline or claim to preempt
blocking IO or an in-progress compiler operation.

All five model-work counters, grammar-mask visits, input/output bytes and
request counts retain their existing nonrenewable corpus ceilings. Flush resets
only the duplicate-ID epoch after delivery; it does not refund failed work.
The wire budget includes provenance on every line and reserved terminal framing.
The existing candidate writer preserves inner canonical event bytes and exact
extracted JSON strings; no JSON number is parsed and reserialized by wrapping.

Every event retains `task=extract`, `scope=real-artifact-current-candidate` and
`evidence=non_authoritative`. Native extraction results include exact JSON,
grounding information and complete model work. A failed record is `doc_error`,
not null/empty successful extraction; a terminal error stops admission. Any
failed document makes the process exit nonzero even if the stream was drained
and emitted `run_complete`. Partial writes/failed flushes poison transport;
no retry or extra terminal frame is appended to an unknown output prefix.

## Validation scope

Added regression sources exercise task routing, conflicting flags, exact schema
bytes and decimals, configuration/schema rejection, every budget axis, full KV
pricing, per-record overrides, actual per-document source binding, feature-off
no-IO behavior and unchanged aggregate/framing accounting. They do not simulate
successful native model inference. Compilation, Rust tests, real-model runs,
quality/performance and controller DSR qualification were not executed in this
session. Existing artifact, numerical and release gates remain independent.
