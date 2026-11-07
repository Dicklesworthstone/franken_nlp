# Exact root-array reduction for INT4 document extraction

Complete-document extraction can now return one combined record array while
retaining every independently validated chunk result. This is an explicit
candidate option on `extract-map`, `extract-corpus`, and
`extract-document-job start|resume`:

```text
--root-array-union
```

The library equivalent is
`Int4ExtractionMapLimits.root_array_union: Some(RootArrayUnionLimits { ... })`.
Serial execution and `--cohort-chunks N` use the same reducer after all chunks
finish. The default remains the original independent-chunk result envelope.

## Record semantics

The schema must have an array at its root. Every chunk still executes the
original schema independently. The reducer visits chunks in source order and
elements in their original array order. A record's first occurrence determines
its position in the final array.

Equality is exact canonical JSON equality. Object property order and equivalent
JSON string escapes do not distinguish records. Numbers use the repository's
bounded exact-decimal representation: `1`, `1.0`, and `1e0` are equal, while
distinct accepted 38-digit integers remain distinct. No record passes through
an `f64` conversion. Arrays inside a record retain their element order.

Each distinct record retains **all** of its origins. An origin names the
original chunk, its array-element index, its original-document byte/scalar
span, and references into that chunk's retained field evidence. Duplicate
records therefore share one final value without losing their separate source
locations. The original exact chunk JSON strings remain in the result.

The fixed policy identifier is
`exact-root-array-union-source-order-v1`. The completed result adds an optional
`root_array_union` envelope containing its exact array JSON and record lineage.
Canonical spelling in this additional array can differ from a chunk's original
JSON spelling while representing the same exact value.

The final array is independently checked against the original schema. Its
`maxItems`, item types, required properties, exact constants, and string bounds
still apply. A document whose combined array exceeds that schema fails as a
whole. There is no silent truncation or choice of a preferred conflicting
record: distinct values remain distinct until a declared limit rejects them.

## Example

A bounded record schema could describe source-backed names:

```json
{
  "type": "array",
  "maxItems": 64,
  "items": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "maxLength": 128,
        "x-fnlp-source": "verbatim"
      }
    },
    "required": ["name"],
    "additionalProperties": false
  }
}
```

With that schema saved locally, add the reducer to the existing explicit-model
map route:

```bash
fnlp candidate source-int4 extract-map document.txt \
  --schema records.schema.json --source-membership \
  --snapshot /models/nanbeige.int4.snapshot \
  --expected-model-sha256 "$EXPECTED_MODEL_SHA256" \
  --memory-mib 32768 --max-chunks 64 \
  --root-array-union \
  --max-union-records 64 --max-union-origins 4096 \
  --max-union-input-bytes 16777216 \
  --max-union-json-bytes 1048576 \
  --max-union-nodes 65536 \
  --max-union-result-bytes 2097152 \
  --max-map-result-bytes 16777216 > document-result.json
```

The process memory ceiling is checked against the declared model, context,
chunk, evidence and reducer envelopes. The example is a configuration example,
not a measured memory or throughput result. Explicit source conversion remains
available through `--source-dir DIR --source-manifest FILE`.

`extract-corpus` accepts the same flags for each complete `text` in its ordered
NDJSON input. A document is published only after its complete reduction and
validation. Earlier completed documents remain valid if a later document fails.

## Independent bounds and memory ownership

The optional reducer has six complete-document limits:

| CLI flag | Bounds | Default |
| --- | --- | --- |
| `--max-union-records` | Distinct final records | 4096 |
| `--max-union-origins` | All input array elements, including duplicates | 16384 |
| `--max-union-input-bytes` | Sum of original chunk JSON bytes inspected | `--max-total-value-bytes` |
| `--max-union-json-bytes` | Canonical final array bytes | `--max-map-result-bytes` |
| `--max-union-nodes` | Parsed JSON value nodes across chunks and final validation | 262144 |
| `--max-union-result-bytes` | Additional aggregate envelope bytes | `--max-map-result-bytes` |

Each override requires `--root-array-union`. Records, origins and nodes accept
1 through 1,000,000, and records cannot exceed origins. Byte limits accept
2 bytes through 64 MiB. The array must fit its aggregate envelope, and both
declared output caps must fit the whole-document result cap. The **combined**
chunk-plus-union result must also fit `--max-map-result-bytes` when serialized.
These are upper bounds, not a promise that every combination will fit in the
declared process memory.

The reducer limits JSON depth to 64 and schema object-key bytes to 1024 so its
independent parsing and diagnostic-path storage have a bounded reservation.
An unsupported depth, key size, schema or cap is a typed refusal.

`RootArrayUnionLimits::requirements()` prices the additional exact parsing,
canonical record index, origin metadata and output workspace. The shared
`Int4ExtractionMapLimits::root_array_workspace_bytes()` validates its relation
to the document envelope. CLI, hosted maps, corpus admission and document jobs
add that same charge once per active document, including cohort execution.
The guarded result remains owned through publication or durable commit.

The reducer performs no model calls and does not renew or widen any native,
grammar-mask or source-verification allowance. Its own complete-document byte,
node and record limits apply across every chunk. Cancellation stops reduction
without publishing a partial document.

## Retained jobs

Document jobs authenticate the fixed reducer policy and every reducer limit in
their private recipe. Repeat the same flags on resume, including retained-input
replay. Enabling, disabling or changing the policy for an existing job causes
a recipe mismatch before pending model execution. Existing serial recipes
without a reducer retain their original omitted-field representation.

The durable unit remains the entire original document. All chunks and the final
array must validate before that document's result is committed. Stdout remains
metadata only; extracted arrays and source lineage belong to the explicitly
retained result. Public candidate provenance names the reduction policy and
numeric limits without including the private schema or extracted records.

## Practical limits and qualification

This is exact record union over chunk outputs. It does not resolve semantically
equivalent but differently worded records, merge arbitrary objects, infer
cross-chunk relationships, or find occurrences in chunks that did not extract
the value. Source membership proves byte membership under its declared policy;
it does not prove the extraction is semantically correct. Chunked inference
does not claim equivalence to one full-context model invocation.

Focused fixtures cover exact decimals, duplicate origins, schema limits,
cancellation, bounded allocation, schedule equivalence, admission and recipe
authentication. They are authored but unrun in this code-first checkpoint.
Nonwriting Rust parsing and source review do not establish compilation, native
parity, crash recovery, quality or measured performance; those remain subject
to the repository's controller-selected DSR and model-present gates.
