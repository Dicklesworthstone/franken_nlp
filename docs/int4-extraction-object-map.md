# Complete-document root-object extraction

`fnlp candidate source-int4 extract-object-map` fills an object whose properties
may be found in different chunks of a complete UTF-8 document. It uses one
resident grouped-INT4 model, bounded native chunk extraction and deterministic
reconciliation. It is an explicit command with its own prompt and schema
derivation identity.

## How the object is assembled

Consider this original schema:

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string", "maxLength": 128, "x-fnlp-source": "verbatim"},
    "city": {"type": "string", "maxLength": 128, "x-fnlp-source": "verbatim"}
  },
  "required": ["name", "city"],
  "additionalProperties": false
}
```

Ordinary `extract-map` requires every chunk to produce both fields. The object
route derives a second schema with only the root `required` replaced by `[]`.
Both schemas compile independently, and the implementation checks that every
property schema remains identical. A separate fixed instruction asks the model
to emit only complete root-property values supported by that chunk.

If native chunk outputs are `{"name":"Alice"}`, `{"city":"上海"}` and
`{"name":"Alice"}`, the reducer can produce `{"city":"上海","name":"Alice"}`
while retaining both origins of `name`. This is an illustration of reducer
behavior, not a reported model run. The result must satisfy the original schema,
including both required fields. A second `{"city":"Paris"}` produces a typed
conflict refusal; a document with no recovered `city` produces a missing-required
refusal. No successful partial object is published for either case.

Exact canonical values determine equality, including decimal values outside
binary floating-point precision. Property order is deterministic. Objects and
arrays nested inside a root property are atomic: two different arrays conflict,
and two incomplete nested objects cannot be combined to satisfy nested required
fields. Missing values are not replaced by nulls, empty containers or defaults.
No model repair or conflict-precedence pass runs.

## Running the command

Save the schema as `record.schema.json` and provide one complete text document:

```bash
fnlp candidate source-int4 extract-object-map long-document.txt \
  --schema record.schema.json --source-membership \
  --snapshot nanbeige.int4.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 16384 --max-chunks 64 \
  --max-map-result-bytes 16777216 > complete-record.json
```

Use `-` to read the complete document from stdin. The input is text, not NDJSON.
The alternative model route is `--source-dir DIR --source-manifest FILE`.
Snapshot pins must come from an independently trusted conversion record, and
`--group-width` must match the model. Optional `--save-snapshot NEW_FILE` follows
complete native-plan preflight; a later data-dependent conflict can fail the
task while leaving that successfully saved candidate available.

`--source-membership` requires annotated verbatim fields in the schema. Each
chunk first retains the existing exact source-validation contract. The final
object then rechecks its annotated strings against the complete original source
and retains every exact occurrence, including matches in chunks that did not
emit the field. Original byte and Unicode-scalar coordinates remain separate.
Schema property names and trusted prompt text never authorize evidence.

Without source membership, the command provides the original structural schema
contract. Structural validity, lexical membership and agreement between chunk
outputs do not establish semantic accuracy or extraction completeness. A value
split across chunk boundaries may not be recovered; nested values must fit a
single chunk's complete value contract.

## Limits and publication

All native chunks and their actual prompt capacities preflight before any
native forward. The command reuses the existing complete-document limits for
input bytes, chunks, tokenizer calls, native work, masks, retained reduction
values and output. The final independent source check spends the **remaining**
field, match and scan allowances after chunk verification. It does not receive
a fresh verification budget.

| Additional option | Bound |
| --- | --- |
| `--max-object-properties` | Distinct root properties; default 4096 |
| `--max-object-origins` | Total root-property appearances, including duplicates; default 16384 |
| `--max-object-input-bytes` | Aggregate JSON bytes inspected across all chunk outputs; defaults to the map's total-value cap |
| `--max-object-json-bytes` | Final canonical object JSON; defaults to the complete map-result cap |
| `--max-object-nodes` | Parsed chunk values plus final independent parse; default 262144 |
| `--max-object-result-bytes` | Complete object-and-origins envelope; defaults to the complete map-result cap |

The complete retained **map plus object** must also fit
`--max-map-result-bytes`. Each native chunk continues to obey its own
`--max-result-bytes`. Bounded reconciliation storage is reserved in addition to
the existing planner, model, KV, scratch, reduction and guarded-output storage.
Raising a limit requires enough process memory; exceeding a limit refuses the
complete result without dropping fields or origins.

One elapsed CLI deadline spans preparation, model loading, optional snapshot
save, native execution, reconciliation and checked publication. Hosted
preparation, every chunk and the reducer share the hosted checkpoint allowance.
The output guard stays alive through the complete write and flush. Failed
execution never becomes a successful empty object, and partial writes are not
retried as a new result.

Initial execution is serial with full-vocabulary greedy decoding and byte input.
Chunk cohorts, selected rows, packed prefill, compute teams, Safe-BPE, root-array
union and durable object jobs are not accepted by this route. Those separate
execution contracts need explicit future integration.

## Result and library API

The candidate envelope reports the actual model, group width, kernel, optional
snapshot, reconciliation limits and versioned policy. Its `output` retains:

- `partial_map`: every native partial-schema output and original-coordinate
  field evidence, with the existing complete-map native work accounting.
- `object.json`: the independently validated final object, encoded as an exact
  JSON string so downstream readers can choose an exact-number parser.
- `object.properties`: each root JSON pointer and all originating chunk IDs,
  source ranges and indices into that chunk's retained field evidence.
- `object.original_source_fields`: complete-source occurrences for final
  annotated fields when source membership is selected.
- Verification usage and fixed schema-derivation, prompt and reconciliation
  policy identifiers.

`RootObjectResult` is constructed only by the bounded reducer. It exposes
read-only accessors rather than mutable authority fields. The map preflight
binds the original schema, derived schema, original source, all actual chunk
identities and reconciliation limits; completion verifies the exact retained
chunk outputs and evidence against that binding.

Library callers construct `Int4ObjectExtractionPlanner` from an unconfigured,
model-bound `Int4ExtractionBatchPlanner`, then use
`plan_document_with_control` with `Int4ObjectExtractionMapLimits`. Hosted callers
use `NlpEngine::extract_int4_object_document` with
`Int4ObjectExtractionMapConfig`, keeping the original request in its mapping
config. No caller-supplied prompt hook is introduced.

## Qualification status

The implementation includes reducer, planner, hosted admission and CLI
regression sources for complementary fields, conflicts, required fields, exact
decimals, nested atomic values, complete-source evidence, sealed identities,
remaining verification budgets, cancellation and publication limits. Source
review and syntax parsing do not establish compiled or executed behavior.
Compilation, executable regressions, model quality, numerical behavior and
performance remain unrun pending the controller-owned DSR production checkpoint
and model-present evaluation gates. There is no single-context equivalence or
measured extraction-accuracy claim.
