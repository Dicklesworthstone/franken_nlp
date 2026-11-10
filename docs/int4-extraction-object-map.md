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

Execution defaults to serial, full-vocabulary greedy decoding with byte input.
Two explicit execution options can be used independently or together:

```bash
fnlp candidate source-int4 extract-object-map long-document.txt \
  --schema record.schema.json --source-membership \
  --snapshot nanbeige.int4.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 32768 --max-chunks 64 --cohort-chunks 4 \
  --max-selected-rows 4096 --max-map-result-bytes 16777216 > complete-record.json
```

`--cohort-chunks N` shares native decoder passes across up to N independent
chunks. N must be 1 through 64 and cannot exceed `--max-chunks`. Every chunk
retains its own prompt, grammar, causal KV and task budget. All real windows,
including a short final window, are preflighted before native execution or an
optional snapshot save. Admission covers the componentwise peak native and
decoder storage, the complete retained document map, and the original-schema
reconciliation workspace. One retained peak reservation covers successive
windows; no work, field, match, scan or output allowance renews per cohort.

`--max-selected-rows R` fixes an exact grammar-legal row ceiling for every
chunk, from 1 through 166144. Every legal token is evaluated, including EOS
when the grammar accepts it. A legal set larger than R refuses the complete
document; the engine never prunes that set or silently changes projection
strategy. The policy and row cap bind every chunk's execution identity before
admission. Selected rows change the planned projection work while preserving
the existing physical-memory quote and all schema, source and reducer limits.

The same original-schema reconciliation runs once after every native chunk
finishes. Conflicts, missing required properties, cancellation or exhaustion in
any later window still prevent a successful final object. The completed result
binds the declared cohort width as well as the actual physical counters, so
different schedules cannot be substituted when their batch counts coincide.

Packed prefill, compute teams, Safe-BPE, root-array union and durable object jobs
remain outside this route's execution contract. Cohort grouping and selected
projection have no measured throughput or task-quality claim yet.

## Streaming complete documents

`extract-corpus --reconcile-root-object` applies the same object contract to
every complete document in an ordered NDJSON stream, using one resident model:

```json
{"id":"record-1","text":"The complete original source for the first record."}
{"id":"record-2","text":"The complete original source for the second record."}
```

```bash
fnlp candidate source-int4 extract-corpus documents.ndjson \
  --schema record.schema.json --reconcile-root-object --source-membership \
  --snapshot nanbeige.int4.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 32768 --max-chunks 64 --cohort-chunks 4 \
  --max-selected-rows 4096 --max-map-result-bytes 16777216 > complete-records.ndjson
```

The flag is explicit and conflicts with `--root-array-union`. All
`--max-object-*` overrides require it; omitting the mode preserves ordinary
per-chunk extraction. Each document retains its partial map and one completed
object, with the same conflict refusal, final required-field validation and
complete-source verification as the standalone command. A complete original
object may exceed a single chunk's output cap: only the derived partial schema
is compiled against that cap, while the full object must fit the declared
object and complete-map limits.

The CLI reserves original/partial-schema and reconciliation workspace before
reading the schema or loading the model. It retains preparation ownership
through optional snapshot saving, then transfers that workspace into the
hosted document frontier through result delivery. Cohort windows and selected
rows compose with the same fixed object policy. Provenance adds only policy
versions and declared limits, without exposing schema contents or private
schema commitments.

Whole-run native-work, mask and verification allowances cover every admitted
attempt. Exact `{"flush":true}` controls reset the duplicate-ID epoch without
renewing those allowances. Earlier completed documents remain valid if a later
document fails; require the terminal `run_complete` event and exit status zero
for complete-stream success. Per-record `task_args` must be absent, null or an
empty object, so records cannot replace the schema, policy or limits.

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

Selected execution uses
`grouped-int4-selected-row-partial-root-object-document-map-v1` and records the
cap in `output.partial_map.max_selected_rows`. Cohort CLI output keeps the
complete object at the same `output` path and adds `cohort_schedule`, with the
declared maximum chunk count and the actual cohort, group-step, projection and
work counters. Those counters describe execution; they are not speedup claims.
The bounded cohort metadata shares the existing candidate envelope allowance.

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

Library callers may configure the base planner with `with_selected_rows`
before constructing `Int4ObjectExtractionPlanner`. Prepared object maps expose
`cohort_requirements` and `execute_cohorts_with_control`; the preflight's
`verify_cohorts_completed` checks the complete object and sealed schedule
statistics. Hosted callers use `NlpEngine::extract_int4_object_document_cohorts`
with the same config and set `mapping.limits.mapping.reduction.map_batch_chunks`
to the declared width. `Int4ObjectExtractionMapConfig::cohort_requirements`
quotes the native/decoder window separately from `workspace_bytes`, which
includes the full map, both schema parses and reconciliation. The serial API
requires a width of one.

### Resident complete-document streams

The library stream planner also supports this explicit contract. Construct an
`Int4ExtractionDocumentBatchPlanner` with the **original** schema in its
`Int4ExtractionDocumentBatchConfig`, then call
`with_root_object_reconciliation(RootObjectLimits)`. The factory installs the
same fixed partial-object prompt and schema derivation. It retains the original
schema rather than demanding that every chunk satisfy the original root
required set. Configure `with_selected_rows` and `with_chunk_cohorts` as needed;
serial object execution requires `map_batch_chunks == 1`.

Run the resulting factory through `NlpEngine::batch_int4_extract_documents`.
Before input or model IO, quote its complete ownership with
`int4_extraction_document_object_corpus_requirements`, passing the same compiler
limits and root-object limits. The host independently derives the quote from
the sealed planner and retains the original/partial-schema and reconciliation
workspace through output delivery. The pure
`Int4ExtractionDocumentMemoryLimits::requirements_with_root_object` calculation
supports callers without the hosted feature. Caller-owned factory preparation
must also be covered by its explicit pre-IO reservation.

`NativeInt4DocumentExtractionBatch` returns `Int4ExtractionDocumentRun`. Its
`Map` variant preserves ordinary map JSON; its `Object` variant serializes the
existing complete object envelope directly, without a new wrapper tag. The
`mapped()` accessor provides the retained native chunk results, while
`verification_used()` includes final original-source verification in object
mode. Successful chunks alone cannot satisfy object completion.

Every input document is one complete result. Flush controls reset only the
duplicate-ID epoch. Native work, masks and the complete per-attempt verification
allowance remain debited across documents, failed attempts and flushes. A
conflict, missing required property, cancellation or exhausted allowance prevents
publication of that document; earlier complete records remain available. Changing
the original required fields, reducer limits, selected-row policy or schedule
invalidates previous preparations even when derived partial schemas coincide.

## Qualification status

The implementation includes reducer, planner, hosted admission and CLI
regression sources for complementary fields, conflicts, required fields, exact
decimals, nested atomic values, complete-source evidence, sealed identities,
remaining verification budgets, cancellation and publication limits. Cohort
regressions cover ragged windows, selected-policy binding, exact grouping,
physical counter tampering, underfunded native/decoder reservations and
unchanged original-schema reconciliation. Source
review and syntax parsing do not establish compiled or executed behavior.
Compilation, executable regressions, model quality, numerical behavior and
performance remain unrun pending the controller-owned DSR production checkpoint
and model-present evaluation gates. There is no single-context equivalence or
measured extraction-accuracy claim.
