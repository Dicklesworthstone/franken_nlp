# Native extraction Assay: exact JSON gold and paired model qualification

`fnlp eval native-extract-int4` evaluates an independently pinned native INT4
snapshot against a frozen, user-owned corpus of schema extraction cases.
`fnlp qualify native-extract-int4` compares candidate and baseline snapshots on
exactly the same held-out cases under explicit field-quality gates. Both use
the actual constrained native extraction engine and emit one canonical report
under `fnlp.assay.native-extract-int4` v1.

One suite can contain multiple schemas, including objects, arrays and scalar
roots. Structural extraction and explicitly source-bound verbatim fields can
coexist in the catalogue. Every schema and gold value is validated before a
snapshot opens. Only locked-test documents execute; all source splits remain
bound to the scorecard.

The report measures exact agreement with supplied gold. Schema validity and
source membership are independently checked execution properties; they do not
establish semantic accuracy. The score space is `not_computed`, with no field
probabilities, Brier/ECE, calibration artifact or activation authority.

```sh
fnlp eval native-extract-int4 --schema
fnlp qualify native-extract-int4 --schema
```

Discovery reads no files, key or model and requires no runtime host.

## Declare schemas and freeze the cases

A suite manifest has the following shape:

```json
{
  "schema_version": 1,
  "task": "structured_extraction",
  "dataset_id": "owned-document-extraction",
  "dataset_version": "2026-10-31",
  "named_population": "Owner-labeled documents in the declared schema families",
  "schemas": [
    {
      "id": "invoice",
      "schema_json": "{\"type\":\"object\",\"additionalProperties\":false,\"required\":[\"amount\",\"currency\"],\"properties\":{\"amount\":{\"type\":\"number\"},\"currency\":{\"type\":\"string\",\"enum\":[\"EUR\",\"USD\"]}}}",
      "grounding": "structural"
    },
    {
      "id": "person-name",
      "schema_json": "{\"type\":\"object\",\"additionalProperties\":false,\"required\":[\"name\"],\"properties\":{\"name\":{\"type\":\"string\",\"maxLength\":64,\"x-fnlp-source\":\"verbatim\"}}}",
      "grounding": "source_membership"
    }
  ],
  "expected_input_sha256": "REPLACE_WITH_SHA256_OF_THE_COMPLETE_CASE_FILE",
  "splits": {
    "development": ["dev-001"],
    "calibration": ["cal-001"],
    "locked_test": ["test-001", "test-002"]
  }
}
```

Replace the illustrative split IDs with every ID in your corpus and the pin
with 64 lowercase hexadecimal SHA-256 digits for the exact complete NDJSON
bytes. All three splits must be nonempty and disjoint. Every declared ID must
appear exactly once in its named split. Exact source text is also unique across
the corpus, so repeating a document under a second schema or split is rejected.
Newline changes alter the input pin.

Each case contains seven fields:

```json
{"schema_version":1,"id":"test-001","split":"locked_test","text":"The invoice total is 12.50 EUR.","schema_id":"invoice","language":"en","expected_json":"{\"amount\":12.50,\"currency\":\"EUR\"}"}
{"schema_version":1,"id":"test-002","split":"locked_test","text":"Zoë arrived.","schema_id":"person-name","language":"fr","expected_json":"{\"name\":\"Zoë\"}"}
```

**`schema_json` and `expected_json` are strings containing exact JSON.** They
are not JSON objects embedded in the outer envelope. This preserves the
supported exact decimal domain, including values with 38 significant digits,
without a floating-point serialization round trip. To prepare the envelope,
read an existing schema or gold file as text and JSON-escape that text as a
string. The evaluator independently parses its contents.

Inner schemas use the existing finite native JSON-Schema subset: object,
array, string, number, integer, boolean and null; bounded arrays; scalar
enum/const; required properties; and `additionalProperties: false` for objects.
Unsupported constructs such as `$ref`, recursive definitions and arbitrary
schema combinators refuse. Schema declarations and gold values use the native
normalization and validation rules, including Unicode scalar string lengths,
under the declared compiler and source limits before model loading. Native
planning also checks whether the schema can produce a complete result for the
actual source. For example, an optional verbatim enum with no available source
value can refuse planning even when the gold omits that property. Preflight
does not waive this executable constraint. Neither a schema filename nor a
remote URL is resolved from a case.

`grounding: "source_membership"` requires verbatim source fields and checks
their gold and native values against the exact document. The structural mode
cannot silently weaken a verbatim annotation. Repeated source occurrences stay
explicit in the private native evidence; the evaluator does not choose an
occurrence to manufacture a unique offset. Because extraction gold names JSON
values rather than span locations, an exact string can match its gold even
when that source string has multiple occurrences.

Outer and inner duplicate keys are rejected. Unknown envelope fields,
malformed UTF-8, missing schemas, invalid gold types, invalid source membership,
foreign split membership and stale byte pins refuse before model loading.
Supplied model outputs, scores and execution identities are not accepted.

`language` is owner-declared metadata, not detected language: one to 63 ASCII
letters/digits with optional nonempty hyphen-separated components. Tags are
preserved exactly. Dataset/population descriptions, schema IDs, language tags
and declared host scopes appear in reports and should be nonsecret. Schema
bodies, property names, source text and gold values remain private.

## Run native evaluation

Provide an existing snapshot and a model-content digest retained independently
from trusted conversion evidence. A digest read from the same snapshot header
is not an independent pin. The existing protected key file must contain exactly
32 secret bytes. Choose a new output filename in an owner-controlled directory.

```sh
fnlp eval native-extract-int4 \
  --snapshot candidate.int4 \
  --expected-model-sha256 "$CANDIDATE_MODEL_SHA256" --group-width 128 \
  --suite extraction-suite.json --cases extraction-cases.ndjson \
  --key-file assay.key --key-id local-extraction-key --namespace extraction-v1 \
  --memory-mib "$ASSAY_MEMORY_MIB" \
  --minimum-samples 20 --bootstrap-resamples 200 --bootstrap-seed 0 \
  --max-schemas 16 --max-fields-per-row 256 \
  --context-tokens 2048 --max-new-tokens 512 \
  --output extraction-scorecard.json
```

Only the exact source text, selected schema and fixed task policy enter model
planning. Gold JSON, case IDs, language metadata and split names never become
prompt instructions. Development/calibration rows are validated and privately
bound, but this command does not fit anything on them.

`--cases -` reads bounded stdin. Output omission or `--output -` publishes a
complete report to stdout. Protected file output stages and publishes a
create-new report while retaining its output memory claim through physical
flush. A stdout sink may fail after receiving a prefix; file publication is
the atomic option.

## Interpret field quality

The scorecard is in the outer `report`. `validated_rows` counts the entire
frozen corpus and `executed_rows` counts locked-test documents. The default
`minimum_samples` is 20, with a minimum accepted setting of five documents.
Below the declared minimum, `measurement.status` is `insufficient_data` and no
measured-quality claim is produced.

For a measured population, `measurement.metrics` contains:

| Field | Meaning |
| --- | --- |
| `fields` | Micro field support/predicted/TP/FP/FN counts and precision, recall and F1. |
| `exact_documents`, `document_accuracy` | Count and fraction of complete JSON values matching gold exactly. |
| `field_interval`, `document_accuracy_interval` | Uncertainty from resampling whole documents and recomputing corpus ratios. |
| `by_schema` | Point estimates for each declared schema ID, including zero-row catalogue entries. |
| `by_language` | Point estimates grouped by owner-declared language. |

A field is a terminal scalar or an empty object/array at its **typed structural
location**. Nonempty containers contribute their descendant fields. Object
key order is irrelevant; array order and indices are significant. An object
key named `"0"` is distinct from array index zero. Parent container-type changes
cannot produce accidental matches through a shared textual pointer.

Missing fields count as false negatives, extra fields as false positives and
wrong values at an existing location as one false positive plus one false
negative. `null`, `{}` and `[]` are distinct terminal values, so empty results
still receive meaningful comparison. Strings compare exactly without case,
whitespace, accent or Unicode normalization. Numbers compare by exact decimal
value: `1`, `1.0` and `1e0` agree, and supported high-precision values are never
rounded through `f64`.

For example, gold `{"items":[{"code":"A"},{"code":"B"}]}` and prediction
`{"items":[{"code":"B"},{"code":"A"}]}` have zero matching fields under
this versioned index-sensitive rule. The evaluator does not infer an unordered
set or silently select a fuzzy matching policy.

Bootstrap samples documents, not fields, preserving dependence among fields
from the same source. Paired arms use identical sampled document indices.
Schema/language breakdowns are point estimates and spend no extra bootstrap
draw allowance. Undefined ratios are `null`; a zero-row subgroup also has
`document_accuracy: null`. An interval is `null` if a required resampled ratio
is undefined. `--confidence-level` describes statistical uncertainty, not model
confidence, and `--bootstrap-seed` controls only statistical resampling.

The report also records actual native work on all five axes, mask visits and
source verification. Gold source checks across every split are charged once to
the same global verification allowance. Each executed source-bound result then
accounts for the native source checks plus independent evaluator verification.
For an identical-model comparison, the total includes one gold-validation
charge plus both arms' native verification; it is not blindly twice the entire
single-model total. Structural-only cases spend no source-membership scan work.

Private schema bodies, property names, source/gold strings, exact generated
JSON, tokens and all occurrence evidence are bound through domain-separated
HMAC commitments. Public reports retain aggregate quality, declared metadata
and collection formats/counts. Corpus labels remain human assertions; replay
does not authenticate their semantic correctness.

## Apply explicit paired gates

Every gate field is required. Values must be finite and in `[-1,1]`:

```json
{
  "schema_version": 1,
  "max_field_f1_decrease": 0.01,
  "max_field_recall_decrease": 0.01,
  "max_document_accuracy_decrease": 0.01
}
```

```sh
fnlp qualify native-extract-int4 \
  --snapshot candidate.int4 --expected-model-sha256 "$CANDIDATE_MODEL_SHA256" \
  --baseline-snapshot baseline.int4 --baseline-model-sha256 "$BASELINE_MODEL_SHA256" \
  --group-width 128 --baseline-group-width 128 \
  --suite extraction-suite.json --cases extraction-cases.ndjson \
  --gates extraction-gates.json --host-scope declared-local-host \
  --key-file assay.key --key-id local-extraction-key --namespace extraction-v1 \
  --memory-mib "$ASSAY_PAIRED_MEMORY_MIB" \
  --minimum-samples 20 --bootstrap-resamples 200 --bootstrap-seed 0 \
  --output extraction-qualification.json
```

Gates compare **baseline minus candidate**. Positive tolerances allow that much
decline, zero permits no decline and negative values require improvement.
An undefined required metric fails its gate. The gate basis is the point
estimate; paired changes and intervals accompany that decision.

`decision.paired` reports candidate-minus-baseline field precision/recall/F1
and document-accuracy changes. It also counts documents where the candidate's
field FP+FN total is better, worse or equal. Both sealed scorecards and their
complete native bindings remain in the qualification report.

`decision.status` is `passed`, `rejected` or `insufficient_data`. These are all
successful complete evaluations, so callers must inspect the status rather
than equating process success with qualification. Host scope is explicitly
`caller_declared`, receipts are AuditOnly and `activation_authority` is false.

## Bounds and refusal behavior

| Option | Default and scope |
| --- | --- |
| `--max-schemas` | 16; range 1–64, checked before typed catalogue allocation. |
| `--max-schema-bytes` | 262144; range 1–1048576, per exact schema string and additionally bounded by the complete suite cap. |
| `--max-fields-per-row` | 256; range 1–4096 for each gold or generated value. |
| `--max-json-nodes-per-row` | 16384; range 1–65536, checked during exact JSON parsing. |
| `--max-json-depth` | 64; range 1–128, with the native grammar's own supported-depth limit still enforced. |
| `--max-languages` | 64; range 1–128 across the validated corpus. |
| `--max-occurrences-per-row` | 16384; range 1–65536 for complete source occurrence evidence. |
| `--max-verification-steps` | 67108864; range 1–4294967296, shared by gold checks and all native/evaluator source passes across both arms. |
| `--max-new-tokens` | 512 per complete constrained output, including terminal EOS. |
| `--max-result-bytes` | 1048576 for each complete native result. |
| `--max-bootstrap-draws` | 10000000; qualification charges both arms. |
| `--max-output-bytes` | 4194304 for the final complete envelope and newline. |

Memory admission precedes input/model IO. Its modeled peak covers raw source
and gold strings, retained exact JSON trees, temporary catalogue roots and one
active compiler/source preparation, resident snapshots, loading versus native
execution workspace, report statistics, encoded publication and cleanup. Gold
tree capacity uses both the total input-byte bound and declared row/node
geometry, with a separate line/depth bound for the active parser stack even
when malformed input fails before schema validation. The quote never assumes
the unchecked product of every maximum is actual input. This is an explicit
memory-ledger model, not an RSS measurement. Tighten
input, row, schema and JSON limits to the intended corpus when choosing the
explicit memory ceiling.

Native forward positions, projected logits, attention pairs, dot products,
multiply-accumulates, total mask visits, source verification and elapsed time
are nonrenewable across the complete operation. Both resident models are
priced, but their native execution workspace is sequential. Complete bounded
row work is admitted before execution and actual successful work is charged.
A remaining allowance can refuse a row even when an unusually short completion
might have fit. Caps never silently truncate input, gold, native JSON, source
occurrences or reports.

Long documents must fit the declared prompt/context envelope; this evaluator
does not automatically chunk, summarize, repair JSON or retry unconstrained.
Errors are sanitized and preserve typed input, integrity, resource, work,
cancellation and structured-result categories. There is no NER/classification
gold reinterpretation, semantic second-reader policy or calibrated extraction
surface hidden behind this command.

Regression sources cover discovery, refusal before input, exact-tree memory
admission, paired document work, typed extraction/planner failures and complete
output caps. The ignored model fixture covers three schema families, 38-digit
gold, UTF-8 source membership, exact paired work, protected private output,
insufficient data and refusal before missing model paths. The added sources
require the repository's controller-admitted validation; they establish no
executed test result, model-quality benchmark or measured speedup by themselves.
