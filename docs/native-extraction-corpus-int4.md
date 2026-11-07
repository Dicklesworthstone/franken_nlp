# Complete-document INT4 extraction corpora

`fnlp candidate source-int4 extract-corpus` extracts a fixed schema from every
chunk of each complete original document in an ordered NDJSON stream. One
resident model and one hosted invocation serve the entire stream. A document
can exceed one native context; its original text is never silently truncated.

The command is available with `asupersync-runtime`. It needs no metadata store,
job directory, key or retained input/result spool. Source conversion and a
local snapshot with an independently supplied model digest use the same route.

## Fixed schema and complete original inputs

Supply an exact schema in a local file. For example, `schema.json` can allow a
bounded array of source-verified names, including an empty array when no name
appears in a particular chunk:

```json
{
  "type": "array",
  "maxItems": 16,
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

Each line supplies one complete original source:

```jsonl
{"id":"report-1","text":"Alice and Bob submitted the complete report."}
{"id":"report-2","text":"The second complete original document goes here.","task_args":{}}
{"flush":true}
```

`task_args` may be absent, `null` or an empty object. A record cannot replace
the schema, grounding, task budget, partition policy or execution schedule.
Nonempty objects, arrays and scalar arguments are refused. Flush controls
reset only the duplicate-ID epoch after prior output drains.

```sh
fnlp candidate source-int4 extract-corpus documents.ndjson \
  --source-dir /private/pinned-source \
  --source-manifest /private/pinned-source-manifest.json \
  --schema schema.json --source-membership \
  --group-width 128 --memory-mib "$MEMORY_MIB" \
  --max-chunks 64 --cohort-chunks 4 \
  --max-map-result-bytes 16777216
```

Alternatively, replace the source arguments with
`--snapshot /private/model.snapshot --expected-model-sha256 "$MODEL_SHA256"`.
The digest must come from independent provenance. The command does not infer
it from the file being authenticated. An optional source-mode
`--save-snapshot NEW_FILE` runs after fixed schema, model and maximum-envelope
admission; that complete snapshot can remain if a later document fails.

`MEMORY_MIB` is an explicit configured ceiling. These examples do not establish
a measured memory requirement or model qualification. All output remains
candidate evidence labeled `non_authoritative`.

## Result meaning and native scheduling

Each successful document event retains **every** source chunk, its independent
exact-schema JSON string, and its original byte and Unicode-scalar lineage.
Exact numeric literals stay in the native JSON string. The candidate writer
copies the inner event bytes without parsing and rounding those literals.

With `--source-membership`, every compatible occurrence in the originating
chunk is independently checked and lifted to original-document coordinates.
The command does not invent matches in other chunks or merge independent JSON
values into a global object. Required fields must be satisfiable in every
chunk. Choose a schema that can represent an empty local result when needed.
Schema validity and source membership do not establish semantic accuracy.

Documents execute sequentially. Serial chunk execution is the default;
`--cohort-chunks N` shares native decoder layers across up to `N` independent
chunks of the current document, with `1 <= N <= 64` and `N <= --max-chunks`.
Each chunk retains its own grammar, source language, causal KV and task budget.
Every real chunk and the complete requested schedule preflight before the
document's first native forward. A later failed chunk yields no successful
prefix of that document.

## Whole-run limits

The five native flags are fixed ceilings for the **entire stream**:
`--max-forward-positions`, `--max-projected-logits`, `--max-attention-pairs`,
`--max-dot-products` and `--max-multiply-accumulates`. Raising `--max-requests`
does not raise any of them. No chunk, cohort, document or flush renews work.

Mask and independent original-coordinate verification have separate document
and run ceilings:

| Resource | Per-chunk or complete-document flag | Whole-run flag and default |
| --- | --- | --- |
| Grammar-mask visits | `--max-mask-node-visits` per chunk; `--max-total-mask-node-visits` per document | `--max-run-mask-node-visits`: `128000000000000` |
| Verified fields | `--max-extraction-fields` per document | `--max-run-extraction-fields`: `4096000` |
| Original-coordinate occurrence spans | `--max-extraction-evidence-spans` per document | `--max-run-extraction-evidence-spans`: `16384000` |
| Verification scan steps | `--max-extraction-scan-steps` per document | `--max-run-extraction-scan-steps`: `67108864000` |

Each run ceiling must cover one declared document allowance. It need not fund
the maximum possible number of input records. Every admitted attempt reserves
its complete planned native work, masks and declared verification allowance
atomically before execution; failures and small actual outputs do not refund
that reservation. Exhaustion stops execution without changing the limits.

`--max-input-bytes` bounds each complete decoded source, while
`--max-line-bytes` bounds its entire incoming JSON line. The default line cap
is 4 MiB; decoded source defaults to 1 MiB. `--max-result-bytes` bounds one
native chunk result and `--max-map-result-bytes` bounds the complete semantic
document. Candidate and transport framing receive additional reserved space.
`--max-total-input-bytes` and `--max-total-output-bytes` bound the entire
transport, with 64 MiB defaults. The nonempty-record cap defaults to 1,000.

Pure admission prices the same complete owner set used by the host: native
KV/scratch, compiler and schema preparation, all retained map plans, frozen
chunk identities, the complete reduction frontier, original-coordinate
evidence, cohort decoder state, full output staging and owned IO. Schema
declarations are checked before model or population IO, with raw bytes
preserved. Actual source feasibility is checked against each real chunk.

## Completion and embedding

Output uses the existing ordered event protocol inside
`fnlp-candidate-int4-extract-corpus-v1` candidate envelopes. Require both
`run_complete` and exit status `0` for whole-run success. A drained stream can
contain rejected documents and therefore finish with a nonzero exit. Earlier
complete records remain valid if a later document fails; the CLI adds no second
stdout completion or error after the host's terminal frame. An end-to-end
deadline includes setup, native execution and delivery. Blocking IO observes
cancellation cooperatively; an error can follow already-delivered bytes.

For embedding, construct
`batch::extract::grouped_int4::long::stream::Int4ExtractionDocumentBatchPlanner`
with its fixed `Int4ExtractionDocumentBatchConfig`, then call
`NlpEngine::batch_int4_extract_documents` with owned reader/writer, vocabulary,
`CorpusLimits`, reduction reserve, kernel and cancellation token.
`.with_chunk_cohorts()` uses the declared `map_batch_chunks` width.
`int4_extraction_document_corpus_requirements` and the runtime-independent
`Int4ExtractionDocumentMemoryLimits` share the complete reservation formula.

Use [retained complete-document extraction jobs](owned-document-extraction-jobs.md)
when authenticated restart and private result materialization are required.
Use [ordinary extraction batches](candidate-extraction-batch-cli.md) for the
existing single-context document contract.

Regression definitions cover strict fixed arguments and raw schemas, complete
chunk ownership, all nine run counters, memory admission, source/snapshot
parity, native schedule selection, cancellation and exact output framing.
Compilation and test execution remain **UNRUN** in this implementation session
under the controller-owned DSR policy. Native parity, quality, performance and
release qualification retain their separate evidence requirements.
