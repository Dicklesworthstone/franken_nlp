# Complete-document object extraction with semantic review

`fnlp candidate source-int4 extract-semantic-map` extracts complementary root
properties from a complete UTF-8 document, reconciles one object under the
original schema, and reviews each configured scalar claim against the complete
original source. It preserves the original object, partial chunk results and
source evidence. The review adds diagnostics alongside them.

This is an explicit experimental composition. The same grouped-INT4 model
performs extraction and review, so their errors can be correlated. Whole-source
review aggregates are uncalibrated diagnostics, with no automatic acceptance,
field repair or factual-correctness certificate. Execution requires
`asupersync-runtime`; a build without that feature refuses this route before
private input or model IO.

## From source chunks to one reviewed object

The extraction stage uses the existing
[root-object map contract](int4-extraction-object-map.md). A derived schema
removes only the root required set. Each source chunk may return the complete
root-property values it supports; nested objects and arrays remain atomic.
Deterministic reconciliation combines complementary properties and coalesces
exactly equal values while retaining every origin. Conflicting values or a
missing final required property refuse the whole result. The final JSON must
independently satisfy the original schema.

For example, chunk objects `{"amount":7}` and `{"currency":"USD"}` can
reconcile to `{"amount":7,"currency":"USD"}`. This illustrates the reducer;
it is not a reported model output. A second different amount is a conflict,
not a reason to choose the first value or run a repair pass.

After reconciliation, explicit schema-aware rules render claims using
`prefix + canonical JSON scalar + suffix`. The amount remains an exact decimal
value, and a currency string retains its JSON quotation marks. Rules are
validated against the original schema before source or model loading. Claims
are rendered from the completed object, not from individual partial objects.

Every rendered claim receives a complete-source faithfulness map. Its source
is the entire unchanged document, including chunks that contributed no value
to the object. Extraction and review may use different lossless chunk
boundaries. All actual field maps, chunk prompts and evidence heads prepare
and preflight before the first review forward. No selective retrieval,
truncation or omission of failed source windows supplies a successful review.

## A complete bounded example

Save this supported schema as `invoice.schema.json`:

```json
{
  "type": "object",
  "properties": {
    "amount": {"type": "integer"},
    "currency": {"type": "string", "maxLength": 3}
  },
  "required": ["amount", "currency"],
  "additionalProperties": false
}
```

Save a complete document as `invoice.txt`. For example, this separates the
amount from the currency declaration across the small chunks used below:

```text
Invoice amount: 7 whole units.

The delivery note records that the sealed parcel reached the receiving desk.
The recipient checked the description and acknowledged receipt. This note
does not revise the billed amount. The archive copy includes the delivery
instructions and the acknowledgement so that the invoice can be understood
without a separate attachment.

Payment instructions: the currency code for this invoice is USD.
```

Save the complete `Int4SemanticOptions` value below as
`invoice-semantic.json`. The resource ceilings and decision thresholds are
illustrative settings, not measured performance or recommended acceptance
thresholds.

```json
{
  "specification": {
    "schema_version": 1,
    "experimental_opt_in": true,
    "revision": "invoice-document-review-v1",
    "claims": [
      {
        "id": "amount-v1",
        "path": [{"kind": "property", "name": "amount"}],
        "value_kind": "integer",
        "prefix": "The invoice states that its amount in whole units is ",
        "suffix": "."
      },
      {
        "id": "currency-v1",
        "path": [{"kind": "property", "name": "currency"}],
        "value_kind": "string",
        "prefix": "The invoice states that its currency code is ",
        "suffix": "."
      }
    ],
    "judge_policy": {
      "minimum_candidate_weight_ppm": 600000,
      "minimum_margin_milli": 100,
      "evidence_window_bytes": 128,
      "max_evidence_windows": 4,
      "max_evidence_spans": 4
    },
    "judge_budget": {
      "max_input_tokens": 1985,
      "max_output_tokens": 64,
      "max_output_bytes": 65536,
      "max_grammar_states": 4096,
      "max_kv_bytes": 369098752
    }
  },
  "limits": {
    "max_fields": 2,
    "max_claim_bytes": 256,
    "max_total_claim_bytes": 512,
    "max_template_bytes": 1024,
    "max_walk_nodes": 64,
    "max_input_json_bytes": 65536,
    "max_total_prompt_tokens": 32768,
    "max_total_projected_logits": 100000000,
    "max_output_bytes": 1048576
  },
  "max_model_work": {
    "forward_positions": 100000,
    "projected_logits": 100000000,
    "attention_pairs": 1000000000000,
    "projections": {
      "dot_products": 10000000000000,
      "multiply_accumulates": 1000000000000000
    }
  }
}
```

Use an existing digest-pinned candidate snapshot. Set `MODEL_SHA256` from an
independently trusted conversion record and select the matching group width
with `--group-width` if it differs from the default 128.

```sh
fnlp candidate source-int4 extract-semantic-map invoice.txt \
  --snapshot /private/model.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --context-tokens 2048 --memory-mib 32768 \
  --schema ./invoice.schema.json --semantic-options ./invoice-semantic.json \
  --max-input-bytes 1024 --max-chunks 4 --max-chunk-bytes 256 \
  --max-new-tokens 64 --max-result-bytes 65536 \
  --max-map-result-bytes 262144 \
  --max-live-value-bytes 1048576 --max-total-value-bytes 1048576 \
  --max-object-properties 2 --max-object-origins 8 \
  --max-object-input-bytes 262144 --max-object-nodes 1024 \
  --max-object-json-bytes 65536 --max-object-result-bytes 262144 \
  --review-max-chunks 4 --review-max-chunk-bytes 256 \
  --review-max-tokenizer-calls 1024 \
  --review-max-retained-result-bytes 524288 --review-max-result-bytes 262144 \
  --max-mask-node-visits 128000000 --max-total-mask-node-visits 512000000 \
  > invoice-reviewed.json
```

The `369098752`-byte judge KV allowance covers the complete resident
2,048-token cache. A short prompt does not permit a smaller physical-cache
allowance. Both extraction and review use that resident context, while each
real prompt still must fit its task and chunk bounds. Increasing input bytes
or chunk counts does not increase context capacity or the combined work cap.

This example deliberately reduces the field, chunk and output declarations.
The 32,768 MiB process budget is an admission ceiling, not a measured memory
requirement. The implementation quotes retained plans and result owners from
their declared maxima before private schema/source/model IO. Raise declarations
only for the intended workload and within available process memory. A document
that needs more than four chunks is refused rather than truncated.

### Model source, stdin and CPU team options

Replace the snapshot arguments with `--source-dir DIR --source-manifest FILE`
to convert the pinned source model. Optional `--save-snapshot NEW_FILE` uses
the admitted save path after fixed policy, schema and memory checks. Actual
source-dependent extraction and review planning happen later; their failure
can leave a successfully saved snapshot.

Use `-` as the document argument to read one complete UTF-8 document from stdin.
The input is text, not an NDJSON population. Schema and semantic-options files
remain bounded local files.

Add `--compute-threads 4` to the same command to use one persistent numerical
team across extraction and every review. Omitted width or width 1 executes
serially; widths 2 through 64 reserve the extra team workspace. Chunks and
review heads remain serial tasks. The team partitions numerical output rows,
and every worker physically joins before a completed result is returned.
Physical compute statistics appear in the optional `compute` receipt.

## Reading the result

The CLI emits one JSON envelope with protocol
`fnlp-candidate-int4-semantic-object-document-v1`, candidate model provenance
and the explicit uncalibrated, same-model scope. The completed composition is
under `output`:

| Result path | Meaning |
| --- | --- |
| `output.extraction.object.json` | Exact canonical JSON from original-schema reconciliation, retained as a string |
| `output.extraction.object` | Complete property-origin and original-source evidence envelope |
| `output.extraction.partial_map` | Unchanged native chunk outputs, source geometry and extraction accounting |
| `output.semantic.fields` | Every rendered scalar or explicit `not_checked` field |
| `output.semantic.fields[].review` | Complete original-source faithfulness map, when a claim was attempted |
| `output.semantic.fields[].review.aggregate` | Whole-source aggregate, coverage, abstention and retained evidence/counterevidence |
| `output.input_encoding` | Compact policy and cumulative actual tokenizer work; omitted for byte fallback |

A field with `status: "reviewed"` has a non-abstaining aggregate relation:
`entailed`, `contradicted` or `unsupported`. The relation remains inside the
complete map; `reviewed` does not mean correct or accepted. In particular,
`checked_fields` counts non-abstaining diagnostics, including contradiction
and unsupported outcomes.

Conflicting evidence, insufficient classified coverage or unresolved evidence
can leave the aggregate relation absent. The field then has
`status: "not_checked"` and `not_checked: "judge_abstained"`, with its complete
review and counterevidence retained. The default
`--review-minimum-classified-coverage-ppm 1000000` requires complete classified
coverage under the aggregation policy. Lowering that threshold changes only
the declared aggregate policy; all source chunks must still execute.

Absent optional properties produce no invented claim. Null values, empty
containers and scalars without a claim rule retain their respective
`not_checked` reasons. Verbatim fields remain `verbatim_membership_only`; a
claim rule targeting such a field is refused. Add `--source-membership` only
with an appropriately annotated schema. Its exact matching, all occurrence
coordinates and remaining verification allowance are preserved through
reconciliation and review.

The receipt explicitly sets `single_context_equivalence: false`. Combining
independent chunk judgments is not a judgment made with the entire document
in one native context. Evidence split across chunk boundaries can remain
unresolved, and structural validity, source membership and reconciliation
cannot establish extraction recall or semantic correctness.

## One complete allowance across both stages

The options file's `max_model_work` covers all five native axes: forward
positions, projected logits, attention pairs, grouped dot products and
multiply-accumulates. The full planned extraction reservation is added to the
sum of every actual field map's planned work before any review executes.
Early extraction EOS does not return budget for extra review. The extraction
map's separate `--max-forward-positions`, `--max-projected-logits`,
`--max-attention-pairs`, `--max-dot-products` and
`--max-multiply-accumulates` remain additional extraction-stage ceilings.

`limits.max_total_prompt_tokens` and `limits.max_total_projected_logits`
cover all reviewer fields, chunks and evidence heads together. They do not
restart for each map. The extraction mask allowance is likewise shared across
all extraction chunks through `--max-total-mask-node-visits`; review does not
create another constrained-extraction mask allowance.

Output limits are separate and cumulative ownership is admitted explicitly:

| Setting | Bound and route default |
| --- | --- |
| `--max-result-bytes` | One complete native chunk extraction; 1 MiB |
| `--max-map-result-bytes` | Original partial map plus reconciled object; 4 MiB |
| `--max-object-json-bytes` | Final object JSON; 1 MiB |
| `--max-object-result-bytes` | Object and its origins; defaults to the map-result cap |
| `--review-max-retained-result-bytes` | Retained chunk evidence/results in one field map; 16 MiB |
| `--review-max-result-bytes` | One complete field review; 4 MiB |
| `limits.max_output_bytes` | Entire extraction-plus-review result; explicitly supplied in the options file |

The original-map and per-review result caps must each fit the combined cap.
The object JSON cap must fit `limits.max_input_json_bytes`, and the judge's
per-task output cap must fit the review retained-result allowance. These fixed
inconsistencies are refused before source or model IO. The example overrides
the broader defaults so that two complete field reviews fit a modest combined
output declaration.

Memory admission retains the source and configuration, original and partial
schemas, all extraction chunk programs, reconciliation storage, rendered
claims, every complete field map and intermediate result, final output and
serialization staging. Only the immutable judge compiler, one native cache
and transient encoder workspace are shared. Declaring more fields or review
chunks can substantially increase the quote even if a particular document
would use fewer. It is a modeled payload bound, not an RSS measurement.

### Compact encoding shares the spent extraction ledger

Byte fallback is the default. To explicitly select bounded compact input, add
these flags to the complete command above:

```text
--input-encoding safe-bpe-v1
--max-tokenizer-input-bytes 1048576
--max-tokenizer-output-tokens 1048576
--max-tokenizer-pair-checks 67108864
--max-tokenizer-pair-bytes 536870912
```

One ledger covers extraction's derived-schema/source encoding, accepted and
rejected chunk-sizing probes, then every claim, reviewer probe and evidence
head. Review starts with the work extraction has already spent. No field or
chunk receives a fresh input-byte, output-token, pair-check or examined-pair-byte
allowance. Completed accounting equals the original extraction's work plus
every sealed review delta. These are cumulative tokenizer-work limits, separate
from the original document byte cap and native context size.

The same immutable policy is checked across both stages, with one transient
encoder workspace admitted before private source/model IO. Exact original
bytes and source coordinates remain authoritative, and template-control token
IDs remain excluded from untrusted encoding. Compact encoding changes prompt
tokens; identical model output, quality, parity and speed are not established.

## Resident whole-document populations

`fnlp candidate source-int4 extract-semantic-document-corpus` accepts ordered
NDJSON records whose `text` is a complete original document. One resident
model, immutable original-schema factory, judge, extraction vocabulary and
optional numerical team serve the entire stream. The input is bounded and
streamed; corpus size does not multiply resident model or document-plan memory.

For example, save these records as `invoices.ndjson` and use the schema and
semantic-options files above:

```json
{"id":"invoice-001","text":"Invoice amount: 7 whole units. Payment currency: USD."}
{"id":"invoice-002","text":"Invoice amount: 9 whole units. Payment currency: EUR."}
```

The following declarations fund two complete attempts. These are illustrative
ceilings, not measured costs or a promise of successful model output.

```sh
fnlp candidate source-int4 extract-semantic-document-corpus invoices.ndjson \
  --snapshot /private/model.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --context-tokens 2048 --memory-mib 32768 \
  --schema ./invoice.schema.json --semantic-options ./invoice-semantic.json \
  --max-input-bytes 1024 --max-chunks 4 --max-chunk-bytes 256 \
  --max-new-tokens 64 --max-result-bytes 65536 \
  --max-map-result-bytes 262144 \
  --max-live-value-bytes 1048576 --max-total-value-bytes 1048576 \
  --max-object-properties 2 --max-object-origins 8 \
  --max-object-input-bytes 262144 --max-object-nodes 1024 \
  --max-object-json-bytes 65536 --max-object-result-bytes 262144 \
  --review-max-chunks 4 --review-max-chunk-bytes 256 \
  --review-max-tokenizer-calls 1024 \
  --review-max-retained-result-bytes 524288 --review-max-result-bytes 262144 \
  --max-mask-node-visits 128000000 --max-total-mask-node-visits 512000000 \
  --max-requests 2 \
  --max-run-forward-positions 200000 --max-run-projected-logits 200000000 \
  --max-run-attention-pairs 2000000000000 \
  --max-run-dot-products 20000000000000 \
  --max-run-multiply-accumulates 2000000000000000 \
  --max-run-mask-node-visits 1024000000 \
  --max-run-extraction-fields 8192 --max-run-extraction-evidence-spans 32768 \
  --max-run-extraction-scan-steps 134217728 \
  > invoices-reviewed.ndjson
```

The original map flags retain their per-document scope. The options file's
`max_model_work` covers one complete extraction-plus-review attempt; the five
`--max-run-*` native limits cover the entire stream. The mask and three
source-verification run limits are also separate from their per-document
declarations. Omitted run limits fund **one** complete document; increasing
`--max-requests` never grants more inference authority. A run must have enough
of every axis for each attempted document, including a document without
source-membership annotations, because admission reserves the fixed declared
verification ceiling conservatively.

Records accept `id` and `text`; non-null `task_args` are refused. Exact
`{"flush":true}` controls can close an output epoch, but consume a request slot
and do not renew any work allowance. The CLI also bounds line bytes, total
input bytes, total output bytes and record count. Each successful record
contains the complete reconciled extraction and all field reviews. An
extraction, reconciliation, review or completion-authentication failure cannot
publish a successful partial combined record. Earlier completed records can
remain available if a later record fails.

Require a complete `run_complete` frame **and exit status zero** for whole-run
success. Candidate frames identify the
`fnlp-candidate-int4-semantic-object-corpus-v1` protocol and retain the explicit
same-model, uncalibrated diagnostic scope. Terminal completion follows physical
worker join. Add `--compute-threads 4` to use one numerical team for the whole
stream; `-` selects stdin, and the source-conversion/snapshot options above
retain their meanings.

Owned-source records freeze their original complete-request proof and rebuild
borrowed plans at execution. Both real preparation passes count toward the
same per-document compact-tokenizer allowance. The known reconstruction cost
must fit before it is repeated; the second pass cannot erase the first pass
or obtain a fresh review allowance. Completed compact accounting includes both
preparations and every sealed review delta.

Library callers use
`Int4SemanticObjectDocumentCorpusConfig`,
`int4_semantic_object_document_corpus_requirements` and
`NlpEngine::batch_int4_semantic_object_documents[_with_threads]`.
The quote includes the existing simultaneous whole-document peak, retained
fixed-schema/policy copies, bounded transport and IO staging, and optional
team workspace before private schema/population/model IO. Caller-owned
configuration and weights remain additional, as on the one-shot route.

## Failure, cancellation and current scope

Invalid schema/rules, reconciliation conflicts, missing required properties,
failed review preparation, exhausted work or output limits, and execution
failure produce no successful partial combined result. Completion authenticates
the original object authority, every actual map's source/claim binding and all
additive work before publication. Replacing a valid review with a map from a
different same-length source cannot reuse its completion authority.

One elapsed CLI deadline spans preparation, model load, optional save,
extraction, reconciliation, review and checked publication. Hosted work also
shares one checkpoint allowance. Library errors preserve typed cancellation
causes; CLI failures map to stable error categories without source or claim
content. Source and schema compilation observe cancellation between bounded
calls, not inside every compiler state. Blocking local IO is also cooperative.

The full input and admission owners survive queued cancellation, panic, team
formation failure and physical join. The output guard remains live through
serialization and flush. A write error or late deadline can leave incomplete
or already flushed bytes, so consumers must require a complete JSON envelope
and exit status zero. Publication does not retry as a new inference attempt.

This initial route uses full-vocabulary constrained extraction and
full-vocabulary reviewer scoring, serial chunks and optional numerical CPU
teams. Selected-row projection, chunk cohorts, packed prefill, sampling,
thinking and tools are refused. The resident corpus route above explicitly
selects the complete-document composition.
`extract-semantic-document-job start|resume` adds authenticated retained
populations, lifetime work across retries, complete result storage and optional
retained-input replay; see the
[whole-document job guide](int4-semantic-object-jobs.md) for complete commands.
The existing `extract-semantic-batch` and `extract-semantic-job` have their own
single-context contracts; they do not implicitly select this document map.

Library users can pre-admit with
`int4_semantic_object_document_requirements` and call
`NlpEngine::extract_int4_semantic_object_document` or
`NlpEngine::extract_int4_semantic_object_document_with_threads`. The lower-level
`PreparedInt4SemanticObjectDocument` consumes a real prepared root-object map
and a real model-bound judge; no public execution API accepts substituted
native outputs, source text or schema authority.

The original one-shot implementation received source/API review, Rust syntax
parsing and whitespace checks. The resident-corpus and retained-job increments received
independent source/API review and static delimiter, string, comment,
conflict-marker and whitespace checks; its execution environment was
unavailable. New regression definitions cover exact original-schema/source
authority, all nine run work axes, compact reconstruction accounting,
nonrenewal at flush, typed cancellation, memory admission and CLI dispatch.
Existing private scripted-completion cases check full-source diagnostics and
tampering without claiming native success. Compilation, executable tests,
controller-only DSR runs, native parity, model quality and incremental
error-catching value remain **UNRUN**. The examples above were not executed.
