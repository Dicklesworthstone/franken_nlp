# Retained whole-document object extraction and review

`fnlp candidate source-int4 extract-semantic-document-job start|resume`
persists each complete-document extraction and semantic review as one durable
item. It uses the [whole-document object contract](int4-semantic-object-map.md):
losslessly map every source chunk, reconcile complementary root properties
against the original schema, and review each configured scalar claim against
the complete original source. The unchanged object, partial chunk outputs,
source evidence and every field review are stored together.

A failed extraction, reconciliation or review commits no successful partial
combined result. Earlier committed documents survive and are skipped on
resume. Review remains an explicitly requested, uncalibrated diagnostic from
the same model; it does not repair fields, certify facts or authorize acceptance.

Execution requires `metadata-store` and `asupersync-runtime` on Linux x86-64
or AArch64. Other profiles expose the command and refuse before private IO.
This is a code-first candidate implementation: compilation, executable tests,
authorized DSR and independently pinned native-model qualification remain
**UNRUN**. None of the examples below has been executed.

## Original inputs and protected storage

Use globally unique IDs and complete original document text:

```jsonl
{"id":"invoice-001","text":"Invoice amount: 7 whole units. Payment currency: USD."}
{"id":"invoice-002","text":"Invoice amount: 9 whole units. Payment currency: EUR."}
```

The original schema, grounding mode, review rules, chunk geometry, encoding and
resource declarations are fixed for the population. Non-null `task_args` and
stream flush controls refuse. The immutable item byte cap includes JSON
escaping, the ID and framing; the complete decoded source also has its declared
document cap. Larger inputs are chunked within those bounds, never truncated.

Provision an existing owner-only job directory, a protected file containing
exactly 32 raw secret bytes, and a random 128-bit job ID using the
[existing storage setup](candidate-owned-jobs-cli.md#explicit-retention-and-setup).
The command does not create keys or directories or silently adopt an existing
job. Keep the original secret, ID and limits for every resume.
Authentication does not encrypt the result spool or optional input spool.

`--store-results` is required. Stdout reports bounded completion metadata;
private documents and reviewed results remain in protected storage.
`--materialize` optionally publishes verified ordered `materialized.ndjson`
after every item commits, under the existing no-replace publication contract.
The [model-free management commands](job-management-cli.md) can inspect,
verify and materialize stored progress separately.

## Whole-attempt and lifetime limits

The semantic options file's `max_model_work` funds one complete extraction
and all resulting reviews. Extraction's `--max-total-mask-node-visits`
funds all chunks in that attempt. Before execution, the durable journal charges
the complete declared five-axis native work and mask work. Early EOS, a failed
review, cancellation and retry do not refund those reservations.

The `--limits` file supplies separate immutable lifetime work and attempt
caps. Increasing item count, transport ceilings or CPU width does not multiply
native authority. Budget retry headroom before starting; resume cannot widen
the contract.

For the two-field invoice options in the
[complete-document guide](int4-semantic-object-map.md#a-complete-bounded-example),
save this as `invoice-document-job-limits.json`. It funds two items and up to
three complete attempts at the explicitly selected 512,000,000-mask-visit
per-document ceiling:

```json
{
  "max_items": 2,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 1024,
  "max_snapshot_bytes": 1048576,
  "max_result_bytes": 1048576,
  "max_spool_bytes": 16777216,
  "max_materialized_bytes": 16777216,
  "max_journal_bytes": 67108864,
  "max_attempts": 3,
  "max_work": {
    "model": {
      "forward_positions": 300000,
      "projected_logits": 300000000,
      "attention_pairs": 3000000000000,
      "projections": {
        "dot_products": 30000000000000,
        "multiply_accumulates": 3000000000000000
      }
    },
    "mask_node_visits": 1536000000
  }
}
```

These are finite illustrative ceilings, not measured costs, quality thresholds
or expected performance. They correspond to three times the example's
per-attempt combined work on every axis.

Grounding has three independent ceilings: verified fields, occurrence spans
and scan steps. These are not additional journal work fields. Instead the
fixed recipe funds each full per-document declaration for **every permitted
lifetime attempt**, using checked multiplication by `max_attempts`. Overflow
refuses before private IO. This conservative certificate covers interrupted
attempts and retries without adding a renewed allowance on resume.

`--max-result-bytes` still bounds one extraction chunk. The map, reconciled
object and individual review limits keep their existing scopes.
`semantic_options.limits.max_output_bytes` bounds the complete combined
result and must fit `JobLimits.max_result_bytes`. Spool, materialization,
journal and serializer limits are separately admitted.

## Complete start and resume example

Use `invoice.schema.json` and `invoice-semantic.json` from the
[complete-document example](int4-semantic-object-map.md#a-complete-bounded-example).
Set `MODEL_SHA256` from an independently trusted snapshot record and
`JOB_ID` to the originally provisioned 32-character lowercase hexadecimal ID.

These Bash arrays keep all immutable declarations identical between start and
resume:

```bash
document_options=(
  --snapshot /private/model.snapshot --expected-model-sha256 "$MODEL_SHA256"
  --context-tokens 2048 --memory-mib 65536 --compute-threads 4
  --schema ./invoice.schema.json --semantic-options ./invoice-semantic.json
  --max-input-bytes 1024 --max-chunks 4 --max-chunk-bytes 256
  --max-new-tokens 64 --max-result-bytes 65536
  --max-map-result-bytes 262144
  --max-live-value-bytes 1048576 --max-total-value-bytes 1048576
  --max-object-properties 2 --max-object-origins 8
  --max-object-input-bytes 262144 --max-object-nodes 1024
  --max-object-json-bytes 65536 --max-object-result-bytes 262144
  --review-max-chunks 4 --review-max-chunk-bytes 256
  --review-max-tokenizer-calls 1024
  --review-max-retained-result-bytes 524288 --review-max-result-bytes 262144
  --max-mask-node-visits 128000000 --max-total-mask-node-visits 512000000
)
storage_options=(
  --store-results --job-dir /private/invoice-document-job --job-id "$JOB_ID"
  --key-file /private/invoice-document-job.key
  --limits ./invoice-document-job-limits.json
)

fnlp candidate source-int4 extract-semantic-document-job start invoices.ndjson \
  "${document_options[@]}" "${storage_options[@]}"

fnlp candidate source-int4 extract-semantic-document-job resume invoices.ndjson \
  "${document_options[@]}" "${storage_options[@]}" --materialize
```

The memory setting is an admission ceiling, not a measured requirement.
Both the complete original population and the fresh immutable recipe are
authenticated before pending items execute. Without input retention, supply
every original record again, including already committed items, in its original
order and with its exact original envelope bytes.

Use `-` for original-population stdin. Source conversion is also available:
replace the snapshot flags with `--source-dir DIR --source-manifest FILE`.
Optional `--save-snapshot NEW_FILE` follows fixed declaration/schema
admission. Record-dependent preparation happens later, so a saved snapshot
can remain after a later job failure.

### Explicit retained-input replay

To retain exact private input envelopes, add
`--spool-input --max-retained-input-mib 2` to the start command.
On resume, omit `invoices.ndjson` and use
`--use-retained-input --max-retained-input-mib 2` with the same other settings.

Creation stages and validates bounded envelopes, then authenticates and seals
the retained population before inference. Replay authenticates that complete
population under the job lock and reads pending records through a reusable
bounded buffer. Missing, corrupt or incomplete retained input refuses; it
does not silently fall back to another source. The retained disk cap includes
framing and is separate from work and result storage.

## Exact resume contract

The private keyed recipe binds:

- Both real model-bound extraction and judge identities, group width and
  execution/version profiles.
- Exact original schema bytes and original-schema derivation, grounding and
  root-object reconciliation settings.
- All extraction and review chunk geometry, compiler/source/planning limits,
  claim paths/templates, judge policy and semantic result limits.
- The immutable compact-encoding policy, full combined attempt declaration,
  lifetime native/mask authority and all-attempt grounding certificate.

Schema whitespace changes the contract. Semantic-options settings bind their
typed values; incidental JSON formatting does not change them. Records cannot
override any policy. A prepared record also carries a private factory seal and
complete original-source/schema proof, checked before durable charging and
again during exact reconstruction.

Default resume refuses an uncommitted result tail or staging artifact.
Only `--discard-uncommitted-tail` explicitly permits repair, after
authentication; it never refunds work or changes committed results.
The flag is unavailable on start. There is no automatic retry.

Physical CPU width may change between invocations: omitted width or
`--compute-threads 1` is serial; widths 2 through 64 use one persistent team
across the complete durable invocation. Width changes neither the semantic
recipe nor lifetime work. Whole-document extraction and review remain
full-vocabulary; selected rows, cohorts, packed prefill, sampling, thinking and
tools are unavailable on this route.

Compact `safe-bpe-v1` uses one per-attempt tokenizer allowance across initial
preparation, exact owned-source reconstruction, extraction sizing probes and
all review heads. Both real preparation passes are charged; none of those
stages resets the ledger. Repeat the same policy and caps on resume. Compact
encoding changes prompt tokens and does not promise identical model output.

## Ownership, cancellation and completion

Before key, schema, population or model IO, the CLI checks the complete peak
after bounded fixed options and limits reads. Admission includes the resident
document plan/result peak; original configuration, recipe and immutable factory
copies; canonical recipe buffers; population or retained-record staging;
journal, serializer and IO owners; native KV/scratch; complete combined output;
and the optional numerical team. Caller preparation, weights, model load/save
and cleanup peaks are also included. The bounded schema reader's sentinel
capacity is priced before IO. This is modeled payload admission, not an RSS
measurement.

Actual inputs, compiler/vocabulary assets and reservations remain owned through
queued cancellation, panic, worker-formation failure and physical join.
The combined output's guard remains live through serialization, spool
synchronization and durable acknowledgement. All original typed cancellation
causes are preserved.

Stdout uses `fnlp-candidate-int4-semantic-object-document-job-v1` and reports
candidate provenance, diagnostic scope, progress and optional compute,
encoding and snapshot metadata. Publication checks the same deadline and
cancellation before and after staging and flush. A nonzero exit may follow
durable commits or flushed metadata: authenticate stored status or explicitly
resume, and do not assume rollback.

Library callers construct `Int4SemanticObjectDocumentJobConfig`,
`Int4SemanticObjectDocumentJobHostLimits` and `SourceJobRequest`, pre-admit
with `int4_semantic_object_job_requirements`, then call
`NlpEngine::job_int4_semantic_object_documents` or its
`_with_threads` variant. The immutable planner and durable processor live in
`jobs::runner::extract::grouped_int4::semantic::document`.

Source/API reviews and static hygiene checks cover these implementations and
their regression definitions. Tests cover recipe mutation, all persisted
work axes, checked retry grounding, compact identity, host/config mismatch,
memory/owner lifetimes, refusal before private reads, cancellation, bounded
metadata and explicit retained-input options. The physical lifecycle case is
an ignored controller-only definition. No compilation, executable test,
native numerical parity, model quality or performance claim is made.
