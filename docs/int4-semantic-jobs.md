# Durable INT4 extraction with complete field review

`fnlp candidate source-int4 extract-semantic-job start|resume` retains the
existing native semantic extraction result as one durable item: the original
independently validated schema extraction together with every configured field
review. If extraction or any required review fails, that attempt commits no
combined result. Already committed items survive and are skipped on resume.

This closes the persistence gap between `extract-semantic` and
`extract-semantic-batch`. The same model performs extraction and review; the
review remains an explicitly requested, uncalibrated, correlated diagnostic.
It does not repair fields, certify factual correctness or authorize acceptance.
Null, unmapped, verbatim, empty and abstained cases keep the existing
`not_checked` behavior. See the
[semantic extraction contract and example policy](../README.md#experimental-candidate-int4-extraction-second-reader).

Execution requires `metadata-store` and `asupersync-runtime` on Linux x86-64 or
AArch64. Other profiles expose the command declaration and refuse execution
before private IO. The implementation and regression definitions are present;
compilation, executable tests, DSR and independently pinned model qualification
remain **UNRUN**. These commands are candidate functionality.

## Inputs and protected storage

Each input is one complete NDJSON record with a globally unique ID:

```jsonl
{"id":"invoice-1","text":"The invoice reports an amount of 7."}
{"id":"invoice-2","text":"The invoice reports an amount of 9.","task_args":null}
```

The schema, grounding mode, claim rules, judge configuration and budgets are
shared by the whole job. Non-null `task_args` and batch flush controls are
refused. The outer record byte cap includes JSON escaping, IDs and framing.
The original document and each complete review prompt must also fit the native
context; a larger record ceiling does not permit truncation or selective review.

Provision an existing owner-only job directory, a protected file containing
exactly 32 raw secret bytes, and a random 128-bit job ID. Follow the
[existing storage setup](candidate-owned-jobs-cli.md#explicit-retention-and-setup).
The CLI does not create keys or directories or silently adopt an existing job.
Keep the original secret, ID and limits for every resume. Authentication does
not encrypt the protected result spool or optional input spool.

`--store-results` is required on start and resume. Stdout contains completion
metadata, not private extraction or claim text. `--materialize` optionally
publishes verified, ordered `materialized.ndjson` only after all items commit;
it uses the existing no-replace publication contract. The
[model-free job management commands](job-management-cli.md) can inspect,
verify or materialize stored progress separately.

## One complete attempt and one lifetime allowance

The semantic options file has a five-axis `max_model_work` allowance for one
complete extraction-plus-review attempt. The extraction's
`--max-mask-node-visits` is the per-attempt mask allowance. Before executing an
item, the runner durably spends those full declared allowances, even if the
model later stops early or a field review fails. Work is never refunded.

The separate `--limits` file supplies the immutable job lifetime allowance.
All five native work axes, mask work and attempts cover every item and every
invocation. Increasing `max_items`, `max_attempts`, transport limits or physical
thread count does not multiply native work. Plan retry headroom before starting;
an exhausted lifetime contract cannot be widened on resume.

For the illustrative semantic options in the README, the following limits
allow up to ten items and twelve complete attempts when the command explicitly
sets `--max-mask-node-visits 2000000000`. These are finite resource ceilings,
not model-quality or performance promises:

```json
{
  "max_items": 10,
  "max_id_bytes": 128,
  "max_input_bytes_per_item": 65536,
  "max_snapshot_bytes": 1048576,
  "max_result_bytes": 4194304,
  "max_spool_bytes": 67108864,
  "max_materialized_bytes": 67108864,
  "max_journal_bytes": 67108864,
  "max_attempts": 12,
  "max_work": {
    "model": {
      "forward_positions": 1200000,
      "projected_logits": 1200000000,
      "attention_pairs": 12000000000000,
      "projections": {
        "dot_products": 120000000000000,
        "multiply_accumulates": 12000000000000000
      }
    },
    "mask_node_visits": 24000000000
  }
}
```

The original extraction cap and combined result cap are distinct.
`--max-result-bytes` bounds the extraction stage. The semantic options'
`limits.max_output_bytes` bounds extraction plus all review diagnostics and
must fit `JobLimits.max_result_bytes`. The combined cap is admitted separately
from serializer staging and retained spool capacity.

## Start and resume

Use the README's `amount` schema and matching explicit semantic options file.
Set `JOB_ID` to the originally provisioned 32-character lowercase hexadecimal
ID, and `MODEL_SHA256` to the independently obtained snapshot pin.

```sh
fnlp candidate source-int4 extract-semantic-job start documents.ndjson \
  --snapshot /private/model.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --context-tokens 2048 --memory-mib 32768 --compute-threads 4 \
  --schema ./amount.schema.json --semantic-options ./amount-semantic.json \
  --max-input-bytes 65536 --max-mask-node-visits 2000000000 \
  --store-results --job-dir /private/semantic-job --job-id "$JOB_ID" \
  --key-file /private/semantic-job.key --limits ./semantic-job-limits.json
```

The complete original population is authenticated before inference. Without
explicit input retention, supply that complete original population again on
every resume, including committed records in their original order and with
their original envelope bytes:

```sh
fnlp candidate source-int4 extract-semantic-job resume documents.ndjson \
  --snapshot /private/model.snapshot --expected-model-sha256 "$MODEL_SHA256" \
  --context-tokens 2048 --memory-mib 32768 --compute-threads 4 \
  --schema ./amount.schema.json --semantic-options ./amount-semantic.json \
  --max-input-bytes 65536 --max-mask-node-visits 2000000000 \
  --store-results --job-dir /private/semantic-job --job-id "$JOB_ID" \
  --key-file /private/semantic-job.key --limits ./semantic-job-limits.json \
  --materialize
```

Source conversion is also supported: replace the snapshot arguments with
`--source-dir DIR --source-manifest FILE`; optional `--save-snapshot NEW_FILE`
uses the existing admitted save path. Fixed schema and review policy are
validated first. Per-record preparation and dynamically generated field reviews
occur later, so a later job failure may follow a successfully saved snapshot.

### Explicitly retained input

To store the exact original private envelopes as well as results, add
`--spool-input --max-retained-input-mib 2` to the start example. The explicit
disk cap includes framing and is separate from population and work ceilings.
On resume, omit the population positional argument and use
`--use-retained-input --max-retained-input-mib 2` with the same other settings.

Creation stages bounded envelopes, validates the complete typed population,
then authenticates and seals its spool before inference. Retained replay
authenticates the complete population and fresh recipe under the job lock,
then reads pending records through one reusable record buffer. Missing,
corrupt or incomplete retained input is an error; it never silently falls back
to the original file. An interrupted creation can leave private uncommitted
input bytes. See the input-retention help for the existing recovery contract.

## What resume authenticates

The recipe binds both real extraction and judge compiler identities, the model
and group width, execution versions, exact schema text, grounding mode,
extraction budget, compiler and source limits, claim mapping and templates,
judge policy and planning limits, semantic limits, input encoding policy, and
all native and mask work ceilings. Changing schema whitespace changes the
recipe. Semantic options bind their parsed typed settings, not incidental JSON
file formatting. A record cannot replace any of these settings.

Default resume refuses an uncommitted result tail or staging artifact. Only
`--discard-uncommitted-tail` explicitly permits repair, and authentication of
the original contract and authorized frames happens first. The flag is absent
from start. Repair never refunds a spent attempt or changes committed results.

Physical numerical thread width may change between admitted invocations:
omitted `--compute-threads` or `--compute-threads 1` uses serial execution;
`2..64` uses one persistent team across extraction and all reviews. The thread
width does not change the semantic recipe or renew work. Prefill, selected-row
projection, cohorts and automatic retries are not exposed on this command.

`--input-encoding safe-bpe-v1` uses one bounded tokenizer ledger per complete
document attempt, shared by extraction schema/source encoding and every review
prompt. Repeat the same encoding policy and limits on resume. The allowance
does not restart between fields. This route has no separate aggregate
`--max-total-tokenizer-*` flags; the immutable per-document policy remains
distinct from the six persisted native/mask work axes. Byte fallback is the
default, and compact encoding does not promise identical model output.

## Memory, cancellation and completion

The declaration-only hosted quote accounts for full semantic preparation,
all simultaneously retained review plans, the selected population or retained
record owner, strict parsing and item index, IO, journal and serializer
staging, complete native KV and scratch, combined result ownership, and optional
team workspace. The CLI adds resident weights, its own live preparation,
source/snapshot load or save peaks, allocator allowance and cleanup reserve.
After bounded options and limits reads, this complete peak is checked before
key, schema, population or model IO. A quote is modeled payload admission, not
a measured RSS or throughput estimate.

The host retains actual inputs and memory claims through queued cancellation,
worker formation failure, panic and physical join. Combined outputs retain
their admission guard through serialization, spool synchronization and durable
journal acknowledgement. Cancellation preserves its original typed cause.

The final stdout protocol is `fnlp-candidate-int4-semantic-job-v1`. Its metadata
includes candidate provenance, the explicit same-model diagnostic scope, job
progress and optional compute, compact-policy and snapshot fields. Private
results remain in protected storage. Publication checks cancellation and the
invocation deadline before and after staging and flushing. A nonzero exit can
follow durable progress or even flushed completion bytes; authenticate stored
status before resuming, and require successful command completion when consuming
the report. Failure does not imply rollback.

Library callers construct `Int4SemanticExtractionJobConfig` and use
`int4_semantic_job_requirements` for admission, then call
`NlpEngine::job_int4_semantic_extract` or
`NlpEngine::job_int4_semantic_extract_with_threads` with `SourceJobRequest` and
`JobHostLimits`. The lower-level model-bound planner and durable processor are
in `jobs::runner::extract::grouped_int4::semantic`.
