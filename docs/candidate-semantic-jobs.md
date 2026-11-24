# Retained INT8 extraction with complete field review

`fnlp candidate extract-semantic-job start|resume` processes a fixed-schema
NDJSON population and stores each complete extraction-and-review result as one
durable item. A failed extraction or required review commits no partial
result. Previously committed items remain available and are skipped on resume.

The same model performs both stages. Its review remains explicitly requested,
correlated, experimental and uncalibrated. Original extraction JSON is
unchanged; fields retain the existing `entailed`, `contradicted`,
`unsupported` and `not_checked` states. Review does not repair fields or
decide acceptance. See the [semantic extraction contract](candidate-semantic-extraction.md)
for the exact claim, evidence and diagnostic rules.

Execution requires `metadata-store` and `asupersync-runtime` on Linux x86-64
or AArch64. Other profiles refuse execution before private IO. The code and
regression definitions are present; compilation, executable tests,
independently pinned model equivalence, interruption/resume qualification and
performance remain controller-owned validation gates.

## Fixed inputs and explicit storage

Each record carries one unique ID and its complete original source:

```jsonl
{"id":"report-1","text":"Acme reported revenue of 1250000 US dollars for 2025."}
{"id":"report-2","text":"Acme reported revenue of 1500000 US dollars for 2025.","task_args":null}
```

The exact schema, source-grounding mode, claim rules, judge policy and task
budgets belong to the whole job. Non-null `task_args` and batch flush markers
are refused. The immutable item-byte ceiling includes the original JSON
envelope, escaping and ID. It must fit `--max-input-bytes`; the original
document and every actual review must also fit their context and evidence
limits. A larger envelope cap does not authorize truncated review.

Provision an existing owner-only directory, a protected file containing
exactly 32 raw secret bytes, and a random 128-bit job ID as described in the
[owned-job storage setup](candidate-owned-jobs-cli.md#explicit-retention-and-setup).
Keep the original ID, secret and limits for resume. The command does not
create keys or directories. Protected authenticated spools are not encrypted.

`--store-results` explicitly authorizes storing private extraction and review
content. Stdout contains completion metadata only. `--materialize` optionally
publishes verified ordered `materialized.ndjson` after every item commits,
using the existing no-replace publication protocol. Existing
[job management commands](job-management-cli.md) can inspect, verify and
materialize stored results separately.

## One complete attempt and immutable lifetime work

The [semantic options example](examples/int8-semantic/options.json) declares
the five native work ceilings for **one complete extraction-and-review
attempt**. `--max-mask-node-visits` declares its extraction-mask allowance.

The [job limits example](examples/int8-semantic/job-limits.json) admits two
items and four attempts, with all five native ceilings equal to four complete
attempts. It reserves eight billion mask visits when the command explicitly
selects two billion per attempt. These are finite admission examples, with no
asserted model result, quality score or measured resource usage.

Before the first native forward for an item, the existing durable runner
spends its full declared model and mask allowance. Early EOS, failed review,
cancellation and subsequent resume do not refund that debit. The
`JobLimits.max_work` object is the sole authority for all six lifetime axes;
there are no duplicate batch work flags. `max_items` and `max_attempts` do
not multiply work. Include intended retry headroom in the original limits.

The original extraction cap and combined result cap remain separate.
`--max-result-bytes` limits extraction. The semantic options'
`limits.max_output_bytes` limits the full combined result and must fit
`JobLimits.max_result_bytes`. The declared extraction result must also fit
the semantic walker's input-JSON allowance.

## Start and resume

Set `JOB_ID` to the originally provisioned 32-character lowercase
hexadecimal ID. The following command uses the checked-in schema, semantic
options and four-attempt job limits:

```sh
fnlp candidate extract-semantic-job start reports.ndjson \
  --model /models/candidate-int8.fnlpq --memory-mib 16384 \
  --context-tokens 2048 --max-new-tokens 128 --max-input-bytes 512 \
  --max-result-bytes 16384 --max-mask-node-visits 2000000000 \
  --schema docs/examples/int8-semantic/schema.json --source-membership \
  --semantic-options docs/examples/int8-semantic/options.json \
  --store-results --job-dir /private/semantic-job --job-id "$JOB_ID" \
  --key-file /private/semantic-job.key \
  --limits docs/examples/int8-semantic/job-limits.json \
  --compute-threads 4 --prefill-rows 8
```

Without explicit input retention, supply the **complete original population**
again on resume, including committed records in their original order with
their original envelope bytes:

```sh
fnlp candidate extract-semantic-job resume reports.ndjson \
  --model /models/candidate-int8.fnlpq --memory-mib 16384 \
  --context-tokens 2048 --max-new-tokens 128 --max-input-bytes 512 \
  --max-result-bytes 16384 --max-mask-node-visits 2000000000 \
  --schema docs/examples/int8-semantic/schema.json --source-membership \
  --semantic-options docs/examples/int8-semantic/options.json \
  --store-results --job-dir /private/semantic-job --job-id "$JOB_ID" \
  --key-file /private/semantic-job.key \
  --limits docs/examples/int8-semantic/job-limits.json \
  --compute-threads 4 --prefill-rows 8 --materialize
```

Authentication covers the full original population before pending inference.
Committed rows are skipped; an interrupted or failed attempt remains spent.

### Explicitly retained input

Add `--spool-input --max-retained-input-mib 2` when starting to retain exact
private input envelopes alongside results. The disk cap includes framing and
is independent of model work, result-spool space and RAM admission.

For retained resume, omit the population positional argument and use
`--use-retained-input --max-retained-input-mib 2`, with the same original
semantic and storage settings. The supplied reader is ignored. The runner
authenticates the retained population and recipe under its exclusive lock,
then reads pending records through a bounded reusable record buffer. Missing
or invalid retained input is an error; it does not fall back to stdin.

An interrupted creation can leave private uncommitted input bytes. Follow the
existing input-retention recovery contract rather than silently adopting them.

## Authenticated recipe and physical scheduling

The private recipe binds both actual extraction and judge compiler identities,
model identity, runtime and prompt versions, exact schema text, grounding,
all compiler and source settings, typed claim rules/templates, judge policy,
judge planning limits, complete semantic options, encoding declarations and
all native/mask work ceilings. Schema whitespace changes the recipe. Semantic
options bind their parsed typed settings rather than JSON file formatting.

Default resume refuses uncommitted result tails or staging artifacts.
`--discard-uncommitted-tail` explicitly permits the existing repair path
after authenticating the original contract and authorized frames. It is
available only on resume and never refunds attempts or changes committed
results.

Serial execution is the default. `--compute-threads 2..64` admits one
persistent team across every pending extraction and review; width includes
the coordinator. `--prefill-rows 1..64` bounds the shared prompt workspace.
These settings compose and may change on authenticated resume because they
are physical schedules, separate from the semantic recipe. They create no
new lifetime work authority.

## Compact input and failure accounting

`--input-encoding safe-bpe-v1` selects the compact policy. Its four
`--max-tokenizer-*` ceilings span one complete document attempt: original
schema/source encoding and every generated claim/evidence prompt.

The additional four `--max-total-tokenizer-*` flags bound actual tokenizer
work across this invocation. They default to 67,108,864 input bytes,
4,194,304 output tokens, 268,435,456 pair checks and 4,294,967,296 examined pair
bytes. Both the document policy and invocation limits authenticate in the
recipe and must match on resume.

Actual work remains charged across documents and failed or cancelled
preparation, including late review-plan failure. The declared tokenizer
invocation allowance starts fresh for an explicit resume; persisted native
work, mask work and attempt debits remain spent. This distinction is explicit
in the recipe and metadata. Byte fallback remains the default and rejects
compact-only flags. Compact prompts can change model output.

## Memory, completion and embedding

After reading only bounded options and immutable job limits under caller
preparation admission, the CLI checks the full simultaneous quote before
opening the key, private schema, population or model. It prices both
preparation owners, all retained review plans and source copies, the exact
original population or one retained record, input index/parser storage,
journal and serialization buffers, two bounded private recipe buffers, full
native KV and scratch, coordinator prompt storage, worker storage and a
separate combined-result owner. These are modeled payload reservations, not
an operating-system RSS measurement.

The host keeps the actual immutable compiler pair, recipe, vocabulary,
original population and IO owners through physical worker join. Output guards
remain held through canonical serialization, spool synchronization and durable
acknowledgement. Cancellation retains its typed cause.

The final stdout protocol is
`fnlp-candidate-int8-semantic-extraction-job-v1`. It contains model provenance,
job progress, the explicit same-model diagnostic scope and optional declared
encoding/schedule settings. Private documents, schema, claims and generated
results stay in protected storage. A publication failure or deadline can
occur after durable progress, and even after completion bytes were flushed.
Authenticate stored status before resuming; failure does not imply rollback.

Library callers build `Int8SemanticExtractionJobConfig` and its pinned
`Int8SemanticExtractionJobPlanner` in `jobs::runner::extract::semantic`.
Use `hosted::corpus::jobs::semantic::int8_semantic_job_requirements` for pure
admission, then `NlpEngine::job_int8_semantic_extract` or
`job_int8_semantic_extract_scheduled` with the factory, shared vocabulary,
`SourceJobRequest`, `JobHostLimits` and owned reader. The existing job
runner remains responsible for authentication, durable work, replay,
spool-first acknowledgement and materialization.
