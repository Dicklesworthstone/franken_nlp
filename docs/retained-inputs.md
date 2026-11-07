# Resume durable jobs from retained inputs

Durable candidate jobs can explicitly retain their complete input population
and restore it on a later invocation. This removes the need to keep supplying
the original population file or pipe after a successfully initialized retained
job. The model, exact task configuration, job key, job ID and immutable job
limits remain required.

Input retention is separate from `--store-results`. Without input-retention
flags, a job keeps its existing behavior: it reads the complete original
population on every invocation and does not create an input spool.

## Select the input source

| Operation | Flags | Population argument |
|---|---|---|
| Start without input retention | No input-retention flags | Complete NDJSON file, or `-` for stdin |
| Start and retain inputs | `--spool-input --max-retained-input-mib N` | Complete NDJSON file, or `-` for stdin |
| Resume from an external population | No input-retention flags | Complete original NDJSON file, or `-` for stdin |
| Resume from retained inputs | `--use-retained-input --max-retained-input-mib N` | Omit the positional argument entirely |

Both retention flags require a positive explicit cap. They are mutually
exclusive. `--spool-input` is start-only; `--use-retained-input` is resume-only.
A cap without either selection is refused. Retained replay rejects an explicit
population argument, including an explicitly supplied `-`, so there is no
ambiguous fallback to a file or pipe. Validation occurs before configuration,
key, population or model IO.

The same flags apply to the existing INT8 `candidate job`, `candidate
score-job` and `candidate text-job` routes, and every grouped-INT4 durable job
family: source tasks, extraction, scoring, generation/chat and complete-document
maps, reductions, faithfulness and redaction. For the INT8 redaction command,
use `candidate redact --ndjson --store-results`; its existing `--resume` flag
selects the resume operation.

## Example: retain an extraction population

The following assumes an existing protected job directory, its independently
prepared raw 32-byte key file, an exact limits file and schema, and an
independently pinned local model snapshot. `JOB_ID` contains the chosen random
128-bit ID as 32 lowercase hex characters; `MODEL_SHA256` contains the expected
model digest. The memory ceiling is explicit and must cover the selected
native and job configuration.

```sh
fnlp candidate source-int4 extract-job start documents.ndjson \
  --snapshot model.int4 --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 65536 --schema schema.json \
  --store-results --job-dir private/job --job-id "$JOB_ID" \
  --key-file private/job.key --limits limits.json \
  --spool-input --max-retained-input-mib 64
```

Resume without opening the original population or reading population stdin:

```sh
fnlp candidate source-int4 extract-job resume \
  --snapshot model.int4 --expected-model-sha256 "$MODEL_SHA256" \
  --memory-mib 65536 --schema schema.json \
  --store-results --job-dir private/job --job-id "$JOB_ID" \
  --key-file private/job.key --limits limits.json \
  --use-retained-input --max-retained-input-mib 64 \
  --materialize
```

Repeat every original task, budget and scheduling option. The default tail
policy still refuses uncommitted result bytes or output stages. Add
`--discard-uncommitted-tail` only when selecting that existing recovery policy;
retained-input authentication and the exact task/recipe binding precede any
result-tail repair. Input-spool corruption is never repaired or truncated by
that flag.

Committed items are skipped. Failed attempts retain their work debits. Neither
restoring the population nor changing the invocation deadline replenishes the
job's lifetime work, attempt or verification allowances.

## What is retained and bounded

`inputs.spool` contains the exact admitted NDJSON envelope bytes, in original
item order. This includes IDs, document text, per-record task arguments and
their original JSON representation. It is not a copy of the raw transport
file: ignored empty lines and LF record separators are not replay authority.
The record bytes that the normal population reader freezes, including a CR
following a real record in CRLF input, remain unchanged. Duplicate IDs,
duplicate JSON keys, flush controls and invalid envelopes remain refused.

The input spool is a separate private, owner-only file. The journal records
only authenticated metadata and commitments, not document text. The format
binds ordered frames to the job through domain-separated keyed commitments;
a valid-looking file without the matching committed journal receipt cannot
be adopted as a population.

The `--max-retained-input-mib` cap covers the complete framed input file,
including framing overhead. It does not borrow the result-spool budget or
increase `JobLimits.max_snapshot_bytes`, `max_items`, `max_id_bytes`, or
`max_input_bytes_per_item`. The entire restored population must still fit the
existing admitted in-memory snapshot; external-memory populations are outside
this extension.

On resume, the explicit cap is a current read ceiling. It can be lower than
the original creation cap if the complete authenticated file fits. A larger
read ceiling does not expand the original receipt's sealed storage authority.
The same finite invocation deadline and checkpoint control cover the restore,
verification, planning and execution phases. No extra full-file input copy is
created solely for CLI selection.

## Recovery and inspection

Creation preflights the complete framed size, writes and syncs the input file
and its directory entry, and only then commits its receipt with the initial
job state. An interruption before that journal commit cannot make the input
file authoritative. An error may therefore leave private retained bytes even
if job initialization did not finish; it does not imply rollback or deletion.

Retained replay holds the same exclusive job lock through stored-state
authentication, frame verification, exact population reconstruction, fresh
typed recipe binding and the resumed job. Stored hashes or management metadata
cannot authorize inference. The original job key, ID, limits and native
configuration still have to match. Missing, malformed, reordered, truncated or
tampered retained data fails closed without reading an external population as
a fallback.

`fnlp job status` and `fnlp job verify` authenticate retained input content
along with the stored job. Reports add `retained_inputs` only when a committed
input receipt exists:

```json
{
  "retained_inputs": {
    "items": 2,
    "spool_bytes": 284,
    "max_spool_bytes": 67108864
  }
}
```

The numbers above illustrate the metadata shape. Reports contain no input
envelopes, input IDs, raw private-content hashes or job keys.
`uncommitted_input_spool_present: true` reports an input file without an
authorizing receipt; verification and execution cannot silently adopt it.
Ordinary jobs omit both optional additions when no input file is present.
Management remains model-free and cannot execute or resume native work.

In the candidate job commands' structured error reports, selecting retained
replay for a job that never retained inputs returns `inputs_not_retained` with
structured no-result status. Supply the complete original population through
the existing resume route, or create a new explicitly retained job. No inference, automatic
retention upgrade, input deletion or purge command is introduced by that
failure.

## Pseudonymized redaction

The job-authentication key and pseudonymization key remain independent.
`--key-file` (or INT8 redaction's `--job-key-file`) always names the protected
job key. Pseudonymization still requires its separately authorized binary
`--key-stdin`, key ID, namespace and matching commitment.

When a population comes from a file, the existing separation between that
file and pseudonym-key stdin remains. When `--use-retained-input` restores the
population, stdin may supply only the explicitly selected pseudonym key. An
external population and pseudonym secret cannot both consume stdin. Retaining
input envelopes never retains the caller's pseudonym secret or substitutes
the job key for it.

## Evidence scope

This is candidate storage and execution code, not release certification.
Regression fixtures cover retention consent and byte limits, exact replay,
tampering and missing receipts, lock ownership, refused recipe changes before
repair, parser conflicts, and retained replay that cannot read the original
population stream. Build, test, model-present parity and crash-campaign
verification remain subject to the repository's controller-selected DSR
checkpoint and their own named gates.
