# Resume durable jobs from retained inputs

Durable candidate jobs can explicitly retain their complete input population,
creating and replaying it with one record's payload in memory at a time. This
removes the complete-population payload allocation during retained creation
and the need to supply the original population file or pipe after successful
initialization. The model, exact task configuration, job key, job ID and
immutable job limits remain required.

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

Committed items are checked during the initial full scan and skipped during
resumed execution. Failed attempts retain their work debits. Neither
replaying retained inputs nor changing the invocation deadline replenishes the
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
`max_input_bytes_per_item`. The snapshot ceiling still counts every ID plus
both the original and normalized lengths across the complete population,
even though identity normalization uses one copy of a record's bytes.
The in-memory metadata and whole-population uniqueness index remain bounded
by `max_items`, whose supported ceiling is 100,000. This is not an
external-memory uniqueness index or an unlimited population mode.

On resume, the explicit cap is a current read ceiling. It can be lower than
the original creation cap if the complete authenticated file fits. A larger
read ceiling does not expand the original receipt's sealed storage authority.
The same finite invocation deadline and checkpoint control cover the scan,
verification, planning and execution phases. No extra full-file input copy is
created solely for CLI selection.

## Streaming creation and replay

Creation reads the selected file or one-shot pipe once. It validates each
exact envelope and the current processor's typed arguments, adds the original
bytes to one incremental manifest with whole-population duplicate detection,
and stages a bounded frame in the protected input file. Each frame's full
size, including its header, must fit the explicit retention cap before any
part of that frame is written. Earlier record payloads are released.

The existing v1 frame chain binds the final population commitment, so creation
cannot seal frames until the original stream is complete. A second pass reads
one staged record at a time, compares it against its independently frozen
original binding, and fills only the chain fields in place. Staged bytes cannot
substitute for the original population. The file and its directory entry are
synced, the writable handle closes, and a read-only cursor authenticates the
complete sealed spool before the initial journal receipt is committed. No
second plaintext input file or per-record offset vector is created.

The same exclusive lock covers staging, sealing, receipt publication and
execution. The already admitted native state may exist during ingestion, but
no item preparation, native work debit or inference begins until the complete
typed population and spool have passed validation. Execution starts from the
first authenticated record using the same cursor. Exact envelope bytes,
population commitments and the retained v1 wire format match the existing
borrowed-population creation API.

Retained resume authenticates the complete population before repairing any
result tail or preparing native work. It reads every exact envelope, including
already committed items, verifies the ordered frame chain and final receipt,
and parses every record into the current processor's typed arguments. An
incremental manifest builder independently binds those actual bytes and the
fresh processor identity, recipe and limits. It keeps keyed metadata and
checks duplicate IDs across the whole population. Stored manifest hashes
cannot replace this fresh authorization.

During that scan, the reader saves one opaque bookmark for the next pending
ordinal: its file offset, prior chain state and bounded accounting state.
Only after the complete scan and fresh contract agree does the reader seek
back to that bookmark, before any permitted recovery. Execution reads the pending suffix again,
authenticating each frame before planning or a new durable work debit. It
does not reread the committed prefix during execution. A fully committed job
starts at the authenticated end position and has no further delivery.

The cursor reuses one record buffer, lending its exact bytes through
preparation, execution and result commit. It cannot read another frame while
that record is borrowed. The input file remains with the exclusively locked
job and closes before the lock is released. Read, authentication and
cancellation failures stop the session; continuing requires an explicit
reopen and another complete verification.

These paths trade bounded sequential file passes for avoiding a retained copy
of the complete input population in memory. The metadata indexes used during
fresh authorization remain in memory and obey the existing item ceiling.
Both `--spool-input` creation and `--use-retained-input` replay receive the
smaller input-memory quote. Ordinary creation and resume from an external
population still hold the complete input population. The embedding API
`JobRunner::create_with_retained_inputs` also keeps its caller-owned slice;
use `JobRunner::create_with_retained_ndjson` for streaming creation.

For embeddings, `JobInputMemory::buffer_bytes(JobLimits)` selects the common
job-buffer floor:

| Input mode | Common floor | Additional complete-population allowance |
|---|---|---|
| `JobInputMemory::CompletePopulation` | `1024 × max_items + 8 × max_input_bytes_per_item + P(max_input_bytes_per_item) + 4 × max_result_bytes + 2 MiB` | `2 × max_snapshot_bytes` |
| `JobInputMemory::RetainedRecord` | The same common floor | None |

`P` is the conservative checked parser-workspace model in
[`jobs/memory.rs`](../src/jobs/memory.rs). It prices `serde_json::Value` array
capacity, the pinned standard library's sparse B-tree nodes and split overlap,
and copied/escaped diagnostic paths at the same 64-level limit used by the
input parsers. Malformed task arguments can incur these allocations before
typed rejection, so the raw-record allowance alone is insufficient. Both
input modes include this parser reserve. Large declared per-record ceilings
can require a larger process budget even when ordinary records are small;
the immutable job input ceiling is not silently tightened on resume.

These are checked admission calculations, not measured resident-memory
figures. The floor includes the bounded metadata and per-record parser/result
allowances; it is not the complete process reservation. Hosts still add their
native, journal, preparation, serialization and IO requirements under the
shared process budget. The conservative
`JobHostLimits::required_buffer_bytes` helper retains the complete-population
quote; `required_buffer_bytes_for_input` selects the input mode explicitly.
The actual host requests and CLI preloading calculations derive the mode
from the validated input source. Selecting `--spool-input` or
`--use-retained-input` therefore uses retained-record admission automatically,
with no new memory flag and without changing the job's frozen limits or recipe.
No-model status, verification and materialization share the same record/parser
floor while keeping their separate journal, serialization and IO reserves.

## Recovery and inspection

Streaming creation checks the framed cap before each append, validates the
complete original population and final sealed file, syncs the input file and
its directory entry, and only then commits its receipt with the initial job
state. An interruption or invalid later record before that journal commit
cannot make the input file authoritative. An error may leave private retained
bytes even if job initialization did not finish; it does not imply rollback
or deletion. A later create cannot overwrite or adopt that orphan. A lost
acknowledgement after the header commit requires explicit authenticated resume.

Retained replay holds the same exclusive job lock through stored-state
authentication, complete frame verification, fresh typed population/recipe
binding, any permitted result recovery and the resumed job. Stored hashes or
management metadata cannot authorize inference. The original job key, ID, limits and native
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
file and pseudonym-key stdin remains. When `--use-retained-input` replays the
population, stdin may supply only the explicitly selected pseudonym key. An
external population and pseudonym secret cannot both consume stdin. Retaining
input envelopes never retains the caller's pseudonym secret or substitutes
the job key for it.

## Evidence scope

This is candidate storage and execution code, not release certification.
Regression fixtures cover retention consent and byte limits, exact replay,
tampering and missing receipts, lock ownership, refused recipe changes before
repair, parser conflicts, and retained replay that cannot read the original
population stream. Added fixtures cover reusable record storage, pending
bookmarks that refuse committed-prefix rereads, end-of-input replay, cursor
poisoning, unchanged population commitments, and typed-invalid committed or
pending arguments refusing before result-tail repair and new work debits.
Streaming-creation fixtures additionally cover exact v1 byte compatibility,
late typed and duplicate rejection before preparation, original-reader release,
staging tamper, framed-cap enforcement, cancellation and initialization fault
boundaries. Shared host/CLI fixtures check one-shot source ownership and the
same mode-specific memory admission.

The current implementation checkpoint has independent source and diff review
only. Rust builds/tests, native execution and parity, performance/RSS
measurements, crash campaigns and controller-selected DSR gates remain
**UNRUN**. The smaller admission calculation is not an observed RSS reduction
or throughput result, and these fixtures are not release qualification.
