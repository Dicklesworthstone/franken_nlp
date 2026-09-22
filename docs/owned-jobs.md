# Authenticated, resumable owned jobs

The `jobs` library is an explicit result-storage boundary for serial item-local
work. It is separate from the optional metadata-only history store and does not
change the live NDJSON batch protocol. It adds no runtime, worker, task factory,
network access, or automatic retry loop.

## Available operations

`FrozenManifest::freeze` checks the complete ordered input population before any
job file is opened. `OwnedJob::create` initializes a private job;
`OwnedJob::resume` reopens it after reconstructing the same manifest.
`OwnedJob::begin` returns an exclusively borrowed, durably debited `Attempt`.
The host performs its already-admitted native operation and passes the guarded
result by reference to `Attempt::commit`. `OwnedJob::verify` independently
checks the authorized result prefix, `read_committed` reads an authenticated
result, and `materialize_ordered` publishes the complete ordered output.

The owned filesystem/database operations require `metadata-store` and the
existing Linux x86_64/aarch64 local-output profile. `OWNED_JOBS_AVAILABLE`
reports compile-time availability, not filesystem capability or ratification.
Manifest and commitment construction themselves have no filesystem side effect.
No CLI command or hosted neural adapter is activated by this change.

## Freeze the actual replay contract

The host supplies a fresh random 128-bit `JobId` and a fresh 256-bit
`JobSecret`, generated once through its admitted entropy source. It must retain
the secret under its protected-key policy before creating the job. This module
neither generates nor persists keys. `JobSecret::read` accepts exactly 32 bytes
through the existing private-parent, no-symlink key reader. Never regenerate the
secret on resume; neither a key filename nor a human-readable label proves key
continuity. Best-effort key clearing is not a compiler-proof zeroization claim.

`JobContract` binds the complete canonical `ExecutionIdentity`, immutable
`JobLimits`, and the host's typed semantic recipe. The recipe must include every
meaning-affecting choice not already in the identity: task arguments/defaults,
effective sampling seed and address version, normalization, output schema, and
item-local dependency scope. A journal is not capable of discovering omitted
host semantics or proving that a native engine used the supplied identity.

Each `JobInput` supplies its exact identifier, original bytes, and explicitly
selected normalized bytes. Identity normalization uses the original bytes in
both positions. Inputs are borrowed during freezing and are not spooled.
Domain-separated, length-delimited HMAC-SHA-256 binds identifiers, original and
normalized inputs, the ordered population, execution, recipe, limits, journal
rows, and result frames. No raw private-content SHA-256 is exported. Journal
metadata retains keyed bindings, counters, states, and pointers, not source
text, caller identifier strings, task arguments, prompt text, or result bytes.
The opt-in result spool and materialized output DO contain result bytes; they
are owner-only, not encrypted at rest.

Uniqueness is checked over the whole snapshot, not per batch flush epoch. The
initial implementation uses a bounded in-memory index with a hard ceiling of
100,000 items and explicit aggregate input-byte limits. Larger populations fail
closed; an external-sort/index implementation is not present. Resume requires
reconstructing the full original manifest. Mismatches report closed field
categories rather than printing private old/new values.

## Commit and recovery protocol

For each next frozen ordinal:

1. Verify the exact replay input, atomically persist `admitted` plus the attempt
   count and all native/mask work debits, and complete the journal sync boundary.
2. Persist `running`. Only then return the borrowed attempt to the host.
3. Canonically encode the result, append its fixed-width authenticated frame,
   and sync the result spool.
4. Atomically persist the frame generation, offset, length and commitment with
   `result_committed` and the advanced committed prefix. Complete the database
   and directory barriers before returning successful commit progress.

All five native work dimensions and mask visits remain debited after failure,
early completion, cancellation, or process restart. Attempt limits also survive
resume. The host supplies a sound work ceiling derived from the native plan and
still enforces the actual work and process-memory admission independently. The
journal is not an inference permit, memory ledger, or model-quality receipt.
An unfinished, failed or unwound attempt poisons its session; retry requires
explicitly reopening the job and paying another debit. An acknowledged-or-lost
journal commit is discovered on resume and is not rerun to rebuild its output.

Recovery authenticates the header, every item row, the full contiguous committed
prefix, all work/attempt totals, and each referenced frame. It never scans a
valid-looking tail into authority. Missing/corrupt committed bytes are fatal and
are not repaired by truncation. `TailPolicy::Refuse` diagnoses uncommitted spool
bytes or reserved-name materialization stages without deleting them.
`DiscardUncommitted` explicitly permits truncating only the unauthorized spool
suffix and discarding reserved-name stages AFTER prefix authentication. Unrelated
files are not scavenged. A linked stage is removable only when it names the same
inode as the published destination, which is never removed. Interrupted repairs
are safe to repeat under the same explicit policy.

## Ordered publication and storage profile

Only a fully committed population can publish `materialized.ndjson`. One
canonical result plus LF is streamed per frozen ordinal; the full corpus is
never accumulated in memory. The existing local-output transaction creates an
owner-only stage, syncs it, and uses no-replace hard-link publication with
explicit directory barriers. The journal records `materialized` afterward.
A crash between publication and that journal transition is reconciled by exact
byte comparison with committed frames. An unrelated or different existing
output is refused, never overwritten. Output records do not gain a new envelope;
the host's typed result schema determines their content.

`journal.fsqlite`, `results.spool`, `job.lock`, `materialized.ndjson`, and the
`.fnlp-redact-<pid>-<sequence>.part` stage namespace are reserved within a
pre-existing private job directory. Files are mode 0600, the parent is mode
0700 and owner checked, and lookups are descriptor-rooted through the existing
procfs profile. A kernel-held advisory lock enforces a single cooperating owner;
lock files are never deleted based on stale PID guesses. The database closes
before the lock and directory handles are released. Same-UID/root attackers,
malicious mounts, and arbitrary external mutation are outside this profile.

The fsqlite journal requests and reads back DELETE/FULL mode, 4096-byte pages,
and a database page ceiling; unsupported behavior fails closed. Database,
rollback-sidecar, spool, result, materialization, item, attempt and work ceilings
are explicit. The rollback sidecar has an additional bound of twice the database
ceiling plus 64 KiB. Staging permits one bounded output attempt and refuses new
staging until leftover reserved stages are explicitly recovered. Database-engine,
canonical-tree, index and caller IO allocations still require host memory pricing;
these limits are not an allocator interceptor or an RSS guarantee. Blocking
filesystem/database calls are not made preemptible by surrounding checkpoints.

## Evidence and remaining work

The change includes 12 commitment/manifest/frame tests and 17 concrete owned-job
regression tests. They cover RFC HMAC vectors, whole-population uniqueness,
semantic drift, corrupt and truncated frames, lock/permission boundaries,
persistent retries, injected commit-boundary failures, lost acknowledgements,
corrupt committed prefixes, explicit tail/stage recovery, publication recovery,
and uninterrupted-versus-resumed output equivalence. Synthetic stored results
are not native inference or NER-quality evidence.

Compilation, Rust tests, real-model execution, and real power-loss/disk-full
campaigns were not run in this session; validation remains controller-owned
under `WIRING.md`. No build, test, physical-durability, phase-gate or release pass
is claimed. Read-back PRAGMAs and sync calls are implementation checks, not a
platform durability certificate.

This does not implement public `fnlp job` commands, a hosted native job runner,
input spooling, external-sort manifests, cross-item reductions, cache dependency
invalidation, encryption-at-rest, key generation, or retention/purge commands.
Owned committed records and no-replace publication do not imply exactly-once
delivery to arbitrary stdout consumers. Replaying bytes to an external consumer
requires that consumer's own deduplication or acknowledgement protocol.
