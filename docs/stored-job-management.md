# Stored job management without original inputs

`jobs::StoredJob` supplies management-only access to an existing owned job on
the `metadata-store` Linux x86-64/AArch64 profile. It needs the protected job
key, expected random job ID and exact original `JobLimits`, but no model,
processor, original corpus, execution recipe or tokenizer. Opening it is an
explicit request to read already-retained private results for authentication.
It creates no inference, input spool, directory, runtime or recovery attempt.

The distinction matters: authenticating stored output is not proving that a
newly supplied corpus or recipe matches the job. StoredJob cannot create an
attempt, expose a replay manifest, yield an OwnedJob, or become a processor.
Execution resume still goes through the original complete-population and
semantic-contract checks. The format and original replay APIs are unchanged.

## Authentication and operations

Open holds the same exclusive kernel lock as OwnedJob. It authenticates the
journal header, expected job ID, protected key and original limit commitment;
validates item cardinality and keyed-ID uniqueness; recomputes the ordered
population commitment from every authenticated item binding; and invokes the
existing full prefix verifier over work debits, states and committed spool
frames. A changed, independently authenticated row cannot silently redefine the
population. Corrupt committed content blocks open, even for a status request.

`status` returns a metadata-only snapshot from authenticated open or verification.
The report explicitly states `authenticated-stored-state-not-input-replay-v1`.
It contains the random job ID, counts, spent work, committed spool length,
orphan-tail length and staged/publication state, not private text, input IDs,
key material, execution/recipe digests or content commitments. A valid committed
prefix with an uncommitted suffix remains inspectable: the suffix is reported,
not promoted or discarded.

`verify` checks complete stored-state consistency. Uncommitted tails or stages
are errors; the command never repairs them. If output was published before its
journal acknowledgement, verification compares it byte-for-byte with committed
frames but leaves the journal state unchanged. Merely reporting the presence of
such a destination in status is not a claim that its bytes match.

`materialize_ordered` is the explicit publication operation. It requires all
items committed and no unresolved orphans, and reuses the existing private,
synced, no-replace `materialized.ndjson` transaction. A lost publication
acknowledgement is reconciled only after exact output comparison. Unrelated
output is never overwritten. No work, attempts or model results are created or
refunded by any management operation.

These are non-destructive application-level inspections, not forensic read-only
mount operations: the existing fsqlite opener may perform its normal rollback
recovery and connection PRAGMA setup. Existing platform, same-UID/mount threat,
filesystem durability and database qualification boundaries still apply.
Callers must admit metadata indices, database-engine allocations, bounded frame
buffers, materialization staging and allocator overhead; JobLimits are not RAM
permits. Cancellation is cooperative and does not preempt OS IO.

Fifteen regression-test functions cover original-free status/publication,
unchanged spent work, wrong keys/IDs/limits, original-required execution resume,
orphans, corruption, population reconstruction, exclusive locking, uncertain
publication, foreign destinations and cancellation. They are source fixtures;
Rust compilation/tests and physical crash testing were not run. No model,
quality, production DSR or platform durability gate is promoted.
