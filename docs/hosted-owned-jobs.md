# Process-hosted durable source jobs

`NlpEngine::job_int8_source` connects complete NDJSON population ingestion,
`JobRunner`, the resident INT8 source-task adapter and the existing real
process-ledger admission. It is available with `asupersync-runtime` plus
`metadata-store` on the owned-job Linux x86-64/AArch64 storage profile.

The supported tasks are NER, keyphrases, cited summarization and passage QA.
This is the existing serial native profile, not GEMM/continuous batching,
a new task implementation, a model-quality promotion or a public CLI command.

## Invocation

Supply a resident `ResidentInt8` from the same `NlpEngine`, the pinned source
planner and vocabulary in Arcs, the existing `SourceCorpusConfig`, an owned
`BufRead + Send + 'static` reader, and the same caller cancellation token.
There is no public custom admission/driver parameter: per-item admission uses
`CorpusAdmission`, including the actual resident artifact identity, allocated
KV quantity, and a live result-memory reservation.

`hosted::corpus::jobs::SourceJobRequest` names an existing protected root,
host-protected random `JobSecret`, random `JobId`, immutable `JobLimits`, and
`JobOpenMode::Create` or explicit `Resume(TailPolicy)`. Constructing the request
is an explicit result-storage request; the API generates no weak/default key,
reads no secret from argv, creates no directory and never silently adopts an
existing job. `materialize` separately requests publication after completion.

`JobHostLimits` admits the entire population rather than a live batch window.
It combines checked population/index/byte-staging floors with mandatory caller
prices for planner/vocabulary/default/source/grammar preparation, the reader's
retained backing buffers, fsqlite engine allocations, and serialization trees
and allocator slack. The database file cap is not misrepresented as a database
RAM bound. Native KV/RoPE/scratch and per-result output claims are additional.
These modeled commitments are not an RSS bound or an allocator interceptor.

## Execution and physical ownership

The existing hosted preflight requires the actual process runtime, the same
resident-model resource domain, and the serial profile's single blocking
coordinator. All claims are acquired before dispatch. In the one existing
`spawn_blocking`/`scoped_cpu(0)` invocation, the host first ingests the complete
population. Invalid later input therefore fails before native engine creation,
job storage or inference. The exact typed recipe/envelopes are then frozen by
`JobRunner` before create/resume; per-item native preparation/admission remains
unchanged. The engine and vocabulary are reused across all pending items.

The job buffer charge is held through ingestion, journal/spool use, every
native execution, optional materialization and cleanup. Each result's actual
admission guard travels into the existing spool-first/journal-second commit.
The runner closes its journal/spool/lock before native buffers, population,
planner/vocabulary, reader and job memory charge drain. Captured aggregates
retain storage-before-charge drop order even when queued work never starts.
No live job handle or uncharged output escapes the blocking closure.

The original finite deadline/checkpoint control covers ingestion, planning,
recovery, execution, result commits and materialization; it is not reset per
item. Cancellation does not preempt OS IO or release native buffers early.
The existing actual-completion handshake suppresses success on late cancellation
and waits for physical cleanup. A hosted stop or wrapper/drain failure takes
precedence over a job return value. Use authenticated resume to reconcile a lost
acknowledgement, rather than assuming an error means no durable progress exists.

## Resume and outputs

Replay requires the entire original NDJSON population and exactly the same
key, identity, recipe and immutable job limits. Committed items are not inferred
again. Failed attempts and all six work axes stay charged across invocations,
even though a fresh native adapter starts with new process-local counters.
The newly supplied host limits may change admission capacity but cannot change
that frozen semantic/work contract. There is no automatic retry loop.

Without `materialize`, the return is only `JobProgress`; committed private
results remain in the protected result spool. Explicit materialization uses
the existing verified, synced, no-replace `materialized.ndjson` transaction.
The method touches no ambient stdin/stdout and promises nothing about arbitrary
pipe consumers. Inputs are retained only in the admitted in-memory population;
persistent input spooling and external-sort populations are still separate work.

Nine host-accounting/lifecycle regression-test functions are supplied, including
resume after a failed second item, explicit publication without re-inference,
and changed-original refusal. The existing native source-adapter and dispatch
suites cover their own seams. None of these tests, Rust compilation, real-model
inference, physical crash/drain campaigns, or controller DSR qualification was
run for this checkpoint. Public `fnlp job` command dispatch remains unconnected.
