# Durable item-local batch runner

Implementation source for `franken_nlp-040`, extending `docs/owned-jobs.md`.
This is not a platform durability award, native-model fidelity receipt, or
controller validation result. The controller owns compilation, test execution,
crash injection, and host-admission qualification. None was run for this change.

## Public path

With `metadata-store` on Linux x86-64/AArch64, `jobs::runner::JobRunner` bridges
`OwnedJob` to the existing `BatchProcessor` lifecycle. `create` and `resume`
require a protected directory, a host-protected random job key/ID, immutable
`JobLimits`, a complete borrowed population, and an owned processor implementing
`DurableBatchProcessor`. Importing the module creates no files or runtime.

For this explicitly named `identity-json-envelope-v1` profile, each `JobInput`
contains the original JSON `{id,text,task_args?}` envelope in BOTH `original`
and `normalized`, with the same `id` also supplied as `JobInput.id`. Nonidentity
normalization, duplicate JSON keys, unknown envelope fields, flush commands,
and mismatched IDs refuse. Every envelope is validated before opening storage;
whole-population size and duplicate-ID checks precede that parse pass. The exact
ordered envelopes (including per-item arguments), execution identity, processor
recipe/defaults, and job limits enter the existing keyed frozen manifest.

`step` executes only the next uncommitted ordinal. `run` repeats it until all
items commit. Planning uses the same caller-provided cancellation control.
Before execution, the runner checks the two-axis batch estimate against the
full native estimate, checks the declared result bound, and calls `OwnedJob::begin`
to persist the attempt and all six work debits. It then invokes
`execute_with_context` with stable ordinal-based delivery coordinates, never a
new seed. The output is BORROWED by `Attempt::commit`: its admission guard stays
alive through canonicalization, spool sync, journal commit, and acknowledgement.

Any failure, cancellation, or unwind closes this runner to further execution.
Even a nonfatal batch document failure is not cached as a successful job result.
Retry requires explicit drop/reopen with a fresh clean processor. Committed
items are not prepared or inferred again. Failed/interrupted attempts remain
charged in the existing authenticated journal, including all model and mask
axes and the attempt ceiling; new process-local counters cannot refund them.

`progress`, `read_committed`, `verify`, and `materialize_ordered` expose the
owner's existing protected result/recovery path. Materialization is explicit,
complete-population-only, and never replaces an unrelated existing output.
Raw stdout and downstream consumers gain no exactly-once delivery guarantee.

## Trust and memory boundaries

`DurableBatchProcessor` is a trusted embedding interface, NOT an admission
certificate or deserialized task factory. Its immutable recipe must include all
meaning-affecting settings and effective sampling addresses. It must enforce
its complete work/result ceilings, return only after native cleanup, and carry
its host resource guard with the output. A hostile implementation can lie just
as a hostile implementation of `BatchProcessor` can; built-in adapters must
construct their native processors from the same private frozen settings and
retain the native admitted-identity checks.

`JobLimits` provides bounded metadata/content/work, not a process memory permit.
The embedding host still admits the borrowed input population, tokenizer and
vocabulary, plans, resident model/KV/scratch, fsqlite allocations, canonical JSON
and frame staging, and materialization buffers. No ambient runtime, pool,
loader, detached thread, retry loop, signal handler, or second inference engine
is added. Blocking I/O remains non-preemptible by cooperative checkpoints.

## Regression sources and remaining work

The runner tests cover guard lifetime through serialization and commit
checkpoints; interrupted/uninterrupted ordered equivalence; skipped committed
work; late cancellation; serialization and recoverable processor failures;
six-axis and attempt-budget persistence; recipe/identity/limits/population
mismatches; and complete-envelope rejection before filesystem side effects.
These tests are fixtures, not native accuracy or filesystem crash qualification.

The public CLI dispatcher, input spooling, external-sort populations,
partition/corpus-global execution, live controller/DSR/Beads/bv/Agent-Mail
qualification, and platform-native crash campaigns remain separate work.
