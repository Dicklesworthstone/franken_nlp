# Durable native schema-extraction jobs

`jobs::runner::extract::Int8ExtractionJobPlanner` freezes the actual pinned
raw-schema batch compiler's template/tokenizer identity and exact settings
before native construction. `Int8ExtractionJobProcessor` consumes it into the
existing `NativeInt8ExtractionBatch`, rather than implementing another decoder.
The private recipe includes defaults (schema strings without numeric conversion),
all compiler/source limits, the whole task ceiling, every native work axis and
all grammar-mask budgets. Per-record schema overrides remain in the exact
original input population authenticated by `JobRunner`.

With `metadata-store` and `asupersync-runtime` on the existing Linux x86-64 or
AArch64 owned-storage profile, `NlpEngine::job_int8_extract` accepts the planner,
charged resident model, shared extraction vocabulary, existing `SourceJobRequest`
retention contract, `JobHostLimits`, owned reader and cancellation token.
The request's name is retained for compatibility; its root/key/id/limits/mode
and explicit publication semantics are identical for both native job families.

The host admits complete population, preparation, journal RAM, serialization,
IO, native KV/workspace and result storage. Input ingestion precedes native
engine construction and job-file access. One real blocking/scoped invocation
owns the population, native execution and storage lifecycle through physical
completion. These are modeled reservations, not an OS-enforced RSS ceiling.

`JobRunner` still authenticates every original envelope and the complete
recipe/model/limits before resume can repair any tail or execute pending work.
Committed items are skipped. Failed attempts keep all five model-work charges
and their mask allowance; no epoch/retry refund exists. Results remain guarded
through spool synchronization followed by journal commit. Failed source or
schema validation is not a committed empty value. Optional materialization uses
the unchanged verified no-replace `materialized.ndjson` transaction.

No input persistence, external-sort population, release activation, scheduler,
network, automatic retry or fake-native result producer is introduced. The
host currently receives an already loaded resident model. Source membership
is structural evidence, not semantic accuracy. New model-free regression
sources cover the real pinned compiler, exact schema/source binding, recipe
mutations, override isolation, cancellation and host ownership/accounting.
Rust compilation/tests, model runs, physical-crash qualification, performance
and controller DSR validation have not been executed for this checkpoint.
