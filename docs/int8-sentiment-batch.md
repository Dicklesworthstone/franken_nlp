# Resident INT8 sentiment streams

`tasks::sentiment::batch` connects the existing dimensional sentiment compiler
and native scorer to the existing ordered `batch::run_ndjson` transport. It is
a serial, item-local resident-engine adapter, not a parallel neural batch,
prefix-sharing service, durable job or model-quality qualification.

`SentimentBatchConfig` fixes the model/template/tokenizer identity, task and
planning ceilings, optional defaults, and separate per-item and whole-run
five-axis native work limits. `Int8SentimentBatchPlanner` requires the already
strict-INT8 identity. Records contain `{id,text,task_args?}`; sentiment task
arguments contain only `axes` and `budget`. Policy, EOS, score space and template
come from the immutable planner. A record cannot inject or replace them.

Preparation borrows the same run controller before and after bounded compiler
work. A missing default, repeated/empty axes or over-budget request is a typed
record rejection. Actual native errors, wrong admission identities, incomplete
bundles, invalid accounting and cancellation stop the adapter. They never become
neutral sentiment, successful abstention, a partial axis bundle or a retry.

`NativeInt8SentimentBatch` accepts only a real `StrictInt8Engine` and an explicit
`Int8SentimentAdmission` provider. The private fault-injection driver is not a
public producer of model results. The host must supply actual memory authority
for resident weights, complete KV capacity, workspace, preparation and output.
The returned output guard survives the existing runner's serialization, write
and flush. Admission compares the complete compiled identity, not a filename or
caller ID. Factory seals prevent mixing plans from incompatible configurations.

The live context is the largest independent axis, not the sum of all forwards.
Complete decoder/head projection and branch-attention work is summed exactly by
the existing native sentiment plan. All five counters are subtracted atomically
before fallible admission/execution, with no refund on failure. Neither flush
nor an epoch transition replenishes the stream allowance. Delivery sequences
must advance; no public reset revives a poisoned adapter.

Tests use real pinned planning and a private uniform-logit fixture to exercise
ownership, accounting, identity, per-record overrides and failure transitions.
The fixture is not neural inference. Rust compilation/tests, full-model runs,
quality/performance and controller DSR qualification were not executed while
implementing this module. Candidate and release evidence grades remain unchanged.
