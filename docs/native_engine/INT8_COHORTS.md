# Cross-document INT8 cohorts

Status: source implementation. Compilation, regression execution, real-model
scalar/cohort parity, DSR, physical memory measurements and throughput timings
are UNRUN. This is an explicit current-candidate path, not artifact activation,
profile ratification or a production/performance award. Defaults remain serial.

## Candidate CLI

With an `asupersync-runtime` build and an existing local candidate artifact:

```sh
fnlp candidate text-batch prompts.ndjson --task generate \
  --model local.fnlpq --memory-mib 8192 --cohort-rows 4

fnlp candidate text-batch conversations.ndjson --task chat \
  --model local.fnlpq --memory-mib 8192 --cohort-rows 4
```

The width of four is illustrative, not a benchmark-selected recommendation.
`--cohort-rows` accepts 1..=64 and selects actual grouped document execution;
explicit 1 exercises that implementation with one row. Omission preserves the
existing sequential record path. `--prefill-rows` instead groups prompt positions
within ONE document; combining the two strategies is not yet implemented and
is rejected before input/model IO. Other commands do not accept cohort-rows.

Input remains the existing bounded NDJSON protocol:

```json
{"id":"doc-a","sample_index":0,"prompt":"Explain the first document."}
{"id":"doc-b","sample_index":2,"prompt":"Explain another, longer document."}
```

Chat records replace `prompt` with their existing alternating `messages` array.
IDs must be unique across the entire corpus. Item ID, sample index, prompt and
policy retain their ordinary sampling identities; physical row positions and
cohort numbers never seed a request. Stops, penalties, logit biases, EOS handling,
content-byte limits and raw full-vocabulary logprobs use the existing cursor.

Each bounded input group is fully planned/admitted before its native invocation.
The final partial group executes at its actual width. Only completed, independently
validated task results are published, in input order, with the unchanged candidate
result-frame schema. The complete cohort output guard survives every write and
flush. The mandatory `batch_complete` frame AND successful process exit are still
required for whole-corpus success. Empty input completes without opening a model.

Native failure, cancellation, invalid UTF-8 finalization or another task-level
no-result aborts the whole current cohort: there is no retry, shrinking fallback,
or successful sibling publication from a failed native/finalization invocation.
Previously published records remain valid. Transport can fail after some valid
frames were written; no completion frame follows a failed write/flush.

`--max-checkpoints` applies to ONE native invocation: one document by default,
one complete cohort when selected. It is not multiplied/replenished per row.
`--timeout-seconds` remains one deadline for the entire corpus, including input,
preparation, model loading and output. Blocking IO is cooperative, not preemptible.
Whole-corpus record/forward/head/input/output ceilings never reset between groups.

Preparation admission scales retained document plans/staging by the selected
maximum width. Large cohorts can require a larger explicit `--preparation-mib`.
Native admission separately includes every row's KV capacity, grouped workspace,
simultaneous samplers and all retained results; a large width can therefore be
refused even when one record fits. There is no hidden automatic width reduction.
All byte reservations are modeled resource-ledger charges, not OS RSS guarantees.

## Executable library path

`NlpEngine::execute_int8_chat_cohort` calls the pinned task cohort adapter, which
calls `generation::quantized::cohort`, which drives `Int8CohortSession` using the
existing generation cursors. The session issues `LoopRunner::run_group` with
shared-weight INT8 Q/K/V, output and MLP projections, then a shared lm-head only
for rows that currently need selection. No intermediate prompt head is computed.

A step can mix prompt and decode rows with different context lengths. Every
sequence owns all 44 logical KV slots and its independent causal/RoPE coordinate.
Attention never concatenates documents or broadcasts one row's position. Both
passes still execute 22 layers followed by the same final RMSNorm. Finished
sequences are omitted from later groups, without replacing their membership or
recycling their work allowance into new documents.

The cohort binds one already-materialized model once and shares one RoPE table.
Weights are not cloned or expanded. It runs in one existing blocking coordinator
and one scoped native invocation; it does not introduce threads, parallel CPU
teams, dynamic/continuous admission, prefix sharing or an unbounded daemon.

The hosted entry accepts owned `Vec<PreparedInt8Chat>`, a nonzero first delivery
sequence, `ChatCohortLimits` and the ordinary cancellation token. Limits contain
`native` (per-row context ceiling plus one run budget), aggregate sampler bytes,
preparation reserve bytes and the complete result-envelope cap. Actual row KV
capacities are derived from each plan's worst-case forward count, not padded to
the maximum document length. Complete result admission requires at least the sum
of all task result-byte ceilings plus 4096 bytes of cohort metadata headroom.

The returned `HostedOutput<Int8ChatCohortResult>` retains real output-memory
ownership after physical native cleanup. `results` are in input order;
`group_steps`, `planned_work` and `model_work` expose the grouped schedule and
complete decoder/head/attention accounting. Per-row semantic execution labels
stay unchanged; the wrapper identifies the physical cohort implementation.
Retain the guard through serialization, delivery and flush.

Lower-level native and pinned task `execute_with_sink` APIs support provisional
interleaved token events addressed by unique nonzero `request_seq`. They do not
publish a terminal success frame or replace host-owned admission/drain duties.
No CLI or hosted multiplexed-token-stream surface is introduced here.

## Correctness boundaries

Each sequence has a non-renewable independent work ceiling in addition to the
aggregate projection ledger. Whole-step indices, token IDs, context and work are
checked before native mutation. After native work begins, any fatal result or
unwind poisons the cohort; session Drop clears every cache and hidden state.
Successful receipts reconcile all native per-row work, aggregate work and group
steps against the existing cursors, including byte-refused token proposals.

Added regression definitions cover ragged independent attention and both loop
passes; work/memory isolation; scalar-versus-cohort seeded cursor behavior;
permutation independence; cancellation, malformed logits and uncertain delivery;
pinned finalization and routing; owned host admission; CLI strategy separation;
bounded cohort collection, partial tails, duplicate refusal and complete output
admission. Synthetic fixtures exercise these boundaries but do not establish
full-model token parity, measured traffic savings or end-to-end throughput.
