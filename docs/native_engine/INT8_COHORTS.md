# Cross-document INT8 cohorts

Status: source implementation. Compilation, Rust regression execution, real-model
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
existing sequential record path. By itself, `--prefill-rows` groups prompt
positions within ONE document. The two options can now be combined to select
mixed prompt/decode token packs across documents, as described below. Other
commands do not accept cohort-rows.

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

## Combined prompt/decode token packs

```sh
fnlp candidate text-batch prompts.ndjson --task generate \
  --model local.fnlpq --memory-mib 8192 \
  --cohort-rows 4 --prefill-rows 16
```

This example admits up to FOUR documents and at most SIXTEEN total token rows
per native decoder invocation, NOT sixteen tokens per document. Both ceilings
independently accept 1..=64. The widths are illustrative, not measured choices.
The same combination works with `--task chat`. A token-pack width smaller than
the document count is supported; a rotating round-robin cursor prevents earlier
slots from consuming every pack. Invalid widths are refused before input/model IO.

Each live document advertises its remaining prompt positions or exactly one
generated feedback token. Bounded round-robin grants fill the token pack, which
then traverses both full layer loops with shared-weight linear projections.
Attention appends and attends ONE token at a time inside each document, including
when several positions from that document appear in a pack. Future prompt tokens
and other documents' KV never enter the current token's attention prefix.

A document can begin generating while a longer document is still prefilling.
There is no whole-cohort prompt barrier. Decode rows can contribute only one token
per pack because their next input depends on the pending selection. A vocabulary
projection occurs only when a prompt finishes or a decode token completes; an
intermediate prompt morsel never projects discarded logits. The existing cursor
still owns sampling, penalties, stops, EOS, byte limits and full-vocabulary scores.

Packed execution uses ADDITIONAL per-token activation storage on top of the
resident per-document cohort. The CLI derives that payload from prefill-rows;
`NlpEngine` separately reserves it alongside, not instead of, resident cohort
scratch, samplers, preparation, output and allocator headroom. No unpriced
multiplication by document count or fallback to a smaller pack is performed.

Without prefill-rows, cohort execution retains its previous one-token-per-live-
document schedule. Without cohort-rows, prefill-rows retains its previous
single-document meaning. With neither option, records and prompt tokens retain
the ordinary sequential path. None of these physical options rewrites a plan's
semantic identity, token budget or whole-corpus reserved-work ledger.

## Executable library path

`NlpEngine::execute_int8_chat_cohort` calls the pinned task cohort adapter, which
calls `generation::quantized::cohort`, which drives `Int8CohortSession` using the
existing generation cursors. The session issues `LoopRunner::run_group` with
shared-weight INT8 Q/K/V, output and MLP projections, then a shared lm-head only
for rows that currently need selection. No intermediate prompt head is computed.

The combined strategy is selected by `NlpEngine::execute_int8_chat_cohort_packed`,
which accepts the same arguments plus `Int8PrefillLimits` before the cancellation
token. It dispatches through `tasks::chat::quantized::cohort::packed` and
`generation::quantized::cohort::packed` to `Int8CohortSession::append_packed`.
The low-level `CohortTokenRun` contains a stable sequence slot and consecutive
exact tokens. Runs are strictly ordered by slot and nonempty; their total token
count must fit the separately admitted packed workspace.

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

The hosted entries accept owned `Vec<PreparedInt8Chat>`, a nonzero first delivery
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

For the original cohort strategy, group_steps equals the largest completed
per-document forward count. For packed execution, group_steps counts actual
packed decoder invocations and is independently reconstructed from immutable
prompt lengths, completed per-document forwards and the selected token width.
It is NOT substituted for logical forward work. The packed wrapper identifies
`portable-int8-ragged-token-morsel-cohort-v1`; ordinary cohort finalization will
not accept that label or silently reinterpret its tick count. The existing CLI
per-record wire schema stays unchanged and does not expose the cohort wrapper.

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

The packed path adds regression definitions for total-token admission, consecutive
causal addresses, existing prefixes and both loop slots, exact per-document work,
fairness at widths below document count, mixed prompt/decode scheduling, physical
schedule replay, scalar-versus-packed seeded cursor outputs, byte/stop behavior,
malformed logits/work, cancellation and uncertain delivery. Task/host/CLI fixtures
cover independent finalization, additional scratch pricing, combined-option bounds,
pre-IO refusal, partial tails and unchanged corpus identities/work.

These Rust definitions remain unrun. A separate Python scheduling-model comparison
matched 18,000 exhaustive/randomized cases against an independent ring-based
reference, including exact per-pack allocations. That narrow result is not Rust
execution, DSR, model-weight parity or measured throughput. Full-model and hardware
qualification remain necessary before claiming a numerical or performance award.
