# Process-hosted INT8 judging

With `asupersync-runtime`, `NlpEngine::execute_int8_judge` takes a charged
`ResidentInt8`, a `PreparedInt8Judge`, `NativeLimits`, and `CancellationToken`.
It returns `HostedOutput<Int8JudgeRun>` and retains native failures through
`HostedError::Judge`. The output's memory commitment outlives the native engine
and remains attached through the embedding caller's serialization/delivery.

`NlpEngine::batch_int8_judge` takes the same model, an `Arc<JudgePlanner>`,
`hosted::corpus::JudgeCorpusConfig`, existing `CorpusLimits`, owned reader/writer,
and cancellation token. It requires no caller-written admission adapter and
constructs one native engine for the entire bounded corpus. Both methods use
the existing process runtime, a single real blocking invocation and the existing
coordinator-only native scope. They do not create a pool, clone model tensors,
activate a catalog or change the current-candidate loader's evidence status.

The corpus uses the existing `JudgeBatchArgs` and NDJSON framing. Its `text`
is candidate A for pairwise, the document for rubric scoring, or the complete
source for faithfulness; other fields retain the normal `JudgeRequest` schema.
Per-record arguments replace optional defaults. The pinned planner binds exact
text, template, tokenizer, task policy, model and numeric profile, and refuses
budgets larger than the immutable corpus ceiling. Labels, claims and source
text are not trusted instructions or interpreted task recipes.

Every document preflights all heads before native execution. One actual KV
allocation is checked against every task; the largest live head determines
context capacity, not the sum of independent head forwards. The complete
native attempt is charged before output admission or model work. Forwards,
projected logits, attention pairs, projection dot products and multiply-
accumulates all have nonrefundable whole-stream ceilings. Failed output,
document boundaries and explicit epoch flushes never refund or renew them.
A panic leaves the private processor in a non-reusable running state.

Concrete admission reserves the prepared plan's full result byte allowance in
the actual process ledger. KV/scratch are already charged once for the corpus;
only result storage gets the per-document guard. That guard survives native
finalization, canonical serialization and output write/flush. Planning/IO
staging retain `CorpusLimits`' explicit reservations; unknown external buffers
still need honest embedder estimates. These are modeled commitments, not a
measured RSS or allocator-interception guarantee.

Invalid documents can be rejected without admitting model work. Native failure,
identity mismatch, allocation/serialization failure and cancellation terminate
the stream. A complete-output refusal is recoverable only with a clean, empty
native engine; no code clears a poisoned engine to pretend it is reusable.
Typed cancellation causes survive even when the same failure poisons that
engine. All pairwise orders/criteria/evidence heads succeed or no judgment is
published. Scores remain uncalibrated model judgments, not truth certificates.

Check both the outer hosted result and `BatchSummary.failed`. A successful
stream return reports local EOF and terminal flush, not downstream delivery or
exactly-once execution. Broken writes remain terminal with no append/retry;
late wrapper cancellation can fail the host after bytes have already escaped.
The inherited batch `prepare` interface and tokenizer/OS IO contain bounded or
blocking sections without internal cancellation checkpoints. No latency bound
or preemption claim is made.

Added source regressions cover five-axis work accounting, nonrefundable failed
attempts, overflow, interrupted/fatal processor states, planner/profile binding,
original cancellation causes, output-refusal cleanup requirements and exact
argument mapping in all three modes. They do not manufacture native inference
successes. Rust compilation/tests, model runs and performance measurements
were not performed during this implementation session.
