# Ordered item-local NDJSON batch execution

`batch::run_ndjson` replaces the batch stub with a bounded blocking stream.
`batch::judge::NativeJudgeBatch` runs pairwise, rubric and faithfulness tasks;
`batch::extract::NativeExtractionBatch` runs structural or source-bound schema
extraction. Both use the existing pinned planners/scorers/decoder and borrow
one already-admitted engine. No model load, runtime, worker pool, second KV
cache, unconstrained retry, durable job, or continuous/GEMM batching is added.

This is native/library SOURCE IMPLEMENTATION. Compilation, builds, tests and
real-model measurements have not been run. The inference CLI activation and
process-admission gates remain unchanged. This library protocol is explicitly
named `fnlp-item-local-batch-v1`, schema_version 1; it is not an undeclared
extension of the existing frozen `robot schema` response union.

## Wire and lifecycle

A document is `{ "id": "caller-id", "text": "original text", "task_args": ... }`.
Task arguments may be omitted only when a host-configured default exists.
The sole control request is exactly `{ "flush": true }`. Duplicate/unknown
keys, nested duplicate task keys, invalid UTF-8, whitespace-only lines and
malformed controls produce one correlated `doc_error`. Empty LF/CRLF lines
are ignored. Source text is never trimmed, normalized or retokenized here.
An unterminated final record is processed at EOF.

Every complete nonempty physical record receives a checked sequence before
parsing. Caller IDs are echoed only after strict envelope/ID validation.
Events carry protocol, schema_version, execution, event, epoch, request_seq;
document events additionally carry input_line, byte_offset and caller_id when
known. Success embeds the typed task result, preserving extraction JSON as an
exact string rather than routing its numbers through floating point. Errors
contain only fixed categories and an optional original typed cancellation kind.
No parser paths, arbitrary provider messages, raw input, prompt hashes or
private execution identities enter errors or operational metadata.

Events are `run_start`, `doc`, `doc_error`, `flush`, `run_complete`, `run_error`.
A flushed `run_complete` means EOF was drained, not that every item succeeded:
inspect the summary's failed count. Output contains no clocks/random run IDs
and is canonical. The executor is serial and therefore ordered with no reorder
queue; no concurrency, throughput improvement or model-quality claim follows.

Only one item/result is live. The writer finishes write_all AND flush before
another record is requested, providing synchronous backpressure. A caller's
BufRead implementation may have its own bounded read-ahead buffer. Oversized
records are drained to LF without retaining their tails, then fail once; an
aggregate input ceiling prevents an endless oversized record or blank stream.
Line limits count bytes before LF, including CR in CRLF.

The duplicate-ID set is bounded by count and total ID bytes. Accepted IDs stay
reserved even when later planning/execution fails. They reset only AFTER an
explicit numbered flush has been written and acknowledged by the local writer.
EOF emits a final drain boundary. A failed flush never starts a new epoch.
These epochs are not corpus-global resolve transactions or durable snapshots.

Output is size-preflighted, canonicalized and staged as a complete bounded
record before touching the writer. Ordinary records cannot consume the small
reserved terminal-event allowance. Partial writes and flush failures poison
the stream: no retry, further input, or appended run_error is attempted on
that unknown output prefix. Writer flush proves neither downstream processing
nor persistence. Raw stdout is not an exactly-once or resumable job protocol.

Cooperative checkpoints run between records, during chunked reads and inside
native execution. A blocking I/O operation cannot be preempted by a checkpoint.
The host owns signals, OS I/O cancellation and panic supervision. Panics are
not caught and reclassified as document errors. There is no success record for
an incomplete native result, cancellation or invalid engine state.

## Actual task integration

For judge tasks, text is A (pairwise), document (rubric), or complete source
(faithfulness). The strict task_args enum supplies the remaining fields with
the SAME public JudgeRequest policy/budget/rubric types. `JudgeBatchPlanner`
reuses one pinned JudgePlanner and a frozen host identity/budget ceiling.
Per-document arguments cannot raise that ceiling. All actual judge heads,
including both pairwise orders and every evidence window, remain charged.

For extraction, task_args has `schema` (a string of exact JSON), `grounding`
(`structural` or `source_membership`) and the standard TaskBudget. Schema
numeric constants therefore retain the exact 38-digit domain. Source-bound
schemas explicitly select the source runtime; a verbatim annotation in
structural mode is rejected, never removed. Source-membership mode requires
an actual verbatim field and the independent SourceSpansVerified postcondition.

`ExtractionBatchPlanner` compiles a fixed schema/source prompt. Its renderer
sees only internal placeholders. The caller-authorized declarative schema
occupies a TaskInstruction segment but is byte-fallback encoded, as is source,
so malicious property names/marker spellings cannot emit privileged controls.
There is exactly one Document segment, bound to the original SourceDocument.
Marker containment is not an empirical prompt-injection guarantee. Compiler
caps, runtime, schema, exact prompts and task policy bind the prepared identity.
Unsupported schema keywords and impossible source constraints refuse before
model admission. The original TaskPlan/SourceDocument remain accessible for
explicit semantic second-reader planning; verification is not auto-enabled.

The caller constructs the reusable vocabulary once and supplies the engine
and its actual admission hook. The hook receives the proposed complete private
identity and work ceiling, returning the identity admitted by the host plus
the host's real RAII guard. There is NO default hook inventing authorization.
It is an embedding seam to the ratified admission path, not a local replacement
PermitBroker. The guard stays alive through native execution, independent
validation and logical KV cleanup. Returned identities are compared exactly;
no model/backend/task field is silently repaired at execution time.

The stream charges an aggregate forward/projection ceiling before each
attempt. Failed attempts are not refunded and flush does not replenish work.
Judging has exact cold-head counts. Extraction reserves its maximum prompt plus
nonterminal-output forwards; it may finish earlier and reports actual work in
its existing output. A separate ExtractionMaskBudget bounds per-mask, per-item
and entire-run mask work; its nonrefundable run allowance also survives flush.
`NativeExtractionBatch::reserved_mask_visits()` exposes that charged total.

Native preflight checks every task's context and the complete engine KV
reservation, not merely occupied positions. After execution the adapter requires
empty logical KV and consistent work counts before permitting the next item.
Weights/buffers stay resident. Capacity or task-output refusal can fail one
item; typed cancellation, wrong admitted identity, poisoned/inconsistent engine
state and native failures stop admission. Native cleanup is inherited from the
existing prefix-session and constrained-decoder guards, never simulated here.

## Source regression coverage and remaining work

Twenty-one added source cases cover framing, Unicode, malformed/oversized
records, duplicate epochs, budgets, partial output failure, cancellation,
actual pinned judge planning/scoring with synthetic logits, schema precision,
source bindings and native failure classification. They remain UNRUN.

Still separate: inference CLI/model-activation integration, a frozen robot
schema extension, supervised async I/O, completion-order parallel scheduling,
layer-major continuous batching, corpus-global resolve flush transactions,
durable spool/journal recovery, native execution tests and real-model
qualification. The synchronous embedding API implements no substitute for them.
