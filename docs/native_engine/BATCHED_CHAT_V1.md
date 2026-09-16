# Prepared chat/generate cohorts on the native batch engine

Status: source implementation; Rust compilation/tests and native-model execution
remain UNRUN. This connects the pinned task path to shared-weight layer-major
execution. It does not activate model loading, a CLI, an interactive terminal,
a durable job, or the existing whole-item serial NDJSON runner.

## Entry points

`tasks::chat::batched::{preflight, execute, execute_with_sink}` accept an existing
`EagerBatchEngine`, the host's actual loaded-model identity, a finite slice of
`BatchChatRequest`, an explicit `BatchChatBudget` and the request's cooperative
control. Each request borrows a `PreparedChat` produced by the existing pinned
ChatPlanner (either chat or generate), its ACTUALLY admitted identity, an empty
physical cache slot, and a unique nonzero delivery sequence.

No prompt is rebuilt after admission, no model is cloned, and no private runtime
or fabricated resource guard is introduced. The native cohort checks exact
request identities and the common model/tokenizer/template/backend binding. The
task wrapper checks each selected slot's complete K/V reservation against that
task's own budget, not only the cohort ceiling. Existing busy slots are not reset.
The embedding host retains its real model/process/output guards and accounts for
weights, allocator/metadata overhead and serialized output staging separately.

The wrapper uses the prepared task's pinned tokenizer. There is no public byte
decoder argument at this boundary that could silently substitute another token
mapping. All tasks enter the shared generation Cursor and its existing addressed
sampler, processors, stop/EOS semantics and reserve/permit token-delivery path.
Short prompts can decode while longer prompts continue to prefill; completed
rows close independently. Results preserve original request order.

## Independent finalization

PreparedChat's existing finalizer recognizes exactly two execution variants.
The scalar eager variant computes one full-vocabulary projection for every
prompt/feedback forward. The batch variant omits intermediate prompt lm-heads
and computes full projections only at token-selection positions. Each variant
has its own exact expected projection count; neither can borrow the other's work
numbers. Raw selected-token logprobs still require the actual full-model
normalizer and retain their named score space.

The same finalizer checks exact decoded bytes, UTF-8, EOS/termination, per-row
sample index/seed, token counts, full-vocabulary score fields, observed forward
work and complete serialized row size. The cohort finalizer additionally checks
request routing, aggregate planned/actual work and group-step count. A wrong
identity, route, execution label, work count or decoded-byte invariant is fatal,
not downgraded into an ordinary document failure.

`BatchChatResult` uses the separate `pinned-chat-batch-v1` library envelope. Its
results are tagged `completed` with a ChatResult, or `no_result` with a fixed
reason and delivery/sample coordinates. Incomplete UTF-8 and an oversized
complete row result are ordinary per-row no-result states; valid siblings remain
available. Source bytes, raw decoder diagnostics and private execution/request
hashes are never placed in those error records. A nonallocating size pass checks
the COMPLETE canonical cohort envelope against max_result_bytes before canonical
storage is built, in addition to all individual task limits.

Native or streaming failures/cancellation abort execution; they are not successful
partial cohorts. Token events are untrusted incremental data until finalization.
Already delivered bytes may precede a final no-result/error, but are never
retracted or retried. The cohort's owned sequence handles are closed on native
success, failure and unwind before independent task finalization returns.

Regression sources use actual pinned planning/byte decoding and synthetic native
outputs to check mixed chat/generate finalization, selected-logit accounting,
per-row no-result isolation, routing/work corruption refusal, complete K/V prices
and complete cohort serialization limits. They are UNRUN and do not establish
native-model parity, quality or throughput.
