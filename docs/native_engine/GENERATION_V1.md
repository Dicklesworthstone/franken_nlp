# Native generation, pinned chat, and ordered batch integration

The implementation has three connected entrypoints:

- `native_engine::generation::GenerationPlan`: bounded greedy or addressably
  sampled native eager generation from an exact prompt-token sequence.
- `tasks::chat::ChatPlanner`: pinned `GenerateRequest` and multi-turn
  `ChatRequest` planning, real FreeText TaskIR, native execution and independent
  response validation.
- `batch::generation::NativeGenerationBatch`: the same task path on the existing
  bounded, ordered NDJSON stream, retaining one admitted engine and planner.

These are source implementations. Rust compilation, builds, source regressions,
real-model execution, oracle parity and performance measurements are UNRUN.
They do not bypass model activation, the owned runtime, or process admission.
The existing CLI and blocked model-root activation gate are unchanged. This is
not an assertion that `fnlp chat` or `fnlp batch --task generate` is activated.

## Addressed sampling and effective policy

Generation uses the existing `fnlp-sampler-v1` SHA-256 addressed draw, TopK and
exact Nucleus implementations. Every sampled request supplies a 32-byte effective
seed. A missing seed must be obtained by the admission owner via its approved
random authority; the decoder neither invents one nor consumes an ambient RNG.
The completed result includes the seed as 64 lowercase hex characters.

The private stable request key binds the exact prompt, complete effective
options, declared artifact/backend identity, task and stable caller/job item ID.
Item ID and sample_index also bind the admitted decision-policy identity: the
host cannot admit one sample/item and unknowingly execute another. The draw
adds sample_index, autoregressive step and draw_index zero. Physical batch row,
transport request_seq and flush epoch are not sampling coordinates. The private
key and raw content-derived identity never enter results or errors.

The versioned processor order is static bans and minimum-length EOS exclusion,
repetition penalty, presence penalty, frequency penalty, logit bias, temperature,
top-k, then top-p. Counts include prompt and already committed output tokens.
Settings have explicit bounded fixed-point wire forms. Bias cannot unban tokens.
Bias-map keys must be canonical unsigned decimal token IDs; alternate spellings
such as `1` and `01` cannot silently collapse into one token policy. The decoder
also rejects oversized bias maps, out-of-vocabulary IDs and invalid bias values.
Sampling rejects nonfinite raw logits even at banned rows. Greedy ties retain
the lowest token ID. Sampled ranking and half-open interval arithmetic use the
existing pinned sampler rules; these are named semantics, not an HF-parity award.

With top_k absent, nucleus selection evaluates the complete legal vocabulary,
uses its full normalizer and retains the probability-threshold crossing token.
With top_k present, nucleus is explicitly conditional on that retained top-k
set. It is not mislabeled unconditional model probability. Optional token
logprobs always use the RAW full model vocabulary before processors, name their
score space, and are neither processed sampling probabilities nor confidence.

The separate `eager-addressed-generation-v1` contract does not silently change
legacy `decode::greedy_decode` behavior. The legacy seeded refusal remains;
call the new GenerationPlan or ChatPlanner path for explicit sampled execution.

## Native execution and delivery

Compile checks exact prompt/output bounds and predicts maximum prompt plus
nonterminal-feedback positions, complete projected-logit work and sampler
payload. Native preflight additionally checks the actual engine profile, empty
logical KV and its entire capacity-based KV reservation, not merely used slots.
The host must admit the complete proposed identity; execution compares it
exactly and never repairs model/backend/task fields after admission.

The same existing eager engine executes each prompt and feedback position.
Intermediate full projections are released before subsequent forwards. EOS is
scored and included in token_ids, but has no content bytes and is not fed back.
Minimum length excludes EOS until the declared number of non-EOS tokens commit.
Exact byte-stop suffixes match across token boundaries and retain their bytes,
so streaming never retracts published content. Token and byte limits have
separate finish reasons. A byte-refused proposal is not delivered and its native
work/draw remains counted. No final unnecessary feedback forward occurs.

Cooperative checkpoints precede prompt work, selection/delivery, and feedback.
A checkpoint after sink reservation catches cancellation that arrived while
waiting for output capacity. A failed reserve/permit is fatal and never retried.
The caller owns the sink's truthful delivery contract and physical blocking I/O.
A typed cancellation is an error, not a successful partially cancelled result.
Native logical KV cleanup uses the existing ClearCache guard on normal return,
errors, cancellation and unwind, without discarding caller-owned pre-existing KV.

The sampler allowance is a bounded payload calculation, not measured RSS or
allocator overhead. Full nucleus and exact prefix decoding still allocate; no
allocation-free hot-loop or speedup claim is made. The host must account for
raw model logits, activations, tokenizer assets, output staging and safety margin
in addition to sampler payload. No new thread, runtime, model loader or pool is
created by these APIs.

## Pinned chat and independent finalization

Chat accepts an optional first system message, then alternating user/assistant
messages ending in user. It refuses malformed roles, oversized messages/history,
context overflow and task budgets above the frozen host ceiling. It does not
truncate or summarize history implicitly. Every request cold-prefills its full
transcript; no cross-request prefix cache is asserted. GenerationOptions has a
model-free, non-cloning validation entrypoint; the planner checks it before
rendering/tokenizing any message or cloning options into a sealed plan.

The trusted template renderer sees only authored placeholders and explicit role
enums. Each caller message is separately exact-byte encoded without privileged
control IDs, including literal role/thinking marker spellings. Output excludes
those control IDs except the checked terminal IM_END. This prevents role-token
smuggling at this boundary, not semantic prompt injection. Tools and thinking
are explicitly unsupported, never parsed/executed or silently downgraded.

GenerateRequest supplies one user prompt through the same pinned machinery.
Generate and chat retain distinct static task identities and compile actual
FreeText TaskIR. The prepared task retains its pinned decoder, so execution
cannot silently swap tokenizers. Finalization independently checks envelope,
termination/EOS, exact decoded bytes, full-vocabulary score fields, actual work,
seed and the complete response byte limit. Invalid/incomplete UTF-8 is a typed
no-result, never lossy replacement. A nonallocating serialization-size pass
refuses amplified responses before building their canonical JSON storage; the
original typed value still receives canonical validation and exact size checks.
Results carry untrusted assistant content, not executable tools or exposed
private execution identities.

## NDJSON integration

Use the existing `{id,text,task_args?}` input and exact `{"flush":true}` control.
`GenerationBatchArgs` has generate and chat variants. Generate uses text as the
prompt. Chat appends text as the final user message after its explicit history.
Both carry GenerationOptions, TaskBudget and a stable sample_index. Missing
arguments require host-configured defaults. Defaults and per-item arguments are
size-preflighted independently; oversized default history is rejected before
per-document cloning. The underlying boundary rejects duplicate/unknown keys.

The additive BatchRequestContext carries engine-assigned sequence, epoch and
physical input location to execution. Existing processors use the default
adapter unchanged. Generation echoes the actual transport sequence while the
stable caller ID and sample_index keep their semantic sampling roles.

The runner reserves aggregate forward/projection work before execution and never
refunds it after early stops, refusals or failed attempts. Flush does not renew
this allowance. GenerationBatchAdmission is an explicit embedding hook to the
host's real admission path, not a replacement PermitBroker or model activation
certificate. It receives the exact private identity, maximum work, sampler
payload, actual complete engine KV reservation and result-byte ceiling. It
returns the identity actually admitted and the host's real resource guard.

The native adapter rechecks admitted identity/capacity, executes the real pinned
task, requires cleaned logical KV, and transfers the successful result together
with that guard in GuardedOutput. The guard survives canonical serialization,
write_all and flush, including delivery failures; result storage is dropped
before its guard and before the next input item is executed. Errors distinguish
item-level capacity/no-result refusals from cancellation, identity mismatch or
corruption that must stop admission. No parallelism, durable jobs, automatic
model loading, interactive terminal UI or production CLI activation is added.

## Source regressions

Thirty-one added source cases cover processor order, exact/full nucleus versus
top-k, EOS and byte stops, stream failure, cancellation, transport-independent
sampled replay, work limits, pinned message encoding, role/history refusal,
request identity binding, independent finalization, actual batch request-context
propagation and compatibility with pre-existing BatchProcessor implementations.
They also cover canonical bias-key spelling, stable item/sample admission
binding, cancellation during sink reservation and output-size preflight.
They remain UNRUN. Real-model tests and empirical quality/throughput evidence
are separate requirements, not inferred from these source cases.
