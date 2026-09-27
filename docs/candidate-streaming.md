# Candidate generation and chat token streams

`fnlp candidate stream generate` and `fnlp candidate stream chat` expose the
existing native INT8 token-event path through the process-owned runtime and a
bounded, directly backpressured NDJSON writer. The ordinary `candidate generate`
and `candidate chat` commands still return one completed JSON response. Live
batch and durable-job protocols are unchanged.

This is explicit **current-candidate**, non-certified local inference. It requires
`asupersync-runtime` and a compatible local candidate INT8 `.fnlpq`; it does not
activate a release, fetch a model, enable thinking, or execute generated tools.
Without the runtime feature, validated commands refuse before input/model/output
I/O. Argument-parser usage errors retain the existing clap behavior.

## Invocation

```sh
fnlp candidate stream generate prompt.txt \
  --model ./model.fnlpq --memory-mib 8192 \
  --max-new-tokens 128 --max-output-bytes 65536 \
  --max-stream-bytes 16777216 > tokens.ndjson

fnlp candidate stream chat messages.json \
  --model ./model.fnlpq --memory-mib 8192 --logprobs \
  > chat-tokens.ndjson
```

Omit the input path or use `-` for stdin. Generate input is bounded UTF-8 text.
Chat input is the existing JSON message array: optional first system message,
then alternating user/assistant messages ending with a user. The exact original
transcript is cold-prefilled; nothing is silently truncated or normalized.

The existing seed, temperature, top-k and top-p flags work unchanged. A seed is
exactly 64 lowercase hexadecimal digits; without it decoding is greedy and
sampling controls are refused. Streaming does not alter the stable `cli` item ID,
sample index, prompt, template, effective seed, task limits or model identity.
Both buffered and streaming paths call the same preparation function.
`--logprobs` carries raw full-vocabulary log-softmax values, not processed sampler
probabilities or correctness-confidence scores.

## Wire contract

Stdout consists of canonical JSON objects, each followed by a newline, using
`protocol: "fnlp-candidate-token-stream-v1"` and envelope `schema_version: 1`.
Every frame repeats `scope: "real-artifact-current-candidate"`,
`evidence: "non_authoritative"`, `request_seq: 1`, task (`generate-v1` or
`chat-v1`) and actual public model provenance: model ID, source revision,
source-root digest, logical-model digest and quantization recipe. These are not
public prompt digests, document identifiers or model-qualification receipts.

The frame sequence is:

| Event | Meaning and `data` |
| --- | --- |
| `run_start` | `provisional: true`; declares token-event schema 2, `byte_encoding: "u8-array"`, and `completion_required: true`. |
| `token` | `provisional: true`; `data` is the existing `DecodeTokenEvent`: schema version, request sequence, zero-based token index, token ID, exact `decoded_bytes`, and optional raw logprob. |
| `run_complete` | `provisional: false`; `data` is the complete validated `Int8ChatResult`, including final text, all token IDs, finish reason, effective seed, optional scores and complete native model work. |

The completed result intentionally repeats the accumulated content once. This
lets clients reconcile delivered token events with the same complete task result
returned in the buffered response's `output` field. The outer wrappers differ.

Token bytes may split a UTF-8 character: for example, `[195]` followed by `[169]`
forms `é`. Accumulate bytes or use a strict incremental decoder; do not replace
incomplete token fragments with replacement characters. A terminal EOS token is
included in token IDs and raw scores but carries **no content bytes**. Complete
final text still passes the task's independent UTF-8 and control-token checks.
A failed final text check can therefore follow already delivered provisional
bytes; that is not successful output.

`run_start` is staged with the first token and becomes visible only when its
permit commits. A successful native byte-limit stop with no committed tokens
emits `run_start` followed directly by `run_complete`. Model-loading failure may
produce no stdout at all. A bounded token-limit or byte-limit finish is not the
same as interruption: the native task must validate it before completion.

## Completion and failure

Consumers must require a complete, newline-terminated `run_complete`, reconcile
its token IDs/bytes/scores with the preceding events, and check process exit when
available. **EOF, EOS, a partial JSON line or a count of tokens is not completion.**
A flush failure can occur even after the peer has received some or all bytes;
no transport can retroactively retract them. Nonzero process exit remains failure.

The CLI emits a content-free `run_error` object on stderr for failures after
command parsing, with a static code, exit category, optional cancellation cause
and `provisional_tokens_may_exist: true`. Stderr is not appended to the NDJSON
stdout stream. Do not merge stderr into stdout when consuming the protocol.
Abrupt process termination may prevent any error report: missing completion is
sufficient to mark a stream incomplete.

Input errors, resource admission, budgets/deadlines, cancellation, identity errors
and final task no-result retain typed nonzero categories. A final invalid task is
exit 10; time/poll/cost budget exhaustion is exit 5; ordinary user/parent/shutdown
cancellation is exit 6; resource unavailability is exit 9. Panic, writer failure,
root race-loser and unexplained completion failures are not successful cancellation
results. Nested parser, writer, model and task messages are not echoed to stderr.

The sink reserves and stages the exact event before publishing it. Cancellation
between `reserve` and `permit` drops the unpublished permit. Permits are owned,
non-cloneable and bound to the originating sink and exact event. A mismatched,
abandoned or overlapping permit cannot later produce a successful completion.
Write or flush failure permanently poisons the sink: no retry, replay or terminal
success is appended to a potentially truncated stdout prefix.

The native decoder, sampler and task finalizer are unchanged. Token events are
live, but `run_complete` is staged only after the hosted native scope physically
finishes, dispatch resolves cancellation and cleanup, and the sink verifies the
exact delivered token IDs, bytes, logprobs, seed, task and termination contract
against the validated final result. This is not just a token iterator with a
success marker emitted from the last native callback.

## Bounds and ownership

`--max-output-bytes` limits generated content; `--max-stream-bytes` separately
limits the entire serialized NDJSON stream. The CLI checks a conservative complete
transport allowance before input/model work: four encoded bytes per content byte,
4096 bytes of framing per token, a start frame and the full final-result ceiling.
Every token admission leaves terminal capacity reserved. Increasing transport
capacity never increases native token, context or sampler authority.

Token IDs, bytes and optional score verification buffers have finite declared
capacities. Canonical output staging has both a nonallocating size pass and an
actual encoded-byte check. The host reserves full resident KV, sampler/scratch,
preparation, sink/staging and complete-result allocations. These are modeled
ledger reservations, not an allocator interceptor or OS-enforced RSS guarantee.

The writer moves into the existing blocking/native invocation; no second writer
thread, unbounded channel or detached native task is created. Each permitted token
is written and flushed before generation proceeds. Slow consumers apply direct
backpressure. Blocking writes and reads are not preempted mid-operation: deadline
and cancellation checks are cooperative. This does not implement resumable token
transport or warm multi-turn KV reuse.

The embedding API is `NlpEngine::execute_int8_chat_stream`, with an existing
resident model, prepared chat plan, request sequence, `ChatStreamLimits`, owned
`DecodeEventSink` and cancellation token. Its `HostedChatStream` result exposes the
validated result and a `with_sink` callback only after physical completion. The
same sink and output memory guards remain alive during final publication. Failed
host invocations drop their owned sink after drain; earlier emitted tokens remain
provisional. Embedders must price their own sink buffers and preserve that same
terminal-completion discipline.

## Validation handoff

Added 25 model-free test definitions across host admission (3), CLI routes and
bounds (6, including one feature-disabled test), exact framing and failure state
transitions (11), and real pinned preparation/error routing (5). Transport fixtures
are explicitly not neural-success evidence. Source/API and targeted GitHub diff
review were performed; Rust compilation, Cargo tests, formatting runners,
repository scripts, Actions, DSR and real-model execution were not run here.

Controller-selected clean-SHA validation remains required. In addition to the
new test modules, the important real-model gates are buffered/streamed exact
result equivalence for greedy and seeded requests, EOS and UTF-8 splits,
slow/failed consumers, cancellation during prefill/decode and after the last token,
and absence of a terminal success on native/finalizer/runtime failure. No passing
build, measured latency, throughput improvement or completed model gate is claimed.
