# Source INT4 fixed-batch CLI

`fnlp candidate source-int4 text-batch` connects NDJSON input to shared-model
INT4 generation, pinned chat finalization and the charged host path. It runs
one finite group of 1..64 requests. This is not the INT8 text-batch corpus
protocol, a daemon, dynamic refill, or provisional token streaming. It requires
the existing `asupersync-runtime` build and an explicitly selected local model.

## Input and invocation

For generate, each line has a unique `id`, `prompt`, and optional unsigned
`sample_index` (default zero):

```json
{"id":"document-a","prompt":"Give a short greeting."}
{"id":"document-b","sample_index":7,"prompt":"Explain why leaves change color in one sentence."}
```

Run against an existing local candidate snapshot and an independently retained
model-content pin. `MODEL_SHA256` below must be that trusted 64-character
lowercase pin, not a value extracted from the snapshot being loaded.

```sh
fnlp candidate source-int4 text-batch requests.ndjson --task generate \
  --snapshot model.int4snap --expected-model-sha256 "$MODEL_SHA256" \
  --group-width 128 --memory-mib 8192 --context-tokens 512 \
  --max-new-tokens 32 --max-records 8
```

Omitting the path, or using `-`, reads stdin. To convert the pinned local source
closure, replace the snapshot/pin flags with `--source-dir DIR --source-manifest
FILE`. Optional `--save-snapshot NEW_FILE` retains create-only save behavior.
Saving happens only after the complete batch is planned and admitted; a saved
snapshot remains if later execution fails. A bad snapshot never falls back to
source conversion or another model. For chat, use `--task chat` and `messages`:

```json
{"id":"chat-a","messages":[{"role":"system","content":"Reply briefly."},{"role":"user","content":"Hello."}]}
{"id":"chat-b","messages":[{"role":"user","content":"Name a primary color."}]}
```

The existing strict parser is reused: an optional initial system message, then
alternating user/assistant messages ending in user. Duplicate JSON keys, unknown
record fields, duplicate IDs, invalid UTF-8, blank records, empty input and
records beyond the declared ceiling fail before model loading. A final nonempty
line without a newline is accepted. CR/LF bytes count toward both input limits;
oversized records are not drained without bound.

## Ordinary and packed execution

Without `--prefill-rows`, each native pass advances one prompt/feedback token
per unfinished sequence. Short prompts can decode while longer prompts are
still prefilling. Only final-prompt or feedback positions project the vocabulary
head. Finished rows are not advanced, padded or sampled again.

`--prefill-rows N` now selects bounded token packing for this command. N is the
maximum TOTAL prompt/decode token rows in one layer-major pass, not tokens per
document and not the number of admitted documents. It accepts 1..64, including
values smaller than the cohort. For example:

```sh
fnlp candidate source-int4 text-batch requests.ndjson --task generate \
  --snapshot model.int4snap --expected-model-sha256 "$MODEL_SHA256" \
  --group-width 128 --memory-mib 8192 --context-tokens 512 \
  --max-new-tokens 32 --max-records 8 --prefill-rows 32
```

The packed scheduler uses the same data-only round-robin allocator as INT8
packed generation. It rotates a fairness cursor even when N is smaller than
the number of documents. Prompt rows may contribute consecutive tokens; a
pending decode row advertises only ONE token, so unsampled feedback is never
executed speculatively. There is no whole-prompt barrier. Linear operators
share packed INT4 weight groups across all admitted token rows; each attention
row appends one token to its own cache and then reads only that causal prefix.
The 44 independent KV slots, both loop passes and both final norms are unchanged.

Shared generation flags (`--seed`, processors, penalties, biases, stops,
`--logprobs`, output/token limits) still apply independently to each record.
The exact ID and sample index reach the compiler. Row order, token-pack index
and delivery `request_seq` never replace the semantic sampling key. No second
sampler, numerical profile, kernel or worker implementation is added. Use
`--kernel ID` for an explicit hardware-checked integer leaf, or omit it for
scalar. The single-request `generate`/`chat` prompt-prefill behavior is unchanged;
classification still rejects `--prefill-rows`.

## Complete output, not successful prefixes

The command performs one shared-model hosted invocation. The complete native
pack's routing, positions, row counts and logits are checked before any token
from that pack is delivered. The task layer calls the existing independent
single-request finalizer for every completed result: token re-decoding, UTF-8,
control exclusion, stopping/sample semantics, work and wrapper limits remain
mandatory. It reconciles aggregate work and checks physical pass counts too.
Packed pass counts are replayed from prompt lengths, actual forward counts and
the selected width, rather than trusted or confused with logical token work.

Stdout is one canonical completed JSON object plus newline. Its protocol stays
`fnlp-source-int4-text-batch-v1`, with `schema_version: 1` and
`evidence: non_authoritative`. Model/source identity, group width, actual kernel
and optional snapshot-operation metadata are retained; paths and private prompt
digests are not added. `items[i]` maps the original ID, sample index and delivery
sequence to `output.results[i]` in input order.

`output.execution` is `grouped-int4-ragged-generation-v1` for ordinary batches
and `grouped-int4-ragged-token-morsel-v1` for packed batches. Per-request semantic
execution labels and identities are unchanged. Physical `group_steps` can differ;
that is not a benchmark or numerical-parity receipt. The complete object is
staged before its first write, and result/preparation guards remain held through
serialization, writing and flushing. A later-row failure returns no successful
prefix. Broken transport still exits nonzero; already-written bytes cannot be
retracted. Provisional token streaming is available only through lower library
APIs, not this CLI command.

## Admission and ownership

`--max-records` defaults to 8 and accepts 1..64. It bounds the whole invocation,
not a window. Storage is conservatively admitted for that declared ceiling even
when fewer records arrive. `--context-tokens` is allocated capacity per document.
`--max-input-bytes` bounds a complete NDJSON line; `--max-total-input-bytes`
defaults to 1 MiB and caps the complete transport (maximum 64 MiB).

Admission counts every declared KV capacity/sampler, shared RoPE and native
workspace, retained raw generation payload, final aggregate output, publication
staging, and explicit preparation/allocator reserves. Packed mode additionally
reserves ONE total-token workspace from `Int4PrefillLimits`, independent of the
number of documents. These rails coexist with ordinary cohort storage; they are
not subtracted from it, multiplied by the document count, or left uncharged.
The phase floor uses the maximum load/save/execution occupancy plus always-live
preparation/resident weights/cleanup. These are modeled commitments, not measured
or operating-system-enforced RSS limits.

Every actual compiled request keeps its own context, forward, attention and
group-dot/MAC ceiling. The host repeats packed or ordinary requirements before
allocation and before optional saving. `--max-checkpoints` covers the ONE entire
native invocation; no pack or row renews it. `--timeout-seconds` covers input,
loading, planning, saving and execution through remaining-time checks. Blocking
IO is cooperative, not preemptible. A stale programmatic scheduling quote is
rejected before IO, including when the load phase dominates the memory floor.

## Library entrypoints

`Int4CohortSession::append_packed` accepts ordered `Int4TokenRun` values and
`Int4PrefillLimits`; it uses the existing native executor and ordinary session
poisoning/Drop cleanup. Extra packed rails are explicitly caller-admitted.
`generation::grouped_int4::batch::packed` supplies requirements, preflight and
execution, including provisional streaming. `tasks::chat::grouped_int4::batch`
adds `requirements_packed`, `execute_packed` and `execute_packed_with_sink` with
the same pinned decoder and independent finalizers.

The charged embedding API is `NlpEngine::execute_int4_chat_batch_packed`, with
`Int4BatchHostLimits::requirements_packed` for a pure quote. Model/domain checks,
all component reservations and the single supervised blocking invocation are
shared with the ordinary method. The model, plans and guards remain owned until
physical completion; temporary native buffers drain before aggregate publication.

## Validation status and remaining work

The packed extension adds 18 Rust regression-test functions across native causal
attention/address checks, generation scheduling, pinned task finalization, host
quotes and CLI admission. They cover multiple tokens per sequence, both loop
slot boundaries, future-token/document isolation, widths smaller/larger than
membership, seeded permutation, ordinary-versus-packed cursor results under
synthetic logits, byte-stop feedback, malformed native output, cancellation,
per-member work, exact extra workspace, schedule replay and stale-quote refusal.
Existing ordinary-batch and single-request fixtures are retained.

An independent Python model passed 1,344 scheduling cases, checking 124,138 head
placements and 1,115,772 forwarded positions against a separate ready-ring
allocator, plus 3,600 declared/actual memory shapes. These auxiliary arithmetic
checks do NOT execute Rust, the real sampler, an ISA or a model. Rust compilation,
Rust tests, real-model packed-versus-ordinary parity, production cancellation
interleavings and performance benchmarks have not run. Controller-owned DSR and
model/task qualification remain outstanding; no speedup or parity is asserted.

Automatic corpus windows, dynamic refill, per-record generation overrides,
dedicated SIGINT integration for this route and heterogeneous grammar scheduling
remain separate work. No default execution strategy or kernel preference changes.
