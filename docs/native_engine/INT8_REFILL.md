# Bounded slot-refilling INT8 text execution

Status: source implementation. Rust compilation and regression execution, DSR,
real-model scalar/refill parity, physical memory measurement and throughput
qualification are UNRUN. This is an explicit current-candidate path, not a
production/profile/artifact activation or a measured performance award.

## Candidate CLI

With an asupersync-runtime build and an existing local candidate artifact:

```sh
fnlp candidate text-batch prompts.ndjson --task generate \
  --model local.fnlpq --memory-mib 8192 --preparation-mib 512 \
  --cohort-rows 16 --active-rows 4 --prefill-rows 16
```

The widths are illustrative, not benchmark-selected recommendations. The same
options work with `--task chat` and the existing alternating messages input.

`--cohort-rows` bounds the preplanned input window (1..=64 requests).
`--active-rows` explicitly selects FIFO refill and limits simultaneous KV/sampler
slots (1..=cohort-rows). `--prefill-rows` bounds TOTAL prompt/decode token rows per
native pack (1..=64), not tokens per request. In the example, sixteen requests
can be queued, four can be live, and one decoder pack can contain sixteen tokens.

A completed request frees its slot for the next queued request while another
request can still be prefilling or generating. Queue order is FIFO, vacant slots
are reused in ascending order, and the existing rotating token allocator is fair
even when token width is smaller than live-slot count. Decode contributes only
one feedback token per pack; prompt positions can fill the remaining capacity.

When active-rows is selected without prefill-rows, token width is the actual
live-slot ceiling. A final window shorter than active-rows uses its actual request
count. Explicit prefill-rows remains unchanged for that tail. Invalid geometry is
rejected before input/model IO; native or memory failure never shrinks or retries
an admitted epoch. Other candidate commands do not accept active-rows.

Without active-rows, every existing serial, single-document prefill, ordinary
cohort and packed fixed-cohort mode remains unchanged. The fixed-cohort APIs
and their contracts are described in `INT8_COHORTS.md`.

## Corpus and delivery semantics

All records in a bounded window are parsed, planned and charged to the SAME
whole-corpus ledger before native execution. Input is not read during native
execution. This removes the fixed native membership barrier inside a window;
it does not implement an unbounded arrival queue, socket service or daemon.

Item IDs, sample indices, exact prompts, generation policy and addressed random
draws are unchanged. Physical slot numbers, refill order and pack counts never
seed a request. The ordinary cursor owns stops, penalties, EOS, byte-refused
proposals and full-vocabulary raw logprobs. KV is cleared across all 44 logical
slots before a physical slot is reused. No cross-request prefix cache is used.

Results are retained by INPUT index, not completion order or delivery-number
sort. EVERY request must complete and pass the existing pinned text/UTF-8/control
finalizer before the first result of that window is published. The returned
output guard survives ordered canonical serialization, each write and flush.
The existing per-record NDJSON schema and mandatory batch_complete frame are
unchanged. Successful process exit is also required for whole-corpus success.

A native, cancellation, decoder or task-finalization failure aborts the entire
current window with no terminal sibling results and no retry. Earlier windows
remain valid. Output transport can fail after valid frames were already written;
no completion frame follows an uncertain write or failed flush. Lower-level task
and native event sinks deliver provisional events, never terminal success.

## Resource and work ownership

All request budgets are copied into one immutable finite native epoch before it
opens. A request allowance can be installed exactly once. Retirement preserves
its consumed work in the epoch ledger and discards its unused per-request quota;
it does not renew a sibling's work or give a new request the old allowance.
Completed per-request work and aggregate projection debits are reconciled before
slot reuse and at final completion. Any incomplete native epoch dropped by its
caller poisons and drains the engine through the existing RAII cleanup.

Each recyclable slot reserves the longest planned context in the window, since
completion order is not known in advance. Every queued task must permit that
complete physical KV capacity. This conservative uniform-capacity rule can refuse
a heterogeneous window even when some shorter tasks would fit individually;
there is no capacity-aware rerouting or automatic fallback. Shared weights and
RoPE remain single bindings; native workspace is priced by live slots plus the
separately declared token-pack scratch. Simultaneous sampler admission is the
largest queued sampler workspace times live-slot count.

Preparation and retained result storage still cover ALL queued requests. Smaller
active-rows cannot underprice input/tokenizer graphs, whole-window output or
transport framing. Allocator headroom and cleanup reserves remain separate.
These are modeled process-resource reservations, not measured OS RSS guarantees.

One hosted invocation owns one deadline/checkpoint budget across all refills.
CLI max-checkpoints is per complete window, never replenished per slot. The
invocation-wide timeout still covers the entire corpus including IO, planning,
model loading and delivery. Whole-corpus work/input/output ceilings never reset.

## Library path and receipts

`NlpEngine::execute_int8_chat_refilling` accepts the resident model, owned prepared
plans, first request sequence, ChatCohortLimits, active-sequence count,
Int8PrefillLimits and cancellation token. It returns the existing guarded
Int8ChatCohortResult in input order after physical drain.

The path is `tasks::chat::quantized::cohort::packed::refill` ->
`generation::quantized::cohort::packed::refill` ->
`strict_int8::cohort::packed::refill::Int8RefillSession`. It reuses the actual
packed INT8 projections, causal attention and original generation cursor.
No alternate tokenizer, template, sampler, runtime or worker team is introduced.

The physical wrapper identifies `portable-int8-refilling-token-morsel-epoch-v1`;
per-request semantic execution labels remain unchanged. `group_steps` counts
native packs, not logical forward positions. Generation and task finalization
both reconstruct the expected FIFO-refill schedule from immutable prompt lengths,
completed forward counts, active slots and token width before accepting that
count. Ordinary fixed-cohort finalizers do not accept a relabeled refill receipt.

## Validation boundary

Added Rust regression definitions cover immutable ticket/slot ownership, all-44
KV reset, cumulative work and overflow, early completion and refill, scalar versus
refill seeded cursor results, byte-refused work, routing and malformed logits,
cancellation and uncertain delivery, pinned finalization, live/queued memory
separation, CLI width/tail/default behavior, pre-IO refusal, bounded read-ahead,
unchanged identities and complete queued-output admission. These remain UNRUN.

An independently executed Python schedule model matched a separate deque/ring
simulation in 119,408 exhaustive/randomized cases (seed 20261005), comparing exact
per-pack request allocations. This narrow scheduling check is not Rust execution,
native-model parity, DSR or a throughput benchmark.
