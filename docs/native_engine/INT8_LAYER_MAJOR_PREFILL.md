# Opt-in layer-major INT8 prompt execution

This is source implementation, not a native-model parity or performance award.
Compilation, regression execution, real-model comparison and target-host timing
are unrun. The ordinary sequential execution APIs and CLI defaults are unchanged.

## Executable path

The new path is `NlpEngine::execute_int8_chat_layer_major` ->
`PreparedInt8Chat::execute_layer_major` ->
`Int8GenerationPlan::execute_layer_major` ->
`Int8Session::append_layer_major` -> the existing `LoopRunner::run_group`.
It uses the same pinned prompt compiler, tokenizer, sampler, task finalizer,
resident candidate model, asupersync host, cancellation and output ownership.
It neither loads a second model nor creates a separate runtime or worker team.

The prompt is split into explicitly bounded morsels of 1..=64 rows. Every
morsel executes 22 layers, the final RMSNorm, 22 layers, and the final RMSNorm.
Q/K/V, output projection and MLP operations use shared-weight INT8 projections.
Each weight value feeds up to four exact independent i32 accumulators. The
fixed per-row dequantization and BF16 casts remain the scalar program.

Attention deliberately does NOT preappend the entire morsel. At each layer,
row i's K/V is appended immediately before row i attends to that prefix. RoPE
uses the absolute source position, including an existing prefix and the second
loop. Intermediate prompt/morsel positions never project the lm-head. The
last prompt position projects once; generation then uses ordinary token steps.

## Explicit candidate CLI

A build with `asupersync-runtime` and an existing local candidate artifact may
select `--prefill-rows ROWS` for buffered generate/chat, live token streaming,
or each independent record in text-batch. Omitting the switch preserves the
sequential path. Supplying 1 explicitly exercises the layer-major implementation
with one row. Row counts outside 1..=64 are refused before input or model IO.
The example width is illustrative, not a benchmark-selected recommendation.

```sh
fnlp candidate generate prompt.txt --model local.fnlpq --memory-mib 8192 --prefill-rows 8
fnlp candidate stream generate prompt.txt --model local.fnlpq --memory-mib 8192 --prefill-rows 8
fnlp candidate text-batch prompts.ndjson --task generate --model local.fnlpq --memory-mib 8192 --prefill-rows 8
```

Chat uses the same switch with its existing JSON message-array input. Text-batch
retains one resident model and sequential independent records; the new switch
batches prompt positions WITHIN each record, not documents. Generation controls,
seed addressing, corpus budgets, provisional token frames and required completion
frames are unchanged. The switch is not offered to structured/scored commands
whose execution paths have not been connected to this strategy.

The CLI derives extra scratch from the selected row ceiling; callers cannot
supply a smaller charge. The host must admit that memory in addition to the
ordinary model, KV, native workspace, preparation, sampler and output charges.
An admission or grouped-execution failure does not trigger a hidden sequential
retry. No benchmark win, model fidelity or release activation is inferred.

## Hosted use

Given an existing host, resident model, prepared chat/generate plan and native
limits, select the candidate explicitly:

```rust,ignore
use franken_nlp::native_engine::strict_int8::prefill::Int8PrefillLimits;

let rows = 8;
let prefill = Int8PrefillLimits {
    max_batch_rows: rows,
    max_extra_scratch_bytes:
        Int8PrefillLimits::required_extra_scratch_bytes(rows)?,
};
let output = engine.execute_int8_chat_layer_major(
    &model, prepared, request_seq, native_limits,
    32 * 1024 * 1024, prefill, cancellation,
)?;
// Retain output through serialization, external write and flush.
```

The host adds the derived extra scratch to its real process-memory reservation
BEFORE dispatch and retains the charge until physical native cleanup. Direct
session/plan/task callers must supply equivalent admission themselves; the
numeric limits are not permits and are not an OS RSS guarantee. Task-level
`execute_layer_major_with_sink` additionally supports provisional token events
under the existing reserve/permit contract. Hosted streaming is available through
`NlpEngine::execute_int8_chat_stream_layer_major`; its sink and preparation
charges survive until final publication, independently of native scratch.

## Invariants and limits

The exact prompt, item/sample identity, seed, generation options and semantic
execution identity are unchanged. Grouping cannot consume extra random draws.
The complete model-work count must still equal the shared cursor's actual
forward/head count, including a byte-refused proposal. The strict-profile
contract REQUIRES identical per-row output; the new regression definitions are
not a substitute for running the real-model gate.

Whole-prompt input/context/work and extra scratch are checked before mutation.
Any failure after native work starts poisons the session; RAII clears all 44
logical KV slots. A failed prompt emits no token. Later token events remain
provisional until independent task finalization and host drain succeed.

This is single-sequence prompt batching, not cross-document decode batching,
a parallel scheduler, SIMD dispatch, qualified artifact activation or a speed
claim. Shared-weight integer projections and causal-attention fixtures provide
code paths for subsequent model-present and hardware qualification.
