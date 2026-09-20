# Native INT8 generation and chat integration

The library now has a statically typed raw-text route into the existing
`strict-quantized-v1` model driver:

```text
GenerateRequest / ChatRequest
  -> Int8ChatPlanner (pinned template and control-excluding tokenizer)
  -> PreparedInt8Chat (exact identity, task budget, native plan)
  -> actual StrictInt8Engine
  -> independent completion checks
  -> Int8ChatResult (ChatResult plus complete Int8Work)
```

This is a source-level integration, not a build/parity/performance receipt.
It does not activate the public model-backed CLI or bypass the artifact,
runtime, profile-ratification or model-present evidence gates. The existing
`CurrentCandidateInt8Model` remains explicitly a non-authoritative rehearsal
loader. No BF16 identity is rewritten to obtain a quantized execution plan.

## Single request and streaming

`tasks::chat::quantized::Int8ChatPlanner::pinned` takes the same pinned control
registry, EOS, host task ceiling and chat limits as the eager planner. Its
supplied identity must already name `strict-quantized-v1`, BF16 KV and the exact
`STRICT_INT8_EXECUTION` backend, with thinking disabled and no tools.

`plan_generate` and `plan_chat` accept the existing request types. Multi-turn
history is prefetched in full; malformed roles or oversized history are
refused, not repaired. Caller content is encoded separately from the authored
role scaffold. Literal role/thinking marker spellings in content do not grant
control-token authority. The effective generated-token policy bans all such
controls except the configured terminal EOS.

The prepared plan exposes its exact `ExecutionIdentity`, `TaskPlan`, native
plan and full `Int8Work` ceiling before admission. `preflight` checks the
identity actually admitted against the plan AND the materialized native model.
`execute` and `execute_with_sink` then use the existing native generation
cursor, sampler and exclusively borrowed INT8 session.

Streaming token events are provisional. Consumers need a final successful task
result; bytes already delivered cannot be retracted after a later error. The
finalizer independently checks token/byte agreement, EOS/stop policy, exact
UTF-8, profile/version, effective seed, full-vocabulary raw logprobs and work
counts. It enforces the complete outer result size, including `model_work`,
not only the assistant text or nested ChatResult. It exports no prompt digest
or private sampling key. Addressed sampling keeps delivery sequence separate
from the stable caller ID and sample index.

## Resident-engine NDJSON integration

`batch::generation::quantized::NativeInt8GenerationBatch` implements the
existing `BatchProcessor` trait. It owns no model loader, scheduler or thread
pool and accepts only a borrowed real `StrictInt8Engine`. One planner and one
engine are reused across drained documents. Each request cold-prefills its
history; logical KV is cleared between requests, not reused as hidden history.

The existing `GenerationBatchArgs` wire type is shared: `mode=generate` uses
`text` as the prompt, while `mode=chat` appends `text` as the final User message
after the supplied history. Overrides replace defaults for that item only.
The runner still owns strict NDJSON framing, duplicate-ID epochs, ordered
output, cooperative checkpoints and output poisoning.

The host supplies a real `Int8GenerationBatchAdmission` implementation. Its
request includes the whole identity, complete decoder/attention work, sampler
payload, complete engine KV capacity and complete result limit. Its returned
resource guard survives both inference and the final batch event's write and
flush. There is no default fake certificate or no-op admission implementation.

This wiring function is intended to run inside the embedding host's already
admitted runtime/CPU scope; the host must also account for weights, RoPE,
activation/logit scratch, allocator margin and transport-envelope staging:

```rust
use std::io::{BufRead, Write};
use franken_nlp::{
    batch::{self, BatchFault, BatchLimits, BatchRunError, BatchSummary,
        generation::{GenerationBatchArgs, quantized::{Int8BatchLimits,
            Int8GenerationBatchAdmission, Int8GenerationBatchPlanner,
            NativeInt8GenerationBatch}}},
    native_engine::{decode::DecodeStepControl, strict_int8::StrictInt8Engine},
    tasks::chat::quantized::Int8ChatPlanner,
};

fn run_admitted_int8<A, R, W, C>(
    planner: &Int8ChatPlanner,
    engine: &mut StrictInt8Engine<'_>,
    admission: A,
    defaults: GenerationBatchArgs,
    native_limits: Int8BatchLimits,
    transport_limits: BatchLimits,
    input: &mut R,
    output: &mut W,
    control: &mut C,
) -> Result<BatchSummary, BatchRunError>
where
    A: Int8GenerationBatchAdmission,
    R: BufRead,
    W: Write,
    C: DecodeStepControl,
{
    fn setup_error(fault: BatchFault) -> BatchRunError {
        BatchRunError { fault, summary: BatchSummary::default() }
    }
    let compiler = Int8GenerationBatchPlanner::new(planner, Some(defaults))
        .map_err(setup_error)?;
    let mut processor = NativeInt8GenerationBatch::new(
        compiler, engine, admission, native_limits,
    ).map_err(setup_error)?;
    batch::run_ndjson(input, output, &mut processor, transport_limits, control)
}
```

## Whole-run budgets and failures

`Int8BatchLimits.max_model_work` separately bounds forward positions, projected
logits, attention pairs, projection dot products and multiply-accumulates.
Every checked addition is nonrefundable. An admission refusal or early native
finish does not restore the attempted request's conservative reservation.
Protocol flush does not reset these limits. The runner independently enforces
its forward/logit ceilings and all input/output/record limits.

A fallible native attempt raises the adapter's failure latch before admission.
An unwind leaves it set. Cancellation retains its exact cause. A poisoned or
nonempty native engine cannot become a recoverable per-item error; no retry
is performed. Soft preflight/admission or final-text refusal can continue only
when the driver is clean and quiescent. Failed output is terminal to the NDJSON
runner, and its guard is released only after synchronous native work returned.

Tests use the real pinned tokenizer/compiler and a private synthetic driver
for lifecycle and output-failure injection. The public constructor cannot take
that driver. These are not substitutes for controller-owned Rust validation,
real-model smoke or quantized quality measurements.
