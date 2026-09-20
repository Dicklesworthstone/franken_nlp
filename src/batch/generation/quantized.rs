//! Resident strict-INT8 generation/chat through the existing ordered runner.
//!
//! Public execution accepts only the real native engine. No BF16 fallback,
//! model loader, worker pool, retry or invented resource guard is introduced.
//! The run's complete INT8 work ceiling never resets at a protocol flush.

use super::{GenerationBatchArgs, GuardedOutput, check_argument_size};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext, BatchWork},
    execution_identity::ExecutionIdentity,
    native_engine::{decode::DecodeStepControl, generation::GenerationError,
        generation::quantized::{Int8GenerationBudget, Int8GenerationError}, kv::KV_BYTES_PER_TOKEN,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error}},
    tasks::chat::{ChatError, ChatMessage, ChatRequest, ChatRole, GenerateRequest,
        quantized::{Int8ChatError, Int8ChatPlanner, Int8ChatResult, PreparedInt8Chat}},
};

pub struct Int8GenerationBatchPlanner<'a> {
    planner: &'a Int8ChatPlanner,
    defaults: Option<GenerationBatchArgs>,
}
impl<'a> Int8GenerationBatchPlanner<'a> {
    pub fn new(planner: &'a Int8ChatPlanner, defaults: Option<GenerationBatchArgs>) -> Result<Self, BatchFault> {
        if let Some(args) = &defaults { check_argument_size(args).map_err(|_| BatchCode::InvalidLimits)?; }
        Ok(Self { planner, defaults })
    }
    /// No native work or admission side effect. Each document owns its exact
    /// options; a per-item override does not mutate subsequent run defaults.
    pub fn prepare(&self, document: BatchDocument<GenerationBatchArgs>) -> Result<PreparedInt8Chat, BatchItemFailure> {
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        check_argument_size(&args).map_err(|_| BatchItemFailure::reject(BatchCode::Planning))?;
        let result = match args {
            GenerationBatchArgs::Generate { generation, budget, sample_index } => self.planner.plan_generate(&GenerateRequest {
                item_id: document.id, sample_index, prompt: document.text, generation, budget,
            }),
            GenerationBatchArgs::Chat { mut history, generation, budget, sample_index } => {
                if history.len() >= 128 { return Err(BatchItemFailure::reject(BatchCode::Planning)); }
                history.try_reserve_exact(1).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
                history.push(ChatMessage { role: ChatRole::User, content: document.text });
                self.planner.plan_chat(&ChatRequest { item_id: document.id, sample_index, messages: history, generation, budget })
            }
        };
        result.map_err(planning_failure)
    }
}

/// Host-private request admission. Transport work counts are a projection of
/// model_work, not a replacement for its attention and decoder-MAC bounds.
pub struct Int8GenerationAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int8Work,
    pub sampler_bytes: u64,
    pub kv_reservation_bytes: u64,
    /// Complete Int8ChatResult, excluding the runner's separately admitted
    /// batch envelope/staging. Neither field is an observed RSS measurement.
    pub max_result_bytes: u64,
}
pub trait Int8GenerationBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8GenerationAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

#[derive(Clone, Copy, Debug)]
pub struct Int8BatchLimits {
    pub max_sampler_bytes: u64,
    /// Whole-run conservative charges, including failed attempts. This is in
    /// addition to BatchLimits.max_work and its transport/input/output bounds.
    pub max_model_work: Int8Work,
}
impl Int8BatchLimits {
    fn validate(self) -> Result<(), BatchFault> {
        let w = self.max_model_work;
        if self.max_sampler_bytes == 0 || w.forward_positions == 0 || w.projected_logits == 0
            || w.attention_pairs == 0 || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

pub struct NativeInt8GenerationBatch<'p, 'e, 'weights, A: Int8GenerationBatchAdmission> {
    compiler: Int8GenerationBatchPlanner<'p>,
    engine: &'e mut StrictInt8Engine<'weights>,
    admission: A,
    limits: Int8BatchLimits,
    run: RunState,
}
impl<'p, 'e, 'weights, A: Int8GenerationBatchAdmission> NativeInt8GenerationBatch<'p, 'e, 'weights, A> {
    pub fn new(compiler: Int8GenerationBatchPlanner<'p>, engine: &'e mut StrictInt8Engine<'weights>,
        admission: A, limits: Int8BatchLimits) -> Result<Self, BatchFault> {
        limits.validate()?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) { return Err(BatchCode::InvalidExecution.into()); }
        Ok(Self { compiler, engine, admission, limits, run: RunState::new(limits.max_model_work) })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.run.reserved }
    pub fn is_poisoned(&self) -> bool { self.run.failed || self.engine.is_poisoned() }
}
impl<A: Int8GenerationBatchAdmission> BatchProcessor for NativeInt8GenerationBatch<'_, '_, '_, A> {
    type Args = GenerationBatchArgs;
    type Prepared = PreparedInt8Chat;
    type Output = GuardedOutput<Int8ChatResult, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?;
        let result = (|| {
            let prepared = self.compiler.prepare(document)?;
            prepared.preflight(prepared.execution_identity(), self.engine, budget(&prepared, self.limits.max_sampler_bytes))
                .map_err(execution_failure)?;
            Ok(prepared)
        })();
        if result.as_ref().is_err_and(|error: &BatchItemFailure| error.stop) { self.run.failed = true; }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(prepared, self.engine, &mut self.admission, &mut self.run,
            self.limits.max_sampler_bytes, context, control)
    }
}

fn budget(prepared: &PreparedInt8Chat, max_sampler_bytes: u64) -> Int8GenerationBudget {
    Int8GenerationBudget { native: Int8RunBudget::exact(prepared.planned_work()),
        max_kv_bytes: prepared.task_plan().ir().budget().max_kv_bytes, max_sampler_bytes }
}

// One nonrenewable work ledger. No reset/refund method or external mutation.
// Latch before fallible admission/native work; panic leaves it permanently set.
struct RunState { limit: Int8Work, reserved: Int8Work, last_sequence: u64, failed: bool }
impl RunState {
    fn new(limit: Int8Work) -> Self { Self { limit, reserved: Int8Work::default(), last_sequence: 0, failed: false } }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: Int8Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?;
        self.failed = true;
        if context.request_seq <= self.last_sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let sum = add_work(self.reserved, work).filter(|sum| fits(*sum, self.limit))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.reserved = sum; self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn fits(work: Int8Work, limit: Int8Work) -> bool {
    work.forward_positions <= limit.forward_positions && work.projected_logits <= limit.projected_logits
        && work.attention_pairs <= limit.attention_pairs && work.projections.fits(limit.projections)
}
fn add_work(a: Int8Work, b: Int8Work) -> Option<Int8Work> {
    Some(Int8Work { forward_positions: a.forward_positions.checked_add(b.forward_positions)?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits)?,
        attention_pairs: a.attention_pairs.checked_add(b.attention_pairs)?,
        projections: a.projections.checked_add(b.projections).ok()? })
}

/// Private test seam. The public adapter's constructor accepts ONLY a borrowed
/// StrictInt8Engine, so a fake driver's receipts cannot enter product execution.
trait Driver {
    fn capacity(&self) -> usize;
    fn empty(&self) -> bool;
    fn poisoned(&self) -> bool;
    fn preflight(&self, prepared: &PreparedInt8Chat, identity: &ExecutionIdentity, budget: Int8GenerationBudget)
        -> Result<(), Int8ChatError>;
    fn run<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8Chat, identity: &ExecutionIdentity,
        seq: u64, budget: Int8GenerationBudget, control: &mut C) -> Result<Int8ChatResult, Int8ChatError>;
}
impl Driver for StrictInt8Engine<'_> {
    fn capacity(&self) -> usize { self.kv_cache().capacity_positions() }
    fn empty(&self) -> bool { self.kv_cache().all_slots_have_len(0) }
    fn poisoned(&self) -> bool { self.is_poisoned() }
    fn preflight(&self, prepared: &PreparedInt8Chat, identity: &ExecutionIdentity, budget: Int8GenerationBudget)
        -> Result<(), Int8ChatError> { prepared.preflight(identity, self, budget) }
    fn run<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8Chat, identity: &ExecutionIdentity,
        seq: u64, budget: Int8GenerationBudget, control: &mut C) -> Result<Int8ChatResult, Int8ChatError> {
        prepared.execute(identity, self, seq, budget, control)
    }
}

fn execute_admitted<D: Driver, A: Int8GenerationBatchAdmission, C: DecodeStepControl>(
    prepared: PreparedInt8Chat, engine: &mut D, admission: &mut A, run: &mut RunState,
    max_sampler_bytes: u64, context: BatchRequestContext, control: &mut C,
) -> Result<GuardedOutput<Int8ChatResult, A::Guard>, BatchItemFailure> {
    run.ready()?;
    let work = prepared.planned_work();
    // Charge before admission (including its failures), not after successful
    // inference. The outer runner independently charged its transport ceiling.
    run.begin(work, context)?;
    let outcome = (|| {
        if let Some(cause) = control.checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        let budget = budget(&prepared, max_sampler_bytes);
        engine.preflight(&prepared, prepared.execution_identity(), budget).map_err(execution_failure)?;
        let kv_reservation_bytes = (engine.capacity() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
        let (identity, guard) = admission.admit(Int8GenerationAdmission {
            identity: prepared.execution_identity(), model_work: work,
            sampler_bytes: prepared.native_plan().sampler_bytes(), kv_reservation_bytes,
            max_result_bytes: prepared.task_plan().ir().budget().max_output_bytes,
        })?;
        // Compare the whole admitted identity before any physical forward.
        prepared.verify_identity(&identity).map_err(execution_failure)?;
        engine.preflight(&prepared, &identity, budget).map_err(execution_failure)?;
        let result = engine.run(&prepared, &identity, context.request_seq, budget, control).map_err(execution_failure)?;
        if !valid_output(&prepared, &result, context.request_seq, work) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        // Guard is owned through the canonical batch event's write AND flush.
        Ok(GuardedOutput::new(result, guard))
    })();
    let reusable = engine.empty() && !engine.poisoned();
    match outcome {
        Ok(output) if reusable => { run.failed = false; Ok(output) }
        Ok(_) => Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
        Err(mut failure) => {
            // Preserve typed cancellation even though native cancellation also
            // poisons the engine. Never convert a poisoned engine to soft retry.
            if !reusable { failure.stop = true; }
            if !failure.stop { run.failed = false; }
            Err(failure)
        }
    }
}

fn valid_output(prepared: &PreparedInt8Chat, output: &Int8ChatResult, seq: u64, ceiling: Int8Work) -> bool {
    let result = &output.result;
    let expected = usize::try_from(result.native_work.forward_positions).ok()
        .zip(usize::try_from(result.native_work.projected_logits).ok())
        .and_then(|(positions, rows)| Int8Work::for_sequence(0, positions, rows).ok());
    output.schema_version == 1 && result.schema_version == 1 && result.request_seq == seq
        && result.task == prepared.task_plan().task_spec_identity()
        && result.execution == crate::native_engine::generation::quantized::INT8_GENERATION_VERSION
        && result.numerics_profile == crate::native_engine::strict_int8::STRICT_INT8_PROFILE
        && expected == Some(output.model_work) && fits(output.model_work, ceiling)
}

fn planning_failure(error: Int8ChatError) -> BatchItemFailure {
    match error {
        Int8ChatError::Chat(error) => super::planning_failure(error),
        Int8ChatError::Native(Int8GenerationError::Generation(GenerationError::Allocation)
            | Int8GenerationError::Native(StrictInt8Error::Allocation)) => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8ChatError::Native(Int8GenerationError::ModelIdentity
            | Int8GenerationError::Generation(GenerationError::Identity)) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::reject(BatchCode::Planning),
    }
}
fn execution_failure(error: Int8ChatError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8ChatError::Chat(error) => super::execution_failure(error),
        Int8ChatError::Native(Int8GenerationError::Generation(error)) => super::execution_failure(ChatError::Native(error)),
        Int8ChatError::Native(Int8GenerationError::Native(StrictInt8Error::Context | StrictInt8Error::Memory)) =>
            BatchItemFailure::reject(BatchCode::Admission),
        Int8ChatError::Native(Int8GenerationError::Native(StrictInt8Error::Work)) => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int8ChatError::Native(Int8GenerationError::Native(StrictInt8Error::Allocation)) => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8ChatError::Native(Int8GenerationError::ModelIdentity) => BatchItemFailure::fatal(BatchCode::Admission),
        Int8ChatError::Native(Int8GenerationError::Native(_)) => BatchItemFailure::fatal(BatchCode::Execution),
        Int8ChatError::Native(Int8GenerationError::WorkMismatch) | Int8ChatError::WorkMismatch =>
            BatchItemFailure::fatal(BatchCode::InvalidExecution),
    }
}

#[cfg(test)] mod tests;
