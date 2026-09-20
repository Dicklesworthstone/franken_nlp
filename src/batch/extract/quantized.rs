//! Raw-schema/source extraction on the resident native INT8 NDJSON path.
//!
//! The existing task compiler and constrained decoder own grammar validity,
//! exact decimal JSON and source-occurrence evidence. This adapter owns only
//! corpus admission/lifetime: no new sampler, model loader, runtime or retry.
//! Artifact activation and model-quality certification remain host obligations.

use super::*;
use crate::{
    native_engine::{
        constrained_int8::{self, Int8JsonBudget, Int8JsonError},
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error, STRICT_INT8_PROFILE},
    },
    tasks::extract::quantized::{Int8ExtractError, Int8ExtractPlan, Int8ExtractRun, INT8_EXTRACT_VERSION},
};

/// Owns the shared prompt compiler privately; no eager planner/plan conversion
/// or Deref can route a quantized identity through BF16 execution.
pub struct Int8ExtractionBatchPlanner { compiler: ExtractionBatchPlanner }
pub struct PreparedInt8BatchExtraction {
    plan: Int8ExtractPlan,
    task: TaskPlan,
    source: SourceDocument,
}
impl Int8ExtractionBatchPlanner {
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity,
        ceiling: TaskBudget, compiler_limits: CompileLimits, source_limits: SourceRuntimeLimits,
        defaults: Option<ExtractionBatchArgs>) -> Result<Self, BatchFault> {
        Ok(Self { compiler: ExtractionBatchPlanner::pinned_for_backend(controls, eos, identity,
            ceiling, compiler_limits, source_limits, defaults, ExtractionBackend::Int8)? })
    }
    pub fn prepare(&self, document: BatchDocument<ExtractionBatchArgs>) -> Result<PreparedInt8BatchExtraction, BatchItemFailure> {
        let input = self.compiler.prepare_input(document).map_err(planning_fault)?;
        let ExtractionInput { task, source, schema, options, grounded, .. } = input;
        let c = &self.compiler;
        let plan = if grounded {
            Int8ExtractPlan::from_task_plan_with_source(&task, &schema, options, c.compiler_limits,
                &c.controls, &source, c.source_limits, c.identity.clone())
        } else {
            Int8ExtractPlan::from_task_plan(&task, &schema, options, c.compiler_limits, &c.controls, c.identity.clone())
        }.map_err(planning_failure)?;
        Ok(PreparedInt8BatchExtraction { plan, task, source })
    }
}
impl PreparedInt8BatchExtraction {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.plan.execution_identity() }
    pub fn extraction_plan(&self) -> &Int8ExtractPlan { &self.plan }
    pub fn task_plan(&self) -> &TaskPlan { &self.task }
    pub fn source(&self) -> &SourceDocument { &self.source }
    pub fn planned_work(&self) -> Int8Work { self.plan.planned_work() }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8ExtractError> {
        self.plan.verify_identity(admitted)
    }
    fn budget(&self, masks: ExtractionMaskBudget) -> Int8JsonBudget {
        let work = self.planned_work();
        Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
            max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
            max_kv_bytes: self.task.ir().budget().max_kv_bytes,
            max_total_mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
        } }
    }
}

/// The host additionally admits resident weights, RoPE/activation/logit rails,
/// vocabulary, source/grammar plans, allocator slack and batch-event staging.
/// These quantities are reservation inputs, never an observed RSS assertion.
pub struct Int8ExtractionAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int8Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    /// The complete Int8ExtractRun, including all source-occurrence evidence.
    pub max_result_bytes: u64,
}
pub trait Int8ExtractionBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8ExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}
#[derive(Clone, Copy, Debug)]
pub struct Int8ExtractionBatchLimits {
    pub max_model_work: Int8Work,
    pub masks: ExtractionMaskBudget,
}
impl Int8ExtractionBatchLimits {
    fn validate(self) -> Result<(), BatchFault> {
        let w = self.max_model_work; let m = self.masks;
        if w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
            || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0
            || m.per_mask.max_trie_node_visits == 0 || m.per_mask.checkpoint_interval_nodes == 0
            || m.max_visits_per_item == 0 || m.max_visits_per_run < m.max_visits_per_item {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}

pub struct NativeInt8ExtractionBatch<'e, 'weights, 'v, A: Int8ExtractionBatchAdmission> {
    compiler: Int8ExtractionBatchPlanner,
    driver: NativeDriver<'e, 'weights, 'v>,
    admission: A,
    run: RunState,
}
impl<'e, 'weights, 'v, A: Int8ExtractionBatchAdmission> NativeInt8ExtractionBatch<'e, 'weights, 'v, A> {
    pub fn new(compiler: Int8ExtractionBatchPlanner, engine: &'e mut StrictInt8Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A, limits: Int8ExtractionBatchLimits) -> Result<Self, BatchFault> {
        limits.validate()?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(BatchCode::InvalidExecution.into());
        }
        Ok(Self { compiler, driver: NativeDriver { engine, vocabulary }, admission, run: RunState::new(limits) })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.run.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.run.mask_visits }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !self.driver.clean() }
}
impl<A: Int8ExtractionBatchAdmission> BatchProcessor for NativeInt8ExtractionBatch<'_, '_, '_, A> {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt8BatchExtraction;
    type Output = GuardedOutput<Int8ExtractRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?;
        // An unwind even during source/grammar compilation closes this adapter.
        self.run.failed = true;
        let result = (|| {
            let prepared = self.compiler.prepare(document)?;
            preflight(&prepared, &self.driver)?;
            Ok(prepared)
        })();
        if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) { self.run.failed = false; }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let w = prepared.planned_work();
        BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(prepared, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}

// Atomic, checked and nonrefundable model+mask reservations. Epoch changes
// never reset the whole-run ceiling. No public reset or mutable ledger view.
struct RunState {
    limits: Int8ExtractionBatchLimits, reserved: Int8Work, mask_visits: u64,
    last_sequence: u64, failed: bool,
}
impl RunState {
    fn new(limits: Int8ExtractionBatchLimits) -> Self {
        Self { limits, reserved: Int8Work::default(), mask_visits: 0, last_sequence: 0, failed: false }
    }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: Int8Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.last_sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let sum = add_work(self.reserved, work).filter(|sum| fits(*sum, self.limits.max_model_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.mask_visits.checked_add(self.limits.masks.max_visits_per_item)
            .filter(|&sum| sum <= self.limits.masks.max_visits_per_run)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.reserved = sum; self.mask_visits = masks; self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn add_work(a: Int8Work, b: Int8Work) -> Option<Int8Work> {
    Some(Int8Work { forward_positions: a.forward_positions.checked_add(b.forward_positions)?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits)?,
        attention_pairs: a.attention_pairs.checked_add(b.attention_pairs)?,
        projections: a.projections.checked_add(b.projections).ok()? })
}
fn fits(work: Int8Work, ceiling: Int8Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}

// Private lifecycle seam for fault injection, not a public fake-native driver.
trait Driver {
    fn capacity(&self) -> usize;
    fn clean(&self) -> bool;
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction,
        identity: &ExecutionIdentity, budget: Int8JsonBudget, control: &mut C) -> Result<Int8ExtractRun, Int8ExtractError>;
}
struct NativeDriver<'e, 'weights, 'v> {
    engine: &'e mut StrictInt8Engine<'weights>, vocabulary: &'v ExtractionVocabulary,
}
impl Driver for NativeDriver<'_, '_, '_> {
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn clean(&self) -> bool { !self.engine.is_poisoned() && self.engine.kv_cache().all_slots_have_len(0) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction,
        identity: &ExecutionIdentity, budget: Int8JsonBudget, control: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        // This rechecks the actual materialized model and vocabulary BEFORE
        // opening the native session. Finalization and its last cancellation
        // checkpoint happen inside that same session's lifetime.
        prepared.plan.execute(self.engine, identity, self.vocabulary, budget, control)
    }
}
fn preflight(prepared: &PreparedInt8BatchExtraction, driver: &impl Driver) -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let capacity = u64::try_from(driver.capacity()).map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    let bytes = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if prepared.planned_work().forward_positions > capacity || bytes > prepared.task.ir().budget().max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(bytes)
}
fn execute_admitted<D: Driver, A: Int8ExtractionBatchAdmission, C: DecodeStepControl>(
    prepared: PreparedInt8BatchExtraction, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C,
) -> Result<GuardedOutput<Int8ExtractRun, A::Guard>, BatchItemFailure> {
    let work = prepared.planned_work(); run.begin(work, context)?;
    let outcome = (|| {
        if let Some(cause) = control.checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        let kv_reservation_bytes = preflight(&prepared, driver)?;
        let masks = run.limits.masks;
        let (identity, guard) = admission.admit(Int8ExtractionAdmission {
            identity: prepared.execution_identity(), model_work: work,
            mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes, max_result_bytes: prepared.plan.max_result_bytes(),
        })?;
        prepared.verify_identity(&identity).map_err(execution_failure)?;
        if let Some(cause) = control.checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        let result = driver.execute(&prepared, &identity, prepared.budget(masks), control).map_err(execution_failure)?;
        if !valid_result(&prepared, &result, masks.max_visits_per_item) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        if let Some(cause) = control.checkpoint(result.result.output.token_ids.len()) {
            return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause)));
        }
        Ok(GuardedOutput::new(result, guard))
    })();
    match outcome {
        Ok(output) if driver.clean() => { run.failed = false; Ok(output) }
        Ok(_) => Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
        Err(mut failure) => {
            // Native mask/schema/validation failure may poison the engine even
            // when its error is normally a recoverable per-item refusal.
            if !driver.clean() { failure.stop = true; }
            if !failure.stop { run.failed = false; }
            Err(failure)
        }
    }
}
fn valid_result(prepared: &PreparedInt8BatchExtraction, result: &Int8ExtractRun, masks: u64) -> bool {
    let out = &result.result.output;
    let expected = constrained_int8::planned_work(prepared.plan.prompt_tokens(), out.token_ids.len()).ok();
    result.schema_version == 1 && result.execution == INT8_EXTRACT_VERSION
        && out.schema_version == 1 && out.numerics_profile == STRICT_INT8_PROFILE
        && result.result.task_spec_version == prepared.task.task_spec_identity()
        && out.token_ids.len() <= prepared.plan.options().max_new_tokens
        && expected == Some(result.model_work) && fits(result.model_work, prepared.planned_work())
        && out.forward_positions == result.model_work.forward_positions
        && out.projected_logits == result.model_work.projected_logits && out.mask_node_visit_charge <= masks
}
fn planning_fault(fault: BatchFault) -> BatchItemFailure {
    if matches!(fault.code, BatchCode::Allocation | BatchCode::InvalidExecution | BatchCode::Admission | BatchCode::Serialization) {
        BatchItemFailure::fatal(fault)
    } else { BatchItemFailure::reject(fault) }
}
fn planning_failure(error: Int8ExtractError) -> BatchItemFailure {
    match error {
        Int8ExtractError::Identity | Int8ExtractError::Native(Int8JsonError::ModelIdentity) => BatchItemFailure::fatal(BatchCode::Admission),
        Int8ExtractError::Extraction(ExtractError::AllocationRefused)
            | Int8ExtractError::Native(Int8JsonError::Native(StrictInt8Error::Allocation)) => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8ExtractError::Extraction(ExtractError::Serialization) => BatchItemFailure::fatal(BatchCode::Serialization),
        Int8ExtractError::Native(Int8JsonError::Native(StrictInt8Error::Context)) => BatchItemFailure::reject(BatchCode::Admission),
        _ => BatchItemFailure::reject(BatchCode::Planning),
    }
}
fn execution_failure(error: Int8ExtractError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8ExtractError::Extraction(error) => super::execution_failure(error),
        Int8ExtractError::Native(Int8JsonError::Decode(error)) => super::execution_failure(ExtractError::Decode(error)),
        Int8ExtractError::Identity | Int8ExtractError::Native(Int8JsonError::ModelIdentity) => BatchItemFailure::fatal(BatchCode::Admission),
        Int8ExtractError::Native(Int8JsonError::WorkMismatch) => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        Int8ExtractError::Native(Int8JsonError::Native(error)) => match error {
            StrictInt8Error::Context | StrictInt8Error::Memory => BatchItemFailure::reject(BatchCode::Admission),
            StrictInt8Error::Work => BatchItemFailure::reject(BatchCode::WorkLimit),
            StrictInt8Error::Allocation => BatchItemFailure::fatal(BatchCode::Allocation),
            _ => BatchItemFailure::fatal(BatchCode::Execution),
        },
    }
}

#[cfg(test)] mod tests;
