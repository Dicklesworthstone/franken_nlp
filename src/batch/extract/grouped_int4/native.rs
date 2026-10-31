//! Sequential schema/source extraction over one resident grouped-INT4 engine.
//!
//! The existing task owns constrained decoding and independent finalization.
//! This adapter owns delivery ordering, nonrenewable native/mask allowances,
//! poisoning, and the output guard retained through the batch writer's flush.

use super::*;
use crate::native_engine::{
    constrained_int4::{self, Int4JsonBudget},
    strict_int4::{Int4MemoryRequirement, Int4RunBudget, StrictInt4Engine, StrictInt4Error,
        STRICT_INT4_PROFILE},
};
use crate::tasks::extract::grouped_int4::{Int4ExtractRun, INT4_EXTRACT_VERSION};

/// Host admission must additionally cover immutable weights/vocabulary,
/// preparation, full native workspace, allocator slack and transport buffers.
/// These reservations are modeled commitments, not measurements of RSS.
pub struct Int4ExtractionAdmission<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
    pub kv_reservation_bytes: u64,
    /// Complete Int4ExtractRun, including native work and source evidence.
    pub max_result_bytes: u64,
}
pub trait Int4ExtractionBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4ExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

/// Whole-run ceilings. Epoch/flush boundaries never renew any allowance.
#[derive(Clone, Copy, Debug)]
pub struct Int4ExtractionBatchLimits {
    pub max_model_work: Int4Work,
    pub masks: ExtractionMaskBudget,
}
impl Int4ExtractionBatchLimits {
    pub(crate) fn validate(self) -> Result<(), BatchFault> {
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

/// Borrows one native engine for the complete ordered stream. Every item
/// opens and drains its own session; model weights and workspace stay resident.
pub struct NativeInt4ExtractionBatch<'e, 'weights, 'v, A: Int4ExtractionBatchAdmission> {
    compiler: Int4ExtractionBatchPlanner,
    driver: NativeDriver<'e, 'weights, 'v>,
    admission: A,
    run: RunState,
}
impl<'e, 'weights, 'v, A: Int4ExtractionBatchAdmission> NativeInt4ExtractionBatch<'e, 'weights, 'v, A> {
    pub fn new(compiler: Int4ExtractionBatchPlanner, engine: &'e mut StrictInt4Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A, limits: Int4ExtractionBatchLimits)
        -> Result<Self, BatchFault> {
        limits.validate()?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
            return Err(BatchCode::InvalidExecution.into());
        }
        if compiler.group_width() != engine.model_identity().group_width() {
            return Err(BatchCode::Admission.into());
        }
        constrained_int4::check_model(compiler.base_execution_identity(), engine.model_identity())
            .map_err(|_| BatchCode::Admission)?;
        Ok(Self { compiler, driver: NativeDriver { engine, vocabulary }, admission, run: RunState::new(limits) })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.run.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.run.mask_visits }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !self.driver.clean() }
}
impl<A: Int4ExtractionBatchAdmission> BatchProcessor for NativeInt4ExtractionBatch<'_, '_, '_, A> {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt4BatchExtraction;
    type Output = GuardedOutput<Int4ExtractRun, A::Guard>;

    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?;
        // An unwind in tokenization or source/schema construction closes the
        // adapter too; only a checked recoverable refusal releases this latch.
        self.run.failed = true;
        let result = (|| {
            let prepared = self.compiler.prepare(document).map_err(preparation_failure)?;
            preflight(&prepared, &self.driver)?;
            Ok(prepared)
        })();
        if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) { self.run.failed = false; }
        result
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?;
        self.run.failed = true;
        let result = (|| {
            let prepared = self.compiler.prepare_with_control(document, control).map_err(preparation_failure)?;
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
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.run.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(prepared, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}

// All axes reserve together before admission/native work. A failed attempt
// never refunds either ledger, and there is no public reset at an epoch.
struct RunState {
    limits: Int4ExtractionBatchLimits, reserved: Int4Work, mask_visits: u64,
    last_sequence: u64, failed: bool,
}
impl RunState {
    fn new(limits: Int4ExtractionBatchLimits) -> Self {
        Self { limits, reserved: Int4Work::default(), mask_visits: 0, last_sequence: 0, failed: false }
    }
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: Int4Work, context: BatchRequestContext) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.last_sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let total = self.reserved.checked_add(work).ok().filter(|sum| fits(*sum, self.limits.max_model_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.mask_visits.checked_add(self.limits.masks.max_visits_per_item)
            .filter(|&sum| sum <= self.limits.masks.max_visits_per_run)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.reserved = total; self.mask_visits = masks; self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}

// This private seam exercises lifecycle faults without fabricating a public
// model or allowing callers to supply their own completed native result.
pub(super) trait Driver {
    fn capacity(&self) -> usize;
    fn group_width(&self) -> usize;
    fn clean(&self) -> bool;
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt4BatchExtraction,
        identity: &ExecutionIdentity, budget: Int4JsonBudget, control: &mut C)
        -> Result<Int4ExtractRun, Int4ExtractError>;
}
pub(super) struct NativeDriver<'e, 'weights, 'v> {
    pub(super) engine: &'e mut StrictInt4Engine<'weights>,
    pub(super) vocabulary: &'v ExtractionVocabulary,
}
impl Driver for NativeDriver<'_, '_, '_> {
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.engine.model_identity().group_width() }
    fn clean(&self) -> bool { !self.engine.is_poisoned() && self.engine.kv_cache().all_slots_have_len(0) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt4BatchExtraction,
        identity: &ExecutionIdentity, budget: Int4JsonBudget, control: &mut C)
        -> Result<Int4ExtractRun, Int4ExtractError> {
        // Actual model identity, vocabulary controls, final validation and the
        // final cancellation checkpoint remain inside the real task/session.
        prepared.plan.execute(self.engine, identity, self.vocabulary, budget, control)
    }
}
pub(super) fn preflight(prepared: &PreparedInt4BatchExtraction, driver: &impl Driver) -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    if prepared.plan.group_width() != driver.group_width() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(driver.capacity(), driver.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if prepared.planned_work().forward_positions > driver.capacity() as u64
        || required.kv_bytes > prepared.plan.max_kv_bytes() {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(required.kv_bytes)
}
pub(super) fn native_budget(prepared: &PreparedInt4BatchExtraction, masks: ExtractionMaskBudget) -> Int4JsonBudget {
    let work = prepared.planned_work();
    Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: prepared.plan.max_kv_bytes(),
        max_total_mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
    } }
}
fn execute_admitted<D: Driver, A: Int4ExtractionBatchAdmission, C: DecodeStepControl>(
    prepared: PreparedInt4BatchExtraction, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C,
) -> Result<GuardedOutput<Int4ExtractRun, A::Guard>, BatchItemFailure> {
    let work = prepared.planned_work(); run.begin(work, context)?;
    let outcome = (|| {
        if let Some(cause) = control.checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        let kv_reservation_bytes = preflight(&prepared, driver)?;
        let masks = run.limits.masks;
        let (identity, guard) = admission.admit(Int4ExtractionAdmission {
            identity: prepared.execution_identity(), model_work: work,
            mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes, max_result_bytes: prepared.plan.max_result_bytes(),
        })?;
        prepared.verify_identity(&identity).map_err(execution_failure)?;
        if let Some(cause) = control.checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        let result = driver.execute(&prepared, &identity, native_budget(&prepared, masks), control).map_err(execution_failure)?;
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
            // A nominally recoverable mask/output refusal may have poisoned
            // the native engine; reuse is then forbidden for the whole run.
            if !driver.clean() { failure.stop = true; }
            if !failure.stop { run.failed = false; }
            Err(failure)
        }
    }
}
pub(super) fn valid_result(prepared: &PreparedInt4BatchExtraction, result: &Int4ExtractRun, masks: u64) -> bool {
    let out = &result.result.output;
    let options = prepared.plan.options();
    let expected = constrained_int4::planned_work(prepared.plan.group_width(),
        prepared.plan.prompt_tokens(), out.token_ids.len()).ok();
    result.schema_version == 1 && result.execution == INT4_EXTRACT_VERSION
        && result.group_width == prepared.plan.group_width()
        && out.schema_version == 1 && out.numerics_profile == STRICT_INT4_PROFILE
        && result.result.task_spec_version == prepared.task.task_spec_identity()
        && out.token_ids.len() <= options.max_new_tokens
        && out.token_ids.last() == Some(&options.eos_token_id)
        && out.token_ids.iter().filter(|&&id| id == options.eos_token_id).count() == 1
        && expected == Some(result.model_work) && fits(result.model_work, prepared.planned_work())
        && out.forward_positions == result.model_work.forward_positions
        && out.projected_logits == result.model_work.projected_logits && out.mask_node_visit_charge <= masks
}
fn preparation_failure(mut failure: BatchItemFailure) -> BatchItemFailure {
    if matches!(failure.fault.code, BatchCode::Allocation | BatchCode::InvalidExecution
        | BatchCode::Admission | BatchCode::Serialization | BatchCode::Cancelled) {
        failure.stop = true;
    }
    failure
}
pub(super) fn execution_failure(error: Int4ExtractError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4ExtractError::Extraction(error) => super::super::execution_failure(error),
        Int4ExtractError::Native(Int4JsonError::Decode(error)) => super::super::execution_failure(ExtractError::Decode(error)),
        Int4ExtractError::Identity | Int4ExtractError::Native(Int4JsonError::ModelIdentity) =>
            BatchItemFailure::fatal(BatchCode::Admission),
        Int4ExtractError::Native(Int4JsonError::WorkMismatch) => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        Int4ExtractError::Native(Int4JsonError::Native(error)) => match error {
            StrictInt4Error::Context | StrictInt4Error::Memory => BatchItemFailure::reject(BatchCode::Admission),
            StrictInt4Error::Work => BatchItemFailure::reject(BatchCode::WorkLimit),
            StrictInt4Error::Allocation => BatchItemFailure::fatal(BatchCode::Allocation),
            _ => BatchItemFailure::fatal(BatchCode::Execution),
        },
    }
}

#[cfg(test)] mod tests;
