//! Resident INT8 NER, keyphrases, cited summaries and passage QA on NDJSON.
//! Reuses the source task compiler, native decoder, finalizers and ordered
//! runner. Neither a failed native task nor a failed citation becomes abstention.

use super::*;
use crate::{
    native_engine::{constrained_int8::{self, Int8JsonBudget, Int8JsonError},
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error}},
    tasks::{extract::quantized::Int8ExtractError,
        source_planning::quantized::{Int8SourceError, Int8SourceTaskRun, PreparedInt8SourceTask, INT8_SOURCE_EXECUTION}},
};
// The quantities and ownership contract are identical to schema extraction.
// Reuse the existing host authority; do not require a second permit provider.
pub use crate::batch::extract::quantized::{Int8ExtractionAdmission as Int8SourceAdmission,
    Int8ExtractionBatchAdmission as Int8SourceBatchAdmission, Int8ExtractionBatchLimits as Int8SourceBatchLimits};

/// Additional cap for trusted defaults BEFORE per-document cloning. Transport
/// line/depth/input bounds and source compiler limits remain independent.
pub const MAX_SOURCE_ARGUMENT_BYTES: usize = 1024 * 1024;

pub struct Int8SourceBatchPlanner<'p> {
    planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
    limits: SourcePlanningLimits, defaults: Option<SourceBatchArgs>,
}
impl<'p> Int8SourceBatchPlanner<'p> {
    pub fn new(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        limits: SourcePlanningLimits, defaults: Option<SourceBatchArgs>) -> Result<Self, BatchFault> {
        check_configuration(planner, &identity, ceiling, limits, defaults.as_ref())?;
        Ok(Self { planner, identity, ceiling, limits, defaults })
    }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SourceBatchArgs>,
        control: &mut C) -> Result<PreparedInt8SourceTask, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if args.task().spec().identity() != self.identity.task_spec || argument_size(&args).is_err() {
            return Err(BatchItemFailure::reject(BatchCode::Planning));
        }
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan_int8_with_control(&args.into_request(document.text),
            &context, self.limits, control).map_err(planning_failure)?;
        checkpoint(control)?;
        Ok(plan)
    }
}
pub(crate) fn check_configuration(planner: &SourceTaskPlanner, identity: &ExecutionIdentity,
    ceiling: TaskBudget, limits: SourcePlanningLimits, defaults: Option<&SourceBatchArgs>) -> Result<(), BatchFault> {
    identity.validate().map_err(|_| BatchCode::Admission)?;
    constrained_int8::check_profile(identity).map_err(|_| BatchCode::Admission)?;
    ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if !matches!(identity.task_spec.as_str(), "ner-v1" | "keyphrases-v1" | "summarize-v1" | "answer-v1")
        || identity.template_digest != *planner.template_digest() || identity.tokenizer_digest != planner.tokenizer_digest()
        || defaults.is_some_and(|args| args.task().spec().identity() != identity.task_spec) {
        return Err(BatchCode::Admission.into());
    }
    if !(1..=64 * 1024 * 1024).contains(&limits.max_input_bytes)
        || !(1..=262_144).contains(&limits.max_context_tokens) || !(1..=1024).contains(&limits.max_passages)
        || defaults.is_some_and(|args| argument_size(args).is_err()) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn argument_size(args: &SourceBatchArgs) -> Result<(), ()> {
    struct Counter(usize);
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.checked_sub(bytes.len()).ok_or_else(|| std::io::Error::other("source argument bound"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter(MAX_SOURCE_ARGUMENT_BYTES), args).map_err(|_| ())
}

pub struct NativeInt8SourceBatch<'p, 'e, 'weights, 'v, A: Int8SourceBatchAdmission> {
    compiler: Int8SourceBatchPlanner<'p>, driver: NativeDriver<'e, 'weights, 'v>, admission: A, run: RunState,
}
impl<'p, 'e, 'weights, 'v, A: Int8SourceBatchAdmission> NativeInt8SourceBatch<'p, 'e, 'weights, 'v, A> {
    pub fn new(compiler: Int8SourceBatchPlanner<'p>, engine: &'e mut StrictInt8Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A, limits: Int8SourceBatchLimits) -> Result<Self, BatchFault> {
        validate_limits(limits)?;
        let driver = NativeDriver { engine, vocabulary };
        if !driver.clean() { return Err(BatchCode::InvalidExecution.into()); }
        Ok(Self { compiler, driver, admission, run: RunState::new(limits) })
    }
    pub fn reserved_model_work(&self) -> Int8Work { self.run.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.run.mask_visits }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !self.driver.clean() }
}
impl<A: Int8SourceBatchAdmission> BatchProcessor for NativeInt8SourceBatch<'_, '_, '_, '_, A> {
    type Args = SourceBatchArgs;
    type Prepared = PreparedInt8SourceTask;
    type Output = GuardedOutput<Int8SourceTaskRun, A::Guard>;
    // Standalone callers use the planner's explicit control-taking method.
    // Never invent a fresh/no-op run control to bypass a corpus deadline.
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.ready()?; self.run.failed = true;
        let result = (|| {
            let prepared = self.compiler.prepare_with_control(document, control)?;
            preflight(&prepared, &self.driver)?;
            Ok(prepared)
        })();
        if !result.as_ref().is_err_and(|e: &BatchItemFailure| e.stop) { self.run.failed = false; }
        result
    }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork {
        let work = plan.planned_work(); BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, context: BatchRequestContext,
        control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(prepared, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}

pub(crate) fn validate_limits(limits: Int8SourceBatchLimits) -> Result<(), BatchFault> {
    let w = limits.max_model_work; let m = limits.masks;
    if w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
        || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0
        || m.per_mask.max_trie_node_visits == 0 || m.per_mask.checkpoint_interval_nodes == 0
        || m.max_visits_per_item == 0 || m.max_visits_per_run < m.max_visits_per_item {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
struct RunState {
    limits: Int8SourceBatchLimits, reserved: Int8Work, mask_visits: u64, last_sequence: u64, failed: bool,
}
impl RunState {
    fn new(limits: Int8SourceBatchLimits) -> Self {
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
        // Both counters update together or neither does. No refund/reset path.
        let next = self.reserved.checked_add(work).ok().filter(|w| fits(*w, self.limits.max_model_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let masks = self.mask_visits.checked_add(self.limits.masks.max_visits_per_item)
            .filter(|&n| n <= self.limits.masks.max_visits_per_run)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.reserved = next; self.mask_visits = masks; self.last_sequence = context.request_seq;
        Ok(())
    }
}
fn fits(work: Int8Work, ceiling: Int8Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}

// Private fault-injection seam. Public construction accepts only the actual
// StrictInt8Engine, so a deserialized/fixture result is never native authority.
trait Driver {
    fn capacity(&self) -> usize;
    fn clean(&self) -> bool;
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8SourceTask, identity: &ExecutionIdentity,
        budget: Int8JsonBudget, control: &mut C) -> Result<Int8SourceTaskRun, Int8SourceError>;
}
struct NativeDriver<'e, 'weights, 'v> { engine: &'e mut StrictInt8Engine<'weights>, vocabulary: &'v ExtractionVocabulary }
impl Driver for NativeDriver<'_, '_, '_> {
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn clean(&self) -> bool { !self.engine.is_poisoned() && self.engine.kv_cache().all_slots_have_len(0) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8SourceTask, identity: &ExecutionIdentity,
        budget: Int8JsonBudget, control: &mut C) -> Result<Int8SourceTaskRun, Int8SourceError> {
        prepared.execute_with_control(identity, self.engine, self.vocabulary, budget, control)
    }
}
fn preflight(prepared: &PreparedInt8SourceTask, driver: &impl Driver) -> Result<u64, BatchItemFailure> {
    let capacity = u64::try_from(driver.capacity()).map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    check_capacity(driver.clean(), capacity, prepared.planned_work().forward_positions, prepared.task_budget().max_kv_bytes)?;
    capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or_else(|| BatchItemFailure::fatal(BatchCode::InvalidExecution))
}
fn execute_admitted<D: Driver, A: Int8SourceBatchAdmission, C: DecodeStepControl>(
    prepared: PreparedInt8SourceTask, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C,
) -> Result<GuardedOutput<Int8SourceTaskRun, A::Guard>, BatchItemFailure> {
    let work = prepared.planned_work(); run.begin(work, context)?;
    let result = (|| {
        checkpoint(control)?;
        let kv_reservation_bytes = preflight(&prepared, driver)?;
        let masks = run.limits.masks;
        let (identity, guard) = admission.admit(Int8SourceAdmission { identity: prepared.execution_identity(),
            model_work: work, mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes, max_result_bytes: prepared.max_result_bytes() })?;
        prepared.verify_identity(&identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        checkpoint(control)?;
        let budget = Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
            max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
            max_kv_bytes: prepared.task_budget().max_kv_bytes, max_total_mask_node_visits: masks.max_visits_per_item,
            mask_limits: masks.per_mask,
        } };
        let output = driver.execute(&prepared, &identity, budget, control).map_err(execution_failure)?;
        if !valid_result(&prepared, &output, masks.max_visits_per_item) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    match result {
        Ok(output) if driver.clean() => { run.failed = false; Ok(output) }
        Ok(_) => Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
        Err(mut error) => {
            // Preserve typed cancellation even when the native session is also
            // poisoned. Never clear native state or transform failure into QA abstention.
            if !driver.clean() { error.stop = true; }
            if !error.stop { run.failed = false; }
            Err(error)
        }
    }
}
fn valid_result(plan: &PreparedInt8SourceTask, output: &Int8SourceTaskRun, mask_cap: u64) -> bool {
    let actual = observed(&output.result);
    let task = match &output.result { SourceTaskResult::Ner(_) => "ner-v1", SourceTaskResult::Keyphrases(_) => "keyphrases-v1",
        SourceTaskResult::Summarize(_) => "summarize-v1", SourceTaskResult::Answer(_) => "answer-v1" };
    output.schema_version == 1 && output.execution == INT8_SOURCE_EXECUTION && task == plan.execution_identity().task_spec
        && actual.tokens > 0 && actual.tokens <= plan.task_budget().max_output_tokens as usize
        && constrained_int8::planned_work(plan.prompt_tokens(), actual.tokens).ok() == Some(output.model_work)
        && fits(output.model_work, plan.planned_work()) && actual.positions == output.model_work.forward_positions
        && actual.logits == output.model_work.projected_logits && actual.mask_visits <= mask_cap
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()) }
}
fn planning_failure(error: Int8SourceError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8SourceError::Planning(e) => super::preparation_failure(e),
        Int8SourceError::Extraction(Int8ExtractError::Extraction(e)) => super::preparation_failure(SourcePlanningError::Extraction(e)),
        other => execution_failure(other),
    }
}
fn execution_failure(error: Int8SourceError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8SourceError::Planning(e) => super::execution_failure(e),
        Int8SourceError::Extraction(Int8ExtractError::Extraction(e)) => super::extraction_failure(e),
        Int8SourceError::Extraction(Int8ExtractError::Native(Int8JsonError::Decode(e))) => super::extraction_failure(ExtractError::Decode(e)),
        Int8SourceError::Extraction(Int8ExtractError::Identity | Int8ExtractError::Native(Int8JsonError::ModelIdentity))
            => BatchItemFailure::fatal(BatchCode::Admission),
        Int8SourceError::Extraction(Int8ExtractError::Native(Int8JsonError::Native(StrictInt8Error::Context | StrictInt8Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        Int8SourceError::Extraction(Int8ExtractError::Native(Int8JsonError::Native(StrictInt8Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int8SourceError::Extraction(Int8ExtractError::Native(Int8JsonError::Native(StrictInt8Error::Allocation)))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8SourceError::InvalidResult | Int8SourceError::Extraction(Int8ExtractError::Native(Int8JsonError::WorkMismatch))
            => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)] mod tests;
