//! Resident grouped-INT4 source tasks on the bounded, ordered NDJSON runner.
//! The pinned source compiler and independent finalizers retain their task
//! semantics. Actual model/group identity, complete native work and aggregate
//! mask allowances remain bound across records and flush epochs. A failed
//! task or citation never becomes successful question-answering abstention.

use super::*;
use crate::{
    native_engine::{constrained_int4::{self, Int4JsonBudget, Int4JsonError},
        strict_int4::{Int4MemoryRequirement, Int4RunBudget, Int4Work, StrictInt4Engine,
            StrictInt4Error, STRICT_INT4_PROFILE}},
    tasks::{extract::grouped_int4::Int4ExtractError,
        source_planning::grouped_int4::{Int4SourceError, Int4SourceTaskRun, PreparedInt4SourceTask, INT4_SOURCE_EXECUTION}},
};
// The quantities and ownership contract are identical to schema extraction.
// Reuse the existing host authority; do not require a second permit provider.
pub use crate::batch::extract::grouped_int4::{Int4ExtractionAdmission as Int4SourceAdmission,
    Int4ExtractionBatchAdmission as Int4SourceBatchAdmission, Int4ExtractionBatchLimits as Int4SourceBatchLimits};

/// Additional cap for trusted defaults BEFORE per-document cloning. Transport
/// line/depth/input bounds and source compiler limits remain independent.
pub const MAX_SOURCE_ARGUMENT_BYTES: usize = 1024 * 1024;

pub struct Int4SourceBatchPlanner<'p> {
    planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
    limits: SourcePlanningLimits, defaults: Option<SourceBatchArgs>, group_width: usize,
}
impl<'p> Int4SourceBatchPlanner<'p> {
    pub fn new(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, ceiling: TaskBudget,
        limits: SourcePlanningLimits, defaults: Option<SourceBatchArgs>, group_width: usize) -> Result<Self, BatchFault> {
        check_configuration(planner, &identity, ceiling, limits, defaults.as_ref(), group_width)?;
        Ok(Self { planner, identity, ceiling, limits, defaults, group_width })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SourceBatchArgs>,
        control: &mut C) -> Result<PreparedInt4SourceTask, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.or_else(|| self.defaults.clone())
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if args.task().spec().identity() != self.identity.task_spec || argument_size(&args).is_err() {
            return Err(BatchItemFailure::reject(BatchCode::Planning));
        }
        let context = PlanContext::new(&self.identity, self.ceiling)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.planner.plan_int4_with_control(&args.into_request(document.text),
            &context, self.limits, self.group_width, control).map_err(planning_failure)?;
        checkpoint(control)?;
        Ok(plan)
    }
}
pub(crate) fn check_configuration(planner: &SourceTaskPlanner, identity: &ExecutionIdentity,
    ceiling: TaskBudget, limits: SourcePlanningLimits, defaults: Option<&SourceBatchArgs>,
    group_width: usize) -> Result<(), BatchFault> {
    identity.validate().map_err(|_| BatchCode::Admission)?;
    constrained_int4::check_profile(identity, group_width).map_err(|_| BatchCode::Admission)?;
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

pub struct NativeInt4SourceBatch<'p, 'e, 'weights, 'v, A: Int4SourceBatchAdmission> {
    compiler: Int4SourceBatchPlanner<'p>, driver: NativeDriver<'e, 'weights, 'v>, admission: A, run: RunState,
}
impl<'p, 'e, 'weights, 'v, A: Int4SourceBatchAdmission> NativeInt4SourceBatch<'p, 'e, 'weights, 'v, A> {
    pub fn new(compiler: Int4SourceBatchPlanner<'p>, engine: &'e mut StrictInt4Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A, limits: Int4SourceBatchLimits) -> Result<Self, BatchFault> {
        validate_limits(limits)?;
        // Bind before accepting any record; profile-compatible fabricated or
        // different resident weights cannot acquire a native batch authority.
        if compiler.group_width() != engine.model_identity().group_width() {
            return Err(BatchCode::Admission.into());
        }
        constrained_int4::check_model(compiler.base_execution_identity(), engine.model_identity())
            .map_err(|_| BatchCode::Admission)?;
        let driver = NativeDriver { engine, vocabulary };
        if !driver.clean() { return Err(BatchCode::InvalidExecution.into()); }
        Ok(Self { compiler, driver, admission, run: RunState::new(limits) })
    }
    pub fn reserved_model_work(&self) -> Int4Work { self.run.reserved }
    pub fn reserved_mask_visits(&self) -> u64 { self.run.mask_visits }
    pub fn is_poisoned(&self) -> bool { self.run.failed || !self.driver.clean() }
}
impl<A: Int4SourceBatchAdmission> BatchProcessor for NativeInt4SourceBatch<'_, '_, '_, '_, A> {
    type Args = SourceBatchArgs;
    type Prepared = PreparedInt4SourceTask;
    type Output = GuardedOutput<Int4SourceTaskRun, A::Guard>;
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

pub(crate) fn validate_limits(limits: Int4SourceBatchLimits) -> Result<(), BatchFault> {
    limits.validate()
}
struct RunState {
    limits: Int4SourceBatchLimits, reserved: Int4Work, mask_visits: u64, last_sequence: u64, failed: bool,
}
impl RunState {
    fn new(limits: Int4SourceBatchLimits) -> Self {
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
fn fits(work: Int4Work, ceiling: Int4Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}

// Private fault-injection seam. Public construction accepts only the actual
// StrictInt4Engine, so a deserialized/fixture result is never native authority.
trait Driver {
    fn capacity(&self) -> usize;
    fn group_width(&self) -> usize;
    fn clean(&self) -> bool;
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt4SourceTask, identity: &ExecutionIdentity,
        budget: Int4JsonBudget, control: &mut C) -> Result<Int4SourceTaskRun, Int4SourceError>;
}
struct NativeDriver<'e, 'weights, 'v> { engine: &'e mut StrictInt4Engine<'weights>, vocabulary: &'v ExtractionVocabulary }
impl Driver for NativeDriver<'_, '_, '_> {
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.engine.model_identity().group_width() }
    fn clean(&self) -> bool { !self.engine.is_poisoned() && self.engine.kv_cache().all_slots_have_len(0) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt4SourceTask, identity: &ExecutionIdentity,
        budget: Int4JsonBudget, control: &mut C) -> Result<Int4SourceTaskRun, Int4SourceError> {
        prepared.execute_with_control(identity, self.engine, self.vocabulary, budget, control)
    }
}
fn preflight(prepared: &PreparedInt4SourceTask, driver: &impl Driver) -> Result<u64, BatchItemFailure> {
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    if prepared.group_width() != driver.group_width() {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let capacity = u64::try_from(driver.capacity()).map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    let required = Int4MemoryRequirement::for_context(driver.capacity(), driver.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if prepared.planned_work().forward_positions > capacity || required.kv_bytes > prepared.task_budget().max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(required.kv_bytes)
}
fn execute_admitted<D: Driver, A: Int4SourceBatchAdmission, C: DecodeStepControl>(
    prepared: PreparedInt4SourceTask, driver: &mut D, admission: &mut A,
    run: &mut RunState, context: BatchRequestContext, control: &mut C,
) -> Result<GuardedOutput<Int4SourceTaskRun, A::Guard>, BatchItemFailure> {
    let work = prepared.planned_work(); run.begin(work, context)?;
    let result = (|| {
        checkpoint(control)?;
        let kv_reservation_bytes = preflight(&prepared, driver)?;
        let masks = run.limits.masks;
        let (identity, guard) = admission.admit(Int4SourceAdmission { identity: prepared.execution_identity(),
            model_work: work, mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes, max_result_bytes: prepared.max_result_bytes() })?;
        prepared.verify_identity(&identity).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        checkpoint(control)?;
        let budget = Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
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
fn valid_result(plan: &PreparedInt4SourceTask, output: &Int4SourceTaskRun, mask_cap: u64) -> bool {
    let actual = observed(&output.result);
    let (version, task, declared_task, profile) = match &output.result {
        SourceTaskResult::Ner(r) => (r.schema_version, "ner-v1", r.task_spec_version.as_str(), r.numerics_profile.as_str()),
        SourceTaskResult::Keyphrases(r) => (r.schema_version, "keyphrases-v1", r.task_spec_version.as_str(), r.numerics_profile.as_str()),
        SourceTaskResult::Summarize(r) => (r.schema_version, "summarize-v1", r.task_spec_version.as_str(), r.numerics_profile.as_str()),
        SourceTaskResult::Answer(r) => (r.schema_version, "answer-v1", r.task_spec_version.as_str(), r.numerics_profile.as_str()),
    };
    output.schema_version == 1 && output.execution == INT4_SOURCE_EXECUTION && output.group_width == plan.group_width()
        && version == 1 && task == declared_task && task == plan.execution_identity().task_spec && profile == STRICT_INT4_PROFILE
        && actual.tokens > 0 && actual.tokens <= plan.task_budget().max_output_tokens as usize
        && constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), actual.tokens).ok() == Some(output.model_work)
        && fits(output.model_work, plan.planned_work()) && actual.positions == output.model_work.forward_positions
        && actual.logits == output.model_work.projected_logits && actual.mask_visits <= mask_cap
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()) }
}
fn planning_failure(error: Int4SourceError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4SourceError::Planning(e) => super::preparation_failure(e),
        Int4SourceError::Extraction(Int4ExtractError::Extraction(e)) => super::preparation_failure(SourcePlanningError::Extraction(e)),
        other => execution_failure(other),
    }
}
fn execution_failure(error: Int4SourceError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4SourceError::Planning(e) => super::execution_failure(e),
        Int4SourceError::Extraction(Int4ExtractError::Extraction(e)) => super::extraction_failure(e),
        Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Decode(e))) => super::extraction_failure(ExtractError::Decode(e)),
        Int4SourceError::Extraction(Int4ExtractError::Identity | Int4ExtractError::Native(Int4JsonError::ModelIdentity))
            => BatchItemFailure::fatal(BatchCode::Admission),
        Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::Native(StrictInt4Error::Allocation)))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        Int4SourceError::InvalidResult | Int4SourceError::Extraction(Int4ExtractError::Native(Int4JsonError::WorkMismatch))
            => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        _ => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)] mod tests;
