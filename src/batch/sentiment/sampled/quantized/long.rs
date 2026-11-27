//! Complete sampled INT8 documents on one resident native allocation.
//! Every original source is planned in full before its first forward. Stable
//! item and original chunk coordinates own sample addresses; stream position,
//! flush epochs and failed attempts never change those addresses or renew work.
use super::*;
use crate::tasks::sentiment::sampled::{
    long::SampledSentimentMapError,
    quantized::long::{PreparedInt8SampledSentimentMap, Int8SampledSentimentMapError,
        Int8SampledSentimentMapLimits, Int8SampledSentimentMapRun,
        INT8_SAMPLED_SENTIMENT_MAP_EXECUTION,
        memory::{MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES, Int8SampledSentimentMapMemoryLimits}},
};

pub fn int8_sampled_sentiment_document_preparation_bytes(task: TaskBudget,
    planning: SampledSentimentLimits, mapping: Int8SampledSentimentMapLimits,
    options: &SampledSentimentOptions, encoding: &UntrustedEncoding) -> Result<u64, BatchFault> {
    Int8SampledSentimentMapMemoryLimits {
        context_tokens: mapping.chunks.context_tokens,
        preparation_reserve_bytes: MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: 1, max_axes: SentimentAxis::ALL.len(),
        max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?,
    }.requirements(task, planning, mapping, options, encoding)
        .map(|required| required.input_bytes).map_err(|_| BatchCode::InvalidLimits.into())
}

/// Available without the optional storage backend, so admission precedes IO.
pub fn int8_sampled_sentiment_document_job_preparation_bytes(task: TaskBudget,
    planning: SampledSentimentLimits, mapping: Int8SampledSentimentMapLimits,
    options: &SampledSentimentOptions, encoding: &UntrustedEncoding) -> Result<u64, BatchFault> {
    int8_sampled_sentiment_document_preparation_bytes(task, planning, mapping, options, encoding)?
        .checked_add(super::super::SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64)
        .ok_or_else(|| BatchCode::InvalidLimits.into())
}

pub struct Int8SampledSentimentDocumentPlanner {
    planner: Arc<Int8SampledSentimentPlanner>,
    vocabulary: Arc<SampledSentimentVocabulary>,
    task_budget: TaskBudget,
    mapping: Int8SampledSentimentMapLimits,
    limits: Int8SampledSentimentBatchLimits,
    preparation_bytes: u64,
    seal: Arc<()>,
}
impl Int8SampledSentimentDocumentPlanner {
    pub fn new(planner: Int8SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        task_budget: TaskBudget, mapping: Int8SampledSentimentMapLimits,
        limits: Int8SampledSentimentBatchLimits) -> Result<Self, BatchFault> {
        limits.validate()?;
        mapping.validate(task_budget).map_err(|_| BatchCode::InvalidLimits)?;
        if task_budget != planner.task_budget() || planner.execution_identity().task_spec != "sentiment-v1"
            || planner.execution_identity().sampler_version != JSON_SAMPLER_VERSION
            || !fits(mapping.max_model_work, limits.max_item_work) {
            return Err(BatchCode::InvalidLimits.into());
        }
        constrained_int8::check_profile(planner.execution_identity()).map_err(|_| BatchCode::Admission)?;
        let preparation_bytes = int8_sampled_sentiment_document_preparation_bytes(task_budget,
            planner.limits(), mapping, planner.sampling_options(), planner.input_encoding())?;
        Ok(Self { planner: Arc::new(planner), vocabulary, task_budget, mapping, limits,
            preparation_bytes, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn task_budget(&self) -> TaskBudget { self.task_budget }
    pub fn mapping(&self) -> Int8SampledSentimentMapLimits { self.mapping }
    pub fn limits(&self) -> Int8SampledSentimentBatchLimits { self.limits }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.input_encoding() }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    pub fn planning_limits(&self) -> SampledSentimentLimits { self.planner.limits() }
    pub fn sampling_options(&self) -> &SampledSentimentOptions { self.planner.sampling_options() }
    /// Hosts retain these same allocations outside the team callback through
    /// physical worker join, including unwind and discarded queued work.
    pub(crate) fn retained_owners(&self)
        -> (Arc<Int8SampledSentimentPlanner>, Arc<SampledSentimentVocabulary>) {
        (Arc::clone(&self.planner), Arc::clone(&self.vocabulary))
    }
    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<SampledSentimentBatchArgs>,
        control: &mut C, encoding_work: &mut SafeBpeWork) -> Result<PreparedInt8SampledSentimentMap, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.unwrap_or_default();
        let budget = item_budget(args.budget, self.task_budget)?;
        self.planner.plan_document_for_item_with_encoding_work(&SentimentRequest {
            document: document.text, axes: args.axes, budget }, self.mapping, &document.id,
            control, encoding_work).map_err(map_failure)
    }
}

pub struct Int8SampledSentimentDocumentPlanFacts<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int8Work,
    pub sample_count: usize,
    pub chunk_count: usize,
    pub required_context: usize,
    pub original_task: TaskBudget,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
    pub preparation_bytes: u64,
}
pub struct Int8SampledSentimentDocumentAdmission<'a> {
    pub facts: Int8SampledSentimentDocumentPlanFacts<'a>,
    pub kv_reservation_bytes: u64,
    pub mask_node_visits: u64,
}
pub trait Int8SampledSentimentDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int8SampledSentimentDocumentAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}
struct PreparedRecord<P> { plan: P, mask_node_visits: u64, seal: Arc<()> }
pub struct PreparedInt8SampledSentimentDocumentRecord { record: PreparedRecord<PreparedInt8SampledSentimentMap> }
impl PreparedInt8SampledSentimentDocumentRecord {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.record.plan.execution_identity() }
    pub fn model_work(&self) -> Int8Work { self.record.plan.planned_work() }
    pub fn mask_node_visits(&self) -> u64 { self.record.mask_node_visits }
    pub fn max_result_bytes(&self) -> u64 { self.record.plan.max_result_bytes() }
    pub fn max_output_tokens(&self) -> u64 { self.record.plan.max_output_tokens() }
    pub fn execution_version(&self) -> &'static str { INT8_SAMPLED_SENTIMENT_MAP_EXECUTION }
}

/// The private state is set before any fallible native/admission callback.
/// Unwind, cancellation or dirty completion permanently closes this processor.
pub struct NativeInt8SampledSentimentDocumentBatch<'e, 'w, A: Int8SampledSentimentDocumentBatchAdmission> {
    compiler: Int8SampledSentimentDocumentPlanner,
    driver: NativeDriver<'e, 'w>,
    admission: A,
    run: RunState,
    encoding_work: SafeBpeWork,
    restored: bool,
}
impl<'e, 'w, A: Int8SampledSentimentDocumentBatchAdmission> NativeInt8SampledSentimentDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: Int8SampledSentimentDocumentPlanner, engine: &'e mut StrictInt8Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        Self::new_scheduled(compiler, engine, admission, None)
    }
    pub fn new_layer_major(compiler: Int8SampledSentimentDocumentPlanner, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        Self::new_scheduled(compiler, engine, admission, Some(prefill))
    }
    fn new_scheduled(compiler: Int8SampledSentimentDocumentPlanner, engine: &'e mut StrictInt8Engine<'w>,
        admission: A, prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        compiler.limits.validate()?;
        validate_prefill(prefill)?;
        check_model(compiler.base_execution_identity(), engine.artifact_identity())?;
        if !clean(engine) || compiler.mapping.chunks.context_tokens > engine.kv_cache().capacity_positions() {
            return Err(BatchCode::InvalidExecution.into());
        }
        let run = RunState::new(compiler.limits);
        let vocabulary = Arc::clone(&compiler.vocabulary);
        Ok(Self { compiler, driver: NativeDriver { engine, vocabulary, prefill }, admission, run,
            encoding_work: SafeBpeWork::default(), restored: false })
    }
    pub fn remaining_model_work(&self) -> Int8Work { self.run.remaining }
    pub fn remaining_mask_visits(&self) -> u64 { self.run.remaining_masks }
    pub fn encoding_work(&self) -> SafeBpeWork { self.encoding_work }
    pub fn is_poisoned(&self) -> bool { self.run.state != State::Ready || !self.driver.clean() }
    /// Only tighten a fresh adapter by authenticated history. This does not
    /// grant persistence authority; the existing journal still admits attempts.
    pub fn restore_spent_work(&mut self, spent: Int8Work, masks: u64) -> Result<(), BatchFault> {
        if self.restored || self.run.state != State::Ready || self.run.last_sequence != 0
            || self.encoding_work != SafeBpeWork::default() || !self.driver.clean() {
            return Err(BatchCode::InvalidExecution.into());
        }
        let remaining = subtract(self.run.remaining, spent).map_err(|error| error.fault)?;
        let remaining_masks = self.run.remaining_masks.checked_sub(masks).ok_or(BatchCode::WorkLimit)?;
        self.run.remaining = remaining; self.run.remaining_masks = remaining_masks; self.restored = true;
        Ok(())
    }
}
impl<A: Int8SampledSentimentDocumentBatchAdmission> BatchProcessor for NativeInt8SampledSentimentDocumentBatch<'_, '_, A> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedInt8SampledSentimentDocumentRecord;
    type Output = GuardedOutput<Int8SampledSentimentMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            let plan = self.compiler.prepare(document, control, &mut self.encoding_work)?;
            let record = prepare_document(plan, &self.compiler.seal, &self.driver,
                self.compiler.limits, self.compiler.preparation_bytes, control)?;
            self.run.check_available(record.plan.planned_work(), record.mask_node_visits)?;
            Ok(PreparedInt8SampledSentimentDocumentRecord { record })
        })();
        settle(result, self.driver.clean(), &mut self.run)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { transport_work(prepared.model_work()) }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.run.state = State::Failed;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_document(prepared.record, &self.compiler.seal, &mut self.driver,
            &mut self.admission, &mut self.run, context, control)
    }
}

// Only private seams accept synthetic fixtures. The public constructor always
// uses a model-bound compiler, real native driver and sealed result verifier.
trait DocumentPlan {
    type Output;
    fn facts(&self) -> Int8SampledSentimentDocumentPlanFacts<'_>;
    fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure>;
    fn verify_completed<C: DecodeStepControl>(&self, output: &Self::Output,
        masks: u64, control: &mut C) -> Result<(), BatchItemFailure>;
}
impl DocumentPlan for PreparedInt8SampledSentimentMap {
    type Output = Int8SampledSentimentMapRun;
    fn facts(&self) -> Int8SampledSentimentDocumentPlanFacts<'_> {
        Int8SampledSentimentDocumentPlanFacts { identity: self.execution_identity(), model_work: self.planned_work(),
            sample_count: self.sample_count(), chunk_count: self.chunk_count(), required_context: self.required_context(),
            original_task: self.task_budget(), max_result_bytes: self.max_result_bytes(),
            max_output_tokens: self.max_output_tokens(), preparation_bytes: self.preparation_requirement_bytes() }
    }
    fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
        if admitted != self.execution_identity() { Err(BatchItemFailure::fatal(BatchCode::Admission)) } else { Ok(()) }
    }
    fn verify_completed<C: DecodeStepControl>(&self, output: &Self::Output,
        masks: u64, control: &mut C) -> Result<(), BatchItemFailure> {
        self.preflight_metadata().verify_completed_with_control(output, control).map_err(map_failure)?;
        if output.planned_mask_node_visit_charge != masks || output.mask_node_visit_charge > masks {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(())
    }
}
trait DocumentDriver<P: DocumentPlan> {
    fn prefill(&self) -> Option<Int8PrefillLimits> { None }
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn mask_charge(&self, plan: &P, budget: Int8JsonBudget, sampler_bytes: u64) -> Result<u64, BatchItemFailure>;
    fn execute<C: DecodeStepControl>(&mut self, plan: &P, budget: Int8JsonBudget,
        sampler_bytes: u64, control: &mut C) -> Result<P::Output, BatchItemFailure>;
}
struct NativeDriver<'e, 'w> {
    engine: &'e mut StrictInt8Engine<'w>,
    vocabulary: Arc<SampledSentimentVocabulary>,
    prefill: Option<Int8PrefillLimits>,
}
impl DocumentDriver<PreparedInt8SampledSentimentMap> for NativeDriver<'_, '_> {
    fn prefill(&self) -> Option<Int8PrefillLimits> { self.prefill }
    fn clean(&self) -> bool { clean(self.engine) }
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn mask_charge(&self, plan: &PreparedInt8SampledSentimentMap, budget: Int8JsonBudget,
        sampler_bytes: u64) -> Result<u64, BatchItemFailure> {
        plan.complete_mask_charge(&self.vocabulary, budget, sampler_bytes).map_err(map_failure)
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt8SampledSentimentMap, budget: Int8JsonBudget,
        sampler_bytes: u64, control: &mut C) -> Result<Int8SampledSentimentMapRun, BatchItemFailure> {
        plan.execute_scheduled_with_control(self.engine, &self.vocabulary, budget, sampler_bytes,
            self.prefill, control).map_err(map_failure)
    }
}
fn document_preflight<P: DocumentPlan, D: DocumentDriver<P>>(plan: &P, driver: &D,
    limits: Int8SampledSentimentBatchLimits) -> Result<(u64, u64), BatchItemFailure> {
    validate_prefill(driver.prefill()).map_err(BatchItemFailure::fatal)?;
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let facts = plan.facts();
    if facts.identity.task_spec != "sentiment-v1" || facts.identity.sampler_version != JSON_SAMPLER_VERSION
        || facts.sample_count == 0 || facts.chunk_count == 0 || facts.required_context == 0
        || !nonzero(facts.model_work) || facts.original_task.validate().is_err() || facts.max_output_tokens == 0 {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let required = Int8MemoryRequirement::for_context(driver.capacity())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if facts.required_context > driver.capacity() || required.kv_bytes > facts.original_task.max_kv_bytes
        || !fits(facts.model_work, limits.max_item_work) {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    if facts.max_result_bytes == 0 { return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit)); }
    let masks = driver.mask_charge(plan, native_budget(facts.model_work, required.kv_bytes,
        limits.masks.max_visits_per_item, limits), limits.max_sampler_bytes)?;
    if masks == 0 || masks > limits.masks.max_visits_per_item {
        return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
    }
    Ok((required.kv_bytes, masks))
}
fn prepare_document<P: DocumentPlan, D: DocumentDriver<P>, C: DecodeStepControl>(plan: P,
    seal: &Arc<()>, driver: &D, limits: Int8SampledSentimentBatchLimits, preparation_bytes: u64,
    control: &mut C) -> Result<PreparedRecord<P>, BatchItemFailure> {
    checkpoint(control)?;
    if plan.facts().preparation_bytes == 0 || plan.facts().preparation_bytes > preparation_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let (_, mask_node_visits) = document_preflight(&plan, driver, limits)?;
    checkpoint(control)?;
    Ok(PreparedRecord { plan, mask_node_visits, seal: Arc::clone(seal) })
}
#[allow(clippy::too_many_arguments)]
fn execute_document<P: DocumentPlan, D: DocumentDriver<P>, A: Int8SampledSentimentDocumentBatchAdmission,
    C: DecodeStepControl>(prepared: PreparedRecord<P>, seal: &Arc<()>, driver: &mut D,
    admission: &mut A, run: &mut RunState, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<P::Output, A::Guard>, BatchItemFailure> {
    run.enter()?;
    let result = (|| {
        if !Arc::ptr_eq(&prepared.seal, seal) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let facts = prepared.plan.facts();
        run.charge(facts.model_work, prepared.mask_node_visits, context)?;
        checkpoint(control)?;
        let (kv_bytes, masks) = document_preflight(&prepared.plan, driver, run.limits)?;
        if masks != prepared.mask_node_visits { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        let work = facts.model_work;
        let admitted = admission.admit(Int8SampledSentimentDocumentAdmission {
            facts, kv_reservation_bytes: kv_bytes, mask_node_visits: masks });
        let (identity, guard) = match admitted {
            Err(error) if error.fault.cancellation.is_some() => return Err(error),
            other => { checkpoint(control)?; other? },
        };
        prepared.plan.verify_identity(&identity)?;
        checkpoint(control)?;
        let output = match driver.execute(&prepared.plan, native_budget(work, kv_bytes, masks, run.limits),
            run.limits.max_sampler_bytes, control) {
            Err(error) if error.fault.cancellation.is_some() => return Err(error),
            other => { checkpoint(control)?; other? },
        };
        prepared.plan.verify_completed(&output, masks, control)?;
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    settle(result, driver.clean(), run)
}
fn map_failure(error: Int8SampledSentimentMapError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int8SampledSentimentMapError::Sampled(error) => task_failure(error),
        Int8SampledSentimentMapError::Shared(error) => shared_failure(error),
        Int8SampledSentimentMapError::Limits | Int8SampledSentimentMapError::EmptySource =>
            BatchItemFailure::reject(BatchCode::Planning),
        Int8SampledSentimentMapError::Work => BatchItemFailure::reject(BatchCode::WorkLimit),
        Int8SampledSentimentMapError::Output => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        Int8SampledSentimentMapError::Admission => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        Int8SampledSentimentMapError::Allocation => BatchItemFailure::fatal(BatchCode::Allocation),
        Int8SampledSentimentMapError::Chunk(error) => chunk_failure(error),
        Int8SampledSentimentMapError::Cancelled(_) => BatchItemFailure::fatal(BatchCode::Execution),
    }
}
fn shared_failure(error: SampledSentimentMapError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        SampledSentimentMapError::Limits | SampledSentimentMapError::EmptySource =>
            BatchItemFailure::reject(BatchCode::Planning),
        SampledSentimentMapError::Work => BatchItemFailure::reject(BatchCode::WorkLimit),
        SampledSentimentMapError::Output => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        SampledSentimentMapError::Admission | SampledSentimentMapError::Aggregation =>
            BatchItemFailure::fatal(BatchCode::InvalidExecution),
        SampledSentimentMapError::Allocation => BatchItemFailure::fatal(BatchCode::Allocation),
        SampledSentimentMapError::Serialization => BatchItemFailure::fatal(BatchCode::Serialization),
        SampledSentimentMapError::Sampled(error) => super::super::task_failure(error),
        SampledSentimentMapError::Chunk(error) => chunk_failure(error),
        SampledSentimentMapError::Cancelled(_) => BatchItemFailure::fatal(BatchCode::Execution),
    }
}
fn chunk_failure(error: crate::tasks::mapreduce::MapReduceError) -> BatchItemFailure {
    match error {
        crate::tasks::mapreduce::MapReduceError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
        crate::tasks::mapreduce::MapReduceError::Cancelled => BatchItemFailure::fatal(BatchCode::Execution),
        _ => BatchItemFailure::reject(BatchCode::Planning),
    }
}

#[cfg(test)] mod tests;
