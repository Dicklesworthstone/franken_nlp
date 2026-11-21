//! Complete sampled-sentiment documents on one resident native allocation.
//!
//! Every document is losslessly partitioned and fully planned before its
//! first forward. One stream-wide native/mask ledger prices the complete
//! cold schedule, including rejected admissions and failed executions. A
//! flush, shorter result or reused prefix never replenishes that allowance.

use super::*;
use crate::tasks::sentiment::sampled::{SampledSentimentOptions,
    long::{PreparedSampledSentimentMap, SampledSentimentMapError,
        SampledSentimentMapLimits, SampledSentimentMapRun,
        memory::{MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES, SampledSentimentMapMemoryLimits}}};

/// Pure complete-document preparation quote shared by CLI, hosted streams and
/// durable jobs. The largest valid four-axis request is admitted before input.
/// Native/sampler buffers, transport and the guarded final result are separate
/// charges; this is the existing complete-map input quote, not a new estimate.
pub fn sampled_sentiment_document_preparation_bytes(group_width: usize, task: TaskBudget,
    planning: SampledSentimentLimits, mapping: SampledSentimentMapLimits,
    options: &SampledSentimentOptions, encoding: &UntrustedEncoding) -> Result<u64, BatchFault> {
    SampledSentimentMapMemoryLimits {
        context_tokens: mapping.chunks.context_tokens,
        preparation_reserve_bytes: MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: 1,
        max_axes: SentimentAxis::ALL.len(),
        max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?,
    }.requirements(group_width, task, planning, mapping, options, encoding)
        .map(|required| required.input_bytes).map_err(|_| BatchCode::InvalidLimits.into())
}

/// Actual model-bound compiler, exact whole-document limits and one immutable
/// vocabulary. None of these authorities can be replaced by record arguments.
pub struct SampledSentimentDocumentPlanner {
    planner: SampledSentimentPlanner,
    vocabulary: Arc<SampledSentimentVocabulary>,
    task_budget: TaskBudget,
    mapping: SampledSentimentMapLimits,
    limits: SampledSentimentBatchLimits,
    preparation_bytes: u64,
    seal: Arc<()>,
}
impl SampledSentimentDocumentPlanner {
    pub fn new(planner: SampledSentimentPlanner, vocabulary: Arc<SampledSentimentVocabulary>,
        task_budget: TaskBudget, mapping: SampledSentimentMapLimits,
        limits: SampledSentimentBatchLimits) -> Result<Self, BatchFault> {
        limits.validate()?;
        mapping.validate(task_budget).map_err(|_| BatchCode::InvalidLimits)?;
        if task_budget != planner.task_budget() || planner.execution_identity().task_spec != "sentiment-v1"
            || planner.execution_identity().sampler_version != JSON_SAMPLER_VERSION
            || !fits(mapping.max_model_work, limits.max_item_work) {
            return Err(BatchCode::InvalidLimits.into());
        }
        constrained_int4::check_profile(planner.execution_identity(), planner.group_width())
            .map_err(|_| BatchCode::Admission)?;
        let preparation_bytes = sampled_sentiment_document_preparation_bytes(planner.group_width(),
            task_budget, planner.limits(), mapping, planner.sampling_options(), planner.input_encoding())?;
        Ok(Self { planner, vocabulary, task_budget, mapping, limits, preparation_bytes, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn group_width(&self) -> usize { self.planner.group_width() }
    pub fn task_budget(&self) -> TaskBudget { self.task_budget }
    pub fn mapping(&self) -> SampledSentimentMapLimits { self.mapping }
    pub fn limits(&self) -> SampledSentimentBatchLimits { self.limits }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.input_encoding() }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }
    #[cfg(all(feature = "metadata-store", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn durable_configuration(&self) -> (&SampledSentimentOptions, SampledSentimentLimits) {
        (self.planner.durable_options(), self.planner.limits())
    }

    fn prepare<C: DecodeStepControl>(&self, document: BatchDocument<SampledSentimentBatchArgs>,
        control: &mut C) -> Result<PreparedSampledSentimentMap, BatchItemFailure> {
        checkpoint(control)?;
        let args = document.task_args.unwrap_or_default();
        let budget = item_budget(args.budget, self.task_budget)?;
        // Source, original chunk coordinates and the stable caller ID all
        // enter the private draw keys. Transport sequence/epoch never do.
        self.planner.plan_document_for_item_with_control(&SentimentRequest {
            document: document.text, axes: args.axes, budget }, self.mapping, &document.id, control)
            .map_err(map_failure)
    }
}

/// The original task budget applies to each chunk. Complete-document output
/// and work limits are explicit separate facts, so no host mistakes one
/// chunk's output ceiling for the whole document's retained result.
pub struct SampledSentimentDocumentPlanFacts<'a> {
    pub identity: &'a ExecutionIdentity,
    pub model_work: Int4Work,
    pub group_width: usize,
    pub sample_count: usize,
    pub chunk_count: usize,
    pub required_context: usize,
    pub original_task: TaskBudget,
    pub max_result_bytes: u64,
    pub max_output_tokens: u64,
    pub preparation_bytes: u64,
}
pub struct SampledSentimentDocumentAdmission<'a> {
    pub facts: SampledSentimentDocumentPlanFacts<'a>,
    pub kv_reservation_bytes: u64,
    pub mask_node_visits: u64,
}
pub trait SampledSentimentDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: SampledSentimentDocumentAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure>;
}

pub struct PreparedSampledSentimentDocumentRecord { record: PreparedRecord<PreparedSampledSentimentMap> }
impl PreparedSampledSentimentDocumentRecord {
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.record.plan.execution_identity() }
    pub fn model_work(&self) -> Int4Work { self.record.plan.planned_work() }
    pub fn mask_node_visits(&self) -> u64 { self.record.mask_node_visits }
    pub fn max_result_bytes(&self) -> u64 { self.record.plan.max_result_bytes() }
}

/// One process-admitted engine executes every original chunk, axis and sample
/// in every record. Native sessions still begin/end with empty logical KV.
/// A terminal fault, dirty completion or unwind permanently closes this
/// adapter, and the output guard survives until delivery finishes.
pub struct NativeSampledSentimentDocumentBatch<'e, 'w, A: SampledSentimentDocumentBatchAdmission> {
    compiler: SampledSentimentDocumentPlanner,
    driver: DocumentNativeDriver<'e, 'w>,
    admission: A,
    run: RunState,
    restored: bool,
}
impl<'e, 'w, A: SampledSentimentDocumentBatchAdmission> NativeSampledSentimentDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: SampledSentimentDocumentPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        Self::new_scheduled(compiler, engine, admission, None, false)
    }
    pub fn new_layer_major(compiler: SampledSentimentDocumentPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A, prefill: Int4PrefillLimits) -> Result<Self, BatchFault> {
        Self::new_scheduled(compiler, engine, admission, Some(prefill), false)
    }
    /// Reuse a prefix only within one chunk and axis. The whole cold document
    /// remains the attempt debit; no prefix crosses a chunk/item boundary.
    pub fn new_reusing_prefill(compiler: SampledSentimentDocumentPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A, prefill: Option<Int4PrefillLimits>) -> Result<Self, BatchFault> {
        Self::new_scheduled(compiler, engine, admission, prefill, true)
    }
    fn new_scheduled(compiler: SampledSentimentDocumentPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A, prefill: Option<Int4PrefillLimits>, reuse_prefill: bool) -> Result<Self, BatchFault> {
        compiler.limits.validate()?;
        validate_prefill(prefill)?;
        constrained_int4::check_model(compiler.base_execution_identity(), engine.model_identity())
            .map_err(|_| BatchCode::Admission)?;
        if compiler.group_width() != engine.model_identity().group_width() || !clean(engine)
            || compiler.mapping.chunks.context_tokens > engine.kv_cache().capacity_positions() {
            return Err(BatchCode::InvalidExecution.into());
        }
        let run = RunState::new(compiler.limits);
        let vocabulary = Arc::clone(&compiler.vocabulary);
        Ok(Self { compiler, driver: DocumentNativeDriver { engine, vocabulary, prefill, reuse_prefill },
            admission, run, restored: false })
    }
    pub fn remaining_model_work(&self) -> Int4Work { self.run.remaining }
    pub fn remaining_mask_visits(&self) -> u64 { self.run.remaining_masks }
    pub fn is_poisoned(&self) -> bool { self.run.state != State::Ready || !self.driver.clean() }

    /// Tighten a fresh adapter by authenticated prior attempt debits. This
    /// can only subtract once before execution; it can never replenish work.
    /// Durable storage remains the authority for the supplied history.
    pub fn restore_spent_work(&mut self, spent: Int4Work, masks: u64) -> Result<(), BatchFault> {
        if self.restored || self.run.state != State::Ready || self.run.last_sequence != 0 || !self.driver.clean() {
            return Err(BatchCode::InvalidExecution.into());
        }
        let remaining = subtract(self.run.remaining, spent).map_err(|error| error.fault)?;
        let remaining_masks = self.run.remaining_masks.checked_sub(masks).ok_or(BatchCode::WorkLimit)?;
        self.run.remaining = remaining; self.run.remaining_masks = remaining_masks; self.restored = true;
        Ok(())
    }
}
impl<A: SampledSentimentDocumentBatchAdmission> BatchProcessor for NativeSampledSentimentDocumentBatch<'_, '_, A> {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedSampledSentimentDocumentRecord;
    type Output = GuardedOutput<SampledSentimentMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            let plan = self.compiler.prepare(document, control)?;
            let record = prepare_document(plan, &self.compiler.seal, &self.driver,
                self.compiler.limits, self.compiler.preparation_bytes, control)?;
            self.run.check_available(record.plan.planned_work(), record.mask_node_visits)?;
            Ok(PreparedSampledSentimentDocumentRecord { record })
        })();
        settle(result, self.driver.clean(), &mut self.run)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { transport_work(prepared.model_work()) }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.run.state = State::Failed; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_document(prepared.record, &self.compiler.seal, &mut self.driver,
            &mut self.admission, &mut self.run, context, control)
    }
}

// Private static seams exercise the actual accounting/ownership protocol with
// synthetic fixtures. Public callers cannot mint a native completion or swap
// the task compiler, driver or independent whole-document validator.
trait DocumentPlan {
    type Output;
    fn facts(&self) -> SampledSentimentDocumentPlanFacts<'_>;
    fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure>;
    fn verify_completed<C: DecodeStepControl>(&self, output: &Self::Output,
        masks: u64, reuse_prefill: bool, control: &mut C) -> Result<(), BatchItemFailure>;
}
impl DocumentPlan for PreparedSampledSentimentMap {
    type Output = SampledSentimentMapRun;
    fn facts(&self) -> SampledSentimentDocumentPlanFacts<'_> {
        SampledSentimentDocumentPlanFacts { identity: self.execution_identity(), model_work: self.planned_work(),
            group_width: self.group_width(), sample_count: self.sample_count(), chunk_count: self.chunk_count(),
            required_context: self.required_context(), original_task: self.task_budget(),
            max_result_bytes: self.max_result_bytes(), max_output_tokens: self.max_output_tokens(),
            preparation_bytes: self.preparation_requirement_bytes() }
    }
    fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
        if admitted != self.execution_identity() { Err(BatchItemFailure::fatal(BatchCode::Admission)) } else { Ok(()) }
    }
    fn verify_completed<C: DecodeStepControl>(&self, output: &SampledSentimentMapRun,
        masks: u64, reuse_prefill: bool, control: &mut C) -> Result<(), BatchItemFailure> {
        self.preflight_metadata().verify_completed_with_control(output, reuse_prefill, control).map_err(map_failure)?;
        if output.planned_mask_node_visit_charge != masks || output.mask_node_visit_charge > masks {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(())
    }
}
trait DocumentDriver<P: DocumentPlan> {
    fn prefill(&self) -> Option<Int4PrefillLimits> { None }
    fn reuse_prefill(&self) -> bool { false }
    fn clean(&self) -> bool;
    fn capacity(&self) -> usize;
    fn group_width(&self) -> usize;
    fn mask_charge(&self, plan: &P, budget: Int4JsonBudget, sampler_bytes: u64) -> Result<u64, BatchItemFailure>;
    fn execute<C: DecodeStepControl>(&mut self, plan: &P, budget: Int4JsonBudget,
        sampler_bytes: u64, control: &mut C) -> Result<P::Output, BatchItemFailure>;
}
struct DocumentNativeDriver<'e, 'w> {
    engine: &'e mut StrictInt4Engine<'w>, vocabulary: Arc<SampledSentimentVocabulary>,
    prefill: Option<Int4PrefillLimits>, reuse_prefill: bool,
}
impl DocumentDriver<PreparedSampledSentimentMap> for DocumentNativeDriver<'_, '_> {
    fn prefill(&self) -> Option<Int4PrefillLimits> { self.prefill }
    fn reuse_prefill(&self) -> bool { self.reuse_prefill }
    fn clean(&self) -> bool { clean(self.engine) }
    fn capacity(&self) -> usize { self.engine.kv_cache().capacity_positions() }
    fn group_width(&self) -> usize { self.engine.model_identity().group_width() }
    fn mask_charge(&self, plan: &PreparedSampledSentimentMap, budget: Int4JsonBudget, sampler_bytes: u64)
        -> Result<u64, BatchItemFailure> {
        plan.complete_mask_charge(&self.vocabulary, budget, sampler_bytes).map_err(map_failure)
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedSampledSentimentMap, budget: Int4JsonBudget,
        sampler_bytes: u64, control: &mut C) -> Result<SampledSentimentMapRun, BatchItemFailure> {
        plan.execute_scheduled_with_control(self.engine, &self.vocabulary, budget,
            sampler_bytes, self.prefill, self.reuse_prefill, control).map_err(map_failure)
    }
}

fn document_preflight<P: DocumentPlan, D: DocumentDriver<P>>(plan: &P, driver: &D,
    limits: SampledSentimentBatchLimits) -> Result<(u64, u64), BatchItemFailure> {
    validate_prefill(driver.prefill()).map_err(BatchItemFailure::fatal)?;
    if !driver.clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
    let facts = plan.facts();
    if facts.group_width != driver.group_width() || facts.identity.task_spec != "sentiment-v1"
        || facts.identity.sampler_version != JSON_SAMPLER_VERSION || facts.sample_count == 0
        || facts.chunk_count == 0 || facts.required_context == 0 || !nonzero(facts.model_work)
        || facts.original_task.validate().is_err() || facts.max_output_tokens == 0 {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let required = Int4MemoryRequirement::for_context(driver.capacity(), driver.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if facts.required_context > driver.capacity() || required.kv_bytes > facts.original_task.max_kv_bytes
        || !fits(facts.model_work, limits.max_item_work) {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    // Complete-map bytes have their own admitted ceiling. The original task
    // output limit belongs to each chunk and must not silently truncate a map.
    if facts.max_result_bytes == 0 { return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit)); }
    let masks = driver.mask_charge(plan, native_budget(facts.model_work, required.kv_bytes,
        limits.masks.max_visits_per_item, limits), limits.max_sampler_bytes)?;
    if masks == 0 || masks > limits.masks.max_visits_per_item {
        return Err(BatchItemFailure::reject(BatchCode::WorkLimit));
    }
    Ok((required.kv_bytes, masks))
}
fn prepare_document<P: DocumentPlan, D: DocumentDriver<P>, C: DecodeStepControl>(plan: P,
    seal: &Arc<()>, driver: &D, limits: SampledSentimentBatchLimits,
    preparation_bytes: u64, control: &mut C) -> Result<PreparedRecord<P>, BatchItemFailure> {
    checkpoint(control)?;
    if plan.facts().preparation_bytes == 0 || plan.facts().preparation_bytes > preparation_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let (_, mask_node_visits) = document_preflight(&plan, driver, limits)?;
    checkpoint(control)?;
    Ok(PreparedRecord { plan, mask_node_visits, seal: Arc::clone(seal) })
}
#[allow(clippy::too_many_arguments)]
fn execute_document<P: DocumentPlan, D: DocumentDriver<P>, A: SampledSentimentDocumentBatchAdmission,
    C: DecodeStepControl>(prepared: PreparedRecord<P>, seal: &Arc<()>, driver: &mut D,
    admission: &mut A, run: &mut RunState, context: BatchRequestContext, control: &mut C)
    -> Result<GuardedOutput<P::Output, A::Guard>, BatchItemFailure> {
    run.enter()?;
    let result = (|| {
        if !Arc::ptr_eq(&prepared.seal, seal) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let facts = prepared.plan.facts();
        // Atomically debit every native dimension and the whole-document mask
        // allowance before the caller's admission hook or first native step.
        run.charge(facts.model_work, prepared.mask_node_visits, context)?;
        checkpoint(control)?;
        let (kv_bytes, masks) = document_preflight(&prepared.plan, driver, run.limits)?;
        if masks != prepared.mask_node_visits { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        let work = facts.model_work;
        let (identity, guard) = admission.admit(SampledSentimentDocumentAdmission {
            facts, kv_reservation_bytes: kv_bytes, mask_node_visits: masks })?;
        prepared.plan.verify_identity(&identity)?;
        checkpoint(control)?;
        let output = driver.execute(&prepared.plan, native_budget(work, kv_bytes, masks, run.limits),
            run.limits.max_sampler_bytes, control)?;
        prepared.plan.verify_completed(&output, masks, driver.reuse_prefill(), control)?;
        checkpoint(control)?;
        Ok(GuardedOutput::new(output, guard))
    })();
    settle(result, driver.clean(), run)
}

fn map_failure(error: SampledSentimentMapError) -> BatchItemFailure {
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
        SampledSentimentMapError::Sampled(error) => task_failure(error),
        SampledSentimentMapError::Chunk(error) => match error {
            crate::tasks::mapreduce::MapReduceError::AllocationRefused => BatchItemFailure::fatal(BatchCode::Allocation),
            crate::tasks::mapreduce::MapReduceError::Cancelled => BatchItemFailure::fatal(BatchCode::Execution),
            _ => BatchItemFailure::reject(BatchCode::Planning),
        },
        SampledSentimentMapError::Cancelled(_) => BatchItemFailure::fatal(BatchCode::Execution),
    }
}

#[cfg(test)] mod tests;
