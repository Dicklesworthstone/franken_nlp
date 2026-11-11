//! Complete-document classification on one resident native INT4 stream.
//! The original request is owned between preparation and execution. Replanning
//! verifies every frozen chunk identity; it never borrows through its own owner.
use std::sync::Arc;
use crate::{
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput},
    execution_identity::ExecutionIdentity,
    native_engine::{constrained_int4, decode::DecodeStepControl, int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryRequirement, Int4ModelWeights, Int4RunBudget, Int4Work, StrictInt4Engine,
            StrictInt4Error, scoring::{Int4ScoringBudget, Int4ScoringError}}},
    tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode,
        ClassificationRequest,
        grouped_int4::{Int4ClassificationError, Int4ClassificationPlanner,
            long::{Int4ClassificationMapCohortBudget, Int4ClassificationMapError, Int4ClassificationMapLimits,
                Int4ClassificationMapPreflight, Int4ClassificationMapRun,
                memory::{Int4ClassificationMapCohortMemoryRequirement, Int4ClassificationMapMemoryLimits}}}},
        ir::TaskBudget},
};
use super::ClassificationBatchArgs;
use crate::batch::int4_document::{RunState, subtract};
pub(crate) use crate::batch::int4_document::{fits, nonzero};

mod cohort;
pub use cohort::Int4ClassificationDocumentRun;
pub(crate) use cohort::cohort_head_limit;
use cohort::{Backend, CohortSchedule};

/// Frozen identities, admission copies and document metadata retained while
/// the original request is recompiled. This is additional to the map quote.
pub const CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;

/// Owns the actual model-bound compiler and optional complete task defaults.
/// `mapping.max_model_work` is a per-document cap; `max_run_work` is one
/// nonrenewable invocation cap across all items and flush boundaries.
pub struct Int4ClassificationDocumentBatchPlanner {
    planner: Int4ClassificationPlanner,
    defaults: Option<ClassificationBatchArgs>,
    mapping: Int4ClassificationMapLimits,
    max_run_work: Int4Work,
    cohorts: Option<CohortSchedule>,
    seal: Arc<()>,
}
impl Int4ClassificationDocumentBatchPlanner {
    pub fn new(planner: Int4ClassificationPlanner, defaults: Option<ClassificationBatchArgs>,
        mapping: Int4ClassificationMapLimits, max_run_work: Int4Work) -> Result<Self, BatchFault> {
        planner.document_encoding().validate().map_err(|_| BatchCode::InvalidLimits)?;
        validate_limits(planner.task_budget(), planner.planning_limits(), mapping, max_run_work)?;
        constrained_int4::check_profile(planner.execution_identity(), planner.group_width())
            .map_err(|_| BatchCode::Admission)?;
        if planner.execution_identity().task_spec != "classify-v1" { return Err(BatchCode::Admission.into()); }
        if let Some(args) = &defaults {
            validate_args(args, planner.task_budget(), planner.planning_limits()).map_err(|e| e.fault)?;
            // Byte-mode capacity is content-free. Compact metadata capacity
            // is measured inside each complete document's own allowance.
            if planner.document_encoding() == &UntrustedEncoding::ByteFallback {
                let request = args.clone().into_request("x".to_owned());
                planner.max_document_bytes(&request, mapping.chunks.context_tokens)
                    .map_err(|e| map_failure(e.into(), false).fault)?;
            }
        }
        Ok(Self { planner, defaults, mapping, max_run_work, cohorts: None, seal: Arc::new(()) })
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn group_width(&self) -> usize { self.planner.group_width() }
    pub fn task_budget(&self) -> TaskBudget { self.planner.task_budget() }
    pub fn planning_limits(&self) -> ClassificationLimits { self.planner.planning_limits() }
    pub fn mapping_limits(&self) -> Int4ClassificationMapLimits { self.mapping }
    pub fn max_run_work(&self) -> Int4Work { self.max_run_work }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.planner.document_encoding() }
    pub fn cohort_chunks(&self) -> Option<usize> { self.cohorts.map(|schedule| schedule.chunks) }
    /// Opt into actual chunk-by-head native windows. Unknown records reserve
    /// the largest valid head count; an oversized actual record is refused,
    /// never split into a different schedule or silently executed serially.
    pub fn with_chunk_cohorts(mut self, chunks: usize, memory: Int4ClassificationMapMemoryLimits)
        -> Result<Self, BatchFault> {
        let schedule = CohortSchedule::new(self.group_width(), self.task_budget(), self.planning_limits(),
            self.mapping, chunks, memory)?;
        if let Some(defaults) = &self.defaults {
            schedule.check_request(defaults.mode, defaults.labels.len(), defaults.budget).map_err(|e| e.fault)?;
        }
        self.cohorts = Some(schedule);
        // Prepared serial or differently grouped items cannot transfer into
        // this schedule even though the logical classifier identity is fixed.
        self.seal = Arc::new(());
        Ok(self)
    }
    pub fn cohort_requirements(&self) -> Result<Int4ClassificationMapCohortMemoryRequirement, BatchFault> {
        self.cohorts.map(|schedule| schedule.required).ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    pub fn max_result_bytes(&self) -> u64 {
        self.cohorts.map_or(self.mapping.max_result_bytes, |schedule| schedule.required.memory.max_output_bytes)
    }
    /// Actual transferred capacities, so an embedding caller cannot hide a
    /// greatly overallocated defaults owner inside a small logical label set.
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        let Some(args) = &self.defaults else { return Ok(0); };
        let base = (args.labels.capacity() as u64).checked_mul(std::mem::size_of::<ClassificationLabel>() as u64)
            .ok_or(BatchCode::InvalidLimits)?;
        args.labels.iter().try_fold(base, |bytes, label| bytes.checked_add(label.id.capacity() as u64)
            .and_then(|n| n.checked_add(label.description.capacity() as u64)).ok_or_else(|| BatchCode::InvalidLimits.into()))
    }
    fn request(&self, document: BatchDocument<ClassificationBatchArgs>) -> Result<ClassificationRequest, BatchItemFailure> {
        let request = build_request(document, &self.defaults, self.mapping, self.task_budget(), self.planning_limits())?;
        if let Some(schedule) = self.cohorts {
            schedule.check_request(request.mode, request.labels.len(), request.budget)?;
        }
        Ok(request)
    }
}

/// An opaque original request and a complete frozen preparation snapshot.
/// There is no deserializer, public source replacement or native-score input.
pub struct PreparedInt4ClassificationDocument {
    request: ClassificationRequest,
    expected: Int4ClassificationMapPreflight,
    identities: Vec<ExecutionIdentity>,
    required_context: usize,
    encoding_work: SafeBpeWork,
    seal: Arc<()>,
}
impl PreparedInt4ClassificationDocument {
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn planned_work(&self) -> Int4Work { self.expected.planned_work() }
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    /// Cumulative tokenizer work already spent by this document operation.
    /// Reconstruction spends the remainder of the same fixed allowance.
    pub fn encoding_work(&self) -> SafeBpeWork { self.encoding_work }
}

/// Admission covers every exact chunk identity and the complete document
/// result. Summed forwards do not describe the reused cache's capacity.
pub struct Int4ClassificationDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub group_width: usize,
    pub required_context: usize,
    pub task_budget: TaskBudget,
    /// Physical windows are independent of each chunk's logical task budget.
    pub cohort_chunks: Option<usize>,
    pub head_count: usize,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
}
pub trait Int4ClassificationDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4ClassificationDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt4ClassificationDocumentBatch<'e, 'w, A: Int4ClassificationDocumentBatchAdmission> {
    compiler: Int4ClassificationDocumentBatchPlanner,
    backend: Backend<'e, 'w>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, A: Int4ClassificationDocumentBatchAdmission> NativeInt4ClassificationDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: Int4ClassificationDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        Self::configured(compiler, Backend::Serial(engine), admission)
    }
    pub fn new_cohorts(compiler: Int4ClassificationDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        budget: Int4ClassificationMapCohortBudget, kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        Self::configured(compiler, Backend::Cohorts { weights, budget, kernel }, admission)
    }
    fn configured(compiler: Int4ClassificationDocumentBatchPlanner, backend: Backend<'e, 'w>, admission: A)
        -> Result<Self, BatchFault> {
        backend.check_capacity(&compiler, compiler.task_budget()).map_err(|e| e.fault)?;
        let run = RunState::new(compiler.max_run_work);
        Ok(Self { compiler, backend, admission, run })
    }
    pub fn remaining_work(&self) -> Int4Work { self.run.remaining() }
    pub fn is_poisoned(&self) -> bool { self.run.is_unavailable() || !self.backend.is_clean() }
}
impl<A: Int4ClassificationDocumentBatchAdmission> BatchProcessor for NativeInt4ClassificationDocumentBatch<'_, '_, A> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt4ClassificationDocument;
    type Output = GuardedOutput<Int4ClassificationDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.fail();
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            checkpoint(control)?;
            let request = self.compiler.request(document)?;
            let kv = self.backend.check_capacity(&self.compiler, request.budget)?;
            let mut encoding_work = SafeBpeWork::default();
            let plan = self.compiler.planner.plan_document_with_encoding_work(&request, self.compiler.mapping,
                control, &mut encoding_work)
                .map_err(|e| map_failure(e, false))?;
            subtract(self.run.remaining(), plan.planned_work())?;
            let identities = plan.identities().map_err(|e| map_failure(e, false))?;
            self.backend.preflight(&self.compiler, &plan, &identities, kv, control)
                .map_err(|e| map_failure(e, false))?;
            let expected = plan.preflight_metadata(); let required_context = plan.required_context();
            drop(plan);
            checkpoint(control)?;
            Ok(PreparedInt4ClassificationDocument { request, expected, identities, required_context, encoding_work,
                seal: Arc::clone(&self.compiler.seal) })
        })();
        self.run.finish_preparation(result, self.backend.is_clean())
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.fail();
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, mut prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            if !Arc::ptr_eq(&prepared.seal, &self.compiler.seal) {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            // Charge before replanning, output admission or a native call.
            // All later failures keep this debit and close the processor.
            self.run.charge(prepared.planned_work(), context)?;
            checkpoint(control)?;
            let kv = self.backend.check_capacity(&self.compiler, prepared.request.budget)?;
            let plan = self.compiler.planner.plan_document_with_encoding_work(&prepared.request, self.compiler.mapping,
                control, &mut prepared.encoding_work)
                .map_err(|e| map_failure(e, true))?;
            let actual_ids = plan.identities().map_err(|e| map_failure(e, true))?;
            if plan.preflight_metadata() != prepared.expected || actual_ids != prepared.identities
                || plan.required_context() != prepared.required_context {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            drop(actual_ids);
            self.backend.preflight(&self.compiler, &plan, &prepared.identities, kv, control)
                .map_err(|e| map_failure(e, true))?;
            let (admitted, guard) = self.admission.admit(Int4ClassificationDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.planned_work(),
                group_width: plan.group_width(), required_context: prepared.required_context,
                task_budget: prepared.request.budget, cohort_chunks: self.compiler.cohort_chunks(),
                head_count: plan.head_count(), kv_reservation_bytes: kv,
                max_result_bytes: self.compiler.max_result_bytes(),
            })?;
            // The embedding host must authorize exactly the complete frozen
            // request, not just its first chunk or another compiler identity.
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            checkpoint(control)?;
            let output = self.backend.execute(&self.compiler, &plan, &admitted, kv, control)
                .map_err(|e| map_failure(e, true))?;
            match &output {
                Int4ClassificationDocumentRun::Serial(run) if self.compiler.cohort_chunks().is_none() =>
                    prepared.expected.verify_completed_with_control(run, control).map_err(|e| map_failure(e, true))?,
                Int4ClassificationDocumentRun::Cohorts(run) => {
                    let chunks = self.compiler.cohort_chunks().ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
                    prepared.expected.verify_completed_cohorts_with_control(run, chunks, control)
                        .map_err(|e| map_failure(e, true))?;
                }
                _ => return Err(BatchItemFailure::fatal(BatchCode::Admission)),
            }
            if !self.backend.is_clean() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
            checkpoint(control)?;
            Ok(GuardedOutput::new(output, guard))
        })();
        self.run.finish_execution(result, self.backend.is_clean())
    }
}

pub(crate) fn validate_limits(task: TaskBudget, planning: ClassificationLimits,
    mapping: Int4ClassificationMapLimits, max_run_work: Int4Work) -> Result<(), BatchFault> {
    mapping.validate(task).map_err(|_| BatchCode::InvalidLimits)?;
    planning.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if !nonzero(max_run_work) || !fits(mapping.max_model_work, max_run_work)
        || mapping.chunks.max_input_bytes > planning.max_input_bytes
        || mapping.chunks.context_tokens > planning.max_context_tokens {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn build_request(document: BatchDocument<ClassificationBatchArgs>, defaults: &Option<ClassificationBatchArgs>,
    mapping: Int4ClassificationMapLimits, task: TaskBudget, planning: ClassificationLimits)
    -> Result<ClassificationRequest, BatchItemFailure> {
    if document.text.is_empty() || document.text.len() > mapping.chunks.max_input_bytes {
        return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
    }
    // A supplied object replaces the entire default. Missing/null uses the
    // frozen default; neither case merges label, policy or budget data.
    let args = document.task_args.or_else(|| defaults.clone())
        .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
    validate_args(&args, task, planning)?;
    Ok(args.into_request(document.text))
}
fn validate_args(args: &ClassificationBatchArgs, ceiling: TaskBudget, limits: ClassificationLimits)
    -> Result<(), BatchItemFailure> {
    let budget = args.budget;
    if budget.validate().is_err() || budget.max_input_tokens > ceiling.max_input_tokens
        || budget.max_output_tokens > ceiling.max_output_tokens || budget.max_output_bytes > ceiling.max_output_bytes
        || budget.max_grammar_states > ceiling.max_grammar_states || budget.max_kv_bytes > ceiling.max_kv_bytes
        || args.labels.is_empty() || args.labels.len() > limits.max_labels
        || (args.mode == ClassificationMode::Exclusive && args.labels.len() < 2)
        || args.policy.minimum_candidate_weight_ppm > 1_000_000 || args.policy.minimum_margin_ppm > 1_000_000 {
        return Err(BatchItemFailure::reject(BatchCode::Planning));
    }
    let mut total = 0_usize;
    for (index, label) in args.labels.iter().enumerate() {
        total = total.checked_add(label.id.len()).and_then(|n| n.checked_add(label.description.len()))
            .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if label.id.trim().is_empty() || label.id.chars().any(char::is_control)
            || label.id.len() > limits.max_label_id_bytes || label.description.len() > limits.max_label_description_bytes
            || total > limits.max_total_label_bytes || args.labels[..index].iter().any(|old| old.id == label.id) {
            return Err(BatchItemFailure::reject(BatchCode::Planning));
        }
    }
    Ok(())
}
fn capacity(compiler: &Int4ClassificationDocumentBatchPlanner, task: TaskBudget, engine: &StrictInt4Engine<'_>)
    -> Result<u64, BatchItemFailure> {
    if !clean(engine) || engine.model_identity().group_width() != compiler.group_width()
        || engine.kv_cache().capacity_positions() < compiler.mapping.chunks.context_tokens {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), compiler.group_width())
        .map_err(|_| BatchItemFailure::fatal(BatchCode::InvalidExecution))?;
    if required.kv_bytes > task.max_kv_bytes { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
    Ok(required.kv_bytes)
}
fn clean(engine: &StrictInt4Engine<'_>) -> bool { !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0) }
fn budget(work: Int4Work, max_kv_bytes: u64) -> Int4ScoringBudget {
    Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes }
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4ClassificationMapError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use Int4ClassificationMapError as M;
    use Int4ClassificationError as C;
    let mut failure = match error {
        M::EmptySource => BatchItemFailure::reject(BatchCode::DocumentLimit),
        M::Limits | M::Chunk(_) => BatchItemFailure::reject(BatchCode::Planning),
        M::Work | M::Classification(C::WorkBudget | C::Native(StrictInt4Error::Work)
            | C::Scoring(Int4ScoringError::Native(StrictInt4Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        M::Allocation | M::Classification(C::Scoring(Int4ScoringError::Allocation))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        M::Output | M::Classification(C::Scoring(Int4ScoringError::OutputBudget))
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        M::Serialization | M::Classification(C::Scoring(Int4ScoringError::Serialization))
            => BatchItemFailure::fatal(BatchCode::Serialization),
        M::Classification(C::Planning(e)) => super::planning_failure(e),
        M::Classification(C::Native(StrictInt4Error::Context | StrictInt4Error::Memory)
            | C::Scoring(Int4ScoringError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        M::Admission | M::Classification(C::Identity) => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    if executing { failure.stop = true; }
    failure
}

#[cfg(test)] mod tests;
