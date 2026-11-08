//! Complete sentiment documents on one owned resident native invocation.
//! Frozen original requests are recompiled under the same sealed factory;
//! neither a partial axis bundle nor a partial document can be published.
use std::sync::Arc;
use serde::Serialize;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork, output::GuardedOutput,
        int4_document::{RunState, fits, nonzero, subtract}},
    execution_identity::ExecutionIdentity,
    native_engine::{constrained_int4, decode::DecodeStepControl, int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryRequirement, Int4ModelWeights, Int4RunBudget, Int4Work,
            StrictInt4Engine, StrictInt4Error, cohort::MAX_INT4_SEQUENCES,
            scoring::{Int4ScoringBudget, Int4ScoringError}}},
    tasks::{ir::TaskBudget, sentiment::{SentimentAxis, SentimentError, SentimentLimits,
        SentimentPlanningError, SentimentRequest, batch::SentimentBatchArgs,
        grouped_int4::{Int4SentimentError, Int4SentimentPlanner,
            long::{Int4SentimentMapError, Int4SentimentMapLimits, Int4SentimentMapPreflight,
                Int4SentimentMapRun, Int4SentimentMapCohortRun,
                memory::Int4SentimentMapMemoryLimits}}}},
};
mod backend;
use backend::Backend;

/// Frozen and admitted identities retained while a complete map is recompiled.
/// This ownership is additional to the existing complete-map memory quote.
pub const SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES: u64 = 32_768;
const COHORT_WRAPPER_BYTES: u64 = 4096;

/// Owns one actual model-bound compiler. Policy, EOS and score space remain
/// fixed in that compiler; record arguments replace only axes and task budget.
/// The five native invocation counters never reset at a flush boundary.
pub struct Int4SentimentDocumentBatchPlanner {
    planner: Int4SentimentPlanner,
    defaults: Option<SentimentBatchArgs>,
    mapping: Int4SentimentMapLimits,
    max_run_work: Int4Work,
    max_axes: usize,
    cohort_chunks: Option<usize>,
    seal: Arc<()>,
}
impl Int4SentimentDocumentBatchPlanner {
    pub fn new(planner: Int4SentimentPlanner, defaults: Option<SentimentBatchArgs>,
        mapping: Int4SentimentMapLimits, max_run_work: Int4Work) -> Result<Self, BatchFault> {
        if planner.document_encoding() != &crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback {
            return Err(BatchCode::Admission.into());
        }
        validate_limits(planner.task_budget(), planner.planning_limits(), mapping, max_run_work)?;
        constrained_int4::check_profile(planner.execution_identity(), planner.group_width())
            .map_err(|_| BatchCode::Admission)?;
        if planner.execution_identity().task_spec != "sentiment-v1" { return Err(BatchCode::Admission.into()); }
        if let Some(args) = &defaults {
            validate_args(args, planner.task_budget()).map_err(|e| e.fault)?;
            planner.max_document_bytes(&args.axes, args.budget, mapping.chunks.context_tokens)
                .map_err(|e| map_failure(e.into(), false).fault)?;
        }
        Ok(Self { planner, defaults, mapping, max_run_work, max_axes: 4, cohort_chunks: None, seal: Arc::new(()) })
    }
    /// Bind an explicit capability ceiling for every default and item override.
    /// Defaults alone never restrict the axes that a future item may request.
    pub fn with_max_axes(mut self, max_axes: usize) -> Result<Self, BatchFault> {
        validate_geometry(max_axes, self.cohort_chunks).map_err(|e| e.fault)?;
        if self.defaults.as_ref().is_some_and(|args| args.axes.len() > max_axes) {
            return Err(BatchCode::InvalidLimits.into());
        }
        self.max_axes = max_axes;
        self.seal = Arc::new(());
        Ok(self)
    }
    /// Opt into true simultaneous chunk-by-axis windows. An actual request
    /// whose axes exceed the native window geometry is refused, never clamped
    /// or silently changed to the serial schedule.
    pub fn with_chunk_cohorts(mut self, chunks: usize) -> Result<Self, BatchFault> {
        validate_cohort_chunks(chunks, self.mapping)?;
        validate_geometry(self.max_axes, Some(chunks)).map_err(|e| e.fault)?;
        self.mapping.max_result_bytes.checked_add(COHORT_WRAPPER_BYTES).ok_or(BatchCode::InvalidLimits)?;
        if let Some(args) = &self.defaults { validate_geometry(args.axes.len(), Some(chunks)).map_err(|e| e.fault)?; }
        self.cohort_chunks = Some(chunks);
        self.seal = Arc::new(());
        Ok(self)
    }
    pub fn base_execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    pub fn group_width(&self) -> usize { self.planner.group_width() }
    pub fn task_budget(&self) -> TaskBudget { self.planner.task_budget() }
    pub fn planning_limits(&self) -> SentimentLimits { self.planner.planning_limits() }
    pub fn mapping_limits(&self) -> Int4SentimentMapLimits { self.mapping }
    pub fn max_run_work(&self) -> Int4Work { self.max_run_work }
    pub fn max_axes(&self) -> usize { self.max_axes }
    pub fn cohort_chunks(&self) -> Option<usize> { self.cohort_chunks }
    pub fn max_result_bytes(&self) -> u64 {
        self.mapping.max_result_bytes + if self.cohort_chunks.is_some() { COHORT_WRAPPER_BYTES } else { 0 }
    }
    /// Account for actual transferred capacities, including overallocated
    /// defaults supplied by an embedding caller rather than the NDJSON parser.
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> {
        self.defaults.as_ref().map_or(Ok(0), |args| (args.axes.capacity() as u64)
            .checked_mul(std::mem::size_of::<SentimentAxis>() as u64).ok_or_else(|| BatchCode::InvalidLimits.into()))
    }
    fn request(&self, document: BatchDocument<SentimentBatchArgs>) -> Result<SentimentRequest, BatchItemFailure> {
        build_request(document, &self.defaults, self.mapping, self.task_budget(), self.max_axes, self.cohort_chunks)
    }
}

/// Owned original source and frozen exact semantic identity for every chunk.
/// No deserializer or public result/plan constructor can bypass preparation.
pub struct PreparedInt4SentimentDocument {
    request: SentimentRequest,
    expected: Int4SentimentMapPreflight,
    identities: Vec<ExecutionIdentity>,
    required_context: usize,
    head_count: usize,
    seal: Arc<()>,
}
impl PreparedInt4SentimentDocument {
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
    pub fn planned_work(&self) -> Int4Work { self.expected.planned_work() }
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn head_count(&self) -> usize { self.head_count }
}

/// Preserve the existing standalone serial/cohort wire values verbatim.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Int4SentimentDocumentRun {
    Serial(Int4SentimentMapRun),
    ChunkCohorts(Int4SentimentMapCohortRun),
}

pub struct Int4SentimentDocumentAdmission<'a> {
    pub identities: &'a [ExecutionIdentity],
    pub model_work: Int4Work,
    pub group_width: usize,
    pub required_context: usize,
    pub head_count: usize,
    pub task_budget: TaskBudget,
    pub kv_reservation_bytes: u64,
    pub max_result_bytes: u64,
    pub cohort_chunks: Option<usize>,
}
pub trait Int4SentimentDocumentBatchAdmission {
    type Guard;
    fn admit(&mut self, request: Int4SentimentDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure>;
}

pub struct NativeInt4SentimentDocumentBatch<'e, 'w, A: Int4SentimentDocumentBatchAdmission> {
    compiler: Int4SentimentDocumentBatchPlanner,
    backend: Backend<'e, 'w>,
    admission: A,
    run: RunState,
}
impl<'e, 'w, A: Int4SentimentDocumentBatchAdmission> NativeInt4SentimentDocumentBatch<'e, 'w, A> {
    pub fn new(compiler: Int4SentimentDocumentBatchPlanner, engine: &'e mut StrictInt4Engine<'w>,
        admission: A) -> Result<Self, BatchFault> {
        if compiler.cohort_chunks.is_some() { return Err(BatchCode::InvalidLimits.into()); }
        let backend = Backend::Serial(engine);
        backend.check(&compiler, compiler.task_budget(), compiler.max_axes).map_err(|e| e.fault)?;
        let run = RunState::new(compiler.max_run_work);
        Ok(Self { compiler, backend, admission, run })
    }
    /// The supplied memory geometry must price the declared axis ceiling. It is
    /// an admitted upper bound; every actual ragged window is checked before
    /// any native allocation or forward and retains its own semantic heads.
    pub fn new_cohorts(compiler: Int4SentimentDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        kernel: PackedInt4Kernel, memory: Int4SentimentMapMemoryLimits, admission: A)
        -> Result<Self, BatchFault> {
        let backend = Backend::cohorts(&compiler, weights, kernel, memory)?;
        let run = RunState::new(compiler.max_run_work);
        Ok(Self { compiler, backend, admission, run })
    }
    pub fn remaining_work(&self) -> Int4Work { self.run.remaining() }
    pub fn is_poisoned(&self) -> bool { self.run.is_unavailable() || !self.backend.clean() }
}
impl<A: Int4SentimentDocumentBatchAdmission> BatchProcessor for NativeInt4SentimentDocumentBatch<'_, '_, A> {
    type Args = SentimentBatchArgs;
    type Prepared = PreparedInt4SentimentDocument;
    type Output = GuardedOutput<Int4SentimentDocumentRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.fail(); Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            checkpoint(control)?;
            let request = self.compiler.request(document)?;
            self.backend.check(&self.compiler, request.budget, request.axes.len())?;
            let plan = self.compiler.planner.plan_document_with_control(&request, self.compiler.mapping, control)
                .map_err(|e| map_failure(e, false))?;
            subtract(self.run.remaining(), plan.planned_work())?;
            let identities = plan.identities().map_err(|e| map_failure(e, false))?;
            self.backend.preflight(&plan, &identities, control).map_err(|e| map_failure(e, false))?;
            let expected = plan.preflight_metadata(); let required_context = plan.required_context();
            let head_count = plan.head_count();
            drop(plan); checkpoint(control)?;
            Ok(PreparedInt4SentimentDocument { request, expected, identities, required_context, head_count,
                seal: Arc::clone(&self.compiler.seal) })
        })();
        self.run.finish_preparation(result, self.backend.clean())
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.fail(); Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            if !Arc::ptr_eq(&prepared.seal, &self.compiler.seal) {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            // A full attempt is consumed before any replan, admission or native
            // call. All later refusals retain that debit and end the stream.
            self.run.charge(prepared.planned_work(), context)?;
            checkpoint(control)?;
            let kv = self.backend.check(&self.compiler, prepared.request.budget, prepared.head_count)?;
            let plan = self.compiler.planner.plan_document_with_control(&prepared.request, self.compiler.mapping, control)
                .map_err(|e| map_failure(e, true))?;
            let actual = plan.identities().map_err(|e| map_failure(e, true))?;
            if plan.preflight_metadata() != prepared.expected || actual != prepared.identities
                || plan.required_context() != prepared.required_context || plan.head_count() != prepared.head_count {
                return Err(BatchItemFailure::fatal(BatchCode::Admission));
            }
            self.backend.preflight(&plan, &actual, control).map_err(|e| map_failure(e, true))?;
            let (admitted, guard) = self.admission.admit(Int4SentimentDocumentAdmission {
                identities: &prepared.identities, model_work: prepared.planned_work(),
                group_width: plan.group_width(), required_context: prepared.required_context,
                head_count: prepared.head_count, task_budget: prepared.request.budget,
                kv_reservation_bytes: kv, max_result_bytes: self.compiler.max_result_bytes(),
                cohort_chunks: self.compiler.cohort_chunks })?;
            if admitted != prepared.identities { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
            checkpoint(control)?;
            let result = self.backend.execute(&plan, &admitted, control).map_err(|e| map_failure(e, true))?;
            backend::verify_completed(&prepared.expected, self.compiler.cohort_chunks, &result)
                .map_err(|e| map_failure(e, true))?;
            checkpoint(control)?;
            // GuardedOutput destroys the complete value before its reservation;
            // the sink retains both through serialization, write and flush.
            Ok(GuardedOutput::new(result, guard))
        })();
        self.run.finish_execution(result, self.backend.clean())
    }
}

pub(crate) fn validate_limits(task: TaskBudget, planning: SentimentLimits,
    mapping: Int4SentimentMapLimits, run: Int4Work) -> Result<(), BatchFault> {
    mapping.validate(task).map_err(|_| BatchCode::InvalidLimits)?;
    let axis = planning.per_axis;
    if !nonzero(run) || !fits(mapping.max_model_work, run)
        || axis.max_candidates == 0 || axis.max_total_tokens == 0 || axis.max_nodes == 0
        || axis.max_depth == 0 || axis.max_candidate_id_bytes == 0 || axis.max_projected_logits == 0
        || planning.max_total_prompt_tokens == 0 || planning.max_total_candidate_tokens == 0
        || planning.max_total_nodes == 0 || planning.max_total_projected_logits == 0 || planning.max_output_bytes == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn validate_cohort_chunks(chunks: usize, mapping: Int4SentimentMapLimits) -> Result<(), BatchFault> {
    if chunks == 0 || chunks > MAX_INT4_SEQUENCES || chunks > mapping.chunks.max_chunks {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn validate_geometry(axes: usize, chunks: Option<usize>) -> Result<(), BatchItemFailure> {
    if !(1..=SentimentAxis::ALL.len()).contains(&axes)
        || chunks.is_some_and(|chunks| chunks == 0 || chunks.checked_mul(axes).is_none_or(|n| n > MAX_INT4_SEQUENCES)) {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(())
}
fn validate_args(args: &SentimentBatchArgs, ceiling: TaskBudget) -> Result<(), BatchItemFailure> {
    let b = args.budget;
    if b.validate().is_err() || b.max_input_tokens > ceiling.max_input_tokens
        || b.max_output_tokens > ceiling.max_output_tokens || b.max_output_bytes > ceiling.max_output_bytes
        || b.max_grammar_states > ceiling.max_grammar_states || b.max_kv_bytes > ceiling.max_kv_bytes
        || args.axes.is_empty() || args.axes.len() > SentimentAxis::ALL.len()
        || args.axes.iter().enumerate().any(|(i, axis)| args.axes[..i].contains(axis)) {
        return Err(BatchItemFailure::reject(BatchCode::Planning));
    }
    Ok(())
}
fn build_request(document: BatchDocument<SentimentBatchArgs>, defaults: &Option<SentimentBatchArgs>,
    mapping: Int4SentimentMapLimits, task: TaskBudget, max_axes: usize, cohort_chunks: Option<usize>)
    -> Result<SentimentRequest, BatchItemFailure> {
    if document.text.is_empty() || document.text.len() > mapping.chunks.max_input_bytes {
        return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
    }
    let mut args = document.task_args.or_else(|| defaults.clone())
        .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
    validate_args(&args, task)?;
    if args.axes.len() > max_axes { return Err(BatchItemFailure::reject(BatchCode::Planning)); }
    validate_geometry(args.axes.len(), cohort_chunks)?;
    args.axes.sort_unstable();
    Ok(SentimentRequest { document: document.text, axes: args.axes, budget: args.budget })
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4SentimentMapError, executing: bool) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    use Int4SentimentMapError as M;
    use Int4SentimentError as S;
    let mut failure = match error {
        M::EmptySource | M::Sentiment(S::Planning(SentimentPlanningError::EmptyDocument))
            => BatchItemFailure::reject(BatchCode::DocumentLimit),
        M::Limits | M::Chunk(_) | M::Sentiment(S::Planning(SentimentPlanningError::InvalidAxes
            | SentimentPlanningError::Budget | SentimentPlanningError::DocumentEncoding))
            => BatchItemFailure::reject(BatchCode::Planning),
        M::Work | M::Sentiment(S::WorkBudget | S::Native(StrictInt4Error::Work)
            | S::Scoring(Int4ScoringError::Native(StrictInt4Error::Work)))
            => BatchItemFailure::reject(BatchCode::WorkLimit),
        M::Allocation | M::Sentiment(S::Allocation | S::Planning(SentimentPlanningError::AllocationRefused)
            | S::Planning(SentimentPlanningError::Task(SentimentError::AllocationRefused))
            | S::Task(SentimentError::AllocationRefused) | S::Native(StrictInt4Error::Allocation)
            | S::Scoring(Int4ScoringError::Allocation | Int4ScoringError::Native(StrictInt4Error::Allocation)))
            => BatchItemFailure::fatal(BatchCode::Allocation),
        M::Output | M::Sentiment(S::OutputBudget | S::Scoring(Int4ScoringError::OutputBudget))
            => BatchItemFailure::reject(BatchCode::OutputLineLimit),
        M::Serialization | M::Sentiment(S::Serialization | S::Planning(SentimentPlanningError::Serialization)
            | S::Planning(SentimentPlanningError::Task(SentimentError::Serialization))
            | S::Task(SentimentError::Serialization)
            | S::Scoring(Int4ScoringError::Serialization)) => BatchItemFailure::fatal(BatchCode::Serialization),
        M::Sentiment(S::Planning(SentimentPlanningError::Task(_)) | S::Task(SentimentError::LimitExceeded(_)))
            => BatchItemFailure::reject(BatchCode::Planning),
        M::Sentiment(S::Native(StrictInt4Error::Context | StrictInt4Error::Memory)
            | S::Scoring(Int4ScoringError::Native(StrictInt4Error::Context | StrictInt4Error::Memory)))
            => BatchItemFailure::reject(BatchCode::Admission),
        M::Admission | M::Sentiment(S::Identity | S::Planning(SentimentPlanningError::Identity))
            => BatchItemFailure::fatal(BatchCode::Admission),
        _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
    };
    if executing { failure.stop = true; }
    failure
}

#[cfg(test)] mod tests;
