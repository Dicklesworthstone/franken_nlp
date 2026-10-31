//! Authenticated complete-document source jobs on the actual grouped-INT4 engine.
//! One durable attempt owns all discovery chunks and every permitted synthesis
//! pass. No partial document, failed citation or incomplete hierarchy is a result.
use super::*;
use crate::{
    native_engine::{constrained_int4, int4::PackedInt4Kernel, strict_int4::Int4ModelWeights},
    tasks::{answer::AnswerOptions, ir::PlanContext, keyphrases::KeyphraseOptions,
        mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE},
        summarize::SummaryOptions,
        source_planning::grouped_int4::long::{Int4SourceMapError, Int4SourceMapLimits,
            Int4SourceMapPreflight, Int4SourceMapRun, PreparedInt4SourceMap, SourceMapTask,
            INT4_SOURCE_MAP_EXECUTION,
            cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortMemoryLimits,
                Int4SourceMapCohortRequirement},
            keyphrases::{Int4CorpusKeyphraseError, Int4CorpusKeyphraseRun, Int4KeyphraseLimits,
                INT4_CORPUS_KEYPHRASE_EXECUTION},
            summary::{Int4CorpusSummaryError, Int4CorpusSummaryRun, Int4SummaryLimits,
                INT4_CORPUS_SUMMARY_EXECUTION,
                synthesis::{Int4SummarySynthesisPreflight, Int4SummarySynthesisRun,
                    PreparedInt4SummarySynthesis, SourceSummarySynthesis, SummarySynthesisLimits,
                    INT4_SUMMARY_SYNTHESIS_EXECUTION,
                    hierarchy::{Int4SummaryHierarchyPreflight, Int4SummaryHierarchyRun,
                        PreparedInt4SummaryHierarchy, SummaryHierarchyLimits, INT4_SUMMARY_HIERARCHY_EXECUTION}}},
            question::{Int4QuestionPreflight, Int4QuestionRun, PreparedInt4Question, SourceQuestion,
                INT4_QUESTION_EXECUTION,
                synthesis::{Int4QuestionSynthesisPreflight, Int4QuestionSynthesisRun,
                    PreparedInt4QuestionSynthesis, SourceQuestionSynthesis, QuestionSynthesisLimits,
                    INT4_QUESTION_SYNTHESIS_EXECUTION,
                    hierarchy::{Int4QuestionHierarchyPreflight, Int4QuestionHierarchyRun,
                        PreparedInt4QuestionHierarchy, QuestionHierarchyLimits, INT4_QUESTION_HIERARCHY_EXECUTION}}}}},
    validation::grounded_fields::GroundingBudget,
};
pub use crate::jobs::runner::source::document::SourceDocumentArgs;
mod recipe;
mod planning;
mod execution;
pub use recipe::Int4SourceDocumentJobRecipe;
pub use execution::{Int4SourceDocumentAdmission, Int4SourceDocumentBatchAdmission,
    Int4SourceDocumentJobProcessor, Int4SourceDocumentRun};

/// Fixed data-only policy for the complete ordered population. Private
/// questions deliberately have neither Debug nor a deserializable factory.
pub enum Int4SourceDocumentTask {
    Map(SourceMapTask),
    /// Complete exact-text/evidence union, ranked only at the final root.
    /// The independent reducer scan ceiling includes every durable attempt.
    Keyphrases { options: KeyphraseOptions, limits: Int4KeyphraseLimits, max_lifetime_scan_work: u64 },
    Summary { options: SummaryOptions, limits: Int4SummaryLimits },
    Question(SourceQuestion),
    SummarySynthesis(SourceSummarySynthesis),
    SummaryHierarchy { request: SourceSummarySynthesis, limits: SummaryHierarchyLimits },
    QuestionSynthesis(SourceQuestionSynthesis),
    QuestionHierarchy { request: SourceQuestionSynthesis, limits: QuestionHierarchyLimits },
}
impl Int4SourceDocumentTask {
    fn kind(&self) -> BuiltInTask {
        match self {
            Self::Map(SourceMapTask::Ner(_)) => BuiltInTask::Ner,
            Self::Map(SourceMapTask::Keyphrases(_)) | Self::Keyphrases { .. } => BuiltInTask::Keyphrases,
            Self::Map(SourceMapTask::Summarize(_)) | Self::Summary { .. }
                | Self::SummarySynthesis(_) | Self::SummaryHierarchy { .. } => BuiltInTask::Summarize,
            Self::Question(_) | Self::QuestionSynthesis(_) | Self::QuestionHierarchy { .. } => BuiltInTask::Answer,
        }
    }
    fn execution(&self) -> &'static str {
        match self {
            Self::Map(_) => INT4_SOURCE_MAP_EXECUTION,
            Self::Keyphrases { .. } => INT4_CORPUS_KEYPHRASE_EXECUTION,
            Self::Summary { .. } => INT4_CORPUS_SUMMARY_EXECUTION,
            Self::Question(_) => INT4_QUESTION_EXECUTION,
            Self::SummarySynthesis(_) => INT4_SUMMARY_SYNTHESIS_EXECUTION,
            Self::SummaryHierarchy { .. } => INT4_SUMMARY_HIERARCHY_EXECUTION,
            Self::QuestionSynthesis(_) => INT4_QUESTION_SYNTHESIS_EXECUTION,
            Self::QuestionHierarchy { .. } => INT4_QUESTION_HIERARCHY_EXECUTION,
        }
    }
    fn verification(&self, mapping: Int4SourceMapLimits) -> GroundingBudget {
        match self {
            Self::Map(_) | Self::Keyphrases { .. } | Self::Summary { .. } => mapping.verification,
            Self::SummarySynthesis(request) | Self::SummaryHierarchy { request, .. } => request.limits.verification,
            Self::Question(question) => question.verification,
            Self::QuestionSynthesis(request) | Self::QuestionHierarchy { request, .. } => request.question.verification,
        }
    }
    fn summary_scan_steps(&self) -> u64 {
        match self { Self::Summary { limits, .. } => limits.aggregation.max_scan_steps, _ => 0 }
    }
    fn verify_lifetime(&self, mapping: Int4SourceMapLimits, lifetime: Int4SourceDocumentLifetime,
        attempts: u64) -> Result<(), BatchFault> {
        verify_lifetime(self.verification(mapping), self.summary_scan_steps(), lifetime, attempts)?;
        if let Self::Keyphrases { limits, max_lifetime_scan_work, .. } = self {
            // The persisted attempt count covers failures, cancellation and
            // uncertain acknowledgements too. Keep this new allowance inside
            // the new task variant so existing recipe bytes remain unchanged.
            if limits.aggregation.max_scan_work.checked_mul(attempts)
                .is_none_or(|work| work > *max_lifetime_scan_work) {
                return Err(BatchCode::InvalidLimits.into());
            }
        }
        Ok(())
    }
    fn validate(&self, planning: SourcePlanningLimits, mapping: Int4SourceMapLimits) -> Result<(), BatchFault> {
        let invalid = |_| BatchFault::from(BatchCode::InvalidLimits);
        match self {
            Self::Map(SourceMapTask::Ner(options)) => options.validate().map_err(invalid)?,
            Self::Map(SourceMapTask::Keyphrases(options)) => options.validate().map_err(|_| BatchCode::InvalidLimits)?,
            Self::Map(SourceMapTask::Summarize(options)) => options.validate().map_err(|_| BatchCode::InvalidLimits)?,
            Self::Keyphrases { options, limits, .. } => {
                options.validate().map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate(mapping).map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::Summary { options, limits } => {
                options.validate().map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate(mapping).map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::Question(question) => validate_question(question, planning)?,
            Self::SummarySynthesis(request) => request.validate(planning).map_err(|_| BatchCode::InvalidLimits)?,
            Self::SummaryHierarchy { request, limits } => {
                request.validate(planning).map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::QuestionSynthesis(request) => {
                validate_question(&request.question, planning)?;
                request.limits.validate(planning).map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::QuestionHierarchy { request, limits } => {
                validate_question(&request.question, planning)?;
                request.limits.validate(planning).map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
            }
        }
        Ok(())
    }
}
fn validate_question(question: &SourceQuestion, planning: SourcePlanningLimits) -> Result<(), BatchFault> {
    question.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if question.question.len() > planning.max_input_bytes { return Err(BatchCode::InvalidLimits.into()); }
    Ok(())
}

/// Explicit lifetime authority for independent original-coordinate verification
/// and the exact summary reducer. Ranked keyphrases carry their additional
/// reducer ceiling in the task variant, preserving this existing recipe shape.
/// The persisted attempt counter must fit every ceiling up front.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Int4SourceDocumentLifetime {
    pub verification: GroundingBudget,
    pub summary_scan_steps: u64,
}

/// The real model, compiler and task are owned by the same immutable factory
/// whose private-content projection is committed by JobRunner.
pub struct Int4SourceDocumentJobPlanner {
    planner: Arc<SourceTaskPlanner>,
    identity: ExecutionIdentity,
    group_width: usize,
    budget: TaskBudget,
    planning: SourcePlanningLimits,
    mapping: Int4SourceMapLimits,
    limits: JobLimits,
    recipe: Int4SourceDocumentJobRecipe,
    owner: Arc<()>,
}
impl Int4SourceDocumentJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn for_model(model: &Int4ModelIdentity, planner: Arc<SourceTaskPlanner>,
        task: Int4SourceDocumentTask, budget: TaskBudget, planning: SourcePlanningLimits,
        mapping: Int4SourceMapLimits, limits: JobLimits, lifetime: Int4SourceDocumentLifetime)
        -> Result<Self, BatchFault> {
        validate_mapping(mapping, budget, planning, limits)?;
        let native = Int4SourceBatchLimits { max_model_work: limits.max_work.int4_model_work(),
            masks: SourceMaskBudget { per_mask: mapping.mask_limits,
                max_visits_per_item: mapping.mask_visits_per_chunk,
                max_visits_per_run: limits.max_work.mask_node_visits } };
        let source = Int4SourceJobPlanner::for_model(model, planner, task.kind(), budget, planning, None, native)?;
        Self::configured(source, task, mapping, limits, lifetime)
    }
    // Fixtures can exercise the real pinned compiler without inventing a public
    // model-identity constructor or allowing an injected execution backend.
    fn configured(source: Int4SourceJobPlanner, task: Int4SourceDocumentTask,
        mapping: Int4SourceMapLimits, limits: JobLimits, lifetime: Int4SourceDocumentLifetime)
        -> Result<Self, BatchFault> {
        let budget = source.task_ceiling();
        validate_mapping(mapping, budget, source.planning, limits)?;
        if source.execution_identity().task_spec != task.kind().spec().identity()
            || source.recipe.defaults.is_some() {
            return Err(BatchCode::Admission.into());
        }
        task.validate(source.planning, mapping)?;
        let result_bytes = match &task {
            Int4SourceDocumentTask::Keyphrases { limits, .. } => limits.max_result_bytes,
            Int4SourceDocumentTask::Summary { limits, .. } => limits.max_result_bytes,
            _ => mapping.reduction.max_result_bytes,
        };
        if result_bytes > limits.max_result_bytes { return Err(BatchCode::InvalidLimits.into()); }
        task.verify_lifetime(mapping, lifetime, limits.max_attempts)?;
        let group_width = source.recipe.group_width;
        let recipe = Int4SourceDocumentJobRecipe::new(source.recipe, task, mapping, lifetime, limits.max_attempts)?;
        Ok(Self { planner: source.planner, identity: source.identity, group_width, budget,
            planning: source.planning, mapping, limits, recipe, owner: Arc::new(()) })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn planning_limits(&self) -> SourcePlanningLimits { self.planning }
    pub fn document_limits(&self) -> Int4SourceMapLimits { self.mapping }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn task(&self) -> &Int4SourceDocumentTask { &self.recipe.task }
    pub fn job_recipe(&self) -> &Int4SourceDocumentJobRecipe { &self.recipe }
    /// Opt into native cohorts inside each complete durable document attempt.
    /// The existing immutable map width is also the native cohort ceiling;
    /// no document, work debit or spool commit is split into chunk attempts.
    pub fn with_chunk_cohorts(mut self) -> Result<Self, BatchFault> {
        self.cohort_requirements()?;
        if !self.uses_chunk_cohorts() {
            self.recipe.enable_chunk_cohorts()?;
            // Previously prepared receipts belong to the old scheduling policy,
            // even when their semantic task and native work happen to be identical.
            self.owner = Arc::new(());
        }
        Ok(self)
    }
    pub fn uses_chunk_cohorts(&self) -> bool { self.recipe.uses_chunk_cohorts() }
    /// Static declared native/decoder peak for the whole job invocation.
    /// Actual document windows are checked against this ceiling before native
    /// execution. Per-row task KV and all lifetime work remain unchanged.
    pub fn cohort_requirements(&self) -> Result<Int4SourceMapCohortRequirement, BatchFault> {
        let max_chunks = self.mapping.reduction.map_batch_chunks;
        if max_chunks > self.mapping.chunks.max_chunks { return Err(BatchCode::InvalidLimits.into()); }
        let context = self.planning.max_context_tokens;
        let scalar = Int4MemoryRequirement::for_context(context, self.group_width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if scalar.kv_bytes != self.budget.max_kv_bytes { return Err(BatchCode::Admission.into()); }
        let quote = Int4SourceMapCohortMemoryLimits { max_chunks, context_tokens: context,
            max_output_tokens_per_chunk: u64::from(self.budget.max_output_tokens),
            max_result_bytes_per_chunk: self.budget.max_output_bytes,
        }.requirements(self.group_width).map_err(|_| BatchCode::InvalidLimits)?;
        if matches!(self.task(), Int4SourceDocumentTask::SummarySynthesis(_)
            | Int4SourceDocumentTask::SummaryHierarchy { .. }
            | Int4SourceDocumentTask::QuestionSynthesis(_)
            | Int4SourceDocumentTask::QuestionHierarchy { .. }) {
            quote.with_synthesis_context(context, self.group_width, self.budget.max_kv_bytes)
                .map_err(|_| BatchCode::Admission.into())
        } else { Ok(quote) }
    }
    pub fn max_result_bytes(&self) -> u64 {
        match self.task() {
            Int4SourceDocumentTask::Keyphrases { limits, .. } => limits.max_result_bytes as u64,
            Int4SourceDocumentTask::Summary { limits, .. } => limits.max_result_bytes as u64,
            _ => self.mapping.reduction.max_result_bytes as u64 }
    }
    /// Retained native token arrays in the final result. Discovery QA and exact
    /// summary/keyphrase selection omit transcripts; their temporary storage is separate.
    pub fn output_token_ceiling(&self) -> Result<u64, BatchFault> {
        self.output_tokens(self.mapping.chunks.max_chunks)
    }
    fn output_tokens(&self, chunks: usize) -> Result<u64, BatchFault> {
        let passes = match self.task() {
            Int4SourceDocumentTask::Map(_) => chunks,
            Int4SourceDocumentTask::SummarySynthesis(_) => chunks.checked_add(1).ok_or(BatchCode::InvalidLimits)?,
            Int4SourceDocumentTask::SummaryHierarchy { limits, .. } => chunks.checked_add(limits.max_passes)
                .ok_or(BatchCode::InvalidLimits)?,
            Int4SourceDocumentTask::QuestionHierarchy { limits, .. } => limits.max_passes,
            _ => 0,
        };
        (passes as u64).checked_mul(u64::from(self.budget.max_output_tokens))
            .ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); }
        self.task().verify_lifetime(self.mapping, self.recipe.lifetime, limits.max_attempts)
    }
    fn owns(&self, prepared: &PreparedInt4SourceDocument) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
}

/// Owned original source and complete sealed preparation commitment. A second
/// factory cannot execute this object even when its public task is identical.
pub struct PreparedInt4SourceDocument {
    source: String,
    identities: Vec<ExecutionIdentity>,
    expected: planning::Expected,
    work: JobWork,
    owner: Arc<()>,
}
impl PreparedInt4SourceDocument {
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn planned_work(&self) -> JobWork { self.work }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
}

fn verify_lifetime(per: GroundingBudget, scans: u64, max: Int4SourceDocumentLifetime,
    attempts: u64) -> Result<(), BatchFault> {
    let count = usize::try_from(attempts).map_err(|_| BatchCode::InvalidLimits)?;
    if count == 0 || per.max_fields.checked_mul(count).is_none_or(|n| n > max.verification.max_fields)
        || per.max_matches.checked_mul(count).is_none_or(|n| n > max.verification.max_matches)
        || per.max_scan_steps.checked_mul(attempts).is_none_or(|n| n > max.verification.max_scan_steps)
        || scans.checked_mul(attempts).is_none_or(|n| n > max.summary_scan_steps) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn validate_mapping(m: Int4SourceMapLimits, budget: TaskBudget, p: SourcePlanningLimits, job: JobLimits)
    -> Result<(), BatchFault> {
    job.validate().map_err(|_| BatchCode::InvalidLimits)?;
    budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
    m.chunks.effective_token_limit().map_err(|_| BatchCode::InvalidLimits)?;
    let e = m.reduction; let v = m.verification;
    if m.chunks.max_input_bytes == 0 || !(1..=256).contains(&m.chunks.max_chunks)
        || m.chunks.max_chunk_bytes > p.max_input_bytes || m.chunks.context_tokens > p.max_context_tokens
        || m.chunks.reserved_tokens < budget.max_output_tokens as usize
        || !JobWork::from_int4(m.max_model_work, m.max_mask_visits).fits(job.max_work)
        || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
        || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
        || !(1..=1024).contains(&e.map_batch_chunks) || !(2..=1024).contains(&e.reduce_fan_in)
        || e.max_reduction_levels > 64 || !(1..=1_000_000).contains(&e.max_task_calls)
        || [e.max_value_bytes, e.max_live_value_bytes, e.max_total_value_bytes, e.max_result_bytes]
            .iter().any(|n| !(1..=64 * 1024 * 1024).contains(n))
        || e.max_value_bytes > e.max_live_value_bytes
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
        || v.max_scan_steps == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    source_batch::validate_limits(Int4SourceBatchLimits { max_model_work: m.max_model_work,
        masks: SourceMaskBudget { per_mask: m.mask_limits, max_visits_per_item: m.mask_visits_per_chunk,
            max_visits_per_run: m.max_mask_visits } })
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4SourceMapError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(match error {
        Int4SourceMapError::InvalidLimits => BatchCode::InvalidLimits,
        Int4SourceMapError::EmptySource => BatchCode::DocumentLimit,
        Int4SourceMapError::WorkLimit => BatchCode::WorkLimit,
        Int4SourceMapError::Admission => BatchCode::Admission,
        Int4SourceMapError::Allocation => BatchCode::Allocation,
        Int4SourceMapError::Chunk(_) => BatchCode::Planning,
        _ => BatchCode::InvalidExecution,
    })
}
fn summary_failure(error: Int4CorpusSummaryError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(if matches!(error, Int4CorpusSummaryError::InvalidLimits) {
        BatchCode::InvalidLimits
    } else { BatchCode::InvalidExecution })
}
fn keyphrase_failure(error: Int4CorpusKeyphraseError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(if matches!(error, Int4CorpusKeyphraseError::InvalidLimits) {
        BatchCode::InvalidLimits
    } else { BatchCode::InvalidExecution })
}

#[cfg(test)] mod tests;
