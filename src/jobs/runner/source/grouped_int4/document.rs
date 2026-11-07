//! Authenticated complete-document source jobs on the actual grouped-INT4 engine.
//! One durable attempt owns all discovery chunks and every permitted synthesis
//! pass. No partial document, failed citation or incomplete hierarchy is a result.
use super::*;
use crate::{
    native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights},
    tasks::{answer::AnswerOptions, keyphrases::KeyphraseOptions,
        mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE},
        summarize::SummaryOptions,
        source_planning::grouped_int4::long::{Int4SourceMapLimits, SourceMapTask,
            cohort::{Int4SourceMapCohortBudget,
                Int4SourceMapCohortRequirement, hierarchy::Int4HierarchyCohortSchedule},
            keyphrases::Int4KeyphraseLimits,
            summary::{Int4SummaryLimits, synthesis::{SourceSummarySynthesis, SummarySynthesisLimits,
                hierarchy::SummaryHierarchyLimits}},
            question::{SourceQuestion, synthesis::{SourceQuestionSynthesis, QuestionSynthesisLimits,
                hierarchy::QuestionHierarchyLimits}}}},
    validation::grounded_fields::GroundingBudget,
};
pub use crate::jobs::runner::source::document::SourceDocumentArgs;
mod recipe;
mod planning;
mod execution;
pub use recipe::Int4SourceDocumentJobRecipe;
pub use execution::Int4SourceDocumentJobProcessor;
pub use crate::batch::source::grouped_int4::document::{Int4SourceDocumentTask,
    Int4SourceDocumentLifetime, Int4SourceDocumentAdmission, Int4SourceDocumentBatchAdmission,
    Int4SourceDocumentRun};
use crate::batch::source::grouped_int4::document::{self as native_document, Configuration, DocumentWork, poll};
#[cfg(test)]
use native_document::{task::verify_lifetime, keyphrase_failure};
#[cfg(test)]
use crate::tasks::source_planning::grouped_int4::long::{cohort::Int4SourceMapCohortMemoryLimits,
    keyphrases::{Int4CorpusKeyphraseError, Int4CorpusKeyphraseRun, INT4_CORPUS_KEYPHRASE_EXECUTION}};

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
    fn configuration(&self) -> Configuration<'_> {
        Configuration { planner: &self.planner, identity: &self.identity, group_width: self.group_width,
            task: self.task(), budget: self.budget, planning: self.planning, mapping: self.mapping,
            chunk_cohorts: self.uses_chunk_cohorts(), hierarchy: self.hierarchy_cohort_schedule() }
    }
    /// Opt into native cohorts inside each complete durable document attempt.
    /// The existing immutable map width is also the native cohort ceiling;
    /// no document, work debit or spool commit is split into chunk attempts.
    pub fn with_chunk_cohorts(mut self) -> Result<Self, BatchFault> {
        if !self.uses_chunk_cohorts() {
            self.recipe.enable_chunk_cohorts()?;
            // Previously prepared receipts belong to the old scheduling policy,
            // even when their semantic task and native work happen to be identical.
            self.owner = Arc::new(());
        }
        self.cohort_requirements()?;
        Ok(self)
    }
    pub fn uses_chunk_cohorts(&self) -> bool { self.recipe.uses_chunk_cohorts() }
    /// Independently batch evidence groups within each completely witnessed
    /// hierarchy level. The private resume recipe authenticates this width.
    pub fn with_hierarchy_cohorts(mut self, max_groups: usize) -> Result<Self, BatchFault> {
        let max_passes = match self.task() {
            Int4SourceDocumentTask::SummaryHierarchy { limits, .. } => limits.max_passes,
            Int4SourceDocumentTask::QuestionHierarchy { limits, .. } => limits.max_passes,
            _ => return Err(BatchCode::InvalidLimits.into()),
        };
        let schedule = Int4HierarchyCohortSchedule { max_groups, discovery_cohorts: self.uses_chunk_cohorts() };
        schedule.validate(max_passes).map_err(|_| BatchCode::InvalidLimits)?;
        if self.hierarchy_cohort_schedule() != Some(schedule) {
            self.recipe.enable_hierarchy_cohorts(max_groups)?;
            self.owner = Arc::new(());
        }
        self.cohort_requirements()?;
        Ok(self)
    }
    pub fn hierarchy_cohort_schedule(&self) -> Option<Int4HierarchyCohortSchedule> {
        self.recipe.hierarchy_cohort_schedule()
    }
    pub fn uses_native_cohorts(&self) -> bool {
        self.uses_chunk_cohorts() || self.hierarchy_cohort_schedule().is_some()
    }
    /// Static declared native/decoder peak for the whole job invocation.
    /// Actual document windows are checked against this ceiling before native
    /// execution. Per-row task KV and all lifetime work remain unchanged.
    pub fn cohort_requirements(&self) -> Result<Int4SourceMapCohortRequirement, BatchFault> {
        self.configuration().cohort_requirements()
    }
    pub fn max_result_bytes(&self) -> u64 {
        self.task().max_result_bytes(self.mapping)
    }
    /// Retained native token arrays in the final result. Discovery QA and exact
    /// summary/keyphrase selection omit transcripts; their temporary storage is separate.
    pub fn output_token_ceiling(&self) -> Result<u64, BatchFault> {
        self.output_tokens(self.mapping.chunks.max_chunks)
    }
    fn output_tokens(&self, chunks: usize) -> Result<u64, BatchFault> {
        self.task().output_tokens(chunks, self.budget)
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

fn validate_mapping(m: Int4SourceMapLimits, budget: TaskBudget, p: SourcePlanningLimits, job: JobLimits)
    -> Result<(), BatchFault> {
    job.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if !JobWork::from_int4(m.max_model_work, m.max_mask_visits).fits(job.max_work) {
        return Err(BatchCode::InvalidLimits.into());
    }
    native_document::validate_mapping(m, budget, p)
}

#[cfg(test)] mod tests;
