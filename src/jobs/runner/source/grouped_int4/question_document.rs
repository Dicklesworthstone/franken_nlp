//! Authenticated complete-document QA with one exact private question per item.
//! The sealed corpus compiler owns discovery, synthesis and every hierarchy
//! pass. This adapter adds durable recipe and work authority, never another
//! inference implementation or a successful partial-document result.
use super::*;
use crate::{
    batch::source::grouped_int4::question::{Int4QuestionDocumentBatchConfig,
        Int4QuestionDocumentBatchPlanner, Int4QuestionDocumentBatchAdmission,
        Int4QuestionDocumentRun, NativeInt4QuestionDocumentBatch, PreparedInt4QuestionDocument},
    native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights},
    tasks::{answer::AnswerOptions, mapreduce::{CHUNK_PROFILE, ChunkLimits, ExecutionLimits},
        source_planning::grouped_int4::long::{Int4SourceMapLimits,
            cohort::{Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement,
                hierarchy::Int4HierarchyCohortSchedule},
            question::{INT4_QUESTION_EXECUTION, synthesis::{INT4_QUESTION_SYNTHESIS_EXECUTION,
                QuestionSynthesisLimits, hierarchy::{INT4_QUESTION_HIERARCHY_EXECUTION,
                    QuestionHierarchyLimits}}}}},
    validation::grounded_fields::GroundingBudget,
};
pub use crate::batch::source::grouped_int4::question::QuestionDocumentArgs;
pub mod memory;

const RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// Private fixed-policy projection. Questions and complete source bytes live
/// only in the HMAC-authenticated original item envelopes, never in defaults.
#[derive(Serialize)]
pub struct Int4QuestionDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    reservation_policy: &'static str,
    source: Int4SourceJobRecipe,
    options: AnswerOptions,
    mapping: MappingRecipe,
    verification: GroundingBudget,
    synthesis: Option<SynthesisRecipe>,
    hierarchy: Option<QuestionHierarchyLimits>,
    max_question_bytes: usize,
    max_map_verification: GroundingBudget,
    max_verification: GroundingBudget,
    max_attempts: u64,
    chunk_cohorts: bool,
    hierarchy_cohorts: Option<HierarchyScheduleRecipe>,
    max_stored_result_bytes: u64,
    max_output_tokens: u64,
}
#[derive(Serialize)]
struct MappingRecipe {
    chunks: ChunkLimits,
    reduction: ExecutionLimits,
    max_model_work: Int4Work,
    max_trie_node_visits: usize,
    checkpoint_interval_nodes: usize,
    mask_visits_per_chunk: u64,
    max_mask_visits: u64,
    verification: GroundingBudget,
}
#[derive(Serialize)]
struct SynthesisRecipe { max_evidence_passages: usize, max_evidence_bytes: usize }
#[derive(Serialize)]
struct HierarchyScheduleRecipe { execution: &'static str, max_groups: usize, discovery_cohorts: bool }
impl Int4QuestionDocumentJobRecipe {
    fn new(compiler: &Int4QuestionDocumentBatchPlanner, limits: JobLimits) -> Result<Self, BatchFault> {
        // Exhaustive projections force every new compiler option into the
        // authenticated recipe rather than silently widening resume authority.
        let Int4QuestionDocumentBatchConfig { options, budget, planning, mapping,
            verification, synthesis, hierarchy, max_question_bytes, max_model_work,
            max_mask_visits, max_map_verification, max_verification } = *compiler.config();
        let Int4SourceMapLimits { chunks, reduction, max_model_work: per_document_work,
            mask_limits, mask_visits_per_chunk, max_mask_visits: per_document_masks,
            verification: map_verification } = mapping;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = mask_limits;
        let native = Int4SourceBatchLimits { max_model_work, masks: SourceMaskBudget {
            per_mask: mask_limits, max_visits_per_item: per_document_masks,
            max_visits_per_run: max_mask_visits } };
        let source = Int4SourceJobRecipe::new(compiler.base_execution_identity().task_spec.clone(),
            compiler.group_width(), budget, planning, native, None)?;
        let synthesis = synthesis.map(|QuestionSynthesisLimits { max_evidence_passages,
            max_evidence_bytes }| SynthesisRecipe { max_evidence_passages, max_evidence_bytes });
        let hierarchy = hierarchy.map(|QuestionHierarchyLimits { max_levels, max_passes,
            max_tokenizer_calls, max_tokenizer_bytes }| QuestionHierarchyLimits {
                max_levels, max_passes, max_tokenizer_calls, max_tokenizer_bytes });
        let hierarchy_cohorts = compiler.hierarchy_cohort_schedule().map(|Int4HierarchyCohortSchedule {
            max_groups, discovery_cohorts }| HierarchyScheduleRecipe {
                execution: "native-question-document-hierarchy-group-cohorts-v1", max_groups, discovery_cohorts });
        let execution = if hierarchy.is_some() { INT4_QUESTION_HIERARCHY_EXECUTION }
            else if synthesis.is_some() { INT4_QUESTION_SYNTHESIS_EXECUTION }
            else { INT4_QUESTION_EXECUTION };
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-source-question",
            input_policy: "required-private-question-fixed-answer-policy-v1", execution,
            chunk_profile: CHUNK_PROFILE,
            reservation_policy: "exact-five-axis-mask-debit-verification-across-all-durable-attempts-v1",
            source, options, mapping: MappingRecipe { chunks, reduction, max_model_work: per_document_work,
                max_trie_node_visits, checkpoint_interval_nodes, mask_visits_per_chunk,
                max_mask_visits: per_document_masks, verification: map_verification },
            verification, synthesis, hierarchy, max_question_bytes, max_map_verification, max_verification,
            max_attempts: limits.max_attempts, chunk_cohorts: compiler.uses_chunk_cohorts(), hierarchy_cohorts,
            max_stored_result_bytes: compiler.max_result_bytes(), max_output_tokens: compiler.max_output_tokens() };
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

/// Wrap the actual-model-bound corpus compiler after selecting its immutable
/// discovery and hierarchy schedules. No task or per-item policy override is
/// introduced by retaining the result.
pub struct Int4QuestionDocumentJobPlanner {
    compiler: Int4QuestionDocumentBatchPlanner,
    recipe: Int4QuestionDocumentJobRecipe,
    limits: JobLimits,
}
impl Int4QuestionDocumentJobPlanner {
    pub fn new(compiler: Int4QuestionDocumentBatchPlanner, limits: JobLimits) -> Result<Self, BatchFault> {
        check_limits(*compiler.config(), limits)?;
        let recipe = Int4QuestionDocumentJobRecipe::new(&compiler, limits)?;
        Ok(Self { compiler, recipe, limits })
    }
    pub fn batch_planner(&self) -> &Int4QuestionDocumentBatchPlanner { &self.compiler }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.compiler.base_execution_identity() }
    pub fn config(&self) -> &Int4QuestionDocumentBatchConfig { self.compiler.config() }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn job_recipe(&self) -> &Int4QuestionDocumentJobRecipe { &self.recipe }
    pub fn group_width(&self) -> usize { self.compiler.group_width() }
    pub fn task_budget(&self) -> TaskBudget { self.config().budget }
    pub fn planning_limits(&self) -> SourcePlanningLimits { self.config().planning }
    pub fn document_limits(&self) -> Int4SourceMapLimits { self.config().mapping }
    pub fn max_result_bytes(&self) -> u64 { self.compiler.max_result_bytes() }
    pub fn max_output_tokens(&self) -> u64 { self.compiler.max_output_tokens() }
    pub fn uses_chunk_cohorts(&self) -> bool { self.compiler.uses_chunk_cohorts() }
    pub fn uses_native_cohorts(&self) -> bool { self.compiler.uses_native_cohorts() }
    pub fn hierarchy_cohort_schedule(&self) -> Option<Int4HierarchyCohortSchedule> {
        self.compiler.hierarchy_cohort_schedule()
    }
    pub fn cohort_requirements(&self) -> Result<Int4SourceMapCohortRequirement, BatchFault> {
        self.compiler.cohort_requirements()
    }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> { Ok(0) }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); }
        check_limits(*self.config(), limits)
    }
    /// Pure planning over the exact source/question pair; native capacity is
    /// additionally preflighted by the production processor before execution.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<QuestionDocumentArgs>,
        control: &mut C) -> Result<PreparedInt4QuestionDocument, BatchItemFailure> {
        let document = self.checked_document(document)?;
        let args = document.task_args.ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        self.compiler.prepare_with_control(document.text, args, control)
    }
    fn checked_document(&self, document: BatchDocument<QuestionDocumentArgs>)
        -> Result<BatchDocument<QuestionDocumentArgs>, BatchItemFailure> {
        let args = document.task_args.as_ref().ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if document.text.len().checked_add(args.question.len())
            .is_none_or(|bytes| bytes > self.limits.max_input_bytes_per_item) {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        Ok(document)
    }
    fn durable_work(&self, prepared: &PreparedInt4QuestionDocument) -> Result<JobWork, BatchItemFailure> {
        // A prepared receipt from another factory or a pre-schedule owner must
        // never authorize even a persisted failed-attempt debit.
        self.compiler.owns(prepared)?;
        let work = JobWork::from_int4(prepared.planned_work(), prepared.reserved_mask_visits());
        if !work.fits(self.limits.max_work) { return Err(BatchItemFailure::fatal(BatchCode::WorkLimit)); }
        Ok(work)
    }
}

/// Thin durable adapter. The native corpus executor remains the only owner of
/// all-chunk preflight, reconstruction, admission, cancellation and finalizers.
pub struct Int4QuestionDocumentJobProcessor<'p, 'e, 'w, 'v, A: Int4QuestionDocumentBatchAdmission> {
    planner: &'p Int4QuestionDocumentJobPlanner,
    native: NativeInt4QuestionDocumentBatch<'p, 'e, 'w, 'v, A>,
}
impl<'p, 'e, 'w, 'v, A: Int4QuestionDocumentBatchAdmission> Int4QuestionDocumentJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int4QuestionDocumentJobPlanner, engine: &'e mut StrictInt4Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        planner.check_job_limits(planner.limits)?;
        let native = NativeInt4QuestionDocumentBatch::new(&planner.compiler, engine, vocabulary, admission)?;
        Ok(Self { planner, native })
    }
    pub fn new_cohorts(planner: &'p Int4QuestionDocumentJobPlanner, weights: &'w Int4ModelWeights,
        vocabulary: &'v ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        planner.check_job_limits(planner.limits)?;
        let native = NativeInt4QuestionDocumentBatch::new_cohorts(&planner.compiler,
            weights, vocabulary, budget, kernel, admission)?;
        Ok(Self { planner, native })
    }
}
impl<A: Int4QuestionDocumentBatchAdmission> BatchProcessor for Int4QuestionDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Args = QuestionDocumentArgs;
    type Prepared = PreparedInt4QuestionDocument;
    type Output = GuardedOutput<Int4QuestionDocumentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(self.planner.checked_document(document)?, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int4QuestionDocumentBatchAdmission> DurableBatchProcessor for Int4QuestionDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int4QuestionDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.planner.durable_work(prepared)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}

pub(crate) fn check_limits(config: Int4QuestionDocumentBatchConfig, limits: JobLimits) -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    config.validate()?;
    if limits.max_work != JobWork::from_int4(config.max_model_work, config.max_mask_visits)
        || config.mapping.chunks.max_input_bytes > limits.max_input_bytes_per_item
        || config.max_question_bytes > limits.max_input_bytes_per_item
        || config.mapping.reduction.max_result_bytes > limits.max_result_bytes {
        return Err(BatchCode::InvalidLimits.into());
    }
    // These six counters are not journal columns. The immutable attempt cap
    // therefore bounds their total across every process, including failures;
    // reopening a fresh invocation ledger cannot create more authority.
    verification_lifetime(config.mapping.verification, config.max_map_verification, limits.max_attempts)?;
    verification_lifetime(config.verification, config.max_verification, limits.max_attempts)
}
fn verification_lifetime(per: GroundingBudget, lifetime: GroundingBudget, attempts: u64) -> Result<(), BatchFault> {
    let count = usize::try_from(attempts).map_err(|_| BatchCode::InvalidLimits)?;
    if count == 0 || per.max_fields.checked_mul(count).is_none_or(|n| n > lifetime.max_fields)
        || per.max_matches.checked_mul(count).is_none_or(|n| n > lifetime.max_matches)
        || per.max_scan_steps.checked_mul(attempts).is_none_or(|n| n > lifetime.max_scan_steps) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

#[cfg(test)] mod tests;
