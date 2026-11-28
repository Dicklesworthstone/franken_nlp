//! Durable complete-source INT8 QA with an exact private question per item.
//! The resident factory remains the sole native plan/execution authority.
//! Raw population envelopes bind source/question; no chunk is a journal item.
use super::*;
use std::sync::Arc;
use crate::{
    batch::source::quantized::question::{Int8QuestionDocumentBatchConfig, Int8QuestionDocumentBatchPlanner,
        Int8QuestionDocumentBatchAdmission, Int8QuestionDocumentRun, NativeInt8QuestionDocumentBatch,
        PreparedInt8QuestionDocument, encoding::EncodingRecipe},
    jobs::{JobLimits, question_document::quantized::check_limits},
    tasks::{answer::AnswerOptions, mapreduce::{CHUNK_PROFILE, ChunkLimits, ExecutionLimits},
        source_planning::quantized::long::{Int8SourceMapLimits, question::INT8_QUESTION_EXECUTION}},
    validation::grounded_fields::GroundingBudget,
};
pub use crate::batch::source::quantized::question::QuestionDocumentArgs;
pub mod memory {
    pub use crate::jobs::question_document::quantized::{Int8QuestionDocumentJobMemoryLimits,
        Int8QuestionDocumentJobMemoryRequirement};
}
const RECIPE_BYTES: usize = 1024 * 1024 - 1024;

/// Fixed policy only; private source/question bytes occur exclusively in the
/// authenticated original item envelopes. No deserializable task authority.
#[derive(Serialize)]
pub struct Int8QuestionDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    reservation_policy: &'static str,
    source: Int8SourceJobRecipe,
    options: AnswerOptions,
    mapping: MappingRecipe,
    verification: GroundingBudget,
    max_question_bytes: usize,
    max_verification: GroundingBudget,
    max_attempts: u64,
    max_stored_result_bytes: u64,
    max_output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill: Option<PrefillRecipe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<EncodingRecipe>,
}
#[derive(Serialize)]
struct MappingRecipe {
    chunks: ChunkLimits, reduction: ExecutionLimits, max_model_work: Int8Work,
    max_trie_node_visits: usize, checkpoint_interval_nodes: usize,
    mask_visits_per_chunk: u64, max_mask_visits: u64,
}
#[derive(Serialize)]
struct PrefillRecipe { max_batch_rows: usize, max_extra_scratch_bytes: u64 }
impl Int8QuestionDocumentJobRecipe {
    fn new(compiler: &Int8QuestionDocumentBatchPlanner, limits: JobLimits, prefill: Option<Int8PrefillLimits>)
        -> Result<Self, BatchFault> {
        // Exhaustive projections force future policy fields into resume
        // authority instead of accidentally ignoring a new mutable setting.
        let Int8QuestionDocumentBatchConfig { options, budget, planning, mapping, verification,
            max_question_bytes, max_model_work, max_mask_visits, max_verification } = *compiler.config();
        let Int8SourceMapLimits { chunks, reduction, max_model_work: per_document_work,
            mask_limits, mask_visits_per_chunk, max_mask_visits: per_document_masks } = mapping;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = mask_limits;
        let native = Int8SourceBatchLimits { max_model_work, masks: SourceMaskBudget {
            per_mask: mask_limits, max_visits_per_item: per_document_masks, max_visits_per_run: max_mask_visits } };
        let source = Int8SourceJobRecipe::new(budget, planning, native, None);
        let prefill = prefill.map(|limits| {
            limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
            Ok::<_, BatchFault>(PrefillRecipe { max_batch_rows: limits.max_batch_rows,
                max_extra_scratch_bytes: limits.max_extra_scratch_bytes })
        }).transpose()?;
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-source-question",
            input_policy: "required-private-question-fixed-answer-policy-v1", execution: INT8_QUESTION_EXECUTION,
            chunk_profile: CHUNK_PROFILE,
            reservation_policy: "exact-five-axis-mask-debit-verification-across-all-durable-attempts-v1",
            source, options, mapping: MappingRecipe { chunks, reduction, max_model_work: per_document_work,
                max_trie_node_visits, checkpoint_interval_nodes, mask_visits_per_chunk, max_mask_visits: per_document_masks },
            verification, max_question_bytes, max_verification, max_attempts: limits.max_attempts,
            max_stored_result_bytes: compiler.max_result_bytes(), max_output_tokens: compiler.max_output_tokens(),
            prefill, input_encoding: compiler.encoding_recipe()? };
        bounded_json(&recipe, RECIPE_BYTES).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

pub struct Int8QuestionDocumentJobPlanner {
    compiler: Int8QuestionDocumentBatchPlanner, recipe: Int8QuestionDocumentJobRecipe,
    limits: JobLimits, prefill: Option<Int8PrefillLimits>,
}
impl Int8QuestionDocumentJobPlanner {
    pub fn new(compiler: Int8QuestionDocumentBatchPlanner, limits: JobLimits) -> Result<Self, BatchFault> {
        check_limits(*compiler.config(), limits)?;
        let recipe = Int8QuestionDocumentJobRecipe::new(&compiler, limits, None)?;
        Ok(Self { compiler, recipe, limits, prefill: None })
    }
    /// One retained operation pays prepare and reconstruction for each pending
    /// original pair. Tokenizer work is invocation-local, not a journal counter.
    pub fn with_map_document_encoding(mut self, policy: UntrustedEncoding, complete: SafeBpeWorkLimits)
        -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_map_document_encoding(policy, complete)?;
        self.recipe = Int8QuestionDocumentJobRecipe::new(&self.compiler, self.limits, self.prefill)?;
        Ok(self)
    }
    /// Bind prompt morsels and their complete separate workspace into resume
    /// policy. Changing schedule invalidates every previously prepared owner.
    pub fn with_layer_major_prefill(mut self, prefill: Int8PrefillLimits) -> Result<Self, BatchFault> {
        prefill.validate().map_err(|_| BatchCode::InvalidLimits)?;
        self.compiler.reset_preparation_owner(); self.prefill = Some(prefill);
        self.recipe = Int8QuestionDocumentJobRecipe::new(&self.compiler, self.limits, self.prefill)?;
        Ok(self)
    }
    pub fn prefill_limits(&self) -> Option<Int8PrefillLimits> { self.prefill }
    pub fn batch_planner(&self) -> &Int8QuestionDocumentBatchPlanner { &self.compiler }
    pub fn retained_owners(&self) -> Arc<SourceTaskPlanner> { self.compiler.retained_owners() }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.compiler.execution_identity() }
    pub fn config(&self) -> &Int8QuestionDocumentBatchConfig { self.compiler.config() }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn job_recipe(&self) -> &Int8QuestionDocumentJobRecipe { &self.recipe }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn map_document_encoding_limits(&self) -> Option<SafeBpeWorkLimits> { self.compiler.map_document_encoding_limits() }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> { self.compiler.encoding_workspace_bytes() }
    pub fn task_budget(&self) -> TaskBudget { self.config().budget }
    pub fn planning_limits(&self) -> SourcePlanningLimits { self.config().planning }
    pub fn document_limits(&self) -> Int8SourceMapLimits { self.config().mapping }
    pub fn max_result_bytes(&self) -> u64 { self.compiler.max_result_bytes() }
    pub fn max_output_tokens(&self) -> u64 { self.compiler.max_output_tokens() }
    pub fn retained_defaults_bytes(&self) -> Result<u64, BatchFault> { Ok(0) }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); }
        check_limits(*self.config(), limits)
    }
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<QuestionDocumentArgs>, control: &mut C)
        -> Result<PreparedInt8QuestionDocument, BatchItemFailure> {
        let document = self.checked_document(document)?;
        let args = document.task_args.ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        self.compiler.prepare_with_control(document.text, args, control)
    }
    fn checked_document(&self, document: BatchDocument<QuestionDocumentArgs>)
        -> Result<BatchDocument<QuestionDocumentArgs>, BatchItemFailure> {
        let args = document.task_args.as_ref().ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
        if document.text.len().checked_add(args.question.len()).is_none_or(|bytes| bytes > self.limits.max_input_bytes_per_item) {
            return Err(BatchItemFailure::reject(BatchCode::DocumentLimit));
        }
        Ok(document)
    }
    fn durable_work(&self, prepared: &PreparedInt8QuestionDocument) -> Result<JobWork, BatchItemFailure> {
        self.compiler.owns(prepared)?;
        let work = JobWork { model: prepared.planned_work(), mask_node_visits: prepared.reserved_mask_visits() };
        if !work.fits(self.limits.max_work) { return Err(BatchItemFailure::fatal(BatchCode::WorkLimit)); }
        Ok(work)
    }
}

/// Actual native corpus execution supplies all preparation, exact replan,
/// finalizer and cancellation behavior; only durable recipe/debit seams differ.
pub struct Int8QuestionDocumentJobProcessor<'p, 'e, 'w, 'v, A: Int8QuestionDocumentBatchAdmission> {
    planner: &'p Int8QuestionDocumentJobPlanner,
    native: NativeInt8QuestionDocumentBatch<'p, 'e, 'w, 'v, A>,
}
impl<'p, 'e, 'w, 'v, A: Int8QuestionDocumentBatchAdmission> Int8QuestionDocumentJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int8QuestionDocumentJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        planner.check_job_limits(planner.limits)?;
        let native = match planner.prefill {
            Some(prefill) => NativeInt8QuestionDocumentBatch::new_layer_major(&planner.compiler, engine, vocabulary, admission, prefill),
            None => NativeInt8QuestionDocumentBatch::new(&planner.compiler, engine, vocabulary, admission),
        }?;
        Ok(Self { planner, native })
    }
}
impl<A: Int8QuestionDocumentBatchAdmission> BatchProcessor for Int8QuestionDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Args = QuestionDocumentArgs;
    type Prepared = PreparedInt8QuestionDocument;
    type Output = GuardedOutput<Int8QuestionDocumentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(self.planner.checked_document(document)?)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(self.planner.checked_document(document)?, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, context: BatchRequestContext, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute_with_context(prepared, context, control) }
}
impl<A: Int8QuestionDocumentBatchAdmission> DurableBatchProcessor for Int8QuestionDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int8QuestionDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        self.planner.durable_work(prepared)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}

#[cfg(test)] mod tests;
