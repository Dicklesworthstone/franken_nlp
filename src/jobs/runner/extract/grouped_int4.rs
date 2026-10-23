//! Fixed-schema durable extraction over the actual grouped-INT4 executor.
//! JobRunner owns authenticated replay and durable commit; this sealed bridge
//! binds the SAME compiler/defaults/native ceilings and preserves all work axes.
use super::{CompilerRecipe, DurableBatchProcessor};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionMaskBudget, GuardedOutput, EXTRACTION_BATCH_PROMPT,
            grouped_int4::{Int4ExtractionBatchAdmission, Int4ExtractionBatchLimits,
                Int4ExtractionBatchPlanner, NativeInt4ExtractionBatch, PreparedInt4BatchExtraction}}},
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, mask::MaskWorkLimits,
        runtime::{SourceRuntimeLimits, JSON_RUNTIME_VERSION, SOURCE_JSON_RUNTIME_VERSION}},
    jobs::{JobLimits, JobWork, manifest::bounded_json},
    native_engine::{decode::DecodeStepControl,
        strict_int4::{Int4ModelIdentity, Int4Work, StrictInt4Engine}},
    tasks::{extract::{ExtractionVocabulary, grouped_int4::{Int4ExtractRun, INT4_EXTRACT_VERSION}},
        ir::TaskBudget},
};
use serde::Serialize;

const INPUT_POLICY: &str = "fixed-shared-schema-no-record-overrides-v1";

/// Only a keyed commitment to these private settings enters the journal.
/// The actual strict-v2 content/packing identity is committed independently;
/// this is neither a deserializable task factory nor a public content digest.
#[derive(Serialize)]
pub struct Int4ExtractionJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    json_runtime: &'static str,
    source_runtime: &'static str,
    group_width: usize,
    task_ceiling: TaskBudget,
    compiler: CompilerRecipe,
    source: SourceRuntimeLimits,
    native: NativeRecipe,
    defaults: ExtractionBatchArgs,
}
#[derive(Serialize)]
struct NativeRecipe {
    max_model_work: Int4Work,
    max_trie_node_visits: usize,
    checkpoint_interval_nodes: usize,
    max_visits_per_item: u64,
    max_visits_per_run: u64,
}
impl Int4ExtractionJobRecipe {
    fn new(group_width: usize, task_ceiling: TaskBudget, compiler: CompileLimits,
        source: SourceRuntimeLimits, native: Int4ExtractionBatchLimits, defaults: ExtractionBatchArgs)
        -> Result<Self, BatchFault> {
        native.validate()?;
        task_ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let b = defaults.budget;
        b.validate().map_err(|_| BatchCode::InvalidLimits)?;
        if b.max_input_tokens > task_ceiling.max_input_tokens
            || b.max_output_tokens > task_ceiling.max_output_tokens
            || b.max_output_bytes > task_ceiling.max_output_bytes
            || b.max_grammar_states > task_ceiling.max_grammar_states
            || b.max_kv_bytes != task_ceiling.max_kv_bytes {
            return Err(BatchCode::InvalidLimits.into());
        }
        // Exhaustive projections force new compiler/native axes into the
        // immutable recipe instead of silently omitting them from resume.
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
            max_states, max_transitions, max_mask_bytes } = compiler;
        let Int4ExtractionBatchLimits { max_model_work, masks } = native;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        let recipe = Self { version: 1, dependency_scope: "item-local", input_policy: INPUT_POLICY,
            execution: INT4_EXTRACT_VERSION, prompt_version: EXTRACTION_BATCH_PROMPT,
            json_runtime: JSON_RUNTIME_VERSION, source_runtime: SOURCE_JSON_RUNTIME_VERSION,
            group_width, task_ceiling,
            compiler: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                max_output_bytes, max_states, max_transitions, max_mask_bytes }, source,
            native: NativeRecipe { max_model_work, max_trie_node_visits, checkpoint_interval_nodes,
                max_visits_per_item, max_visits_per_run }, defaults };
        // Bound before cloning exact schema bytes into the native compiler.
        // Keep room for JobRunner's protocol/input-profile wrapper.
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

/// Immutable factory built from real resident source/snapshot model identity.
/// Every record uses the same exact schema and grounding/budget policy. The
/// compiler rebinds source coordinates and prompt identity to EACH document.
pub struct Int4ExtractionJobPlanner {
    compiler: Int4ExtractionBatchPlanner,
    identity: ExecutionIdentity,
    recipe: Int4ExtractionJobRecipe,
    native: Int4ExtractionBatchLimits,
}
impl Int4ExtractionJobPlanner {
    pub fn for_model(model: &Int4ModelIdentity, ceiling: TaskBudget, compiler_limits: CompileLimits,
        source_limits: SourceRuntimeLimits, defaults: ExtractionBatchArgs,
        native: Int4ExtractionBatchLimits) -> Result<Self, BatchFault> {
        let recipe = Int4ExtractionJobRecipe::new(model.group_width(), ceiling, compiler_limits,
            source_limits, native, defaults)?;
        let compiler = Int4ExtractionBatchPlanner::for_model(model, ceiling, compiler_limits,
            source_limits, Some(recipe.defaults.clone()))?.require_shared_defaults()?;
        let identity = compiler.base_execution_identity().clone();
        Ok(Self { compiler, identity, recipe, native })
    }

    // Synthetic identity is only a planning fixture, never a production
    // alternate constructor or authority to manufacture resident weights.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn pinned(controls: &crate::tokenizer::specials::TemplateControlIds, eos: u32,
        identity: ExecutionIdentity, group_width: usize, ceiling: TaskBudget,
        compiler_limits: CompileLimits, source_limits: SourceRuntimeLimits,
        defaults: ExtractionBatchArgs, native: Int4ExtractionBatchLimits) -> Result<Self, BatchFault> {
        let recipe = Int4ExtractionJobRecipe::new(group_width, ceiling, compiler_limits,
            source_limits, native, defaults)?;
        let compiler = Int4ExtractionBatchPlanner::pinned(controls, eos, identity, group_width, ceiling,
            compiler_limits, source_limits, Some(recipe.defaults.clone()))?.require_shared_defaults()?;
        let identity = compiler.base_execution_identity().clone();
        Ok(Self { compiler, identity, recipe, native })
    }

    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn native_limits(&self) -> Int4ExtractionBatchLimits { self.native }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        check_job_limits(&self.recipe, limits)
    }
}
fn check_job_limits(recipe: &Int4ExtractionJobRecipe, limits: JobLimits) -> Result<(), BatchFault> {
    limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
    let expected = JobWork::from_int4(recipe.native.max_model_work, recipe.native.max_visits_per_run);
    if limits.max_work != expected || recipe.task_ceiling.max_output_bytes > limits.max_result_bytes as u64
        || recipe.defaults.budget.max_kv_bytes != recipe.task_ceiling.max_kv_bytes {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

/// One real native processor. Its output guard remains inside the result
/// while JobRunner syncs the result frame and commits its journal pointer.
pub struct Int4ExtractionJobProcessor<'e, 'weights, 'v, A: Int4ExtractionBatchAdmission> {
    native: NativeInt4ExtractionBatch<'e, 'weights, 'v, A>,
    identity: ExecutionIdentity,
    recipe: Int4ExtractionJobRecipe,
}
impl<'e, 'weights, 'v, A: Int4ExtractionBatchAdmission> Int4ExtractionJobProcessor<'e, 'weights, 'v, A> {
    pub fn new(planner: Int4ExtractionJobPlanner, engine: &'e mut StrictInt4Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let Int4ExtractionJobPlanner { compiler, identity, recipe, native } = planner;
        let native = NativeInt4ExtractionBatch::new(compiler, engine, vocabulary, admission, native)?;
        Ok(Self { native, identity, recipe })
    }
}
impl<A: Int4ExtractionBatchAdmission> BatchProcessor for Int4ExtractionJobProcessor<'_, '_, '_, A> {
    // Option<()> rejects every non-null task_args during JobRunner::freeze,
    // before create/resume opens storage. No new runner hook is necessary.
    type Args = ();
    type Prepared = PreparedInt4BatchExtraction;
    type Output = GuardedOutput<Int4ExtractRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(shared_document(document)?)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(shared_document(document)?, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int4ExtractionBatchAdmission> DurableBatchProcessor for Int4ExtractionJobProcessor<'_, '_, '_, A> {
    type Recipe = Int4ExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        check_job_limits(&self.recipe, limits)
    }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        // Copy the already group-aware quote. Recomputing it with INT8
        // formulas would undercharge group dots and corrupt durable authority.
        Ok(JobWork::from_int4(prepared.planned_work(), self.recipe.native.max_visits_per_item))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.extraction_plan().max_result_bytes() }
}
fn shared_document(document: BatchDocument<()>) -> Result<BatchDocument<ExtractionBatchArgs>, BatchItemFailure> {
    if document.task_args.is_some() { return Err(BatchItemFailure::fatal(BatchCode::InvalidEnvelope)); }
    Ok(BatchDocument { id: document.id, text: document.text, task_args: None })
}

#[cfg(test)]
mod tests;
