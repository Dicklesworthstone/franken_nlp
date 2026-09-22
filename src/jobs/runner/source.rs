//! Built-in durable processor for resident INT8 NER, keyphrases, cited
//! summaries and passage QA. Model/vocabulary and admission remain host-owned.
//! This constructs the EXISTING native batch adapter from the exact same
//! private settings that the durable runner freezes. No fixture result or
//! deserialized recipe can replace the native executable or admission identity.
use super::DurableBatchProcessor;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor, BatchRequestContext, BatchWork,
        source::{GuardedOutput, SourceBatchArgs, SourceMaskBudget,
            quantized::{self, Int8SourceBatchAdmission, Int8SourceBatchLimits, Int8SourceBatchPlanner, NativeInt8SourceBatch}}},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::{SourceRuntimeLimits, SOURCE_JSON_RUNTIME_VERSION}},
    jobs::{JobWork, manifest::bounded_json},
    execution_identity::ExecutionIdentity,
    native_engine::{decode::DecodeStepControl, strict_int8::{Int8Work, StrictInt8Engine}},
    tasks::{extract::ExtractionVocabulary, ir::TaskBudget,
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SOURCE_PROMPT_VERSION,
            quantized::{Int8SourceTaskRun, PreparedInt8SourceTask, INT8_SOURCE_EXECUTION}}},
};
use serde::Serialize;

/// A sealed, serializable contract projection. Not Deserialize, not a task
/// factory, and deliberately not Debug: defaults can contain private passages.
#[derive(Serialize)]
pub struct Int8SourceJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    source_runtime: &'static str,
    task_ceiling: TaskBudget,
    planning: PlanningRecipe,
    native: NativeRecipe,
    defaults: Option<SourceBatchArgs>,
}
#[derive(Serialize)]
struct PlanningRecipe {
    max_input_bytes: usize, max_context_tokens: usize, max_passages: usize,
    compiler: CompilerRecipe, source: SourceRuntimeLimits,
}
#[derive(Serialize)]
struct CompilerRecipe {
    max_schema_bytes: usize, max_string_bytes: usize, max_array_items: usize,
    max_output_bytes: usize, max_states: usize, max_transitions: usize, max_mask_bytes: usize,
}
#[derive(Serialize)]
struct NativeRecipe {
    max_model_work: Int8Work, max_trie_node_visits: usize, checkpoint_interval_nodes: usize,
    max_visits_per_item: u64, max_visits_per_run: u64,
}
impl Int8SourceJobRecipe {
    fn new(task_ceiling: TaskBudget, planning: SourcePlanningLimits,
        native: Int8SourceBatchLimits, defaults: Option<SourceBatchArgs>) -> Self {
        // Exhaustive destructuring intentionally fails compilation if these
        // non-Serialize limit structs gain an unbound configuration field.
        let SourcePlanningLimits { max_input_bytes, max_context_tokens, max_passages, compiler, source } = planning;
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
            max_states, max_transitions, max_mask_bytes } = compiler;
        let Int8SourceBatchLimits { max_model_work, masks } = native;
        let SourceMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        Self { version: 1, dependency_scope: "item-local", execution: INT8_SOURCE_EXECUTION,
            prompt_version: SOURCE_PROMPT_VERSION, source_runtime: SOURCE_JSON_RUNTIME_VERSION,
            task_ceiling, planning: PlanningRecipe { max_input_bytes, max_context_tokens, max_passages,
                compiler: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                    max_output_bytes, max_states, max_transitions, max_mask_bytes }, source },
            native: NativeRecipe { max_model_work, max_trie_node_visits, checkpoint_interval_nodes,
                max_visits_per_item, max_visits_per_run }, defaults }
    }
}

/// Processor for JobRunner, not a standalone durable job. It exclusively
/// borrows the same resident native engine and immutable vocabulary/planner.
/// The host's real admission guard is retained in GuardedOutput until the
/// runner finishes the durable commit, not merely until inference returns.
pub struct Int8SourceJobProcessor<'p, 'e, 'weights, 'v, A: Int8SourceBatchAdmission> {
    native: NativeInt8SourceBatch<'p, 'e, 'weights, 'v, A>,
    identity: ExecutionIdentity,
    recipe: Int8SourceJobRecipe,
}
impl<'p, 'e, 'weights, 'v, A: Int8SourceBatchAdmission> Int8SourceJobProcessor<'p, 'e, 'weights, 'v, A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(planner: &'p SourceTaskPlanner, identity: ExecutionIdentity, task_ceiling: TaskBudget,
        planning: SourcePlanningLimits, defaults: Option<SourceBatchArgs>,
        engine: &'e mut StrictInt8Engine<'weights>, vocabulary: &'v ExtractionVocabulary,
        admission: A, native_limits: Int8SourceBatchLimits) -> Result<Self, BatchFault> {
        // Bound private defaults before cloning and retain the same exact
        // configuration in the compiler and the keyed job recipe.
        quantized::check_configuration(planner, &identity, task_ceiling, planning, defaults.as_ref())?;
        quantized::validate_limits(native_limits)?;
        let compiler = Int8SourceBatchPlanner::new(planner, identity.clone(), task_ceiling, planning, defaults.clone())?;
        let recipe = Int8SourceJobRecipe::new(task_ceiling, planning, native_limits, defaults);
        // Bound the whole projection, not only its optional default arguments.
        // Leave space for the runner's outer protocol/input-profile envelope.
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchFault::from(BatchCode::InvalidLimits))?;
        let native = NativeInt8SourceBatch::new(compiler, engine, vocabulary, admission, native_limits)?;
        Ok(Self { native, identity, recipe })
    }
}
impl<A: Int8SourceBatchAdmission> BatchProcessor for Int8SourceJobProcessor<'_, '_, '_, '_, A> {
    type Args = SourceBatchArgs;
    type Prepared = PreparedInt8SourceTask;
    type Output = GuardedOutput<Int8SourceTaskRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(document, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.native.execute(prepared, control)
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // Existing code admits actual identity/work/KV/masks/result storage,
        // rechecks native model and vocabulary, finalizes inside the native
        // session, validates observed work, and refuses nonempty/poisoned KV.
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int8SourceBatchAdmission> DurableBatchProcessor for Int8SourceJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int8SourceJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(JobWork { model: prepared.planned_work(), mask_node_visits: self.recipe.native.max_visits_per_item })
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)] mod tests;
