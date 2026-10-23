//! Authenticated item-local jobs for the grouped-INT4 source-task portfolio.
//!
//! The immutable factory binds the actual model and every compiler/default/
//! native limit before storage is opened. JobRunner persists all group-aware
//! work debits before execution and retains guarded results through durable
//! spool sync and journal acknowledgement. No task failure becomes abstention.
use super::{CompilerRecipe, PlanningRecipe};
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        source::{GuardedOutput, SourceBatchArgs, SourceMaskBudget,
            grouped_int4::{self as source_batch, Int4SourceBatchAdmission,
                Int4SourceBatchLimits, Int4SourceBatchPlanner, NativeInt4SourceBatch}}},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SOURCE_JSON_RUNTIME_VERSION},
    jobs::{JobLimits, JobWork, manifest::bounded_json, runner::DurableBatchProcessor},
    native_engine::{decode::DecodeStepControl, generation::grouped_int4::bind_candidate_model,
        strict_int4::{Int4MemoryRequirement, Int4ModelIdentity, Int4Work, StrictInt4Engine,
            STRICT_INT4_EXECUTION}},
    tasks::{BuiltInTask, extract::ExtractionVocabulary, ir::TaskBudget,
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, SOURCE_PROMPT_VERSION,
            grouped_int4::{Int4SourceTaskRun, PreparedInt4SourceTask, INT4_SOURCE_EXECUTION}}},
};
use serde::Serialize;
use std::sync::Arc;

const INPUT_POLICY: &str = "typed-source-records-fixed-task-full-resident-kv-v1";

/// A private-content recipe projection, not a deserializable task factory.
/// Only its per-job keyed commitment enters persistent metadata. In particular,
/// caller-provided QA passages are never exposed by Debug or public digests.
#[derive(Serialize)]
pub struct Int4SourceJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    input_policy: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    source_runtime: &'static str,
    task_spec: String,
    group_width: usize,
    task_ceiling: TaskBudget,
    planning: PlanningRecipe,
    native: NativeRecipe,
    defaults: Option<SourceBatchArgs>,
}
#[derive(Serialize)]
struct NativeRecipe {
    max_model_work: Int4Work,
    max_trie_node_visits: usize,
    checkpoint_interval_nodes: usize,
    max_visits_per_item: u64,
    max_visits_per_run: u64,
}
impl Int4SourceJobRecipe {
    fn new(task_spec: String, group_width: usize, task_ceiling: TaskBudget,
        planning: SourcePlanningLimits, native: Int4SourceBatchLimits,
        defaults: Option<SourceBatchArgs>) -> Result<Self, BatchFault> {
        // Exhaustive projections force every new limit into resume authority.
        // SourceRuntimeLimits is serialized in full, including both its index
        // and independent verification budgets.
        let SourcePlanningLimits { max_input_bytes, max_context_tokens, max_passages, compiler, source } = planning;
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
            max_states, max_transitions, max_mask_bytes } = compiler;
        let Int4SourceBatchLimits { max_model_work, masks } = native;
        let SourceMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        let recipe = Self { version: 1, dependency_scope: "item-local", input_policy: INPUT_POLICY,
            execution: INT4_SOURCE_EXECUTION, prompt_version: SOURCE_PROMPT_VERSION,
            source_runtime: SOURCE_JSON_RUNTIME_VERSION, task_spec, group_width, task_ceiling,
            planning: PlanningRecipe { max_input_bytes, max_context_tokens, max_passages,
                compiler: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                    max_output_bytes, max_states, max_transitions, max_mask_bytes }, source },
            native: NativeRecipe { max_model_work, max_trie_node_visits, checkpoint_interval_nodes,
                max_visits_per_item, max_visits_per_run }, defaults };
        // Bound complete private defaults before the native compiler clones
        // them. Leave room for JobRunner's protocol/input-profile envelope.
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

/// An actual-model-bound source-task factory. One task is fixed for the whole
/// frozen population; each typed record may choose bounded options/passages
/// and tighten token/output/grammar limits. Complete resident KV is invariant.
pub struct Int4SourceJobPlanner {
    planner: Arc<SourceTaskPlanner>,
    identity: ExecutionIdentity,
    planning: SourcePlanningLimits,
    native: Int4SourceBatchLimits,
    recipe: Int4SourceJobRecipe,
}
impl Int4SourceJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn for_model(model: &Int4ModelIdentity, planner: Arc<SourceTaskPlanner>, task: BuiltInTask,
        ceiling: TaskBudget, planning: SourcePlanningLimits, defaults: Option<SourceBatchArgs>,
        native: Int4SourceBatchLimits) -> Result<Self, BatchFault> {
        let absent = Sha256Digest::of_bytes(b"null");
        // Unbound model fields never escape: bind_candidate_model derives
        // source, content and packing identity from the loaded weights.
        let identity = bind_candidate_model(ExecutionIdentity {
            schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
            artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(),
            packing_set_digest: absent, tokenizer_digest: planner.tokenizer_digest(),
            template_digest: *planner.template_digest(), task_spec: task.spec().identity(),
            taskir_digest: absent, prompt_digest: absent, grammar_compiler_version: "uncompiled".to_owned(),
            schema_digest: absent, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
            kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: absent, decision_policy_digest: absent,
            backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
        }, model).map_err(|_| BatchCode::Admission)?;
        Self::configured(planner, identity, model.group_width(), ceiling, planning, defaults, native)
    }

    // Private setup seam also used by planning/storage fixtures. Public
    // callers can construct this factory only from a real Int4ModelIdentity.
    fn configured(planner: Arc<SourceTaskPlanner>, identity: ExecutionIdentity, group_width: usize,
        ceiling: TaskBudget, planning: SourcePlanningLimits, defaults: Option<SourceBatchArgs>,
        native: Int4SourceBatchLimits) -> Result<Self, BatchFault> {
        source_batch::check_configuration(&planner, &identity, ceiling, planning, defaults.as_ref(), group_width)?;
        source_batch::validate_limits(native)?;
        let resident = Int4MemoryRequirement::for_context(planning.max_context_tokens, group_width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if resident.kv_bytes == 0 || ceiling.max_kv_bytes != resident.kv_bytes {
            return Err(BatchCode::InvalidLimits.into());
        }
        if let Some(args) = &defaults {
            check_args(args, &identity.task_spec, ceiling)?;
        }
        let recipe = Int4SourceJobRecipe::new(identity.task_spec.clone(), group_width,
            ceiling, planning, native, defaults)?;
        Ok(Self { planner, identity, planning, native, recipe })
    }

    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn native_limits(&self) -> Int4SourceBatchLimits { self.native }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let expected = JobWork::from_int4(self.native.max_model_work, self.native.masks.max_visits_per_run);
        if limits.max_work != expected || self.task_ceiling().max_output_bytes > limits.max_result_bytes as u64 {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }

    fn compiler(&self) -> Result<Int4SourceBatchPlanner<'_>, BatchFault> {
        Int4SourceBatchPlanner::new(&self.planner, self.identity.clone(), self.task_ceiling(),
            self.planning, self.recipe.defaults.clone(), self.recipe.group_width)
    }

    fn checked_document(&self, document: BatchDocument<SourceBatchArgs>)
        -> Result<BatchDocument<SourceBatchArgs>, BatchItemFailure> {
        if let Some(args) = &document.task_args {
            check_args(args, &self.identity.task_spec, self.task_ceiling()).map_err(BatchItemFailure::reject)?;
        }
        Ok(document)
    }
}

fn check_args(args: &SourceBatchArgs, task_spec: &str, ceiling: TaskBudget) -> Result<(), BatchFault> {
    let (kind, budget) = match args {
        SourceBatchArgs::Ner { budget, .. } => (BuiltInTask::Ner, *budget),
        SourceBatchArgs::Keyphrases { budget, .. } => (BuiltInTask::Keyphrases, *budget),
        SourceBatchArgs::Summarize { budget, .. } => (BuiltInTask::Summarize, *budget),
        SourceBatchArgs::Answer { budget, .. } => (BuiltInTask::Answer, *budget),
    };
    budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if kind.spec().identity() != task_spec || budget.max_input_tokens > ceiling.max_input_tokens
        || budget.max_output_tokens > ceiling.max_output_tokens || budget.max_output_bytes > ceiling.max_output_bytes
        || budget.max_grammar_states > ceiling.max_grammar_states || budget.max_kv_bytes != ceiling.max_kv_bytes {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}

/// One exclusive native processor borrowing its immutable factory. Its
/// GuardedOutput stays intact while JobRunner serializes and commits results.
pub struct Int4SourceJobProcessor<'p, 'e, 'weights, 'v, A: Int4SourceBatchAdmission> {
    native: NativeInt4SourceBatch<'p, 'e, 'weights, 'v, A>,
    planner: &'p Int4SourceJobPlanner,
}
impl<'p, 'e, 'weights, 'v, A: Int4SourceBatchAdmission> Int4SourceJobProcessor<'p, 'e, 'weights, 'v, A> {
    pub fn new(planner: &'p Int4SourceJobPlanner, engine: &'e mut StrictInt4Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        let native = NativeInt4SourceBatch::new(planner.compiler()?, engine, vocabulary, admission, planner.native)?;
        Ok(Self { native, planner })
    }
}
impl<A: Int4SourceBatchAdmission> BatchProcessor for Int4SourceJobProcessor<'_, '_, '_, '_, A> {
    type Args = SourceBatchArgs;
    type Prepared = PreparedInt4SourceTask;
    type Output = GuardedOutput<Int4SourceTaskRun, A::Guard>;
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
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int4SourceBatchAdmission> DurableBatchProcessor for Int4SourceJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int4SourceJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { &self.planner.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.planner.check_job_limits(limits)
    }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        // These are actual group-aware planned counts. A fresh native ledger
        // on resume never refunds the journal's previous failed attempts.
        Ok(JobWork::from_int4(prepared.planned_work(), self.planner.native.masks.max_visits_per_item))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}

#[cfg(test)]
mod tests;
