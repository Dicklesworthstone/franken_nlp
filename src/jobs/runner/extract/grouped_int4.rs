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
    native_engine::{constrained_int4::sparse::{Int4JsonSparseLimits, INT4_SPARSE_JSON_EXECUTION},
        decode::DecodeStepControl, strict_int4::{Int4ModelIdentity, Int4Work, StrictInt4Engine,
            prefill::Int4PrefillLimits}},
    tasks::{extract::{ExtractionVocabulary,
        grouped_int4::{Int4ExtractRun, INT4_EXTRACT_VERSION, INT4_SPARSE_EXTRACT_VERSION}},
        ir::TaskBudget},
    tokenizer::untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding},
};
use serde::Serialize;

/// Durable complete-document extraction with exact native chunk accounting.
pub mod document;

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
    /// Compact encoding is explicit private resume authority. Omission
    /// preserves the original byte-fallback recipe exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<UntrustedEncoding>,
    /// Single-context jobs retain one allowance across every record and
    /// failed preparation in an invocation. Document jobs keep their existing
    /// separately authenticated complete-document preparation contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_work_limits: Option<SafeBpeWorkLimits>,
    /// The complete legal-row strategy and its refusal ceiling are immutable
    /// resume authority. Omission preserves every existing full-head recipe.
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_rows: Option<SelectedRowsRecipe>,
}
#[derive(Serialize)]
struct SelectedRowsRecipe {
    execution: &'static str,
    native_execution: &'static str,
    max_rows_per_step: usize,
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
                max_visits_per_item, max_visits_per_run }, defaults, input_encoding: None,
            encoding_work_limits: None, selected_rows: None };
        // Bound before cloning exact schema bytes into the native compiler.
        // Keep room for JobRunner's protocol/input-profile wrapper.
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
    // Only the single-context adapter calls this check. Complete-document
    // recipes authenticate their selected strategy in their outer map recipe.
    fn checked_selected_rows(&self) -> Result<Option<Int4JsonSparseLimits>, BatchFault> {
        match &self.selected_rows {
            None if self.execution == INT4_EXTRACT_VERSION => Ok(None),
            Some(selected) if self.execution == INT4_SPARSE_EXTRACT_VERSION
                && selected.execution == INT4_SPARSE_EXTRACT_VERSION
                && selected.native_execution == INT4_SPARSE_JSON_EXECUTION => {
                let limits = Int4JsonSparseLimits { max_rows_per_step: selected.max_rows_per_step };
                limits.validate().map_err(|_| BatchCode::Admission)?;
                Ok(Some(limits))
            }
            _ => Err(BatchCode::Admission.into()),
        }
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
    encoding_work: SafeBpeWork,
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
        Ok(Self { compiler, identity, recipe, native, encoding_work: SafeBpeWork::default() })
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
        Ok(Self { compiler, identity, recipe, native, encoding_work: SafeBpeWork::default() })
    }

    /// Project every legal grammar row under one fixed per-step ceiling.
    /// Exceeding the ceiling refuses the attempt; it never truncates the legal
    /// set or retries with the full head. Resume authenticates the strategy,
    /// ceiling and both execution contracts before repairing any retained tail.
    pub fn with_selected_rows(mut self, limits: Int4JsonSparseLimits) -> Result<Self, BatchFault> {
        self.check_selected_rows()?;
        self.compiler = self.compiler.with_selected_rows(limits)?;
        self.recipe.execution = INT4_SPARSE_EXTRACT_VERSION;
        self.recipe.selected_rows = Some(SelectedRowsRecipe {
            execution: INT4_SPARSE_EXTRACT_VERSION,
            native_execution: INT4_SPARSE_JSON_EXECUTION,
            max_rows_per_step: limits.max_rows_per_step,
        });
        self.check_selected_rows()?;
        bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    pub fn selected_rows(&self) -> Option<Int4JsonSparseLimits> { self.compiler.selected_rows() }
    fn check_selected_rows(&self) -> Result<(), BatchFault> {
        if self.selected_rows() != self.recipe.checked_selected_rows()? {
            return Err(BatchCode::Admission.into());
        }
        Ok(())
    }

    /// Select compact schema/source preparation before opening any job files.
    /// Both the per-record policy and all invocation ceilings are authenticated.
    /// One ledger moves with this factory into execution; changing records,
    /// failed preparation or reconstruction cannot renew it.
    pub fn with_document_encoding(mut self, policy: UntrustedEncoding,
        total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        self.check_document_encoding()?;
        encoding_workspace(&policy, total)?;
        if self.document_encoding() == &policy && self.recipe.encoding_work_limits == total {
            return Ok(self);
        }
        if self.encoding_work != SafeBpeWork::default() {
            return Err(BatchCode::InvalidExecution.into());
        }
        self.compiler = self.compiler.with_document_encoding(policy.clone())?;
        self.recipe.input_encoding = match policy {
            UntrustedEncoding::ByteFallback => None,
            compact @ UntrustedEncoding::SafeBpeV1(_) => Some(compact),
        };
        self.recipe.encoding_work_limits = total;
        self.check_document_encoding()?;
        bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn encoding_work_limits(&self) -> Option<SafeBpeWorkLimits> { self.recipe.encoding_work_limits }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        self.check_document_encoding()?;
        encoding_workspace(self.document_encoding(), self.encoding_work_limits())
    }
    fn check_document_encoding(&self) -> Result<(), BatchFault> {
        let expected = match &self.recipe.input_encoding {
            None => &UntrustedEncoding::ByteFallback,
            Some(UntrustedEncoding::ByteFallback) => return Err(BatchCode::Admission.into()),
            Some(compact @ UntrustedEncoding::SafeBpeV1(_)) => compact,
        };
        if self.document_encoding() != expected { return Err(BatchCode::Admission.into()); }
        encoding_workspace(expected, self.recipe.encoding_work_limits)?;
        Ok(())
    }

    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn native_limits(&self) -> Int4ExtractionBatchLimits { self.native }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.check_document_encoding()?;
        self.check_selected_rows()?;
        check_job_limits(&self.recipe, limits)
    }
}
fn encoding_workspace(policy: &UntrustedEncoding, total: Option<SafeBpeWorkLimits>)
    -> Result<u64, BatchFault> {
    match (policy, total) {
        (UntrustedEncoding::ByteFallback, None) => Ok(0),
        (UntrustedEncoding::SafeBpeV1(_), Some(total)) => {
            total.validate().map_err(|_| BatchCode::InvalidLimits)?;
            crate::batch::extract::grouped_int4::memory::encoding_workspace_bytes(policy)
        }
        _ => Err(BatchCode::Admission.into()),
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
    encoding_work: SafeBpeWork,
}
impl<'e, 'weights, 'v, A: Int4ExtractionBatchAdmission> Int4ExtractionJobProcessor<'e, 'weights, 'v, A> {
    pub fn new(planner: Int4ExtractionJobPlanner, engine: &'e mut StrictInt4Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        planner.check_document_encoding()?;
        planner.check_selected_rows()?;
        let Int4ExtractionJobPlanner { compiler, identity, recipe, native, encoding_work } = planner;
        let native = NativeInt4ExtractionBatch::new_with_document_encoding(
            compiler, engine, vocabulary, admission, native)?;
        Ok(Self { native, identity, recipe, encoding_work })
    }
    /// Execute each pending record with bounded layer-major prompt morsels.
    /// Row width is physical scheduling and does not change the private
    /// recipe, selected-row policy or any durable native/mask/attempt debit.
    /// The embedding host retains the additional workspace through the job.
    pub fn new_layer_major(planner: Int4ExtractionJobPlanner, engine: &'e mut StrictInt4Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A, prefill: Int4PrefillLimits)
        -> Result<Self, BatchFault> {
        planner.check_document_encoding()?;
        planner.check_selected_rows()?;
        let Int4ExtractionJobPlanner { compiler, identity, recipe, native, encoding_work } = planner;
        let native = NativeInt4ExtractionBatch::new_layer_major(
            compiler, engine, vocabulary, admission, native, prefill)?;
        Ok(Self { native, identity, recipe, encoding_work })
    }
}
impl<A: Int4ExtractionBatchAdmission> BatchProcessor for Int4ExtractionJobProcessor<'_, '_, '_, A> {
    // Option<()> rejects every non-null task_args during JobRunner::freeze,
    // before create/resume opens storage. No new runner hook is necessary.
    type Args = ();
    type Prepared = PreparedInt4BatchExtraction;
    type Output = GuardedOutput<Int4ExtractRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        struct Uncontrolled;
        impl DecodeStepControl for Uncontrolled {
            fn checkpoint(&mut self, _: usize)
                -> Option<crate::native_engine::decode::DecodeCancellationKind> { None }
        }
        self.prepare_with_control(document, &mut Uncontrolled)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_invocation_encoding_work(shared_document(document)?,
            &mut self.encoding_work, self.recipe.encoding_work_limits, control)
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
        self.recipe.checked_selected_rows()?;
        check_job_limits(&self.recipe, limits)
    }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        durable_plan_work(prepared, &self.recipe)
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.extraction_plan().max_result_bytes() }
}
fn durable_plan_work(prepared: &PreparedInt4BatchExtraction, recipe: &Int4ExtractionJobRecipe)
    -> Result<JobWork, BatchItemFailure> {
    let selected_rows = recipe.checked_selected_rows().map_err(BatchItemFailure::fatal)?;
    let encoding = recipe.input_encoding.as_ref().unwrap_or(&UntrustedEncoding::ByteFallback);
    // Public processor traits can receive a foreign prepared plan. Strategy,
    // exact row ceiling and encoder policy must match before durable accounting.
    if prepared.extraction_plan().selected_rows() != selected_rows
        || prepared.extraction_plan().execution_version() != recipe.execution
        || prepared.source().document_encoding() != encoding {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    // Copy the already group-aware quote, including the complete selected-row
    // ceiling and unchanged mask allowance. Actual row use never refunds work.
    Ok(JobWork::from_int4(prepared.planned_work(), recipe.native.max_visits_per_item))
}
fn shared_document(document: BatchDocument<()>) -> Result<BatchDocument<ExtractionBatchArgs>, BatchItemFailure> {
    if document.task_args.is_some() { return Err(BatchItemFailure::fatal(BatchCode::InvalidEnvelope)); }
    Ok(BatchDocument { id: document.id, text: document.text, task_args: None })
}

#[cfg(test)]
mod tests;
