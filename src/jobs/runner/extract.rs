//! Durable user-schema extraction using the existing native batch executor.
//! Exact schemas remain strings; source binding happens against each original
//! document. The frozen recipe and the native compiler own the SAME settings.
use super::DurableBatchProcessor;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchItemFailure, BatchProcessor,
        BatchRequestContext, BatchWork,
        extract::{ExtractionBatchArgs, ExtractionMaskBudget, GuardedOutput, EXTRACTION_BATCH_PROMPT,
            quantized::{Int8ExtractionBatchAdmission, Int8ExtractionBatchLimits,
                Int8ExtractionBatchPlanner, NativeInt8ExtractionBatch, PreparedInt8BatchExtraction}}},
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, mask::MaskWorkLimits,
        runtime::{SourceRuntimeLimits, JSON_RUNTIME_VERSION, SOURCE_JSON_RUNTIME_VERSION}},
    jobs::{JobWork, manifest::bounded_json},
    native_engine::{decode::DecodeStepControl, strict_int8::{Int8Work, StrictInt8Engine, prefill::Int8PrefillLimits}},
    tasks::{extract::{ExtractionVocabulary, quantized::{Int8ExtractRun, INT8_EXTRACT_VERSION}}, ir::TaskBudget},
    tokenizer::{specials::TemplateControlIds,
        untrusted::{SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding}},
};
use serde::Serialize;

/// Whole-document jobs are a distinct frozen recipe, not an implicit change
/// to the existing single-context extraction job contract.
pub mod document;
/// Fixed shared-schema jobs using the distinct grouped-INT4 execution profile.
pub mod grouped_int4;

/// Only a keyed commitment to this private projection enters the journal.
/// Never Deserialize/Debug or turn it into an alternate executable recipe.
#[derive(Serialize)]
pub struct Int8ExtractionJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    execution: &'static str,
    prompt_version: &'static str,
    json_runtime: &'static str,
    source_runtime: &'static str,
    task_ceiling: TaskBudget,
    compiler: CompilerRecipe,
    source: SourceRuntimeLimits,
    native: NativeRecipe,
    defaults: Option<ExtractionBatchArgs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<UntrustedEncoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_work_limits: Option<SafeBpeWorkLimits>,
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
impl Int8ExtractionJobRecipe {
    fn new(task_ceiling: TaskBudget, compiler: CompileLimits, source: SourceRuntimeLimits,
        native: Int8ExtractionBatchLimits, defaults: Option<ExtractionBatchArgs>) -> Self {
        // Future limit fields must be deliberately added to the frozen recipe.
        let CompileLimits { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
            max_states, max_transitions, max_mask_bytes } = compiler;
        let Int8ExtractionBatchLimits { max_model_work, masks } = native;
        let ExtractionMaskBudget { per_mask, max_visits_per_item, max_visits_per_run } = masks;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = per_mask;
        Self { version: 1, dependency_scope: "item-local", execution: INT8_EXTRACT_VERSION,
            prompt_version: EXTRACTION_BATCH_PROMPT, json_runtime: JSON_RUNTIME_VERSION,
            source_runtime: SOURCE_JSON_RUNTIME_VERSION, task_ceiling,
            compiler: CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items,
                max_output_bytes, max_states, max_transitions, max_mask_bytes }, source,
            native: NativeRecipe { max_model_work, max_trie_node_visits, checkpoint_interval_nodes,
                max_visits_per_item, max_visits_per_run }, defaults,
            input_encoding: None, encoding_work_limits: None }
    }
}

/// Model-free factory, consumed at the native handoff without recompilation.
/// Its identity contains the ACTUAL compiler's template/tokenizer binding.
/// Per-record schemas and source bytes are separately frozen by JobRunner.
pub struct Int8ExtractionJobPlanner {
    compiler: Int8ExtractionBatchPlanner,
    identity: ExecutionIdentity,
    recipe: Int8ExtractionJobRecipe,
    native: Int8ExtractionBatchLimits,
    encoding_work: SafeBpeWork,
}
impl Int8ExtractionJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity,
        ceiling: TaskBudget, compiler_limits: CompileLimits, source_limits: SourceRuntimeLimits,
        defaults: Option<ExtractionBatchArgs>, native: Int8ExtractionBatchLimits) -> Result<Self, BatchFault> {
        native.validate()?;
        ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
        // Bound ALL settings before cloning defaults into the native compiler.
        // Leave room for the runner's protocol/input-profile wrapper.
        let recipe = Int8ExtractionJobRecipe::new(ceiling, compiler_limits, source_limits, native, defaults);
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        if let Some(defaults) = &recipe.defaults {
            let b = defaults.budget;
            b.validate().map_err(|_| BatchCode::InvalidLimits)?;
            if b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
                || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
                || b.max_kv_bytes > ceiling.max_kv_bytes {
                return Err(BatchCode::InvalidLimits.into());
            }
        }
        let compiler = Int8ExtractionBatchPlanner::pinned(controls, eos, identity, ceiling,
            compiler_limits, source_limits, recipe.defaults.clone())?;
        let identity = compiler.base_execution_identity().clone();
        Ok(Self { compiler, identity, recipe, native, encoding_work: SafeBpeWork::default() })
    }
    /// Select bounded compact schema/source input before job storage opens.
    /// The fixed per-record policy and all invocation ceilings authenticate on
    /// resume. One spent ledger moves with this factory into the real native
    /// processor; rejected/cancelled preparation does not refund it.
    pub fn with_document_encoding(mut self, policy: UntrustedEncoding,
        total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        self.check_document_encoding()?;
        encoding_workspace(&policy, total)?;
        if self.document_encoding() == &policy && self.encoding_work_limits() == total {
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
        let expected = self.recipe.input_encoding.as_ref().unwrap_or(&UntrustedEncoding::ByteFallback);
        if matches!(self.recipe.input_encoding.as_ref(), Some(UntrustedEncoding::ByteFallback))
            || self.document_encoding() != expected {
            return Err(BatchCode::Admission.into());
        }
        encoding_workspace(expected, self.recipe.encoding_work_limits)?;
        Ok(())
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.task_ceiling }
    pub fn native_limits(&self) -> Int8ExtractionBatchLimits { self.native }
}

fn encoding_workspace(policy: &UntrustedEncoding, total: Option<SafeBpeWorkLimits>)
    -> Result<u64, BatchFault> {
    match (policy, total) {
        (UntrustedEncoding::ByteFallback, None) => Ok(0),
        (UntrustedEncoding::SafeBpeV1(call), Some(total)) => {
            total.validate().map_err(|_| BatchCode::InvalidLimits)?;
            call.requirements().map(|quote| quote.workspace_bytes).map_err(|_| BatchCode::InvalidLimits.into())
        }
        _ => Err(BatchCode::Admission.into()),
    }
}

/// Real native execution only. The external host's guard travels inside each
/// result until JobRunner has synced its frame and committed its journal row.
pub struct Int8ExtractionJobProcessor<'e, 'weights, 'v, A: Int8ExtractionBatchAdmission> {
    native: NativeInt8ExtractionBatch<'e, 'weights, 'v, A>,
    identity: ExecutionIdentity,
    recipe: Int8ExtractionJobRecipe,
    encoding_work: SafeBpeWork,
}
impl<'e, 'weights, 'v, A: Int8ExtractionBatchAdmission> Int8ExtractionJobProcessor<'e, 'weights, 'v, A> {
    pub fn new(planner: Int8ExtractionJobPlanner, engine: &'e mut StrictInt8Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::scheduled(planner, engine, vocabulary, admission, None)
    }
    pub(crate) fn scheduled(planner: Int8ExtractionJobPlanner, engine: &'e mut StrictInt8Engine<'weights>,
        vocabulary: &'v ExtractionVocabulary, admission: A,
        prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        planner.check_document_encoding()?;
        let Int8ExtractionJobPlanner { compiler, identity, recipe, native, encoding_work } = planner;
        let native = match prefill {
            None => NativeInt8ExtractionBatch::new(compiler, engine, vocabulary, admission, native),
            Some(prefill) => NativeInt8ExtractionBatch::new_layer_major(compiler, engine, vocabulary, admission, native, prefill),
        }?;
        Ok(Self { native, identity, recipe, encoding_work })
    }
}
impl<A: Int8ExtractionBatchAdmission> BatchProcessor for Int8ExtractionJobProcessor<'_, '_, '_, A> {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt8BatchExtraction;
    type Output = GuardedOutput<Int8ExtractRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        // Compact preparation requires the original caller's cancellation and
        // deadline authority; never create a fresh control at this boundary.
        if self.recipe.input_encoding.is_some() {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_invocation_encoding_work(document, &mut self.encoding_work,
            self.recipe.encoding_work_limits, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { self.native.planned_work(prepared) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        self.native.execute(prepared, control)
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: Int8ExtractionBatchAdmission> DurableBatchProcessor for Int8ExtractionJobProcessor<'_, '_, '_, A> {
    type Recipe = Int8ExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        if prepared.source().document_encoding()
            != self.recipe.input_encoding.as_ref().unwrap_or(&UntrustedEncoding::ByteFallback) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(JobWork { model: prepared.planned_work(), mask_node_visits: self.recipe.native.max_visits_per_item })
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.extraction_plan().max_result_bytes() }
}

#[cfg(test)] mod tests;
