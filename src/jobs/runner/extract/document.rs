//! Resumable complete-document extraction over the sealed native map adapter.
//! Journal attempts price WHOLE documents, never partially committed chunks.
use super::*;
use crate::{
    batch::extract::quantized::long::{Int8ExtractionMapLimits,
        INT8_EXTRACTION_MAP_EXECUTION,
        object::{Int8ObjectExtractionMapError, INT8_OBJECT_MAP_EXECUTION, ROOT_OBJECT_MAP_PROMPT},
        stream::{DocumentCompiler, Int8ExtractionDocumentRun, ExtractionDocumentArgs,
            ExtractionDocumentBatchAdmission, ExtractionDocumentBatchConfig,
            NativeInt8DocumentExtractionBatch, PreparedExtractionDocument}},
    jobs::JobLimits,
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE},
        extract::object_reconcile::{RootObjectError, RootObjectLimits,
            ROOT_OBJECT_DERIVATION, ROOT_OBJECT_RECONCILIATION},
        source_planning::quantized::long::Int8SourceMapLimits},
    validation::grounded_fields::GroundingBudget,
};

/// The original schema remains in `request`. Its derived partial schema and
/// task-owned prompt are deterministic under these versions and the exact
/// compiler bounds; no public raw content hash becomes resume authority.
#[derive(Serialize)]
struct RootObjectRecipe {
    policy: &'static str,
    schema_derivation: &'static str,
    prompt_version: &'static str,
    limits: RootObjectLimits,
}

/// Private settings projected into a keyed manifest commitment, never a public
/// executable recipe or a wire constructor. Exact schema bytes remain strings.
#[derive(Serialize)]
pub struct Int8DocumentExtractionJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    reservation_policy: &'static str,
    extraction: Int8ExtractionJobRecipe,
    request: ExtractionBatchArgs,
    chunks: ChunkLimits,
    reduction: ExecutionLimits,
    per_document_work: Int8Work,
    mask_visits_per_chunk: u64,
    max_mask_visits_per_document: u64,
    verification: GroundingBudget,
    max_verification: GroundingBudget,
    /// Absent preserves the original independent-map recipe bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    root_object_reconciliation: Option<RootObjectRecipe>,
}

/// Model-free, immutable factory. The compiler, frozen recipe and native
/// adapter are built from these SAME settings. No externally supplied compiler
/// can omit its tokenizer/template/source-limit settings from the commitment.
pub struct Int8DocumentExtractionJobPlanner {
    compiler: DocumentCompiler,
    identity: ExecutionIdentity,
    config: ExtractionDocumentBatchConfig,
    recipe: Int8DocumentExtractionJobRecipe,
}
impl Int8DocumentExtractionJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn pinned(controls: &TemplateControlIds, eos: u32, identity: ExecutionIdentity,
        ceiling: TaskBudget, compiler_limits: CompileLimits, source_limits: SourceRuntimeLimits,
        config: ExtractionDocumentBatchConfig) -> Result<Self, BatchFault> {
        config.validate()?;
        let b = config.request.budget;
        if b.max_input_tokens > ceiling.max_input_tokens || b.max_output_tokens > ceiling.max_output_tokens
            || b.max_output_bytes > ceiling.max_output_bytes || b.max_grammar_states > ceiling.max_grammar_states
            || b.max_kv_bytes > ceiling.max_kv_bytes || config.request.schema.len() > compiler_limits.max_schema_bytes {
            return Err(BatchCode::InvalidLimits.into());
        }
        // Exhaustive destructuring forces new limit axes to be considered for
        // the frozen contract, rather than silently absent on a later resume.
        let ExtractionDocumentBatchConfig { request, limits, max_model_work, max_mask_visits, max_verification } = config.clone();
        let Int8ExtractionMapLimits { mapping, verification } = limits;
        let Int8SourceMapLimits { chunks, reduction, max_model_work: per_document_work,
            mask_limits, mask_visits_per_chunk, max_mask_visits: document_masks } = mapping;
        let native = Int8ExtractionBatchLimits { max_model_work, masks: ExtractionMaskBudget {
            per_mask: mask_limits, max_visits_per_item: document_masks, max_visits_per_run: max_mask_visits } };
        let Int8ExtractionJobPlanner { compiler, identity, recipe: extraction, .. } =
            Int8ExtractionJobPlanner::pinned(controls, eos, identity, ceiling,
                compiler_limits, source_limits, None, native)?;
        let recipe = Int8DocumentExtractionJobRecipe { version: 1, dependency_scope: "item-local-complete-document",
            execution: INT8_EXTRACTION_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            reservation_policy: "whole-document-ceiling-per-durable-attempt-v1", extraction, request,
            chunks, reduction, per_document_work, mask_visits_per_chunk,
            max_mask_visits_per_document: document_masks, verification, max_verification,
            root_object_reconciliation: None };
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { compiler: DocumentCompiler::Map(compiler), identity, config, recipe })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.extraction.task_ceiling }
    pub fn document_limits(&self) -> Int8ExtractionMapLimits { self.config.limits }
    pub fn document_budget(&self) -> TaskBudget { self.config.request.budget }
    pub fn job_recipe(&self) -> &Int8DocumentExtractionJobRecipe { &self.recipe }
    pub fn request(&self) -> &ExtractionBatchArgs { &self.config.request }
    pub fn compiler_limits(&self) -> CompileLimits {
        let CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
            max_states, max_transitions, max_mask_bytes } = &self.recipe.extraction.compiler;
        CompileLimits { max_schema_bytes: *max_schema_bytes, max_string_bytes: *max_string_bytes,
            max_array_items: *max_array_items, max_output_bytes: *max_output_bytes,
            max_states: *max_states, max_transitions: *max_transitions, max_mask_bytes: *max_mask_bytes }
    }
    pub fn root_object_reconciliation(&self) -> Option<RootObjectLimits> {
        self.compiler.root_object_reconciliation()
    }

    /// Explicitly reconcile all partial chunk objects into one complete result
    /// validated against the exact ORIGINAL schema. This changes task semantics,
    /// the real compiler's template identity and the authenticated private
    /// recipe before any storage recovery. Only the root required set relaxes;
    /// nested values stay atomic and conflicts or missing fields never commit.
    /// The existing invocation tokenizer ledger travels with the compiler.
    /// Repeating the identical policy is a no-op; a different installed policy
    /// is refused instead of silently changing a resumable job.
    pub fn with_root_object_reconciliation(mut self, reconciliation: RootObjectLimits)
        -> Result<Self, BatchFault> {
        self.check_document_encoding()?;
        if self.root_object_reconciliation() == Some(reconciliation) { return Ok(self); }
        self.compiler = self.compiler.with_root_object_reconciliation(
            &self.config.request, self.config.limits, reconciliation).map_err(object_failure)?;
        self.identity = self.compiler.base_planner().base_execution_identity().clone();
        self.recipe.execution = INT8_OBJECT_MAP_EXECUTION;
        self.recipe.extraction.prompt_version = ROOT_OBJECT_MAP_PROMPT;
        self.recipe.root_object_reconciliation = Some(RootObjectRecipe {
            policy: ROOT_OBJECT_RECONCILIATION, schema_derivation: ROOT_OBJECT_DERIVATION,
            prompt_version: ROOT_OBJECT_MAP_PROMPT, limits: reconciliation,
        });
        self.check_document_encoding()?;
        bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }

    /// Configure one bounded tokenizer allowance for this invocation before
    /// authenticating storage. All document probes, chunk compilation and
    /// reconstruction share the same compiler-owned ledger. Resume recreates
    /// the same authenticated policy/caps; committed rows are skipped without
    /// tokenization, while durable native work and attempts remain spent.
    /// Reapplying identical settings preserves work already consumed. Once
    /// compact input is selected, changing or removing its policy is refused.
    pub fn with_document_encoding(mut self, policy: UntrustedEncoding,
        total: Option<SafeBpeWorkLimits>) -> Result<Self, BatchFault> {
        self.check_document_encoding()?;
        encoding_workspace(&policy, total)?;
        if self.document_encoding() == &policy && self.encoding_work_limits() == total {
            return Ok(self);
        }
        if self.document_encoding() != &UntrustedEncoding::ByteFallback {
            return Err(BatchCode::InvalidExecution.into());
        }
        let total = total.ok_or(BatchCode::InvalidLimits)?;
        self.compiler = self.compiler.with_document_encoding(policy.clone(), Some(total))?;
        self.identity = self.compiler.base_planner().base_execution_identity().clone();
        self.recipe.extraction.input_encoding = Some(policy);
        self.recipe.extraction.encoding_work_limits = Some(total);
        self.check_document_encoding()?;
        bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(self)
    }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    pub fn encoding_work_limits(&self) -> Option<SafeBpeWorkLimits> {
        self.compiler.base_planner().map_document_encoding_limits()
    }
    pub fn encoding_work(&self) -> Result<Option<SafeBpeWork>, BatchFault> {
        self.compiler.base_planner().map_document_encoding_work()
    }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        self.check_document_encoding()?;
        self.compiler.encoding_workspace_bytes()
    }
    fn check_document_encoding(&self) -> Result<(), BatchFault> {
        let recipe = &self.recipe.extraction;
        let expected = recipe.input_encoding.as_ref().unwrap_or(&UntrustedEncoding::ByteFallback);
        if matches!(recipe.input_encoding.as_ref(), Some(UntrustedEncoding::ByteFallback))
            || self.document_encoding() != expected || self.encoding_work_limits() != recipe.encoding_work_limits
            || &self.identity != self.compiler.base_planner().base_execution_identity() {
            return Err(BatchCode::Admission.into());
        }
        match (&self.recipe.root_object_reconciliation, self.root_object_reconciliation()) {
            (None, None) if self.recipe.execution == INT8_EXTRACTION_MAP_EXECUTION
                && recipe.prompt_version == EXTRACTION_BATCH_PROMPT => {}
            (Some(object), Some(limits)) if object.limits == limits
                && object.policy == ROOT_OBJECT_RECONCILIATION
                && object.schema_derivation == ROOT_OBJECT_DERIVATION
                && object.prompt_version == ROOT_OBJECT_MAP_PROMPT
                && self.recipe.execution == INT8_OBJECT_MAP_EXECUTION
                && recipe.prompt_version == ROOT_OBJECT_MAP_PROMPT => {}
            _ => return Err(BatchCode::Admission.into()),
        }
        self.compiler.encoding_workspace_bytes()?;
        Ok(())
    }
    pub fn document_work(&self) -> JobWork {
        JobWork { model: self.recipe.per_document_work,
            mask_node_visits: self.recipe.max_mask_visits_per_document }
    }
    /// A process restart must not replenish coordinate-verification authority.
    /// The authenticated journal already persists a nonrefundable attempt count
    /// BEFORE execution. Require its entire lifetime to fit every verification
    /// axis. This conservative proof needs no journal-format migration and
    /// includes failed, cancelled and acknowledgement-uncertain attempts.
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.check_document_encoding()?;
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let maximum = JobWork { model: self.config.max_model_work, mask_node_visits: self.config.max_mask_visits };
        if !self.document_work().fits(limits.max_work) || !limits.max_work.fits(maximum)
            || self.recipe.reduction.max_result_bytes > limits.max_result_bytes
            || !verification_covers_attempts(self.recipe.verification, self.recipe.max_verification, limits.max_attempts) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
}
fn object_failure(error: Int8ObjectExtractionMapError) -> BatchFault {
    if let Some(cause) = error.cancellation() { return BatchFault::cancelled(cause); }
    match error {
        Int8ObjectExtractionMapError::Admission => BatchCode::Admission.into(),
        Int8ObjectExtractionMapError::Map(_) => BatchCode::Planning.into(),
        Int8ObjectExtractionMapError::Reconciliation(error) => match error {
            RootObjectError::InvalidLimits | RootObjectError::Limit => BatchCode::InvalidLimits.into(),
            RootObjectError::Schema => BatchCode::Planning.into(),
            RootObjectError::Allocation => BatchCode::Allocation.into(),
            RootObjectError::Serialization => BatchCode::Serialization.into(),
            _ => BatchCode::InvalidExecution.into(),
        },
    }
}

fn verification_covers_attempts(per: GroundingBudget, max: GroundingBudget, attempts: u64) -> bool {
    let Ok(count) = usize::try_from(attempts) else { return false; };
    count != 0 && per.max_fields.checked_mul(count).is_some_and(|n| n <= max.max_fields)
        && per.max_matches.checked_mul(count).is_some_and(|n| n <= max.max_matches)
        && per.max_scan_steps.checked_mul(attempts).is_some_and(|n| n <= max.max_scan_steps)
}

/// Borrowing the immutable factory keeps the native adapter non-self-referential.
/// The caller retains its preparation charge until this processor is dropped.
/// Only real native extraction is available; no result-injection constructor.
pub struct Int8DocumentExtractionJobProcessor<'p, 'e, 'w, 'v, A: ExtractionDocumentBatchAdmission> {
    native: NativeInt8DocumentExtractionBatch<'p, 'e, 'w, 'v, A>,
    planner: &'p Int8DocumentExtractionJobPlanner,
}
impl<'p, 'e, 'w, 'v, A: ExtractionDocumentBatchAdmission>
    Int8DocumentExtractionJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int8DocumentExtractionJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        Self::scheduled(planner, engine, vocabulary, admission, None)
    }
    pub(crate) fn scheduled(planner: &'p Int8DocumentExtractionJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A, prefill: Option<Int8PrefillLimits>) -> Result<Self, BatchFault> {
        planner.check_document_encoding()?;
        let config = planner.config.clone();
        let native = NativeInt8DocumentExtractionBatch::new_scheduled(
            &planner.compiler, config, engine, vocabulary, admission, prefill)?;
        Ok(Self { native, planner })
    }
}
impl<A: ExtractionDocumentBatchAdmission> BatchProcessor
    for Int8DocumentExtractionJobProcessor<'_, '_, '_, '_, A> {
    type Args = ExtractionDocumentArgs;
    type Prepared = PreparedExtractionDocument;
    type Output = GuardedOutput<Int8ExtractionDocumentRun, A::Guard>;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare(document)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.native.prepare_with_control(document, control)
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        // A conservative whole-document ceiling, not the actual work of just
        // one chunk. durable_work reports exactly the same two visible axes.
        let model = self.planner.document_work().model;
        BatchWork { forward_positions: model.forward_positions, projected_logits: model.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> { self.native.execute(prepared, control) }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.native.execute_with_context(prepared, context, control)
    }
}
impl<A: ExtractionDocumentBatchAdmission> DurableBatchProcessor
    for Int8DocumentExtractionJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int8DocumentExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, _: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        if self.native.is_poisoned() { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(self.planner.document_work())
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.recipe.reduction.max_result_bytes as u64 }
}

#[cfg(test)] mod tests;
