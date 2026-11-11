//! Durable exact-schema extraction over complete grouped-INT4 documents.
//! One attempt owns every ordered chunk and the lossless reduction. Its debit
//! comes from the real prepared native map, never an INT8 or per-document proxy.
use super::*;
use crate::{
    batch::extract::grouped_int4::long::{Int4ExtractionMapError, Int4ExtractionMapLimits,
        Int4ExtractionMappingLimits, INT4_EXTRACTION_MAP_EXECUTION, INT4_EXTRACTION_MAP_SELECTED_EXECUTION,
        object::{Int4ObjectExtractionMapError,
            INT4_OBJECT_MAP_EXECUTION, INT4_OBJECT_MAP_SELECTED_EXECUTION, ROOT_OBJECT_MAP_PROMPT},
        stream::{DocumentCompiler, DocumentPlan, DocumentPreflight, Int4ExtractionDocumentRun},
        cohort::{Int4ExtractionMapCohortBudget, Int4ExtractionMapCohortMemoryLimits,
            Int4ExtractionMapCohortRequirement}},
    native_engine::{constrained_int4::{self, sparse::Int4JsonSparseLimits}, int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryRequirement, Int4ModelWeights}},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE},
        extract::{array_union::{RootArrayUnionLimits, ROOT_ARRAY_UNION_POLICY},
            object_reconcile::{RootObjectLimits, RootObjectError,
                ROOT_OBJECT_DERIVATION, ROOT_OBJECT_RECONCILIATION}}},
    validation::grounded_fields::GroundingBudget,
};
use std::sync::Arc;
// This is only the empty record-arguments shape; no INT8 execution or work
// authority is reused by the grouped-INT4 document adapter.
pub use crate::batch::extract::quantized::long::stream::ExtractionDocumentArgs;
mod execution;
pub use execution::{Int4ExtractionDocumentAdmission, Int4ExtractionDocumentBatchAdmission,
    Int4DocumentExtractionJobProcessor};

const CHUNK_COHORT_SCHEDULE: &str = "native-document-chunk-cohorts-v1";

/// The reducer changes result semantics, so both its fixed version and every
/// bounded allowance belong in the authenticated private recipe.
#[derive(Serialize)]
struct RootArrayUnionRecipe {
    policy: &'static str,
    limits: RootArrayUnionLimits,
}

/// Original schema bytes remain in `extraction.defaults`. The independently
/// derived partial schema and the task-owned prompt are deterministic under
/// these exact versions and compiler bounds; changing any value changes the
/// private job commitment before recovery or native execution.
#[derive(Serialize)]
struct RootObjectRecipe {
    policy: &'static str,
    schema_derivation: &'static str,
    prompt_version: &'static str,
    limits: RootObjectLimits,
}

/// The complete legal set is projected or the step refuses. The cap and both
/// native/task contracts are private resume authority, not tuning metadata.
#[derive(Serialize)]
struct SelectedRowsRecipe {
    execution: &'static str,
    native_execution: &'static str,
    max_rows_per_step: usize,
}

/// Private immutable settings projected into JobRunner's keyed commitment.
/// Exact schema bytes stay in the sealed extraction defaults, including number
/// lexemes and whitespace; there is no public content digest or wire factory.
#[derive(Serialize)]
pub struct Int4DocumentExtractionJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    reservation_policy: &'static str,
    extraction: Int4ExtractionJobRecipe,
    chunks: ChunkLimits,
    reduction: ExecutionLimits,
    per_document_model_work: Int4Work,
    max_trie_node_visits: usize,
    checkpoint_interval_nodes: usize,
    mask_visits_per_chunk: u64,
    max_mask_visits_per_document: u64,
    verification: GroundingBudget,
    max_lifetime_verification: GroundingBudget,
    max_attempts: u64,
    /// Absent preserves the exact original serial recipe commitment.
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_cohorts: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_array_union: Option<RootArrayUnionRecipe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_rows: Option<SelectedRowsRecipe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_object_reconciliation: Option<RootObjectRecipe>,
}
impl Int4DocumentExtractionJobRecipe {
    fn new(extraction: Int4ExtractionJobRecipe, document: Int4ExtractionMapLimits,
        max_lifetime_verification: GroundingBudget, max_attempts: u64) -> Result<Self, BatchFault> {
        // Exhaustive projections prevent newly introduced map/mask settings
        // from silently escaping the private resume commitment.
        let Int4ExtractionMapLimits { mapping, verification, root_array_union } = document;
        let Int4ExtractionMappingLimits { chunks, reduction, max_model_work, mask_limits,
            mask_visits_per_chunk, max_mask_visits } = mapping;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = mask_limits;
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-document",
            execution: INT4_EXTRACTION_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            reservation_policy: "exact-all-chunk-native-work-per-durable-attempt-v1",
            extraction, chunks, reduction, per_document_model_work: max_model_work,
            max_trie_node_visits, checkpoint_interval_nodes, mask_visits_per_chunk,
            max_mask_visits_per_document: max_mask_visits, verification,
            max_lifetime_verification, max_attempts, chunk_cohorts: None, selected_rows: None,
            root_object_reconciliation: None,
            root_array_union: root_array_union.map(|limits| RootArrayUnionRecipe {
                policy: ROOT_ARRAY_UNION_POLICY, limits }) };
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

/// Model-bound immutable factory. This owns the same actual pinned compiler,
/// fixed raw schema and source-verification settings used for every chunk.
pub struct Int4DocumentExtractionJobPlanner {
    compiler: DocumentCompiler,
    identity: ExecutionIdentity,
    document: Int4ExtractionMapLimits,
    limits: JobLimits,
    recipe: Int4DocumentExtractionJobRecipe,
    owner: Arc<()>,
}
impl Int4DocumentExtractionJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn for_model(model: &Int4ModelIdentity, ceiling: TaskBudget, compiler: CompileLimits,
        source: SourceRuntimeLimits, request: ExtractionBatchArgs, document: Int4ExtractionMapLimits,
        job: JobLimits, max_lifetime_verification: GroundingBudget) -> Result<Self, BatchFault> {
        validate_limits(model.group_width(), ceiling, request.budget, document, job)?;
        let native = native_limits(document, job);
        let extraction = Int4ExtractionJobPlanner::for_model(model, ceiling, compiler, source, request, native)?;
        Self::configured(extraction, document, job, max_lifetime_verification)
    }
    /// Execute explicitly partial root objects and reconcile one complete
    /// original-schema result per durable item. Only the root required set is
    /// relaxed for native chunks; nested values remain atomic. The original
    /// schema, fixed prompt/derivation, reducer limits and every verification
    /// allowance are authenticated before recovery. No ordinary extraction
    /// job acquires these semantics implicitly.
    #[allow(clippy::too_many_arguments)]
    pub fn for_object_model(model: &Int4ModelIdentity, ceiling: TaskBudget, compiler: CompileLimits,
        source: SourceRuntimeLimits, request: ExtractionBatchArgs, document: Int4ExtractionMapLimits,
        job: JobLimits, max_lifetime_verification: GroundingBudget, reconciliation: RootObjectLimits)
        -> Result<Self, BatchFault> {
        validate_limits(model.group_width(), ceiling, request.budget, document, job)?;
        let extraction = Int4ExtractionJobRecipe::new(model.group_width(), ceiling, compiler,
            source, native_limits(document, job), request)?;
        // A full required object may exceed one chunk's output capacity. Do
        // not run the ordinary shared-default compiler on that schema first.
        let compiler = Int4ExtractionBatchPlanner::for_model(model, ceiling, compiler, source, None)?;
        Self::configured_object(compiler, extraction, document, job, max_lifetime_verification, reconciliation)
    }
    fn configured_object(compiler: Int4ExtractionBatchPlanner, mut extraction: Int4ExtractionJobRecipe,
        document: Int4ExtractionMapLimits, limits: JobLimits, max_lifetime_verification: GroundingBudget,
        reconciliation: RootObjectLimits) -> Result<Self, BatchFault> {
        validate_limits(extraction.group_width, extraction.task_ceiling, extraction.defaults.budget, document, limits)?;
        verify_lifetime(document.verification, max_lifetime_verification, limits.max_attempts)?;
        let compiler = DocumentCompiler::Map(compiler).with_root_object_reconciliation(
            &extraction.defaults, document, reconciliation).map_err(|e| object_failure(e).fault)?;
        let identity = compiler.base_planner().base_execution_identity().clone();
        extraction.prompt_version = ROOT_OBJECT_MAP_PROMPT;
        let mut recipe = Int4DocumentExtractionJobRecipe::new(extraction, document,
            max_lifetime_verification, limits.max_attempts)?;
        recipe.execution = INT4_OBJECT_MAP_EXECUTION;
        recipe.root_object_reconciliation = Some(RootObjectRecipe {
            policy: ROOT_OBJECT_RECONCILIATION, schema_derivation: ROOT_OBJECT_DERIVATION,
            prompt_version: ROOT_OBJECT_MAP_PROMPT, limits: reconciliation,
        });
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(Self { compiler, identity, document, limits, recipe, owner: Arc::new(()) })
    }
    // Fixtures use the real pinned compiler without exposing an alternate
    // production model constructor or any fabricated native execution result.
    fn configured(extraction: Int4ExtractionJobPlanner, document: Int4ExtractionMapLimits,
        limits: JobLimits, max_lifetime_verification: GroundingBudget) -> Result<Self, BatchFault> {
        let width = extraction.recipe.group_width;
        validate_limits(width, extraction.task_ceiling(), extraction.recipe.defaults.budget, document, limits)?;
        verify_lifetime(document.verification, max_lifetime_verification, limits.max_attempts)?;
        if let Some(union) = document.root_array_union {
            let CompilerRecipe { max_schema_bytes, max_string_bytes, max_array_items, max_output_bytes,
                max_states, max_transitions, max_mask_bytes } = &extraction.recipe.compiler;
            let declaration = crate::grammar::runtime::JsonValidationSchema::compile(
                &extraction.recipe.defaults.schema, CompileLimits {
                    max_schema_bytes: *max_schema_bytes, max_string_bytes: *max_string_bytes,
                    max_array_items: *max_array_items, max_output_bytes: *max_output_bytes,
                    max_states: *max_states, max_transitions: *max_transitions, max_mask_bytes: *max_mask_bytes,
                }).map_err(|_| BatchCode::Planning)?;
            union.validate_schema(declaration.root()).map_err(|_| BatchCode::Planning)?;
        }
        // The compiler's immutable extraction policy and the document recipe
        // must agree even for internal planning fixtures.
        let expected = native_limits(document, limits);
        let actual = extraction.native;
        if actual.max_model_work != expected.max_model_work
            || actual.masks.per_mask.max_trie_node_visits != expected.masks.per_mask.max_trie_node_visits
            || actual.masks.per_mask.checkpoint_interval_nodes != expected.masks.per_mask.checkpoint_interval_nodes
            || actual.masks.max_visits_per_item != expected.masks.max_visits_per_item
            || actual.masks.max_visits_per_run != expected.masks.max_visits_per_run {
            return Err(BatchCode::Admission.into());
        }
        let Int4ExtractionJobPlanner { compiler, identity, recipe: extraction, .. } = extraction;
        let recipe = Int4DocumentExtractionJobRecipe::new(extraction, document,
            max_lifetime_verification, limits.max_attempts)?;
        Ok(Self { compiler: DocumentCompiler::Map(compiler), identity, document, limits, recipe, owner: Arc::new(()) })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.extraction.task_ceiling }
    pub fn document_budget(&self) -> TaskBudget { self.request().budget }
    pub fn document_limits(&self) -> Int4ExtractionMapLimits { self.document }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn request(&self) -> &ExtractionBatchArgs { &self.recipe.extraction.defaults }
    pub fn group_width(&self) -> usize { self.recipe.extraction.group_width }
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
    /// Seal the complete document's input policy before population recovery.
    /// Schema encoding, chunk fit probes and final chunk compilation share
    /// one allowance per complete-document preparation. Reconstruction starts
    /// a separately bounded preparation under the same authenticated policy;
    /// tokenizer work is not a persisted lifetime debit. The private recipe
    /// authenticates every limit, and a previously prepared byte-fallback
    /// attempt cannot acquire compact semantics. Root-object reconciliation
    /// remains byte-fallback only.
    pub fn with_document_encoding(mut self, policy: UntrustedEncoding) -> Result<Self, BatchFault> {
        self.check_document_encoding()?;
        if self.document_encoding() == &policy { return Ok(self); }
        self.compiler = self.compiler.with_document_encoding(policy.clone())?;
        self.recipe.extraction.input_encoding = match policy {
            UntrustedEncoding::ByteFallback => None,
            compact @ UntrustedEncoding::SafeBpeV1(_) => Some(compact),
        };
        self.check_document_encoding()?;
        bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        self.owner = Arc::new(());
        Ok(self)
    }
    pub fn document_encoding(&self) -> &UntrustedEncoding { self.compiler.document_encoding() }
    /// One transient encoder is live during serial preparation/reconstruction,
    /// independently of the number of retained chunks or native cohort rows.
    pub fn encoding_workspace_bytes(&self) -> Result<u64, BatchFault> {
        self.check_document_encoding()?;
        self.compiler.encoding_workspace_bytes()
    }
    fn check_document_encoding(&self) -> Result<(), BatchFault> {
        let expected = match &self.recipe.extraction.input_encoding {
            None => &UntrustedEncoding::ByteFallback,
            Some(UntrustedEncoding::ByteFallback) => return Err(BatchCode::Admission.into()),
            Some(compact @ UntrustedEncoding::SafeBpeV1(_)) => compact,
        };
        if self.document_encoding() != expected
            || (self.root_object_reconciliation().is_some() && expected != &UntrustedEncoding::ByteFallback) {
            return Err(BatchCode::Admission.into());
        }
        self.compiler.encoding_workspace_bytes()?;
        Ok(())
    }
    /// Seal the same all-legal-row strategy into every chunk and the immutable
    /// job recipe. Switching strategy or cap requires a different job; a
    /// preexisting full-head attempt cannot be reused under this factory.
    pub fn with_selected_rows(mut self, limits: Int4JsonSparseLimits) -> Result<Self, BatchFault> {
        self.compiler = self.compiler.with_selected_rows(limits)?;
        let execution = crate::tasks::extract::grouped_int4::INT4_SPARSE_EXTRACT_VERSION;
        self.recipe.execution = if self.root_object_reconciliation().is_some() {
            INT4_OBJECT_MAP_SELECTED_EXECUTION
        } else { INT4_EXTRACTION_MAP_SELECTED_EXECUTION };
        self.recipe.extraction.execution = execution;
        self.recipe.selected_rows = Some(SelectedRowsRecipe {
            execution,
            native_execution: constrained_int4::sparse::INT4_SPARSE_JSON_EXECUTION,
            max_rows_per_step: limits.max_rows_per_step,
        });
        bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        self.owner = Arc::new(());
        Ok(self)
    }
    pub fn selected_rows(&self) -> Option<Int4JsonSparseLimits> { self.compiler.base_planner().selected_rows() }
    /// Opt into native windows within each durable document attempt. The
    /// already-authenticated map_batch_chunks fixes their maximum width.
    pub fn with_chunk_cohorts(mut self) -> Result<Self, BatchFault> {
        self.cohort_requirements()?;
        if !self.uses_chunk_cohorts() {
            self.recipe.chunk_cohorts = Some(CHUNK_COHORT_SCHEDULE);
            bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
            // Prepared serial attempts cannot acquire the newly opted schedule.
            self.owner = Arc::new(());
        }
        Ok(self)
    }
    pub fn uses_chunk_cohorts(&self) -> bool { self.recipe.chunk_cohorts == Some(CHUNK_COHORT_SCHEDULE) }
    /// Static whole-window ownership for the complete job invocation. Short
    /// documents and ragged final windows never reduce the host's reservation.
    pub fn cohort_requirements(&self) -> Result<Int4ExtractionMapCohortRequirement, BatchFault> {
        let mapping = self.document.mapping;
        if mapping.reduction.map_batch_chunks > mapping.chunks.max_chunks {
            return Err(BatchCode::InvalidLimits.into());
        }
        Int4ExtractionMapCohortMemoryLimits {
            max_chunks: mapping.reduction.map_batch_chunks, context_tokens: mapping.chunks.context_tokens,
            max_output_tokens_per_chunk: u64::from(self.document_budget().max_output_tokens),
            max_result_bytes_per_chunk: self.document_budget().max_output_bytes,
        }.requirements(self.group_width()).map_err(|error| map_failure(error).fault)
    }
    pub fn max_result_bytes(&self) -> u64 { self.document.mapping.reduction.max_result_bytes as u64 }
    pub fn output_token_ceiling(&self) -> Result<u64, BatchFault> {
        self.output_tokens(self.document.mapping.chunks.max_chunks)
    }
    pub fn job_recipe(&self) -> &Int4DocumentExtractionJobRecipe { &self.recipe }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        self.check_document_encoding()?;
        if limits != self.limits || (self.root_object_reconciliation().is_some()
            && !self.uses_chunk_cohorts() && self.document.mapping.reduction.map_batch_chunks != 1) {
            return Err(BatchCode::InvalidLimits.into());
        }
        verify_lifetime(self.document.verification, self.recipe.max_lifetime_verification, limits.max_attempts)
    }
    fn output_tokens(&self, chunks: usize) -> Result<u64, BatchFault> {
        u64::try_from(chunks).ok().and_then(|n| n.checked_mul(u64::from(self.document_budget().max_output_tokens)))
            .ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    fn owns(&self, prepared: &PreparedInt4ExtractionDocument) -> Result<(), BatchItemFailure> {
        self.check_document_encoding().map_err(BatchItemFailure::fatal)?;
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
    fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, control: &mut C)
        -> Result<DocumentPlan<'s>, BatchItemFailure> {
        poll(control)?;
        self.check_document_encoding().map_err(BatchItemFailure::fatal)?;
        if source.len() > self.limits.max_input_bytes_per_item {
            return Err(BatchItemFailure::fatal(BatchCode::DocumentLimit));
        }
        self.compiler.plan(source, self.request(), self.document, control).map_err(object_failure)
    }
    fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<ExtractionDocumentArgs>, control: &mut C)
        -> Result<PreparedInt4ExtractionDocument, BatchItemFailure> {
        let plan = self.plan(&document.text, control)?;
        let expected = plan.preflight_metadata();
        let mapped = expected.mapped();
        let mut identities = Vec::new();
        identities.try_reserve_exact(mapped.chunk_count()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in plan.execution_identities() {
            poll(control)?;
            identities.push(identity.clone());
        }
        if identities.len() != mapped.chunk_count() || mapped.chunk_count() == 0
            || mapped.group_width() != self.group_width() || mapped.selected_rows() != self.selected_rows()
            || &mapped.document_encoding() != self.document_encoding()
            || expected.execution_version() != self.recipe.execution
            || plan.max_kv_bytes() != self.document_budget().max_kv_bytes
            || plan.max_result_bytes() != self.max_result_bytes() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let work = receipt_work(expected);
        if !work.fits(self.limits.max_work) { return Err(BatchItemFailure::fatal(BatchCode::WorkLimit)); }
        // Release all borrowed native plans before moving the exact original
        // source. Reconstruction uses this factory and verifies every receipt.
        drop(plan);
        poll(control)?;
        Ok(PreparedInt4ExtractionDocument { source: document.text, identities, expected, work,
            owner: Arc::clone(&self.owner) })
    }
    fn compare<C: DecodeStepControl>(&self, plan: &DocumentPlan<'_>,
        prepared: &PreparedInt4ExtractionDocument, control: &mut C) -> Result<(), BatchItemFailure> {
        self.owns(prepared)?;
        if plan.preflight_metadata() != prepared.expected || receipt_work(prepared.expected) != prepared.work
            || plan.max_kv_bytes() != self.document_budget().max_kv_bytes
            || plan.max_result_bytes() != self.max_result_bytes()
            || plan.execution_identities().len() != prepared.identities.len() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        for (actual, expected) in plan.execution_identities().zip(&prepared.identities) {
            poll(control)?;
            if actual != expected { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        }
        poll(control)
    }
}

/// Owned original input and the complete native preparation commitment.
/// Neither source text nor a partially completed chunk set is serializable.
pub struct PreparedInt4ExtractionDocument {
    source: String,
    identities: Vec<ExecutionIdentity>,
    expected: DocumentPreflight,
    work: JobWork,
    owner: Arc<()>,
}
impl PreparedInt4ExtractionDocument {
    pub fn chunk_count(&self) -> usize { self.expected.mapped().chunk_count() }
    pub fn planned_work(&self) -> JobWork { self.work }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
}
fn receipt_work(expected: DocumentPreflight) -> JobWork {
    let expected = expected.mapped();
    JobWork::from_int4(expected.planned_work(), expected.reserved_mask_visits())
}
fn native_limits(document: Int4ExtractionMapLimits, job: JobLimits) -> Int4ExtractionBatchLimits {
    Int4ExtractionBatchLimits { max_model_work: job.max_work.int4_model_work(),
        masks: ExtractionMaskBudget { per_mask: document.mapping.mask_limits,
            max_visits_per_item: document.mapping.mask_visits_per_chunk,
            max_visits_per_run: job.max_work.mask_node_visits } }
}
fn verify_lifetime(per: GroundingBudget, max: GroundingBudget, attempts: u64) -> Result<(), BatchFault> {
    // JobRunner persists the attempt BEFORE execution. Including every failed,
    // cancelled and acknowledgement-uncertain attempt prevents a restart from
    // replenishing original-coordinate verification without a journal change.
    let count = usize::try_from(attempts).map_err(|_| BatchCode::InvalidLimits)?;
    if count == 0 || per.max_fields.checked_mul(count).is_none_or(|n| n > max.max_fields)
        || per.max_matches.checked_mul(count).is_none_or(|n| n > max.max_matches)
        || per.max_scan_steps.checked_mul(attempts).is_none_or(|n| n > max.max_scan_steps) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
fn validate_limits(width: usize, ceiling: TaskBudget, request: TaskBudget,
    document: Int4ExtractionMapLimits, job: JobLimits) -> Result<(), BatchFault> {
    job.validate().map_err(|_| BatchCode::InvalidLimits)?;
    ceiling.validate().map_err(|_| BatchCode::InvalidLimits)?;
    request.validate().map_err(|_| BatchCode::InvalidLimits)?;
    let m = document.mapping; let e = m.reduction; let v = document.verification;
    document.root_array_workspace_bytes().map_err(|_| BatchCode::InvalidLimits)?;
    m.chunks.effective_token_limit().map_err(|_| BatchCode::InvalidLimits)?;
    let kv = Int4MemoryRequirement::for_context(m.chunks.context_tokens, width)
        .map_err(|_| BatchCode::InvalidLimits)?.kv_bytes;
    if ceiling.max_kv_bytes != kv || request.max_kv_bytes != kv
        || m.chunks.max_input_bytes == 0 || !(1..=256).contains(&m.chunks.max_chunks)
        || !JobWork::from_int4(m.max_model_work, m.max_mask_visits).fits(job.max_work)
        || !(1..=1024).contains(&e.map_batch_chunks) || !(2..=1024).contains(&e.reduce_fan_in)
        || e.max_reduction_levels > 64 || !(1..=1_000_000).contains(&e.max_task_calls)
        || [e.max_value_bytes, e.max_live_value_bytes, e.max_total_value_bytes, e.max_result_bytes]
            .iter().any(|n| !(1..=64 * 1024 * 1024).contains(n))
        || e.max_value_bytes > e.max_live_value_bytes || e.max_result_bytes > job.max_result_bytes
        || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
        || v.max_scan_steps == 0 {
        return Err(BatchCode::InvalidLimits.into());
    }
    Int4ExtractionBatchLimits { max_model_work: m.max_model_work,
        masks: ExtractionMaskBudget { per_mask: m.mask_limits,
            max_visits_per_item: m.mask_visits_per_chunk, max_visits_per_run: m.max_mask_visits } }.validate()
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()),
    }
}
fn map_failure(error: Int4ExtractionMapError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(match error {
        Int4ExtractionMapError::InvalidLimits => BatchCode::InvalidLimits,
        Int4ExtractionMapError::EmptySource => BatchCode::DocumentLimit,
        Int4ExtractionMapError::WorkLimit => BatchCode::WorkLimit,
        Int4ExtractionMapError::Admission => BatchCode::Admission,
        Int4ExtractionMapError::Allocation => BatchCode::Allocation,
        Int4ExtractionMapError::Chunk(_) => BatchCode::Planning,
        Int4ExtractionMapError::Extraction(_) | Int4ExtractionMapError::Reduction(_)
            | Int4ExtractionMapError::ArrayUnion(_) => BatchCode::InvalidExecution,
    })
}

fn object_failure(error: Int4ObjectExtractionMapError) -> BatchItemFailure {
    if let Some(cause) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    match error {
        Int4ObjectExtractionMapError::Map(error) => map_failure(error),
        Int4ObjectExtractionMapError::Admission => BatchItemFailure::fatal(BatchCode::Admission),
        Int4ObjectExtractionMapError::Reconciliation(error) => BatchItemFailure::fatal(match error {
            RootObjectError::InvalidLimits => BatchCode::InvalidLimits,
            RootObjectError::Schema => BatchCode::Planning,
            RootObjectError::Allocation => BatchCode::Allocation,
            _ => BatchCode::InvalidExecution,
        }),
    }
}

#[cfg(test)] mod tests;
