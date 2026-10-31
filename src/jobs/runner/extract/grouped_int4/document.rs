//! Durable exact-schema extraction over complete grouped-INT4 documents.
//! One attempt owns every ordered chunk and the lossless reduction. Its debit
//! comes from the real prepared native map, never an INT8 or per-document proxy.
use super::*;
use crate::{
    batch::extract::grouped_int4::long::{Int4ExtractionMapError, Int4ExtractionMapLimits,
        Int4ExtractionMappingLimits, Int4ExtractionMapPreflight, Int4ExtractionMapRun,
        PreparedInt4ExtractionMap, INT4_EXTRACTION_MAP_EXECUTION,
        cohort::{Int4ExtractionMapCohortBudget, Int4ExtractionMapCohortMemoryLimits,
            Int4ExtractionMapCohortRequirement}},
    native_engine::{constrained_int4, int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryRequirement, Int4ModelWeights}},
    tasks::mapreduce::{ChunkLimits, ExecutionLimits, CHUNK_PROFILE},
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
}
impl Int4DocumentExtractionJobRecipe {
    fn new(extraction: Int4ExtractionJobRecipe, document: Int4ExtractionMapLimits,
        max_lifetime_verification: GroundingBudget, max_attempts: u64) -> Result<Self, BatchFault> {
        // Exhaustive projections prevent newly introduced map/mask settings
        // from silently escaping the private resume commitment.
        let Int4ExtractionMapLimits { mapping, verification } = document;
        let Int4ExtractionMappingLimits { chunks, reduction, max_model_work, mask_limits,
            mask_visits_per_chunk, max_mask_visits } = mapping;
        let MaskWorkLimits { max_trie_node_visits, checkpoint_interval_nodes } = mask_limits;
        let recipe = Self { version: 1, dependency_scope: "item-local-complete-document",
            execution: INT4_EXTRACTION_MAP_EXECUTION, chunk_profile: CHUNK_PROFILE,
            reservation_policy: "exact-all-chunk-native-work-per-durable-attempt-v1",
            extraction, chunks, reduction, per_document_model_work: max_model_work,
            max_trie_node_visits, checkpoint_interval_nodes, mask_visits_per_chunk,
            max_mask_visits_per_document: max_mask_visits, verification,
            max_lifetime_verification, max_attempts, chunk_cohorts: None };
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        Ok(recipe)
    }
}

/// Model-bound immutable factory. This owns the same actual pinned compiler,
/// fixed raw schema and source-verification settings used for every chunk.
pub struct Int4DocumentExtractionJobPlanner {
    compiler: Int4ExtractionBatchPlanner,
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
    // Fixtures use the real pinned compiler without exposing an alternate
    // production model constructor or any fabricated native execution result.
    fn configured(extraction: Int4ExtractionJobPlanner, document: Int4ExtractionMapLimits,
        limits: JobLimits, max_lifetime_verification: GroundingBudget) -> Result<Self, BatchFault> {
        let width = extraction.recipe.group_width;
        validate_limits(width, extraction.task_ceiling(), extraction.recipe.defaults.budget, document, limits)?;
        verify_lifetime(document.verification, max_lifetime_verification, limits.max_attempts)?;
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
        Ok(Self { compiler, identity, document, limits, recipe, owner: Arc::new(()) })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_ceiling(&self) -> TaskBudget { self.recipe.extraction.task_ceiling }
    pub fn document_budget(&self) -> TaskBudget { self.request().budget }
    pub fn document_limits(&self) -> Int4ExtractionMapLimits { self.document }
    pub fn job_limits(&self) -> JobLimits { self.limits }
    pub fn request(&self) -> &ExtractionBatchArgs { &self.recipe.extraction.defaults }
    pub fn group_width(&self) -> usize { self.recipe.extraction.group_width }
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
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); }
        verify_lifetime(self.document.verification, self.recipe.max_lifetime_verification, limits.max_attempts)
    }
    fn output_tokens(&self, chunks: usize) -> Result<u64, BatchFault> {
        u64::try_from(chunks).ok().and_then(|n| n.checked_mul(u64::from(self.document_budget().max_output_tokens)))
            .ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    fn owns(&self, prepared: &PreparedInt4ExtractionDocument) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
    fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, control: &mut C)
        -> Result<PreparedInt4ExtractionMap<'s>, BatchItemFailure> {
        poll(control)?;
        if source.len() > self.limits.max_input_bytes_per_item {
            return Err(BatchItemFailure::fatal(BatchCode::DocumentLimit));
        }
        self.compiler.plan_document_with_control(source, self.request(), self.document, control).map_err(map_failure)
    }
    fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<ExtractionDocumentArgs>, control: &mut C)
        -> Result<PreparedInt4ExtractionDocument, BatchItemFailure> {
        let plan = self.plan(&document.text, control)?;
        let expected = plan.preflight_metadata();
        let mut identities = Vec::new();
        identities.try_reserve_exact(expected.chunk_count()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in plan.execution_identities() {
            poll(control)?;
            identities.push(identity.clone());
        }
        if identities.len() != expected.chunk_count() || expected.chunk_count() == 0
            || expected.group_width() != self.group_width() || plan.max_kv_bytes() != self.document_budget().max_kv_bytes
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
    fn compare<C: DecodeStepControl>(&self, plan: &PreparedInt4ExtractionMap<'_>,
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
    expected: Int4ExtractionMapPreflight,
    work: JobWork,
    owner: Arc<()>,
}
impl PreparedInt4ExtractionDocument {
    pub fn chunk_count(&self) -> usize { self.expected.chunk_count() }
    pub fn planned_work(&self) -> JobWork { self.work }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
}
fn receipt_work(expected: Int4ExtractionMapPreflight) -> JobWork {
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
        Int4ExtractionMapError::Extraction(_) | Int4ExtractionMapError::Reduction(_) => BatchCode::InvalidExecution,
    })
}

#[cfg(test)] mod tests;
