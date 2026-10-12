//! Durable, whole-document native source maps. No chunk is a journal item.
//! Exact map work is prepared before the durable debit, then reconstructed and
//! compared before admission. Results preserve independent chunk semantics.
use super::*;
use std::sync::Arc;
use serde::Deserialize;
use crate::{
    jobs::JobLimits,
    native_engine::kv::KV_BYTES_PER_TOKEN,
    tasks::{ir::PlanContext, mapreduce::{CHUNK_PROFILE, ChunkLimits, ExecutionLimits},
        source_planning::quantized::long::{INT8_SOURCE_MAP_EXECUTION, Int8SourceMapError,
            Int8SourceMapLimits, Int8SourceMapRun, PreparedInt8SourceMap, SourceMapTask}},
    tokenizer::specials::TemplateControlIds,
};
pub mod reduction;
// Reuse the all-identities/whole-output admission contract, NOT extraction's
// task-specific host implementation. The source host checks its fixed task.
pub use crate::batch::extract::quantized::long::stream::{
    ExtractionDocumentAdmission as SourceDocumentAdmission,
    ExtractionDocumentBatchAdmission as SourceDocumentBatchAdmission,
};

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDocumentArgs {}

/// Private projection for a keyed manifest, never a deserializable executable.
#[derive(Serialize)]
pub struct Int8SourceDocumentJobRecipe {
    version: u32,
    dependency_scope: &'static str,
    execution: &'static str,
    chunk_profile: &'static str,
    reservation_policy: &'static str,
    source: Int8SourceJobRecipe,
    chunks: ChunkLimits,
    reduction: ExecutionLimits,
    document_ceiling: JobWork,
    mask_visits_per_chunk: u64,
    // Omission preserves ordinary v1 map-job commitments exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    reduction_task: Option<reduction::SourceDocumentReduction>,
}

/// Owns the actual pinned source compiler and the SAME settings that are frozen.
/// No caller-supplied task factory, arbitrary recipe, or result-injection seam.
pub struct Int8SourceDocumentJobPlanner {
    source: SourceTaskPlanner,
    identity: ExecutionIdentity,
    task: SourceMapTask,
    budget: TaskBudget,
    planning: SourcePlanningLimits,
    mapping: Int8SourceMapLimits,
    limits: JobLimits,
    recipe: Int8SourceDocumentJobRecipe,
    owner: Arc<()>,
}
impl Int8SourceDocumentJobPlanner {
    #[allow(clippy::too_many_arguments)]
    pub fn pinned<C: DecodeStepControl>(controls: &TemplateControlIds, eos: u32,
        mut identity: ExecutionIdentity, task: SourceMapTask, budget: TaskBudget,
        planning: SourcePlanningLimits, mut mapping: Int8SourceMapLimits,
        limits: JobLimits, control: &mut C) -> Result<Self, BatchFault> {
        poll(control).map_err(|e| e.fault)?;
        limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
        budget.validate().map_err(|_| BatchCode::InvalidLimits)?;
        validate_mapping(mapping, budget, planning, limits)?;
        // Bound options BEFORE cloning any caller-owned label/type strings.
        bounded_json(&task, 1024 * 1024 - 16 * 1024).map_err(|_| BatchCode::InvalidLimits)?;
        let defaults = match &task {
            SourceMapTask::Ner(options) => SourceBatchArgs::Ner { options: options.clone(), budget },
            SourceMapTask::Keyphrases(options) => SourceBatchArgs::Keyphrases { options: *options, budget },
            SourceMapTask::Summarize(options) => SourceBatchArgs::Summarize { options: *options, budget },
        };
        let source = SourceTaskPlanner::pinned(controls, eos).map_err(|_| BatchCode::Planning)?;
        // Bind the ACTUAL factory, rather than a caller's tokenizer placeholders.
        identity.template_digest = *source.template_digest();
        identity.tokenizer_digest = source.tokenizer_digest();
        quantized::check_configuration(&source, &identity, budget, planning, Some(&defaults))?;
        let context = PlanContext::new(&identity, budget).map_err(|_| BatchCode::Admission)?;
        let capacity = source.int8_map_capacity_with_control(&task, budget, &context, planning, control)
            .map_err(|e| map_failure(e.into()).fault)?;
        mapping.chunks = capacity.constrain_chunks(mapping.chunks).map_err(|e| map_failure(e.into()).fault)?;
        validate_mapping(mapping, budget, planning, limits)?;
        let native = Int8SourceBatchLimits { max_model_work: limits.max_work.model,
            masks: SourceMaskBudget { per_mask: mapping.mask_limits,
                max_visits_per_item: mapping.max_mask_visits, max_visits_per_run: limits.max_work.mask_node_visits } };
        quantized::validate_limits(native)?;
        let recipe = Int8SourceDocumentJobRecipe { version: 1,
            dependency_scope: "item-local-complete-document", execution: INT8_SOURCE_MAP_EXECUTION,
            chunk_profile: CHUNK_PROFILE, reservation_policy: "exact-complete-map-per-durable-attempt-v1",
            source: Int8SourceJobRecipe::new(budget, planning, native, Some(defaults)),
            chunks: mapping.chunks, reduction: mapping.reduction,
            document_ceiling: JobWork { model: mapping.max_model_work, mask_node_visits: mapping.max_mask_visits },
            mask_visits_per_chunk: mapping.mask_visits_per_chunk, reduction_task: None };
        bounded_json(&recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        poll(control).map_err(|e| e.fault)?;
        Ok(Self { source, identity, task, budget, planning, mapping, limits, recipe, owner: Arc::new(()) })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn planning_limits(&self) -> SourcePlanningLimits { self.planning }
    pub fn document_limits(&self) -> Int8SourceMapLimits { self.mapping }
    pub fn job_recipe(&self) -> &Int8SourceDocumentJobRecipe { &self.recipe }
    pub fn check_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        // Exact immutable lifetime authority; reopening cannot raise any axis.
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); }
        Ok(())
    }
    fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, control: &mut C)
        -> Result<PreparedInt8SourceMap<'s>, BatchItemFailure> {
        poll(control)?;
        if source.is_empty() || source.len() > self.mapping.chunks.max_input_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::DocumentLimit));
        }
        let context = PlanContext::new(&self.identity, self.budget)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let plan = self.source.plan_int8_map_with_control(source, &self.task, self.budget, &context,
            self.planning, self.mapping, control).map_err(map_failure)?;
        if let Some(policy) = self.reduction_policy() { policy.check_plan(&plan)?; }
        Ok(plan)
    }
    /// No model or filesystem work. Owns the exact text and private identity
    /// snapshot; reconstruction avoids a self-referential prepared object.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SourceDocumentArgs>,
        control: &mut C) -> Result<PreparedSourceDocument, BatchItemFailure> {
        let plan = self.plan(&document.text, control)?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(plan.chunk_count()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        for identity in plan.execution_identities() {
            poll(control)?;
            identities.push(identity.clone());
        }
        let work = map_work(&plan);
        drop(plan);
        poll(control)?;
        Ok(PreparedSourceDocument { source: document.text, identities, work, owner: Arc::clone(&self.owner) })
    }
    fn owns(&self, prepared: &PreparedSourceDocument) -> Result<(), BatchItemFailure> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        Ok(())
    }
}

/// No Clone/Deserialize/public mutation. A prepared document cannot cross from
/// a different factory, even when that other factory has the same public task.
pub struct PreparedSourceDocument {
    source: String,
    identities: Vec<ExecutionIdentity>,
    work: JobWork,
    owner: Arc<()>,
}
impl PreparedSourceDocument {
    pub fn chunk_count(&self) -> usize { self.identities.len() }
    pub fn planned_work(&self) -> JobWork { self.work }
    pub fn execution_identities(&self) -> &[ExecutionIdentity] { &self.identities }
}
fn map_work(plan: &PreparedInt8SourceMap<'_>) -> JobWork {
    JobWork { model: plan.planned_work(), mask_node_visits: plan.reserved_mask_visits() }
}
fn snapshot_matches(plan: &PreparedInt8SourceMap<'_>, prepared: &PreparedSourceDocument) -> bool {
    map_work(plan) == prepared.work && plan.execution_identities().eq(prepared.identities.iter())
}

pub struct Int8SourceDocumentJobProcessor<'p, 'e, 'w, 'v, A: SourceDocumentBatchAdmission> {
    planner: &'p Int8SourceDocumentJobPlanner,
    engine: &'e mut StrictInt8Engine<'w>,
    vocabulary: &'v ExtractionVocabulary,
    admission: A,
    ledger: Ledger,
}
impl<'p, 'e, 'w, 'v, A: SourceDocumentBatchAdmission> Int8SourceDocumentJobProcessor<'p, 'e, 'w, 'v, A> {
    pub fn new(planner: &'p Int8SourceDocumentJobPlanner, engine: &'e mut StrictInt8Engine<'w>,
        vocabulary: &'v ExtractionVocabulary, admission: A) -> Result<Self, BatchFault> {
        // A reduced recipe must never execute through the ordinary map route.
        if planner.reduction_policy().is_some() { return Err(BatchCode::InvalidLimits.into()); }
        check_capacity(planner, engine).map_err(|e| e.fault)?;
        Ok(Self { planner, engine, vocabulary, admission, ledger: Ledger::default() })
    }
}
impl<A: SourceDocumentBatchAdmission> BatchProcessor for Int8SourceDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Args = SourceDocumentArgs;
    type Prepared = PreparedSourceDocument;
    type Output = GuardedOutput<Int8SourceMapRun, A::Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?; self.ledger.failed = true;
        check_capacity(self.planner, self.engine)?;
        let prepared = self.planner.prepare_with_control(document, control)?;
        self.ledger.failed = false;
        Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: prepared.work.model.forward_positions,
            projected_logits: prepared.work.model.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.ledger.failed = true; Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        // The journal already charged this exact complete map. This additional
        // invocation ledger is not durable authority and never replaces begin().
        self.ledger.begin(prepared.work, context, self.planner.limits)?;
        self.planner.owns(&prepared)?;
        poll(control)?;
        let kv = check_capacity(self.planner, self.engine)?;
        let plan = self.planner.plan(&prepared.source, control)?;
        if !snapshot_matches(&plan, &prepared) { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        plan.preflight(&prepared.identities, self.engine).map_err(map_failure)?;
        let (admitted, guard) = self.admission.admit(SourceDocumentAdmission {
            identities: &prepared.identities, model_work: prepared.work.model,
            mask_node_visits: prepared.work.mask_node_visits, mask_limits: self.planner.mapping.mask_limits,
            kv_reservation_bytes: kv, max_result_bytes: plan.max_result_bytes(),
        })?;
        poll(control)?;
        // Rechecks EVERY returned identity, not only the first chunk, before
        // the first forward. The actual driver keeps the same clean engine.
        let output = plan.execute_with_control(&admitted, self.engine, self.vocabulary, control).map_err(map_failure)?;
        if output.planned_model_work != prepared.work.model
            || output.reserved_mask_node_visits != prepared.work.mask_node_visits
            || !(JobWork { model: output.model_work, mask_node_visits: output.mask_node_visit_charge }).fits(prepared.work)
            || output.mapped.root().value().len() != prepared.chunk_count() {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        check_capacity(self.planner, self.engine)?;
        poll(control)?;
        self.ledger.failed = false;
        // This guard outlives the complete result's spool sync/journal commit.
        Ok(GuardedOutput::new(output, guard))
    }
}
impl<A: SourceDocumentBatchAdmission> DurableBatchProcessor for Int8SourceDocumentJobProcessor<'_, '_, '_, '_, A> {
    type Recipe = Int8SourceDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.ledger.ready()?;
        self.planner.owns(prepared)?;
        check_capacity(self.planner, self.engine)?;
        Ok(prepared.work)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.mapping.reduction.max_result_bytes as u64 }
}

#[derive(Default)]
struct Ledger { work: JobWork, attempts: u64, sequence: u64, failed: bool }
impl Ledger {
    fn ready(&self) -> Result<(), BatchItemFailure> {
        if self.failed { Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)) } else { Ok(()) }
    }
    fn begin(&mut self, work: JobWork, context: BatchRequestContext, limits: JobLimits) -> Result<(), BatchItemFailure> {
        self.ready()?; self.failed = true;
        if context.request_seq <= self.sequence || context.epoch == 0 || context.input_line == 0 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        let attempts = self.attempts.checked_add(1).filter(|&n| n <= limits.max_attempts)
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        let next = self.work.checked_add(work).ok().filter(|w| w.fits(limits.max_work))
            .ok_or_else(|| BatchItemFailure::fatal(BatchCode::WorkLimit))?;
        self.work = next; self.attempts = attempts; self.sequence = context.request_seq;
        Ok(())
    }
}
fn check_capacity(planner: &Int8SourceDocumentJobPlanner, engine: &StrictInt8Engine<'_>) -> Result<u64, BatchItemFailure> {
    capacity_bytes(!engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0),
        engine.kv_cache().capacity_positions(), planner.planning.max_context_tokens, planner.budget.max_kv_bytes)
}
fn capacity_bytes(clean: bool, capacity: usize, required: usize, maximum: u64) -> Result<u64, BatchItemFailure> {
    if !clean || capacity < required { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
    (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).filter(|&n| n != 0 && n <= maximum)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.prefill_checkpoint(0) { Some(c) => Err(BatchItemFailure::fatal(BatchFault::cancelled(c))), None => Ok(()) }
}
fn map_failure(error: Int8SourceMapError) -> BatchItemFailure {
    if let Some(c) = error.cancellation() { return BatchItemFailure::fatal(BatchFault::cancelled(c)); }
    let code = match error {
        Int8SourceMapError::InvalidLimits => BatchCode::InvalidLimits,
        Int8SourceMapError::EmptySource => BatchCode::DocumentLimit,
        Int8SourceMapError::WorkLimit => BatchCode::WorkLimit,
        Int8SourceMapError::Admission => BatchCode::Admission,
        Int8SourceMapError::Allocation => BatchCode::Allocation,
        Int8SourceMapError::Chunk(_) => BatchCode::Planning,
        _ => BatchCode::InvalidExecution,
    };
    BatchItemFailure::fatal(code)
}
fn validate_mapping(m: Int8SourceMapLimits, budget: TaskBudget, p: SourcePlanningLimits, job: JobLimits)
    -> Result<(), BatchFault> {
    m.chunks.effective_token_limit().map_err(|_| BatchCode::InvalidLimits)?;
    let e = m.reduction;
    if m.chunks.max_input_bytes == 0 || m.chunks.max_chunks > 256
        || m.chunks.max_chunk_bytes > p.max_input_bytes || m.chunks.context_tokens > p.max_context_tokens
        || m.chunks.reserved_tokens < budget.max_output_tokens as usize
        || !(JobWork { model: m.max_model_work, mask_node_visits: m.max_mask_visits }).fits(job.max_work)
        || m.max_mask_visits < m.mask_visits_per_chunk || m.mask_visits_per_chunk == 0
        || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
        || !(1..=1024).contains(&e.map_batch_chunks) || !(2..=1024).contains(&e.reduce_fan_in)
        || e.max_reduction_levels > 64 || !(1..=1_000_000).contains(&e.max_task_calls)
        || [e.max_value_bytes, e.max_live_value_bytes, e.max_total_value_bytes, e.max_result_bytes]
            .iter().any(|n| !(1..=64 * 1024 * 1024).contains(n))
        || e.max_value_bytes > e.max_live_value_bytes || e.max_result_bytes > job.max_result_bytes {
        return Err(BatchCode::InvalidLimits.into());
    }
    // A document ceiling must fund native work on every axis independently.
    quantized::validate_limits(Int8SourceBatchLimits { max_model_work: m.max_model_work,
        masks: SourceMaskBudget { per_mask: m.mask_limits,
            max_visits_per_item: m.mask_visits_per_chunk, max_visits_per_run: m.max_mask_visits } })
}

#[cfg(test)] pub(crate) mod tests;
