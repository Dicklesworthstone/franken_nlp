//! Raw long-document summary execution over the existing NativeSourceBatch.
//!
//! Planning uses the pinned encoder and a real compiled summary scaffold. All
//! chunk work is budgeted before model admission. Native maps run sequentially
//! on ONE supplied engine; reduction never invokes another model or runtime.
//! This is not durable-job/cache authority or a binary CLI activation path.

use std::{cell::{Cell, RefCell}, error::Error, fmt, mem::size_of};
use serde::Serialize;
use crate::{
    batch::{BatchCode, BatchDocument, BatchFault, BatchProcessor, BatchWork,
        source::{NativeSourceBatch, SourceBatchAdmission, SourceBatchArgs, SourceBatchPlanner,
            SourceMaskBudget, GuardedOutput}},
    execution_identity::ExecutionIdentity,
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        hf_bf16_eager::HfBf16EagerEngine, kv::KV_BYTES_PER_TOKEN, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::{extract::ExtractionVocabulary, ir::{PlanContext, TaskBudget},
        mapreduce::{self, ChunkLimits, ChunkPlan, ExecutionError, ExecutionLimits, MapReduceError, SourceChunk, CHUNK_PROFILE},
        source_planning::{SourcePlanningError, SourcePlanningLimits, SourceTaskPlanner, SourceTaskRequest, SourceTaskResult},
        summarize::{SummaryError, SummaryOptions, SummaryResult, SUMMARIZE_TASK_VERSION}},
    validation::grounded_fields::VerifiedSourceSpan,
};
use super::summarize::{CorpusSummaryError, CorpusSummaryLimits, CorpusSummaryResult, CorpusSummaryTask, SummaryPass, check_bytes};

pub const NATIVE_CORPUS_SUMMARY: &str = "native-serial-cited-summary-map-exact-reduce-v1";

#[derive(Clone, Copy, Debug)]
pub struct NativeCorpusSummaryConfig {
    pub chunks: ChunkLimits,
    pub execution: ExecutionLimits,
    pub aggregation: CorpusSummaryLimits,
    pub options: SummaryOptions,
    pub budget: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub masks: SourceMaskBudget,
    pub max_work: BatchWork,
    pub max_bullets: usize,
    pub max_result_bytes: usize,
    /// Inline storage of retained host guard values; the host separately owns
    /// the accounting of any heap allocations inside its opaque guard type.
    pub max_retained_guard_bytes: usize,
}

#[derive(Debug)]
pub enum NativeCorpusSummaryError {
    InvalidLimits, WorkBudget, Accounting, AllocationRefused,
    Cancelled(DecodeCancellationKind), Planning(SourcePlanningError), Chunk(MapReduceError),
    Batch(BatchFault), Summary(SummaryError),
    Execution(ExecutionError<CorpusSummaryError<BatchFault>>),
}
impl fmt::Display for NativeCorpusSummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid native corpus summary limits",
            Self::WorkBudget => "native corpus summary whole-run work budget exceeded",
            Self::Accounting => "native corpus summary execution accounting mismatch",
            Self::AllocationRefused => "native corpus summary allocation refused",
            Self::Cancelled(_) => "native corpus summary cancelled",
            Self::Planning(_) => "native corpus summary planning failed",
            Self::Chunk(_) => "native corpus summary partition failed",
            Self::Batch(_) => "native corpus summary admission or map failed",
            Self::Summary(_) => "native corpus summary finalization failed",
            Self::Execution(_) => "native corpus summary map/reduce failed",
        })
    }
}
impl Error for NativeCorpusSummaryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Chunk(e) => Some(e), Self::Batch(e) => Some(e),
            Self::Summary(e) => Some(e), Self::Execution(e) => Some(e), _ => None }
    }
}

/// Borrows the exact source and planner. No wire/Debug constructor or external
/// source digest; per-chunk complete execution identities are minted by the
/// same planner and checked by NativeSourceBatch's real host admission hook.
pub struct PreparedSummaryCorpus<'s, 'p> {
    chunks: ChunkPlan<'s>,
    planner: &'p SourceTaskPlanner,
    identity: ExecutionIdentity,
    config: NativeCorpusSummaryConfig,
    scaffold_tokens: usize,
    work: BatchWork,
    maximum_forward_positions: u64,
    reserved_mask_visits: u64,
}
impl PreparedSummaryCorpus<'_, '_> {
    pub fn chunks(&self) -> &ChunkPlan<'_> { &self.chunks }
    pub fn planned_work(&self) -> BatchWork { self.work }
    pub fn reserved_mask_visits(&self) -> u64 { self.reserved_mask_visits }
    pub fn scaffold_tokens(&self) -> usize { self.scaffold_tokens }

    /// The embedding host must supply its actual whole-run memory/output guard
    /// in addition to the per-map admission hook. It must cover the configured
    /// live reduction/serialization workspace and retained metadata. This API
    /// grants no new process authority and never constructs a fake permit.
    /// All guards move into the final result and outlive its storage/delivery.
    pub fn execute_native<A: SourceBatchAdmission, C: DecodeStepControl, G>(&self,
        engine: &mut HfBf16EagerEngine, vocabulary: &ExtractionVocabulary,
        admission: A, run_guard: G, control: &mut C,
    ) -> Result<GuardedOutput<NativeCorpusSummaryResult, (Vec<A::Guard>, G)>, NativeCorpusSummaryError> {
        if !engine.kv_cache().all_slots_have_len(0) { return Err(NativeCorpusSummaryError::Accounting); }
        let capacity = engine.kv_cache().capacity_positions() as u64;
        if capacity < self.maximum_forward_positions || capacity.checked_mul(KV_BYTES_PER_TOKEN as u64)
            .is_none_or(|n| n > self.config.budget.max_kv_bytes) { return Err(NativeCorpusSummaryError::WorkBudget); }
        let guard_bytes = self.chunks.chunks().len().checked_mul(size_of::<A::Guard>())
            .ok_or(NativeCorpusSummaryError::InvalidLimits)?;
        if guard_bytes > self.config.max_retained_guard_bytes { return Err(NativeCorpusSummaryError::WorkBudget); }
        let mut guards = Vec::new();
        guards.try_reserve_exact(self.chunks.chunks().len()).map_err(|_| NativeCorpusSummaryError::AllocationRefused)?;
        let compiler = SourceBatchPlanner::new(self.planner, self.identity.clone(), self.config.budget,
            self.config.planning, None).map_err(NativeCorpusSummaryError::Batch)?;
        let processor = NativeSourceBatch::new(compiler, engine, vocabulary, admission, self.config.masks)
            .map_err(NativeCorpusSummaryError::Batch)?;
        // Calls are synchronous and sequential. Short RefCell borrows share the
        // SAME cancellation controller with partition/map/reduce checkpoints;
        // no trait defaults or cancellation kinds are discarded by an adapter.
        let control = RefCell::new(control); let cause = Cell::new(None);
        let pass = BatchSummaryPass { processor, control: &control, options: self.config.options,
            budget: self.config.budget, scaffold_tokens: self.scaffold_tokens,
            remaining: self.work, guards, failed: false };
        let mut task = CorpusSummaryTask::new(pass, self.config.aggregation).map_err(NativeCorpusSummaryError::Summary)?;
        let reduced = mapreduce::execute(&self.chunks, &mut task, self.config.execution,
            || checkpoint(&control, &cause)).map_err(|e| match cause.get() {
                Some(reason) => NativeCorpusSummaryError::Cancelled(reason),
                None => NativeCorpusSummaryError::Execution(e),
            })?;
        checkpoint(&control, &cause).map_err(|e| match cause.get() {
            Some(reason) => NativeCorpusSummaryError::Cancelled(reason), None => NativeCorpusSummaryError::Chunk(e),
        })?;
        let metadata = (reduced.map_batches(), reduced.reduce_calls(), reduced.reduction_levels(), reduced.root().source_span());
        let verification_scan_steps = self.config.aggregation.max_scan_steps - task.scan_steps_remaining();
        let pass = task.into_pass();
        if pass.remaining != BatchWork::default() { return Err(NativeCorpusSummaryError::Accounting); }
        // Keep guards alive while the aggregate is finalized and copied into
        // its publication envelope, including all error paths below.
        let BatchSummaryPass { guards, .. } = pass;
        let summary = reduced.into_value().into_ranked(self.config.max_bullets, self.config.max_result_bytes)
            .map_err(NativeCorpusSummaryError::Summary)?;
        if summary.forward_positions > self.work.forward_positions || summary.projected_logits > self.work.projected_logits
            || summary.forward_positions.checked_mul(NANBEIGE_VOCAB_SIZE as u64) != Some(summary.projected_logits)
            || summary.mask_node_visit_charge > self.reserved_mask_visits {
            return Err(NativeCorpusSummaryError::Accounting);
        }
        let result = NativeCorpusSummaryResult { schema_version: 1, execution: NATIVE_CORPUS_SUMMARY,
            chunk_profile: CHUNK_PROFILE, source_span: metadata.3, map_batches: metadata.0,
            reduce_calls: metadata.1, reduction_levels: metadata.2, reserved_work: self.work,
            reserved_mask_visits: self.reserved_mask_visits, verification_scan_steps,
            untrusted_fields: ["summary.bullets"], summary };
        check_bytes(&result, self.config.max_result_bytes).map_err(NativeCorpusSummaryError::Summary)?;
        checkpoint(&control, &cause).map_err(|e| match cause.get() {
            Some(reason) => NativeCorpusSummaryError::Cancelled(reason), None => NativeCorpusSummaryError::Chunk(e),
        })?;
        Ok(GuardedOutput::new(result, (guards, run_guard)))
    }
}

#[derive(Serialize)]
pub struct NativeCorpusSummaryResult {
    pub schema_version: u32,
    pub execution: &'static str,
    pub chunk_profile: &'static str,
    /// Includes all mapped input, even when final selection omits its bullets.
    pub source_span: VerifiedSourceSpan,
    pub map_batches: usize,
    pub reduce_calls: usize,
    pub reduction_levels: usize,
    /// Conservative precharged ceilings; actual counters are inside summary.
    pub reserved_work: BatchWork,
    pub reserved_mask_visits: u64,
    pub verification_scan_steps: u64,
    pub untrusted_fields: [&'static str; 1],
    pub summary: CorpusSummaryResult,
}

/// Model-free preparation. Caller ceilings can only shrink. A real one-byte
/// summary plan measures the pinned scaffold; source counting uses the SAME
/// byte-preserving encoder as native execution, not an estimated token ratio.
pub fn prepare_summary_corpus<'s, 'p, C: DecodeStepControl>(source: &'s str,
    planner: &'p SourceTaskPlanner, identity: &ExecutionIdentity, config: NativeCorpusSummaryConfig,
    control: &mut C) -> Result<PreparedSummaryCorpus<'s, 'p>, NativeCorpusSummaryError> {
    config.options.validate().map_err(NativeCorpusSummaryError::Summary)?;
    config.aggregation.validate().map_err(NativeCorpusSummaryError::Summary)?;
    config.budget.validate().map_err(|_| NativeCorpusSummaryError::InvalidLimits)?;
    config.chunks.effective_token_limit().map_err(NativeCorpusSummaryError::Chunk)?;
    if identity.task_spec != SUMMARIZE_TASK_VERSION || source.is_empty()
        || !(1..=1024).contains(&config.max_bullets) || !(1..=64 * 1024 * 1024).contains(&config.max_result_bytes)
        || config.max_retained_guard_bytes > 64 * 1024 * 1024
        || config.masks.max_visits_per_item == 0 || config.masks.per_mask.max_trie_node_visits == 0
        || config.masks.per_mask.checkpoint_interval_nodes == 0 {
        return Err(NativeCorpusSummaryError::InvalidLimits);
    }
    if source.len() > config.chunks.max_input_bytes { return Err(NativeCorpusSummaryError::Chunk(MapReduceError::InputBudget)); }
    if let Some(reason) = control.prefill_checkpoint(0) { return Err(NativeCorpusSummaryError::Cancelled(reason)); }
    let context = PlanContext::new(identity, config.budget).map_err(|_| NativeCorpusSummaryError::InvalidLimits)?;
    let probe = planner.plan(&SourceTaskRequest::Summarize { document: "x".to_owned(), options: config.options,
        budget: config.budget }, &context, config.planning).map_err(NativeCorpusSummaryError::Planning)?;
    let encoded_probe = planner.source_encoder().encode("x", 1, usize::MAX)
        .map_err(|e| NativeCorpusSummaryError::Planning(SourcePlanningError::Extraction(e)))?;
    let scaffold_tokens = probe.prompt_tokens().checked_sub(encoded_probe.token_ids().len())
        .ok_or(NativeCorpusSummaryError::Accounting)?;
    let mut limits = effective_chunks(config, scaffold_tokens)?;
    limits.max_chunk_bytes = limits.max_chunk_bytes.min(config.planning.max_input_bytes);
    let mut cause = None;
    let chunks = ChunkPlan::build_with_checkpoints(source, limits,
        |text| planner.source_encoder().encode(text, limits.max_chunk_bytes, usize::MAX)
            .map(|d| d.token_ids().len()).map_err(|_| MapReduceError::Tokenizer),
        || match control.prefill_checkpoint(0) {
            Some(reason) => { cause = Some(reason); Err(MapReduceError::Cancelled) }, None => Ok(()),
        }).map_err(|e| match cause { Some(reason) => NativeCorpusSummaryError::Cancelled(reason),
            None => NativeCorpusSummaryError::Chunk(e) })?;
    let mut work = BatchWork::default(); let mut maximum_forward_positions = 0;
    for chunk in chunks.chunks() {
        let charge = chunk_work(chunk.tokens(), scaffold_tokens, config.budget.max_output_tokens)
            .map_err(NativeCorpusSummaryError::Batch)?;
        work.forward_positions = work.forward_positions.checked_add(charge.forward_positions)
            .ok_or(NativeCorpusSummaryError::WorkBudget)?;
        work.projected_logits = work.projected_logits.checked_add(charge.projected_logits)
            .ok_or(NativeCorpusSummaryError::WorkBudget)?;
        maximum_forward_positions = maximum_forward_positions.max(charge.forward_positions);
    }
    let reserved_mask_visits = (chunks.chunks().len() as u64).checked_mul(config.masks.max_visits_per_item)
        .ok_or(NativeCorpusSummaryError::WorkBudget)?;
    if work.forward_positions > config.max_work.forward_positions || work.projected_logits > config.max_work.projected_logits
        || reserved_mask_visits > config.masks.max_visits_per_run { return Err(NativeCorpusSummaryError::WorkBudget); }
    if let Some(reason) = control.prefill_checkpoint(0) { return Err(NativeCorpusSummaryError::Cancelled(reason)); }
    Ok(PreparedSummaryCorpus { chunks, planner, identity: identity.clone(), config, scaffold_tokens,
        work, maximum_forward_positions, reserved_mask_visits })
}

fn effective_chunks(config: NativeCorpusSummaryConfig, scaffold: usize) -> Result<ChunkLimits, NativeCorpusSummaryError> {
    let output = config.budget.max_output_tokens as usize;
    let reserved = scaffold.checked_add(output).ok_or(NativeCorpusSummaryError::InvalidLimits)?;
    let prompt_room = (config.budget.max_input_tokens as usize).checked_sub(scaffold)
        .ok_or(NativeCorpusSummaryError::InvalidLimits)?;
    let kv_positions = config.budget.max_kv_bytes / KV_BYTES_PER_TOKEN as u64;
    let kv_room = kv_positions.checked_sub(reserved.checked_sub(1).ok_or(NativeCorpusSummaryError::InvalidLimits)? as u64)
        .ok_or(NativeCorpusSummaryError::InvalidLimits)?;
    let kv_room = usize::try_from(kv_room).unwrap_or(usize::MAX);
    let mut limits = config.chunks;
    limits.reserved_tokens = limits.reserved_tokens.max(reserved);
    limits.context_tokens = limits.context_tokens.min(config.planning.max_context_tokens);
    limits.max_chunk_tokens = limits.max_chunk_tokens.min(prompt_room).min(kv_room);
    limits.effective_token_limit().map_err(NativeCorpusSummaryError::Chunk)?;
    Ok(limits)
}
fn chunk_work(tokens: usize, scaffold: usize, output: u32) -> Result<BatchWork, BatchFault> {
    let forward = (tokens as u64).checked_add(scaffold as u64).and_then(|n| n.checked_add(u64::from(output)))
        .and_then(|n| n.checked_sub(1)).ok_or(BatchCode::WorkLimit)?;
    Ok(BatchWork { forward_positions: forward,
        projected_logits: forward.checked_mul(NANBEIGE_VOCAB_SIZE as u64).ok_or(BatchCode::WorkLimit)? })
}
fn reserve(remaining: &mut BatchWork, charge: BatchWork) -> Result<(), BatchFault> {
    let next = BatchWork {
        forward_positions: remaining.forward_positions.checked_sub(charge.forward_positions).ok_or(BatchCode::WorkLimit)?,
        projected_logits: remaining.projected_logits.checked_sub(charge.projected_logits).ok_or(BatchCode::WorkLimit)?,
    };
    *remaining = next; Ok(())
}
fn checkpoint<C: DecodeStepControl>(control: &RefCell<&mut C>, cause: &Cell<Option<DecodeCancellationKind>>)
    -> Result<(), MapReduceError> {
    match control.borrow_mut().prefill_checkpoint(0) {
        Some(reason) => { cause.set(Some(reason)); Err(MapReduceError::Cancelled) }, None => Ok(()),
    }
}

/// Internal static adapter, generic solely so orchestration/guard semantics can
/// be tested without a model. Production constructs it with NativeSourceBatch.
struct BatchSummaryPass<'a, 'c, P, C, G> {
    processor: P,
    control: &'a RefCell<&'c mut C>,
    options: SummaryOptions,
    budget: TaskBudget,
    scaffold_tokens: usize,
    remaining: BatchWork,
    guards: Vec<G>,
    failed: bool,
}
impl<P, C, G> SummaryPass for BatchSummaryPass<'_, '_, P, C, G>
where P: BatchProcessor<Args = SourceBatchArgs, Output = GuardedOutput<SourceTaskResult, G>>, C: DecodeStepControl {
    type Error = BatchFault;
    fn options(&self) -> SummaryOptions { self.options }
    fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<SummaryResult, BatchFault> {
        if self.failed { return Err(BatchCode::InvalidExecution.into()); }
        self.failed = true;
        if self.guards.len() == self.guards.capacity() { return Err(BatchCode::Allocation.into()); }
        let mut text = String::new(); text.try_reserve_exact(chunk.text().len()).map_err(|_| BatchCode::Allocation)?;
        text.push_str(chunk.text());
        let prepared = self.processor.prepare(BatchDocument { id: chunk.id().to_string(), text,
            task_args: Some(SourceBatchArgs::Summarize { options: self.options, budget: self.budget }) }).map_err(|e| e.fault)?;
        let charge = self.processor.planned_work(&prepared);
        if charge != chunk_work(chunk.tokens(), self.scaffold_tokens, self.budget.max_output_tokens)? {
            return Err(BatchCode::InvalidExecution.into());
        }
        reserve(&mut self.remaining, charge)?; // never refunded after this point
        let (result, guard) = self.processor.execute(prepared, &mut **self.control.borrow_mut())
            .map_err(|e| e.fault)?.into_parts();
        self.guards.push(guard);
        let SourceTaskResult::Summarize(summary) = result else { return Err(BatchCode::InvalidExecution.into()); };
        self.failed = false; Ok(summary)
    }
}

#[cfg(test)]
mod tests;
