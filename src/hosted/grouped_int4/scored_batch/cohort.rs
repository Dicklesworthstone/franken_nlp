//! Cross-document scoring windows inside one physically owned stream.
use super::*;
use crate::batch::{grouped::{self, GroupLimits}, scored_int4::{
    Int4ScoredCohortBatchLimits, Int4ScoredCohortBatchStats, NativeInt4ScoredCohortBatch,
    memory::{Int4ScoredCohortMemoryLimits, Int4ScoredCohortMemoryRequirement},
}};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ScoredCohortHostLimits {
    /// Bounded input window, including rejected/malformed documents.
    pub max_documents: usize,
    /// Total simultaneous scoring heads across all documents in a window.
    pub max_heads: usize,
}
impl Int4ScoredCohortHostLimits {
    /// Shared with CLI admission even when the runtime feature is disabled.
    pub fn requirements(self, group_width: usize, work: Int4ScoredBatchLimits,
        limits: CorpusLimits) -> Result<Int4ScoredCohortMemoryRequirement, HostedError> {
        limits.native.run.validate()?;
        Int4ScoredCohortMemoryLimits { max_documents: self.max_documents,
            max_heads: self.max_heads, context_tokens: limits.native.context_tokens,
            preparation_reserve_bytes: limits.preparation_reserve_bytes,
            io_reserve_bytes: limits.io_reserve_bytes,
            allocator_reserve_bytes: limits.native.allocator_reserve_bytes,
        }.requirements(group_width, work, limits.transport).map_err(HostedError::BatchSetup)
    }
}

/// Semantic per-document output is unchanged. Physical counters are once per
/// invocation, separate from the original task result and transport summary.
pub struct Int4ScoredCohortBatchResult {
    pub summary: BatchSummary,
    pub cohorts: Int4ScoredCohortBatchStats,
}

impl NlpEngine {
    /// Execute bounded windows through real shared-layer scoring cohorts. The
    /// native head count follows each window's exact plans, including the tail;
    /// no dummy heads or per-document forwards substitute for grouping.
    ///
    /// All queued plans, one lookahead, grouped workspace and maximum native
    /// payload are reserved before input is read. Complete output guards stay
    /// together until the last window result is written and flushed. Flush/EOF
    /// drain the window before the transport acknowledges an epoch boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_int4_scored_cohorts<P, R, W>(&self, model: &ResidentInt4,
        planner: Int4ScoredBatchPlanner<P>, limits: CorpusLimits,
        cohorts: Int4ScoredCohortHostLimits, kernel: PackedInt4Kernel,
        reader: R, writer: W, cancellation: CancellationToken)
        -> Result<Int4ScoredCohortBatchResult, HostedError>
    where P: Int4ScoredTask, R: BufRead + Send + 'static, W: Write + Send + 'static {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(planner.base_execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if planner.group_width() != model.model_identity().group_width()
            || planner.base_execution_identity().task_spec != P::TASK {
            return Err(HostedError::ModelIdentity);
        }
        let work = planner.limits();
        let required = cohorts.requirements(planner.group_width(), work, limits)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(StreamInput { planner: Some(planner), reader, writer }))?;
        // These remain reservations across dynamic cohort allocations. An
        // empty stream has no native allocation to claim as committed memory.
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let native = Int4ScoredCohortBatchLimits {
            max_documents: cohorts.max_documents, max_heads: cohorts.max_heads,
            context_limit: limits.native.context_tokens,
            native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound },
            max_workspace_bytes: required.task_workspace_bytes,
            max_item_output_bytes: limits.transport.max_output_line_bytes as u64,
            max_output_bytes: required.max_output_bytes,
        };
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let mut input = input;
            let native_guards = (kv, scratch);
            checkpoint(control)?;
            let planner = input.value.planner.take().ok_or(HostedError::CompletionMissing)?;
            let admission = CohortAdmission { lease: &lease, model: model.model_identity(),
                context: limits.native.context_tokens, max_heads: cohorts.max_heads,
                kv_bytes: required.native.kv_bytes, max_item_work: work.max_item_work,
                task_spec: P::TASK, output_bytes: limits.transport.max_output_line_bytes as u64 };
            let mut processor = NativeInt4ScoredCohortBatch::new(planner,
                &model.inner.loaded.value, native, kernel, admission).map_err(HostedError::BatchSetup)?;
            let result = grouped::run_ndjson(&mut input.value.reader, &mut input.value.writer,
                &mut processor, limits.transport,
                GroupLimits { max_records: cohorts.max_documents }, control).map_err(HostedError::Batch);
            let statistics = processor.cohort_stats();
            drop(processor);
            drop(input);
            drop(native_guards);
            drop(lease);
            result.map(|summary| Int4ScoredCohortBatchResult { summary, cohorts: statistics })
        })
    }
}

struct CohortAdmission<'a> {
    lease: &'a EngineLease,
    model: &'a Int4ModelIdentity,
    context: usize,
    max_heads: usize,
    kv_bytes: u64,
    max_item_work: Int4Work,
    task_spec: &'static str,
    output_bytes: u64,
}
impl Int4ScoredBatchAdmission for CohortAdmission<'_> {
    type Guard = Pending;
    fn admit(&mut self, request: Int4ScoredAdmission<'_>)
        -> Result<(ExecutionIdentity, Self::Guard), BatchItemFailure> {
        constrained_int4::check_model(request.facts.identity, self.model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        check_cohort_request(self.model.group_width(), self.context, self.max_heads,
            self.kv_bytes, self.max_item_work, self.task_spec, self.output_bytes, &request)?;
        let guard = output_claim(self.lease, request.facts.max_result_bytes, 0).map_err(|error| match error {
            HostedError::Reservation(_) => BatchItemFailure::fatal(BatchCode::Admission),
            _ => BatchItemFailure::fatal(BatchCode::InvalidExecution),
        })?;
        Ok((request.facts.identity.clone(), guard))
    }
}

#[allow(clippy::too_many_arguments)]
fn check_cohort_request(width: usize, context: usize, max_heads: usize, kv_bytes: u64,
    ceiling: Int4Work, task_spec: &str, output_bytes: u64, request: &Int4ScoredAdmission<'_>)
    -> Result<(), BatchItemFailure> {
    let facts = &request.facts;
    constrained_int4::check_profile(facts.identity, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    if !matches!(task_spec, "classify-v1" | "sentiment-v1" | "judge-v1")
        || facts.identity.task_spec != task_spec || facts.group_width != width
        || facts.task_budget.validate().is_err() || facts.head_count == 0
        || facts.head_count > max_heads || facts.required_context == 0 || facts.required_context > context {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let per_head = Int4MemoryRequirement::for_context(context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let shortest = Int4MemoryRequirement::for_context(1, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?.kv_bytes;
    let longest = Int4MemoryRequirement::for_context(facts.required_context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?.kv_bytes;
    let minimum = shortest.checked_mul((facts.head_count - 1) as u64)
        .and_then(|bytes| bytes.checked_add(longest))
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
    let maximum = per_head.kv_bytes.checked_mul(facts.head_count as u64)
        .ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
    // Native planning derives the exact head-context sum. Admission requires
    // that complete simultaneous sum, not the largest reusable serial cache.
    if request.kv_reservation_bytes < minimum || request.kv_reservation_bytes > maximum
        || request.kv_reservation_bytes > kv_bytes
        || request.kv_reservation_bytes > facts.task_budget.max_kv_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let work = facts.model_work;
    if work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || work.forward_positions > ceiling.forward_positions || work.projected_logits > ceiling.projected_logits
        || work.attention_pairs > ceiling.attention_pairs || !work.projections.fits(ceiling.projections) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    if facts.max_result_bytes == 0 || facts.max_result_bytes > facts.task_budget.max_output_bytes
        || facts.max_result_bytes > output_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}

#[cfg(test)] mod tests;
