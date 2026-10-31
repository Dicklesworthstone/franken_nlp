//! Actual native multi-document scoring behind the bounded grouped transport.
use super::*;
use serde::{Deserialize, Serialize};
use crate::{batch::grouped::{GroupedBatchProcessor, GroupedOutput, GroupedRequest, GroupedRow, MAX_GROUP_RECORDS},
    native_engine::{int4::PackedInt4Kernel,
        strict_int4::{Int4MemoryBudget, Int4ModelWeights, StrictInt4Error,
            cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES},
            scoring::cohort::{INT4_SCORING_COHORT_EXECUTION, Int4ScoringCohortBudget,
                required_task_workspace_bytes}}}};

/// Modeled payload limits supplied under an embedding host's live reservation.
/// The processor borrows resident weights; it never creates a runtime or IO.
#[derive(Clone, Copy, Debug)]
pub struct Int4ScoredCohortBatchLimits {
    pub max_documents: usize,
    pub max_heads: usize,
    pub context_limit: usize,
    pub native_memory: Int4MemoryBudget,
    pub max_workspace_bytes: u64,
    pub max_item_output_bytes: u64,
    pub max_output_bytes: u64,
}
impl Int4ScoredCohortBatchLimits {
    pub fn validate(self, width: usize) -> Result<(), BatchFault> {
        if !(1..=MAX_GROUP_RECORDS).contains(&self.max_documents)
            || !(1..=MAX_INT4_SEQUENCES).contains(&self.max_heads)
            || self.max_workspace_bytes == 0 || self.max_item_output_bytes == 0
            || self.max_item_output_bytes > self.max_output_bytes
            || self.max_output_bytes > memory::MAX_COHORT_OUTPUT_BYTES {
            return Err(BatchCode::InvalidLimits.into());
        }
        let contexts = [self.context_limit; MAX_INT4_SEQUENCES];
        let required = Int4MemoryRequirement::for_cohort(&contexts[..self.max_heads], width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        if !memory_fits(required, self.native_memory) { return Err(BatchCode::InvalidLimits.into()); }
        Ok(())
    }
}

/// Observed complete native execution, once per invocation. Reserved ceilings
/// remain in BatchSummary; these counters never purport to measure speed/RSS.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4ScoredCohortBatchStats {
    pub cohorts: u64,
    pub documents: u64,
    pub heads: u64,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub model_work: Int4Work,
}

/// Each window allocates the exact head count under the host's maximum payload
/// reservation. Physical slots never carry over to another document/window.
pub struct NativeInt4ScoredCohortBatch<'w, P: Int4ScoredTask, A: Int4ScoredBatchAdmission> {
    compiler: Int4ScoredBatchPlanner<P>,
    weights: &'w Int4ModelWeights,
    limits: Int4ScoredCohortBatchLimits,
    kernel: PackedInt4Kernel,
    admission: A,
    remaining: Int4Work,
    last_context: Option<BatchRequestContext>,
    failed: bool,
    statistics: Int4ScoredCohortBatchStats,
}
impl<'w, P: Int4ScoredTask, A: Int4ScoredBatchAdmission> NativeInt4ScoredCohortBatch<'w, P, A> {
    pub fn new(compiler: Int4ScoredBatchPlanner<P>, weights: &'w Int4ModelWeights,
        limits: Int4ScoredCohortBatchLimits, kernel: PackedInt4Kernel, admission: A) -> Result<Self, BatchFault> {
        compiler.limits.validate()?; limits.validate(compiler.group_width())?;
        constrained_int4::check_model(compiler.base_execution_identity(), weights.identity())
            .map_err(|_| BatchCode::Admission)?;
        if compiler.group_width() != weights.group_width() { return Err(BatchCode::Admission.into()); }
        let remaining = compiler.limits.max_model_work;
        Ok(Self { compiler, weights, limits, kernel, admission, remaining,
            last_context: None, failed: false, statistics: Int4ScoredCohortBatchStats::default() })
    }
    pub fn remaining_work(&self) -> Int4Work { self.remaining }
    pub fn is_poisoned(&self) -> bool { self.failed }
    pub fn cohort_stats(&self) -> Int4ScoredCohortBatchStats { self.statistics }

    fn execute<C: DecodeStepControl>(&mut self, requests: Vec<GroupedRequest<PreparedInt4Scored<P>>>,
        control: &mut C) -> Result<GroupedOutput<P::Output, Vec<A::Guard>>, BatchFault> {
        validate_contexts(&requests, self.last_context, self.limits.max_documents)?;
        checkpoint(control).map_err(|error| error.fault)?;
        // Guards are declared before all retained native/result allocations and
        // transferred together to transport; no sibling loses its output claim.
        let mut guards = reserve(requests.len())?;
        let mut identities = reserve(requests.len())?;
        let mut contexts = reserve(self.limits.max_heads)?;
        let mut work = Int4Work::default(); let mut output_bytes = 0_u64;
        for request in &requests {
            if !Arc::ptr_eq(&request.prepared.seal, &self.compiler.seal) { return Err(BatchCode::Admission.into()); }
            let plan = &request.prepared.plan;
            let head_contexts = P::cohort_contexts(plan).map_err(|error| error.fault)?;
            let facts = P::facts(plan);
            let kv_bytes = price_plan(&facts, &head_contexts, self.limits).map_err(|error| error.fault)?;
            if contexts.len().checked_add(head_contexts.len()).is_none_or(|n| n > self.limits.max_heads) {
                return Err(BatchCode::Admission.into());
            }
            work = work.checked_add(facts.model_work).map_err(native_error)?;
            output_bytes = output_bytes.checked_add(facts.max_result_bytes).ok_or(BatchCode::OutputLimit)?;
            if output_bytes > self.limits.max_output_bytes { return Err(BatchCode::OutputLimit.into()); }
            let (identity, guard) = self.admission.admit(Int4ScoredAdmission { facts, kv_reservation_bytes: kv_bytes })
                .map_err(|error| error.fault)?;
            // Retain the guard even when a forged final identity is refused.
            guards.push(guard);
            P::verify_identity(plan, &identity).map_err(|error| error.fault)?;
            identities.push(identity); contexts.extend_from_slice(&head_contexts);
        }
        let required = Int4MemoryRequirement::for_cohort(&contexts, self.compiler.group_width()).map_err(native_error)?;
        if !memory_fits(required, self.limits.native_memory) { return Err(BatchCode::Admission.into()); }
        let workspace = required_task_workspace_bytes(work, contexts.len(), self.limits.max_output_bytes)
            .map_err(|_| BatchCode::Admission)?;
        if workspace > self.limits.max_workspace_bytes { return Err(BatchCode::Admission.into()); }
        checkpoint(control).map_err(|error| error.fault)?;
        let mut engine = Int4CohortEngine::new(self.weights, &contexts, self.limits.native_memory)
            .map_err(native_error)?.with_kernel(self.kernel);
        let mut documents = reserve(requests.len())?;
        for (request, identity) in requests.iter().zip(&identities) {
            documents.push((&request.prepared.plan, identity));
        }
        let result = P::execute_document_cohort(&documents, &mut engine,
            Int4ScoringCohortBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes,
                max_output_bytes: self.limits.max_output_bytes }, control).map_err(|error| error.fault)?;
        engine.check_idle().map_err(native_error)?;
        drop(engine);
        if result.schema_version != 1 || result.execution != INT4_SCORING_COHORT_EXECUTION
            || result.output.len() != requests.len() || result.group_steps == 0
            || result.group_steps > work.forward_positions || result.projection_groups == 0 {
            return Err(BatchCode::InvalidExecution.into());
        }
        for (request, output) in requests.iter().zip(&result.output) {
            if !P::completed(&request.prepared.plan, output) { return Err(BatchCode::InvalidExecution.into()); }
        }
        checkpoint(control).map_err(|error| error.fault)?;
        let statistics = add_statistics(self.statistics, requests.len(), contexts.len(),
            result.group_steps, result.projection_groups, work)?;
        let mut rows = reserve(requests.len())?;
        for (request, output) in requests.iter().zip(result.output) {
            rows.push(GroupedRow { request_seq: request.context.request_seq, result: Ok(output) });
        }
        self.last_context = requests.last().map(|request| request.context);
        self.statistics = statistics;
        Ok(GroupedOutput::new(rows, guards))
    }
}
impl<P: Int4ScoredTask, A: Int4ScoredBatchAdmission> GroupedBatchProcessor for NativeInt4ScoredCohortBatch<'_, P, A> {
    type Args = P::Args;
    type Prepared = PreparedInt4Scored<P>;
    type Output = P::Output;
    type Guard = Vec<A::Guard>;
    fn max_group_size(&self) -> usize { self.limits.max_documents }
    fn max_group_weight(&self) -> usize { self.limits.max_heads }
    fn group_weight(&self, prepared: &Self::Prepared) -> Result<usize, BatchItemFailure> {
        if self.failed || !Arc::ptr_eq(&prepared.seal, &self.compiler.seal) {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(prepared.head_count())
    }
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.failed = true;
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        if self.failed { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        let result = prepare_document(&self.compiler, document, self.limits, &mut self.remaining, control);
        if let Err(error) = &result { self.failed |= error.stop; }
        result
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> Result<BatchWork, BatchItemFailure> {
        self.group_weight(prepared)?;
        let work = prepared.model_work();
        Ok(BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits })
    }
    fn execute_group<C: DecodeStepControl>(&mut self, requests: Vec<GroupedRequest<Self::Prepared>>, control: &mut C)
        -> Result<GroupedOutput<Self::Output, Self::Guard>, BatchFault> {
        if self.failed { return Err(BatchCode::InvalidExecution.into()); }
        self.failed = true;
        let result = self.execute(requests, control);
        if result.is_ok() { self.failed = false; }
        result
    }
}

pub(super) fn prepare_document<P: Int4ScoredTask, C: DecodeStepControl>(compiler: &Int4ScoredBatchPlanner<P>,
    document: BatchDocument<P::Args>, limits: Int4ScoredCohortBatchLimits, remaining: &mut Int4Work, control: &mut C)
    -> Result<PreparedInt4Scored<P>, BatchItemFailure> {
    let prepared = compiler.prepare(document, control)?;
    check_plan_bounds(&P::facts(&prepared.plan), limits)?;
    let contexts = P::cohort_contexts(&prepared.plan)?;
    price_plan(&P::facts(&prepared.plan), &contexts, limits)?;
    // All five axes are invocation-wide, including queued/lookahead work later
    // discarded by cancellation or a transport rejection. Subtraction succeeds
    // on every axis before replacing the live allowance; an over-limit document
    // cannot partially spend the allowance of a following valid document.
    *remaining = subtract(*remaining, prepared.model_work())?;
    checkpoint(control)?;
    Ok(prepared)
}

fn price_plan(facts: &Int4ScoredPlanFacts<'_>, contexts: &[usize], limits: Int4ScoredCohortBatchLimits)
    -> Result<u64, BatchItemFailure> {
    check_plan_bounds(facts, limits)?;
    if contexts.len() != facts.head_count || contexts.iter().copied().max() != Some(facts.required_context) {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    let mut kv = 0_u64;
    for &context in contexts {
        if context == 0 || context > limits.context_limit { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        let bytes = Int4MemoryRequirement::for_context(context, facts.group_width)
            .map_err(|_| BatchItemFailure::reject(BatchCode::Admission))?.kv_bytes;
        kv = kv.checked_add(bytes).ok_or_else(|| BatchItemFailure::reject(BatchCode::Admission))?;
    }
    if kv > facts.task_budget.max_kv_bytes || kv > limits.native_memory.max_kv_bytes {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(kv)
}
// Check width BEFORE asking the native layer to materialize contexts: a valid
// serial task can exceed the cohort layer's 64-head ceiling. That document is
// an isolated admission rejection, not corruption of all previously queued work.
fn check_plan_bounds(facts: &Int4ScoredPlanFacts<'_>, limits: Int4ScoredCohortBatchLimits)
    -> Result<(), BatchItemFailure> {
    if facts.head_count == 0 || facts.head_count > limits.max_heads
        || facts.task_budget.validate().is_err() || !nonzero(facts.model_work)
        || facts.required_context == 0 || facts.required_context > limits.context_limit {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    if facts.max_result_bytes == 0 || facts.max_result_bytes > facts.task_budget.max_output_bytes
        || facts.max_result_bytes > limits.max_item_output_bytes {
        return Err(BatchItemFailure::reject(BatchCode::OutputLineLimit));
    }
    Ok(())
}
fn memory_fits(required: Int4MemoryRequirement, budget: Int4MemoryBudget) -> bool {
    required.kv_bytes <= budget.max_kv_bytes && required.rope_bytes <= budget.max_rope_bytes
        && required.scratch_payload_bound <= budget.max_scratch_payload_bytes
}
fn validate_contexts<T>(requests: &[GroupedRequest<T>], previous: Option<BatchRequestContext>, max_documents: usize)
    -> Result<(), BatchFault> {
    if requests.is_empty() || requests.len() > max_documents { return Err(BatchCode::InvalidExecution.into()); }
    let epoch = requests[0].context.epoch;
    let mut previous = previous;
    for request in requests {
        let context = request.context;
        if context.epoch == 0 || context.epoch != epoch || context.request_seq == 0 || context.input_line == 0
            || previous.is_some_and(|last| context.epoch < last.epoch || context.request_seq <= last.request_seq
                || context.input_line <= last.input_line || context.byte_offset <= last.byte_offset) {
            return Err(BatchCode::InvalidExecution.into());
        }
        previous = Some(context);
    }
    Ok(())
}
fn add_statistics(previous: Int4ScoredCohortBatchStats, documents: usize, heads: usize,
    steps: u64, projections: u64, work: Int4Work) -> Result<Int4ScoredCohortBatchStats, BatchFault> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(BatchCode::SequenceOverflow);
    Ok(Int4ScoredCohortBatchStats { cohorts: add(previous.cohorts, 1)?,
        documents: add(previous.documents, documents as u64)?, heads: add(previous.heads, heads as u64)?,
        group_steps: add(previous.group_steps, steps)?, projection_groups: add(previous.projection_groups, projections)?,
        model_work: previous.model_work.checked_add(work).map_err(native_error)? })
}
fn native_error(error: StrictInt4Error) -> BatchFault {
    match error {
        StrictInt4Error::Cancelled(cause) => BatchFault::cancelled(cause),
        StrictInt4Error::Allocation => BatchCode::Allocation.into(),
        StrictInt4Error::Context | StrictInt4Error::Memory => BatchCode::Admission.into(),
        StrictInt4Error::Work => BatchCode::WorkLimit.into(),
        _ => BatchCode::InvalidExecution.into(),
    }
}
fn reserve<T>(count: usize) -> Result<Vec<T>, BatchFault> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| BatchCode::Allocation)?; Ok(values)
}

#[cfg(test)] mod tests;
