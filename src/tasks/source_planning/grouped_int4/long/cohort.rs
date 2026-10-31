//! Native chunk cohorts for complete-document source maps. Every row retains
//! its own source grammar, identity, causal KV and original semantic finalizer.
//! The same independently checked evidence lifting and lossless reducer serve
//! both serial and grouped execution; no partial document escapes on failure.
use super::*;
use crate::{
    batch::extract::grouped_int4::{memory::MAX_COHORT_RESULT_BYTES,
        long::cohort::Int4ExtractionMapCohortMemoryLimits},
    native_engine::{int4::PackedInt4Kernel,
        constrained_int4::cohort::{Int4JsonCohortBudget, required_workspace_bytes},
        strict_int4::{Int4MemoryBudget, Int4ModelWeights, StrictInt4Error,
            cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES}}},
    tasks::{extract::grouped_int4::cohort::Int4ExtractCohortBudget,
        source_planning::grouped_int4::cohort::{self as source_group,
            Int4SourceCohortRequest, Int4SourceCohortRun, INT4_SOURCE_COHORT_EXECUTION}},
};

pub const INT4_SOURCE_MAP_COHORT_EXECUTION: &str = "grouped-int4-source-document-cohorts-v1";
pub const COHORT_ENVELOPE_BYTES: u64 = 4096;

/// Maximum simultaneous chunk payload, quotable before source or model IO.
/// Complete preparation, reduction, output and allocator reserves remain
/// separately owned by the host for the entire invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceMapCohortMemoryLimits {
    pub max_chunks: usize,
    pub context_tokens: usize,
    pub max_output_tokens_per_chunk: u64,
    pub max_result_bytes_per_chunk: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SourceMapCohortRequirement {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    /// Complete transient cohort result, separately from the final document.
    pub max_result_bytes: u64,
}
impl Int4SourceMapCohortMemoryLimits {
    pub fn requirements(self, group_width: usize) -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        // Source tasks share the same bounded JSON decoder and retained typed
        // result allowance. Keep that static quote in one implementation.
        let quote = Int4ExtractionMapCohortMemoryLimits { max_chunks: self.max_chunks,
            context_tokens: self.context_tokens, max_output_tokens_per_chunk: self.max_output_tokens_per_chunk,
            max_result_bytes_per_chunk: self.max_result_bytes_per_chunk }
            .requirements(group_width).map_err(|_| Int4SourceMapError::InvalidLimits)?;
        Ok(Int4SourceMapCohortRequirement { native: quote.native,
            workspace_bytes: quote.workspace_bytes, max_result_bytes: quote.max_result_bytes })
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Int4SourceMapCohortBudget {
    pub context_limit: usize,
    pub native_memory: Int4MemoryBudget,
    pub max_workspace_bytes: u64,
    pub max_result_bytes: u64,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Int4SourceMapCohortStats {
    pub cohorts: u64,
    pub chunks: u64,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}
#[derive(Serialize)]
pub struct Int4SourceMapCohortRun {
    pub schema_version: u32,
    pub execution: &'static str,
    /// Original source-map envelope; map_batches records the selected schedule.
    pub document: Int4SourceMapRun,
    pub cohorts: Int4SourceMapCohortStats,
}

struct Window {
    contexts: [usize; MAX_INT4_SEQUENCES],
    count: usize,
    requirement: Int4SourceMapCohortRequirement,
    work: Int4Work,
}
impl PreparedInt4SourceMap<'_> {
    /// Exact componentwise peak over all actual windows, including the short
    /// final window. Each row must fit its original task KV ceiling unchanged.
    pub fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        self.check_cohort_shape()?;
        Int4MemoryRequirement::for_context(context_limit, self.group_width()).map_err(native_error)?;
        let mut peak = None;
        let mut work = Int4Work::default();
        for plans in self.plans.chunks(self.limits.reduction.map_batch_chunks) {
            let window = window(plans, self.group_width(), context_limit)?;
            work = add_work(work, window.work).ok_or(Int4SourceMapError::WorkLimit)?;
            peak = Some(match peak { None => window.requirement, Some(previous) => maximum(previous, window.requirement) });
        }
        let masks = self.limits.mask_visits_per_chunk.checked_mul(self.plans.len() as u64)
            .ok_or(Int4SourceMapError::WorkLimit)?;
        if work != self.expected.work || !fits(work, self.limits.max_model_work)
            || masks != self.expected.masks || masks > self.limits.max_mask_visits {
            return Err(Int4SourceMapError::WorkLimit);
        }
        peak.ok_or(Int4SourceMapError::EmptySource)
    }

    /// Execute every prepared source chunk through real shared decoder layers.
    /// ALL chunk identities, work, controls and memory are checked before the
    /// first engine allocation. The caller owns the quoted peak and complete
    /// document preparation, reduction and output until this call has drained.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4SourceMapCohortRun, Int4SourceMapError> {
        checkpoint(control)?;
        self.check_cohort_shape()?;
        if admitted.len() != self.plans.len() || weights.group_width() != self.group_width() {
            return Err(Int4SourceMapError::Admission);
        }
        for identity in admitted {
            checkpoint(control)?;
            constrained_int4::check_model(identity, weights.identity()).map_err(Int4SourceError::from)?;
        }
        self.execute_cohorts_with_driver(admitted, vocabulary, budget,
            Native { weights, vocabulary, budget, kernel }, control)
    }

    fn check_cohort_shape(&self) -> Result<(), Int4SourceMapError> {
        if !(1..=MAX_INT4_SEQUENCES).contains(&self.limits.reduction.map_batch_chunks)
            || self.plans.is_empty() || self.plans.len() != self.expected.chunks
            || self.plans.len() != self.chunks.chunks().len() {
            return Err(Int4SourceMapError::InvalidLimits);
        }
        Ok(())
    }

    // Private scripted seam only. Public execution requires real immutable
    // model weights and never authenticates caller-supplied result records.
    fn execute_cohorts_with_driver<D: CohortDriver, C: DecodeStepControl>(self,
        admitted: &[ExecutionIdentity], vocabulary: &ExtractionVocabulary,
        budget: Int4SourceMapCohortBudget, driver: D, control: &mut C)
        -> Result<Int4SourceMapCohortRun, Int4SourceMapError> {
        checkpoint(control)?;
        check_budget(self.cohort_requirements(budget.context_limit)?, budget)?;
        if admitted.len() != self.plans.len() { return Err(Int4SourceMapError::Admission); }
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            plan.verify_identity(identity)?;
            if plan.execution_identity().task_spec != self.expected.task_spec {
                return Err(Int4SourceMapError::Admission);
            }
            let context = usize::try_from(plan.planned_work().forward_positions).map_err(|_| Int4SourceMapError::Admission)?;
            source_group::preflight(&Int4SourceCohortRequest { prepared: plan, admitted_identity: identity,
                budget: row_budget(plan, context, self.limits)? }, vocabulary)?;
        }
        checkpoint(control)?;
        let control = RefCell::new(control);
        let mut task = Executor { plans: &self.plans, admitted, driver, control: &control,
            next: 0, masks: 0, remaining: self.limits.verification, limits: self.limits,
            stats: Int4SourceMapCohortStats::default() };
        let mut cancelled = None;
        let mapped = mapreduce::execute(&self.chunks, &mut task, self.limits.reduction, || {
            match checkpoint(&mut **control.borrow_mut()) {
                Ok(()) => Ok(()), Err(error) => { cancelled = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = cancelled { return Err(error.into()); }
        let mapped = mapped.map_err(Int4SourceMapError::Reduction)?;
        checkpoint(&mut **control.borrow_mut())?;
        if task.next != self.expected.chunks || task.stats.chunks != self.expected.chunks as u64
            || task.stats.cohorts != mapped.map_batches() as u64 || task.stats.planned_work != self.expected.work {
            return Err(Int4SourceError::InvalidResult.into());
        }
        let document = execution::complete(self.expected, self.limits.verification, task.remaining,
            task.stats.model_work, task.masks, mapped, self.max_result_bytes())?;
        let result = Int4SourceMapCohortRun { schema_version: 1, execution: INT4_SOURCE_MAP_COHORT_EXECUTION,
            document, cohorts: task.stats };
        let cap = self.max_result_bytes().checked_add(COHORT_ENVELOPE_BYTES).ok_or(Int4SourceMapError::WorkLimit)?;
        check_size(&result, cap).map_err(Int4SourceError::from)?;
        checkpoint(&mut **control.borrow_mut())?;
        Ok(result)
    }
}

fn requirement(native: Int4MemoryRequirement, count: usize, tokens: u64, bytes: u64)
    -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
    let max_result_bytes = bytes.checked_add(COHORT_ENVELOPE_BYTES).and_then(|n| n.checked_add(count as u64))
        .filter(|&n| n <= MAX_COHORT_RESULT_BYTES).ok_or(Int4SourceMapError::InvalidLimits)?;
    let workspace_bytes = required_workspace_bytes(count, tokens, max_result_bytes).map_err(Int4SourceError::from)?;
    Ok(Int4SourceMapCohortRequirement { native, workspace_bytes, max_result_bytes })
}
fn window(plans: &[PreparedInt4SourceTask], width: usize, context_limit: usize)
    -> Result<Window, Int4SourceMapError> {
    if plans.is_empty() || plans.len() > MAX_INT4_SEQUENCES { return Err(Int4SourceMapError::InvalidLimits); }
    let mut contexts = [0; MAX_INT4_SEQUENCES]; let mut work = Int4Work::default();
    let mut tokens = 0_u64; let mut bytes = 0_u64;
    for (slot, plan) in plans.iter().enumerate() {
        let context = usize::try_from(plan.planned_work().forward_positions).map_err(|_| Int4SourceMapError::Admission)?;
        if plan.group_width() != width || context == 0 || context > context_limit || plan.max_result_bytes() == 0 {
            return Err(Int4SourceMapError::Admission);
        }
        let native = Int4MemoryRequirement::for_context(context, width).map_err(native_error)?;
        if native.kv_bytes > plan.max_kv_bytes() || native.kv_bytes > plan.task_budget().max_kv_bytes {
            return Err(Int4SourceMapError::Admission);
        }
        contexts[slot] = context;
        work = add_work(work, plan.planned_work()).ok_or(Int4SourceMapError::WorkLimit)?;
        tokens = tokens.checked_add(plan.extraction.options().max_new_tokens as u64).ok_or(Int4SourceMapError::WorkLimit)?;
        bytes = bytes.checked_add(plan.max_result_bytes()).ok_or(Int4SourceMapError::WorkLimit)?;
    }
    let native = Int4MemoryRequirement::for_cohort(&contexts[..plans.len()], width).map_err(native_error)?;
    Ok(Window { contexts, count: plans.len(), requirement: requirement(native, plans.len(), tokens, bytes)?, work })
}
fn maximum(a: Int4SourceMapCohortRequirement, b: Int4SourceMapCohortRequirement) -> Int4SourceMapCohortRequirement {
    Int4SourceMapCohortRequirement { native: Int4MemoryRequirement {
        kv_bytes: a.native.kv_bytes.max(b.native.kv_bytes), rope_bytes: a.native.rope_bytes.max(b.native.rope_bytes),
        scratch_payload_bound: a.native.scratch_payload_bound.max(b.native.scratch_payload_bound) },
        workspace_bytes: a.workspace_bytes.max(b.workspace_bytes), max_result_bytes: a.max_result_bytes.max(b.max_result_bytes) }
}
fn check_budget(need: Int4SourceMapCohortRequirement, budget: Int4SourceMapCohortBudget) -> Result<(), Int4SourceMapError> {
    if need.native.kv_bytes > budget.native_memory.max_kv_bytes || need.native.rope_bytes > budget.native_memory.max_rope_bytes
        || need.native.scratch_payload_bound > budget.native_memory.max_scratch_payload_bytes
        || need.workspace_bytes > budget.max_workspace_bytes || need.max_result_bytes > budget.max_result_bytes
        || budget.max_result_bytes > MAX_COHORT_RESULT_BYTES { return Err(Int4SourceMapError::Admission); }
    Ok(())
}
fn row_budget(plan: &PreparedInt4SourceTask, context: usize, limits: Int4SourceMapLimits)
    -> Result<Int4JsonBudget, Int4SourceError> {
    let work = plan.planned_work();
    let native = Int4MemoryRequirement::for_context(context, plan.group_width()).map_err(Int4JsonError::from)?;
    Ok(Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: native.kv_bytes, max_total_mask_node_visits: limits.mask_visits_per_chunk,
        mask_limits: limits.mask_limits,
    } })
}
fn native_error(error: StrictInt4Error) -> Int4SourceMapError { Int4SourceError::from(Int4JsonError::from(error)).into() }
fn task_error(error: Int4SourceMapError) -> Int4SourceError {
    match error { Int4SourceMapError::Source(error) => error,
        Int4SourceMapError::Allocation => SourcePlanningError::AllocationRefused.into(),
        Int4SourceMapError::WorkLimit => ExtractError::OutputBudgetExceeded.into(),
        _ => Int4SourceError::InvalidResult }
}

trait CohortDriver {
    fn execute<C: DecodeStepControl>(&mut self, plans: &[PreparedInt4SourceTask], admitted: &[ExecutionIdentity],
        limits: Int4SourceMapLimits, control: &mut C) -> Result<Int4SourceCohortRun, Int4SourceError>;
}
struct Native<'w, 'v> {
    weights: &'w Int4ModelWeights, vocabulary: &'v ExtractionVocabulary,
    budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel,
}
impl CohortDriver for Native<'_, '_> {
    fn execute<C: DecodeStepControl>(&mut self, plans: &[PreparedInt4SourceTask], admitted: &[ExecutionIdentity],
        limits: Int4SourceMapLimits, control: &mut C) -> Result<Int4SourceCohortRun, Int4SourceError> {
        if plans.len() != admitted.len() { return Err(Int4SourceError::InvalidResult); }
        let window = window(plans, self.weights.group_width(), self.budget.context_limit).map_err(task_error)?;
        check_budget(window.requirement, self.budget).map_err(task_error)?;
        let mut rows = Vec::new(); rows.try_reserve_exact(plans.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
        for ((plan, identity), &context) in plans.iter().zip(admitted).zip(&window.contexts[..window.count]) {
            rows.push(Int4SourceCohortRequest { prepared: plan, admitted_identity: identity,
                budget: row_budget(plan, context, limits)? });
        }
        let masks = limits.mask_visits_per_chunk.checked_mul(plans.len() as u64)
            .filter(|&n| n <= limits.max_mask_visits).ok_or(ExtractError::OutputBudgetExceeded)?;
        checkpoint(control)?;
        let mut engine = Int4CohortEngine::new(self.weights, &window.contexts[..window.count], self.budget.native_memory)
            .map_err(Int4JsonError::from)?.with_kernel(self.kernel);
        let result = source_group::execute(&mut engine, &rows, self.vocabulary, Int4ExtractCohortBudget {
            decode: Int4JsonCohortBudget { native: Int4RunBudget::exact(window.work),
                max_kv_bytes: window.requirement.native.kv_bytes, max_mask_node_visits: masks,
                max_result_bytes: window.requirement.max_result_bytes },
            max_result_bytes: window.requirement.max_result_bytes,
        }, control)?;
        engine.check_idle().map_err(Int4JsonError::from)?;
        Ok(result)
    }
}

struct Executor<'a, 'c, D, C> {
    plans: &'a [PreparedInt4SourceTask], admitted: &'a [ExecutionIdentity], driver: D,
    control: &'a RefCell<&'c mut C>, next: usize, masks: u64, remaining: GroundingBudget,
    limits: Int4SourceMapLimits, stats: Int4SourceMapCohortStats,
}
impl<D: CohortDriver, C: DecodeStepControl> MapReduceTask for Executor<'_, '_, D, C> {
    type Value = SourceMapValue;
    type Error = Int4SourceError;
    fn policy(&self) -> ReductionPolicy { execution::policy() }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<Self::Value>>, Self::Error> {
        let end = self.next.checked_add(chunks.len()).ok_or(Int4SourceError::InvalidResult)?;
        if chunks.is_empty() || chunks.iter().enumerate().any(|(index, chunk)| chunk.id() != self.next + index) {
            return Err(Int4SourceError::InvalidResult);
        }
        let plans = self.plans.get(self.next..end).ok_or(Int4SourceError::InvalidResult)?;
        let admitted = self.admitted.get(self.next..end).ok_or(Int4SourceError::InvalidResult)?;
        let mut control = self.control.borrow_mut(); checkpoint(&mut **control)?;
        let run = self.driver.execute(plans, admitted, self.limits, &mut **control)?;
        validate_group(plans, &run)?;
        let stats = accumulate(self.stats, &run)?;
        let mut outputs = Vec::new(); outputs.try_reserve_exact(chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
        for ((chunk, plan), native) in chunks.iter().zip(plans).zip(run.sequences) {
            checkpoint(&mut **control)?;
            let masks = execution::mask_charge(&native.result);
            outputs.push(execution::map_output(chunk, plan, native, self.limits.mask_visits_per_chunk,
                &mut self.remaining, &mut || checkpoint(&mut **control))?);
            self.masks = self.masks.checked_add(masks).ok_or(Int4SourceError::InvalidResult)?;
        }
        self.next = end; self.stats = stats;
        checkpoint(&mut **control)?;
        Ok(outputs)
    }
    fn reduce(&mut self, input: ReduceInput<'_, Self::Value>) -> Result<Self::Value, Self::Error> {
        execution::reduce(input, &mut || checkpoint(&mut **self.control.borrow_mut()))
    }
}
fn validate_group(plans: &[PreparedInt4SourceTask], run: &Int4SourceCohortRun) -> Result<(), Int4SourceError> {
    if plans.is_empty() || run.schema_version != 1 || run.execution != INT4_SOURCE_COHORT_EXECUTION
        || run.sequences.len() != plans.len() { return Err(Int4SourceError::InvalidResult); }
    let mut planned = Int4Work::default(); let mut actual = Int4Work::default();
    let (mut longest, mut longest_head, mut heads) = (0, 0, 0_u64);
    for (plan, row) in plans.iter().zip(&run.sequences) {
        planned = add_work(planned, plan.planned_work()).ok_or(Int4SourceError::InvalidResult)?;
        actual = add_work(actual, row.model_work).ok_or(Int4SourceError::InvalidResult)?;
        longest = longest.max(row.model_work.forward_positions);
        let tokens = match &row.result {
            SourceTaskResult::Ner(r) => r.generated_token_ids.len(),
            SourceTaskResult::Keyphrases(r) => r.generated_token_ids.len(),
            SourceTaskResult::Summarize(r) => r.generated_token_ids.len(),
            SourceTaskResult::Answer(_) => return Err(Int4SourceError::InvalidResult),
        } as u64;
        longest_head = longest_head.max(tokens); heads = heads.checked_add(tokens).ok_or(Int4SourceError::InvalidResult)?;
    }
    if planned != run.planned_work || actual != run.model_work || longest != run.group_steps
        || run.projection_groups == 0 || run.projection_groups < longest_head
        || run.projection_groups > heads || run.projection_groups > run.group_steps {
        return Err(Int4SourceError::InvalidResult);
    }
    Ok(())
}
fn accumulate(previous: Int4SourceMapCohortStats, run: &Int4SourceCohortRun)
    -> Result<Int4SourceMapCohortStats, Int4SourceError> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(Int4SourceError::InvalidResult);
    Ok(Int4SourceMapCohortStats { cohorts: add(previous.cohorts, 1)?, chunks: add(previous.chunks, run.sequences.len() as u64)?,
        group_steps: add(previous.group_steps, run.group_steps)?, projection_groups: add(previous.projection_groups, run.projection_groups)?,
        planned_work: add_work(previous.planned_work, run.planned_work).ok_or(Int4SourceError::InvalidResult)?,
        model_work: add_work(previous.model_work, run.model_work).ok_or(Int4SourceError::InvalidResult)? })
}

#[cfg(test)] mod tests;
