//! Real shared-layer chunk cohorts with the original lossless document reducer.
//! Native engines have exactly the current window's independent source rows;
//! the caller retains one peak reservation across every window and finalizer.
use super::*;
use std::cell::RefCell;
use crate::{batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES,
    native_engine::{int4::PackedInt4Kernel,
        constrained_int4::{Int4JsonBudget, cohort::required_workspace_bytes},
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelWeights,
            Int4RunBudget, StrictInt4Error, cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES}}},
    tasks::extract::grouped_int4::cohort::{self as extraction, Int4ExtractCohortRequest,
        Int4ExtractCohortBudget, Int4ExtractCohortRun, INT4_EXTRACT_COHORT_EXECUTION},
    native_engine::constrained_int4::cohort::Int4JsonCohortBudget};

pub const INT4_EXTRACTION_MAP_COHORT_EXECUTION: &str = "grouped-int4-exact-schema-document-cohorts-v1";
pub const COHORT_ENVELOPE_BYTES: u64 = 4096;

/// Declared simultaneous chunk bounds, quotable before document or model IO.
/// Preparation, retained reduction values, output ownership and allocator
/// headroom are separate host reservations. These are payload bounds, not RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractionMapCohortMemoryLimits {
    pub max_chunks: usize,
    pub context_tokens: usize,
    pub max_output_tokens_per_chunk: u64,
    pub max_result_bytes_per_chunk: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractionMapCohortRequirement {
    pub native: Int4MemoryRequirement,
    /// Additional decoder cursors/masks/results beyond native payload.
    pub workspace_bytes: u64,
    /// Complete transient native cohort envelope, not the document result.
    pub max_result_bytes: u64,
}
impl Int4ExtractionMapCohortMemoryLimits {
    pub fn requirements(self, group_width: usize) -> Result<Int4ExtractionMapCohortRequirement, Int4ExtractionMapError> {
        if !(1..=MAX_INT4_SEQUENCES).contains(&self.max_chunks)
            || self.max_output_tokens_per_chunk == 0 || self.max_output_tokens_per_chunk > self.context_tokens as u64
            || self.max_result_bytes_per_chunk == 0 {
            return Err(Int4ExtractionMapError::InvalidLimits);
        }
        let count = self.max_chunks as u64;
        let contexts = [self.context_tokens; MAX_INT4_SEQUENCES];
        let native = Int4MemoryRequirement::for_cohort(&contexts[..self.max_chunks], group_width).map_err(native_error)?;
        let tokens = count.checked_mul(self.max_output_tokens_per_chunk).ok_or(Int4ExtractionMapError::WorkLimit)?;
        let bytes = count.checked_mul(self.max_result_bytes_per_chunk).ok_or(Int4ExtractionMapError::WorkLimit)?;
        requirement(native, self.max_chunks, tokens, bytes)
    }
}

/// Peak authority for the complete invocation. No new memory or work allowance
/// is granted by starting another window, native engine, or reducer level.
#[derive(Clone, Copy, Debug)]
pub struct Int4ExtractionMapCohortBudget {
    pub context_limit: usize,
    pub native_memory: Int4MemoryBudget,
    pub max_workspace_bytes: u64,
    pub max_result_bytes: u64,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Int4ExtractionMapCohortStats {
    pub cohorts: u64,
    pub chunks: u64,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}
#[derive(Serialize)]
pub struct Int4ExtractionMapCohortRun {
    pub schema_version: u32,
    pub execution: &'static str,
    /// Existing semantic result, including unchanged per-chunk native output
    /// and original-coordinate evidence. map_batches records this schedule.
    pub document: Int4ExtractionMapRun,
    pub cohorts: Int4ExtractionMapCohortStats,
}

struct Window {
    contexts: [usize; MAX_INT4_SEQUENCES],
    count: usize,
    requirement: Int4ExtractionMapCohortRequirement,
    work: Int4Work,
}
impl PreparedInt4ExtractionMap<'_> {
    /// Quote the actual componentwise peak across ALL prepared windows,
    /// including a short final window. Row task budgets are never widened.
    pub fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4ExtractionMapCohortRequirement, Int4ExtractionMapError> {
        self.check_cohort_shape()?;
        Int4MemoryRequirement::for_context(context_limit, self.group_width()).map_err(native_error)?;
        let mut peak: Option<Int4ExtractionMapCohortRequirement> = None;
        let mut work = Int4Work::default();
        for plans in self.plans.chunks(self.limits.mapping.reduction.map_batch_chunks) {
            let window = window(plans, self.group_width(), context_limit)?;
            work = add_work(work, window.work).ok_or(Int4ExtractionMapError::WorkLimit)?;
            peak = Some(match peak { None => window.requirement, Some(previous) => maximum(previous, window.requirement) });
        }
        let masks = self.limits.mapping.mask_visits_per_chunk.checked_mul(self.plans.len() as u64)
            .ok_or(Int4ExtractionMapError::WorkLimit)?;
        if work != self.expected.work || !fits(work, self.limits.mapping.max_model_work)
            || masks != self.expected.masks || masks > self.limits.mapping.max_mask_visits {
            return Err(Int4ExtractionMapError::WorkLimit);
        }
        peak.ok_or(Int4ExtractionMapError::EmptySource)
    }

    /// Execute the sealed document through actual native cohorts. The caller
    /// owns complete preparation, peak native/decoder memory, reduction values
    /// and final output storage throughout this synchronous invocation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        budget: Int4ExtractionMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4ExtractionMapCohortRun, Int4ExtractionMapError> {
        step(control)?;
        self.check_cohort_shape()?;
        if admitted.len() != self.plans.len() || weights.group_width() != self.group_width() {
            return Err(Int4ExtractionMapError::Admission);
        }
        // Authenticate every later chunk before the first model forward.
        for identity in admitted {
            step(control)?;
            constrained_int4::check_model(identity, weights.identity()).map_err(Int4ExtractError::from)?;
        }
        self.execute_cohorts_with_driver(admitted, vocabulary, budget,
            Native { weights, vocabulary, budget, kernel }, control)
    }

    fn check_cohort_shape(&self) -> Result<(), Int4ExtractionMapError> {
        if !(1..=MAX_INT4_SEQUENCES).contains(&self.limits.mapping.reduction.map_batch_chunks)
            || self.plans.is_empty() || self.plans.len() != self.expected.chunks
            || self.plans.len() != self.chunks.chunks().len() {
            return Err(Int4ExtractionMapError::InvalidLimits);
        }
        Ok(())
    }

    // Private deterministic injection only. Public execution requires real
    // immutable weights and performs the complete actual-model binding above.
    fn execute_cohorts_with_driver<D: CohortDriver, C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        vocabulary: &ExtractionVocabulary, budget: Int4ExtractionMapCohortBudget, driver: D, control: &mut C)
        -> Result<Int4ExtractionMapCohortRun, Int4ExtractionMapError> {
        step(control)?;
        let required = self.cohort_requirements(budget.context_limit)?;
        check_budget(required, budget)?;
        if admitted.len() != self.plans.len() { return Err(Int4ExtractionMapError::Admission); }
        let masks = ExtractionMaskBudget { per_mask: self.limits.mapping.mask_limits,
            max_visits_per_item: self.limits.mapping.mask_visits_per_chunk,
            max_visits_per_run: self.limits.mapping.max_mask_visits };
        for ((plan, identity), chunk) in self.plans.iter().zip(admitted).zip(self.chunks.chunks()) {
            step(control)?;
            plan.verify_identity(identity)?;
            if plan.source().text() != chunk.text() { return Err(Int4ExtractionMapError::Admission); }
            let context = usize::try_from(plan.planned_work().forward_positions).map_err(|_| Int4ExtractionMapError::Admission)?;
            extraction::preflight(&Int4ExtractCohortRequest { prepared: &plan.plan, admitted_identity: identity,
                budget: row_budget(plan, context, masks)? }, vocabulary)?;
        }
        step(control)?;
        let control = RefCell::new(control);
        let mut task = Executor { plans: &self.plans, admitted, driver, control: &control,
            next: 0, masks: 0, remaining: self.limits.verification, mask_budget: masks,
            stats: Int4ExtractionMapCohortStats::default() };
        let mut cancelled = None;
        let mapped = mapreduce::execute(&self.chunks, &mut task, self.limits.mapping.reduction, || {
            match step(&mut **control.borrow_mut()) {
                Ok(()) => Ok(()), Err(error) => { cancelled = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = cancelled { return Err(error.into()); }
        let mapped = mapped.map_err(Int4ExtractionMapError::Reduction)?;
        step(&mut **control.borrow_mut())?;
        if task.next != self.expected.chunks || task.stats.chunks != self.expected.chunks as u64
            || task.stats.cohorts != mapped.map_batches() as u64 || task.stats.planned_work != self.expected.work {
            return Err(ExtractError::InvalidResult.into());
        }
        let document = execution::complete(self.expected, self.limits.verification, task.remaining,
            task.stats.model_work, task.masks, mapped, self.max_result_bytes())?;
        let run = Int4ExtractionMapCohortRun { schema_version: 1, execution: INT4_EXTRACTION_MAP_COHORT_EXECUTION,
            document, cohorts: task.stats };
        let cap = self.max_result_bytes().checked_add(COHORT_ENVELOPE_BYTES).ok_or(Int4ExtractionMapError::WorkLimit)?;
        check_size(&run, cap)?;
        step(&mut **control.borrow_mut())?;
        Ok(run)
    }
}

fn requirement(native: Int4MemoryRequirement, count: usize, tokens: u64, bytes: u64)
    -> Result<Int4ExtractionMapCohortRequirement, Int4ExtractionMapError> {
    let max_result_bytes = bytes.checked_add(COHORT_ENVELOPE_BYTES).and_then(|n| n.checked_add(count as u64))
        .filter(|&n| n <= MAX_COHORT_RESULT_BYTES).ok_or(Int4ExtractionMapError::InvalidLimits)?;
    let workspace_bytes = required_workspace_bytes(count, tokens, max_result_bytes)
        .map_err(Int4ExtractError::from)?;
    Ok(Int4ExtractionMapCohortRequirement { native, workspace_bytes, max_result_bytes })
}
fn window(plans: &[PreparedInt4BatchExtraction], width: usize, context_limit: usize)
    -> Result<Window, Int4ExtractionMapError> {
    if plans.is_empty() || plans.len() > MAX_INT4_SEQUENCES { return Err(Int4ExtractionMapError::InvalidLimits); }
    let mut contexts = [0; MAX_INT4_SEQUENCES]; let mut work = Int4Work::default();
    let mut tokens = 0_u64; let mut bytes = 0_u64;
    for (slot, prepared) in plans.iter().enumerate() {
        let plan = &prepared.plan;
        let context = usize::try_from(plan.planned_work().forward_positions).map_err(|_| Int4ExtractionMapError::Admission)?;
        if plan.group_width() != width || context == 0 || context > context_limit || plan.max_result_bytes() == 0 {
            return Err(Int4ExtractionMapError::Admission);
        }
        let native = Int4MemoryRequirement::for_context(context, width).map_err(native_error)?;
        if native.kv_bytes > plan.max_kv_bytes() { return Err(Int4ExtractionMapError::Admission); }
        contexts[slot] = context;
        work = add_work(work, plan.planned_work()).ok_or(Int4ExtractionMapError::WorkLimit)?;
        tokens = tokens.checked_add(plan.options().max_new_tokens as u64).ok_or(Int4ExtractionMapError::WorkLimit)?;
        bytes = bytes.checked_add(plan.max_result_bytes()).ok_or(Int4ExtractionMapError::WorkLimit)?;
    }
    let native = Int4MemoryRequirement::for_cohort(&contexts[..plans.len()], width).map_err(native_error)?;
    Ok(Window { contexts, count: plans.len(), requirement: requirement(native, plans.len(), tokens, bytes)?, work })
}
fn maximum(a: Int4ExtractionMapCohortRequirement, b: Int4ExtractionMapCohortRequirement)
    -> Int4ExtractionMapCohortRequirement {
    Int4ExtractionMapCohortRequirement { native: Int4MemoryRequirement {
        kv_bytes: a.native.kv_bytes.max(b.native.kv_bytes), rope_bytes: a.native.rope_bytes.max(b.native.rope_bytes),
        scratch_payload_bound: a.native.scratch_payload_bound.max(b.native.scratch_payload_bound) },
        workspace_bytes: a.workspace_bytes.max(b.workspace_bytes), max_result_bytes: a.max_result_bytes.max(b.max_result_bytes) }
}
fn check_budget(need: Int4ExtractionMapCohortRequirement, budget: Int4ExtractionMapCohortBudget)
    -> Result<(), Int4ExtractionMapError> {
    if need.native.kv_bytes > budget.native_memory.max_kv_bytes || need.native.rope_bytes > budget.native_memory.max_rope_bytes
        || need.native.scratch_payload_bound > budget.native_memory.max_scratch_payload_bytes
        || need.workspace_bytes > budget.max_workspace_bytes || need.max_result_bytes > budget.max_result_bytes
        || budget.max_result_bytes > MAX_COHORT_RESULT_BYTES { return Err(Int4ExtractionMapError::Admission); }
    Ok(())
}
fn row_budget(plan: &PreparedInt4BatchExtraction, context: usize, masks: ExtractionMaskBudget)
    -> Result<Int4JsonBudget, Int4ExtractError> {
    let mut budget = native_budget(plan, masks);
    budget.json.max_kv_bytes = (context as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(ExtractError::InvalidResult)?;
    Ok(budget)
}
fn native_error(error: StrictInt4Error) -> Int4ExtractionMapError {
    Int4ExtractError::from(Int4JsonError::from(error)).into()
}

// A test-only injected implementation exercises the coordinator and independent
// evidence checks. The public API never accepts a driver or serialized output.
trait CohortDriver {
    fn execute<C: DecodeStepControl>(&mut self, plans: &[PreparedInt4BatchExtraction], admitted: &[ExecutionIdentity],
        masks: ExtractionMaskBudget, control: &mut C) -> Result<Int4ExtractCohortRun, Int4ExtractError>;
}
struct Native<'w, 'v> {
    weights: &'w Int4ModelWeights, vocabulary: &'v ExtractionVocabulary,
    budget: Int4ExtractionMapCohortBudget, kernel: PackedInt4Kernel,
}
impl CohortDriver for Native<'_, '_> {
    fn execute<C: DecodeStepControl>(&mut self, plans: &[PreparedInt4BatchExtraction], admitted: &[ExecutionIdentity],
        masks: ExtractionMaskBudget, control: &mut C) -> Result<Int4ExtractCohortRun, Int4ExtractError> {
        if plans.len() != admitted.len() { return Err(ExtractError::InvalidResult.into()); }
        let window = window(plans, self.weights.group_width(), self.budget.context_limit).map_err(task_error)?;
        check_budget(window.requirement, self.budget).map_err(task_error)?;
        let mut rows = Vec::new(); rows.try_reserve_exact(plans.len()).map_err(|_| ExtractError::AllocationRefused)?;
        for ((plan, identity), &context) in plans.iter().zip(admitted).zip(&window.contexts[..window.count]) {
            rows.push(Int4ExtractCohortRequest { prepared: &plan.plan, admitted_identity: identity,
                budget: row_budget(plan, context, masks)? });
        }
        let mask_cap = masks.max_visits_per_item.checked_mul(plans.len() as u64)
            .filter(|&n| n <= masks.max_visits_per_run).ok_or(ExtractError::OutputBudgetExceeded)?;
        step(control)?;
        let mut engine = Int4CohortEngine::new(self.weights, &window.contexts[..window.count], self.budget.native_memory)
            .map_err(Int4JsonError::from)?.with_kernel(self.kernel);
        let result = extraction::execute(&mut engine, &rows, self.vocabulary, Int4ExtractCohortBudget {
            decode: Int4JsonCohortBudget { native: Int4RunBudget::exact(window.work),
                max_kv_bytes: window.requirement.native.kv_bytes, max_mask_node_visits: mask_cap,
                max_result_bytes: window.requirement.max_result_bytes },
            max_result_bytes: window.requirement.max_result_bytes,
        }, control)?;
        engine.check_idle().map_err(Int4JsonError::from)?;
        Ok(result)
    }
}
fn task_error(error: Int4ExtractionMapError) -> Int4ExtractError {
    match error {
        Int4ExtractionMapError::Extraction(error) => error,
        Int4ExtractionMapError::Allocation => ExtractError::AllocationRefused.into(),
        Int4ExtractionMapError::WorkLimit => ExtractError::OutputBudgetExceeded.into(),
        _ => ExtractError::InvalidResult.into(),
    }
}

struct Executor<'a, 'c, D, C> {
    plans: &'a [PreparedInt4BatchExtraction], admitted: &'a [ExecutionIdentity], driver: D,
    control: &'a RefCell<&'c mut C>, next: usize, masks: u64, remaining: GroundingBudget,
    mask_budget: ExtractionMaskBudget, stats: Int4ExtractionMapCohortStats,
}
impl<D: CohortDriver, C: DecodeStepControl> MapReduceTask for Executor<'_, '_, D, C> {
    type Value = ExtractionMapValue;
    type Error = Int4ExtractError;
    fn policy(&self) -> ReductionPolicy { execution::policy() }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<Self::Value>>, Self::Error> {
        let end = self.next.checked_add(chunks.len()).ok_or(ExtractError::InvalidResult)?;
        if chunks.is_empty() || chunks.iter().enumerate().any(|(index, chunk)| chunk.id() != self.next + index) {
            return Err(ExtractError::InvalidResult.into());
        }
        let plans = self.plans.get(self.next..end).ok_or(ExtractError::InvalidResult)?;
        let admitted = self.admitted.get(self.next..end).ok_or(ExtractError::InvalidResult)?;
        let mut control = self.control.borrow_mut(); step(&mut **control)?;
        let run = self.driver.execute(plans, admitted, self.mask_budget, &mut **control)?;
        validate_group(plans, &run)?;
        let next_stats = accumulate(self.stats, &run)?;
        let mut output = Vec::new(); output.try_reserve_exact(chunks.len()).map_err(|_| ExtractError::AllocationRefused)?;
        for ((chunk, plan), native) in chunks.iter().zip(plans).zip(run.sequences) {
            step(&mut **control)?;
            let masks = native.result.output.mask_node_visit_charge;
            output.push(execution::map_output(chunk, plan, native, self.mask_budget.max_visits_per_item,
                &mut self.remaining, &mut **control)?);
            self.masks = self.masks.checked_add(masks).ok_or(ExtractError::InvalidResult)?;
        }
        self.next = end; self.stats = next_stats;
        step(&mut **control)?;
        Ok(output)
    }
    fn reduce(&mut self, input: ReduceInput<'_, Self::Value>) -> Result<Self::Value, Self::Error> {
        execution::reduce(input, &mut **self.control.borrow_mut())
    }
}
fn validate_group(plans: &[PreparedInt4BatchExtraction], run: &Int4ExtractCohortRun) -> Result<(), Int4ExtractError> {
    if plans.is_empty() || run.schema_version != 1 || run.execution != INT4_EXTRACT_COHORT_EXECUTION
        || run.sequences.len() != plans.len() { return Err(ExtractError::InvalidResult.into()); }
    let mut planned = Int4Work::default(); let mut actual = Int4Work::default();
    let (mut longest, mut longest_head, mut heads) = (0, 0, 0_u64);
    for (plan, row) in plans.iter().zip(&run.sequences) {
        planned = add_work(planned, plan.planned_work()).ok_or(ExtractError::InvalidResult)?;
        actual = add_work(actual, row.model_work).ok_or(ExtractError::InvalidResult)?;
        longest = longest.max(row.model_work.forward_positions);
        let tokens = row.result.output.token_ids.len() as u64;
        longest_head = longest_head.max(tokens); heads = heads.checked_add(tokens).ok_or(ExtractError::InvalidResult)?;
    }
    if planned != run.planned_work || actual != run.model_work || longest != run.group_steps
        || run.projection_groups == 0 || run.projection_groups < longest_head
        || run.projection_groups > heads || run.projection_groups > run.group_steps {
        return Err(ExtractError::InvalidResult.into());
    }
    Ok(())
}
fn accumulate(previous: Int4ExtractionMapCohortStats, run: &Int4ExtractCohortRun)
    -> Result<Int4ExtractionMapCohortStats, Int4ExtractError> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(ExtractError::InvalidResult);
    Ok(Int4ExtractionMapCohortStats { cohorts: add(previous.cohorts, 1)?,
        chunks: add(previous.chunks, run.sequences.len() as u64)?,
        group_steps: add(previous.group_steps, run.group_steps)?,
        projection_groups: add(previous.projection_groups, run.projection_groups)?,
        planned_work: add_work(previous.planned_work, run.planned_work).ok_or(ExtractError::InvalidResult)?,
        model_work: add_work(previous.model_work, run.model_work).ok_or(ExtractError::InvalidResult)? })
}

#[cfg(test)] mod tests;
