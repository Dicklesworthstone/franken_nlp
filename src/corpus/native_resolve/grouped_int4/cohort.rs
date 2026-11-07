//! Complete candidate pairs in bounded native scoring windows.
//!
//! Both presentation orders stay together. All future windows are checked
//! before the first native forward; only one window's metadata, causal state
//! and temporary scores are retained at a time. Graph semantics are unchanged.
use super::*;
use crate::native_engine::{int4::PackedInt4Kernel,
    strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelWeights,
        cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES},
        scoring::cohort::{self as native, CompiledRequest, Int4CandidateCohortRun,
            Int4ScoredCohort, Int4ScoringCohortBudget}}};

pub const INT4_RESOLUTION_COHORT_EXECUTION: &str = "grouped-int4-resolution-pair-cohorts-v1";
pub const MAX_INT4_RESOLUTION_COHORT_PAIRS: usize = MAX_INT4_SEQUENCES / 2;
pub const INT4_RESOLUTION_COHORT_WRAPPER_BYTES: u64 = 4096;
const MAX_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
// A head has exactly three fixed label IDs, finite scalar scores, fixed
// score-space strings, five u64 work counters and its versioned envelope.
// This bound contains no source-derived strings. Native serialization still
// checks the actual byte length before the numerical session can complete.
const HEAD_RESULT_BYTES: u64 = 16 * 1024;

/// These are physical execution limits, not a larger logical task budget.
/// `memory.max_kv_bytes` prices the sum of simultaneous head capacities;
/// each original head independently retains `TaskBudget.max_kv_bytes`.
#[derive(Clone, Copy, Debug)]
pub struct Int4ResolutionCohortBudget {
    pub native: Int4RunBudget,
    pub memory: Int4MemoryBudget,
    pub max_task_workspace_bytes: u64,
    /// Bounds each temporary native window and the final physical wrapper.
    /// The existing semantic result is still bounded by its original plan.
    pub max_output_bytes: u64,
}

/// A pure declared peak quote, usable before private population IO. Existing
/// graph/prompt preparation, frozen pair scores and original input ownership
/// remain separately admitted by the embedding host.
#[derive(Clone, Copy, Debug)]
pub struct Int4ResolutionCohortMemoryLimits {
    /// Maximum pairs in one physical window, not the snapshot's pair count.
    pub max_pairs: usize,
    /// Capacity of each original scoring head, never the sum of head lengths.
    pub context_tokens: usize,
    pub max_model_work: Int4Work,
    /// The existing complete semantic resolution output ceiling.
    pub max_result_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ResolutionCohortRequirements {
    pub native: Int4MemoryRequirement,
    /// Includes bounded requests, window metadata, cursors, native results,
    /// canonical serialization and temporary bidirectional score staging.
    pub workspace_bytes: u64,
    /// Peak output allowance: the larger of a native window and the complete
    /// unchanged semantic result plus the fixed physical wrapper.
    pub max_result_bytes: u64,
}
impl Int4ResolutionCohortMemoryLimits {
    pub fn requirements(self, group_width: usize) -> Result<Int4ResolutionCohortRequirements, Int4ResolveError> {
        check_pairs(self.max_pairs)?;
        if !(1..=MAX_OUTPUT_BYTES).contains(&self.max_result_bytes) {
            return Err(Int4ResolveError::WorkBudget);
        }
        let heads = self.max_pairs.checked_mul(2).ok_or(Int4ResolveError::WorkBudget)?;
        // Fixed at no more than 64 words: quotation itself needs no heap.
        let contexts = [self.context_tokens; MAX_INT4_SEQUENCES];
        let native = Int4MemoryRequirement::for_cohort(&contexts[..heads], group_width)?;
        let window_bytes = (heads as u64).checked_mul(HEAD_RESULT_BYTES)
            .and_then(|n| n.checked_add(INT4_RESOLUTION_COHORT_WRAPPER_BYTES))
            .ok_or(Int4ResolveError::WorkBudget)?;
        let max_result_bytes = self.max_result_bytes.checked_add(INT4_RESOLUTION_COHORT_WRAPPER_BYTES)
            .ok_or(Int4ResolveError::WorkBudget)?.max(window_bytes);
        // Each three-label head needs prompt_len + 3 forwards while its
        // simultaneous context is prompt_len + 1. Work paid by the snapshot
        // and the physical head/context count both bound staging memory.
        let physical_positions = (self.context_tokens as u64).checked_add(2)
            .and_then(|n| n.checked_mul(heads as u64)).ok_or(Int4ResolveError::WorkBudget)?;
        let mut workspace_work = self.max_model_work;
        workspace_work.forward_positions = workspace_work.forward_positions.min(physical_positions);
        let workspace_bytes = native::required_task_workspace_bytes(workspace_work, heads,
            max_result_bytes.min(MAX_OUTPUT_BYTES))?;
        Ok(Int4ResolutionCohortRequirements { native, workspace_bytes, max_result_bytes })
    }
}

/// Native grouping statistics are separate from the exact serial semantic
/// result. No model execution is invented for a graph with zero pairs.
#[derive(Serialize)]
pub struct Int4ResolutionCohortRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub pair_cohorts: usize,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub output: Int4ResolutionRun,
}

impl PreparedInt4Resolution<'_, '_, '_> {
    /// Model-bound all-window admission without allocating a native engine.
    /// In particular, a larger late pair cannot run after a cheap first
    /// window if its context, memory, workspace or identity would refuse.
    pub fn preflight_cohorts_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        model: &Int4ModelIdentity, cohort_pairs: usize, budget: Int4ResolutionCohortBudget,
        control: &mut C) -> Result<(), Int4ResolveError> {
        self.check_cohort_model(admitted, model, control)?;
        self.preflight_cohort_windows(admitted, cohort_pairs, budget, control)
    }

    /// One immutable complete snapshot, with exact native execution in each
    /// explicitly sized window. There is no width reduction, serial fallback,
    /// per-window budget renewal or partial graph publication.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, kernel: PackedInt4Kernel, cohort_pairs: usize,
        budget: Int4ResolutionCohortBudget, control: &mut C)
        -> Result<Int4ResolutionCohortRun, Int4ResolveError> {
        self.check_cohort_model(admitted, weights.identity(), control)?;
        let planner = self.inner.planner;
        let task = self.task_budget();
        let width = self.group_width;
        let expected = self.work;
        self.collect_cohorts_with_control(admitted, cohort_pairs, budget, control,
            |pairs, window, progress, control| {
                resolve::checkpoint(control)?;
                let requests = compiled_requests(pairs, &window.schedules, planner, task)?;
                let mut engine = Int4CohortEngine::new(weights, &window.contexts, budget.memory)?.with_kernel(kernel);
                let run = native::execute_compiled_with(&mut engine, &requests, window.budget(budget), control,
                    |run, control| finish_window(run, window, progress, width, planner.eos, expected,
                        budget.max_output_bytes.min(MAX_OUTPUT_BYTES), control))?;
                engine.check_idle()?;
                drop(engine);
                resolve::checkpoint(control)?;
                Ok(run)
            })
    }

    fn check_cohort_model<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        model: &Int4ModelIdentity, control: &mut C) -> Result<(), Int4ResolveError> {
        resolve::checkpoint(control)?;
        self.verify_admitted(admitted, control)?;
        check_model(&self.identity, model)?;
        if model.group_width() != self.group_width { return Err(Int4ResolveError::Identity); }
        for identity in admitted { resolve::checkpoint(control)?; check_model(identity, model)?; }
        Ok(())
    }

    // Kept separate from the actual-model check solely for model-free source
    // regressions. Every public preflight/execution authenticates real weights.
    fn preflight_cohort_windows<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        cohort_pairs: usize, budget: Int4ResolutionCohortBudget, control: &mut C)
        -> Result<(), Int4ResolveError> {
        resolve::checkpoint(control)?;
        check_pairs(cohort_pairs)?; check_budget_shape(budget)?;
        self.verify_admitted(admitted, control)?;
        check_budget(self.work, budget.native)?;
        let mut total = Int4Work::default();
        // Do not retain O(snapshot_pairs) descriptors. One bounded window is
        // checked and dropped; execution later rebuilds only this cheap layout.
        for (pairs, schedules) in self.inner.pairs.chunks(cohort_pairs).zip(self.schedules.chunks(cohort_pairs)) {
            let window = Window::checked(pairs, schedules, self.inner.planner, self.task_budget(),
                self.group_width, budget, control)?;
            total = total.checked_add(window.work)?;
        }
        if total != self.work { return Err(Int4ResolveError::Accounting); }
        resolve::checkpoint(control)?;
        Ok(())
    }

    // Private numerical handoff used by real native execution and explicitly
    // synthetic fixtures. Tickets, complete-head validation and clustering
    // stay in the same implementation for serial and cohort execution.
    fn collect_cohorts_with_control<C, F>(self, admitted: &[ExecutionIdentity], cohort_pairs: usize,
        budget: Int4ResolutionCohortBudget, control: &mut C, mut execute: F)
        -> Result<Int4ResolutionCohortRun, Int4ResolveError>
    where C: DecodeStepControl,
        F: FnMut(&[NativePair<'_, '_>], &Window, &mut ResolutionProgress, &mut C)
            -> Result<Int4ScoredCohort<Vec<BidirectionalScores>>, Int4ResolveError> {
        self.preflight_cohort_windows(admitted, cohort_pairs, budget, control)?;
        let max_result_bytes = self.max_result_bytes();
        let expected_heads = self.head_count();
        let mut progress = ResolutionProgress::new(self.pair_count())?;
        let task = self.task_budget();
        let mut pairs = self.inner.pairs.into_iter();
        let mut schedules = self.schedules.into_iter();
        let (mut pair_cohorts, mut group_steps, mut projection_groups) = (0_usize, 0_u64, 0_u64);
        while pairs.len() != 0 {
            resolve::checkpoint(control)?;
            let count = pairs.len().min(cohort_pairs);
            let mut current_pairs = resolve::reserved(count)?;
            let mut current_schedules = resolve::reserved(count)?;
            for _ in 0..count {
                current_pairs.push(pairs.next().ok_or(Int4ResolveError::Accounting)?);
                current_schedules.push(schedules.next().ok_or(Int4ResolveError::Accounting)?);
            }
            let window = Window::checked(&current_pairs, &current_schedules, self.inner.planner,
                task, self.group_width, budget, control)?;
            let before = (progress.work, progress.rewound, progress.heads);
            let result = execute(&current_pairs, &window, &mut progress, control)?;
            resolve::checkpoint(control)?;
            if result.schema_version != 1 || result.execution != native::INT4_SCORING_COHORT_EXECUTION
                || result.output.len() != count
                || progress.work != before.0.checked_add(window.work)?
                || before.1.checked_add((count as u64) * 4) != Some(progress.rewound)
                || before.2.checked_add(count * 2) != Some(progress.heads) {
                return Err(Int4ResolveError::Accounting);
            }
            window.check_statistics(result.group_steps, result.projection_groups)?;
            for (pair, scores) in current_pairs.into_iter().zip(result.output) {
                resolve::checkpoint(control)?;
                progress.scores.push(pair.ticket.finish(scores));
            }
            pair_cohorts = pair_cohorts.checked_add(1).ok_or(Int4ResolveError::Accounting)?;
            group_steps = group_steps.checked_add(result.group_steps).ok_or(Int4ResolveError::Accounting)?;
            projection_groups = projection_groups.checked_add(result.projection_groups).ok_or(Int4ResolveError::Accounting)?;
        }
        if schedules.len() != 0 { return Err(Int4ResolveError::Accounting); }
        let output = progress.finish(self.inner.plan, self.group_width, self.work,
            expected_heads, max_result_bytes as usize, control)?;
        let result = Int4ResolutionCohortRun { schema_version: 1, execution: INT4_RESOLUTION_COHORT_EXECUTION,
            pair_cohorts, group_steps, projection_groups, output };
        let cap = max_result_bytes.checked_add(INT4_RESOLUTION_COHORT_WRAPPER_BYTES)
            .ok_or(Int4ResolveError::WorkBudget)?.min(budget.max_output_bytes);
        resolve::check_output(&result, usize::try_from(cap).map_err(|_| Int4ResolveError::WorkBudget)?)?;
        resolve::checkpoint(control)?;
        Ok(result)
    }
}

struct Window {
    contexts: Vec<usize>,
    schedules: Vec<CandidateSchedule>,
    work: Int4Work,
    minimum_group_steps: u64,
    minimum_projection_groups: u64,
    projection_steps: u64,
}
impl Window {
    #[allow(clippy::too_many_arguments)]
    fn checked<C: DecodeStepControl>(pairs: &[NativePair<'_, '_>], schedules: &[[CandidateSchedule; 2]],
        planner: &ResolutionPlanner, task: TaskBudget, width: usize, budget: Int4ResolutionCohortBudget,
        control: &mut C) -> Result<Self, Int4ResolveError> {
        resolve::checkpoint(control)?;
        check_pairs(pairs.len())?;
        if pairs.len() != schedules.len() { return Err(Int4ResolveError::Accounting); }
        let heads = pairs.len().checked_mul(2).ok_or(Int4ResolveError::WorkBudget)?;
        let mut contexts = resolve::reserved(heads)?;
        let mut flat = resolve::reserved(heads)?;
        let mut work = Int4Work::default();
        let (mut minimum_group_steps, mut minimum_projection_groups, mut projection_steps) = (0_u64, 0_u64, 0_u64);
        for pair in schedules {
            for &schedule in pair {
                resolve::checkpoint(control)?;
                if schedule.group_width != width || schedule.scoring.prefix_evaluations != 4
                    || schedule.scoring.scored_edges != 6 || schedule.rewound != 2 {
                    return Err(Int4ResolveError::Accounting);
                }
                contexts.push(schedule.context); flat.push(schedule);
                work = work.checked_add(schedule.model)?;
                minimum_group_steps = minimum_group_steps.max(schedule.model.forward_positions);
                minimum_projection_groups = minimum_projection_groups.max(schedule.scoring.prefix_evaluations as u64);
                projection_steps = projection_steps.checked_add(schedule.scoring.prefix_evaluations as u64)
                    .ok_or(Int4ResolveError::WorkBudget)?;
            }
        }
        let required = Int4MemoryRequirement::for_cohort(&contexts, width)?;
        if required.kv_bytes > budget.memory.max_kv_bytes || required.rope_bytes > budget.memory.max_rope_bytes
            || required.scratch_payload_bound > budget.memory.max_scratch_payload_bytes {
            return Err(Int4ResolveError::WorkBudget);
        }
        let window = Self { contexts, schedules: flat, work, minimum_group_steps,
            minimum_projection_groups, projection_steps };
        let requests = compiled_requests(pairs, &window.schedules, planner, task)?;
        // The same exact geometry authority runs again inside the native
        // session. It checks every per-head task KV cap AND the physical sum.
        if native::check_geometry(&requests, &window.contexts, window.budget(budget))? != work
            || projection_steps.checked_mul(NANBEIGE_VOCAB_SIZE as u64) != Some(work.projected_logits)
            || native::required_task_workspace_bytes(work, heads,
                budget.max_output_bytes.min(MAX_OUTPUT_BYTES))? > budget.max_task_workspace_bytes {
            return Err(Int4ResolveError::WorkBudget);
        }
        resolve::checkpoint(control)?;
        Ok(window)
    }
    fn budget(&self, budget: Int4ResolutionCohortBudget) -> Int4ScoringCohortBudget {
        Int4ScoringCohortBudget { native: Int4RunBudget::exact(self.work),
            max_kv_bytes: budget.memory.max_kv_bytes,
            max_output_bytes: budget.max_output_bytes.min(MAX_OUTPUT_BYTES) }
    }
    fn check_statistics(&self, group_steps: u64, projection_groups: u64) -> Result<(), Int4ResolveError> {
        if group_steps < self.minimum_group_steps || group_steps > self.work.forward_positions
            || projection_groups < self.minimum_projection_groups || projection_groups > self.projection_steps {
            return Err(Int4ResolveError::Accounting);
        }
        Ok(())
    }
}

fn compiled_requests<'a>(pairs: &'a [NativePair<'_, '_>], schedules: &[CandidateSchedule],
    planner: &'a ResolutionPlanner, task: TaskBudget) -> Result<Vec<CompiledRequest<'a>>, Int4ResolveError> {
    if pairs.len().checked_mul(2) != Some(schedules.len()) { return Err(Int4ResolveError::Accounting); }
    let mut requests = resolve::reserved(schedules.len())?;
    for (pair, schedules) in pairs.iter().zip(schedules.chunks_exact(2)) {
        for (prompt, &schedule) in pair.prompts.iter().zip(schedules) {
            requests.push(CompiledRequest { identity: &pair.identity, prompt, scorer: &planner.scorer,
                mode: ScoringMode::FullVocabulary, schedule, max_output_bytes: task.max_output_bytes,
                budget: Int4ScoringBudget { native: Int4RunBudget::exact(schedule.model), max_kv_bytes: task.max_kv_bytes } });
        }
    }
    Ok(requests)
}

#[allow(clippy::too_many_arguments)]
fn finish_window<C: DecodeStepControl>(run: Int4CandidateCohortRun, window: &Window,
    progress: &mut ResolutionProgress, width: usize, eos: u32, expected: Int4Work,
    cap: u64, control: &mut C) -> Result<Int4ScoredCohort<Vec<BidirectionalScores>>, Int4ResolveError> {
    resolve::checkpoint(control)?;
    window.check_statistics(run.group_steps, run.projection_groups)?;
    native::task::finish(run, &window.schedules, window.work, cap, |heads| {
        let mut pairs = resolve::reserved(window.schedules.len() / 2)?;
        for schedules in window.schedules.chunks_exact(2) {
            resolve::checkpoint(control)?;
            let a = heads.next().ok_or(Int4ResolveError::Accounting)?;
            let forward = progress.accept_head(&a, schedules[0], width, eos, expected)?;
            let b = heads.next().ok_or(Int4ResolveError::Accounting)?;
            let reverse = progress.accept_head(&b, schedules[1], width, eos, expected)?;
            pairs.push(BidirectionalScores { forward, reverse });
        }
        resolve::checkpoint(control)?;
        Ok(pairs)
    })
}

fn check_pairs(pairs: usize) -> Result<(), Int4ResolveError> {
    if !(1..=MAX_INT4_RESOLUTION_COHORT_PAIRS).contains(&pairs) { return Err(Int4ResolveError::WorkBudget); }
    Ok(())
}
fn check_budget_shape(budget: Int4ResolutionCohortBudget) -> Result<(), Int4ResolveError> {
    if budget.max_task_workspace_bytes == 0
        || !(1..=MAX_OUTPUT_BYTES + INT4_RESOLUTION_COHORT_WRAPPER_BYTES).contains(&budget.max_output_bytes) {
        return Err(Int4ResolveError::WorkBudget);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
