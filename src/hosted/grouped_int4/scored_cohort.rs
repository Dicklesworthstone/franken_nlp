//! Process-owned simultaneous INT4 scoring heads on one resident model.
//! Every prompt/cursor/KV branch is admitted together and retained until drain.
use super::*;
use serde::Serialize;
use crate::{execution_identity::ExecutionIdentity,
    native_engine::strict_int4::{Int4Work, cohort::Int4CohortEngine,
        scoring::cohort::{required_task_workspace_bytes, Int4ScoredCohort, Int4ScoringCohortBudget}},
    tasks::{ir::TaskBudget,
        classify::grouped_int4::{Int4ClassificationRun, PreparedInt4Classification},
        sentiment::grouped_int4::{Int4SentimentError, Int4SentimentRun, PreparedInt4Sentiment},
        judge::grouped_int4::{Int4JudgeError, Int4JudgeRun, PreparedInt4Judge}}};

#[derive(Clone, Copy)]
struct Facts { task: TaskBudget, work: Int4Work, heads: usize, group_width: usize, output_bytes: u64 }
struct Input<P> { prepared: P, contexts: Vec<usize> }

// Closed static dispatch keeps caller-supplied scores, receipts and drivers
// outside this native execution boundary.
trait Plan: Send + 'static {
    type Output: Serialize + Send + 'static;
    fn identity(&self) -> &ExecutionIdentity;
    fn facts(&self) -> Facts;
    fn contexts(&self) -> Result<Vec<usize>, HostedError>;
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int4CohortEngine<'_>, budget: Int4ScoringCohortBudget,
        control: &mut C) -> Result<Int4ScoredCohort<Self::Output>, HostedError>;
}
impl Plan for PreparedInt4Classification {
    type Output = Int4ClassificationRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn facts(&self) -> Facts { Facts { task: self.task_budget(), work: self.planned_work(), heads: self.head_count(),
        group_width: self.group_width(), output_bytes: self.task_budget().max_output_bytes } }
    fn contexts(&self) -> Result<Vec<usize>, HostedError> {
        self.cohort_contexts().map_err(|error| classification_error(Int4ClassificationError::Scoring(error)))
    }
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int4CohortEngine<'_>, budget: Int4ScoringCohortBudget,
        control: &mut C) -> Result<Int4ScoredCohort<Self::Output>, HostedError> {
        self.execute_cohort_with_control(self.execution_identity(), engine, budget, control).map_err(classification_error)
    }
}
impl Plan for PreparedInt4Sentiment {
    type Output = Int4SentimentRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn facts(&self) -> Facts { Facts { task: self.task_budget(), work: self.planned_work(), heads: self.head_count(),
        group_width: self.group_width(), output_bytes: self.max_result_bytes() } }
    fn contexts(&self) -> Result<Vec<usize>, HostedError> {
        self.cohort_contexts().map_err(|error| sentiment_error(Int4SentimentError::Scoring(error)))
    }
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int4CohortEngine<'_>, budget: Int4ScoringCohortBudget,
        control: &mut C) -> Result<Int4ScoredCohort<Self::Output>, HostedError> {
        self.execute_cohort_with_control(self.execution_identity(), engine, budget, control).map_err(sentiment_error)
    }
}
impl Plan for PreparedInt4Judge {
    type Output = Int4JudgeRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn facts(&self) -> Facts { Facts { task: self.task_budget(), work: self.planned_work(), heads: self.head_count(),
        group_width: self.group_width(), output_bytes: self.max_result_bytes() } }
    fn contexts(&self) -> Result<Vec<usize>, HostedError> {
        self.cohort_contexts().map_err(|error| judge_error(Int4JudgeError::Scoring(error)))
    }
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int4CohortEngine<'_>, budget: Int4ScoringCohortBudget,
        control: &mut C) -> Result<Int4ScoredCohort<Self::Output>, HostedError> {
        self.execute_cohort_with_control(self.execution_identity(), engine, budget, control).map_err(judge_error)
    }
}

impl NlpEngine {
    /// Execute every classification head in one simultaneous native cohort.
    /// context_tokens is per head; the sealed task cap covers total KV. The
    /// retained preparation reserve is additional to the derived transient one.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_classify_cohort(&self, model: &ResidentInt4, prepared: PreparedInt4Classification,
        native: NativeLimits, preparation_reserve_bytes: u64, max_model_work: Int4Work,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ScoredCohort<Int4ClassificationRun>>, HostedError> {
        self.execute_int4_scored_cohort(model, prepared, native, preparation_reserve_bytes, max_model_work, kernel, cancellation)
    }
    /// Share physical layers across independent sentiment axes with complete
    /// aggregate work admission and no shared probability denominator.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_sentiment_cohort(&self, model: &ResidentInt4, prepared: PreparedInt4Sentiment,
        native: NativeLimits, preparation_reserve_bytes: u64, max_model_work: Int4Work,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ScoredCohort<Int4SentimentRun>>, HostedError> {
        self.execute_int4_scored_cohort(model, prepared, native, preparation_reserve_bytes, max_model_work, kernel, cancellation)
    }
    /// Complete both orders, all criteria or all evidence heads in one call.
    /// No partial judgment, renewed per-head allowance or serial fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_judge_cohort(&self, model: &ResidentInt4, prepared: PreparedInt4Judge,
        native: NativeLimits, preparation_reserve_bytes: u64, max_model_work: Int4Work,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ScoredCohort<Int4JudgeRun>>, HostedError> {
        self.execute_int4_scored_cohort(model, prepared, native, preparation_reserve_bytes, max_model_work, kernel, cancellation)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_int4_scored_cohort<P: Plan>(&self, model: &ResidentInt4, prepared: P, native: NativeLimits,
        preparation: u64, max_work: Int4Work, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ScoredCohort<P::Output>>, HostedError> {
        dispatch::preflight(self, native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let rebound = bind_candidate_model(prepared.identity().clone(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let facts = prepared.facts();
        if &rebound != prepared.identity() || facts.group_width != model.model_identity().group_width() {
            return Err(HostedError::ModelIdentity);
        }
        validate_work(facts, native, preparation, max_work)?;
        let input_bytes = sum(&[preparation, transient_bytes(facts)?])?;
        let lease = self.resources().acquire_lease();
        // Reserve before context vectors, flattened prompts and cursors are
        // constructed. Their charge travels with the complete prepared input.
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?, || {
            let contexts = prepared.contexts()?;
            geometry(facts, native, &contexts)?;
            Ok(Input { prepared, contexts })
        })?;
        let required = geometry(facts, native, &input.value.contexts)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?)?;
        let output = output_claim(&lease, facts.output_bytes, 0)?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, workspace, || Int4CohortEngine::new(&model.inner.loaded.value,
                &input.value.contexts, Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
                    max_rope_bytes: required.rope_bytes, max_scratch_payload_bytes: required.scratch_payload_bound })
                .map(|engine| engine.with_kernel(kernel)).map_err(native_error))?;
            let result = allocate(output, || {
                let result = input.value.prepared.execute(&mut engine.value, Int4ScoringCohortBudget {
                    native: Int4RunBudget::exact(facts.work), max_kv_bytes: required.kv_bytes,
                    max_output_bytes: facts.output_bytes,
                }, control)?;
                engine.value.check_idle().map_err(native_error)?;
                Ok(result)
            })?;
            drop(engine); drop(input); drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn classification_error(error: Int4ClassificationError) -> HostedError {
    HostedError::Int4(Int4HostError::Classification(error))
}
fn sentiment_error(error: Int4SentimentError) -> HostedError { HostedError::Int4(Int4HostError::Sentiment(error)) }
fn judge_error(error: Int4JudgeError) -> HostedError { HostedError::Int4(Int4HostError::Judge(error)) }
fn validate_work(facts: Facts, native: NativeLimits, preparation: u64, cap: Int4Work) -> Result<(), HostedError> {
    native.run.validate()?;
    facts.task.validate().map_err(|_| HostedError::Limits("INT4 scored cohort task budget"))?;
    if preparation == 0 || native.allocator_reserve_bytes == 0 || facts.heads == 0 || facts.heads > 64
        || !(1..=64 * 1024 * 1024).contains(&facts.output_bytes) || facts.output_bytes > facts.task.max_output_bytes {
        return Err(HostedError::Limits("INT4 scored cohort preparation, width or complete output"));
    }
    Int4MemoryRequirement::for_context(native.context_tokens, facts.group_width).map_err(native_error)?;
    let work = facts.work;
    if work.forward_positions == 0 || work.forward_positions > cap.forward_positions
        || work.projected_logits > cap.projected_logits || work.attention_pairs > cap.attention_pairs
        || !work.projections.fits(cap.projections) { return Err(HostedError::Limits("INT4 scored cohort complete work")); }
    Ok(())
}
fn geometry(facts: Facts, native: NativeLimits, contexts: &[usize]) -> Result<Int4MemoryRequirement, HostedError> {
    if contexts.len() != facts.heads || contexts.iter().any(|&context| context == 0 || context > native.context_tokens) {
        return Err(HostedError::Limits("INT4 scored cohort exact physical contexts"));
    }
    let required = Int4MemoryRequirement::for_cohort(contexts, facts.group_width).map_err(native_error)?;
    if required.kv_bytes > facts.task.max_kv_bytes { return Err(HostedError::Limits("INT4 scored cohort simultaneous task KV")); }
    Ok(required)
}
fn transient_bytes(facts: Facts) -> Result<u64, HostedError> {
    required_task_workspace_bytes(facts.work, facts.heads, facts.output_bytes)
        .map_err(|_| HostedError::Limits("INT4 scored cohort prompt/cursor/result workspace"))
}
#[cfg(test)] mod tests;
