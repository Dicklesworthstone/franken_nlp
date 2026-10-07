//! Process-owned finite-head cohorts on the existing resident model and runtime.
//! All prompts/cursors/KV coexist; no serial-memory discount or renewed work.
use super::*;
use serde::Serialize;
use crate::{native_engine::{decode::DecodeStepControl, strict_int8::{Int8Work, cohort::Int8CohortEngine,
    scoring::cohort::{Int8ScoredCohort, Int8ScoringCohortBudget}}},
    tasks::sentiment::quantized::{Int8SentimentError, Int8SentimentRun, PreparedInt8Sentiment}};

#[derive(Clone, Copy)]
struct Facts { task: TaskBudget, work: Int8Work, heads: usize, output_bytes: u64 }
struct Input<P> { prepared: P, contexts: Vec<usize> }

// Closed static dispatch only. Callers cannot plug in a fake plan/receipt or
// substitute a metadata-only admission provider for native execution.
trait Plan: Send + 'static {
    type Output: Serialize + Send + 'static;
    fn identity(&self) -> &ExecutionIdentity;
    fn facts(&self) -> Facts;
    fn contexts(&self) -> Result<Vec<usize>, HostedError>;
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int8CohortEngine<'_>, budget: Int8ScoringCohortBudget,
        control: &mut C) -> Result<Int8ScoredCohort<Self::Output>, HostedError>;
}
impl Plan for PreparedInt8Classification {
    type Output = Int8ClassificationRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn facts(&self) -> Facts { Facts { task: self.task_budget(), work: self.planned_work(),
        heads: self.head_count(), output_bytes: self.task_budget().max_output_bytes } }
    fn contexts(&self) -> Result<Vec<usize>, HostedError> {
        self.cohort_contexts().map_err(|e| HostedError::Classification(Int8ClassificationError::Scoring(e)))
    }
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int8CohortEngine<'_>, budget: Int8ScoringCohortBudget,
        control: &mut C) -> Result<Int8ScoredCohort<Self::Output>, HostedError> {
        self.execute_cohort_with_control(self.execution_identity(), engine, budget, control).map_err(HostedError::Classification)
    }
}
impl Plan for PreparedInt8Sentiment {
    type Output = Int8SentimentRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn facts(&self) -> Facts { Facts { task: self.task_budget(), work: self.planned_work(),
        heads: self.head_count(), output_bytes: self.max_result_bytes() } }
    fn contexts(&self) -> Result<Vec<usize>, HostedError> {
        self.cohort_contexts().map_err(|e| HostedError::Sentiment(Int8SentimentError::Scoring(e)))
    }
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int8CohortEngine<'_>, budget: Int8ScoringCohortBudget,
        control: &mut C) -> Result<Int8ScoredCohort<Self::Output>, HostedError> {
        self.execute_cohort_with_control(self.execution_identity(), engine, budget, control).map_err(HostedError::Sentiment)
    }
}
impl Plan for PreparedInt8Judge {
    type Output = Int8JudgeRun;
    fn identity(&self) -> &ExecutionIdentity { self.execution_identity() }
    fn facts(&self) -> Facts { Facts { task: self.task_budget(), work: self.planned_work(),
        heads: self.head_count(), output_bytes: self.max_result_bytes() } }
    fn contexts(&self) -> Result<Vec<usize>, HostedError> {
        self.cohort_contexts().map_err(|e| HostedError::Judge(Int8JudgeError::Scoring(e)))
    }
    fn execute<C: DecodeStepControl>(&self, engine: &mut Int8CohortEngine<'_>, budget: Int8ScoringCohortBudget,
        control: &mut C) -> Result<Int8ScoredCohort<Self::Output>, HostedError> {
        self.execute_cohort_with_control(self.execution_identity(), engine, budget, control).map_err(HostedError::Judge)
    }
}

impl NlpEngine {
    /// All classification heads share real native layers. native.context_tokens
    /// is a per-head ceiling; full simultaneous KV is charged to the sealed
    /// task's single max_kv_bytes. Caller-priced retained preparation is IN
    /// ADDITION to the derived transient cursor/prompt/result reservation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_classify_cohort(&self, model: &ResidentInt8, prepared: PreparedInt8Classification,
        native: NativeLimits, preparation_reserve_bytes: u64, max_model_work: Int8Work, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ScoredCohort<Int8ClassificationRun>>, HostedError> {
        self.execute_scored_cohort(model, prepared, native, preparation_reserve_bytes, max_model_work, cancellation)
    }
    /// Complete independent sentiment axes, one process-owned native invocation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_sentiment_cohort(&self, model: &ResidentInt8, prepared: PreparedInt8Sentiment,
        native: NativeLimits, preparation_reserve_bytes: u64, max_model_work: Int8Work, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ScoredCohort<Int8SentimentRun>>, HostedError> {
        self.execute_scored_cohort(model, prepared, native, preparation_reserve_bytes, max_model_work, cancellation)
    }
    /// Complete pairwise, rubric or full-source/evidence task. No partial head
    /// output, model reload, extra runtime, hidden retry or serial fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_judge_cohort(&self, model: &ResidentInt8, prepared: PreparedInt8Judge,
        native: NativeLimits, preparation_reserve_bytes: u64, max_model_work: Int8Work, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ScoredCohort<Int8JudgeRun>>, HostedError> {
        self.execute_scored_cohort(model, prepared, native, preparation_reserve_bytes, max_model_work, cancellation)
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_scored_cohort<P: Plan>(&self, model: &ResidentInt8, prepared: P,
        native: NativeLimits, preparation: u64, max_work: Int8Work, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ScoredCohort<P::Output>>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.identity())?;
        let facts = prepared.facts();
        validate_work(facts, native, preparation, max_work)?;
        let input_bytes = sum(&[preparation, transient_bytes(facts)?])?;
        let lease = self.resources().acquire_lease();
        // Reserve before even the context vector, flattened prompts or cursor
        // arrays are constructed. Capture storage and its charge as one value.
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
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .cohort_engine(&input.value.contexts, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let result = input.value.prepared.execute(&mut engine.value, Int8ScoringCohortBudget {
                    native: Int8RunBudget::exact(facts.work), max_kv_bytes: required.kv_bytes,
                    max_output_bytes: facts.output_bytes,
                }, control)?;
                engine.value.check_idle().map_err(HostedError::Native)?;
                Ok(result)
            })?;
            drop(engine); drop(input); drop(lease);
            // The final result retains its real reservation through external
            // serialization/write/flush, not just through native completion.
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn validate_work(facts: Facts, native: NativeLimits, preparation: u64, cap: Int8Work) -> Result<(), HostedError> {
    native.run.validate()?;
    facts.task.validate().map_err(|_| HostedError::Limits("scored cohort task budget"))?;
    if preparation == 0 || native.allocator_reserve_bytes == 0 || facts.heads == 0 || facts.heads > 64
        || !(1..=64 * 1024 * 1024).contains(&facts.output_bytes) || facts.output_bytes > facts.task.max_output_bytes {
        return Err(HostedError::Limits("scored cohort preparation, width or complete output"));
    }
    Int8MemoryRequirement::for_context(native.context_tokens).map_err(HostedError::Native)?;
    let work = facts.work;
    if work.forward_positions == 0 || work.forward_positions > cap.forward_positions
        || work.projected_logits > cap.projected_logits || work.attention_pairs > cap.attention_pairs
        || !work.projections.fits(cap.projections) { return Err(HostedError::Limits("scored cohort complete work")); }
    Ok(())
}
fn geometry(facts: Facts, native: NativeLimits, contexts: &[usize]) -> Result<Int8MemoryRequirement, HostedError> {
    if contexts.len() != facts.heads || contexts.iter().any(|&c| c == 0 || c > native.context_tokens) {
        return Err(HostedError::Limits("scored cohort exact physical contexts"));
    }
    let required = Int8MemoryRequirement::for_cohort(contexts).map_err(HostedError::Native)?;
    if required.kv_bytes > facts.task.max_kv_bytes { return Err(HostedError::Limits("scored cohort simultaneous task KV")); }
    Ok(required)
}
fn transient_bytes(facts: Facts) -> Result<u64, HostedError> {
    // Cursor scalar tables, both prefix rails and flattened prompt tokens scale
    // with total distinct-prefix forwards (EOS leaves add at most that many).
    // Native head results and canonical/semantic staging have a separate bound.
    // Retained plans/tokenizers/allocator overhead are still caller-modeled;
    // this additional commitment is not a measured RSS assertion.
    let prompts = facts.work.forward_positions.checked_mul(64).ok_or(HostedError::Limits("scored cursor arithmetic"))?;
    let outputs = facts.output_bytes.checked_mul(8).ok_or(HostedError::Limits("scored staging arithmetic"))?;
    let rows = (facts.heads as u64).checked_mul(65536).ok_or(HostedError::Limits("scored row arithmetic"))?;
    sum(&[prompts, outputs, rows])
}
#[cfg(test)] mod tests;
