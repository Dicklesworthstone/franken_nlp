//! One controlled, model-free compiler for live and retained judgment records.
use super::*;
use std::io;

const MAX_DEFAULT_BYTES: usize = 1024 * 1024;

impl JudgeCorpusConfig {
    /// Fixed configuration only. Never invent a document to validate a future
    /// stream, and never rewrite the supplied model/task identity.
    pub fn validate(&self, planner: &JudgePlanner) -> Result<(), HostedError> {
        self.identity.validate().map_err(|_| HostedError::ModelIdentity)?;
        check_binding(&self.identity, planner.template_digest(), planner.tokenizer_digest())?;
        self.task_ceiling.validate().map_err(|_| HostedError::Limits("judge corpus task ceiling"))?;
        validate_work(self.max_model_work)?;
        let limits = self.planning;
        let head = limits.per_head;
        if head.max_candidates == 0 || head.max_total_tokens == 0 || head.max_nodes == 0
            || head.max_depth == 0 || head.max_candidate_id_bytes == 0 || head.max_projected_logits == 0
            || limits.max_total_prompt_tokens == 0 || limits.max_total_candidate_tokens == 0
            || limits.max_total_projected_logits == 0 || !(1..=64 * 1024 * 1024).contains(&limits.max_output_bytes) {
            return Err(HostedError::Limits("judge corpus planning limits"));
        }
        if let Some(args) = &self.defaults {
            // Count borrowed serialization before any defaults clone. No JSON
            // tree or unbounded staging buffer is allocated just to check size.
            check_serialized_size(args, MAX_DEFAULT_BYTES)?;
            check_args(args, self.task_ceiling)
                .map_err(|_| HostedError::Limits("judge corpus defaults"))?;
        }
        Ok(())
    }
}

pub(super) fn prepare<C: DecodeStepControl>(planner: &JudgePlanner, config: &JudgeCorpusConfig,
    document: BatchDocument<JudgeBatchArgs>, control: &mut C) -> Result<PreparedInt8Judge, BatchItemFailure> {
    checkpoint(control)?;
    let args = document.task_args.or_else(|| config.defaults.clone())
        .ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?;
    check_args(&args, config.task_ceiling)?;
    let context = PlanContext::new(&config.identity, config.task_ceiling)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let plan = planner.plan_int8_with_control(&args.into_request(document.text),
        &context, config.planning, control).map_err(planning_failure)?;
    checkpoint(control)?;
    Ok(plan)
}

fn check_args(args: &JudgeBatchArgs, ceiling: TaskBudget) -> Result<(), BatchItemFailure> {
    let reject = || BatchItemFailure::reject(BatchCode::Planning);
    let budget = match args {
        JudgeBatchArgs::Pairwise { criterion, b, budget, .. } => {
            if criterion.is_empty() || b.is_empty() { return Err(reject()); }
            // Pairwise margins are u32 milli-log-odds, not ppm probabilities.
            *budget
        }
        JudgeBatchArgs::Rubric { rubric, policy, budget } => {
            rubric.validate().map_err(|_| reject())?;
            if policy.minimum_peak_weight_ppm > 1_000_000 || policy.maximum_normalized_entropy_ppm > 1_000_000 {
                return Err(reject());
            }
            *budget
        }
        JudgeBatchArgs::Faithfulness { claim, policy, budget } => {
            if claim.is_empty() { return Err(reject()); }
            policy.validate().map_err(|_| reject())?;
            *budget
        }
    };
    budget.validate().map_err(|_| reject())?;
    if budget.max_input_tokens > ceiling.max_input_tokens || budget.max_output_tokens > ceiling.max_output_tokens
        || budget.max_output_bytes > ceiling.max_output_bytes || budget.max_grammar_states > ceiling.max_grammar_states
        || budget.max_kv_bytes > ceiling.max_kv_bytes { return Err(reject()); }
    Ok(())
}

pub(super) fn check_serialized_size<T: Serialize + ?Sized>(value: &T, cap: usize) -> Result<(), HostedError> {
    struct Counter { bytes: usize, cap: usize }
    impl io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes = self.bytes.checked_add(bytes.len()).filter(|&n| n <= self.cap)
                .ok_or_else(|| io::Error::other("judge settings size limit"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    serde_json::to_writer(Counter { bytes: 0, cap }, value)
        .map_err(|_| HostedError::Limits("judge settings serialization bound"))
}

#[cfg(test)] pub(super) mod tests;
