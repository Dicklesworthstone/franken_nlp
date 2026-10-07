//! Physically batched sentiment dimensions retain independent score spaces.
use super::*;
use crate::native_engine::strict_int8::{cohort::Int8CohortEngine,
    scoring::cohort::{self as native, CompiledRequest, Int8CandidateCohortRun, Int8ScoredCohort, Int8ScoringCohortBudget}};
impl PreparedInt8Sentiment {
    pub fn cohort_contexts(&self) -> Result<Vec<usize>, Int8ScoringError> {
        if self.inner.heads.len() != self.schedules.len() { return Err(Int8ScoringError::Accounting); }
        native::task::contexts(&self.schedules)
    }
    /// One shared-layer cohort for all requested axes. No shared softmax or
    /// cross-axis confidence. The whole resident KV fits the original task cap.
    pub fn execute_cohort_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut Int8CohortEngine<'_>, budget: Int8ScoringCohortBudget, control: &mut C)
        -> Result<Int8ScoredCohort<Int8SentimentRun>, Int8SentimentError> {
        checkpoint(control)?;
        self.verify_identity(admitted)?; check_model(admitted, engine.artifact_identity())?;
        check_work(self.work, budget.native)?;
        self.cohort_contexts()?;
        let budget = Int8ScoringCohortBudget { max_kv_bytes: budget.max_kv_bytes.min(self.budget.max_kv_bytes),
            max_output_bytes: budget.max_output_bytes.min(self.max_result_bytes()), ..budget };
        let prompts = self.cohort_prompts(control)?;
        let mut requests = Vec::new();
        requests.try_reserve_exact(self.head_count()).map_err(|_| Int8SentimentError::Allocation)?;
        for ((head, prompt), &schedule) in self.inner.heads.iter().zip(&prompts).zip(&self.schedules) {
            requests.push(CompiledRequest { identity: admitted, prompt, scorer: &head.scorer,
                mode: self.inner.options.mode, schedule, max_output_bytes: budget.max_output_bytes,
                budget: Int8ScoringBudget { native: Int8RunBudget::exact(schedule.model), max_kv_bytes: budget.max_kv_bytes } });
        }
        native::execute_compiled_with(engine, &requests, budget, control,
            |run, control| self.finish_cohort(run, budget.max_output_bytes, control))
    }
    fn cohort_prompts<C: DecodeStepControl>(&self, control: &mut C) -> Result<Vec<Vec<u32>>, Int8SentimentError> {
        let mut prompts = Vec::new();
        prompts.try_reserve_exact(self.head_count()).map_err(|_| Int8SentimentError::Allocation)?;
        for head in &self.inner.heads {
            checkpoint(control)?;
            let mut prompt = Vec::new();
            prompt.try_reserve_exact(head.prompt_len).map_err(|_| Int8SentimentError::Allocation)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|segment| segment.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int8SentimentError::Accounting); }
            prompts.push(prompt);
        }
        Ok(prompts)
    }
    fn finish_cohort<C: DecodeStepControl>(&self, run: Int8CandidateCohortRun, cap: u64, control: &mut C)
        -> Result<Int8ScoredCohort<Int8SentimentRun>, Int8SentimentError> {
        native::task::finish(run, &self.schedules, self.work, cap, |heads|
            self.execute_heads(control, |_, _, _, _| heads.next().ok_or(Int8SentimentError::Accounting)))
    }
}
#[cfg(test)] mod tests;
