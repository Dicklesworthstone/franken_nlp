//! Complete classification heads share physical native layers, never probabilities.
use super::*;
use crate::native_engine::strict_int8::{cohort::Int8CohortEngine,
    scoring::cohort::{self as native, CompiledRequest, Int8CandidateCohortRun, Int8ScoredCohort, Int8ScoringCohortBudget}};

impl PreparedInt8Classification {
    /// Exact simultaneous contexts in stable head order. More than 64 heads
    /// refuses this explicit mode; serial execution remains available unchanged.
    pub fn cohort_contexts(&self) -> Result<Vec<usize>, Int8ScoringError> {
        if self.inner.heads.len() != self.schedules.len() { return Err(Int8ScoringError::Accounting); }
        native::task::contexts(&self.schedules)
    }
    /// All exact prompts and independent KV branches must be admitted together.
    /// The aggregate resident KV remains bounded by the SEALED task budget.
    /// Final decisions and the outer envelope finish inside the native session.
    pub fn execute_cohort_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut Int8CohortEngine<'_>, budget: Int8ScoringCohortBudget, control: &mut C)
        -> Result<Int8ScoredCohort<Int8ClassificationRun>, Int8ClassificationError> {
        planning::checkpoint(control)?;
        self.verify_identity(admitted)?; check_model(admitted, engine.artifact_identity())?;
        check_work_ceiling(self.work, budget.native)?;
        self.cohort_contexts()?;
        let budget = Int8ScoringCohortBudget { max_kv_bytes: budget.max_kv_bytes.min(self.inner.budget.max_kv_bytes),
            max_output_bytes: budget.max_output_bytes.min(self.inner.budget.max_output_bytes), ..budget };
        let prompts = self.cohort_prompts(control)?;
        let mut requests = planning::reserved(self.head_count())?;
        for ((head, prompt), &schedule) in self.inner.heads.iter().zip(&prompts).zip(&self.schedules) {
            requests.push(CompiledRequest { identity: admitted, prompt, scorer: &head.classifier.scorer,
                mode: ScoringMode::FullVocabulary, schedule, max_output_bytes: budget.max_output_bytes,
                budget: Int8ScoringBudget { native: Int8RunBudget::exact(schedule.model), max_kv_bytes: budget.max_kv_bytes } });
        }
        native::execute_compiled_with(engine, &requests, budget, control,
            |run, control| self.finish_cohort(run, budget.max_output_bytes, control))
    }
    fn cohort_prompts<C: DecodeStepControl>(&self, control: &mut C) -> Result<Vec<Vec<u32>>, Int8ClassificationError> {
        let mut prompts = planning::reserved(self.head_count())?;
        for head in &self.inner.heads {
            planning::checkpoint(control)?;
            let mut prompt = planning::reserved(head.prompt_len)?;
            prompt.extend(head.task.ir().prompt_segments().iter().flat_map(|segment| segment.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int8ClassificationError::Accounting); }
            prompts.push(prompt);
        }
        Ok(prompts)
    }
    fn finish_cohort<C: DecodeStepControl>(&self, run: Int8CandidateCohortRun, cap: u64, control: &mut C)
        -> Result<Int8ScoredCohort<Int8ClassificationRun>, Int8ClassificationError> {
        native::task::finish(run, &self.schedules, self.work, cap, |heads|
            self.execute_heads(control, |_, _, _, _| heads.next().ok_or(Int8ClassificationError::Accounting)))
    }
}
#[cfg(test)] mod tests;
