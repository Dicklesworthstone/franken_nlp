//! Both orders, all criteria or every source/evidence head in one native cohort.
use super::*;
use crate::native_engine::strict_int8::{cohort::Int8CohortEngine,
    scoring::cohort::{self as native, CompiledRequest, Int8CandidateCohortRun, Int8ScoredCohort, Int8ScoringCohortBudget}};
impl PreparedInt8Judge {
    pub fn cohort_contexts(&self) -> Result<Vec<usize>, Int8ScoringError> {
        if self.inner.executable.bundle().heads.len() != self.schedules.len() { return Err(Int8ScoringError::Accounting); }
        native::task::contexts(&self.schedules)
    }
    /// Complete sealed task, not one successful order/criterion at a time.
    /// Membership cannot exceed native width or silently fall back to serial.
    /// The embedding host retains all prompt, KV and result reservations.
    pub fn execute_cohort_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut Int8CohortEngine<'_>, budget: Int8ScoringCohortBudget, control: &mut C)
        -> Result<Int8ScoredCohort<Int8JudgeRun>, Int8JudgeError> {
        checkpoint(control)?;
        self.verify_identity(admitted)?; check_model(admitted, engine.artifact_identity())?;
        check_work(self.work, budget.native)?;
        self.cohort_contexts()?;
        let budget = Int8ScoringCohortBudget { max_kv_bytes: budget.max_kv_bytes.min(self.budget.max_kv_bytes),
            max_output_bytes: budget.max_output_bytes.min(self.max_result_bytes()), ..budget };
        let prompts = self.cohort_prompts(control)?;
        let mut requests = reserved(self.head_count())?;
        for ((head, prompt), &schedule) in self.inner.executable.bundle().heads.iter().zip(&prompts).zip(&self.schedules) {
            requests.push(CompiledRequest { identity: admitted, prompt, scorer: &head.scorer,
                mode: ScoringMode::FullVocabulary, schedule, max_output_bytes: budget.max_output_bytes,
                budget: Int8ScoringBudget { native: Int8RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(head.ir.budget().max_kv_bytes) } });
        }
        native::execute_compiled_with(engine, &requests, budget, control,
            |run, control| self.finish_cohort(run, budget.max_output_bytes, control))
    }
    fn cohort_prompts<C: DecodeStepControl>(&self, control: &mut C) -> Result<Vec<Vec<u32>>, Int8JudgeError> {
        let mut prompts = reserved(self.head_count())?;
        for head in &self.inner.executable.bundle().heads {
            checkpoint(control)?;
            let mut prompt = reserved(head.prompt_len)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|segment| segment.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int8JudgeError::Accounting); }
            prompts.push(prompt);
        }
        Ok(prompts)
    }
    fn finish_cohort<C: DecodeStepControl>(&self, run: Int8CandidateCohortRun, cap: u64, control: &mut C)
        -> Result<Int8ScoredCohort<Int8JudgeRun>, Int8JudgeError> {
        native::task::finish(run, &self.schedules, self.work, cap, |heads|
            self.evaluate_heads(control, |_, _, _, _| heads.next().ok_or(Int8JudgeError::Accounting)))
    }
}
#[cfg(test)] mod tests;
