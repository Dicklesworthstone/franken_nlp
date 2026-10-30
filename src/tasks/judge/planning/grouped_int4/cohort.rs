//! Both orders, every criterion and all source/evidence heads in one cohort.
use super::*;
use crate::native_engine::strict_int4::{cohort::Int4CohortEngine,
    scoring::cohort::{self as native, CompiledRequest, Int4CandidateCohortRun, Int4ScoredCohort, Int4ScoringCohortBudget}};

impl PreparedInt4Judge {
    pub fn cohort_contexts(&self) -> Result<Vec<usize>, Int4ScoringError> {
        if self.inner.executable.bundle().heads.len() != self.schedules.len() { return Err(Int4ScoringError::Accounting); }
        native::task::contexts(&self.schedules)
    }
    /// Execute the whole sealed judgment within one native session. One head
    /// failure discards every score. No renewed work or serial memory discount.
    pub fn execute_cohort_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut Int4CohortEngine<'_>, budget: Int4ScoringCohortBudget, control: &mut C)
        -> Result<Int4ScoredCohort<Int4JudgeRun>, Int4JudgeError> {
        checkpoint(control)?;
        self.verify_identity(admitted)?; check_model(admitted, engine.model_identity())?;
        if self.group_width != engine.model_identity().group_width() { return Err(Int4JudgeError::Identity); }
        check_work(self.work, budget.native)?;
        self.cohort_contexts()?;
        let budget = Int4ScoringCohortBudget { max_kv_bytes: budget.max_kv_bytes.min(self.budget.max_kv_bytes),
            max_output_bytes: budget.max_output_bytes.min(self.max_result_bytes()), ..budget };
        let prompts = self.cohort_prompts(control)?;
        let mut requests = reserved(self.head_count())?;
        for ((head, prompt), &schedule) in self.inner.executable.bundle().heads.iter().zip(&prompts).zip(&self.schedules) {
            requests.push(CompiledRequest { identity: admitted, prompt, scorer: &head.scorer,
                mode: ScoringMode::FullVocabulary, schedule, max_output_bytes: budget.max_output_bytes,
                budget: Int4ScoringBudget { native: Int4RunBudget::exact(schedule.model),
                    max_kv_bytes: budget.max_kv_bytes.min(head.ir.budget().max_kv_bytes) } });
        }
        native::execute_compiled_with(engine, &requests, budget, control,
            |run, control| self.finish_cohort(run, budget.max_output_bytes, control))
    }
    fn cohort_prompts<C: DecodeStepControl>(&self, control: &mut C) -> Result<Vec<Vec<u32>>, Int4JudgeError> {
        let mut prompts = reserved(self.head_count())?;
        for head in &self.inner.executable.bundle().heads {
            checkpoint(control)?;
            let mut prompt = reserved(head.prompt_len)?;
            prompt.extend(head.ir.prompt_segments().iter().flat_map(|segment| segment.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(Int4JudgeError::Accounting); }
            prompts.push(prompt);
        }
        Ok(prompts)
    }
    fn finish_cohort<C: DecodeStepControl>(&self, run: Int4CandidateCohortRun, cap: u64, control: &mut C)
        -> Result<Int4ScoredCohort<Int4JudgeRun>, Int4JudgeError> {
        native::task::finish(run, &self.schedules, self.work, cap, |heads|
            self.evaluate_heads(control, |_, _, _, _| heads.next().ok_or(Int4JudgeError::Accounting)))
    }
}
#[cfg(test)] mod tests;
