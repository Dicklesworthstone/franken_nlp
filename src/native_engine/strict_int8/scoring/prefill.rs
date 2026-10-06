//! Opt-in bounded layer-major prompt execution for exact candidate scoring.
//!
//! Only the initial prompt is grouped. CandidateScorer still owns every
//! probability denominator, EOS edge and branch visit. Backtracking retains
//! one live KV branch and never reruns the prompt or projects a stale hidden.
//! Extra scratch is separately admitted by the host, not by these limits.
use super::*;
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;

impl Int8CandidatePlan {
    /// Physical prompt scheduling only: no changed candidate language, score
    /// space, output identity or default. No model-parity or speed award.
    pub fn execute_layer_major<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>,
        budget: Int8ScoringBudget, prefill: Int8PrefillLimits, control: &mut C)
        -> Result<Int8CandidateRun, Int8ScoringError> {
        execute_compiled_with_prefill(&self.prompt, &self.scorer, self.mode, self.schedule,
            self.max_output_bytes, engine, budget, Some(prefill), control)
    }
}

/// Closed task drivers select scheduling AFTER binding exact task identity.
/// The native session receives an exact per-head slice of aggregate work.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_compiled_with_prefill<C: DecodeStepControl>(prompt: &[u32], scorer: &CandidateScorer,
    mode: ScoringMode, schedule: CandidateSchedule, max_output_bytes: u64,
    engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget,
    prefill: Option<Int8PrefillLimits>, control: &mut C) -> Result<Int8CandidateRun, Int8ScoringError> {
    let Some(prefill) = prefill else {
        return execute_compiled(prompt, scorer, mode, schedule, max_output_bytes, engine, budget, control);
    };
    prefill.validate()?;
    check_prompt(prompt)?; check_output_limit(max_output_bytes)?;
    schedule.preflight(engine, budget)?;
    let mut session = engine.session(Int8RunBudget::exact(schedule.model), control)?;
    execute_layer_major_driver(prompt, scorer, mode, schedule, max_output_bytes, prefill, &mut session)
}

trait PromptDriver: Driver {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8ScoringError>;
}
impl<C: DecodeStepControl> PromptDriver for Int8Session<'_, '_, C> {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8ScoringError> {
        Int8Session::append_layer_major(self, prompt, limits).map_err(Into::into)
    }
}
struct Grouped<'a, D> { driver: &'a mut D, limits: Int8PrefillLimits }
impl<D: PromptDriver> Driver for Grouped<'_, D> {
    fn position(&self) -> Result<usize, Int8ScoringError> { self.driver.position() }
    fn rewind(&mut self, retain: usize) -> Result<(), Int8ScoringError> { self.driver.rewind(retain) }
    fn append(&mut self, token: u32) -> Result<(), Int8ScoringError> { self.driver.append(token) }
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int8ScoringError> {
        self.driver.append_layer_major(prompt, self.limits)
    }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int8ScoringError> { self.driver.logits(rows) }
    fn work(&self) -> Int8Work { self.driver.work() }
    fn abort(&mut self) { self.driver.abort(); }
}
fn execute_layer_major_driver<D: PromptDriver>(prompt: &[u32], scorer: &CandidateScorer, mode: ScoringMode,
    schedule: CandidateSchedule, max_output_bytes: u64, limits: Int8PrefillLimits, driver: &mut D)
    -> Result<Int8CandidateRun, Int8ScoringError> {
    limits.validate()?;
    execute_driver(prompt, scorer, mode, schedule, max_output_bytes, &mut Grouped { driver, limits })
}

#[cfg(test)] mod tests;
