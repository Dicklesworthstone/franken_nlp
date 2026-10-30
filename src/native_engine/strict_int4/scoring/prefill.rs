//! Opt-in layer-major prompt execution for grouped-INT4 candidate scoring.
//!
//! Only the initial prompt is grouped. The shared scorer still owns every
//! candidate edge, EOS and probability denominator; continuations retain one
//! live KV branch. Extra scratch is a separate host admission requirement.
use super::*;
use crate::native_engine::strict_int4::prefill::Int4PrefillLimits;

impl Int4CandidatePlan {
    /// Use bounded prompt morsels without changing score space, identity or
    /// native work. The caller must reserve the validated extra scratch until
    /// the session has drained. Sequential execution remains the default.
    pub fn execute_layer_major<C: DecodeStepControl>(&self, engine: &mut StrictInt4Engine<'_>,
        budget: Int4ScoringBudget, prefill: Int4PrefillLimits, control: &mut C)
        -> Result<Int4CandidateRun, Int4ScoringError> {
        execute_compiled_with_prefill(&self.prompt, &self.scorer, self.mode, self.schedule,
            self.max_output_bytes, engine, budget, Some(prefill), control)
    }
}

/// The sealed task composition binds the model and all heads before choosing
/// physical prompt scheduling. Each session receives exactly one head's work.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_compiled_with_prefill<C: DecodeStepControl>(prompt: &[u32], scorer: &CandidateScorer,
    mode: ScoringMode, schedule: CandidateSchedule, max_output_bytes: u64,
    engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget,
    prefill: Option<Int4PrefillLimits>, control: &mut C) -> Result<Int4CandidateRun, Int4ScoringError> {
    let Some(prefill) = prefill else {
        return execute_compiled(prompt, scorer, mode, schedule, max_output_bytes, engine, budget, control);
    };
    prefill.validate()?;
    check_prompt(prompt)?; check_output_limit(max_output_bytes)?;
    schedule.preflight(engine, budget)?;
    let mut session = engine.session(Int4RunBudget::exact(schedule.model), control)?;
    execute_layer_major_driver(prompt, scorer, mode, schedule, max_output_bytes, prefill, &mut session)
}

trait PromptDriver: Driver {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4ScoringError>;
}
impl<C: DecodeStepControl> PromptDriver for Int4Session<'_, '_, C> {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4ScoringError> {
        Int4Session::append_layer_major(self, prompt, limits).map_err(Into::into)
    }
}
struct Grouped<'a, D> { driver: &'a mut D, limits: Int4PrefillLimits }
impl<D: PromptDriver> Driver for Grouped<'_, D> {
    fn checkpoint(&mut self) -> Result<(), Int4ScoringError> { self.driver.checkpoint() }
    fn position(&self) -> Result<usize, Int4ScoringError> { self.driver.position() }
    fn rewind(&mut self, retain: usize) -> Result<(), Int4ScoringError> { self.driver.rewind(retain) }
    fn append(&mut self, token: u32) -> Result<(), Int4ScoringError> { self.driver.append(token) }
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int4ScoringError> {
        self.driver.append_layer_major(prompt, self.limits)
    }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError> { self.driver.logits(rows) }
    fn work(&self) -> Int4Work { self.driver.work() }
    fn abort(&mut self) { self.driver.abort(); }
}
fn execute_layer_major_driver<D: PromptDriver>(prompt: &[u32], scorer: &CandidateScorer, mode: ScoringMode,
    schedule: CandidateSchedule, max_output_bytes: u64, limits: Int4PrefillLimits, driver: &mut D)
    -> Result<Int4CandidateRun, Int4ScoringError> {
    limits.validate()?;
    execute_driver(prompt, scorer, mode, schedule, max_output_bytes, &mut Grouped { driver, limits })
}

#[cfg(test)] mod tests;
