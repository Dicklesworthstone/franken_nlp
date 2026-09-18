//! Quantized finite-language scoring on one live KV branch.
//!
//! The existing CandidateScorer remains the probability/EOS authority. Its
//! prefix-first traversal needs one prompt prefill and one forward per distinct
//! nonempty continuation prefix. Backtracking discards KV suffixes; it is NOT
//! an independently retained fork or an unpriced hidden-state cache.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{canonjson, tasks::ir::Candidate,
    native_engine::lmhead::scoring::{CandidateLogits, CandidateScorer, CandidateScores,
        ProjectionRows, ScoringError, ScoringLimits, ScoringMode, ScoringWork}};
use super::{Int8RunBudget, Int8Session, Int8Work, StrictInt8Engine, StrictInt8Error,
    STRICT_INT8_PROFILE, V, KV_SLOT_COUNT, QUERY_HEAD_COUNT, KV_BYTES_PER_TOKEN,
    DEFAULT_ADMITTED_CONTEXT_CAP, ProjectionWork, LinearRows, DecodeStepControl,
    DecodeCancellationKind, decoder_projection_work};

pub const INT8_SCORING_EXECUTION: &str = "portable-int8-single-kv-candidate-trie-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Int8ScoringError {
    Input, Traversal, Accounting, Allocation, OutputBudget, Serialization,
    Scoring(ScoringError), Native(StrictInt8Error),
}
impl fmt::Display for Int8ScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "int8 scoring requires a bounded exact prompt and finite language",
            Self::Traversal => "int8 scorer diverged from prefix-first traversal",
            Self::Accounting => "int8 scorer work disagrees with its complete language",
            Self::Allocation => "int8 scorer bounded allocation refused",
            Self::OutputBudget => "complete int8 scoring output exceeds its byte budget",
            Self::Serialization => "int8 scoring serialization failed",
            Self::Scoring(_) => "int8 finite-candidate scoring failed",
            Self::Native(_) => "int8 scoring native execution failed",
        })
    }
}
impl Error for Int8ScoringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<ScoringError> for Int8ScoringError { fn from(e: ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt8Error> for Int8ScoringError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl Int8ScoringError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Native(StrictInt8Error::Cancelled(c)) => Some(*c), _ => None }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int8ScoringBudget { pub native: Int8RunBudget, pub max_kv_bytes: u64 }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8CandidateRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub scores: CandidateScores,
    pub model_work: Int8Work,
    pub rewound_positions: u64,
}

/// Owned exact-token leaf plan, not an artifact or runtime admission capability.
/// The host binds prompt/language/score-space to its admitted task identity.
/// Raw-text classification supplies that identity-checked composition separately.
/// This type cannot deserialize caller-provided work certificates.
pub struct Int8CandidatePlan {
    prompt: Vec<u32>, scorer: CandidateScorer, mode: ScoringMode,
    schedule: CandidateSchedule, max_output_bytes: u64,
}
impl Int8CandidatePlan {
    pub fn compile(prompt: Vec<u32>, candidates: &[Candidate], eos: u32, mode: ScoringMode,
        limits: ScoringLimits, max_output_bytes: u64) -> Result<Self, Int8ScoringError> {
        check_prompt(&prompt)?;
        check_output_limit(max_output_bytes)?;
        let schedule = CandidateSchedule::new(prompt.len(), candidates, mode)?;
        let scorer = CandidateScorer::compile(candidates, V, eos, limits)?;
        if schedule.scoring.projected_logits > limits.max_projected_logits {
            return Err(ScoringError::LimitExceeded("projected_logits").into());
        }
        Ok(Self { prompt, scorer, mode, schedule, max_output_bytes })
    }
    pub fn planned_work(&self) -> Int8Work { self.schedule.model }
    pub fn scoring_work(&self) -> ScoringWork { self.schedule.scoring }
    pub fn required_context(&self) -> usize { self.schedule.context }
    pub fn execute<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>,
        budget: Int8ScoringBudget, control: &mut C) -> Result<Int8CandidateRun, Int8ScoringError> {
        execute_compiled(&self.prompt, &self.scorer, self.mode, self.schedule,
            self.max_output_bytes, engine, budget, control)
    }
}

/// Geometry only: never holds another prompt, scorer, logits or hidden table.
#[derive(Clone, Copy)]
pub(crate) struct CandidateSchedule {
    pub(crate) scoring: ScoringWork,
    pub(crate) model: Int8Work,
    pub(crate) context: usize,
    max_prefix: usize,
    rewound: u64,
}
impl CandidateSchedule {
    pub(crate) fn new(prompt: usize, candidates: &[Candidate], mode: ScoringMode) -> Result<Self, Int8ScoringError> {
        if prompt == 0 || prompt > DEFAULT_ADMITTED_CONTEXT_CAP || candidates.is_empty() || candidates.len() > 4096 {
            return Err(Int8ScoringError::Input);
        }
        let mut ordered = reserve(candidates.len())?;
        ordered.extend(candidates.iter().map(|c| c.continuation().token_ids()));
        ordered.sort_unstable();
        let mut previous: &[u32] = &[];
        let (mut edges, mut depths, mut maximum) = (0_u64, 0_u64, 0_usize);
        for tokens in ordered {
            if tokens.is_empty() || tokens == previous || tokens.len() > DEFAULT_ADMITTED_CONTEXT_CAP {
                return Err(Int8ScoringError::Input);
            }
            let common = previous.iter().zip(tokens).take_while(|(a, b)| a == b).count();
            let added = (tokens.len() - common) as u64;
            edges = add(edges, added)?;
            depths = add(depths, triangle(tokens.len() as u64)? - triangle(common as u64)?)?;
            maximum = maximum.max(tokens.len()); previous = tokens;
        }
        let context = prompt.checked_add(maximum).filter(|&n| n <= DEFAULT_ADMITTED_CONTEXT_CAP)
            .ok_or(StrictInt8Error::Context)?;
        let evaluations = add(edges, 1)?;
        let scored_edges = add(edges, candidates.len() as u64)?;
        let rows = if mode == ScoringMode::FullVocabulary { mul(evaluations, V as u64)? } else { scored_edges };
        let mut model = Int8Work::for_sequence(0, prompt, usize::try_from(rows).map_err(|_| StrictInt8Error::Work)?)?;
        let decoder = decoder_projection_work()?;
        model.forward_positions = add(prompt as u64, edges)?;
        model.projections = model.projections.checked_add(ProjectionWork {
            dot_products: mul(decoder.dot_products, edges)?,
            multiply_accumulates: mul(decoder.multiply_accumulates, edges)?,
        }).map_err(StrictInt8Error::from)?;
        // Prefix reuse changes logical positions. Pricing the total forwards
        // as one ever-growing sequence would be the wrong attention program.
        let causal_lengths = add(triangle(prompt as u64)?, add(mul(prompt as u64, edges)?, depths)?)?;
        model.attention_pairs = mul(causal_lengths, (KV_SLOT_COUNT * QUERY_HEAD_COUNT) as u64)?;
        Ok(Self { scoring: ScoringWork {
            prefix_evaluations: usize::try_from(evaluations).map_err(|_| StrictInt8Error::Work)?,
            scored_edges: usize::try_from(scored_edges).map_err(|_| StrictInt8Error::Work)?, projected_logits: rows,
        }, model, context, max_prefix: maximum, rewound: edges - previous.len() as u64 })
    }
    pub(crate) fn preflight(self, engine: &StrictInt8Engine<'_>, budget: Int8ScoringBudget) -> Result<(), Int8ScoringError> {
        if engine.state.active || engine.state.poisoned || !engine.cache.all_slots_have_len(0) {
            return Err(StrictInt8Error::EngineUnavailable.into());
        }
        check_capacity(self, engine.cache.capacity_positions(), budget)
    }
}

/// Sum independent heads without treating their reset contexts as one sequence.
impl Int8Work {
    pub fn checked_add(self, other: Self) -> Result<Self, StrictInt8Error> {
        let sum = |a: u64, b: u64| a.checked_add(b).ok_or(StrictInt8Error::Work);
        Ok(Self { forward_positions: sum(self.forward_positions, other.forward_positions)?,
            projected_logits: sum(self.projected_logits, other.projected_logits)?,
            attention_pairs: sum(self.attention_pairs, other.attention_pairs)?,
            projections: self.projections.checked_add(other.projections)? })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_compiled<C: DecodeStepControl>(prompt: &[u32], scorer: &CandidateScorer,
    mode: ScoringMode, schedule: CandidateSchedule, max_output_bytes: u64,
    engine: &mut StrictInt8Engine<'_>, budget: Int8ScoringBudget, control: &mut C)
    -> Result<Int8CandidateRun, Int8ScoringError> {
    check_prompt(prompt)?; check_output_limit(max_output_bytes)?;
    schedule.preflight(engine, budget)?;
    // The native session gets EXACTLY this head, never another copy of the
    // enclosing bundle's whole allowance. The host retains real memory guards.
    let mut session = engine.session(Int8RunBudget::exact(schedule.model), control)?;
    execute_driver(prompt, scorer, mode, schedule, max_output_bytes, &mut session)
}

fn execute_driver<D: Driver>(prompt: &[u32], scorer: &CandidateScorer, mode: ScoringMode,
    schedule: CandidateSchedule, max_output_bytes: u64, driver: &mut D) -> Result<Int8CandidateRun, Int8ScoringError> {
    let result = (|| {
        let mut backend = PrefixEvaluator { prompt, previous: reserve(schedule.max_prefix)?,
            primed: false, poisoned: false, last_error: None, max_prefix: schedule.max_prefix,
            evaluations: 0, rewound: 0, driver: &mut *driver };
        let scores = match scorer.score(&mut backend, mode) {
            Ok(scores) => scores,
            Err(e) => return Err(backend.last_error.take().unwrap_or_else(|| e.into())),
        };
        let model_work = backend.driver.work();
        if scores.work != schedule.scoring || model_work != schedule.model
            || backend.evaluations != schedule.scoring.prefix_evaluations || backend.rewound != schedule.rewound {
            return Err(Int8ScoringError::Accounting);
        }
        let run = Int8CandidateRun { schema_version: 1, execution: INT8_SCORING_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), scores, model_work, rewound_positions: backend.rewound };
        if canonjson::canonical_bytes(&run).map_err(|_| Int8ScoringError::Serialization)?.len() as u64 > max_output_bytes {
            return Err(Int8ScoringError::OutputBudget);
        }
        Ok(run)
    })();
    if result.is_err() { driver.abort(); }
    result
}

struct PrefixEvaluator<'a, D> {
    prompt: &'a [u32], previous: Vec<u32>, primed: bool, poisoned: bool,
    max_prefix: usize, evaluations: usize, rewound: u64,
    last_error: Option<Int8ScoringError>, driver: &'a mut D,
}
impl<D: Driver> PrefixEvaluator<'_, D> {
    fn project_inner(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Int8ScoringError> {
        let selection = checked_rows(rows)?;
        if self.poisoned || prefix.len() > self.max_prefix || prefix.iter().any(|&id| id as usize >= V)
            || (!self.primed && !prefix.is_empty()) || (self.primed && prefix <= self.previous.as_slice()) {
            return Err(Int8ScoringError::Traversal);
        }
        let current = if self.primed { self.prompt.len() + self.previous.len() } else { 0 };
        if self.driver.position()? != current { return Err(Int8ScoringError::Traversal); }
        let common = self.previous.iter().zip(prefix).take_while(|(a, b)| a == b).count();
        let retain = if self.primed { self.prompt.len() + common } else { 0 };
        self.poisoned = true; // Unwind cannot leave a retryable half-transition.
        self.driver.rewind(retain)?;
        self.rewound = add(self.rewound, (current - retain) as u64)?;
        if !self.primed { for &token in self.prompt { self.driver.append(token)?; } }
        // After any rewind at least one new token recomputes the correct hidden
        // state. No stale descendant hidden is used for an ancestor projection.
        for &token in &prefix[common..] { self.driver.append(token)?; }
        if self.driver.position()? != self.prompt.len() + prefix.len() { return Err(Int8ScoringError::Traversal); }
        let logits = self.driver.logits(selection)?;
        self.evaluations += 1; self.previous.clear(); self.previous.extend_from_slice(prefix);
        self.primed = true; self.poisoned = false; Ok(logits)
    }
}
impl<D: Driver> CandidateLogits for PrefixEvaluator<'_, D> {
    type Error = Int8ScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let result = self.project_inner(prefix, rows);
        if let Err(error) = &result {
            self.poisoned = true; self.driver.abort();
            if self.last_error.is_none() { self.last_error = Some(error.clone()); }
        }
        result
    }
}

// A private static seam for differential tests. Only the real quantized
// engine can produce the public native execution receipt.
trait Driver {
    fn position(&self) -> Result<usize, Int8ScoringError>;
    fn rewind(&mut self, retain: usize) -> Result<(), Int8ScoringError>;
    fn append(&mut self, token: u32) -> Result<(), Int8ScoringError>;
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int8ScoringError>;
    fn work(&self) -> Int8Work;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int8Session<'_, '_, C> {
    fn position(&self) -> Result<usize, Int8ScoringError> { Int8Session::position(self).map_err(Into::into) }
    fn rewind(&mut self, retain: usize) -> Result<(), Int8ScoringError> {
        let current = Int8Session::position(self)?;
        if retain > current { return Err(Int8ScoringError::Traversal); }
        if retain != current {
            self.engine.cache.rewind_completed(retain).map_err(|_| StrictInt8Error::Cache)?;
            self.hidden = None;
        }
        Ok(())
    }
    fn append(&mut self, token: u32) -> Result<(), Int8ScoringError> { Int8Session::append(self, token).map_err(Into::into) }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int8ScoringError> { Int8Session::logits(self, rows).map_err(Into::into) }
    fn work(&self) -> Int8Work { Int8Session::work(self) }
    fn abort(&mut self) { Int8Session::abort(self); }
}
fn checked_rows(rows: ProjectionRows<'_>) -> Result<LinearRows<'_>, Int8ScoringError> {
    let rows = match rows {
        ProjectionRows::FullVocabulary { vocabulary_size } if vocabulary_size == V => LinearRows::All,
        ProjectionRows::Selected(ids) => LinearRows::Selected(ids),
        _ => return Err(Int8ScoringError::Traversal),
    };
    rows.checked_count(V).map_err(StrictInt8Error::from)?; Ok(rows)
}
fn check_prompt(prompt: &[u32]) -> Result<(), Int8ScoringError> {
    if prompt.is_empty() || prompt.len() > DEFAULT_ADMITTED_CONTEXT_CAP || prompt.iter().any(|&id| id as usize >= V) {
        return Err(Int8ScoringError::Input);
    }
    Ok(())
}
fn check_output_limit(limit: u64) -> Result<(), Int8ScoringError> {
    if !(1..=64 * 1024 * 1024).contains(&limit) { return Err(Int8ScoringError::Input); } Ok(())
}
fn check_capacity(schedule: CandidateSchedule, capacity: usize, budget: Int8ScoringBudget) -> Result<(), Int8ScoringError> {
    if schedule.context > capacity { return Err(StrictInt8Error::Context.into()); }
    let bytes = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt8Error::Memory)?;
    if bytes > budget.max_kv_bytes { return Err(StrictInt8Error::Memory.into()); }
    let w = schedule.model;
    if w.forward_positions > budget.native.max_forward_positions || w.attention_pairs > budget.native.max_attention_pairs
        || !w.projections.fits(budget.native.max_projection_work) { return Err(StrictInt8Error::Work.into()); }
    Ok(())
}
fn add(a: u64, b: u64) -> Result<u64, Int8ScoringError> { a.checked_add(b).ok_or_else(|| StrictInt8Error::Work.into()) }
fn mul(a: u64, b: u64) -> Result<u64, Int8ScoringError> { a.checked_mul(b).ok_or_else(|| StrictInt8Error::Work.into()) }
fn triangle(n: u64) -> Result<u64, Int8ScoringError> { Ok(mul(n, add(n, 1)?)? / 2) }
fn reserve<T>(n: usize) -> Result<Vec<T>, Int8ScoringError> {
    let mut v = Vec::new(); v.try_reserve_exact(n).map_err(|_| Int8ScoringError::Allocation)?; Ok(v)
}

#[cfg(test)] mod tests;
