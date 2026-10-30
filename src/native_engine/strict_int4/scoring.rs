//! Grouped-INT4 finite-language scoring on one live KV branch.
//!
//! CandidateScorer remains the shared probability, traversal and EOS authority.
//! Only its native adapter and group-aware work schedule live here. Every
//! nonempty distinct continuation prefix is forwarded once; backtracking
//! discards completed KV suffixes, never retaining unpriced hidden/KV forks.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{canonjson, tasks::ir::Candidate,
    native_engine::lmhead::scoring::{CandidateLogits, CandidateScorer, CandidateScores,
        ProjectionRows, ScoringError, ScoringLimits, ScoringMode, ScoringWork}};
use super::{Int4RunBudget, Int4Session, Int4Work, StrictInt4Engine, StrictInt4Error,
    STRICT_INT4_PROFILE, V, KV_SLOT_COUNT, QUERY_HEAD_COUNT, KV_BYTES_PER_TOKEN,
    DEFAULT_ADMITTED_CONTEXT_CAP, MAX_MODEL_K, LinearRows, DecodeStepControl,
    DecodeCancellationKind, scale_work};

pub mod prefill;

pub const INT4_SCORING_EXECUTION: &str = "grouped-int4-single-kv-candidate-trie-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Int4ScoringError {
    Input, Identity, Traversal, Accounting, Allocation, OutputBudget, Serialization,
    Scoring(ScoringError), Native(StrictInt4Error),
}
impl fmt::Display for Int4ScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "INT4 scoring requires a bounded exact prompt, group and finite language",
            Self::Identity => "INT4 scoring model or group partition differs",
            Self::Traversal => "INT4 scorer diverged from prefix-first traversal",
            Self::Accounting => "INT4 scorer work disagrees with its complete language",
            Self::Allocation => "INT4 scorer bounded allocation refused",
            Self::OutputBudget => "complete INT4 scoring output exceeds its byte budget",
            Self::Serialization => "INT4 scoring serialization failed",
            Self::Scoring(_) => "INT4 finite-candidate scoring failed",
            Self::Native(_) => "INT4 scoring native execution failed",
        })
    }
}
impl Error for Int4ScoringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Scoring(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<ScoringError> for Int4ScoringError { fn from(e: ScoringError) -> Self { Self::Scoring(e) } }
impl From<StrictInt4Error> for Int4ScoringError { fn from(e: StrictInt4Error) -> Self { Self::Native(e) } }
impl Int4ScoringError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Native(StrictInt4Error::Cancelled(c)) => Some(*c), _ => None }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int4ScoringBudget { pub native: Int4RunBudget, pub max_kv_bytes: u64 }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4CandidateRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub group_width: usize,
    pub scores: CandidateScores,
    pub model_work: Int4Work,
    pub rewound_positions: u64,
}

/// Exact-token leaf plan, not a model-identity or runtime-admission capability.
/// Embedders bind prompt/language/model identity in their task composition.
/// The group partition is checked again against actual engine-owned weights.
/// No public driver injection or deserialized work certificate is accepted.
pub struct Int4CandidatePlan {
    prompt: Vec<u32>, scorer: CandidateScorer, mode: ScoringMode,
    schedule: CandidateSchedule, max_output_bytes: u64,
}
impl Int4CandidatePlan {
    #[allow(clippy::too_many_arguments)]
    pub fn compile(prompt: Vec<u32>, candidates: &[Candidate], eos: u32, mode: ScoringMode,
        group_width: usize, limits: ScoringLimits, max_output_bytes: u64) -> Result<Self, Int4ScoringError> {
        check_prompt(&prompt)?; check_output_limit(max_output_bytes)?; check_group(group_width)?;
        // Enforce the shared finite-language limits BEFORE scheduling traversal.
        let scorer = CandidateScorer::compile(candidates, V, eos, limits)?;
        let schedule = CandidateSchedule::new(group_width, prompt.len(), candidates, mode)?;
        if schedule.scoring.projected_logits > limits.max_projected_logits {
            return Err(ScoringError::LimitExceeded("projected_logits").into());
        }
        Ok(Self { prompt, scorer, mode, schedule, max_output_bytes })
    }
    pub fn planned_work(&self) -> Int4Work { self.schedule.model }
    pub fn scoring_work(&self) -> ScoringWork { self.schedule.scoring }
    pub fn required_context(&self) -> usize { self.schedule.context }
    pub fn group_width(&self) -> usize { self.schedule.group_width }
    pub fn execute<C: DecodeStepControl>(&self, engine: &mut StrictInt4Engine<'_>,
        budget: Int4ScoringBudget, control: &mut C) -> Result<Int4CandidateRun, Int4ScoringError> {
        execute_compiled(&self.prompt, &self.scorer, self.mode, self.schedule,
            self.max_output_bytes, engine, budget, control)
    }
}

/// Geometry only; neither duplicate prompts/scorers nor retained hidden tables.
#[derive(Clone, Copy)]
pub(crate) struct CandidateSchedule {
    pub(crate) scoring: ScoringWork,
    pub(crate) model: Int4Work,
    pub(crate) context: usize,
    group_width: usize, max_prefix: usize, rewound: u64,
}
impl CandidateSchedule {
    pub(crate) fn new(group_width: usize, prompt: usize, candidates: &[Candidate], mode: ScoringMode)
        -> Result<Self, Int4ScoringError> {
        check_group(group_width)?;
        if prompt == 0 || prompt > DEFAULT_ADMITTED_CONTEXT_CAP || candidates.is_empty() || candidates.len() > 4096 {
            return Err(Int4ScoringError::Input);
        }
        let mut ordered = reserve(candidates.len())?;
        ordered.extend(candidates.iter().map(|c| c.continuation().token_ids()));
        ordered.sort_unstable();
        let mut previous: &[u32] = &[];
        let (mut edges, mut depths, mut maximum) = (0_u64, 0_u64, 0_usize);
        for tokens in ordered {
            if tokens.is_empty() || tokens == previous || tokens.len() > DEFAULT_ADMITTED_CONTEXT_CAP {
                return Err(Int4ScoringError::Input);
            }
            let common = previous.iter().zip(tokens).take_while(|(a, b)| a == b).count();
            let added = (tokens.len() - common) as u64;
            edges = add(edges, added)?;
            depths = add(depths, triangle(tokens.len() as u64)? - triangle(common as u64)?)?;
            maximum = maximum.max(tokens.len()); previous = tokens;
        }
        let context = prompt.checked_add(maximum).filter(|&n| n <= DEFAULT_ADMITTED_CONTEXT_CAP)
            .ok_or(StrictInt4Error::Context)?;
        let evaluations = add(edges, 1)?;
        let scored_edges = add(edges, candidates.len() as u64)?;
        let rows = if mode == ScoringMode::FullVocabulary { mul(evaluations, V as u64)? } else { scored_edges };
        let mut model = Int4Work::for_sequence(group_width, 0, prompt,
            usize::try_from(rows).map_err(|_| StrictInt4Error::Work)?)?;
        let decoder = Int4Work::for_sequence(group_width, 0, 1, 0)?.projections;
        model.forward_positions = add(prompt as u64, edges)?;
        model.projections = model.projections.checked_add(scale_work(decoder, edges)?)
            .map_err(StrictInt4Error::from)?;
        // Group dots include ceil(K/group_width) for each of the seven layer
        // projections and the head. MACs use logical K, never padded width.
        // Attention follows actual branch depths, NOT total forwarded tokens.
        let causal_lengths = add(triangle(prompt as u64)?, add(mul(prompt as u64, edges)?, depths)?)?;
        model.attention_pairs = mul(causal_lengths, (KV_SLOT_COUNT * QUERY_HEAD_COUNT) as u64)?;
        Ok(Self { scoring: ScoringWork {
            prefix_evaluations: usize::try_from(evaluations).map_err(|_| StrictInt4Error::Work)?,
            scored_edges: usize::try_from(scored_edges).map_err(|_| StrictInt4Error::Work)?, projected_logits: rows,
        }, model, context, group_width, max_prefix: maximum, rewound: edges - previous.len() as u64 })
    }
    pub(crate) fn preflight(self, engine: &StrictInt4Engine<'_>, budget: Int4ScoringBudget) -> Result<(), Int4ScoringError> {
        if engine.model_identity().group_width() != self.group_width { return Err(Int4ScoringError::Identity); }
        if engine.state.active || engine.state.poisoned || !engine.cache.all_slots_have_len(0) {
            return Err(StrictInt4Error::EngineUnavailable.into());
        }
        check_capacity(self, engine.cache.capacity_positions(), budget)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_compiled<C: DecodeStepControl>(prompt: &[u32], scorer: &CandidateScorer,
    mode: ScoringMode, schedule: CandidateSchedule, max_output_bytes: u64,
    engine: &mut StrictInt4Engine<'_>, budget: Int4ScoringBudget, control: &mut C)
    -> Result<Int4CandidateRun, Int4ScoringError> {
    check_prompt(prompt)?; check_output_limit(max_output_bytes)?;
    schedule.preflight(engine, budget)?;
    // Exactly one head slice, never a renewed whole-bundle work allowance.
    let mut session = engine.session(Int4RunBudget::exact(schedule.model), control)?;
    execute_driver(prompt, scorer, mode, schedule, max_output_bytes, &mut session)
}

fn execute_driver<D: Driver>(prompt: &[u32], scorer: &CandidateScorer, mode: ScoringMode,
    schedule: CandidateSchedule, max_output_bytes: u64, driver: &mut D) -> Result<Int4CandidateRun, Int4ScoringError> {
    let result = (|| {
        driver.checkpoint()?;
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
            return Err(Int4ScoringError::Accounting);
        }
        let run = Int4CandidateRun { schema_version: 1, execution: INT4_SCORING_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: schedule.group_width,
            scores, model_work, rewound_positions: backend.rewound };
        backend.driver.checkpoint()?;
        if canonjson::canonical_bytes(&run).map_err(|_| Int4ScoringError::Serialization)?.len() as u64 > max_output_bytes {
            return Err(Int4ScoringError::OutputBudget);
        }
        backend.driver.checkpoint()?;
        Ok(run)
    })();
    if result.is_err() { driver.abort(); }
    result
}

struct PrefixEvaluator<'a, D> {
    prompt: &'a [u32], previous: Vec<u32>, primed: bool, poisoned: bool,
    max_prefix: usize, evaluations: usize, rewound: u64,
    last_error: Option<Int4ScoringError>, driver: &'a mut D,
}
impl<D: Driver> PrefixEvaluator<'_, D> {
    fn project_inner(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Int4ScoringError> {
        let selection = checked_rows(rows)?;
        if self.poisoned || prefix.len() > self.max_prefix || prefix.iter().any(|&id| id as usize >= V)
            || (!self.primed && !prefix.is_empty()) || (self.primed && prefix <= self.previous.as_slice()) {
            return Err(Int4ScoringError::Traversal);
        }
        self.driver.checkpoint()?;
        let current = if self.primed { self.prompt.len() + self.previous.len() } else { 0 };
        if self.driver.position()? != current { return Err(Int4ScoringError::Traversal); }
        let common = self.previous.iter().zip(prefix).take_while(|(a, b)| a == b).count();
        let retain = if self.primed { self.prompt.len() + common } else { 0 };
        self.poisoned = true;
        self.driver.rewind(retain)?;
        self.rewound = add(self.rewound, (current - retain) as u64)?;
        if !self.primed { self.driver.append_prompt(self.prompt)?; }
        // Strictly increasing traversal guarantees a new token after a rewind.
        // It recomputes hidden state; stale descendant hidden is never reused.
        for &token in &prefix[common..] { self.driver.append(token)?; }
        if self.driver.position()? != self.prompt.len() + prefix.len() { return Err(Int4ScoringError::Traversal); }
        let logits = self.driver.logits(selection)?;
        self.evaluations += 1; self.previous.clear(); self.previous.extend_from_slice(prefix);
        self.primed = true; self.poisoned = false; Ok(logits)
    }
}
impl<D: Driver> CandidateLogits for PrefixEvaluator<'_, D> {
    type Error = Int4ScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let result = self.project_inner(prefix, rows);
        if let Err(error) = &result {
            self.poisoned = true; self.driver.abort();
            if self.last_error.is_none() { self.last_error = Some(error.clone()); }
        }
        result
    }
}

// Private static differential-test seam. Public execution takes only the real
// immutable-model engine and its exclusive RAII session, never a caller driver.
trait Driver {
    fn checkpoint(&mut self) -> Result<(), Int4ScoringError>;
    fn position(&self) -> Result<usize, Int4ScoringError>;
    fn rewind(&mut self, retain: usize) -> Result<(), Int4ScoringError>;
    fn append(&mut self, token: u32) -> Result<(), Int4ScoringError>;
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int4ScoringError> {
        for &token in prompt { self.append(token)?; }
        Ok(())
    }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError>;
    fn work(&self) -> Int4Work;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int4Session<'_, '_, C> {
    fn checkpoint(&mut self) -> Result<(), Int4ScoringError> { super::poll(self.control()).map_err(Into::into) }
    fn position(&self) -> Result<usize, Int4ScoringError> { Int4Session::position(self).map_err(Into::into) }
    fn rewind(&mut self, retain: usize) -> Result<(), Int4ScoringError> {
        let current = Int4Session::position(self)?;
        if retain > current { return Err(Int4ScoringError::Traversal); }
        if retain != current {
            self.engine.cache.rewind_completed(retain).map_err(|_| StrictInt4Error::Cache)?;
            self.hidden = None;
        }
        Ok(())
    }
    fn append(&mut self, token: u32) -> Result<(), Int4ScoringError> { Int4Session::append(self, token).map_err(Into::into) }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError> { Int4Session::logits(self, rows).map_err(Into::into) }
    fn work(&self) -> Int4Work { Int4Session::work(self) }
    fn abort(&mut self) { Int4Session::abort(self); }
}
fn checked_rows(rows: ProjectionRows<'_>) -> Result<LinearRows<'_>, Int4ScoringError> {
    let rows = match rows {
        ProjectionRows::FullVocabulary { vocabulary_size } if vocabulary_size == V => LinearRows::All,
        ProjectionRows::Selected(ids) => LinearRows::Selected(ids),
        _ => return Err(Int4ScoringError::Traversal),
    };
    rows.checked_count(V).map_err(StrictInt4Error::from)?; Ok(rows)
}
fn check_prompt(prompt: &[u32]) -> Result<(), Int4ScoringError> {
    if prompt.is_empty() || prompt.len() > DEFAULT_ADMITTED_CONTEXT_CAP || prompt.iter().any(|&id| id as usize >= V) {
        return Err(Int4ScoringError::Input);
    }
    Ok(())
}
fn check_group(width: usize) -> Result<(), Int4ScoringError> {
    if width == 0 || width > MAX_MODEL_K { return Err(Int4ScoringError::Input); } Ok(())
}
fn check_output_limit(limit: u64) -> Result<(), Int4ScoringError> {
    if !(1..=64 * 1024 * 1024).contains(&limit) { return Err(Int4ScoringError::Input); } Ok(())
}
fn check_capacity(schedule: CandidateSchedule, capacity: usize, budget: Int4ScoringBudget) -> Result<(), Int4ScoringError> {
    if schedule.context > capacity { return Err(StrictInt4Error::Context.into()); }
    let bytes = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt4Error::Memory)?;
    if bytes > budget.max_kv_bytes { return Err(StrictInt4Error::Memory.into()); }
    let w = schedule.model;
    if w.forward_positions > budget.native.max_forward_positions || w.attention_pairs > budget.native.max_attention_pairs
        || !w.projections.fits(budget.native.max_projection_work) { return Err(StrictInt4Error::Work.into()); }
    Ok(())
}
fn add(a: u64, b: u64) -> Result<u64, Int4ScoringError> { a.checked_add(b).ok_or_else(|| StrictInt4Error::Work.into()) }
fn mul(a: u64, b: u64) -> Result<u64, Int4ScoringError> { a.checked_mul(b).ok_or_else(|| StrictInt4Error::Work.into()) }
fn triangle(n: u64) -> Result<u64, Int4ScoringError> {
    let next = add(n, 1)?;
    if n % 2 == 0 { mul(n / 2, next) } else { mul(n, next / 2) }
}
fn reserve<T>(n: usize) -> Result<Vec<T>, Int4ScoringError> {
    let mut v = Vec::new(); v.try_reserve_exact(n).map_err(|_| Int4ScoringError::Allocation)?; Ok(v)
}

#[cfg(test)] mod tests;
