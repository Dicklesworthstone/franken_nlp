//! Grammar-first native JSON decoding with true selected-row projection.
//!
//! This is an explicit execution strategy, not a silent replacement for the
//! universal full-vocabulary reference in `constrained`. Greedy selection needs
//! no vocabulary-wide probability denominator, but every nonterminal chosen
//! token still executes the model and populates all 44 KV slots. No forced-token
//! shortcut skips hidden-state evolution. Only legal projected rows are checked
//! for finite logits; this route does not certify unrequested vocabulary rows.

use std::{cell::RefCell, error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{canonjson, grammar::{
    mask::{DenseTokenMask, MaskWorkLimits, VocabMaskOracle},
    runtime::{JsonProgram, JsonState},
}};
use super::{
    constrained::{JsonDecodeError, JsonDecodeOptions, JsonDecodeOutput, JsonWorkBudget},
    decode::{DecodeCancellationKind, DecodeStepControl},
    hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE,
        candidate_scoring::{EagerPrefixSession, PrefixBudget, PrefixScoringError}},
    lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{CandidateLogits, ProjectionRows}},
};

pub const SPARSE_JSON_EXECUTION: &str = "eager-grammar-first-selected-rows-v1";
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_OUTPUT_TOKENS: usize = 1_000_000;

/// A row limit refuses the entire operation; it NEVER prunes legal tokens.
/// The output ceiling applies to the complete versioned envelope, not merely
/// its JSON payload. Model/KV/process admission remains the caller's authority.
#[derive(Clone, Copy, Debug)]
pub struct SparseJsonLimits {
    pub max_rows_per_step: usize,
    pub max_output_bytes: usize,
}
impl Default for SparseJsonLimits {
    fn default() -> Self { Self { max_rows_per_step: NANBEIGE_VOCAB_SIZE, max_output_bytes: 4 * 1024 * 1024 } }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SparseJsonDecodeOutput {
    pub schema_version: u32,
    pub execution: String,
    pub output: JsonDecodeOutput,
}

#[derive(Debug)]
pub enum SparseJsonError {
    Decode(JsonDecodeError),
    Native(PrefixScoringError),
    Projection,
    InvalidRows,
    OutputBudget,
    Serialization,
}
impl fmt::Display for SparseJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(e) => e.fmt(f),
            Self::Native(e) => e.fmt(f),
            Self::Projection => f.write_str("sparse JSON native projection failed"),
            Self::InvalidRows => f.write_str("sparse JSON projection rows are invalid"),
            Self::OutputBudget => f.write_str("sparse JSON complete output budget exceeded"),
            Self::Serialization => f.write_str("sparse JSON serialization failed"),
        }
    }
}
impl Error for SparseJsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Decode(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<JsonDecodeError> for SparseJsonError { fn from(e: JsonDecodeError) -> Self { Self::Decode(e) } }
impl From<PrefixScoringError> for SparseJsonError { fn from(e: PrefixScoringError) -> Self { Self::Native(e) } }

/// Decode through an already-admitted native engine. The complete exact prompt
/// is prefetched only once, without intermediate lm_head work or L2 tap copies.
/// Each step materializes its full legal grammar mask BEFORE projecting rows.
/// EOS enters that set only at an accepting state and remains a scored choice,
/// not an automatic early stop. Runtime, artifact, and source-trust admission
/// are not changed by selecting this strategy.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_eager_sparse<C: DecodeStepControl>(
    engine: &mut HfBf16EagerEngine,
    prompt: &[u32],
    program: &JsonProgram,
    vocabulary: &VocabMaskOracle,
    options: &JsonDecodeOptions,
    budget: JsonWorkBudget,
    limits: SparseJsonLimits,
    control: &mut C,
) -> Result<SparseJsonDecodeOutput, SparseJsonError> {
    if !engine.kv_cache().all_slots_have_len(0) { return Err(JsonDecodeError::EngineAlreadyPrimed.into()); }
    if vocabulary.trie().vocab_size() != NANBEIGE_VOCAB_SIZE {
        return Err(JsonDecodeError::InvalidRequest("model/tokenizer vocabulary mismatch").into());
    }
    preflight(prompt, vocabulary.trie().vocab_size(), options, budget, limits)?;
    // Both adapters borrow the SAME caller-owned cancellation state. Each
    // callback drops its RefMut before native/mask code continues; no borrow
    // crosses a projection and no second deadline or token budget is created.
    let shared = RefCell::new(control);
    let mut native_control = SharedControl(&shared);
    let mut decode_control = SharedControl(&shared);
    let mut session = EagerPrefixSession::new(engine, prompt, options.max_new_tokens - 1,
        budget.max_kv_bytes, PrefixBudget { max_forward_positions: budget.max_forward_positions,
            max_projected_logits: budget.max_projected_logits }, &mut native_control)?;
    let result = run(prompt, program, vocabulary, options, budget, limits, &mut decode_control, &mut session);
    let result = match result {
        Ok(result) => result,
        Err(error) => return Err(match session.last_error() {
            Some(native) => SparseJsonError::Native(native.clone()), None => error,
        }),
    };
    let work = session.work();
    if work.forward_positions != result.output.forward_positions
        || work.projected_logits != result.output.projected_logits
        || work.prefix_evaluations != result.output.token_ids.len() as u64
        || work.prompt_positions != prompt.len() as u64
        || work.continuation_positions != result.output.token_ids.len() as u64 - 1
        || work.rewound_positions != 0 {
        return Err(PrefixScoringError::InvalidExecution.into());
    }
    drop(session); // Clear logical KV on success; all error/unwind paths also drop.
    Ok(result)
}

struct SharedControl<'a, 'b, C>(&'a RefCell<&'b mut C>);
impl<C: DecodeStepControl> DecodeStepControl for SharedControl<'_, '_, C> {
    fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
        self.0.borrow_mut().checkpoint(step)
    }
    fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
        self.0.borrow_mut().prefill_checkpoint(position)
    }
}

fn preflight(prompt: &[u32], width: usize, options: &JsonDecodeOptions,
    budget: JsonWorkBudget, limits: SparseJsonLimits) -> Result<(), SparseJsonError> {
    if prompt.is_empty() || width == 0 || width > NANBEIGE_VOCAB_SIZE {
        return Err(JsonDecodeError::InvalidRequest("empty prompt or invalid vocabulary").into());
    }
    if options.max_new_tokens == 0 || options.max_new_tokens > MAX_OUTPUT_TOKENS
        || limits.max_rows_per_step == 0 || limits.max_rows_per_step > NANBEIGE_VOCAB_SIZE
        || limits.max_output_bytes == 0 || limits.max_output_bytes > MAX_RESPONSE_BYTES {
        return Err(JsonDecodeError::InvalidRequest("invalid sparse decode limits").into());
    }
    if prompt.iter().chain(options.excluded_token_ids.iter()).chain(std::iter::once(&options.eos_token_id))
        .any(|&id| id as usize >= width) {
        return Err(JsonDecodeError::InvalidRequest("out-of-vocabulary token").into());
    }
    if budget.mask_limits.max_trie_node_visits == 0 || budget.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(JsonDecodeError::InvalidRequest("zero mask work bound").into());
    }
    let positions = prompt.len().checked_add(options.max_new_tokens - 1)
        .and_then(|n| u64::try_from(n).ok()).ok_or(JsonDecodeError::BudgetExceeded("position arithmetic"))?;
    if positions > budget.max_forward_positions { return Err(JsonDecodeError::BudgetExceeded("forward work").into()); }
    if budget.max_projected_logits == 0 { return Err(JsonDecodeError::BudgetExceeded("projection work").into()); }
    // Legal-row counts depend on runtime grammar states. Charge each complete
    // set before projecting, rather than silently applying a full-vocabulary
    // P*V preflight that would make sparse execution unreachable on long input.
    Ok(())
}

trait Vocabulary {
    fn width(&self) -> usize;
    fn bytes(&self, token: u32) -> Option<&[u8]>;
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize;
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, limits: MaskWorkLimits,
        control: &mut C, step: usize) -> Result<DenseTokenMask, JsonDecodeError>;
}
impl Vocabulary for VocabMaskOracle {
    fn width(&self) -> usize { self.trie().vocab_size() }
    fn bytes(&self, token: u32) -> Option<&[u8]> { self.trie().token_bytes(token) }
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize { self.trie().node_count().min(limits.max_trie_node_visits) }
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, limits: MaskWorkLimits,
        control: &mut C, step: usize) -> Result<DenseTokenMask, JsonDecodeError> {
        let mut cancellation = None;
        let mask = self.materialize(state, limits, |_| {
            cancellation = control.checkpoint(step); cancellation.is_none()
        });
        if let Some(cause) = cancellation { return Err(JsonDecodeError::Cancelled(cause)); }
        mask.map_err(JsonDecodeError::Mask)
    }
}

fn poll<C: DecodeStepControl>(control: &mut C, step: usize) -> Result<(), JsonDecodeError> {
    match control.checkpoint(step) { Some(cause) => Err(JsonDecodeError::Cancelled(cause)), None => Ok(()) }
}
fn legal_rows<C: DecodeStepControl>(mask: &DenseTokenMask, accepting: bool,
    options: &JsonDecodeOptions, cap: usize, control: &mut C, step: usize) -> Result<Vec<u32>, SparseJsonError> {
    let legal = |id: u32| if id == options.eos_token_id { accepting }
        else { mask.contains(id) && !options.excluded_token_ids.contains(&id) };
    let mut count = 0;
    for index in 0..mask.vocab_size() {
        if index % 1024 == 0 { poll(control, step)?; }
        if legal(index as u32) {
            count += 1;
            if count > cap { return Err(JsonDecodeError::BudgetExceeded("legal projection rows").into()); }
        }
    }
    if count == 0 { return Err(JsonDecodeError::NoLegalToken.into()); }
    let mut ids = Vec::new();
    ids.try_reserve_exact(count).map_err(|_| JsonDecodeError::AllocationRefused)?;
    for index in 0..mask.vocab_size() {
        if index % 1024 == 0 { poll(control, step)?; }
        if legal(index as u32) { ids.push(index as u32); }
    }
    Ok(ids)
}
fn select_rows(ids: &[u32], logits: &[f32]) -> Result<u32, SparseJsonError> {
    if ids.is_empty() || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SparseJsonError::InvalidRows);
    }
    if ids.len() != logits.len() || logits.iter().any(|v| !v.is_finite()) {
        return Err(JsonDecodeError::InvalidLogits.into());
    }
    let mut best = 0;
    for index in 1..ids.len() {
        // Lowest token ID wins ties; +0 and -0 intentionally compare equal.
        if logits[index] > logits[best] { best = index; }
    }
    Ok(ids[best])
}

#[allow(clippy::too_many_arguments)]
fn run<V: Vocabulary, C: DecodeStepControl, M: CandidateLogits>(
    prompt: &[u32], program: &JsonProgram, vocabulary: &V, options: &JsonDecodeOptions,
    budget: JsonWorkBudget, limits: SparseJsonLimits, control: &mut C, model: &mut M,
) -> Result<SparseJsonDecodeOutput, SparseJsonError> {
    preflight(prompt, vocabulary.width(), options, budget, limits)?;
    let mut tokens = Vec::new();
    tokens.try_reserve_exact(options.max_new_tokens).map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(program.max_output_bytes().min(limits.max_output_bytes))
        .map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut state = program.initial_state();
    let mut mask_charge = 0_u64;
    let mut projected = 0_u64;
    for step in 0..options.max_new_tokens {
        poll(control, step)?;
        mask_charge = mask_charge.checked_add(vocabulary.mask_charge(budget.mask_limits) as u64)
            .filter(|&n| n <= budget.max_total_mask_node_visits)
            .ok_or(JsonDecodeError::BudgetExceeded("mask work"))?;
        let mask = vocabulary.mask(&state, budget.mask_limits, control, step)?;
        if mask.vocab_size() != vocabulary.width() { return Err(SparseJsonError::InvalidRows); }
        let ids = legal_rows(&mask, state.is_accepting(), options, limits.max_rows_per_step, control, step)?;
        projected = projected.checked_add(ids.len() as u64)
            .filter(|&n| n <= budget.max_projected_logits)
            .ok_or(JsonDecodeError::BudgetExceeded("projection work"))?;
        // Even a one-bit mask projects that row and advances the whole native
        // prefix. Never turn grammar certainty into skipped transformer work.
        let logits = model.project(&tokens, ProjectionRows::Selected(&ids)).map_err(|_| SparseJsonError::Projection)?;
        let selected = select_rows(&ids, &logits)?;
        poll(control, step)?; // No token commits after cancellation.
        tokens.push(selected);
        if selected == options.eos_token_id {
            let json = String::from_utf8(bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
            program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
            let positions = (prompt.len() as u64).checked_add(tokens.len() as u64 - 1)
                .ok_or(JsonDecodeError::BudgetExceeded("position arithmetic"))?;
            let output = SparseJsonDecodeOutput { schema_version: 1, execution: SPARSE_JSON_EXECUTION.to_owned(),
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(),
                    token_ids: tokens, json, forward_positions: positions, projected_logits: projected,
                    mask_node_visit_charge: mask_charge } };
            if canonjson::canonical_bytes(&output).map_err(|_| SparseJsonError::Serialization)?.len() > limits.max_output_bytes {
                return Err(SparseJsonError::OutputBudget);
            }
            return Ok(output);
        }
        let emitted = vocabulary.bytes(selected).filter(|value| !value.is_empty()).ok_or(JsonDecodeError::IllegalTransition)?;
        let next_bytes = bytes.len().checked_add(emitted.len()).ok_or(SparseJsonError::OutputBudget)?;
        if next_bytes > program.max_output_bytes() || next_bytes > limits.max_output_bytes {
            return Err(SparseJsonError::OutputBudget);
        }
        if !state.consume_bytes(emitted) { return Err(JsonDecodeError::IllegalTransition.into()); }
        bytes.extend_from_slice(emitted);
        // On the next iteration `tokens` includes this nonterminal even when
        // its emitted bytes already form valid JSON. Native KV must advance
        // before scoring EOS against any still-legal longer continuations.
    }
    Err(JsonDecodeError::BudgetExceeded("output tokens before EOS").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use crate::grammar::CompileLimits;
    struct Table(Vec<Vec<u8>>);
    impl Vocabulary for Table {
        fn width(&self) -> usize { self.0.len() }
        fn bytes(&self, token: u32) -> Option<&[u8]> { self.0.get(token as usize).map(Vec::as_slice) }
        fn mask_charge(&self, _: MaskWorkLimits) -> usize { self.width() }
        fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, _: MaskWorkLimits, _: &mut C, _: usize)
            -> Result<DenseTokenMask, JsonDecodeError> {
            let mut mask = DenseTokenMask::empty(self.width());
            for (id, bytes) in self.0.iter().enumerate() {
                if !bytes.is_empty() && state.clone().consume_bytes(bytes) { mask.set_legal(id as u32).unwrap(); }
            }
            Ok(mask)
        }
    }
    struct Model { logits: Vec<f32>, calls: Vec<(Vec<u32>, Vec<u32>)> }
    impl CandidateLogits for Model {
        type Error = &'static str;
        fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
            let ProjectionRows::Selected(ids) = rows else { panic!("sparse decode must not project full vocabulary"); };
            self.calls.push((prefix.to_vec(), ids.to_vec()));
            Ok(ids.iter().map(|&id| self.logits[id as usize]).collect())
        }
    }
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    fn options(n: usize) -> JsonDecodeOptions {
        JsonDecodeOptions { max_new_tokens: n, eos_token_id: 0, excluded_token_ids: BTreeSet::new() }
    }
    fn budget() -> JsonWorkBudget { JsonWorkBudget {
        max_forward_positions: 100, max_projected_logits: 1000, max_kv_bytes: 100000,
        max_total_mask_node_visits: 10000, mask_limits: MaskWorkLimits::default(),
    } }
    fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
    fn model(logits: &[f32]) -> Model { Model { logits: logits.to_vec(), calls: Vec::new() } }
    #[test]
    fn illegal_rows_and_early_eos_are_never_projected() {
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec(), b"malformed".to_vec()]);
        let mut m = model(&[100.0, 1.0, f32::NAN]);
        let out = run(&[2, 2], &p, &v, &options(2), budget(), SparseJsonLimits::default(), &mut Continue, &mut m).unwrap();
        assert_eq!(out.output.json, "true");
        assert_eq!(out.output.token_ids, [1, 0]);
        assert_eq!(m.calls, [(vec![], vec![1]), (vec![1], vec![0])]);
        assert_eq!(out.output.forward_positions, 3);
        assert_eq!(out.output.projected_logits, 2);
    }
    #[test]
    fn accepting_numeric_prefix_still_competes_with_its_longer_continuation() {
        let p = program(r#"{"type":"integer","enum":[1,10]}"#);
        let v = Table(vec![vec![], b"1".to_vec(), b"0".to_vec()]);
        let mut m = model(&[0.0, 2.0, 1.0]);
        let out = run(&[1], &p, &v, &options(3), budget(), SparseJsonLimits::default(), &mut Continue, &mut m).unwrap();
        assert_eq!(out.output.json, "10");
        assert_eq!(m.calls, [(vec![], vec![1]), (vec![1], vec![0, 2]), (vec![1, 2], vec![0])]);
        assert_eq!(out.output.projected_logits, 4);
    }
    #[test]
    fn exclusions_apply_but_terminal_eos_is_still_explicitly_allowed() {
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec(), b"false".to_vec()]);
        let mut opts = options(2); opts.excluded_token_ids.extend([0, 1]);
        let mut m = model(&[0.0, 100.0, 1.0]);
        let out = run(&[1], &p, &v, &opts, budget(), SparseJsonLimits::default(), &mut Continue, &mut m).unwrap();
        assert_eq!(out.output.json, "false"); assert_eq!(out.output.token_ids, [2, 0]);
        assert_eq!(m.calls[0].1, [2]);
    }
    #[test]
    fn complete_legal_set_budget_refuses_without_pruning_or_projecting() {
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec(), b"false".to_vec()]);
        for row_cap in [false, true] {
            let mut b = budget(); let mut l = SparseJsonLimits::default();
            if row_cap { l.max_rows_per_step = 1; } else { b.max_projected_logits = 1; }
            let mut m = model(&[0.0, 1.0, 2.0]);
            assert!(run(&[1], &p, &v, &options(2), b, l, &mut Continue, &mut m).is_err());
            assert!(m.calls.is_empty());
        }
    }
    #[test]
    fn projected_nonfinite_and_signed_zero_ties_preserve_selection_contract() {
        assert_eq!(select_rows(&[1, 2], &[-0.0, 0.0]).unwrap(), 1);
        assert!(matches!(select_rows(&[1, 2], &[1.0, f32::NAN]), Err(SparseJsonError::Decode(JsonDecodeError::InvalidLogits))));
        assert!(select_rows(&[2, 1], &[0.0, 0.0]).is_err());
    }
    #[test]
    fn no_eos_or_small_complete_envelope_never_returns_partial_json() {
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec()]);
        assert!(run(&[1], &p, &v, &options(1), budget(), SparseJsonLimits::default(), &mut Continue, &mut model(&[0.0, 1.0])).is_err());
        let limits = SparseJsonLimits { max_output_bytes: 8, ..SparseJsonLimits::default() };
        assert!(matches!(run(&[1], &p, &v, &options(2), budget(), limits, &mut Continue, &mut model(&[0.0, 1.0])), Err(SparseJsonError::OutputBudget)));
    }
    #[test]
    fn cancellation_and_mask_budget_stop_before_later_projection() {
        struct Cancel;
        impl DecodeStepControl for Cancel {
            fn checkpoint(&mut self, step: usize) -> Option<DecodeCancellationKind> {
                (step == 1).then_some(DecodeCancellationKind::Deadline)
            }
        }
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec()]); let mut m = model(&[0.0, 1.0]);
        assert!(matches!(run(&[1], &p, &v, &options(2), budget(), SparseJsonLimits::default(), &mut Cancel, &mut m),
            Err(SparseJsonError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)))));
        assert_eq!(m.calls.len(), 1);
        let mut b = budget(); b.max_total_mask_node_visits = 3;
        let mut m = model(&[0.0, 1.0]);
        assert!(matches!(run(&[1], &p, &v, &options(2), b, SparseJsonLimits::default(), &mut Continue, &mut m),
            Err(SparseJsonError::Decode(JsonDecodeError::BudgetExceeded("mask work")))));
        assert_eq!(m.calls.len(), 1);
    }
}
