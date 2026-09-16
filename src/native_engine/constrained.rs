//! Greedy schema-constrained decoding through the native eager engine.
//!
//! This is the universal full-projection path, not a forced-token or sparse
//! projection optimization. No partial JSON is returned or streamed. EOS is
//! scored only at accepting states; success additionally requires independent
//! whole-value validation. Runtime ownership and artifact admission stay with
//! the caller, and the already-admitted engine is never cloned or loaded here.

use std::{collections::BTreeSet, error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::grammar::{
    mask::{DenseTokenMask, MaskOracleError, MaskWorkLimits, VocabMaskOracle},
    runtime::JsonProgram,
};
use super::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    hf_bf16_eager::{HfBf16EagerEngine, HfBf16EagerError, candidate_scoring::ClearCache},
    kv::{KV_ELEMENTS_PER_POSITION, KV_SLOT_COUNT},
    lmhead::NANBEIGE_VOCAB_SIZE,
};

/// Semantic options. The excluded set must include the template-control
/// alphabet and unwanted special tokens. The one explicit EOS id is legal
/// only on acceptance, even when it also belongs to that excluded set.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JsonDecodeOptions {
    /// Includes the terminal EOS token, which contributes no JSON bytes.
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub excluded_token_ids: BTreeSet<u32>,
}

/// Caller-selected real work and admitted KV-payload ceilings.
#[derive(Clone, Copy, Debug)]
pub struct JsonWorkBudget {
    pub max_forward_positions: u64,
    pub max_projected_logits: u64,
    pub max_kv_bytes: u64,
    pub max_total_mask_node_visits: u64,
    pub mask_limits: MaskWorkLimits,
}

/// Successful output only. Tokens include EOS; `json` never does.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JsonDecodeOutput {
    pub schema_version: u32,
    pub numerics_profile: String,
    pub token_ids: Vec<u32>,
    pub json: String,
    pub forward_positions: u64,
    pub projected_logits: u64,
    /// Conservative work charge, not a measured trie-node visit count.
    pub mask_node_visit_charge: u64,
}

#[derive(Debug)]
pub enum JsonDecodeError {
    InvalidRequest(&'static str),
    EngineAlreadyPrimed,
    BudgetExceeded(&'static str),
    AllocationRefused,
    InvalidLogits,
    NoLegalToken,
    IllegalTransition,
    IndependentValidation,
    Cancelled(DecodeCancellationKind),
    Mask(MaskOracleError),
    Engine(HfBf16EagerError),
}
impl fmt::Display for JsonDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(category) => write!(f, "constrained decode request refused: {category}"),
            Self::EngineAlreadyPrimed => f.write_str("constrained decode requires an empty engine cache"),
            Self::BudgetExceeded(axis) => write!(f, "constrained decode has no result: {axis} budget exhausted"),
            Self::AllocationRefused => f.write_str("constrained decode allocation refused"),
            Self::InvalidLogits => f.write_str("constrained decode requires a finite complete logit vector"),
            Self::NoLegalToken => f.write_str("constrained decode has no legal token"),
            Self::IllegalTransition => f.write_str("constrained token disagrees with the executable grammar"),
            Self::IndependentValidation => f.write_str("constrained output failed independent validation"),
            Self::Cancelled(kind) => write!(f, "constrained decode cancelled: {kind:?}"),
            Self::Mask(error) => write!(f, "constrained mask refused: {error}"),
            Self::Engine(_) => f.write_str("constrained native forward failed"),
        }
    }
}
impl Error for JsonDecodeError {}

/// Execute on the exact existing bf16 engine. Cleanup retains its weight and
/// reserved KV buffers but discards all request KV on success, failure, or
/// unwinding. A caller's nonempty cache is refused without altering it.
pub fn decode_json_eager<C: DecodeStepControl>(
    engine: &mut HfBf16EagerEngine,
    prompt: &[u32],
    program: &JsonProgram,
    vocabulary: &VocabMaskOracle,
    options: &JsonDecodeOptions,
    budget: JsonWorkBudget,
    control: &mut C,
) -> Result<JsonDecodeOutput, JsonDecodeError> {
    if !engine.kv_cache().all_slots_have_len(0) { return Err(JsonDecodeError::EngineAlreadyPrimed); }
    if vocabulary.trie().vocab_size() != NANBEIGE_VOCAB_SIZE {
        return Err(JsonDecodeError::InvalidRequest("model/tokenizer vocabulary mismatch"));
    }
    let positions = preflight(prompt, vocabulary.width(), options, budget)?;
    if positions > engine.kv_cache().capacity_positions() as u64 {
        return Err(JsonDecodeError::BudgetExceeded("context"));
    }
    let kv_bytes = (engine.kv_cache().capacity_positions() as u64)
        .checked_mul((KV_SLOT_COUNT * KV_ELEMENTS_PER_POSITION * 2 * size_of::<u16>()) as u64)
        .ok_or(JsonDecodeError::BudgetExceeded("KV arithmetic"))?;
    if kv_bytes > budget.max_kv_bytes { return Err(JsonDecodeError::BudgetExceeded("KV")); }
    let profile = engine.profile();
    let mut guard = ClearCache(engine);
    run(prompt, program, vocabulary, options, budget, profile, control, |token| {
        guard.0.decode(token).map(|forward| forward.logits).map_err(JsonDecodeError::Engine)
    })
}

fn preflight(prompt: &[u32], width: usize, options: &JsonDecodeOptions, budget: JsonWorkBudget) -> Result<u64, JsonDecodeError> {
    if prompt.is_empty() || width == 0 || width > u32::MAX as usize {
        return Err(JsonDecodeError::InvalidRequest("empty prompt or invalid vocabulary"));
    }
    if options.max_new_tokens == 0 { return Err(JsonDecodeError::InvalidRequest("zero output token budget")); }
    if prompt.iter().chain(options.excluded_token_ids.iter()).chain(std::iter::once(&options.eos_token_id)).any(|&id| id as usize >= width) {
        return Err(JsonDecodeError::InvalidRequest("out-of-vocabulary token"));
    }
    if budget.mask_limits.max_trie_node_visits == 0 || budget.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(JsonDecodeError::InvalidRequest("zero mask work bound"));
    }
    let positions = prompt.len().checked_add(options.max_new_tokens - 1)
        .and_then(|n| u64::try_from(n).ok()).ok_or(JsonDecodeError::BudgetExceeded("position arithmetic"))?;
    let logits = positions.checked_mul(width as u64).ok_or(JsonDecodeError::BudgetExceeded("projection arithmetic"))?;
    if positions > budget.max_forward_positions || logits > budget.max_projected_logits {
        return Err(JsonDecodeError::BudgetExceeded("forward work"));
    }
    Ok(positions)
}

trait Vocabulary {
    fn width(&self) -> usize;
    fn bytes(&self, token: u32) -> Option<&[u8]>;
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize;
    fn mask<C: DecodeStepControl>(&self, state: &crate::grammar::runtime::JsonState<'_>, limits: MaskWorkLimits, control: &mut C, step: usize) -> Result<DenseTokenMask, JsonDecodeError>;
}
impl Vocabulary for VocabMaskOracle {
    fn width(&self) -> usize { self.trie().vocab_size() }
    fn bytes(&self, token: u32) -> Option<&[u8]> { self.trie().token_bytes(token) }
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize { self.trie().node_count().min(limits.max_trie_node_visits) }
    fn mask<C: DecodeStepControl>(&self, state: &crate::grammar::runtime::JsonState<'_>, limits: MaskWorkLimits, control: &mut C, step: usize) -> Result<DenseTokenMask, JsonDecodeError> {
        let mut cancellation = None;
        let result = self.materialize(state, limits, |_| {
            cancellation = control.checkpoint(step); cancellation.is_none()
        });
        if let Some(kind) = cancellation { return Err(JsonDecodeError::Cancelled(kind)); }
        result.map_err(JsonDecodeError::Mask)
    }
}

fn checked_logits(logits: &[f32], width: usize) -> Result<(), JsonDecodeError> {
    if logits.len() != width || logits.iter().any(|v| !v.is_finite()) { return Err(JsonDecodeError::InvalidLogits); }
    Ok(())
}

fn select(logits: &[f32], mask: &DenseTokenMask, accepting: bool, options: &JsonDecodeOptions) -> Result<u32, JsonDecodeError> {
    checked_logits(logits, mask.vocab_size())?;
    let mut best = None;
    for (id, &value) in logits.iter().enumerate() {
        let token = id as u32;
        let legal = if token == options.eos_token_id { accepting }
            else { mask.contains(token) && !options.excluded_token_ids.contains(&token) };
        // Ordinary comparison intentionally treats +0 and -0 as equal.
        if legal && best.is_none_or(|(_, current)| value > current) { best = Some((token, value)); }
    }
    best.map(|(id, _)| id).ok_or(JsonDecodeError::NoLegalToken)
}

#[allow(clippy::too_many_arguments)]
fn run<V: Vocabulary, C: DecodeStepControl, F: FnMut(u32) -> Result<Vec<f32>, JsonDecodeError>>(
    prompt: &[u32], program: &JsonProgram, vocabulary: &V, options: &JsonDecodeOptions,
    budget: JsonWorkBudget, profile: &str, control: &mut C, mut forward: F,
) -> Result<JsonDecodeOutput, JsonDecodeError> {
    preflight(prompt, vocabulary.width(), options, budget)?;
    let mut tokens = Vec::new();
    tokens.try_reserve_exact(options.max_new_tokens).map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(program.max_output_bytes()).map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut logits = Vec::new();
    let mut positions = 0;
    for (index, &token) in prompt.iter().enumerate() {
        if let Some(kind) = control.prefill_checkpoint(index) { return Err(JsonDecodeError::Cancelled(kind)); }
        drop(logits);
        logits = forward(token)?;
        positions += 1;
        checked_logits(&logits, vocabulary.width())?;
    }
    let mut state = program.initial_state();
    let mut charged = 0_u64;
    for step in 0..options.max_new_tokens {
        if let Some(kind) = control.checkpoint(step) { return Err(JsonDecodeError::Cancelled(kind)); }
        charged = charged.checked_add(vocabulary.mask_charge(budget.mask_limits) as u64)
            .filter(|&n| n <= budget.max_total_mask_node_visits).ok_or(JsonDecodeError::BudgetExceeded("mask work"))?;
        let mask = vocabulary.mask(&state, budget.mask_limits, control, step)?;
        let selected = select(&logits, &mask, state.is_accepting(), options)?;
        if let Some(kind) = control.checkpoint(step) { return Err(JsonDecodeError::Cancelled(kind)); }
        tokens.push(selected);
        if selected == options.eos_token_id {
            let json = String::from_utf8(bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
            program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
            return Ok(JsonDecodeOutput {
                schema_version: 1, numerics_profile: profile.to_owned(), token_ids: tokens, json,
                forward_positions: positions, projected_logits: positions * vocabulary.width() as u64,
                mask_node_visit_charge: charged,
            });
        }
        let emitted = vocabulary.bytes(selected).filter(|b| !b.is_empty()).ok_or(JsonDecodeError::IllegalTransition)?;
        if !state.consume_bytes(emitted) { return Err(JsonDecodeError::IllegalTransition); }
        bytes.extend_from_slice(emitted);
        if step + 1 == options.max_new_tokens { return Err(JsonDecodeError::BudgetExceeded("output tokens before EOS")); }
        // Always feed back nonterminal selected tokens, even if their bytes
        // already form valid JSON. The next logits must score explicit EOS.
        drop(logits);
        logits = forward(selected)?;
        positions += 1;
        checked_logits(&logits, vocabulary.width())?;
    }
    Err(JsonDecodeError::BudgetExceeded("output tokens"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{CompileLimits, runtime::JsonState};

    struct Table(Vec<Vec<u8>>);
    impl Vocabulary for Table {
        fn width(&self) -> usize { self.0.len() }
        fn bytes(&self, token: u32) -> Option<&[u8]> { self.0.get(token as usize).map(Vec::as_slice) }
        fn mask_charge(&self, _: MaskWorkLimits) -> usize { self.width() }
        fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, _: MaskWorkLimits, _: &mut C, _: usize) -> Result<DenseTokenMask, JsonDecodeError> {
            let mut mask = DenseTokenMask::empty(self.width());
            for (id, bytes) in self.0.iter().enumerate() {
                if !bytes.is_empty() && state.clone().consume_bytes(bytes) { mask.set_legal(id as u32).unwrap(); }
            }
            Ok(mask)
        }
    }
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    fn options(n: usize) -> JsonDecodeOptions { JsonDecodeOptions { max_new_tokens: n, eos_token_id: 0, excluded_token_ids: BTreeSet::new() } }
    fn budget() -> JsonWorkBudget { JsonWorkBudget {
        max_forward_positions: 100, max_projected_logits: 10000, max_kv_bytes: 100000,
        max_total_mask_node_visits: 10000, mask_limits: MaskWorkLimits::default(),
    } }
    fn program(s: &str) -> JsonProgram { JsonProgram::compile(s, CompileLimits::default()).unwrap() }

    #[test]
    fn illegal_top_logit_and_early_eos_are_masked_before_selection() {
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec(), b"malformed".to_vec()]);
        let mut seen = Vec::new();
        let out = run(&[2], &p, &v, &options(2), budget(), "fixture", &mut Continue, |token| {
            seen.push(token); Ok(vec![100.0, 1.0, 1000.0])
        }).unwrap();
        assert_eq!(out.json, "true"); assert_eq!(out.token_ids, vec![1, 0]);
        assert_eq!(seen, vec![2, 1]); assert_eq!(out.forward_positions, 2);
        assert_eq!(out.projected_logits, 6); assert_eq!(out.mask_node_visit_charge, 6);
    }

    #[test]
    fn multi_byte_tokens_can_cross_multiple_json_boundaries() {
        let p = program(r#"{"type":"object","properties":{"x":{"type":"boolean"}},"required":["x"],"additionalProperties":false}"#);
        let v = Table(vec![vec![], br#"{"x":true}"#.to_vec()]);
        let out = run(&[1], &p, &v, &options(2), budget(), "fixture", &mut Continue, |_| Ok(vec![0.0, 1.0])).unwrap();
        assert_eq!(out.json, r#"{"x":true}"#);
    }

    #[test]
    fn token_budget_never_returns_partial_or_eosless_json() {
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec()]);
        assert!(matches!(run(&[1], &p, &v, &options(1), budget(), "fixture", &mut Continue, |_| Ok(vec![0.0, 1.0])), Err(JsonDecodeError::BudgetExceeded(_))));
    }

    #[test]
    fn excluded_payload_tokens_do_not_reenter_through_grammar() {
        let p = program(r#"{"type":"boolean"}"#);
        let v = Table(vec![vec![], b"true".to_vec(), b"false".to_vec()]);
        let mut opts = options(2); opts.excluded_token_ids.insert(1);
        let out = run(&[1], &p, &v, &opts, budget(), "fixture", &mut Continue, |_| Ok(vec![0.0, 100.0, 1.0])).unwrap();
        assert_eq!(out.json, "false");
    }

    #[test]
    fn nonfinite_logits_reject_even_when_the_bad_row_is_illegal() {
        let p = program(r#"{"type":"boolean"}"#); let v = Table(vec![vec![], b"true".to_vec()]);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(run(&[1], &p, &v, &options(2), budget(), "fixture", &mut Continue, |_| Ok(vec![bad, 1.0])), Err(JsonDecodeError::InvalidLogits)));
        }
    }

    #[test]
    fn numeric_acceptance_does_not_force_shorter_prefix_candidate() {
        let p = program(r#"{"type":"integer","enum":[1,10]}"#);
        let v = Table(vec![vec![], b"1".to_vec(), b"0".to_vec()]);
        let out = run(&[1], &p, &v, &options(3), budget(), "fixture", &mut Continue, |_| Ok(vec![0.0, 2.0, 1.0])).unwrap();
        assert_eq!(out.json, "10"); assert_eq!(out.token_ids, vec![1, 2, 0]);
    }

    #[test]
    fn tie_breaking_is_lowest_token_id_including_signed_zero() {
        let mut mask = DenseTokenMask::empty(3); mask.set_legal(1).unwrap(); mask.set_legal(2).unwrap();
        assert_eq!(select(&[1.0, -0.0, 0.0], &mask, false, &options(2)).unwrap(), 1);
    }

    #[test]
    fn preflight_rejects_work_before_any_forward() {
        let p = program(r#"{"type":"boolean"}"#); let v = Table(vec![vec![], b"true".to_vec()]);
        let mut b = budget(); b.max_projected_logits = 1;
        assert!(run(&[1], &p, &v, &options(2), b, "fixture", &mut Continue, |_| panic!("must not forward")).is_err());
    }

    #[test]
    fn cancellation_during_prefill_has_no_result() {
        struct Cancel;
        impl DecodeStepControl for Cancel {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
            fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> { (index == 1).then_some(DecodeCancellationKind::Deadline) }
        }
        let p = program(r#"{"type":"boolean"}"#); let v = Table(vec![vec![], b"true".to_vec()]); let mut calls = 0;
        let out = run(&[1, 1], &p, &v, &options(2), budget(), "fixture", &mut Cancel, |_| { calls += 1; Ok(vec![0.0, 1.0]) });
        assert!(matches!(out, Err(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)))); assert_eq!(calls, 1);
    }

    #[test]
    fn aggregate_mask_work_is_enforced_across_steps() {
        let p = program(r#"{"type":"boolean"}"#); let v = Table(vec![vec![], b"true".to_vec()]); let mut b = budget(); b.max_total_mask_node_visits = 3;
        assert!(matches!(run(&[1], &p, &v, &options(2), b, "fixture", &mut Continue, |_| Ok(vec![0.0, 1.0])), Err(JsonDecodeError::BudgetExceeded("mask work"))));
    }
}
