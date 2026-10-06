//! Synthetic logits exercise the real scorer and KV traversal, NOT model parity.
use super::*;
use crate::{tasks::ir::TokenSequence, native_engine::lmhead::scoring::SequenceScoreRule};

fn candidate(id: &str, tokens: &[u32]) -> Candidate { Candidate::new(id, TokenSequence::new(tokens.to_vec())) }
fn plan(mode: ScoringMode) -> Int8CandidatePlan {
    Int8CandidatePlan::compile(vec![7, 8, 9, 10, 11], &[
        candidate("short", &[1]), candidate("long", &[1, 2, 3]),
        candidate("sibling", &[1, 4]), candidate("other", &[5, 6]),
    ], 0, mode, ScoringLimits::default(), 1_000_000).unwrap()
}
fn limits(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn logits(tokens: &[u32], rows: LinearRows<'_>) -> Vec<f32> {
    let h = tokens.iter().fold(23_u64, |h, &t| h.wrapping_mul(31).wrapping_add(u64::from(t)));
    let value = |r: u32| (h.wrapping_add(u64::from(r).wrapping_mul(13)) % 37) as f32 * 0.25;
    match rows { LinearRows::All => (0..V as u32).map(value).collect(),
        LinearRows::Selected(ids) => ids.iter().map(|&r| value(r)).collect() }
}
#[derive(Default)]
struct Model {
    tokens: Vec<u32>, work: Int8Work, forwards: Vec<(usize, u32)>, projections: Vec<Vec<u32>>,
    groups: Vec<usize>, prompt_calls: usize, aborted: bool,
    fail_at: Option<usize>, invalid_head: bool, forged_work: bool, short_prompt: bool,
}
impl Driver for Model {
    fn position(&self) -> Result<usize, Int8ScoringError> { Ok(self.tokens.len()) }
    fn rewind(&mut self, retain: usize) -> Result<(), Int8ScoringError> {
        if retain > self.tokens.len() { return Err(Int8ScoringError::Traversal); }
        self.tokens.truncate(retain); Ok(())
    }
    fn append(&mut self, token: u32) -> Result<(), Int8ScoringError> {
        if self.fail_at == Some(self.forwards.len()) {
            return Err(StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline).into());
        }
        self.work = self.work.checked_add(Int8Work::for_sequence(self.tokens.len(), 1, 0)?)?;
        self.forwards.push((self.tokens.len(), token)); self.tokens.push(token); Ok(())
    }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int8ScoringError> {
        self.work = self.work.checked_add(Int8Work::for_sequence(self.tokens.len(), 0,
            rows.checked_count(V).map_err(StrictInt8Error::from)?)?)?;
        self.projections.push(self.tokens.clone());
        let mut values = logits(&self.tokens, rows);
        if self.invalid_head { values[0] = f32::NAN; } Ok(values)
    }
    fn work(&self) -> Int8Work {
        let mut work = self.work; if self.forged_work { work.attention_pairs += 1; } work
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl PromptDriver for Model {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8ScoringError> {
        self.prompt_calls += 1;
        let prompt = if self.short_prompt { &prompt[..prompt.len() - 1] } else { prompt };
        for chunk in prompt.chunks(limits.max_batch_rows) {
            self.groups.push(chunk.len());
            for &token in chunk { self.append(token)?; }
        }
        Ok(())
    }
}
fn grouped(p: &Int8CandidatePlan, width: usize, model: &mut Model) -> Result<Int8CandidateRun, Int8ScoringError> {
    execute_layer_major_driver(&p.prompt, &p.scorer, p.mode, p.schedule, p.max_output_bytes, limits(width), model)
}
struct Replay<'a>(&'a [u32]);
impl CandidateLogits for Replay<'_> {
    type Error = Int8ScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let mut tokens = self.0.to_vec(); tokens.extend_from_slice(prefix);
        Ok(logits(&tokens, checked_rows(rows)?))
    }
}

#[test]
fn all_probability_spaces_preserve_serial_scores_and_independent_replay() {
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit }] {
        let p = plan(mode); let mut serial = Model::default();
        let expected = execute_driver(&p.prompt, &p.scorer, p.mode, p.schedule, p.max_output_bytes, &mut serial).unwrap();
        let replay = p.scorer.score(&mut Replay(&p.prompt), mode).unwrap();
        assert_eq!(expected.scores, replay);
        for width in [1, 2, 4, 64] {
            let mut model = Model::default(); let result = grouped(&p, width, &mut model).unwrap();
            assert_eq!(result, expected); assert_eq!(result.scores, replay);
            assert_eq!(model.forwards, serial.forwards); assert_eq!(model.projections, serial.projections);
            assert_eq!(model.prompt_calls, 1); assert!(!model.aborted);
            assert_eq!(model.groups, p.prompt.chunks(width).map(|c| c.len()).collect::<Vec<_>>());
        }
    }
}
#[test]
fn branching_never_reprefills_or_prices_rewinds_as_a_growing_sequence() {
    let p = plan(ScoringMode::TrieConditional); let mut model = Model::default();
    let out = grouped(&p, 4, &mut model).unwrap();
    assert_eq!(model.groups, [4, 1]); assert_eq!(model.prompt_calls, 1);
    assert_eq!(&model.forwards[5..], &[(5, 1), (6, 2), (7, 3), (6, 4), (5, 5), (6, 6)]);
    assert_eq!(out.rewound_positions, 4); assert_eq!(out.model_work.forward_positions, 11);
    assert_eq!(out.model_work.attention_pairs, (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 7 + 6 + 7) * 44 * 48);
    assert_eq!(out.scores.work.scored_edges, 10); assert_eq!(out.scores.work.prefix_evaluations, 7);
    assert!(model.forwards.iter().all(|&(_, token)| token != 0));
}
#[test]
fn every_prompt_and_continuation_cancellation_is_fatal_without_partial_scores() {
    let p = plan(ScoringMode::TrieConditional);
    for position in 0..p.planned_work().forward_positions as usize {
        let mut model = Model { fail_at: Some(position), ..Model::default() };
        let error = grouped(&p, 4, &mut model).unwrap_err();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(model.aborted); assert_eq!(model.forwards.len(), position);
        if position < p.prompt.len() { assert!(model.projections.is_empty()); }
        assert_eq!(model.prompt_calls, 1);
    }
}
#[test]
fn invalid_geometry_and_insufficient_workspace_refuse_before_any_driver_operation() {
    let p = plan(ScoringMode::TrieConditional); let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    for limits in [short, Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX }] {
        let mut model = Model::default();
        assert!(execute_layer_major_driver(&p.prompt, &p.scorer, p.mode, p.schedule,
            p.max_output_bytes, limits, &mut model).is_err());
        assert!(model.forwards.is_empty()); assert!(model.projections.is_empty());
        assert_eq!(model.prompt_calls, 0); assert!(!model.aborted);
    }
}
#[test]
fn malformed_prompt_completion_cannot_project_a_stale_hidden_state() {
    let p = plan(ScoringMode::TrieConditional); let mut model = Model { short_prompt: true, ..Model::default() };
    assert_eq!(grouped(&p, 4, &mut model).unwrap_err(), Int8ScoringError::Traversal);
    assert!(model.aborted); assert!(model.projections.is_empty());
}
#[test]
fn nonfinite_logits_and_forged_native_work_never_finalize() {
    let p = plan(ScoringMode::TrieConditional);
    for invalid_head in [false, true] {
        let mut model = Model { invalid_head, forged_work: !invalid_head, ..Model::default() };
        let error = grouped(&p, 4, &mut model).unwrap_err();
        assert!(matches!(error, Int8ScoringError::Accounting | Int8ScoringError::Scoring(_)));
        assert!(model.aborted); assert_eq!(model.prompt_calls, 1);
    }
}
#[test]
fn complete_grouped_output_obeys_the_exact_same_outer_byte_ceiling() {
    let mut p = plan(ScoringMode::TrieConditional);
    let result = grouped(&p, 4, &mut Model::default()).unwrap();
    p.max_output_bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    assert_eq!(grouped(&p, 4, &mut Model::default()).unwrap(), result);
    p.max_output_bytes -= 1; let mut model = Model::default();
    assert_eq!(grouped(&p, 4, &mut model).unwrap_err(), Int8ScoringError::OutputBudget);
    assert!(model.aborted);
}
