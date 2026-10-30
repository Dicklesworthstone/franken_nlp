//! Traversal and accounting fixtures use synthetic logits, not model evidence.
use super::*;
use crate::{tasks::ir::TokenSequence, native_engine::lmhead::scoring::SequenceScoreRule};

fn plan(group: usize, mode: ScoringMode) -> Int4CandidatePlan {
    let candidate = |id, tokens: &[u32]| Candidate::new(id, TokenSequence::new(tokens.to_vec()));
    Int4CandidatePlan::compile(vec![7, 8, 9, 10, 11], &[
        candidate("short", &[1]), candidate("long", &[1, 2, 3]),
        candidate("sibling", &[1, 4]), candidate("other", &[5, 6]),
    ], 0, mode, group, ScoringLimits::default(), 1_000_000).unwrap()
}
fn limits(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn logits(tokens: &[u32], rows: LinearRows<'_>) -> Vec<f32> {
    let h = tokens.iter().fold(23_u64, |h, &t| h.wrapping_mul(31).wrapping_add(u64::from(t)));
    let value = |r: u32| (h.wrapping_add(u64::from(r).wrapping_mul(13)) % 37) as f32 * 0.25;
    match rows { LinearRows::All => (0..V as u32).map(value).collect(),
        LinearRows::Selected(ids) => ids.iter().map(|&r| value(r)).collect() }
}
struct Model {
    group: usize, tokens: Vec<u32>, work: Int4Work, forwards: Vec<(usize, u32)>,
    projections: Vec<Vec<u32>>, groups: Vec<usize>, prompt_calls: usize,
    aborted: bool, fail_at: Option<usize>, short_prompt: bool, forged_work: bool,
}
impl Model {
    fn new(group: usize) -> Self {
        Self { group, tokens: Vec::new(), work: Int4Work::default(), forwards: Vec::new(),
            projections: Vec::new(), groups: Vec::new(), prompt_calls: 0, aborted: false,
            fail_at: None, short_prompt: false, forged_work: false }
    }
}
impl Driver for Model {
    fn checkpoint(&mut self) -> Result<(), Int4ScoringError> { Ok(()) }
    fn position(&self) -> Result<usize, Int4ScoringError> { Ok(self.tokens.len()) }
    fn rewind(&mut self, retain: usize) -> Result<(), Int4ScoringError> {
        if retain > self.tokens.len() { return Err(Int4ScoringError::Traversal); }
        self.tokens.truncate(retain); Ok(())
    }
    fn append(&mut self, token: u32) -> Result<(), Int4ScoringError> {
        if self.fail_at == Some(self.forwards.len()) {
            return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline).into());
        }
        self.work = self.work.checked_add(Int4Work::for_sequence(self.group, self.tokens.len(), 1, 0)?)?;
        self.forwards.push((self.tokens.len(), token)); self.tokens.push(token); Ok(())
    }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError> {
        self.work = self.work.checked_add(Int4Work::for_sequence(self.group, self.tokens.len(), 0,
            rows.checked_count(V).map_err(StrictInt4Error::from)?)?)?;
        self.projections.push(self.tokens.clone()); Ok(logits(&self.tokens, rows))
    }
    fn work(&self) -> Int4Work {
        let mut work = self.work; if self.forged_work { work.projections.dot_products += 1; } work
    }
    fn abort(&mut self) { self.aborted = true; }
}
impl PromptDriver for Model {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int4PrefillLimits) -> Result<(), Int4ScoringError> {
        self.prompt_calls += 1;
        let prompt = if self.short_prompt { &prompt[..prompt.len() - 1] } else { prompt };
        for chunk in prompt.chunks(limits.max_batch_rows) {
            self.groups.push(chunk.len()); for &token in chunk { self.append(token)?; }
        }
        Ok(())
    }
}
fn grouped(p: &Int4CandidatePlan, rows: usize, model: &mut Model) -> Result<Int4CandidateRun, Int4ScoringError> {
    execute_layer_major_driver(&p.prompt, &p.scorer, p.mode, p.schedule, p.max_output_bytes, limits(rows), model)
}
struct Replay<'a>(&'a [u32]);
impl CandidateLogits for Replay<'_> {
    type Error = Int4ScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let mut tokens = self.0.to_vec(); tokens.extend_from_slice(prefix);
        Ok(logits(&tokens, checked_rows(rows)?))
    }
}
#[test]
fn all_score_spaces_and_group_partitions_preserve_serial_and_independent_replay() {
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit }] {
        for group in [3, 32, 128] {
            let p = plan(group, mode); let mut serial = Model::new(group);
            let expected = execute_driver(&p.prompt, &p.scorer, p.mode, p.schedule, p.max_output_bytes, &mut serial).unwrap();
            assert_eq!(expected.scores, p.scorer.score(&mut Replay(&p.prompt), mode).unwrap());
            for rows in [1, 2, 4, 64] {
                let mut model = Model::new(group);
                assert_eq!(grouped(&p, rows, &mut model).unwrap(), expected);
                assert_eq!(model.forwards, serial.forwards); assert_eq!(model.projections, serial.projections);
                assert_eq!(model.prompt_calls, 1); assert!(!model.aborted);
                assert_eq!(model.groups, p.prompt.chunks(rows).map(|c| c.len()).collect::<Vec<_>>());
            }
        }
    }
}
#[test]
fn branch_rewinds_never_repeat_prompt_or_inflate_causal_attention() {
    let p = plan(3, ScoringMode::TrieConditional); let mut model = Model::new(3);
    let run = grouped(&p, 4, &mut model).unwrap();
    assert_eq!(model.groups, [4, 1]); assert_eq!(model.prompt_calls, 1);
    assert_eq!(&model.forwards[5..], &[(5, 1), (6, 2), (7, 3), (6, 4), (5, 5), (6, 6)]);
    assert_eq!(run.rewound_positions, 4); assert_eq!(run.model_work.forward_positions, 11);
    assert_eq!(run.model_work.attention_pairs, (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 7 + 6 + 7) * 44 * 48);
    assert_eq!(run.scores.work.scored_edges, 10); assert_eq!(run.scores.work.prefix_evaluations, 7);
    assert!(model.forwards.iter().all(|&(_, token)| token != 0));
}
#[test]
fn every_prompt_and_continuation_cancellation_discards_partial_scores() {
    let p = plan(128, ScoringMode::TrieConditional);
    for position in 0..p.planned_work().forward_positions as usize {
        let mut model = Model::new(128); model.fail_at = Some(position);
        assert_eq!(grouped(&p, 4, &mut model).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(model.aborted); assert_eq!(model.forwards.len(), position);
        if position < p.prompt.len() { assert!(model.projections.is_empty()); }
        assert_eq!(model.prompt_calls, 1);
    }
}
#[test]
fn malformed_rows_or_underpriced_scratch_refuse_before_driver_access() {
    let p = plan(32, ScoringMode::TrieConditional); let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    for limits in [short, Int4PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX }] {
        let mut model = Model::new(32);
        assert!(execute_layer_major_driver(&p.prompt, &p.scorer, p.mode, p.schedule,
            p.max_output_bytes, limits, &mut model).is_err());
        assert!(model.forwards.is_empty()); assert!(model.projections.is_empty());
        assert_eq!(model.prompt_calls, 0); assert!(!model.aborted);
    }
}
#[test]
fn incomplete_prompt_and_forged_group_work_cannot_finalize() {
    let p = plan(32, ScoringMode::TrieConditional);
    let mut model = Model::new(32); model.short_prompt = true;
    assert_eq!(grouped(&p, 4, &mut model).unwrap_err(), Int4ScoringError::Traversal);
    assert!(model.aborted); assert!(model.projections.is_empty());
    let mut model = Model::new(32); model.forged_work = true;
    assert_eq!(grouped(&p, 4, &mut model).unwrap_err(), Int4ScoringError::Accounting);
    assert!(model.aborted);
    let mut model = Model::new(16);
    assert_eq!(grouped(&p, 4, &mut model).unwrap_err(), Int4ScoringError::Accounting);
    assert!(model.aborted);
}
#[test]
fn complete_grouped_output_obeys_the_same_exact_byte_limit() {
    let mut p = plan(32, ScoringMode::TrieConditional);
    let result = grouped(&p, 4, &mut Model::new(32)).unwrap();
    p.max_output_bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    assert_eq!(grouped(&p, 4, &mut Model::new(32)).unwrap(), result);
    p.max_output_bytes -= 1; let mut model = Model::new(32);
    assert_eq!(grouped(&p, 4, &mut model).unwrap_err(), Int4ScoringError::OutputBudget);
    assert!(model.aborted);
}
