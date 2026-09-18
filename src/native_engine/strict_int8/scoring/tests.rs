//! Real CandidateScorer vs independent stateless prefix replay; synthetic logits.
use super::*;
use super::super::H;
use crate::tasks::ir::TokenSequence;
use crate::native_engine::lmhead::scoring::SequenceScoreRule;

fn candidates() -> Vec<Candidate> {
    vec![candidate("short", &[1]), candidate("long", &[1, 2, 3]),
        candidate("sibling", &[1, 4]), candidate("other", &[5, 6])]
}
fn candidate(id: &str, tokens: &[u32]) -> Candidate { Candidate::new(id, TokenSequence::new(tokens.to_vec())) }
fn plan(mode: ScoringMode) -> Int8CandidatePlan {
    Int8CandidatePlan::compile(vec![7, 8, 9], &candidates(), 0, mode, ScoringLimits::default(), 1_000_000).unwrap()
}
fn logits(tokens: &[u32], rows: LinearRows<'_>) -> Vec<f32> {
    let hash = tokens.iter().fold(17_u64, |h, &t| h.wrapping_mul(31).wrapping_add(u64::from(t)));
    let value = |row: u32| ((hash.wrapping_add(u64::from(row).wrapping_mul(13))) % 37) as f32 * 0.25;
    match rows { LinearRows::All => (0..V as u32).map(value).collect(),
        LinearRows::Selected(ids) => ids.iter().map(|&r| value(r)).collect() }
}
#[derive(Default)]
struct Model {
    tokens: Vec<u32>, work: Int8Work, forwards: Vec<(usize, u32)>, projections: Vec<Vec<u32>>,
    aborted: bool, cancel_at: Option<usize>, corrupt: bool, forge: bool,
}
impl Driver for Model {
    fn position(&self) -> Result<usize, Int8ScoringError> { Ok(self.tokens.len()) }
    fn rewind(&mut self, retain: usize) -> Result<(), Int8ScoringError> {
        assert!(retain <= self.tokens.len()); self.tokens.truncate(retain); Ok(())
    }
    fn append(&mut self, token: u32) -> Result<(), Int8ScoringError> {
        if self.cancel_at == Some(self.forwards.len()) {
            return Err(StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline).into());
        }
        self.work = self.work.checked_add(Int8Work::for_sequence(self.tokens.len(), 1, 0)?)?;
        self.forwards.push((self.tokens.len(), token)); self.tokens.push(token); Ok(())
    }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int8ScoringError> {
        self.projections.push(self.tokens.clone());
        self.work = self.work.checked_add(Int8Work::for_sequence(self.tokens.len(), 0,
            rows.checked_count(V).map_err(StrictInt8Error::from)?)?)?;
        let mut values = logits(&self.tokens, rows);
        if self.corrupt { values[0] = f32::NAN; }
        Ok(values)
    }
    fn work(&self) -> Int8Work {
        let mut w = self.work; if self.forge { w.attention_pairs += 1; } w
    }
    fn abort(&mut self) { self.aborted = true; }
}
fn run(p: &Int8CandidatePlan, model: &mut Model) -> Result<Int8CandidateRun, Int8ScoringError> {
    execute_driver(&p.prompt, &p.scorer, p.mode, p.schedule, p.max_output_bytes, model)
}
struct Replay<'a>(&'a [u32]);
impl CandidateLogits for Replay<'_> {
    type Error = Int8ScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let mut full = self.0.to_vec(); full.extend_from_slice(prefix); Ok(logits(&full, checked_rows(rows)?))
    }
}

#[test]
fn all_score_spaces_match_independent_stateless_replay() {
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit }] {
        let p = plan(mode); let mut model = Model::default();
        let run = run(&p, &mut model).unwrap();
        let oracle = p.scorer.score(&mut Replay(&p.prompt), mode).unwrap();
        assert_eq!(run.scores, oracle); assert_eq!(run.model_work, p.planned_work());
        assert_eq!(run.numerics_profile, STRICT_INT8_PROFILE); assert!(!model.aborted);
    }
}
#[test]
fn one_prefill_then_each_unique_nonempty_prefix_is_forwarded_once() {
    let p = plan(ScoringMode::TrieConditional); let mut model = Model::default();
    let output = run(&p, &mut model).unwrap();
    assert_eq!(&model.forwards[..3], &[(0, 7), (1, 8), (2, 9)]);
    assert_eq!(&model.forwards[3..], &[(3, 1), (4, 2), (5, 3), (4, 4), (3, 5), (4, 6)]);
    assert_eq!(output.rewound_positions, 4); assert_eq!(model.tokens, [7, 8, 9, 5, 6]);
    assert_eq!(output.model_work.forward_positions, 9);
    assert!(model.forwards.iter().all(|&(_, token)| token != 0));
}
#[test]
fn branch_attention_prices_actual_causal_depth_not_total_forward_count() {
    let p = plan(ScoringMode::FullVocabulary);
    let expected = (1 + 2 + 3 + 4 + 5 + 6 + 5 + 4 + 5) * 44 * 48;
    assert_eq!(p.planned_work().attention_pairs, expected);
    assert!(p.planned_work().attention_pairs < Int8Work::for_sequence(0, 9, 0).unwrap().attention_pairs);
    assert_eq!(p.scoring_work().prefix_evaluations, 7);
    assert_eq!(p.scoring_work().scored_edges, 10);
}
#[test]
fn selected_modes_price_only_real_selected_head_rows() {
    let full = plan(ScoringMode::FullVocabulary); let sliced = plan(ScoringMode::TrieConditional);
    assert_eq!(full.planned_work().projected_logits, 7 * V as u64);
    assert_eq!(sliced.planned_work().projected_logits, 10);
    assert_eq!(full.planned_work().attention_pairs, sliced.planned_work().attention_pairs);
    assert_eq!(full.planned_work().projections.multiply_accumulates - sliced.planned_work().projections.multiply_accumulates,
        (7 * V as u64 - 10) * H as u64);
}
#[test]
fn input_permutation_does_not_change_scores_schedule_or_original_candidate_ids() {
    let expected = plan(ScoringMode::TrieConditional); let mut values = candidates();
    let baseline = run(&expected, &mut Model::default()).unwrap();
    for shift in 0..values.len() {
        values.rotate_left(shift); values.reverse();
        let p = Int8CandidatePlan::compile(vec![7, 8, 9], &values, 0, ScoringMode::TrieConditional,
            ScoringLimits::default(), 1_000_000).unwrap();
        assert_eq!(run(&p, &mut Model::default()).unwrap(), baseline);
    }
}
#[test]
fn nested_candidates_preserve_shorter_candidate_eos_edges() {
    let p = Int8CandidatePlan::compile(vec![9], &[candidate("a", &[1]), candidate("b", &[1, 2]), candidate("c", &[1, 2, 3])],
        0, ScoringMode::TrieConditional, ScoringLimits::default(), 1_000_000).unwrap();
    let output = run(&p, &mut Model::default()).unwrap();
    assert_eq!(output.scores.candidates.iter().map(|c| c.scored_tokens).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(output.rewound_positions, 0); assert_eq!(output.scores.work.scored_edges, 6);
}
#[test]
fn invalid_prompt_language_eos_and_duplicate_encodings_refuse_before_execution() {
    for prompt in [vec![], vec![V as u32], vec![1; DEFAULT_ADMITTED_CONTEXT_CAP]] {
        assert!(Int8CandidatePlan::compile(prompt, &candidates(), 0, ScoringMode::TrieConditional,
            ScoringLimits::default(), 1_000_000).is_err());
    }
    for language in [vec![], vec![candidate("a", &[])], vec![candidate("a", &[0])],
        vec![candidate("a", &[1]), candidate("b", &[1])], vec![candidate("a", &[1]), candidate("a", &[2])]] {
        assert!(Int8CandidatePlan::compile(vec![9], &language, 0, ScoringMode::TrieConditional,
            ScoringLimits::default(), 1_000_000).is_err());
    }
}
#[test]
fn full_vocabulary_cost_can_refuse_while_selected_mode_fits() {
    let limits = ScoringLimits { max_projected_logits: 10, ..ScoringLimits::default() };
    assert!(Int8CandidatePlan::compile(vec![9], &candidates(), 0, ScoringMode::TrieConditional, limits, 1_000_000).is_ok());
    assert!(Int8CandidatePlan::compile(vec![9], &candidates(), 0, ScoringMode::FullVocabulary, limits, 1_000_000).is_err());
}
#[test]
fn exact_context_memory_and_every_native_budget_axis_are_preflighted() {
    let p = plan(ScoringMode::FullVocabulary); let s = p.schedule;
    let b = Int8ScoringBudget { native: Int8RunBudget::exact(s.model), max_kv_bytes: s.context as u64 * KV_BYTES_PER_TOKEN as u64 };
    check_capacity(s, s.context, b).unwrap();
    assert!(check_capacity(s, s.context - 1, b).is_err()); assert!(check_capacity(s, s.context + 1, b).is_err());
    for axis in 0..5 {
        let mut lower = b;
        match axis { 0 => lower.max_kv_bytes -= 1, 1 => lower.native.max_forward_positions -= 1,
            2 => lower.native.max_attention_pairs -= 1, 3 => lower.native.max_projection_work.dot_products -= 1,
            _ => lower.native.max_projection_work.multiply_accumulates -= 1 }
        assert!(check_capacity(s, s.context, lower).is_err());
    }
}
#[test]
fn cancellation_keeps_cause_and_never_returns_partial_candidate_scores() {
    let p = plan(ScoringMode::TrieConditional); let mut model = Model { cancel_at: Some(4), ..Model::default() };
    let error = run(&p, &mut model).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert_eq!(model.forwards.len(), 4); assert!(model.aborted);
}
#[test]
fn corrupt_projection_and_counter_mismatch_abort_the_native_driver() {
    let p = plan(ScoringMode::TrieConditional);
    let mut model = Model { corrupt: true, ..Model::default() };
    assert!(matches!(run(&p, &mut model), Err(Int8ScoringError::Scoring(ScoringError::NonFiniteLogit { .. }))));
    assert!(model.aborted);
    let mut model = Model { forge: true, ..Model::default() };
    assert!(matches!(run(&p, &mut model), Err(Int8ScoringError::Accounting))); assert!(model.aborted);
}
#[test]
fn complete_native_envelope_obeys_exact_output_bound() {
    let mut p = plan(ScoringMode::TrieConditional);
    let out = run(&p, &mut Model::default()).unwrap(); let size = canonjson::canonical_bytes(&out).unwrap().len() as u64;
    p.max_output_bytes = size; assert!(run(&p, &mut Model::default()).is_ok());
    p.max_output_bytes -= 1; let mut model = Model::default();
    assert!(matches!(run(&p, &mut model), Err(Int8ScoringError::OutputBudget))); assert!(model.aborted);
}
#[test]
fn independent_head_work_sums_do_not_invent_cross_head_attention() {
    let p = plan(ScoringMode::TrieConditional); let w = p.planned_work(); let twice = w.checked_add(w).unwrap();
    assert_eq!(twice.attention_pairs, 2 * w.attention_pairs);
    assert_eq!(twice.projections.multiply_accumulates, 2 * w.projections.multiply_accumulates);
    assert!(w.checked_add(Int8Work { attention_pairs: u64::MAX, ..Int8Work::default() }).is_err());
}
#[test]
fn repeated_or_backward_prefix_queries_fail_instead_of_using_stale_hidden() {
    let mut model = Model::default();
    let mut backend = PrefixEvaluator { prompt: &[7, 8], previous: Vec::with_capacity(3), primed: false,
        poisoned: false, max_prefix: 3, evaluations: 0, rewound: 0, last_error: None, driver: &mut model };
    backend.project(&[], ProjectionRows::Selected(&[1])).unwrap();
    backend.project(&[1, 2], ProjectionRows::Selected(&[0])).unwrap();
    assert!(matches!(backend.project(&[1], ProjectionRows::Selected(&[0])), Err(Int8ScoringError::Traversal)));
    assert!(backend.poisoned); assert!(backend.driver.aborted);
    assert!(backend.project(&[3], ProjectionRows::Selected(&[0])).is_err());
}
