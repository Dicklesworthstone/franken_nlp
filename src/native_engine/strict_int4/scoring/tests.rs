//! The real CandidateScorer against stateless prefix replay with synthetic
//! logits. These are traversal/accounting tests, NOT full-model INT4 evidence.
use super::*;
use crate::tasks::ir::TokenSequence;
use crate::native_engine::lmhead::scoring::SequenceScoreRule;
use super::super::{H, PROJECTION_SHAPES};

fn candidate(id: &str, tokens: &[u32]) -> Candidate { Candidate::new(id, TokenSequence::new(tokens.to_vec())) }
fn language() -> Vec<Candidate> {
    vec![candidate("short", &[1]), candidate("long", &[1, 2, 3]),
        candidate("sibling", &[1, 4]), candidate("other", &[5, 6])]
}
fn plan(width: usize, mode: ScoringMode) -> Int4CandidatePlan {
    Int4CandidatePlan::compile(vec![7, 8, 9], &language(), 0, mode, width,
        ScoringLimits::default(), 1_000_000).unwrap()
}
fn logits(tokens: &[u32], rows: LinearRows<'_>) -> Vec<f32> {
    let hash = tokens.iter().fold(17_u64, |h, &t| h.wrapping_mul(31).wrapping_add(u64::from(t)));
    let value = |row: u32| ((hash.wrapping_add(u64::from(row).wrapping_mul(13))) % 37) as f32 * 0.25;
    match rows { LinearRows::All => (0..V as u32).map(value).collect(),
        LinearRows::Selected(ids) => ids.iter().map(|&r| value(r)).collect() }
}
struct Model {
    width: usize, tokens: Vec<u32>, work: Int4Work, forwards: Vec<(usize, u32)>,
    projections: Vec<Vec<u32>>, aborted: bool, cancel_at: Option<usize>,
    corrupt: bool, forge: bool, polls: usize, cancel_poll: Option<usize>,
}
impl Model {
    fn new(width: usize) -> Self {
        Self { width, tokens: Vec::new(), work: Int4Work::default(), forwards: Vec::new(),
            projections: Vec::new(), aborted: false, cancel_at: None, corrupt: false,
            forge: false, polls: 0, cancel_poll: None }
    }
}
impl Driver for Model {
    fn checkpoint(&mut self) -> Result<(), Int4ScoringError> {
        self.polls += 1;
        if self.cancel_poll == Some(self.polls) {
            return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::User).into());
        }
        Ok(())
    }
    fn position(&self) -> Result<usize, Int4ScoringError> { Ok(self.tokens.len()) }
    fn rewind(&mut self, retain: usize) -> Result<(), Int4ScoringError> {
        assert!(retain <= self.tokens.len()); self.tokens.truncate(retain); Ok(())
    }
    fn append(&mut self, token: u32) -> Result<(), Int4ScoringError> {
        if self.cancel_at == Some(self.forwards.len()) {
            return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline).into());
        }
        self.work = self.work.checked_add(Int4Work::for_sequence(self.width, self.tokens.len(), 1, 0)?)?;
        self.forwards.push((self.tokens.len(), token)); self.tokens.push(token); Ok(())
    }
    fn logits(&mut self, rows: LinearRows<'_>) -> Result<Vec<f32>, Int4ScoringError> {
        self.projections.push(self.tokens.clone());
        self.work = self.work.checked_add(Int4Work::for_sequence(self.width, self.tokens.len(), 0,
            rows.checked_count(V).map_err(StrictInt4Error::from)?)?)?;
        let mut values = logits(&self.tokens, rows);
        if self.corrupt { values[0] = f32::NAN; }
        Ok(values)
    }
    fn work(&self) -> Int4Work {
        let mut w = self.work; if self.forge { w.attention_pairs += 1; } w
    }
    fn abort(&mut self) { self.aborted = true; }
}
fn run(p: &Int4CandidatePlan, model: &mut Model) -> Result<Int4CandidateRun, Int4ScoringError> {
    execute_driver(&p.prompt, &p.scorer, p.mode, p.schedule, p.max_output_bytes, model)
}
struct Replay<'a>(&'a [u32]);
impl CandidateLogits for Replay<'_> {
    type Error = Int4ScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let mut full = self.0.to_vec(); full.extend_from_slice(prefix); Ok(logits(&full, checked_rows(rows)?))
    }
}

#[test]
fn every_score_space_matches_independent_stateless_prefix_replay() {
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit }] {
        for width in [3, 16, 32, 128] {
            let p = plan(width, mode); let mut model = Model::new(width);
            let result = run(&p, &mut model).unwrap();
            assert_eq!(result.scores, p.scorer.score(&mut Replay(&p.prompt), mode).unwrap());
            assert_eq!(result.model_work, p.planned_work());
            assert_eq!(result.group_width, width); assert_eq!(result.numerics_profile, STRICT_INT4_PROFILE);
            assert!(!model.aborted);
        }
    }
}
#[test]
fn prefix_sharing_forwards_every_unique_nonempty_prefix_once_without_forwarding_eos() {
    let p = plan(32, ScoringMode::TrieConditional); let mut model = Model::new(32);
    let result = run(&p, &mut model).unwrap();
    assert_eq!(model.forwards, [(0, 7), (1, 8), (2, 9), (3, 1), (4, 2), (5, 3), (4, 4), (3, 5), (4, 6)]);
    assert_eq!(result.rewound_positions, 4); assert_eq!(model.tokens, [7, 8, 9, 5, 6]);
    assert_eq!(result.model_work.forward_positions, 9);
    assert!(model.forwards.iter().all(|&(_, token)| token != 0));
}
#[test]
fn branch_attention_is_priced_at_causal_depth_not_total_forward_count() {
    let p = plan(128, ScoringMode::FullVocabulary);
    assert_eq!(p.planned_work().attention_pairs, (1 + 2 + 3 + 4 + 5 + 6 + 5 + 4 + 5) * 44 * 48);
    assert!(p.planned_work().attention_pairs < Int4Work::for_sequence(128, 0, 9, 0).unwrap().attention_pairs);
    assert_eq!(p.scoring_work().prefix_evaluations, 7); assert_eq!(p.scoring_work().scored_edges, 10);
}
#[test]
fn all_group_partitions_price_unpadded_macs_and_each_actual_group_dot() {
    let candidates = language();
    for width in 1..=MAX_MODEL_K {
        let s = CandidateSchedule::new(width, 3, &candidates, ScoringMode::TrieConditional).unwrap();
        let decoder_dots: u64 = PROJECTION_SHAPES.iter().map(|&(n, k)| (n * k.div_ceil(width)) as u64).sum::<u64>() * 44;
        let decoder_macs: u64 = PROJECTION_SHAPES.iter().map(|&(n, k)| (n * k) as u64).sum::<u64>() * 44;
        assert_eq!(s.model.projections.dot_products, 9 * decoder_dots + 10 * H.div_ceil(width) as u64);
        assert_eq!(s.model.projections.multiply_accumulates, 9 * decoder_macs + 10 * H as u64);
        assert_eq!(s.context, 6);
    }
}
#[test]
fn selected_modes_do_not_price_or_project_the_unrequested_vocabulary() {
    for width in [1, 3, 16, 32, 128, MAX_MODEL_K] {
        let full = plan(width, ScoringMode::FullVocabulary); let selected = plan(width, ScoringMode::TrieConditional);
        let omitted = 7 * V as u64 - 10;
        assert_eq!(full.planned_work().projected_logits, 7 * V as u64);
        assert_eq!(selected.planned_work().projected_logits, 10);
        assert_eq!(full.planned_work().projections.dot_products - selected.planned_work().projections.dot_products,
            omitted * H.div_ceil(width) as u64);
        assert_eq!(full.planned_work().projections.multiply_accumulates - selected.planned_work().projections.multiply_accumulates,
            omitted * H as u64);
    }
}
#[test]
fn input_permutation_preserves_ids_scores_and_work() {
    let expected = plan(3, ScoringMode::TrieConditional); let mut values = language();
    let baseline = run(&expected, &mut Model::new(3)).unwrap();
    for shift in 0..values.len() {
        values.rotate_left(shift); values.reverse();
        let p = Int4CandidatePlan::compile(vec![7, 8, 9], &values, 0, ScoringMode::TrieConditional, 3,
            ScoringLimits::default(), 1_000_000).unwrap();
        assert_eq!(run(&p, &mut Model::new(3)).unwrap(), baseline);
    }
}
#[test]
fn nested_candidates_score_the_shorter_candidates_eos_edges() {
    let p = Int4CandidatePlan::compile(vec![9], &[candidate("a", &[1]), candidate("b", &[1, 2]), candidate("c", &[1, 2, 3])],
        0, ScoringMode::TrieConditional, 16, ScoringLimits::default(), 1_000_000).unwrap();
    let result = run(&p, &mut Model::new(16)).unwrap();
    assert_eq!(result.scores.candidates.iter().map(|c| c.scored_tokens).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(result.rewound_positions, 0); assert_eq!(result.scores.work.scored_edges, 6);
}
#[test]
fn invalid_prompt_partition_language_eos_and_duplicate_encodings_refuse() {
    for prompt in [vec![], vec![V as u32], vec![1; DEFAULT_ADMITTED_CONTEXT_CAP]] {
        assert!(Int4CandidatePlan::compile(prompt, &language(), 0, ScoringMode::TrieConditional, 32,
            ScoringLimits::default(), 1_000_000).is_err());
    }
    for width in [0, MAX_MODEL_K + 1, usize::MAX] {
        assert!(Int4CandidatePlan::compile(vec![9], &language(), 0, ScoringMode::TrieConditional, width,
            ScoringLimits::default(), 1_000_000).is_err());
    }
    for candidates in [vec![], vec![candidate("a", &[])], vec![candidate("a", &[0])],
        vec![candidate("a", &[1]), candidate("b", &[1])], vec![candidate("a", &[1]), candidate("a", &[2])]] {
        assert!(Int4CandidatePlan::compile(vec![9], &candidates, 0, ScoringMode::TrieConditional, 32,
            ScoringLimits::default(), 1_000_000).is_err());
    }
}
#[test]
fn full_vocabulary_budget_refuses_even_when_selected_rows_fit() {
    let limits = ScoringLimits { max_projected_logits: 10, ..ScoringLimits::default() };
    assert!(Int4CandidatePlan::compile(vec![9], &language(), 0, ScoringMode::TrieConditional, 32, limits, 1_000_000).is_ok());
    assert!(Int4CandidatePlan::compile(vec![9], &language(), 0, ScoringMode::FullVocabulary, 32, limits, 1_000_000).is_err());
}
#[test]
fn whole_capacity_and_every_native_budget_axis_are_preflighted() {
    for width in [3, 16, 128] {
        let p = plan(width, ScoringMode::FullVocabulary); let s = p.schedule;
        let b = Int4ScoringBudget { native: Int4RunBudget::exact(s.model), max_kv_bytes: s.context as u64 * KV_BYTES_PER_TOKEN as u64 };
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
}
#[test]
fn cancellation_cause_survives_every_forward_boundary_with_no_partial_result() {
    let p = plan(32, ScoringMode::TrieConditional);
    for at in 0..p.planned_work().forward_positions as usize {
        let mut model = Model::new(32); model.cancel_at = Some(at);
        let error = run(&p, &mut model).unwrap_err();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert_eq!(model.forwards.len(), at); assert!(model.aborted);
    }
}
#[test]
fn cancellation_after_final_projection_or_serialization_is_not_success() {
    let p = plan(32, ScoringMode::TrieConditional); let mut completed = Model::new(32);
    run(&p, &mut completed).unwrap();
    for at in [1, completed.polls - 1, completed.polls] {
        let mut model = Model::new(32); model.cancel_poll = Some(at);
        assert_eq!(run(&p, &mut model).unwrap_err().cancellation(), Some(DecodeCancellationKind::User));
        assert!(model.aborted);
    }
}
#[test]
fn corrupt_logits_wrong_partition_work_and_forged_counters_abort() {
    let p = plan(32, ScoringMode::TrieConditional);
    let mut model = Model::new(32); model.corrupt = true;
    assert!(matches!(run(&p, &mut model), Err(Int4ScoringError::Scoring(ScoringError::NonFiniteLogit { .. }))));
    assert!(model.aborted);
    let mut model = Model::new(32); model.forge = true;
    assert!(matches!(run(&p, &mut model), Err(Int4ScoringError::Accounting))); assert!(model.aborted);
    let mut model = Model::new(16);
    assert!(matches!(run(&p, &mut model), Err(Int4ScoringError::Accounting))); assert!(model.aborted);
}
#[test]
fn complete_envelope_obeys_exact_output_limit() {
    let mut p = plan(32, ScoringMode::TrieConditional);
    let result = run(&p, &mut Model::new(32)).unwrap();
    p.max_output_bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    assert!(run(&p, &mut Model::new(32)).is_ok());
    p.max_output_bytes -= 1; let mut model = Model::new(32);
    assert!(matches!(run(&p, &mut model), Err(Int4ScoringError::OutputBudget))); assert!(model.aborted);
}
#[test]
fn independent_heads_sum_work_without_cross_head_attention_or_overflow() {
    let w = plan(32, ScoringMode::TrieConditional).planned_work(); let twice = w.checked_add(w).unwrap();
    assert_eq!(twice.attention_pairs, 2 * w.attention_pairs);
    assert_eq!(twice.projections.dot_products, 2 * w.projections.dot_products);
    assert!(w.checked_add(Int4Work { attention_pairs: u64::MAX, ..Int4Work::default() }).is_err());
}
#[test]
fn repeated_backward_and_nonroot_initial_queries_never_reuse_stale_hidden() {
    let mut model = Model::new(32);
    let mut backend = PrefixEvaluator { prompt: &[7, 8], previous: Vec::with_capacity(3), primed: false,
        poisoned: false, max_prefix: 3, evaluations: 0, rewound: 0, last_error: None, driver: &mut model };
    backend.project(&[], ProjectionRows::Selected(&[1])).unwrap();
    backend.project(&[1, 2], ProjectionRows::Selected(&[0])).unwrap();
    assert!(matches!(backend.project(&[1], ProjectionRows::Selected(&[0])), Err(Int4ScoringError::Traversal)));
    assert!(backend.poisoned); assert!(backend.driver.aborted);
    assert!(backend.project(&[3], ProjectionRows::Selected(&[0])).is_err());
    let mut model = Model::new(32);
    let mut backend = PrefixEvaluator { prompt: &[7, 8], previous: Vec::with_capacity(3), primed: false,
        poisoned: false, max_prefix: 3, evaluations: 0, rewound: 0, last_error: None, driver: &mut model };
    assert!(matches!(backend.project(&[1], ProjectionRows::Selected(&[0])), Err(Int4ScoringError::Traversal)));
    assert!(backend.driver.forwards.is_empty());
}
