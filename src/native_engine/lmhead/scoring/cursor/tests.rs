//! Compare against the frozen pre-cursor implementation, not another cursor.
use super::*;
use crate::tasks::ir::TokenSequence;

fn modes() -> [ScoringMode; 5] {
    [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit }]
}
fn candidate(id: &str, tokens: &[u32]) -> Candidate { Candidate::new(id, TokenSequence::new(tokens.to_vec())) }
fn language() -> Vec<Candidate> {
    vec![candidate("short", &[1]), candidate("long", &[1, 2, 3]), candidate("sibling", &[1, 4]),
        candidate("other", &[5, 6]), candidate("nested", &[1, 2])]
}
fn scorer() -> CandidateScorer { CandidateScorer::compile(&language(), 8, 0, ScoringLimits::default()).unwrap() }
#[derive(Default)]
struct Model { calls: Vec<Vec<u32>>, seed: u64, constant: Option<f32> }
impl CandidateLogits for Model {
    type Error = &'static str;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        self.calls.push(prefix.to_vec());
        let h = prefix.iter().fold(self.seed, |h, &t| h.wrapping_mul(31).wrapping_add(u64::from(t)));
        let value = |t: u32| self.constant.unwrap_or((h.wrapping_add(u64::from(t) * 13) % 71) as f32 * 0.25 - 10.0);
        Ok(match rows { ProjectionRows::FullVocabulary { vocabulary_size } => (0..vocabulary_size as u32).map(value).collect(),
            ProjectionRows::Selected(ids) => ids.iter().map(|&id| value(id)).collect() })
    }
}
fn advance(cursor: &mut CandidateScoreCursor<'_>, model: &mut Model) -> bool {
    let Some(request) = cursor.request().unwrap() else { return false };
    let ordinal = request.ordinal;
    let values = model.project(request.prefix, request.rows).unwrap();
    cursor.accept(ordinal, &values).unwrap(); true
}

#[test]
fn all_five_modes_match_frozen_serial_arithmetic_and_projection_order() {
    let scorer = scorer();
    for mode in modes() {
        for seed in 0..17 {
            let mut old = Model { seed, ..Model::default() };
            let expected = scorer.score_serial_reference(&mut old, mode).unwrap();
            let mut current = Model { seed, ..Model::default() };
            assert_eq!(scorer.score(&mut current, mode).unwrap(), expected);
            assert_eq!(current.calls, old.calls);
            let mut direct = Model { seed, ..Model::default() };
            let mut cursor = scorer.cursor(mode).unwrap();
            assert_eq!(cursor.planned_work(), expected.work);
            while advance(&mut cursor, &mut direct) {}
            assert_eq!(cursor.finish().unwrap(), expected); assert_eq!(direct.calls, old.calls);
        }
    }
}
#[test]
fn ragged_interleaving_cannot_mix_candidate_normalizers_or_row_state() {
    let a = scorer();
    let b = CandidateScorer::compile(&[candidate("x", &[2]), candidate("y", &[3])], 8, 0, ScoringLimits::default()).unwrap();
    let mut rows = [a.cursor(ScoringMode::FullVocabulary).unwrap(), b.cursor(ScoringMode::TrieConditional).unwrap()];
    let mut models = [Model { seed: 3, ..Model::default() }, Model { seed: 29, ..Model::default() }];
    for _ in 0..32 {
        // Deliberately different order and pauses; completed rows are skipped.
        advance(&mut rows[1], &mut models[1]); advance(&mut rows[0], &mut models[0]);
    }
    let [left, right] = rows;
    assert_eq!(left.finish().unwrap(), a.score_serial_reference(&mut Model { seed: 3, ..Model::default() }, ScoringMode::FullVocabulary).unwrap());
    assert_eq!(right.finish().unwrap(), b.score_serial_reference(&mut Model { seed: 29, ..Model::default() }, ScoringMode::TrieConditional).unwrap());
    assert_eq!(models[1].calls, [vec![], vec![2], vec![3]]);
}
#[test]
fn pending_request_is_stable_and_eos_is_never_forwarded() {
    let scorer = scorer(); let mut cursor = scorer.cursor(ScoringMode::TrieConditional).unwrap();
    let mut model = Model::default(); let mut ordinal = 0;
    while let Some(a) = cursor.request().unwrap() {
        let b = cursor.request().unwrap().unwrap();
        assert_eq!(a.ordinal, ordinal); assert_eq!(a.ordinal, b.ordinal); assert_eq!(a.prefix, b.prefix);
        assert!(!a.prefix.contains(&0));
        let ProjectionRows::Selected(ids) = a.rows else { unreachable!() };
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        advance(&mut cursor, &mut model); ordinal += 1;
    }
    let run = cursor.finish().unwrap(); assert_eq!(run.work.prefix_evaluations, ordinal);
    assert_eq!(run.candidates.len(), language().len());
    assert_eq!(run.candidates.iter().find(|c| c.id == "short").unwrap().scored_tokens, 2);
    assert_eq!(run.candidates.iter().find(|c| c.id == "nested").unwrap().scored_tokens, 3);
}
#[test]
fn malformed_projection_permanently_poisoned_no_retry_or_partial_finish() {
    let scorer = scorer();
    for values in [vec![], vec![0.0; 7], vec![0.0; 9], vec![f32::NAN; 8], vec![f32::INFINITY; 8], vec![f32::NEG_INFINITY; 8]] {
        let mut cursor = scorer.cursor(ScoringMode::FullVocabulary).unwrap();
        assert!(cursor.accept(0, &values).is_err());
        assert!(cursor.request().is_err()); assert_eq!(cursor.accept(0, &[0.0; 8]), Err(ScoringError::CursorState));
        assert_eq!(cursor.finish().unwrap_err(), ScoringError::CursorState);
    }
}
#[test]
fn stale_future_and_post_completion_submissions_refuse() {
    let scorer = scorer();
    for ordinal in [1, usize::MAX] {
        let mut cursor = scorer.cursor(ScoringMode::FullVocabulary).unwrap();
        assert_eq!(cursor.accept(ordinal, &[0.0; 8]), Err(ScoringError::CursorState));
        assert!(cursor.request().is_err());
    }
    let mut cursor = scorer.cursor(ScoringMode::FullVocabulary).unwrap();
    cursor.accept(0, &[0.0; 8]).unwrap();
    assert_eq!(cursor.accept(0, &[0.0; 8]), Err(ScoringError::CursorState));
    let mut cursor = scorer.cursor(ScoringMode::FullVocabulary).unwrap(); let mut model = Model::default();
    while advance(&mut cursor, &mut model) {}
    let ordinal = cursor.planned_work().prefix_evaluations;
    assert_eq!(cursor.accept(ordinal, &[0.0; 8]), Err(ScoringError::CursorState));
    assert!(cursor.finish().is_err());
}
#[test]
fn incomplete_finish_cannot_publish_any_of_the_already_scored_terminals() {
    let scorer = scorer(); let total = scorer.cursor(ScoringMode::FullVocabulary).unwrap().planned_work().prefix_evaluations;
    for completed in 0..total {
        let mut cursor = scorer.cursor(ScoringMode::FullVocabulary).unwrap(); let mut model = Model::default();
        for _ in 0..completed { advance(&mut cursor, &mut model); }
        assert_eq!(cursor.finish().unwrap_err(), ScoringError::CursorState);
    }
}
#[test]
fn extreme_finite_logits_preserve_frozen_denominators_and_underflowed_candidates() {
    let scorer = scorer();
    for mode in modes() {
        for constant in [f32::MAX, -f32::MAX, 0.0, f32::MIN_POSITIVE] {
            let mut old = Model { constant: Some(constant), ..Model::default() };
            let mut new = Model { constant: Some(constant), ..Model::default() };
            let expected = scorer.score_serial_reference(&mut old, mode).unwrap();
            let actual = scorer.score(&mut new, mode).unwrap();
            assert_eq!(actual, expected); assert_eq!(actual.candidates.len(), language().len());
        }
    }
}
#[test]
fn work_limit_is_checked_before_a_cursor_can_request_a_projection() {
    let scorer = CandidateScorer::compile(&[candidate("x", &[1])], 8, 0,
        ScoringLimits { max_projected_logits: 15, ..ScoringLimits::default() }).unwrap();
    assert!(matches!(scorer.cursor(ScoringMode::FullVocabulary), Err(ScoringError::LimitExceeded("projected_logits"))));
    assert!(scorer.cursor(ScoringMode::TrieConditional).is_ok());
}
