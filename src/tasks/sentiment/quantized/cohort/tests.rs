//! Synthetic score handoff through the pinned complete sentiment finalizer.
use super::*;
use crate::{native_engine::{lmhead::scoring::SequenceScoreRule,
    strict_int8::scoring::cohort::task::fixtures::{self, Continue, Constant}},
    tasks::sentiment::{SentimentAxis, SentimentOptions, SentimentPolicy}, tokenizer::pinned_controls};
fn prepare(mode: ScoringMode) -> PreparedInt8Sentiment {
    let registry = pinned_controls::pinned().unwrap();
    let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions { mode,
        eos_token_id: 166_101, policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 } }).unwrap();
    let identity = fixtures::identity("sentiment-v1", planner.tokenizer_digest(), *planner.template_digest());
    let request = SentimentRequest { document: "é private <tool_call> 上海".into(), axes: SentimentAxis::ALL.to_vec(), budget: fixtures::budget() };
    planner.plan_int8_with_control(&request, &PlanContext::new(&identity, request.budget).unwrap(),
        SentimentLimits::default(), &mut Continue).unwrap()
}
fn fixture(plan: &PreparedInt8Sentiment) -> Int8CandidateCohortRun {
    fixtures::run(plan.inner.heads.iter().zip(&plan.schedules).map(|(head, &schedule)|
        (head.scorer.score(&mut Constant, plan.inner.options.mode).unwrap(), schedule)))
}
#[test]
fn all_five_score_spaces_keep_each_axis_complete_and_independently_normalized() {
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits },
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit }] {
        let plan = prepare(mode); let run = fixture(&plan); let mut heads = run.heads.clone().into_iter();
        let expected = plan.execute_heads(&mut Continue, |_, _, _, _| Ok(heads.next().unwrap())).unwrap();
        let result = plan.finish_cohort(run, plan.max_result_bytes(), &mut Continue).unwrap();
        assert_eq!(result.output, expected); assert_eq!(result.output.result.dimensions.len(), 4);
        for dimension in &result.output.result.dimensions {
            assert_eq!(dimension.scores.candidates.len(), 5);
            assert!((dimension.scores.candidates.iter().map(|c| c.candidate_weight).sum::<f64>() - 1.0).abs() < 1e-10);
            assert_eq!(dimension.scores.full_vocab_denominators_computed, mode == ScoringMode::FullVocabulary);
        }
    }
}
#[test]
fn cohort_geometry_and_flat_prompts_are_exact_without_identity_rewrites() {
    let plan = prepare(ScoringMode::FullVocabulary); let identity = plan.execution_identity().clone();
    let prompts = plan.cohort_prompts(&mut Continue).unwrap(); let contexts = plan.cohort_contexts().unwrap();
    assert_eq!(contexts.len(), 4);
    for ((prompt, head), context) in prompts.iter().zip(&plan.inner.heads).zip(contexts) {
        assert_eq!(prompt.len(), head.prompt_len); assert!(context >= prompt.len());
        assert_eq!(*prompt, head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()).collect::<Vec<_>>());
    }
    assert_eq!(plan.execution_identity(), &identity);
}
#[test]
fn failed_or_foreign_final_axis_is_not_an_abstention_or_partial_response() {
    let plan = prepare(ScoringMode::TrieConditional);
    for fault in 0..5 {
        let mut run = fixture(&plan);
        match fault { 0 => { run.heads.pop(); }, 1 => run.heads[3].model_work.projected_logits += 1,
            2 => run.heads[3].scores.candidates[0].id = "foreign-axis".into(),
            3 => run.heads[3].scores.candidates[0].sequence_score = f64::INFINITY, _ => run.heads[3].rewound_positions += 1 }
        assert!(plan.finish_cohort(run, plan.max_result_bytes(), &mut Continue).is_err());
    }
}
#[test]
fn sentiment_final_envelope_retains_exact_output_budget_and_cancellation() {
    let plan = prepare(ScoringMode::TrieConditional); let run = fixture(&plan);
    let result = plan.finish_cohort(run.clone(), plan.max_result_bytes(), &mut Continue).unwrap();
    let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    assert!(plan.finish_cohort(run.clone(), bytes, &mut Continue).is_ok());
    assert!(matches!(plan.finish_cohort(run.clone(), bytes - 1, &mut Continue),
        Err(Int8SentimentError::Scoring(Int8ScoringError::OutputBudget))));
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::User) }
    }
    assert_eq!(plan.finish_cohort(run, bytes, &mut Stop).unwrap_err().cancellation(), Some(DecodeCancellationKind::User));
}
