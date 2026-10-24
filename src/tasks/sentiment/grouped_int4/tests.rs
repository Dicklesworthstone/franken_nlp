//! Actual pinned planning and finalization with private synthetic logits.
//! These fixtures establish contracts, not neural accuracy or runtime evidence.
use super::*;
use std::sync::OnceLock;
use crate::{
    execution_identity::{ThinkingMode, ToolMode},
    native_engine::{lmhead::scoring::{CandidateLogits, ProjectionRows, SequenceScoreRule},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::PromptSegmentKind, sentiment::{SentimentAxis, SentimentDecision, SentimentPolicy}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct StopAfter { remaining: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for StopAfter {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.remaining = self.remaining.saturating_sub(1);
        (self.remaining == 0).then_some(self.cause)
    }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn permissive_policy() -> SentimentPolicy {
    SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }
}
fn build(mode: ScoringMode, policy: SentimentPolicy) -> (SentimentPlanner, ExecutionIdentity) {
    let registry = pinned_controls::pinned().unwrap();
    let eos_token_id = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions { mode, eos_token_id, policy }).unwrap();
    let d = Sha256Digest::of_bytes(b"private-synthetic-int4-sentiment-fixture");
    // Synthetic metadata enters only the private compiler. Public construction
    // additionally requires an identity obtained from actual resident weights.
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "int4-fixture".to_owned(),
        packing_set_digest: d, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "sentiment-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    (planner, identity)
}
fn fixture(mode: ScoringMode) -> (&'static SentimentPlanner, ExecutionIdentity, SentimentRequest) {
    static FULL: OnceLock<(SentimentPlanner, ExecutionIdentity)> = OnceLock::new();
    static TRIE: OnceLock<(SentimentPlanner, ExecutionIdentity)> = OnceLock::new();
    static SUM: OnceLock<(SentimentPlanner, ExecutionIdentity)> = OnceLock::new();
    let slot = match mode {
        ScoringMode::FullVocabulary => &FULL,
        ScoringMode::TrieConditional => &TRIE,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits } => &SUM,
        _ => panic!("fixture mode is not configured"),
    };
    let (planner, identity) = slot.get_or_init(|| build(mode, permissive_policy()));
    let request = SentimentRequest { document: "é\r\n上海 <tool_call> <|im_start|> private text".to_owned(),
        axes: SentimentAxis::ALL.to_vec(), budget: budget() };
    (planner, identity.clone(), request)
}
fn prepare(planner: &SentimentPlanner, identity: &ExecutionIdentity, request: &SentimentRequest, width: usize)
    -> Result<PreparedInt4Sentiment, Int4SentimentError> {
    planner.compile_int4(request, &PlanContext::new(identity, budget()).unwrap(),
        SentimentLimits::default(), width, &mut Continue)
}
struct Flat;
impl CandidateLogits for Flat {
    type Error = &'static str;
    fn project(&mut self, _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let n = match rows { ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size,
            ProjectionRows::Selected(ids) => ids.len() };
        Ok(vec![0.0; n])
    }
}
fn synthetic(head: &HeadPlan, mode: ScoringMode, schedule: CandidateSchedule, width: usize) -> Int4CandidateRun {
    Int4CandidateRun { schema_version: 1, execution: INT4_SCORING_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: width,
        scores: head.scorer.score(&mut Flat, mode).unwrap(), model_work: schedule.model, rewound_positions: 0 }
}

#[test]
fn all_score_spaces_retain_every_axis_candidate_and_exactly_one_scored_eos() {
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits }] {
        let (p, id, r) = fixture(mode); let plan = prepare(p, &id, &r, 32).unwrap();
        let run = plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s, 32))).unwrap();
        assert_eq!(run.head_count, 4); assert_eq!(run.result.dimensions.len(), 4);
        assert_eq!(run.result.work.candidates, 20); assert_eq!(run.model_work, plan.planned_work());
        assert_eq!(run.execution, INT4_SENTIMENT_EXECUTION); assert_eq!(run.group_width, 32);
        assert_eq!(run.result.calibration, "uncalibrated_not_a_psychological_measurement");
        assert_eq!(run.result.interpretation, "independent_conditional_candidate_weight_projections");
        for (dimension, head) in run.result.dimensions.iter().zip(&plan.inner.heads) {
            assert_eq!(dimension.scores.candidates.len(), 5);
            assert_eq!(dimension.scores.eos_token_id, plan.inner.options.eos_token_id);
            assert_eq!(dimension.scores.eos_rule, "append_and_score_exactly_one_eos");
            assert_eq!(dimension.scores.full_vocab_denominators_computed, mode == ScoringMode::FullVocabulary);
            assert!((dimension.scores.candidates.iter().map(|c| c.candidate_weight).sum::<f64>() - 1.0).abs() < 1e-10);
            let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else { panic!("finite candidates") };
            for score in &dimension.scores.candidates {
                let candidate = candidates.iter().find(|c| c.id() == score.id).unwrap();
                assert_eq!(score.scored_tokens, candidate.continuation().token_ids().len() + 1);
                assert!(!candidate.continuation().token_ids().contains(&plan.inner.options.eos_token_id));
            }
        }
    }
}

#[test]
fn group_partition_changes_group_dot_cost_without_padded_macs_or_changed_language() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional);
    let small = prepare(p, &id, &r, 16).unwrap(); let large = prepare(p, &id, &r, 128).unwrap();
    assert_eq!(small.work.forward_positions, large.work.forward_positions);
    assert_eq!(small.work.projected_logits, large.work.projected_logits);
    assert_eq!(small.work.attention_pairs, large.work.attention_pairs);
    assert_eq!(small.work.projections.multiply_accumulates, large.work.projections.multiply_accumulates);
    assert!(small.work.projections.dot_products > large.work.projections.dot_products);
    assert_eq!(small.identity.schema_digest, large.identity.schema_digest);
    assert_ne!(small.identity.decision_policy_digest, large.identity.decision_policy_digest);
    for width in [3, 16, 128, 10752] {
        let plan = prepare(p, &id, &r, width).unwrap();
        let sum = plan.schedules.iter().try_fold(Int4Work::default(), |total, s| total.checked_add(s.model)).unwrap();
        assert_eq!(sum, plan.work); assert_eq!(plan.group_width(), width);
        assert!(sum.forward_positions > plan.required_context() as u64);
        assert_eq!(plan.required_context(), plan.schedules.iter().map(|s| s.context).max().unwrap());
    }
    for width in [0, 10753, usize::MAX] { assert!(prepare(p, &id, &r, width).is_err()); }
}

#[test]
fn all_five_work_axes_are_cumulative_across_the_complete_bundle() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(p, &id, &r, 32).unwrap();
    check_work(plan.work, Int4RunBudget::exact(plan.work)).unwrap();
    for axis in 0..4 {
        let mut ceiling = Int4RunBudget::exact(plan.work);
        match axis { 0 => ceiling.max_forward_positions -= 1, 1 => ceiling.max_attention_pairs -= 1,
            2 => ceiling.max_projection_work.dot_products -= 1, _ => ceiling.max_projection_work.multiply_accumulates -= 1 }
        assert!(matches!(check_work(plan.work, ceiling), Err(Int4SentimentError::WorkBudget)));
    }
    let limits = SentimentLimits { max_total_projected_logits: plan.work.projected_logits - 1, ..SentimentLimits::default() };
    assert!(plan.schedules.iter().all(|s| s.model.projected_logits <= limits.max_total_projected_logits));
    assert!(matches!(p.compile_int4(&r, &PlanContext::new(&id, budget()).unwrap(), limits, 32, &mut Continue),
        Err(Int4SentimentError::Planning(SentimentPlanningError::Task(SentimentError::LimitExceeded("projected_logits"))))));
}

#[test]
fn int4_planning_refuses_other_profiles_backends_assets_and_privileged_modes() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional);
    for field in 0..9 {
        let mut changed = id.clone();
        match field { 0 => changed.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => changed.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => changed.backend_semantic_version = "different-cast-program".to_owned(),
            3 => changed.kv_dtype = "int8".to_owned(), 4 => changed.thinking_mode = ThinkingMode::Enabled,
            5 => changed.tool_mode = ToolMode::Json, 6 => changed.task_spec = "classify-v1".to_owned(),
            7 => changed.tokenizer_digest = Sha256Digest::of_bytes(b"other"),
            _ => changed.template_digest = Sha256Digest::of_bytes(b"other") }
        assert!(matches!(prepare(p, &changed, &r, 32), Err(Int4SentimentError::Planning(SentimentPlanningError::Identity))));
    }
    let context = PlanContext::new(&id, budget()).unwrap();
    assert!(p.plan(&r, &context, SentimentLimits::default()).is_err());
    assert!(p.plan_int8_with_control(&r, &context, SentimentLimits::default(), &mut Continue).is_err());
}

#[test]
fn eager_int8_and_int4_share_exact_semantics_without_identity_coercion() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional); let before = id.clone();
    let int4 = prepare(p, &id, &r, 32).unwrap();
    let mut eager_id = id.clone(); eager_id.numerics_profile = NumericsProfile::HfBf16Eager;
    let eager = p.plan(&r, &PlanContext::new(&eager_id, budget()).unwrap(), SentimentLimits::default()).unwrap();
    assert_eq!(int4.inner.binding_digest(), eager.binding_digest());
    let mut int8_id = id.clone(); int8_id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    int8_id.backend_semantic_version = crate::native_engine::strict_int8::STRICT_INT8_EXECUTION.to_owned();
    let int8 = p.plan_int8_with_control(&r, &PlanContext::new(&int8_id, budget()).unwrap(), SentimentLimits::default(), &mut Continue).unwrap();
    assert_eq!(int4.identity.prompt_digest, int8.execution_identity().prompt_digest);
    assert_eq!(int4.identity.schema_digest, int8.execution_identity().schema_digest);
    assert_ne!(int4.identity.decision_policy_digest, int8.execution_identity().decision_policy_digest);
    assert_eq!(id, before);
}

#[test]
fn source_bytes_are_preserved_only_as_untrusted_document_segments_for_every_axis() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional); let plan = prepare(p, &id, &r, 32).unwrap();
    let registry = pinned_controls::pinned().unwrap(); let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    for head in &plan.inner.heads {
        let data: Vec<_> = head.ir.prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document).collect();
        assert_eq!(data.len(), 1);
        assert!(data[0].token_ids().iter().all(|&id| !registry.template_controls().contains(id)));
        assert_eq!(tokenizer.tokenizer().decode_bytes(data[0].token_ids()).unwrap(), r.document.as_bytes());
    }
}

#[test]
fn exact_identity_binds_source_axes_mode_policy_output_cap_and_admitted_fields() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional); let plan = prepare(p, &id, &r, 32).unwrap();
    assert!(plan.verify_identity(&id).is_err()); plan.verify_identity(plan.execution_identity()).unwrap();
    let mut reordered = r.clone(); reordered.axes.reverse();
    assert_eq!(plan.identity, prepare(p, &id, &reordered, 32).unwrap().identity);
    for field in 0..3 {
        let mut changed = r.clone();
        match field { 0 => changed.document.push('!'), 1 => { changed.axes.pop(); }, _ => changed.budget.max_output_bytes -= 1 }
        assert_ne!(plan.identity, prepare(p, &id, &changed, 32).unwrap().identity);
    }
    let (other, other_id, _) = fixture(ScoringMode::FullVocabulary);
    assert_ne!(plan.identity, prepare(other, &other_id, &r, 32).unwrap().identity);
    let (other, other_id) = build(ScoringMode::TrieConditional,
        SentimentPolicy { minimum_peak_weight_ppm: 700_000, ..permissive_policy() });
    assert_ne!(plan.identity, prepare(&other, &other_id, &r, 32).unwrap().identity);
    let value = serde_json::to_value(plan.execution_identity()).unwrap();
    for key in value.as_object().unwrap().keys() {
        let mut changed = value.clone();
        changed[key] = match &value[key] {
            serde_json::Value::String(s) if s.len() == 64 => serde_json::json!(Sha256Digest::of_bytes(b"other").to_hex()),
            serde_json::Value::Number(_) => serde_json::json!(99), _ => serde_json::json!("different"),
        };
        if let Ok(changed) = serde_json::from_value::<ExecutionIdentity>(changed) {
            assert!(plan.verify_identity(&changed).is_err(), "accepted {key}");
        }
    }
}

struct Prefer<'a> { continuation: &'a [u32], eos: u32 }
impl CandidateLogits for Prefer<'_> {
    type Error = &'static str;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let preferred = self.continuation.starts_with(prefix)
            .then(|| self.continuation.get(prefix.len()).copied().unwrap_or(self.eos));
        let value = |token| if Some(token) == preferred { 0.0 } else { -32.0 };
        Ok(match rows {
            ProjectionRows::FullVocabulary { vocabulary_size } => (0..vocabulary_size).map(|id| value(id as u32)).collect(),
            ProjectionRows::Selected(ids) => ids.iter().map(|&id| value(id)).collect(),
        })
    }
}
#[test]
fn positive_and_negative_coordinates_are_independent_without_cross_axis_normalization() {
    let mode = ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits };
    let (p, id, r) = fixture(mode); let plan = prepare(p, &id, &r, 32).unwrap();
    let run = plan.execute_heads(&mut Continue, |head, mode, schedule, _| {
        let target = if head.axis == SentimentAxis::Dominance { "bin-0" } else { "bin-4" };
        let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else { panic!("finite candidates") };
        let continuation = candidates.iter().find(|c| c.id() == target).unwrap().continuation().token_ids();
        let scores = head.scorer.score(&mut Prefer { continuation, eos: plan.inner.options.eos_token_id }, mode).unwrap();
        Ok(Int4CandidateRun { scores, ..synthetic(head, mode, schedule, 32) })
    }).unwrap();
    for dimension in &run.result.dimensions {
        assert_eq!(dimension.decision, SentimentDecision::Estimated);
        if dimension.axis == SentimentAxis::Dominance { assert!(dimension.estimate.unwrap() < -0.99); }
        else { assert!(dimension.estimate.unwrap() > 0.99); }
    }
    let high: f64 = run.result.dimensions.iter().filter(|d| d.axis != SentimentAxis::Dominance)
        .map(|d| d.scores.candidates.iter().find(|c| c.id == "bin-4").unwrap().candidate_weight).sum();
    assert!(high > 2.99);
}

#[test]
fn diffuse_axes_abstain_without_discarding_the_complete_distributions() {
    let mode = ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits };
    let (_, _, r) = fixture(mode);
    let (p, id) = build(mode, SentimentPolicy { minimum_peak_weight_ppm: 700_000, maximum_normalized_entropy_ppm: 500_000 });
    let plan = prepare(&p, &id, &r, 32).unwrap();
    let run = plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s, 32))).unwrap();
    for dimension in run.result.dimensions {
        assert_eq!(dimension.decision, SentimentDecision::Abstained); assert!(dimension.estimate.is_none());
        assert_eq!(dimension.scores.candidates.len(), 5); assert!(dimension.normalized_entropy > 0.99);
    }
}

#[test]
fn full_vocabulary_scores_retain_probability_mass_outside_the_finite_language() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(p, &id, &r, 32).unwrap();
    let run = plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s, 32))).unwrap();
    for dimension in run.result.dimensions {
        let captured: f64 = dimension.scores.candidates.iter().map(|c| c.sequence_score.exp()).sum();
        assert!(captured > 0.0 && captured < 1e-8);
        assert!((dimension.scores.candidates.iter().map(|c| c.candidate_weight).sum::<f64>() - 1.0).abs() < 1e-10);
    }
}

#[test]
fn corrupt_native_metadata_group_and_any_work_axis_refuse_the_entire_bundle() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional); let plan = prepare(p, &id, &r, 32).unwrap();
    for field in 0..12 {
        let result = plan.execute_heads(&mut Continue, |h, m, s, _| {
            let mut run = synthetic(h, m, s, 32);
            match field { 0 => run.schema_version += 1, 1 => run.execution.push('x'), 2 => run.numerics_profile.push('x'),
                3 => run.group_width += 1, 4 => run.model_work.forward_positions += 1,
                5 => run.model_work.projected_logits += 1, 6 => run.model_work.attention_pairs += 1,
                7 => run.model_work.projections.dot_products += 1, 8 => run.model_work.projections.multiply_accumulates += 1,
                9 => run.scores.work.projected_logits += 1, 10 => run.scores.work.scored_edges += 1,
                _ => run.scores.work.prefix_evaluations += 1 }
            Ok(run)
        });
        assert!(matches!(result, Err(Int4SentimentError::Accounting)));
    }
    assert!(plan.execute_heads(&mut Continue, |h, m, s, _| {
        let mut run = synthetic(h, m, s, 32); run.scores.candidates.pop(); Ok(run)
    }).is_err());
}

#[test]
fn later_head_failure_and_rewind_overflow_never_publish_partial_sentiment() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional); let plan = prepare(p, &id, &r, 32).unwrap();
    let mut calls = 0;
    let error = plan.execute_heads(&mut Continue, |h, m, s, _| {
        calls += 1;
        if calls == 2 { return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::FailFast).into()); }
        Ok(synthetic(h, m, s, 32))
    }).unwrap_err();
    assert_eq!(calls, 2); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::FailFast));
    assert!(matches!(plan.execute_heads(&mut Continue, |h, m, s, _| {
        let mut run = synthetic(h, m, s, 32); run.rewound_positions = u64::MAX; Ok(run)
    }), Err(Int4SentimentError::Accounting)));
}

#[test]
fn preparation_head_boundaries_and_final_publication_keep_typed_cancellation() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional); let plan = prepare(p, &id, &r, 32).unwrap();
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled] {
        for remaining in [1, 2, 4, 7] {
            let error = p.compile_int4(&r, &PlanContext::new(&id, budget()).unwrap(), SentimentLimits::default(), 32,
                &mut StopAfter { remaining, cause }).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause));
        }
        // Two checks per axis, followed by the final complete-result check.
        for remaining in [1, 3, 8, 9] {
            let error = plan.execute_heads(&mut StopAfter { remaining, cause },
                |h, m, s, _| Ok(synthetic(h, m, s, 32))).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause));
        }
        let error = Int4SentimentError::Scoring(Int4ScoringError::Native(StrictInt4Error::Cancelled(cause)));
        assert_eq!(error.cancellation(), Some(cause));
    }
}

#[test]
fn complete_wrapper_must_fit_even_when_the_inner_result_exactly_fits() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional); let mut plan = prepare(p, &id, &r, 32).unwrap();
    let run = plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s, 32))).unwrap();
    plan.inner.max_output_bytes = canonjson::canonical_bytes(&run.result).unwrap().len() as u64;
    assert!(matches!(plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s, 32))),
        Err(Int4SentimentError::OutputBudget)));
}

#[test]
fn invalid_axes_and_planning_limits_are_refused_without_truncation() {
    let (p, id, r) = fixture(ScoringMode::TrieConditional);
    for axes in [Vec::new(), vec![SentimentAxis::Valence, SentimentAxis::Valence]] {
        let mut changed = r.clone(); changed.axes = axes;
        assert!(matches!(prepare(p, &id, &changed, 32), Err(Int4SentimentError::Planning(SentimentPlanningError::InvalidAxes))));
    }
    let mut changed = r.clone(); changed.document.clear();
    assert!(matches!(prepare(p, &id, &changed, 32), Err(Int4SentimentError::Planning(SentimentPlanningError::EmptyDocument))));
    changed = r.clone(); changed.budget.max_input_tokens = 1;
    assert!(prepare(p, &id, &changed, 32).is_err());
    let plan = prepare(p, &id, &r, 32).unwrap();
    let total: usize = plan.inner.heads.iter().map(|h| h.prompt_len).sum();
    let limits = SentimentLimits { max_total_prompt_tokens: total - 1, ..SentimentLimits::default() };
    assert!(p.compile_int4(&r, &PlanContext::new(&id, budget()).unwrap(), limits, 32, &mut Continue).is_err());
    assert_eq!(r.document, "é\r\n上海 <tool_call> <|im_start|> private text");
}

#[test]
fn redacted_errors_and_owned_types_do_not_export_private_plans() {
    let error = Int4SentimentError::Task(SentimentError::LimitExceeded("private-payload"));
    assert!(!format!("{error}").contains("private-payload"));
    assert!(!format!("{error:?}").contains("private-payload"));
    fn send<T: Send + 'static>() {}
    send::<Int4SentimentPlanner>(); send::<PreparedInt4Sentiment>(); send::<Int4SentimentRun>();
}
