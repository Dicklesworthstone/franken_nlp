//! Pinned planning and synthetic numerical receipts at a private seam.
//! These fixtures do not establish native parity, model quality or calibration.
use super::*;
use std::sync::OnceLock;
use crate::{
    corpus::resolve::{ResolutionDocument, MentionInput, ResolveOptions, ResolveLimits, PairDecision},
    native_engine::lmhead::scoring::{CandidateScore, ScoringWork},
    tokenizer::pinned_controls,
    validation::{SourceSpan, validate_source_span, grounded_fields::VerifiedSourceSpan},
};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> &'static ResolutionPlanner {
    static PINNED: OnceLock<ResolutionPlanner> = OnceLock::new();
    PINNED.get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        let eos = controls.template_controls().entries().iter()
            .find(|entry| entry.special && entry.surface == IM_END).unwrap().id;
        ResolutionPlanner::pinned(controls.template_controls(), eos).unwrap()
    })
}
fn identity() -> ExecutionIdentity {
    let p = planner();
    let d = Sha256Digest::of_bytes(b"private-int4-resolution-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int4-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: p.template_digest(), task_spec: RESOLVE_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn docs() -> Vec<ResolutionDocument> {
    ["a", "b", "c"].iter().map(|&id| ResolutionDocument {
        id: id.to_owned(), text: "Smith <think> instructions\r\n上海 😀".to_owned(),
        mentions: vec![MentionInput { entity_type: "PERSON".to_owned(), surface: "Smith".to_owned(),
            span: VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 } }],
    }).collect()
}
fn graph(d: &[ResolutionDocument]) -> ResolutionPlan<'_> {
    ResolutionPlan::prepare(d, ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap()
}
fn limits() -> Int4ResolveLimits {
    let mut work = Int4Work::default();
    work.forward_positions = u64::MAX;
    work.projected_logits = u64::MAX;
    work.attention_pairs = u64::MAX;
    work.projections.dot_products = u64::MAX;
    work.projections.multiply_accumulates = u64::MAX;
    Int4ResolveLimits { planning: NativeResolveLimits::default(), max_model_work: work }
}
fn prepared<'p, 's>(plan: &'p ResolutionPlan<'s>, width: usize) -> PreparedInt4Resolution<'static, 'p, 's> {
    planner().compile_int4(plan, &identity(), limits(), width, &mut Continue).unwrap()
}
fn admitted(prepared: &PreparedInt4Resolution<'_, '_, '_>) -> Vec<ExecutionIdentity> {
    prepared.execution_identities().cloned().collect()
}
fn synthetic(schedule: CandidateSchedule, width: usize, decision: PairDecision) -> Int4CandidateRun {
    let values: [f64; 3] = match decision {
        PairDecision::Same => [-4.0, -1.0, -4.0],
        PairDecision::Different => [-1.0, -4.0, -4.0],
        PairDecision::Uncertain => [-3.0; 3],
    };
    let sum: f64 = values.iter().map(|p| p.exp()).sum();
    let scores = CandidateScores { score_space: ScoreSpace::FullVocabSequenceLogprob,
        normalization_scope: "synthetic-candidate-set".to_owned(), length_rule: "none".to_owned(),
        eos_rule: "scored".to_owned(), eos_token_id: planner().eos, full_vocab_denominators_computed: true,
        candidates: ["different", "same", "uncertain"].iter().zip(values).map(|(&id, p)| CandidateScore {
            id: id.to_owned(), scored_tokens: 2, sequence_score: p, candidate_weight: p.exp() / sum,
        }).collect(), work: ScoringWork { prefix_evaluations: 4, scored_edges: 6, projected_logits: HEAD_ROWS } };
    Int4CandidateRun { schema_version: 1, execution: INT4_SCORING_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: width,
        scores, model_work: schedule.model, rewound_positions: 2 }
}
fn all_same(prepared: PreparedInt4Resolution<'_, '_, '_>) -> Int4ResolutionRun {
    let ids = admitted(&prepared);
    let width = prepared.group_width();
    prepared.execute_heads(&ids, &mut Continue, |_, _, _, schedule, _| {
        Ok(synthetic(schedule, width, PairDecision::Same))
    }).unwrap()
}

#[test]
fn both_group_widths_execute_every_order_with_exact_native_work() {
    let d = docs(); let plan = graph(&d);
    for width in [16, 32] {
        let prepared = prepared(&plan, width); let expected = prepared.planned_work();
        assert_eq!(prepared.pair_count(), 3);
        assert_eq!(prepared.head_count(), 6);
        let ids = admitted(&prepared); let mut sequence = Vec::new();
        let run = prepared.execute_heads(&ids, &mut Continue, |pair, order, _, schedule, _| {
            sequence.push((pair, order)); Ok(synthetic(schedule, width, PairDecision::Same))
        }).unwrap();
        assert_eq!(sequence, vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
        assert_eq!(run.group_width, width); assert!(run.model_evaluated);
        assert_eq!(run.head_count, 6); assert_eq!(run.rewound_positions, 12);
        assert_eq!(run.model_work, expected); assert_eq!(run.planned_model_work, expected);
        assert_eq!(run.result.clusters.len(), 1); assert_eq!(run.result.clusters[0].mentions, vec![0, 1, 2]);
    }
}

#[test]
fn group_partition_changes_group_dots_and_policy_but_not_prompt_or_logical_macs() {
    let d = docs(); let plan = graph(&d);
    let g16 = prepared(&plan, 16); let g32 = prepared(&plan, 32);
    assert_eq!(g16.work.forward_positions, g32.work.forward_positions);
    assert_eq!(g16.work.projected_logits, g32.work.projected_logits);
    assert_eq!(g16.work.attention_pairs, g32.work.attention_pairs);
    assert_eq!(g16.work.projections.multiply_accumulates, g32.work.projections.multiply_accumulates);
    assert_eq!(g16.work.projections.dot_products, g32.work.projections.dot_products * 2);
    for (a, b) in g16.inner.pairs.iter().zip(&g32.inner.pairs) {
        assert_eq!(a.prompts, b.prompts);
        assert_ne!(a.identity.decision_policy_digest, b.identity.decision_policy_digest);
    }
}

#[test]
fn schedules_account_for_sibling_branches_instead_of_sequential_attention() {
    let d = docs(); let plan = graph(&d);
    for width in [16, 32] {
        let prepared = prepared(&plan, width); let mut total = Int4Work::default();
        for (pair, schedules) in prepared.inner.pairs.iter().zip(&prepared.schedules) {
            for (prompt, schedule) in pair.prompts.iter().zip(schedules) {
                assert_eq!(schedule.context, prompt.len() + 1);
                assert_eq!(schedule.model.forward_positions, prompt.len() as u64 + 3);
                assert_eq!(schedule.scoring.prefix_evaluations, 4);
                assert_eq!(schedule.scoring.scored_edges, 6);
                assert_eq!(schedule.scoring.projected_logits, HEAD_ROWS);
                let sequential = Int4Work::for_sequence(width, 0, prompt.len() + 3, HEAD_ROWS as usize).unwrap();
                assert!(schedule.model.attention_pairs < sequential.attention_pairs);
                assert_eq!(schedule.model.projections, sequential.projections);
                total = total.checked_add(schedule.model).unwrap();
            }
        }
        assert_eq!(prepared.work, total);
    }
}

#[test]
fn shared_prompt_compiler_keeps_control_text_as_untrusted_data() {
    let d = docs(); let plan = graph(&d); let id = identity();
    let before = canonjson::canonical_bytes(&id).unwrap();
    let int4 = prepared(&plan, 32);
    let mut eager_identity = id.clone(); eager_identity.numerics_profile = NumericsProfile::HfBf16Eager;
    let eager = planner().prepare(&plan, &eager_identity, limits().planning, &mut Continue).unwrap();
    for (a, b) in eager.pairs.iter().zip(&int4.inner.pairs) {
        assert_eq!(a.prompts, b.prompts);
        assert_ne!(a.identity.decision_policy_digest, b.identity.decision_policy_digest);
    }
    assert!(planner().prepare(&plan, &id, limits().planning, &mut Continue).is_err());
    assert!(planner().compile_int4(&plan, &eager_identity, limits(), 32, &mut Continue).is_err());
    assert_eq!(before, canonjson::canonical_bytes(&id).unwrap());
}

#[test]
fn each_whole_snapshot_axis_can_independently_refuse_before_execution() {
    let d = docs(); let plan = graph(&d);
    for width in [16, 32] {
        let exact = prepared(&plan, width).planned_work();
        let mut cap = limits(); cap.max_model_work = exact;
        planner().compile_int4(&plan, &identity(), cap, width, &mut Continue).unwrap();
        for axis in 0..5 {
            let mut short = cap;
            match axis {
                0 => short.max_model_work.forward_positions -= 1,
                1 => short.max_model_work.projected_logits -= 1,
                2 => short.max_model_work.attention_pairs -= 1,
                3 => short.max_model_work.projections.dot_products -= 1,
                _ => short.max_model_work.projections.multiply_accumulates -= 1,
            }
            assert!(matches!(planner().compile_int4(&plan, &identity(), short, width, &mut Continue),
                Err(Int4ResolveError::WorkBudget)), "width {width} axis {axis}");
        }
    }
}

#[test]
fn pair_local_allowance_cannot_pay_for_a_complete_snapshot() {
    let d = docs(); let plan = graph(&d); let prepared = prepared(&plan, 32);
    let first = prepared.schedules[0][0].model.checked_add(prepared.schedules[0][1].model).unwrap();
    let mut cap = limits(); cap.max_model_work = first;
    assert!(matches!(planner().compile_int4(&plan, &identity(), cap, 32, &mut Continue),
        Err(Int4ResolveError::WorkBudget)));
}

#[test]
fn last_pair_identity_substitution_refuses_before_the_first_head() {
    let d = docs(); let plan = graph(&d);
    for axis in 0..10 {
        let prepared = prepared(&plan, 32); let mut ids = admitted(&prepared);
        let last = ids.last_mut().unwrap(); let changed = Sha256Digest::of_bytes(b"substitution");
        match axis {
            0 => last.prompt_digest = changed, 1 => last.logical_model_digest = changed,
            2 => last.taskir_digest = changed, 3 => last.decision_policy_digest = changed,
            4 => last.packing_set_digest = changed, 5 => last.calibration_digest = changed,
            6 => last.schema_digest = changed, 7 => last.quant_recipe = "other-group".to_owned(),
            8 => last.source_revision = "other-source".to_owned(),
            _ => last.backend_semantic_version = "other".to_owned(),
        }
        let mut calls = 0;
        let result = prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| {
            calls += 1; Ok(synthetic(s, 32, PairDecision::Same))
        });
        assert!(matches!(result, Err(Int4ResolveError::Identity)), "axis {axis}");
        assert_eq!(calls, 0);
    }
}

#[test]
fn missing_extra_and_reordered_admissions_are_not_partial_success() {
    let d = docs(); let plan = graph(&d);
    for mutation in 0..3 {
        let prepared = prepared(&plan, 32); let mut ids = admitted(&prepared);
        match mutation { 0 => { ids.pop(); }, 1 => ids.push(ids[0].clone()), _ => ids.reverse() }
        let mut calls = 0;
        assert!(matches!(prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| {
            calls += 1; Ok(synthetic(s, 32, PairDecision::Same))
        }), Err(Int4ResolveError::Identity)));
        assert_eq!(calls, 0);
    }
}

#[test]
fn disagreeing_orders_abstain_instead_of_averaging_away_the_conflict() {
    let d = docs(); let plan = graph(&d); let prepared = prepared(&plan, 32); let ids = admitted(&prepared);
    let run = prepared.execute_heads(&ids, &mut Continue, |_, order, _, s, _| {
        Ok(synthetic(s, 32, if order == 0 { PairDecision::Same } else { PairDecision::Different }))
    }).unwrap();
    assert!(run.result.judgments.iter().all(|j| j.decision == PairDecision::Uncertain));
    assert_eq!(run.result.clusters.len(), 3);
}

#[test]
fn an_explicit_different_pair_blocks_transitive_same_merges() {
    let d = docs(); let plan = graph(&d); let prepared = prepared(&plan, 32); let ids = admitted(&prepared);
    let run = prepared.execute_heads(&ids, &mut Continue, |pair, _, _, s, _| {
        Ok(synthetic(s, 32, if pair == 1 { PairDecision::Different } else { PairDecision::Same }))
    }).unwrap();
    assert_eq!(run.result.clusters.len(), 2); assert!(run.result.blocked_merges > 0);
    assert!(!run.result.clusters.iter().any(|c| c.mentions.contains(&0) && c.mentions.contains(&2)));
}

#[test]
fn permutation_of_documents_preserves_the_complete_canonical_result() {
    let d = docs(); let mut reversed = d.clone(); reversed.reverse();
    let a = graph(&d); let b = graph(&reversed);
    let a = all_same(prepared(&a, 32)); let b = all_same(prepared(&b, 32));
    assert_eq!(canonjson::canonical_bytes(&a).unwrap(), canonjson::canonical_bytes(&b).unwrap());
}

#[test]
fn no_pair_graph_requires_no_native_work_and_does_not_invent_inference() {
    let d = docs(); let plan = graph(&d[..1]);
    for width in [16, 32] {
        let mut cap = limits(); cap.max_model_work = Int4Work::default();
        cap.planning.max_work = BatchWork::default();
        let prepared = planner().compile_int4(&plan, &identity(), cap, width, &mut Continue).unwrap();
        assert_eq!(prepared.required_context_tokens(), 0); assert_eq!(prepared.head_count(), 0);
        let run = prepared.finalize_without_model(&mut Continue).unwrap();
        assert_eq!(run.group_width, width); assert!(!run.model_evaluated);
        assert_eq!(run.head_count, 0); assert_eq!(run.model_work, Int4Work::default());
        assert_eq!(run.rewound_positions, 0); assert_eq!(run.result.clusters.len(), 1);
    }
}

#[test]
fn no_model_finalizer_cannot_skip_required_comparisons() {
    let d = docs(); let plan = graph(&d);
    assert!(matches!(prepared(&plan, 32).finalize_without_model(&mut Continue), Err(Int4ResolveError::Accounting)));
}

#[test]
fn empty_graph_still_rejects_invalid_group_and_identity() {
    let d = docs(); let plan = graph(&d[..1]);
    for width in [0, usize::MAX] {
        assert!(planner().compile_int4(&plan, &identity(), limits(), width, &mut Continue).is_err());
    }
    let mut id = identity(); id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    assert!(planner().compile_int4(&plan, &id, limits(), 32, &mut Continue).is_err());
}

#[test]
fn late_second_order_failure_discards_the_entire_snapshot() {
    let d = docs(); let plan = graph(&d); let prepared = prepared(&plan, 32); let ids = admitted(&prepared);
    let mut calls = 0;
    let error = prepared.execute_heads(&ids, &mut Continue, |pair, order, _, s, _| {
        calls += 1;
        if pair == 1 && order == 1 { Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline).into()) }
        else { Ok(synthetic(s, 32, PairDecision::Same)) }
    }).err().unwrap();
    assert_eq!(calls, 4); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn post_head_cancellation_prevents_the_next_order() {
    struct Control(bool);
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.then_some(DecodeCancellationKind::User) }
    }
    let d = docs(); let plan = graph(&d); let prepared = prepared(&plan, 32); let ids = admitted(&prepared);
    let mut calls = 0;
    let error = prepared.execute_heads(&ids, &mut Control(false), |_, _, _, s, control| {
        calls += 1; control.0 = true; Ok(synthetic(s, 32, PairDecision::Same))
    }).err().unwrap();
    assert_eq!(calls, 1); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::User));
}

#[test]
fn empty_graph_finalization_observes_cancellation() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Shutdown) }
    }
    let d = docs(); let plan = graph(&d[..1]);
    let error = prepared(&plan, 32).finalize_without_model(&mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Shutdown));
}

#[test]
fn every_native_receipt_axis_is_checked_before_the_next_head() {
    let d = docs(); let plan = graph(&d);
    for axis in 0..20 {
        let prepared = prepared(&plan, 32); let ids = admitted(&prepared); let mut calls = 0;
        let result = prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| {
            calls += 1; let mut run = synthetic(s, 32, PairDecision::Same);
            match axis {
                0 => run.schema_version = 99, 1 => run.execution = "other".to_owned(),
                2 => run.numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(), 3 => run.group_width = 16,
                4 => run.model_work.forward_positions += 1, 5 => run.model_work.projected_logits -= 1,
                6 => run.model_work.attention_pairs -= 1, 7 => run.model_work.projections.dot_products -= 1,
                8 => run.model_work.projections.multiply_accumulates -= 1, 9 => run.rewound_positions = 0,
                10 => run.scores.full_vocab_denominators_computed = false, 11 => run.scores.eos_token_id ^= 1,
                12 => run.scores.candidates[0].sequence_score = f64::NAN,
                13 => run.scores.candidates[0].scored_tokens = 1, 14 => run.scores.work.scored_edges -= 1,
                15 => run.scores.work.prefix_evaluations -= 1, 16 => run.scores.work.projected_logits -= 1,
                17 => run.scores.candidates[0].candidate_weight = f64::INFINITY,
                18 => run.scores.candidates[0].id = "unknown".to_owned(),
                _ => run.scores.candidates.reverse(),
            }
            Ok(run)
        });
        assert!(result.is_err(), "axis {axis}"); assert_eq!(calls, 1);
    }
}

#[test]
fn full_result_byte_bound_is_exact_and_never_returns_truncation() {
    let d = docs(); let plan = graph(&d);
    let bytes = canonjson::canonical_bytes(&all_same(prepared(&plan, 32))).unwrap().len();
    for short in [false, true] {
        let mut cap = limits(); cap.planning.max_result_bytes = bytes - usize::from(short);
        let prepared = planner().compile_int4(&plan, &identity(), cap, 32, &mut Continue).unwrap();
        let ids = admitted(&prepared);
        let result = prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| Ok(synthetic(s, 32, PairDecision::Same)));
        if short { assert!(matches!(result, Err(Int4ResolveError::Resolution(ResolveError::OutputBudget)))); }
        else { assert_eq!(canonjson::canonical_bytes(&result.unwrap()).unwrap().len(), bytes); }
    }
}

#[test]
fn cluster_work_exhaustion_does_not_publish_a_partial_graph() {
    let d = docs(); let mut graph_limits = ResolveLimits::default(); graph_limits.max_cluster_checks = 1;
    let plan = ResolutionPlan::prepare(&d, ResolveOptions::default(), graph_limits, &mut Continue).unwrap();
    let prepared = prepared(&plan, 32); let ids = admitted(&prepared);
    assert!(matches!(prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| Ok(synthetic(s, 32, PairDecision::Same))),
        Err(Int4ResolveError::Resolution(ResolveError::ClusterBudget))));
}

#[test]
fn profile_backend_tools_and_pinned_asset_identity_are_never_repaired() {
    let d = docs(); let plan = graph(&d);
    for axis in 0..9 {
        let mut id = identity();
        match axis {
            0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => id.backend_semantic_version = "other".to_owned(), 3 => id.kv_dtype = "int8".to_owned(),
            4 => id.thinking_mode = ThinkingMode::Enabled, 5 => id.tool_mode = ToolMode::Json,
            6 => id.template_digest = Sha256Digest::of_bytes(b"other"),
            7 => id.tokenizer_digest = Sha256Digest::of_bytes(b"other"), _ => id.task_spec = "judge-v1".to_owned(),
        }
        assert!(planner().compile_int4(&plan, &id, limits(), 32, &mut Continue).is_err(), "axis {axis}");
    }
}

#[test]
fn source_validation_rejects_wrong_offsets_before_pair_planning() {
    for scalar in [false, true] {
        let mut d = docs();
        if scalar { d[2].mentions[0].span.scalar_end += 1; } else { d[2].mentions[0].span.byte_end += 1; }
        assert!(matches!(ResolutionPlan::prepare(&d, ResolveOptions::default(), ResolveLimits::default(), &mut Continue),
            Err(ResolveError::InvalidAnchor)));
    }
}

#[test]
fn unicode_mentions_preserve_original_byte_and_scalar_coordinates() {
    let d: Vec<_> = ["a", "b"].iter().map(|&id| ResolutionDocument {
        id: id.to_owned(), text: "é\r\n上海😀".to_owned(), mentions: vec![MentionInput {
            entity_type: "PLACE".to_owned(), surface: "上海".to_owned(),
            span: VerifiedSourceSpan { byte_start: 4, byte_end: 10, scalar_start: 3, scalar_end: 5 },
        }],
    }).collect();
    let plan = graph(&d); let run = all_same(prepared(&plan, 16));
    for mention in &run.result.mentions {
        let source = &d.iter().find(|doc| doc.id == mention.document_id).unwrap().text;
        let s = mention.span;
        validate_source_span(source, &mention.surface, SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
        assert_eq!(s, d[0].mentions[0].span);
    }
    assert_eq!(run.result.clusters.len(), 1);
}

#[test]
fn native_budget_cannot_omit_any_of_its_four_counter_axes() {
    let d = docs(); let plan = graph(&d); let work = prepared(&plan, 32).planned_work();
    check_budget(work, Int4RunBudget::exact(work)).unwrap();
    for axis in 0..4 {
        let mut cap = Int4RunBudget::exact(work);
        match axis { 0 => cap.max_forward_positions -= 1, 1 => cap.max_attention_pairs -= 1,
            2 => cap.max_projection_work.dot_products -= 1, _ => cap.max_projection_work.multiply_accumulates -= 1 }
        assert!(matches!(check_budget(work, cap), Err(Int4ResolveError::WorkBudget)));
    }
}

#[test]
fn full_resident_kv_capacity_is_charged_not_only_the_live_context() {
    let bytes = 1024 * KV_BYTES_PER_TOKEN as u64;
    check_kv_capacity(1024, 512, bytes).unwrap();
    assert!(matches!(check_kv_capacity(1024, 512, bytes - 1), Err(Int4ResolveError::WorkBudget)));
    assert!(matches!(check_kv_capacity(511, 512, bytes), Err(Int4ResolveError::WorkBudget)));
    assert!(matches!(check_kv_capacity(usize::MAX, 512, u64::MAX), Err(Int4ResolveError::WorkBudget)));
}

#[test]
fn all_cancellation_kinds_survive_each_supported_error_layer() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        for error in [Int4ResolveError::Resolution(ResolveError::Cancelled(cause)),
            Int4ResolveError::Planning(NativeResolveError::Resolution(ResolveError::Cancelled(cause))),
            Int4ResolveError::Scoring(Int4ScoringError::Native(StrictInt4Error::Cancelled(cause))),
            Int4ResolveError::Native(StrictInt4Error::Cancelled(cause))] {
            assert_eq!(error.cancellation(), Some(cause));
            assert_eq!(format!("{error}"), format!("{error:?}"));
        }
    }
}

#[test]
fn public_result_keeps_uncalibrated_warnings_without_private_model_keys() {
    let d = docs(); let plan = graph(&d); let run = all_same(prepared(&plan, 32));
    assert_eq!(run.result.calibration, "uncalibrated");
    assert_eq!(run.result.dependency_scope, DependencyScope::CorpusGlobal);
    assert_eq!(run.result.score_space, ScoreSpace::FullVocabSequenceLogprob);
    assert!(run.result.warnings.contains(&"pair_scores_are_not_identity_confidence"));
    assert!(run.result.warnings.contains(&"cluster_ids_are_snapshot_local"));
    let public = canonjson::canonical_string(&run).unwrap();
    for absent in ["packing_set_digest", "prompt_digest", "logical_model_digest", "taskir_digest", "source_root_sha256"] {
        assert!(!public.contains(absent));
    }
}
