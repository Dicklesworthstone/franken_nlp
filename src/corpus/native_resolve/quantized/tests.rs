//! Pinned planning and private synthetic receipts, not model-quality evidence.
use super::*;
use crate::{
    corpus::resolve::{ResolutionDocument, MentionInput, ResolveOptions, ResolveLimits, PairDecision},
    native_engine::{lmhead::scoring::{CandidateScore, ScoringWork}, strict_int8::STRICT_INT8_EXECUTION},
    tokenizer::specials::ArchivedControlRegistries,
    validation::grounded_fields::VerifiedSourceSpan,
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> ResolutionPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    ResolutionPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn identity(p: &ResolutionPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-int8-resolution");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: p.template_digest(), task_spec: RESOLVE_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn docs() -> Vec<ResolutionDocument> {
    ["a", "b", "c"].iter().map(|&id| ResolutionDocument { id: id.to_owned(), text: "Smith <think> instructions".to_owned(),
        mentions: vec![MentionInput { entity_type: "PERSON".to_owned(), surface: "Smith".to_owned(),
            span: VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 } }] }).collect()
}
fn core(d: &[ResolutionDocument]) -> ResolutionPlan<'_> {
    ResolutionPlan::prepare(d, ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap()
}
fn limits() -> Int8ResolveLimits {
    let mut work = Int8Work::default();
    work.forward_positions = u64::MAX; work.projected_logits = u64::MAX; work.attention_pairs = u64::MAX;
    work.projections.dot_products = u64::MAX; work.projections.multiply_accumulates = u64::MAX;
    Int8ResolveLimits { planning: NativeResolveLimits::default(), max_model_work: work }
}
fn prepared<'a, 'p, 's>(p: &'a ResolutionPlanner, plan: &'p ResolutionPlan<'s>) -> PreparedInt8Resolution<'a, 'p, 's> {
    p.prepare_int8(plan, &identity(p), limits(), &mut Continue).unwrap()
}
fn admitted(p: &PreparedInt8Resolution<'_, '_, '_>) -> Vec<ExecutionIdentity> { p.execution_identities().cloned().collect() }
fn fake_head(schedule: CandidateSchedule, eos: u32, decision: PairDecision) -> Int8CandidateRun {
    let values: [f64; 3] = match decision { PairDecision::Same => [-4.0, -1.0, -4.0],
        PairDecision::Different => [-1.0, -4.0, -4.0], PairDecision::Uncertain => [-3.0; 3] };
    let sum: f64 = values.iter().map(|p| p.exp()).sum();
    let scores = CandidateScores { score_space: ScoreSpace::FullVocabSequenceLogprob,
        normalization_scope: "synthetic-candidate-set".to_owned(), length_rule: "none".to_owned(),
        eos_rule: "scored".to_owned(), eos_token_id: eos, full_vocab_denominators_computed: true,
        candidates: ["different", "same", "uncertain"].iter().zip(values).map(|(&id, p)| CandidateScore {
            id: id.to_owned(), scored_tokens: 2, sequence_score: p, candidate_weight: p.exp() / sum,
        }).collect(), work: ScoringWork { prefix_evaluations: 4, scored_edges: 6, projected_logits: HEAD_ROWS } };
    Int8CandidateRun { schema_version: 1, execution: INT8_SCORING_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), scores, model_work: schedule.model, rewound_positions: 2 }
}
#[test]
fn strict_planning_preserves_prompt_bytes_but_cannot_enter_eager_execution() {
    let p = planner(); let d = docs(); let plan = core(&d); let id = identity(&p);
    let before = canonjson::canonical_bytes(&id).unwrap(); let quantized = prepared(&p, &plan);
    assert!(p.prepare(&plan, &id, limits().planning, &mut Continue).is_err());
    let mut eager_id = id.clone(); eager_id.numerics_profile = NumericsProfile::HfBf16Eager;
    let eager = p.prepare(&plan, &eager_id, limits().planning, &mut Continue).unwrap();
    assert!(p.prepare_int8(&plan, &eager_id, limits(), &mut Continue).is_err());
    for (a, b) in eager.pairs.iter().zip(&quantized.inner.pairs) {
        assert_eq!(a.prompts, b.prompts);
        assert_ne!(a.identity.decision_policy_digest, b.identity.decision_policy_digest);
    }
    assert_eq!(before, canonjson::canonical_bytes(&id).unwrap());
}
#[test]
fn all_pair_orders_have_exact_branch_aware_native_geometry() {
    let p = planner(); let d = docs(); let plan = core(&d); let prepared = prepared(&p, &plan);
    assert_eq!(prepared.pair_count(), 3); let mut work = Int8Work::default();
    for (pair, schedules) in prepared.inner.pairs.iter().zip(&prepared.schedules) {
        for (prompt, s) in pair.prompts.iter().zip(schedules) {
            assert_eq!(s.context, prompt.len() + 1); assert_eq!(s.model.forward_positions, prompt.len() as u64 + 3);
            assert_eq!(s.scoring.projected_logits, HEAD_ROWS);
            // Three sibling labels rewind to the prompt; they are NOT three
            // sequentially extending output positions in one causal triangle.
            let wrong = Int8Work::for_sequence(0, prompt.len() + 3, HEAD_ROWS as usize).unwrap();
            assert!(s.model.attention_pairs < wrong.attention_pairs);
            assert_eq!(s.model.projections, wrong.projections);
            work = work.checked_add(s.model).unwrap();
        }
    }
    assert_eq!(prepared.planned_work(), work);
}
#[test]
fn each_whole_snapshot_counter_is_admitted_before_any_execution() {
    let p = planner(); let d = docs(); let plan = core(&d); let exact = prepared(&p, &plan).planned_work();
    let mut l = limits(); l.max_model_work = exact;
    p.prepare_int8(&plan, &identity(&p), l, &mut Continue).unwrap();
    for axis in 0..5 {
        let mut short = l;
        match axis { 0 => short.max_model_work.forward_positions -= 1, 1 => short.max_model_work.projected_logits -= 1,
            2 => short.max_model_work.attention_pairs -= 1, 3 => short.max_model_work.projections.dot_products -= 1,
            _ => short.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(matches!(p.prepare_int8(&plan, &identity(&p), short, &mut Continue), Err(Int8ResolveError::WorkBudget)));
    }
}
#[test]
fn mutated_last_identity_blocks_the_first_native_head() {
    let p = planner(); let d = docs(); let plan = core(&d);
    for axis in 0..5 {
        let prepared = prepared(&p, &plan); let mut ids = admitted(&prepared); let last = ids.last_mut().unwrap();
        let digest = Sha256Digest::of_bytes(b"substitution");
        match axis { 0 => last.prompt_digest = digest, 1 => last.logical_model_digest = digest,
            2 => last.taskir_digest = digest, 3 => last.decision_policy_digest = digest,
            _ => last.backend_semantic_version = "other".to_owned() }
        let mut calls = 0;
        assert!(matches!(prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| {
            calls += 1; Ok(fake_head(s, p.eos, PairDecision::Same))
        }), Err(Int8ResolveError::Identity)));
        assert_eq!(calls, 0);
    }
}
#[test]
fn missing_and_extra_pair_admissions_are_not_partial_success() {
    let p = planner(); let d = docs(); let plan = core(&d);
    for extra in [false, true] {
        let prepared = prepared(&p, &plan); let mut ids = admitted(&prepared);
        if extra { ids.push(ids[0].clone()); } else { ids.pop(); }
        let mut calls = 0;
        assert!(prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| { calls += 1; Ok(fake_head(s, p.eos, PairDecision::Same)) }).is_err());
        assert_eq!(calls, 0);
    }
}
#[test]
fn both_orders_and_all_pairs_feed_the_actual_complete_link_finalizer() {
    let p = planner(); let d = docs(); let plan = core(&d); let prepared = prepared(&p, &plan);
    let ids = admitted(&prepared); let expected = prepared.planned_work(); let mut sequence = Vec::new();
    let run = prepared.execute_heads(&ids, &mut Continue, |pair, order, _, s, _| {
        sequence.push((pair, order)); Ok(fake_head(s, p.eos, PairDecision::Same))
    }).unwrap();
    assert_eq!(sequence, vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
    assert_eq!(run.model_work, expected); assert_eq!(run.planned_model_work, expected);
    assert_eq!(run.head_count, 6); assert_eq!(run.rewound_positions, 12); assert!(run.model_evaluated);
    assert_eq!(run.result.clusters.len(), 1); assert_eq!(run.result.clusters[0].mentions, vec![0, 1, 2]);
    assert_eq!(run.result.calibration, "uncalibrated"); assert_eq!(run.result.mentions[0].span, d[0].mentions[0].span);
}
#[test]
fn disagreeing_presentation_orders_abstain_instead_of_averaging_away_conflict() {
    let p = planner(); let d = docs(); let plan = core(&d); let prepared = prepared(&p, &plan); let ids = admitted(&prepared);
    let run = prepared.execute_heads(&ids, &mut Continue, |_, order, _, s, _| {
        Ok(fake_head(s, p.eos, if order == 0 { PairDecision::Same } else { PairDecision::Different }))
    }).unwrap();
    assert!(run.result.judgments.iter().all(|j| j.decision == PairDecision::Uncertain));
    assert_eq!(run.result.clusters.len(), 3);
}
#[test]
fn transitive_same_edges_cannot_overrule_an_explicit_different_pair() {
    let p = planner(); let d = docs(); let plan = core(&d); let prepared = prepared(&p, &plan); let ids = admitted(&prepared);
    let run = prepared.execute_heads(&ids, &mut Continue, |pair, _, _, s, _| {
        Ok(fake_head(s, p.eos, if pair == 1 { PairDecision::Different } else { PairDecision::Same }))
    }).unwrap();
    assert_eq!(run.result.clusters.len(), 2); assert!(run.result.blocked_merges > 0);
    assert!(!run.result.clusters.iter().any(|c| c.mentions.contains(&0) && c.mentions.contains(&2)));
}
#[test]
fn no_candidate_graph_returns_singletons_without_fabricated_inference() {
    let p = planner(); let mut d = docs(); d.truncate(1); let plan = core(&d);
    let mut l = limits(); l.max_model_work = Int8Work::default(); l.planning.max_work = BatchWork::default();
    let prepared = p.prepare_int8(&plan, &identity(&p), l, &mut Continue).unwrap(); let mut calls = 0;
    let run = prepared.execute_heads(&[], &mut Continue, |_, _, _, s, _| { calls += 1; Ok(fake_head(s, p.eos, PairDecision::Same)) }).unwrap();
    assert_eq!(calls, 0); assert_eq!(run.head_count, 0); assert!(!run.model_evaluated);
    assert_eq!(run.model_work, Int8Work::default()); assert_eq!(run.result.clusters.len(), 1);
}
#[test]
fn late_pair_cancellation_discards_the_entire_snapshot() {
    let p = planner(); let d = docs(); let plan = core(&d); let prepared = prepared(&p, &plan); let ids = admitted(&prepared); let mut calls = 0;
    let error = prepared.execute_heads(&ids, &mut Continue, |pair, order, _, s, _| {
        calls += 1;
        if pair == 1 && order == 1 { Err(StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline).into()) }
        else { Ok(fake_head(s, p.eos, PairDecision::Same)) }
    }).err().unwrap();
    assert_eq!(calls, 4); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn cancellation_immediately_after_a_head_prevents_the_next_order() {
    struct Control(bool);
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.then_some(DecodeCancellationKind::User) }
    }
    let p = planner(); let d = docs(); let plan = core(&d); let prepared = prepared(&p, &plan); let ids = admitted(&prepared); let mut calls = 0;
    let error = prepared.execute_heads(&ids, &mut Control(false), |_, _, _, s, control| {
        calls += 1; control.0 = true; Ok(fake_head(s, p.eos, PairDecision::Same))
    }).err().unwrap();
    assert_eq!(calls, 1); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::User));
}
#[test]
fn every_native_receipt_axis_is_checked_before_the_second_order() {
    let p = planner(); let d = docs(); let plan = core(&d);
    for axis in 0..14 {
        let prepared = prepared(&p, &plan); let ids = admitted(&prepared); let mut calls = 0;
        let result = prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| {
            calls += 1; let mut r = fake_head(s, p.eos, PairDecision::Same);
            match axis {
                0 => r.schema_version = 99, 1 => r.execution = "other".to_owned(),
                2 => r.numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(),
                3 => r.model_work.forward_positions += 1, 4 => r.model_work.projected_logits -= 1,
                5 => r.model_work.attention_pairs -= 1, 6 => r.model_work.projections.dot_products -= 1,
                7 => r.model_work.projections.multiply_accumulates -= 1, 8 => r.rewound_positions = 0,
                9 => r.scores.full_vocab_denominators_computed = false, 10 => r.scores.eos_token_id ^= 1,
                11 => r.scores.candidates[0].sequence_score = f64::NAN, 12 => r.scores.candidates[0].scored_tokens = 1,
                _ => r.scores.work.scored_edges -= 1,
            }
            Ok(r)
        });
        assert!(result.is_err(), "axis {axis}"); assert_eq!(calls, 1);
    }
}
#[test]
fn output_and_cluster_limits_never_yield_a_partial_graph() {
    let p = planner(); let d = docs();
    for output in [true, false] {
        let mut graph_limits = ResolveLimits::default(); let mut l = limits();
        if output { l.planning.max_result_bytes = 1; } else { graph_limits.max_cluster_checks = 1; }
        let plan = ResolutionPlan::prepare(&d, ResolveOptions::default(), graph_limits, &mut Continue).unwrap();
        let prepared = p.prepare_int8(&plan, &identity(&p), l, &mut Continue).unwrap(); let ids = admitted(&prepared);
        let error = prepared.execute_heads(&ids, &mut Continue, |_, _, _, s, _| Ok(fake_head(s, p.eos, PairDecision::Same))).err().unwrap();
        assert!(matches!(error, Int8ResolveError::Resolution(ResolveError::OutputBudget | ResolveError::ClusterBudget)));
    }
}
#[test]
fn strict_profile_backend_tools_and_pinned_assets_are_not_repaired() {
    let p = planner(); let d = docs(); let plan = core(&d);
    for axis in 0..8 {
        let mut id = identity(&p);
        match axis { 0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            2 => id.backend_semantic_version = "other".to_owned(), 3 => id.kv_dtype = "int8".to_owned(),
            4 => id.thinking_mode = ThinkingMode::Enabled, 5 => id.tool_mode = ToolMode::Json,
            6 => id.template_digest = Sha256Digest::of_bytes(b"other"), _ => id.tokenizer_digest = Sha256Digest::of_bytes(b"other") }
        assert!(p.prepare_int8(&plan, &id, limits(), &mut Continue).is_err());
    }
}
#[test]
fn core_source_validation_precedes_every_int8_planning_or_model_call() {
    let mut d = docs(); d[2].mentions[0].span.byte_end += 1;
    assert!(matches!(ResolutionPlan::prepare(&d, ResolveOptions::default(), ResolveLimits::default(), &mut Continue), Err(ResolveError::InvalidAnchor)));
}
#[test]
fn enclosing_native_budget_cannot_omit_attention_or_integer_work() {
    let p = planner(); let d = docs(); let plan = core(&d); let work = prepared(&p, &plan).planned_work();
    check_budget(work, Int8RunBudget::exact(work)).unwrap();
    for axis in 0..4 {
        let mut b = Int8RunBudget::exact(work);
        match axis { 0 => b.max_forward_positions -= 1, 1 => b.max_attention_pairs -= 1,
            2 => b.max_projection_work.dot_products -= 1, _ => b.max_projection_work.multiply_accumulates -= 1 }
        assert!(matches!(check_budget(work, b), Err(Int8ResolveError::WorkBudget)));
    }
}
#[test]
fn typed_cancellation_survives_source_planning_and_scorer_wrappers() {
    let cause = DecodeCancellationKind::Deadline;
    for e in [Int8ResolveError::Resolution(ResolveError::Cancelled(cause)),
        Int8ResolveError::Planning(NativeResolveError::Resolution(ResolveError::Cancelled(cause))),
        Int8ResolveError::Scoring(Int8ScoringError::Native(StrictInt8Error::Cancelled(cause)))] {
        assert_eq!(e.cancellation(), Some(cause));
        assert_eq!(format!("{e}"), format!("{e:?}"));
    }
}

#[test]
fn no_model_finalization_cannot_skip_required_comparisons() {
    let p = planner(); let d = docs(); let plan = core(&d);
    assert!(matches!(prepared(&p, &plan).finalize_without_model(&mut Continue), Err(Int8ResolveError::Accounting)));
    let d = &d[..1]; let plan = core(d);
    let run = prepared(&p, &plan).finalize_without_model(&mut Continue).unwrap();
    assert!(!run.model_evaluated); assert_eq!(run.head_count, 0); assert_eq!(run.result.clusters.len(), 1);
}
#[test]
fn no_model_finalization_still_observes_cancellation() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Shutdown) } }
    let p = planner(); let d = docs(); let plan = core(&d[..1]);
    let error = prepared(&p, &plan).finalize_without_model(&mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Shutdown));
}
