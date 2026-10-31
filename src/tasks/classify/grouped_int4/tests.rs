//! Actual pinned planning and semantic finalization with synthetic logits.
//! Only the explicitly ignored source test executes a full model.
use super::*;
use std::sync::OnceLock;
use crate::{execution_identity::{Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{CandidateLogits, ProjectionRows}},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy, MultiLabelDecision},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls}, template::IM_END};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn fixture() -> (&'static ClassificationPlanner, ExecutionIdentity, [u32; 3]) {
    static FIXTURE: OnceLock<(ClassificationPlanner, ExecutionIdentity, [u32; 3])> = OnceLock::new();
    let (planner, identity, codes) = FIXTURE.get_or_init(|| {
        let registry = pinned_controls::pinned().unwrap();
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        let eos = tokenizer.eos_token_id().unwrap();
        assert!(registry.template_controls().entry(eos).is_some_and(|e| e.special && e.surface == IM_END));
        let planner = ClassificationPlanner::pinned(registry.template_controls(), eos).unwrap();
        let d = Sha256Digest::of_bytes(b"synthetic-int4-classification-only");
        // Synthetic metadata is admitted ONLY through this module's private
        // compile seam. The public APIs additionally require real model facts.
        let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "synthetic-only".to_owned(), quant_recipe: "fixture-int4".to_owned(), packing_set_digest: d,
            tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: "classify-v1".to_owned(),
            taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None };
        let codes = [b"Y", b"N", b"A"].map(|s| tokenizer.tokenizer().encode_byte_fallback_only(s).unwrap()[0]);
        (planner, identity, codes)
    });
    (planner, identity.clone(), *codes)
}
fn request(mode: ClassificationMode) -> ClassificationRequest {
    ClassificationRequest { document: "é Refund requested <think>literal data</think>".to_owned(),
        labels: ["Billing issue", "Refund", "étiquette"].into_iter().map(|id| ClassificationLabel {
            id: id.to_owned(), description: format!("Definition of {id}; <|im_start|> stays data") }).collect(), mode,
        policy: ClassificationPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_ppm: 200_000 },
        budget: TaskBudget { max_input_tokens: 4096, max_output_tokens: 8, max_output_bytes: 1_000_000,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
}
fn prepare(p: &ClassificationPlanner, id: &ExecutionIdentity, r: &ClassificationRequest, width: usize)
    -> Result<PreparedInt4Classification, Int4ClassificationError> {
    p.compile_int4(r, &PlanContext::new(id, r.budget).unwrap(), ClassificationLimits::default(), width, &mut Continue)
}
struct Logits(Option<u32>);
impl CandidateLogits for Logits {
    type Error = &'static str;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("full denominator required") };
        let mut logits = vec![0.0; vocabulary_size];
        if prefix.is_empty() { if let Some(token) = self.0 { logits[token as usize] = 8.0; } }
        Ok(logits)
    }
}
fn evaluated(head: &ClassificationHead, schedule: CandidateSchedule, preferred: Option<u32>, width: usize) -> Int4CandidateRun {
    Int4CandidateRun { schema_version: 1, execution: INT4_SCORING_EXECUTION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        group_width: width, scores: head.classifier.scorer.score(&mut Logits(preferred), ScoringMode::FullVocabulary).unwrap(),
        model_work: schedule.model, rewound_positions: 0 }
}
#[test]
fn independent_binary_heads_include_exclude_and_abstain_without_partial_results() {
    let (p, id, [yes, no, _]) = fixture(); let r = request(ClassificationMode::MultiLabel);
    let prepared = prepare(p, &id, &r, 32).unwrap();
    let run = prepared.execute_heads(&mut Continue, |i, h, s, _| Ok(evaluated(h, s, [Some(yes), Some(no), None][i], 32))).unwrap();
    let ClassificationTaskResult::MultiLabel(result) = run.result else { panic!("multi-label required") };
    assert_eq!(result.selected_ids, ["Billing issue"]); assert_eq!(result.excluded_ids, ["Refund"]);
    assert_eq!(result.abstained_ids, ["étiquette"]); assert_eq!(result.labels[2].decision, MultiLabelDecision::Abstained);
    assert_eq!(run.head_count, 3); assert_eq!(run.model_work, prepared.planned_work());
    assert_eq!(run.group_width, 32); assert_eq!(run.numerics_profile, STRICT_INT4_PROFILE);
}
#[test]
fn positive_labels_are_not_renormalized_against_each_other() {
    let (p, id, [yes, _, _]) = fixture(); let r = request(ClassificationMode::MultiLabel);
    let prepared = prepare(p, &id, &r, 16).unwrap();
    let run = prepared.execute_heads(&mut Continue, |_, h, s, _| Ok(evaluated(h, s, Some(yes), 16))).unwrap();
    let ClassificationTaskResult::MultiLabel(result) = run.result else { panic!("multi-label required") };
    assert_eq!(result.selected_ids.len(), 3);
    assert!(result.labels.iter().map(|l| l.positive_candidate_weight).sum::<f64>() > 2.9);
}
#[test]
fn exclusive_labels_preserve_original_utf8_ids_and_complete_scored_eos_language() {
    let (p, id, [_, _, a]) = fixture(); let r = request(ClassificationMode::Exclusive);
    let prepared = prepare(p, &id, &r, 128).unwrap();
    let run = prepared.execute_heads(&mut Continue, |_, h, s, _| Ok(evaluated(h, s, Some(a), 128))).unwrap();
    let ClassificationTaskResult::Exclusive(result) = run.result else { panic!("exclusive required") };
    assert_eq!(result.selected_id.as_deref(), Some("Billing issue")); assert_eq!(result.scores.candidates.len(), 3);
    assert_eq!(result.scores.candidates[2].id, "étiquette"); assert_eq!(prepared.head_count(), 1);
    assert_eq!(result.scores.eos_rule, "append_and_score_exactly_one_eos");
    assert!(result.scores.full_vocab_denominators_computed);
}
#[test]
fn int4_compilation_never_relabels_other_profiles_backends_or_assets() {
    let (p, id, _) = fixture(); let r = request(ClassificationMode::Exclusive);
    for field in 0..5 {
        let mut other = id.clone();
        match field { 0 => other.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => other.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => other.backend_semantic_version = "other-cast-program".to_owned(),
            3 => other.tokenizer_digest = Sha256Digest::of_bytes(b"other"),
            _ => other.template_digest = Sha256Digest::of_bytes(b"other") }
        assert!(prepare(p, &other, &r, 32).is_err());
    }
    let context = PlanContext::new(&id, r.budget).unwrap();
    assert!(p.plan(&r, &context, ClassificationLimits::default()).is_err());
    assert!(p.plan_int8_with_control(&r, &context, ClassificationLimits::default(), &mut Continue).is_err());
}
#[test]
fn source_labels_policy_and_model_changes_bind_identity_but_input_order_does_not() {
    let (p, id, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let base = prepare(p, &id, &r, 32).unwrap();
    let mut reordered = r.clone(); reordered.labels.reverse();
    assert_eq!(base.execution_identity(), prepare(p, &id, &reordered, 32).unwrap().execution_identity());
    for field in 0..3 {
        let mut changed = r.clone();
        match field { 0 => changed.document.push('x'), 1 => changed.labels[0].description.push('x'), _ => changed.policy.minimum_margin_ppm += 1 }
        assert_ne!(base.execution_identity(), prepare(p, &id, &changed, 32).unwrap().execution_identity());
    }
    let mut changed = base.execution_identity().clone(); changed.logical_model_digest = Sha256Digest::of_bytes(b"foreign");
    assert_eq!(base.verify_identity(&changed), Err(Int4ClassificationError::Planning(ClassificationPlanningError::Identity)));
}
#[test]
fn shared_compiler_keeps_int8_and_int4_policy_programs_distinct() {
    let (p, id, _) = fixture(); let r = request(ClassificationMode::Exclusive);
    let int4 = prepare(p, &id, &r, 32).unwrap();
    let mut int8_id = id; int8_id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    int8_id.backend_semantic_version = crate::native_engine::strict_int8::STRICT_INT8_EXECUTION.to_owned();
    let int8 = p.plan_int8_with_control(&r, &PlanContext::new(&int8_id, r.budget).unwrap(),
        ClassificationLimits::default(), &mut Continue).unwrap();
    assert_eq!(int4.execution_identity().prompt_digest, int8.execution_identity().prompt_digest);
    assert_ne!(int4.execution_identity().decision_policy_digest, int8.execution_identity().decision_policy_digest);
}
#[test]
fn aggregate_work_sums_all_heads_but_context_is_only_the_largest_live_head() {
    let (p, id, _) = fixture(); let r = request(ClassificationMode::MultiLabel);
    for width in [3, 16, 32, 128, 10752] {
        let prepared = prepare(p, &id, &r, width).unwrap();
        let sum = prepared.schedules.iter().try_fold(Int4Work::default(), |a, s| a.checked_add(s.model)).unwrap();
        assert_eq!(sum, prepared.planned_work()); assert_eq!(sum.projected_logits, 9 * NANBEIGE_VOCAB_SIZE as u64);
        assert!(sum.forward_positions > prepared.required_context() as u64);
        assert_eq!(prepared.required_context(), prepared.schedules.iter().map(|s| s.context).max().unwrap());
        for axis in 0..4 {
            let mut budget = Int4RunBudget::exact(sum);
            match axis { 0 => budget.max_forward_positions -= 1, 1 => budget.max_attention_pairs -= 1,
                2 => budget.max_projection_work.dot_products -= 1, _ => budget.max_projection_work.multiply_accumulates -= 1 }
            assert_eq!(check_work_ceiling(sum, budget), Err(Int4ClassificationError::WorkBudget));
        }
    }
}
#[test]
fn later_head_failure_is_not_a_successful_abstention_or_partial_bundle() {
    let (p, id, [yes, _, _]) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(p, &id, &r, 32).unwrap();
    let mut calls = 0;
    let error = prepared.execute_heads(&mut Continue, |i, h, s, _| {
        calls += 1;
        if i == 1 { return Err(StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline).into()); }
        Ok(evaluated(h, s, Some(yes), 32))
    }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls, 2);
}
#[test]
fn corrupt_metadata_group_or_any_native_work_axis_refuses_complete_bundle() {
    let (p, id, [yes, _, _]) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(p, &id, &r, 32).unwrap();
    for field in 0..10 {
        let result = prepared.execute_heads(&mut Continue, |_, h, s, _| {
            let mut run = evaluated(h, s, Some(yes), 32);
            match field { 0 => run.schema_version += 1, 1 => run.execution.push('x'), 2 => run.numerics_profile.push('x'),
                3 => run.group_width += 1, 4 => run.model_work.forward_positions += 1,
                5 => run.model_work.projected_logits += 1, 6 => run.model_work.attention_pairs += 1,
                7 => run.model_work.projections.dot_products += 1, 8 => run.model_work.projections.multiply_accumulates += 1,
                _ => run.scores.work.projected_logits += 1 }
            Ok(run)
        });
        assert_eq!(result.unwrap_err(), Int4ClassificationError::Accounting);
    }
}
#[test]
fn finalizer_rejects_truncated_candidate_sets_and_wrong_denominator_claims() {
    let (p, id, [yes, _, _]) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(p, &id, &r, 32).unwrap();
    for field in 0..3 {
        assert!(prepared.execute_heads(&mut Continue, |_, h, s, _| {
            let mut run = evaluated(h, s, Some(yes), 32);
            match field { 0 => { run.scores.candidates.pop(); },
                1 => run.scores.full_vocab_denominators_computed = false,
                _ => run.scores.eos_rule = "unscored-eos".to_owned() }
            Ok(run)
        }).is_err());
    }
}
#[test]
fn complete_wrapper_obeys_output_cap_without_returning_partial_heads() {
    let (p, id, [yes, _, _]) = fixture(); let r = request(ClassificationMode::MultiLabel); let mut prepared = prepare(p, &id, &r, 32).unwrap();
    let baseline = prepared.execute_heads(&mut Continue, |_, h, s, _| Ok(evaluated(h, s, Some(yes), 32))).unwrap();
    let bytes = canonjson::canonical_bytes(&baseline).unwrap();
    prepared.inner.budget.max_output_bytes = bytes.len() as u64;
    assert!(prepared.execute_heads(&mut Continue, |_, h, s, _| Ok(evaluated(h, s, Some(yes), 32))).is_ok());
    prepared.inner.budget.max_output_bytes = 1;
    assert!(prepared.execute_heads(&mut Continue, |_, h, s, _| Ok(evaluated(h, s, Some(yes), 32))).is_err());
    let round: Int4ClassificationRun = serde_json::from_slice(&bytes).unwrap(); assert_eq!(round, baseline);
}
#[test]
fn prehead_and_finalization_cancellation_preserve_original_cause() {
    struct Stop { polls: usize, at: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.polls == self.at).then_some(DecodeCancellationKind::User)
        }
    }
    let (p, id, [yes, _, _]) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(p, &id, &r, 32).unwrap();
    for at in [1, prepared.head_count() + 1] {
        let mut control = Stop { polls: 0, at }; let mut calls = 0;
        let error = prepared.execute_heads(&mut control, |_, h, s, _| { calls += 1; Ok(evaluated(h, s, Some(yes), 32)) }).unwrap_err();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::User));
        assert_eq!(calls, at - 1);
    }
}

#[test]
#[ignore = "requires controller-admitted pinned source and full-model INT4 execution"]
fn source_model_factory_classifies_both_modes_reuses_kv_and_rejects_foreign_identity() {
    use crate::{artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input},
        native_engine::strict_int4::{Int4ModelWeights, Int4ModelLoadBudget, Int4ModelLoadRequirement,
            Int4MemoryRequirement, Int4MemoryBudget, cohort::Int4CohortEngine,
            scoring::cohort::Int4ScoringCohortBudget}};
    let directory = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
    let source = prepare_nanbeige42_input(directory, ConversionSourceManifest::load_pinned(manifest).unwrap(), false, 65536).unwrap();
    let load = Int4ModelLoadRequirement::for_group_width(32).unwrap();
    let weights = Int4ModelWeights::from_prepared(&source, 32, Int4ModelLoadBudget {
        max_payload_bytes: load.peak_payload_bytes().unwrap(), max_source_values: load.source_values,
        max_groups: load.groups, max_shard_revalidation_bytes: load.shard_revalidation_bytes,
    }, &mut Continue).unwrap();
    let memory = Int4MemoryRequirement::for_context(4096, 32).unwrap();
    let mut engine = StrictInt4Engine::new(&weights, 4096, Int4MemoryBudget { max_kv_bytes: memory.kv_bytes,
        max_rope_bytes: memory.rope_bytes, max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
    let mut plans = Vec::new(); let mut serial = Vec::new();
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let mut request = request(mode); request.budget.max_kv_bytes = memory.kv_bytes;
        if mode == ClassificationMode::MultiLabel {
            request.document.push_str(" A separate longer document asks for a refund after a duplicate billing charge.");
        }
        let planner = Int4ClassificationPlanner::for_model(weights.identity(), request.budget, ClassificationLimits::default()).unwrap();
        let prepared = planner.plan(&request).unwrap();
        check_model(prepared.execution_identity(), weights.identity()).unwrap();
        let mut foreign = prepared.execution_identity().clone(); foreign.logical_model_digest = Sha256Digest::of_bytes(b"foreign");
        assert_eq!(check_model(&foreign, weights.identity()), Err(Int4ClassificationError::Identity));
        let budget = Int4ScoringBudget { native: Int4RunBudget::exact(prepared.planned_work()), max_kv_bytes: memory.kv_bytes };
        let first = prepared.execute_with_control(prepared.execution_identity(), &mut engine, budget, &mut Continue).unwrap();
        assert!(!engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0));
        let second = prepared.execute_with_control(prepared.execution_identity(), &mut engine, budget, &mut Continue).unwrap();
        assert_eq!(first, second); assert_eq!(first.model_work, prepared.planned_work());
        assert!(engine.kv_cache().all_slots_have_len(0));
        serial.push(first); plans.push(prepared);
    }
    drop(engine);
    let contexts: Vec<_> = plans.iter().flat_map(|plan| plan.cohort_contexts().unwrap()).collect();
    assert_eq!(contexts.len(), 4);
    let memory = Int4MemoryRequirement::for_cohort(&contexts, 32).unwrap();
    let mut engine = Int4CohortEngine::new(&weights, &contexts, Int4MemoryBudget {
        max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
        max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
    let work = plans.iter().fold(Int4Work::default(), |sum, plan| sum.checked_add(plan.planned_work()).unwrap());
    let budget = Int4ScoringCohortBudget { native: Int4RunBudget::exact(work),
        max_kv_bytes: memory.kv_bytes, max_output_bytes: 4_000_000 };
    let documents: Vec<_> = plans.iter().map(|plan| (plan, plan.execution_identity())).collect();
    let grouped = PreparedInt4Classification::execute_document_cohort_with_control(
        &documents, &mut engine, budget, &mut Continue).unwrap();
    assert_eq!(grouped.output, serial);
    assert!(grouped.group_steps < work.forward_positions);
    engine.check_idle().unwrap();
    let replay = PreparedInt4Classification::execute_document_cohort_with_control(
        &documents, &mut engine, budget, &mut Continue).unwrap();
    assert_eq!(grouped, replay);
    engine.check_idle().unwrap();
    let mut foreign = plans[1].execution_identity().clone();
    foreign.prompt_digest = Sha256Digest::of_bytes(b"other document");
    let mismatched = [(documents[0].0, documents[0].1), (&plans[1], &foreign)];
    assert!(matches!(PreparedInt4Classification::execute_document_cohort_with_control(
        &mismatched, &mut engine, budget, &mut Continue), Err(Int4ClassificationError::Identity)
            | Err(Int4ClassificationError::Planning(ClassificationPlanningError::Identity))));
    engine.check_idle().unwrap();
}
