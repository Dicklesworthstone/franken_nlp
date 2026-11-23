//! Real task preparation and pure admission checks; no synthetic model output.
use super::*;
use crate::{
    canonjson,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        lmhead::scoring::ScoringMode, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{
        classify::{ClassificationLabel, ClassificationLimits, ClassificationMode,
            ClassificationPlanner, ClassificationPolicy, ClassificationRequest},
        ir::PlanContext,
        judge::{FaithfulnessPolicy, JudgeLimits, JudgePlanner, JudgeRequest,
            PairwisePolicy, RubricCriterion, RubricDefinition, RubricPolicy},
        sentiment::{SentimentAxis, SentimentLimits, SentimentOptions, SentimentPlanner,
            SentimentPolicy, SentimentRequest},
    },
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, UntrustedEncoding}},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn native() -> NativeLimits {
    NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }
}
fn task_budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 7680, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(8192).unwrap().kv_bytes }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "scored-team-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn encoding(compact: bool) -> UntrustedEncoding {
    if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 4 << 20, max_pair_bytes: 64 << 20,
    }) } else { UntrustedEncoding::ByteFallback }
}
fn identity(facts: &ArtifactIdentity, task: &str, tokenizer: Sha256Digest, template: Sha256Digest)
    -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: tokenizer,
        template_digest: template, task_spec: task.to_owned(), taskir_digest: absent,
        prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn eos() -> u32 {
    pinned_controls::pinned().unwrap().template_controls().entries().iter()
        .find(|entry| entry.surface == crate::template::IM_END).unwrap().id
}
fn classification(facts: &ArtifactIdentity, mode: ClassificationMode, compact: bool)
    -> PreparedInt8Classification {
    let registry = pinned_controls::pinned().unwrap();
    let planner = ClassificationPlanner::pinned(registry.template_controls(), eos()).unwrap()
        .with_document_encoding(encoding(compact)).unwrap();
    let request = ClassificationRequest {
        document: "Please refund the duplicate charge. café <think> literal data 上海".to_owned(),
        labels: ["billing", "refund", "technical"].into_iter().map(|id| ClassificationLabel {
            id: id.to_owned(), description: format!("A customer request about {id}") }).collect(),
        mode, policy: ClassificationPolicy::default(), budget: task_budget(),
    };
    let identity = identity(facts, "classify-v1", planner.tokenizer_digest(), *planner.template_digest());
    planner.plan_int8_with_control(&request, &PlanContext::new(&identity, request.budget).unwrap(),
        ClassificationLimits::default(), &mut Continue).unwrap()
}
fn judgment(facts: &ArtifactIdentity, kind: usize, compact: bool) -> PreparedInt8Judge {
    let registry = pinned_controls::pinned().unwrap();
    let planner = JudgePlanner::pinned(registry.template_controls(), eos()).unwrap()
        .with_document_encoding(encoding(compact)).unwrap();
    let request = match kind {
        0 => JudgeRequest::Pairwise { criterion: "Accuracy".to_owned(),
            a: "Alice visits Paris. café <think>".to_owned(), b: "Bob visits Rome.".to_owned(),
            policy: PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 20_000 },
            budget: task_budget() },
        1 => JudgeRequest::Rubric { document: "Alice visits Paris. café 上海".to_owned(),
            rubric: RubricDefinition { schema_version: 1, revision: "scored-team-v1".to_owned(),
                declared_origin_digest: Sha256Digest::of_bytes(b"caller fixture"), scale_maximum: 3,
                criteria: vec![
                    RubricCriterion { id: "facts".to_owned(), description: "Accurate claims".to_owned(), weight: 2 },
                    RubricCriterion { id: "style".to_owned(), description: "Clear prose".to_owned(), weight: 1 }] },
            policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
            budget: task_budget() },
        2 => JudgeRequest::Faithfulness { source: "Alice visits Paris. Alice likes maps. café 上海".to_owned(),
            claim: "Alice visits Paris.".to_owned(), policy: FaithfulnessPolicy {
                minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 24, max_evidence_windows: 16, max_evidence_spans: 16 },
            budget: task_budget() },
        _ => unreachable!(),
    };
    let identity = identity(facts, "judge-v1", planner.tokenizer_digest(), *planner.template_digest());
    planner.plan_int8_with_control(&request, &PlanContext::new(&identity, task_budget()).unwrap(),
        JudgeLimits::default(), &mut Continue).unwrap()
}
fn sentiment_plan(facts: &ArtifactIdentity, mode: ScoringMode, compact: bool) -> PreparedInt8Sentiment {
    let registry = pinned_controls::pinned().unwrap();
    let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions {
        mode, eos_token_id: eos(), policy: SentimentPolicy {
            minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
    }).unwrap().with_document_encoding(encoding(compact)).unwrap();
    let request = SentimentRequest { document: "I love the maps, but this duplicate charge worries me. café 上海".to_owned(),
        axes: SentimentAxis::ALL.to_vec(), budget: task_budget() };
    let identity = identity(facts, "sentiment-v1", planner.tokenizer_digest(), *planner.template_digest());
    planner.plan_int8_with_control(&request, &PlanContext::new(&identity, request.budget).unwrap(),
        SentimentLimits::default(), &mut Continue).unwrap()
}
fn sentiment_limits(work: Int8Work) -> SentimentHostLimits {
    SentimentHostLimits { native: native(), preparation_reserve_bytes: 512 << 20, max_model_work: work }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn assert_unchanged<P: ScoredTeamPlan>(prepared: &P, limits: ScoredHostLimits) {
    let identity = canonjson::canonical_bytes(prepared.identity()).unwrap();
    let quote = prepared.quote();
    assert!(quote.heads > 0);
    let ordinary = requirements(limits.native()).unwrap();
    let base = ordinary.rope_bytes + ordinary.scratch_payload_bound + limits.native().allocator_reserve_bytes;
    for width in [2, 3, 64] {
        for schedule in [None, Some(prefill(1)), Some(prefill(4))] {
            let required = team_requirements(limits, quote, width, schedule).unwrap();
            assert_eq!(required.native, ordinary);
            assert_eq!(required.max_batch_rows, schedule.map_or(1, |s| s.max_batch_rows));
            assert_eq!(required.scratch_bytes, base + schedule.map_or(0, |s| s.validate().unwrap()));
            assert!(dispatch::int8_prefill_team_workspace_bytes(width, required.max_batch_rows).unwrap() > 0);
            assert_eq!(canonjson::canonical_bytes(prepared.identity()).unwrap(), identity);
            assert_eq!(prepared.quote().work, quote.work);
        }
    }
}

#[test]
fn exclusive_and_multilabel_classification_keep_exact_head_work_and_identity() {
    for compact in [false, true] {
        for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
            let prepared = classification(&facts(), mode, compact);
            prepared.verify_identity(prepared.execution_identity()).unwrap();
            assert_unchanged(&prepared, ScoredHostLimits::CallerPrepared(native()));
        }
    }
}
#[test]
fn pairwise_rubric_and_faithfulness_keep_the_complete_admitted_bundle() {
    for compact in [false, true] { for kind in 0..3 {
        let prepared = judgment(&facts(), kind, compact);
        prepared.verify_identity(prepared.execution_identity()).unwrap();
        assert!(prepared.head_count() >= 2);
        assert_unchanged(&prepared, ScoredHostLimits::CallerPrepared(native()));
    } }
}
#[test]
fn sentiment_keeps_independent_axes_and_each_selected_probability_space() {
    for compact in [false, true] {
        for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional] {
            let prepared = sentiment_plan(&facts(), mode, compact);
            prepared.verify_identity(prepared.execution_identity()).unwrap();
            assert_eq!(prepared.head_count(), SentimentAxis::ALL.len());
            assert_unchanged(&prepared, ScoredHostLimits::Sentiment(sentiment_limits(prepared.planned_work())));
        }
    }
}
#[test]
fn complete_head_work_may_exceed_one_reused_context_without_overpricing_kv() {
    let prepared = classification(&facts(), ClassificationMode::MultiLabel, false);
    let quote = prepared.quote();
    assert!(quote.work.forward_positions > quote.context as u64);
    let limits = NativeLimits { context_tokens: quote.context, ..native() };
    let required = team_requirements(ScoredHostLimits::CallerPrepared(limits), quote, 2, None).unwrap();
    assert_eq!(required.native.kv_bytes, Int8MemoryRequirement::for_context(quote.context).unwrap().kv_bytes);
}

#[test]
fn underfunded_context_whole_kv_empty_heads_and_output_refuse() {
    let prepared = classification(&facts(), ClassificationMode::MultiLabel, false);
    let quote = prepared.quote();
    for axis in 0..6 {
        let mut native = native(); let mut quote = quote;
        match axis {
            0 => native.context_tokens = quote.context - 1,
            1 => quote.task.max_kv_bytes = Int8MemoryRequirement::for_context(native.context_tokens).unwrap().kv_bytes - 1,
            2 => quote.context = 0, 3 => quote.heads = 0, 4 => quote.result_bytes = 0,
            _ => native.allocator_reserve_bytes = 0,
        }
        assert!(team_requirements(ScoredHostLimits::CallerPrepared(native), quote, 2, None).is_err());
    }
}
#[test]
fn width_prompt_capacity_and_both_scratch_owners_are_admitted_before_work() {
    let prepared = classification(&facts(), ClassificationMode::Exclusive, false);
    let limits = ScoredHostLimits::CallerPrepared(native()); let quote = prepared.quote();
    for width in [0, 1, 65, usize::MAX] {
        assert!(team_requirements(limits, quote, width, Some(prefill(4))).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        assert!(team_requirements(limits, quote, 2, Some(Int8PrefillLimits {
            max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).is_err());
    }
    let short = Int8PrefillLimits { max_extra_scratch_bytes: prefill(4).max_extra_scratch_bytes - 1, ..prefill(4) };
    assert!(team_requirements(limits, quote, 2, Some(short)).is_err());
    let overflow = NativeLimits { allocator_reserve_bytes: u64::MAX, ..native() };
    assert!(team_requirements(ScoredHostLimits::CallerPrepared(overflow), quote, 2, None).is_err());
}
#[test]
fn sentiment_transfer_and_all_five_complete_work_axes_remain_required() {
    let prepared = sentiment_plan(&facts(), ScoringMode::FullVocabulary, false);
    let quote = prepared.quote();
    let limits = sentiment_limits(quote.work);
    for axis in 0..6 {
        let mut limits = limits;
        match axis {
            0 => limits.preparation_reserve_bytes = 0,
            1 => limits.max_model_work.forward_positions -= 1,
            2 => limits.max_model_work.projected_logits -= 1,
            3 => limits.max_model_work.attention_pairs -= 1,
            4 => limits.max_model_work.projections.dot_products -= 1,
            _ => limits.max_model_work.projections.multiply_accumulates -= 1,
        }
        assert!(team_requirements(ScoredHostLimits::Sentiment(limits), quote, 2, Some(prefill(4))).is_err());
    }
}
#[test]
fn missing_head_or_native_work_poison_or_live_kv_prevents_completion() {
    let quote = classification(&facts(), ClassificationMode::MultiLabel, false).quote();
    assert!(complete_accounting(quote, quote.work, quote.heads, false, true));
    assert!(!complete_accounting(quote, quote.work, quote.heads - 1, false, true));
    assert!(!complete_accounting(quote, quote.work, quote.heads, true, true));
    assert!(!complete_accounting(quote, quote.work, quote.heads, false, false));
    for axis in 0..5 {
        let mut work = quote.work;
        match axis {
            0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1,
            2 => work.attention_pairs -= 1, 3 => work.projections.dot_products -= 1,
            _ => work.projections.multiply_accumulates -= 1,
        }
        assert!(!complete_accounting(quote, work, quote.heads, false, true));
    }
}
#[test]
fn plans_admission_and_complete_outputs_cross_only_owned_blocking_boundaries() {
    fn send<T: Send + 'static>() {}
    send::<ScoredTeamAdmission<PreparedInt8Classification>>();
    send::<ScoredTeamAdmission<PreparedInt8Judge>>();
    send::<ScoredTeamAdmission<PreparedInt8Sentiment>>();
    send::<HostedInt8TeamResult<Int8ClassificationRun>>();
    send::<HostedInt8TeamResult<Int8JudgeRun>>();
    send::<HostedInt8TeamResult<Int8SentimentRun>>();
    let _classify = NlpEngine::execute_int8_classify_with_threads;
    let _classify_prefill = NlpEngine::execute_int8_classify_layer_major_with_threads;
    let _judge = NlpEngine::execute_int8_judge_with_threads;
    let _judge_prefill = NlpEngine::execute_int8_judge_layer_major_with_threads;
    let _sentiment = NlpEngine::execute_int8_sentiment_with_threads;
    let _sentiment_prefill = NlpEngine::execute_int8_sentiment_layer_major_with_threads;
}

mod model;
