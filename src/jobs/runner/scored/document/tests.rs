//! Real pinned factories, immutable keyed recipes and lifetime contracts.
//! These checks require no model and do not stand in for native inference,
//! spool interruption or task-quality qualification.
use super::*;
use std::sync::OnceLock;
use crate::{
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringMode,
        strict_int8::{Int8MemoryRequirement, STRICT_INT8_EXECUTION}},
    tasks::{classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy},
        sentiment::{SentimentAxis, SentimentOptions, SentimentPolicy}},
    tokenizer::{pinned_controls, untrusted::SafeBpeLimits},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 262_144,
        max_output_tokens: 262_144, max_pair_checks: 1 << 25, max_pair_bytes: 1 << 28 })
}
fn total() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 24, max_output_tokens: 1 << 24,
        max_pair_checks: 1 << 30, max_pair_bytes: 1 << 34 }
}
fn classifier(compact: bool) -> Arc<ClassificationPlanner> {
    static BYTE: OnceLock<Arc<ClassificationPlanner>> = OnceLock::new();
    static COMPACT: OnceLock<Arc<ClassificationPlanner>> = OnceLock::new();
    Arc::clone((if compact { &COMPACT } else { &BYTE }).get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        Arc::new(ClassificationPlanner::pinned(controls.template_controls(), 166_101).unwrap()
            .with_document_encoding(if compact { policy() } else { UntrustedEncoding::ByteFallback }).unwrap())
    }))
}
fn sentiment(compact: bool) -> Arc<SentimentPlanner> {
    static BYTE: OnceLock<Arc<SentimentPlanner>> = OnceLock::new();
    static COMPACT: OnceLock<Arc<SentimentPlanner>> = OnceLock::new();
    Arc::clone((if compact { &COMPACT } else { &BYTE }).get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        Arc::new(SentimentPlanner::pinned(controls.template_controls(), SentimentOptions {
            mode: ScoringMode::FullVocabulary, eos_token_id: 166_101,
            policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
        }).unwrap().with_document_encoding(if compact { policy() } else { UntrustedEncoding::ByteFallback }).unwrap())
    }))
}
fn judge(compact: bool) -> Arc<JudgePlanner> {
    static BYTE: OnceLock<Arc<JudgePlanner>> = OnceLock::new();
    static COMPACT: OnceLock<Arc<JudgePlanner>> = OnceLock::new();
    Arc::clone((if compact { &COMPACT } else { &BYTE }).get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        Arc::new(JudgePlanner::pinned(controls.template_controls(), 166_101).unwrap()
            .with_document_encoding(if compact { policy() } else { UntrustedEncoding::ByteFallback }).unwrap())
    }))
}
fn identity(task: &str, template: Sha256Digest, tokenizer: Sha256Digest) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"document-job-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(),
        packing_set_digest: d, tokenizer_digest: tokenizer, template_digest: template,
        task_spec: task.to_owned(), taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 1 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(1024).unwrap().kv_bytes * 4 }
}
fn work() -> Int8Work { Int8Work::for_sequence(0, 65536, 100_000_000).unwrap() }
fn chunks() -> ChunkLimits {
    ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 768,
        context_tokens: 1024, reserved_tokens: 16, max_chunks: 64, max_tokenizer_calls: 4096 }
}
fn classification_mapping() -> Int8ClassificationMapLimits {
    Int8ClassificationMapLimits { chunks: chunks(), max_model_work: work(),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 1 << 20,
        aggregation: ClassificationAggregationPolicy::default() }
}
fn sentiment_mapping() -> Int8SentimentMapLimits {
    Int8SentimentMapLimits { chunks: chunks(), max_model_work: work(),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 1 << 20,
        aggregation: SentimentAggregationPolicy::default() }
}
fn faithfulness_mapping() -> Int8FaithfulnessMapLimits {
    Int8FaithfulnessMapLimits { chunks: chunks(), max_model_work: work(),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 1 << 20,
        aggregation: FaithfulnessAggregationPolicy::default() }
}
fn faithfulness_policy() -> FaithfulnessPolicy {
    FaithfulnessPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_milli: 100,
        evidence_window_bytes: 64, max_evidence_windows: 31, max_evidence_spans: 8 }
}
fn faithfulness(compact: bool, planning: JudgeLimits, mapping: Int8FaithfulnessMapLimits,
    policy: FaithfulnessPolicy, model: Int8Work, totals: Option<SafeBpeWorkLimits>)
    -> Int8FaithfulnessDocumentJobPlanner {
    let planner = judge(compact);
    let identity = identity("judge-v1", *planner.template_digest(), planner.tokenizer_digest());
    Int8FaithfulnessDocumentJobPlanner::new(planner, identity, budget(), planning,
        mapping, policy, model, totals).unwrap()
}
fn defaults() -> ClassificationBatchArgs {
    ClassificationBatchArgs { labels: ["alpha", "beta"].map(|id| ClassificationLabel {
        id: id.to_owned(), description: format!("Private definition of {id}") }).to_vec(),
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: budget() }
}
fn classification(compact: bool, mapping: Int8ClassificationMapLimits,
    defaults: Option<ClassificationBatchArgs>, total: Option<SafeBpeWorkLimits>)
    -> Int8ClassificationDocumentJobPlanner {
    let planner = classifier(compact);
    let identity = identity("classify-v1", *planner.template_digest(), planner.tokenizer_digest());
    Int8ClassificationDocumentJobPlanner::new(planner, identity, budget(), ClassificationLimits::default(),
        mapping, defaults, work(), total).unwrap()
}
fn sentiment_config(planner: &SentimentPlanner, compact: bool) -> Int8SentimentDocumentBatchConfig {
    Int8SentimentDocumentBatchConfig {
        identity: identity("sentiment-v1", *planner.template_digest(), planner.tokenizer_digest()),
        task_ceiling: budget(), planning: SentimentLimits::default(), mapping: sentiment_mapping(),
        defaults: Some(SentimentBatchArgs { axes: SentimentAxis::ALL.to_vec(), budget: budget() }),
        max_model_work: work(), max_tokenizer_work: compact.then(total),
    }
}
fn lifetime() -> JobLimits {
    JobLimits { max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 2 << 20,
        max_spool_bytes: 16 << 20, max_materialized_bytes: 16 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 16,
        max_work: JobWork { model: work(), mask_node_visits: 0 } }
}
const INPUT: &[u8] = br#"{"id":"source","text":"original complete private document"}"#;
fn freeze<R: Serialize>(identity: &ExecutionIdentity, recipe: &R, input: &[u8], secret: u8) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([secret; 32]), JobContract {
        job_id: JobId([19; 16]), execution: identity, recipe, limits: lifetime() },
        [JobInput { id: "source", original: input, normalized: input }], &mut Continue).unwrap()
}

#[test]
fn complete_classification_freezes_every_chunk_limit_and_reducer_setting() {
    let factory = classification(false, classification_mapping(), Some(defaults()), None);
    let original = freeze(factory.execution_identity(), factory.job_recipe(), INPUT, 7);
    for axis in 0..10 {
        let mut mapping = classification_mapping();
        match axis {
            0 => mapping.chunks.max_input_bytes += 1,
            1 => mapping.chunks.max_chunk_bytes += 1,
            2 => mapping.chunks.max_chunk_tokens += 1,
            3 => mapping.chunks.context_tokens += 1,
            4 => mapping.chunks.reserved_tokens += 1,
            5 => mapping.chunks.max_chunks += 1,
            6 => mapping.chunks.max_tokenizer_calls += 1,
            7 => mapping.max_retained_result_bytes += 1,
            8 => mapping.max_result_bytes += 1,
            _ => mapping.aggregation.minimum_classified_coverage_ppm -= 1,
        }
        let changed = classification(false, mapping, Some(defaults()), None);
        assert_eq!(original.binding.compare(&freeze(changed.execution_identity(), changed.job_recipe(), INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "map axis {axis}");
    }
}

#[test]
fn original_source_defaults_order_and_secret_remain_resume_authorities() {
    let factory = classification(false, classification_mapping(), Some(defaults()), None);
    let original = freeze(factory.execution_identity(), factory.job_recipe(), INPUT, 7);
    original.binding.compare(&freeze(factory.execution_identity(), factory.job_recipe(), INPUT, 7).binding).unwrap();
    for axis in 0..4 {
        let mut args = defaults();
        match axis { 0 => args.labels[0].description.push('!'), 1 => args.labels.swap(0, 1),
            2 => args.mode = ClassificationMode::MultiLabel, _ => args.policy.minimum_margin_ppm += 1 }
        let changed = classification(false, classification_mapping(), Some(args), None);
        assert_eq!(original.binding.compare(&freeze(changed.execution_identity(), changed.job_recipe(), INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    assert_eq!(original.binding.compare(&freeze(factory.execution_identity(), factory.job_recipe(),
        br#"{"id":"source","text":"changed source"}"#, 7).binding),
        Err(JobError::Mismatch(MismatchField::Population)));
    assert_eq!(original.binding.compare(&freeze(factory.execution_identity(), factory.job_recipe(), INPUT, 8).binding),
        Err(JobError::Mismatch(MismatchField::Secret)));
}

#[test]
fn all_compact_invocation_counters_are_bound_even_with_identical_inputs() {
    let factory = classification(true, classification_mapping(), Some(defaults()), Some(total()));
    let original = freeze(factory.execution_identity(), factory.job_recipe(), INPUT, 7);
    for axis in 0..4 {
        let mut changed = total();
        match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_pair_checks += 1, _ => changed.max_pair_bytes += 1 }
        let changed = classification(true, classification_mapping(), Some(defaults()), Some(changed));
        assert_eq!(original.binding.compare(&freeze(changed.execution_identity(), changed.job_recipe(), INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    let bytes = canonjson::canonical_bytes(factory.job_recipe()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["scoring"]["input_encoding"], serde_json::to_value(policy()).unwrap());
    assert_eq!(value["scoring"]["encoding_work_limits"], serde_json::to_value(total()).unwrap());
}

#[test]
fn sentiment_documents_authenticate_complete_axes_aggregation_and_both_work_scopes() {
    let planner = sentiment(true);
    let original = Int8SentimentDocumentJobPlanner::new(Arc::clone(&planner), sentiment_config(&planner, true)).unwrap();
    let binding = freeze(original.execution_identity(), original.job_recipe(), INPUT, 7);
    for axis in 0..10 {
        let mut config = sentiment_config(&planner, true);
        match axis {
            0 => { config.defaults.as_mut().unwrap().axes.pop(); },
            1 => config.mapping.aggregation.minimum_estimated_coverage_ppm -= 1,
            2 => config.mapping.max_result_bytes += 1,
            3 => config.mapping.max_retained_result_bytes += 1,
            4 => config.mapping.chunks.max_chunk_bytes += 1,
            5 => config.mapping.max_model_work.forward_positions -= 1,
            6 => config.mapping.max_model_work.attention_pairs -= 1,
            7 => config.max_model_work.projected_logits += 1,
            8 => config.max_model_work.projections.multiply_accumulates += 1,
            _ => config.max_tokenizer_work.as_mut().unwrap().max_pair_bytes += 1,
        }
        let changed = Int8SentimentDocumentJobPlanner::new(Arc::clone(&planner), config).unwrap();
        assert_eq!(binding.binding.compare(&freeze(changed.execution_identity(), changed.job_recipe(), INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "sentiment axis {axis}");
    }
    assert_ne!(original.job_recipe().execution, INT8_SENTIMENT_EXECUTION);
    assert_eq!(original.job_recipe().dependency_scope, "item-local-complete-document");
}

#[test]
fn complete_maps_must_fit_input_result_and_every_persisted_native_counter() {
    let factory = classification(false, classification_mapping(), Some(defaults()), None);
    factory.check_job_limits(lifetime()).unwrap();
    let planner = sentiment(false);
    let sentiment = Int8SentimentDocumentJobPlanner::new(Arc::clone(&planner), sentiment_config(&planner, false)).unwrap();
    sentiment.check_job_limits(lifetime()).unwrap();
    for axis in 0..7 {
        let mut limits = lifetime();
        match axis {
            0 => limits.max_input_bytes_per_item = chunks().max_input_bytes - 1,
            1 => limits.max_result_bytes = classification_mapping().max_result_bytes as usize - 1,
            2 => limits.max_work.model.forward_positions -= 1,
            3 => limits.max_work.model.projected_logits -= 1,
            4 => limits.max_work.model.attention_pairs -= 1,
            5 => limits.max_work.model.projections.dot_products -= 1,
            _ => limits.max_work.model.projections.multiply_accumulates -= 1,
        }
        assert!(factory.check_job_limits(limits).is_err(), "classification axis {axis}");
        assert!(sentiment.check_job_limits(limits).is_err(), "sentiment axis {axis}");
    }
}

#[test]
fn missing_compact_invocation_allowance_is_refused_by_both_factories() {
    let p = classifier(true);
    assert!(Int8ClassificationDocumentJobPlanner::new(Arc::clone(&p),
        identity("classify-v1", *p.template_digest(), p.tokenizer_digest()), budget(),
        ClassificationLimits::default(), classification_mapping(), Some(defaults()), work(), None).is_err());
    let p = sentiment(true); let mut config = sentiment_config(&p, true); config.max_tokenizer_work = None;
    assert!(Int8SentimentDocumentJobPlanner::new(p, config).is_err());
}

#[test]
fn factory_keepalive_retains_original_overallocated_defaults_and_recipe_copies() {
    let mut args = defaults();
    args.labels.reserve_exact(20);
    args.labels[0].description.reserve_exact(10_000);
    let original = classification_defaults_bytes(Some(&args)).unwrap();
    let factory = Arc::new(classification(false, classification_mapping(), Some(args), None));
    assert!(factory.retained_defaults_bytes().unwrap() > original);
    let weak = Arc::downgrade(&factory);
    let retained: Box<dyn Send> = Box::new(Arc::clone(&factory));
    drop(factory);
    assert!(weak.upgrade().is_some());
    drop(retained);
    assert!(weak.upgrade().is_none());
}

#[test]
fn prepared_invocation_identity_cannot_be_reconstructed_by_an_equivalent_factory() {
    let original = Arc::new(());
    verify_owner(&original, &Arc::clone(&original)).unwrap();
    let replacement = Arc::new(());
    assert!(verify_owner(&original, &replacement).unwrap_err().stop);
}

const FAITHFULNESS_INPUT: &[u8] =
    br#"{"id":"source","text":"complete private source","task_args":{"claim":"original private claim"}}"#;

#[test]
fn faithfulness_resume_authenticates_every_map_and_evidence_setting() {
    let factory = faithfulness(false, JudgeLimits::default(), faithfulness_mapping(),
        faithfulness_policy(), work(), None);
    let original = freeze(factory.execution_identity(), factory.job_recipe(), FAITHFULNESS_INPUT, 7);
    for axis in 0..15 {
        let mut mapping = faithfulness_mapping(); let mut policy = faithfulness_policy();
        match axis {
            0 => mapping.chunks.max_input_bytes += 1,
            1 => mapping.chunks.max_chunk_bytes += 1,
            2 => mapping.chunks.max_chunk_tokens += 1,
            3 => mapping.chunks.context_tokens += 1,
            4 => mapping.chunks.reserved_tokens += 1,
            5 => mapping.chunks.max_chunks += 1,
            6 => mapping.chunks.max_tokenizer_calls += 1,
            7 => mapping.max_retained_result_bytes += 1,
            8 => mapping.max_result_bytes += 1,
            9 => mapping.aggregation.minimum_classified_coverage_ppm -= 1,
            10 => policy.minimum_candidate_weight_ppm += 1,
            11 => policy.minimum_margin_milli += 1,
            12 => policy.evidence_window_bytes += 1,
            13 => policy.max_evidence_windows -= 1,
            _ => policy.max_evidence_spans += 1,
        }
        let changed = faithfulness(false, JudgeLimits::default(), mapping, policy, work(), None);
        assert_eq!(original.binding.compare(&freeze(changed.execution_identity(),
            changed.job_recipe(), FAITHFULNESS_INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "faithfulness axis {axis}");
    }
    assert_eq!(factory.job_recipe().execution, INT8_FAITHFULNESS_MAP_EXECUTION);
    assert_eq!(factory.job_recipe().dependency_scope, "item-local-complete-source-claim");
}

#[test]
fn faithfulness_resume_freezes_every_head_planning_limit_and_work_axis() {
    let factory = faithfulness(false, JudgeLimits::default(), faithfulness_mapping(),
        faithfulness_policy(), work(), None);
    let original = freeze(factory.execution_identity(), factory.job_recipe(), FAITHFULNESS_INPUT, 7);
    for axis in 0..20 {
        let mut planning = JudgeLimits::default(); let mut mapping = faithfulness_mapping();
        let mut invocation = work();
        match axis {
            0 => planning.per_head.max_candidates += 1,
            1 => planning.per_head.max_total_tokens += 1,
            2 => planning.per_head.max_nodes += 1,
            3 => planning.per_head.max_depth += 1,
            4 => planning.per_head.max_candidate_id_bytes += 1,
            5 => planning.per_head.max_projected_logits += 1,
            6 => planning.max_total_prompt_tokens += 1,
            7 => planning.max_total_candidate_tokens += 1,
            8 => planning.max_total_projected_logits += 1,
            9 => planning.max_output_bytes += 1,
            10 => mapping.max_model_work.forward_positions -= 1,
            11 => mapping.max_model_work.projected_logits -= 1,
            12 => mapping.max_model_work.attention_pairs -= 1,
            13 => mapping.max_model_work.projections.dot_products -= 1,
            14 => mapping.max_model_work.projections.multiply_accumulates -= 1,
            15 => invocation.forward_positions += 1,
            16 => invocation.projected_logits += 1,
            17 => invocation.attention_pairs += 1,
            18 => invocation.projections.dot_products += 1,
            _ => invocation.projections.multiply_accumulates += 1,
        }
        let changed = faithfulness(false, planning, mapping, faithfulness_policy(), invocation, None);
        assert_eq!(original.binding.compare(&freeze(changed.execution_identity(),
            changed.job_recipe(), FAITHFULNESS_INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "faithfulness planning axis {axis}");
    }
}

#[test]
fn faithfulness_keeps_exact_claim_source_and_secret_as_population_authorities() {
    let factory = faithfulness(false, JudgeLimits::default(), faithfulness_mapping(),
        faithfulness_policy(), work(), None);
    let original = freeze(factory.execution_identity(), factory.job_recipe(), FAITHFULNESS_INPUT, 7);
    original.binding.compare(&freeze(factory.execution_identity(), factory.job_recipe(), FAITHFULNESS_INPUT, 7).binding).unwrap();
    for input in [
        br#"{"id":"source","text":"changed private source","task_args":{"claim":"original private claim"}}"#.as_slice(),
        br#"{"id":"source","text":"complete private source","task_args":{"claim":"different private claim"}}"#.as_slice(),
    ] {
        assert_eq!(original.binding.compare(&freeze(factory.execution_identity(), factory.job_recipe(), input, 7).binding),
            Err(JobError::Mismatch(MismatchField::Population)));
    }
    assert_eq!(original.binding.compare(&freeze(factory.execution_identity(), factory.job_recipe(), FAITHFULNESS_INPUT, 8).binding),
        Err(JobError::Mismatch(MismatchField::Secret)));
    let stored = canonjson::canonical_bytes(&(&original.binding, &original.items)).unwrap();
    assert!(!String::from_utf8(stored).unwrap().contains("private"));
}

#[test]
fn faithfulness_compact_encoding_and_each_invocation_counter_authenticate_on_resume() {
    let factory = faithfulness(true, JudgeLimits::default(), faithfulness_mapping(),
        faithfulness_policy(), work(), Some(total()));
    let original = freeze(factory.execution_identity(), factory.job_recipe(), FAITHFULNESS_INPUT, 7);
    for axis in 0..4 {
        let mut changed = total();
        match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_pair_checks += 1, _ => changed.max_pair_bytes += 1 }
        let changed = faithfulness(true, JudgeLimits::default(), faithfulness_mapping(),
            faithfulness_policy(), work(), Some(changed));
        assert_eq!(original.binding.compare(&freeze(changed.execution_identity(),
            changed.job_recipe(), FAITHFULNESS_INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "tokenizer total axis {axis}");
    }
    let value = serde_json::to_value(factory.job_recipe()).unwrap();
    assert_eq!(value["input_encoding"], serde_json::to_value(policy()).unwrap());
    assert_eq!(value["encoding_work_limits"], serde_json::to_value(total()).unwrap());
    let planner = judge(true);
    assert!(Int8FaithfulnessDocumentJobPlanner::new(Arc::clone(&planner),
        identity("judge-v1", *planner.template_digest(), planner.tokenizer_digest()), budget(),
        JudgeLimits::default(), faithfulness_mapping(), faithfulness_policy(), work(), None).is_err());
    let planner = judge(false);
    assert!(Int8FaithfulnessDocumentJobPlanner::new(Arc::clone(&planner),
        identity("judge-v1", *planner.template_digest(), planner.tokenizer_digest()), budget(),
        JudgeLimits::default(), faithfulness_mapping(), faithfulness_policy(), work(), Some(total())).is_err());
}

#[test]
fn faithfulness_whole_document_must_fit_output_input_and_all_lifetime_work_counters() {
    let factory = faithfulness(false, JudgeLimits::default(), faithfulness_mapping(),
        faithfulness_policy(), work(), None);
    factory.check_job_limits(lifetime()).unwrap();
    for axis in 0..7 {
        let mut limits = lifetime();
        match axis {
            0 => limits.max_input_bytes_per_item = chunks().max_input_bytes - 1,
            1 => limits.max_result_bytes = faithfulness_mapping().max_result_bytes as usize - 1,
            2 => limits.max_work.model.forward_positions -= 1,
            3 => limits.max_work.model.projected_logits -= 1,
            4 => limits.max_work.model.attention_pairs -= 1,
            5 => limits.max_work.model.projections.dot_products -= 1,
            _ => limits.max_work.model.projections.multiply_accumulates -= 1,
        }
        assert!(factory.check_job_limits(limits).is_err(), "faithfulness lifetime axis {axis}");
    }
}

#[test]
fn faithfulness_factory_refuses_wrong_pinned_identity_before_storage() {
    let planner = judge(false);
    for axis in 0..5 {
        let mut changed = identity("judge-v1", *planner.template_digest(), planner.tokenizer_digest());
        match axis {
            0 => changed.template_digest = Sha256Digest::of_bytes(b"substituted template"),
            1 => changed.tokenizer_digest = Sha256Digest::of_bytes(b"substituted tokenizer"),
            2 => changed.task_spec = "classify-v1".to_owned(),
            3 => changed.backend_semantic_version.push('x'),
            _ => changed.numerics_profile = NumericsProfile::HfBf16Eager,
        }
        assert!(Int8FaithfulnessDocumentJobPlanner::new(Arc::clone(&planner), changed, budget(),
            JudgeLimits::default(), faithfulness_mapping(), faithfulness_policy(), work(), None).is_err(),
            "faithfulness identity axis {axis}");
    }
}

#[test]
fn faithfulness_factory_and_actual_compiler_survive_the_physical_join_keepalive() {
    fn send_sync<T: Send + Sync + 'static>() {}
    send_sync::<Int8FaithfulnessDocumentJobPlanner>();
    let factory = Arc::new(faithfulness(false, JudgeLimits::default(), faithfulness_mapping(),
        faithfulness_policy(), work(), None));
    let compiler = Arc::downgrade(&factory.compiler);
    let retained: Box<dyn Send> = Box::new(Arc::clone(&factory));
    drop(factory);
    assert!(compiler.upgrade().is_some());
    drop(retained);
    assert!(compiler.upgrade().is_none());
}
