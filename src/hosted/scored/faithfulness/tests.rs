//! Model-free ownership, admission and identity regressions. These do not
//! establish native execution parity, model accuracy or measured RSS bounds.
use super::*;
use crate::{
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork},
    tasks::{judge::{FaithfulnessAggregationPolicy, PairwisePolicy, partition_evidence},
        mapreduce::ChunkLimits},
    tokenizer::untrusted::SafeBpeLimits,
};

fn fixture() -> (JudgeRequest, Int8FaithfulnessMapHostConfig) {
    let task = TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(1024).unwrap().kv_bytes };
    let request = JudgeRequest::Faithfulness {
        source: "The original café source.\r\nTail <think>. ".repeat(4),
        claim: "The exact claim <|im_start|> remains data.".to_owned(),
        policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_milli: 100,
            evidence_window_bytes: 16, max_evidence_windows: 4, max_evidence_spans: 4 },
        budget: task,
    };
    let config = Int8FaithfulnessMapHostConfig {
        native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65_536 } },
        planning: JudgeLimits::default(),
        mapping: Int8FaithfulnessMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 64, max_chunk_tokens: 768,
                context_tokens: 1024, reserved_tokens: 16, max_chunks: 16, max_tokenizer_calls: 1024 },
            max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
                attention_pairs: u64::MAX, projections: ProjectionWork {
                    dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
            max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
            aggregation: FaithfulnessAggregationPolicy::default(),
        },
        preparation_reserve_bytes: MIN_PREPARATION_BYTES,
    };
    (request, config)
}
fn task(request: &JudgeRequest) -> TaskBudget {
    let JudgeRequest::Faithfulness { budget, .. } = request else { panic!("faithfulness fixture") };
    *budget
}
fn planner(encoding: UntrustedEncoding) -> JudgePlanner {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    JudgePlanner::pinned(controls, eos).unwrap().with_document_encoding(encoding).unwrap()
}
fn identity(planner: &JudgePlanner) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1, source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: Sha256Digest::of_bytes(b"unit-model"), artifact_format: "fnlpq-current-candidate-v1".to_owned(),
        quant_recipe: "int8-all".to_owned(), packing_set_digest: absent, tokenizer_digest: planner.tokenizer_digest(),
        template_digest: *planner.template_digest(), task_spec: "judge-v1".to_owned(), taskir_digest: absent,
        prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }).unwrap()
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn every_chunk_and_evidence_head_is_priced_but_only_one_physical_cache_is_owned() {
    let (request, mut config) = fixture();
    let expected = Int8MemoryRequirement::for_context(config.native.context_tokens).unwrap();
    let mut previous = 0;
    for chunks in [1, 8, 64] {
        config.mapping.chunks.max_chunks = chunks;
        let required = config.requirements(task(&request)).unwrap();
        assert_eq!(required.native, expected);
        assert_eq!(required.workspace_bytes, expected.rope_bytes + expected.scratch_payload_bound + config.native.allocator_reserve_bytes);
        assert_eq!(required.max_output_bytes, config.mapping.max_result_bytes);
        assert!(required.output_reservation_bytes >= 8 * config.mapping.max_result_bytes);
        assert!(required.input_bytes > previous);
        previous = required.input_bytes;
    }
    let mut insufficient = task(&request); insufficient.max_kv_bytes -= 1;
    assert!(config.requirements(insufficient).is_err());
    let mut generous = task(&request); generous.max_kv_bytes *= MAX_FAITHFULNESS_MAP_HEADS as u64;
    assert_eq!(config.requirements(generous).unwrap(), config.requirements(task(&request)).unwrap());
}

#[test]
fn quotes_refuse_invalid_geometry_and_checked_arithmetic_overflow() {
    let (request, config) = fixture();
    for axis in 0..14 {
        let mut changed = config;
        match axis {
            0 => changed.preparation_reserve_bytes -= 1,
            1 => changed.native.allocator_reserve_bytes = 0,
            2 => changed.native.run.max_checkpoints = 1,
            3 => changed.native.context_tokens = changed.mapping.chunks.context_tokens - 1,
            4 => changed.planning.max_total_prompt_tokens = 0,
            5 => changed.planning.max_total_candidate_tokens = 0,
            6 => changed.planning.max_total_projected_logits = 0,
            7 => changed.planning.max_output_bytes = 0,
            8 => changed.planning.per_head.max_projected_logits = 0,
            9 => changed.planning.per_head.max_nodes = 0,
            10 => changed.mapping.chunks.max_chunks = 257,
            11 => changed.mapping.max_retained_result_bytes = 0,
            12 => changed.mapping.max_result_bytes = 0,
            _ => changed.preparation_reserve_bytes = u64::MAX,
        }
        assert!(changed.requirements(task(&request)).is_err(), "axis {axis}");
    }
}

#[test]
fn closed_nli_language_does_not_allocate_generic_trie_limits() {
    let (request, mut config) = fixture();
    let ordinary = config.requirements(task(&request)).unwrap();
    config.planning.per_head.max_nodes = usize::MAX;
    config.planning.per_head.max_depth = usize::MAX;
    config.planning.per_head.max_candidates = usize::MAX;
    config.planning.per_head.max_total_tokens = usize::MAX;
    assert_eq!(config.requirements(task(&request)).unwrap(), ordinary);
    // A rejected next chunk can coexist with the retained plan frontier.
    config.mapping.max_model_work.forward_positions = 1;
    let limited = config.requirements(task(&request)).unwrap();
    assert!(limited.input_bytes < ordinary.input_bytes);
    let full_next_chunk = u64::from(task(&request).max_input_tokens) * MAX_FAITHFULNESS_MAP_HEADS as u64 * 64;
    assert!(limited.input_bytes >= MIN_PREPARATION_BYTES + full_next_chunk);
}

#[test]
fn whole_source_and_claim_share_the_input_boundary_without_a_one_context_partition_gate() {
    let (request, config) = fixture();
    let JudgeRequest::Faithfulness { source, policy, .. } = &request else { unreachable!() };
    assert!(partition_evidence(source, *policy).is_err());
    assert_eq!(validate_request(&request, config).unwrap(), task(&request));
    for field in 0..7 {
        let mut changed = request.clone();
        let JudgeRequest::Faithfulness { source, claim, policy, budget } = &mut changed else { unreachable!() };
        match field {
            0 => source.clear(),
            1 => claim.clear(),
            2 => *source = "x".repeat(config.mapping.chunks.max_input_bytes),
            3 => policy.minimum_candidate_weight_ppm = 1_000_001,
            4 => policy.evidence_window_bytes = 0,
            5 => policy.max_evidence_spans = policy.max_evidence_windows + 1,
            _ => budget.max_output_bytes = 0,
        }
        assert!(validate_request(&changed, config).is_err(), "field {field}");
    }
    let other = JudgeRequest::Pairwise { criterion: "quality".to_owned(), a: "a".to_owned(), b: "b".to_owned(),
        policy: PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 500 },
        budget: task(&request) };
    assert!(validate_request(&other, config).is_err());
}

#[test]
fn owned_preparation_preserves_every_chunk_and_exact_policy_head_cap() {
    let (request, config) = fixture();
    let planner = planner(UntrustedEncoding::ByteFallback);
    let input = MapInput { request, identity: identity(&planner), config, encoding: UntrustedEncoding::ByteFallback };
    let (prepared, admitted) = prepare(&planner, &input, &mut Continue).unwrap();
    assert!(prepared.chunk_count() > 1);
    assert!(prepared.head_count() > prepared.chunk_count());
    assert!(prepared.max_head_count() <= 5);
    assert_eq!(admitted.len(), prepared.chunk_count());
    assert_eq!(prepared.task_budget(), task(&input.request));
    for identity in admitted {
        check_planner_binding(&planner, &identity, &input.encoding).unwrap();
        assert_eq!(identity.logical_model_digest, input.identity.logical_model_digest);
        assert_eq!(identity.quant_recipe, input.identity.quant_recipe);
    }
    assert_eq!(policy_head_limit(FaithfulnessPolicy { max_evidence_windows: 1,
        max_evidence_spans: 1, minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0, evidence_window_bytes: 16 }), 1);
    let mut changed = MapInput { request: input.request.clone(), identity: input.identity.clone(),
        config, encoding: UntrustedEncoding::ByteFallback };
    let JudgeRequest::Faithfulness { policy, .. } = &mut changed.request else { unreachable!() };
    policy.max_evidence_windows = 1; policy.max_evidence_spans = 1;
    changed.config.mapping.chunks.max_chunks = 64;
    let (prepared, _) = prepare(&planner, &changed, &mut Continue).unwrap();
    assert_eq!(prepared.head_count(), prepared.chunk_count());
    assert_eq!(prepared.max_head_count(), 1);
}

#[test]
fn compact_workspace_is_added_once_and_never_replaces_complete_plans_or_output() {
    let (request, config) = fixture();
    let limits = SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 131_072,
        max_pair_checks: 33_554_432, max_pair_bytes: 268_435_456 };
    let encoding = UntrustedEncoding::SafeBpeV1(limits);
    let byte = config.requirements(task(&request)).unwrap();
    let compact = config.requirements_with_encoding(task(&request), &encoding).unwrap();
    assert_eq!(compact.input_bytes, byte.input_bytes + limits.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, byte.native);
    assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
    assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
    assert_eq!(compact, byte.with_input_encoding(&encoding).unwrap());
    assert_eq!(config.requirements_with_encoding(task(&request), &UntrustedEncoding::ByteFallback).unwrap(), byte);
    let invalid = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_pair_checks: 0, ..limits });
    assert!(config.requirements_with_encoding(task(&request), &invalid).is_err());
    let mut overflow = byte; overflow.input_bytes = u64::MAX;
    assert!(overflow.with_input_encoding(&encoding).is_err());
}

#[test]
fn transferred_source_claim_and_identity_spare_capacities_retain_their_full_charge() {
    let (mut request, config) = fixture();
    let planner = planner(UntrustedEncoding::ByteFallback);
    let mut identity = identity(&planner);
    let required = config.requirements(task(&request)).unwrap();
    let JudgeRequest::Faithfulness { source, claim, .. } = &mut request else { unreachable!() };
    let mut owned_source = String::with_capacity(config.mapping.chunks.max_input_bytes + 1024);
    owned_source.push_str(source); *source = owned_source;
    let mut owned_claim = String::with_capacity(config.mapping.chunks.max_input_bytes + 2048);
    owned_claim.push_str(claim); *claim = owned_claim;
    let mut recipe = String::with_capacity(BASE_IDENTITY_BYTES as usize + 1024);
    recipe.push_str(&identity.quant_recipe); identity.quant_recipe = recipe;
    let capacity = (source.capacity() + claim.capacity()) as u64;
    validate_request(&request, config).unwrap(); validate_base_identity(&identity).unwrap();
    let identity_bytes = std::mem::size_of::<ExecutionIdentity>() as u64
        + identity_strings(&identity).iter().map(|value| value.capacity() as u64).sum::<u64>();
    let excess = capacity - config.mapping.chunks.max_input_bytes as u64 + identity_bytes - BASE_IDENTITY_BYTES;
    assert_eq!(owned_input_bytes(&request, &identity, config, required).unwrap(), required.input_bytes + excess);
}

#[test]
fn exact_profile_model_template_and_encoding_claims_are_rejected_without_repair() {
    let planner = planner(UntrustedEncoding::ByteFallback);
    let identity = identity(&planner);
    check_planner_binding(&planner, &identity, &UntrustedEncoding::ByteFallback).unwrap();
    for field in 0..11 {
        let mut wrong = identity.clone();
        match field {
            0 => wrong.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            1 => wrong.backend_semantic_version.push('x'),
            2 => wrong.kv_dtype = "int8".to_owned(),
            3 => wrong.task_spec = "sentiment-v1".to_owned(),
            4 => wrong.thinking_mode = ThinkingMode::Enabled,
            5 => wrong.tool_mode = ToolMode::Json,
            6 => wrong.template_digest = Sha256Digest::of_bytes(b"foreign-template"),
            7 => wrong.tokenizer_digest = Sha256Digest::of_bytes(b"foreign-tokenizer"),
            8 => wrong.artifact_format = "x".repeat(257),
            9 => wrong.host_class = Some("ambient-host".to_owned()),
            _ => wrong.compiler_identity = Some("ambient-compiler".to_owned()),
        }
        let before = wrong.clone();
        assert!(check_planner_binding(&planner, &wrong, &UntrustedEncoding::ByteFallback).is_err(), "field {field}");
        assert_eq!(wrong, before);
    }
    let compact = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 262_144,
        max_output_tokens: 131_072, max_pair_checks: 33_554_432, max_pair_bytes: 268_435_456 });
    assert!(check_planner_binding(&planner, &identity, &compact).is_err());
    let compact_planner = self::planner(compact.clone());
    assert!(check_planner_binding(&compact_planner, &identity, &compact).is_err());
}

#[test]
fn all_native_work_axes_remain_complete_document_admission_bounds() {
    let work = Int8Work { forward_positions: 17, projected_logits: 19, attention_pairs: 23,
        projections: ProjectionWork { dot_products: 29, multiply_accumulates: 31 } };
    check_work(work, work).unwrap();
    for axis in 0..5 {
        let mut cap = work;
        match axis {
            0 => cap.forward_positions -= 1,
            1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1,
            3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1,
        }
        assert!(matches!(check_work(work, cap), Err(HostedError::FaithfulnessMap(Int8FaithfulnessMapError::Work))));
    }
}

#[test]
fn all_original_cancellation_causes_survive_hosted_and_preparation_boundaries() {
    use DecodeCancellationKind::*;
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    let (request, config) = fixture();
    let planner = planner(UntrustedEncoding::ByteFallback);
    let input = MapInput { request, identity: identity(&planner), config, encoding: UntrustedEncoding::ByteFallback };
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let error = checkpoint(&mut Stop(cause)).unwrap_err();
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert_eq!(error.to_string(), "hosted native faithfulness document failed");
        let HostedError::FaithfulnessMap(error) = error else { panic!("typed map error required") };
        assert_eq!(error.cancellation(), Some(cause));
        let error = match prepare(&planner, &input, &mut Stop(cause)) {
            Ok(_) => panic!("cancelled preparation cannot succeed"), Err(error) => error,
        };
        let HostedError::FaithfulnessMap(error) = error else { panic!("typed preparation error required") };
        assert_eq!(error.cancellation(), Some(cause));
    }
}

#[test]
fn owned_inputs_and_complete_guarded_results_cross_the_real_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<MapInput>();
    send::<Int8FaithfulnessMapRun>();
    send::<HostedOutput<Int8FaithfulnessMapRun>>();
    let _byte = NlpEngine::map_int8_faithfulness;
    let _compact = NlpEngine::map_int8_faithfulness_with_encoding;
}
