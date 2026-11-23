//! Model-free admission, identity and ownership regressions for document maps.
use super::*;
use crate::{
    native_engine::{decode::DecodeCancellationKind, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int8::Int8Work},
    tasks::{classify::{ClassificationAggregationPolicy, ClassificationPolicy},
        mapreduce::ChunkLimits},
    tokenizer::untrusted::SafeBpeLimits,
};

fn fixture() -> (ClassificationRequest, Int8ClassificationMapHostConfig) {
    let task = TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(1024).unwrap().kv_bytes };
    let request = ClassificationRequest { document: "the entire original document".to_owned(),
        labels: ["alpha", "beta"].map(|id| ClassificationLabel { id: id.to_owned(), description: String::new() }).to_vec(),
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: task };
    let config = Int8ClassificationMapHostConfig {
        native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } },
        planning: ClassificationLimits { max_input_bytes: 4096, max_labels: 4, max_total_label_bytes: 1024, ..Default::default() },
        mapping: Int8ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
            max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
            max_model_work: Int8Work::for_sequence(0, 2048, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
            max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
            aggregation: ClassificationAggregationPolicy::default() },
        preparation_reserve_bytes: MIN_PREPARATION_BYTES,
    };
    (request, config)
}

fn planner(encoding: UntrustedEncoding) -> ClassificationPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    ClassificationPlanner::pinned(controls, eos).unwrap().with_document_encoding(encoding).unwrap()
}

fn identity(planner: &ClassificationPlanner) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1, source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: Sha256Digest::of_bytes(b"unit-model"), artifact_format: "fnlpq-current-candidate-v1".to_owned(),
        quant_recipe: "int8-all".to_owned(), packing_set_digest: absent, tokenizer_digest: planner.tokenizer_digest(),
        template_digest: *planner.template_digest(), task_spec: "classify-v1".to_owned(), taskir_digest: absent,
        prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }).unwrap()
}

#[test]
fn complete_document_memory_uses_one_fully_charged_native_context() {
    let (request, mut config) = fixture();
    let expected = Int8MemoryRequirement::for_context(config.native.context_tokens).unwrap();
    let mut previous_input = 0;
    for chunks in [1, 8, 64] {
        config.mapping.chunks.max_chunks = chunks;
        let required = config.requirements(request.budget).unwrap();
        assert_eq!(required.native, expected);
        assert_eq!(required.workspace_bytes, expected.rope_bytes + expected.scratch_payload_bound + config.native.allocator_reserve_bytes);
        assert_eq!(required.max_output_bytes, config.mapping.max_result_bytes);
        assert!(required.output_reservation_bytes >= 8 * config.mapping.max_result_bytes);
        assert!(required.input_bytes > previous_input);
        previous_input = required.input_bytes;
    }
    let mut insufficient = request.budget;
    insufficient.max_kv_bytes -= 1;
    assert!(config.requirements(insufficient).is_err());
}

#[test]
fn host_quote_rejects_invalid_geometry_and_cannot_hide_overflow() {
    let (request, config) = fixture();
    for axis in 0..9 {
        let mut changed = config;
        match axis {
            0 => changed.preparation_reserve_bytes -= 1,
            1 => changed.native.allocator_reserve_bytes = 0,
            2 => changed.native.run.max_checkpoints = 1,
            3 => changed.native.context_tokens = changed.mapping.chunks.context_tokens - 1,
            4 => changed.planning.max_input_bytes = changed.mapping.chunks.max_input_bytes - 1,
            5 => changed.planning.max_context_tokens = changed.native.context_tokens - 1,
            6 => changed.planning.scoring.max_projected_logits = 0,
            7 => changed.planning.scoring.max_nodes = usize::MAX,
            _ => changed.preparation_reserve_bytes = u64::MAX,
        }
        assert!(changed.requirements(request.budget).is_err(), "axis {axis}");
    }
}

#[test]
fn whole_document_and_label_requests_refuse_without_any_model_access() {
    for field in 0..8 {
        let (mut request, config) = fixture();
        match field {
            0 => request.document.clear(),
            1 => request.document = "x".repeat(config.mapping.chunks.max_input_bytes + 1),
            2 => { let _ = request.labels.pop(); },
            3 => request.labels[1].id = request.labels[0].id.clone(),
            4 => request.labels[0].id = "\nprivate-label".to_owned(),
            5 => request.labels[0].description = "x".repeat(config.planning.max_total_label_bytes + 1),
            6 => request.policy.minimum_candidate_weight_ppm = 1_000_001,
            _ => request.policy.minimum_margin_ppm = 1_000_001,
        }
        assert!(validate_request(&request, config).is_err(), "field {field}");
    }
}

#[test]
fn compact_encoder_memory_is_charged_once_without_replacing_retained_plans() {
    let (request, config) = fixture();
    let limits = SafeBpeLimits { max_input_bytes: 262144, max_output_tokens: 131072,
        max_pair_checks: 33554432, max_pair_bytes: 268435456 };
    let encoding = UntrustedEncoding::SafeBpeV1(limits);
    let byte = config.requirements(request.budget).unwrap();
    let compact = config.requirements_with_encoding(request.budget, &encoding).unwrap();
    assert_eq!(compact.input_bytes, byte.input_bytes + limits.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, byte.native);
    assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
    assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
    assert_eq!(compact, byte.with_input_encoding(&encoding).unwrap());
    assert_eq!(config.requirements_with_encoding(request.budget, &UntrustedEncoding::ByteFallback).unwrap(), byte);
    let invalid = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_pair_checks: 0, ..limits });
    assert!(config.requirements_with_encoding(request.budget, &invalid).is_err());
    let mut overflow = byte; overflow.input_bytes = u64::MAX;
    assert!(overflow.with_input_encoding(&encoding).is_err());
}

#[test]
fn source_labels_and_identity_spare_capacities_survive_transfer_with_their_charge() {
    let (mut request, config) = fixture();
    let planner = planner(UntrustedEncoding::ByteFallback);
    let mut identity = identity(&planner);
    let required = config.requirements(request.budget).unwrap();
    let mut document = String::with_capacity(config.mapping.chunks.max_input_bytes + 1024);
    document.push_str(&request.document); request.document = document;
    let mut labels = Vec::with_capacity(config.planning.max_labels + 4);
    labels.extend(request.labels); request.labels = labels;
    let mut label = String::with_capacity(2048); label.push_str(&request.labels[0].id); request.labels[0].id = label;
    let mut description = String::with_capacity(4096); description.push('x'); request.labels[0].description = description;
    let mut recipe = String::with_capacity(BASE_IDENTITY_BYTES as usize + 1024);
    recipe.push_str(&identity.quant_recipe); identity.quant_recipe = recipe;
    validate_request(&request, config).unwrap();
    validate_base_identity(&identity).unwrap();
    let text: usize = request.labels.iter().map(|label| label.id.capacity() + label.description.capacity()).sum();
    let identity_bytes = std::mem::size_of::<ExecutionIdentity>()
        + identity_strings(&identity).iter().map(|value| value.capacity()).sum::<usize>();
    let excess = (request.document.capacity() - config.mapping.chunks.max_input_bytes) as u64
        + ((request.labels.capacity() - config.planning.max_labels) * std::mem::size_of::<ClassificationLabel>()) as u64
        + (text - config.planning.max_total_label_bytes) as u64
        + identity_bytes as u64 - BASE_IDENTITY_BYTES;
    assert_eq!(owned_input_bytes(&request, &identity, config, required).unwrap(), required.input_bytes + excess);
}

#[test]
fn exact_native_profile_and_compiler_policy_are_checked_without_repair() {
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
    let compact = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 262144,
        max_output_tokens: 131072, max_pair_checks: 33554432, max_pair_bytes: 268435456 });
    assert!(check_planner_binding(&planner, &identity, &compact).is_err());
    let compact_planner = self::planner(compact.clone());
    assert!(check_planner_binding(&compact_planner, &identity, &compact).is_err());
}

#[test]
fn all_original_cancellation_causes_keep_their_typed_host_boundary() {
    use DecodeCancellationKind::*;
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let error = checkpoint(&mut Stop(cause)).unwrap_err();
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert_eq!(error.to_string(), "hosted native classification document failed");
        let HostedError::ClassificationMap(error) = error else { panic!("typed map error required") };
        assert_eq!(error.cancellation(), Some(cause));
    }
}

#[test]
fn owned_map_inputs_and_complete_guarded_results_cross_the_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<MapInput>();
    send::<Int8ClassificationMapRun>();
    send::<HostedOutput<Int8ClassificationMapRun>>();
    let _byte = NlpEngine::map_int8_classify;
    let _compact = NlpEngine::map_int8_classify_with_encoding;
}
