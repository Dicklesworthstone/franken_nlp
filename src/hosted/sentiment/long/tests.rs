//! Hosted ownership/refusal contracts without loading a native model.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{
        decode::DecodeCancellationKind, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int8::STRICT_INT8_EXECUTION,
    },
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy},
    tokenizer::untrusted::SafeBpeLimits,
};

fn fixture() -> (Int8SentimentMapHostConfig, SentimentRequest) {
    let digest = Sha256Digest::of_bytes(b"hosted-int8-sentiment-map-fixture");
    let native = NativeLimits {
        context_tokens: 256,
        allocator_reserve_bytes: 65_536,
        run: RunLimits {
            max_elapsed: Duration::from_secs(30),
            max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536,
        },
    };
    let budget = TaskBudget {
        max_input_tokens: 128,
        max_output_tokens: 16,
        max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(native.context_tokens)
            .unwrap()
            .kv_bytes
            * 4,
    };
    let identity = ExecutionIdentity {
        schema_version: 1,
        source_revision: "fixture".to_owned(),
        logical_model_digest: digest,
        artifact_format: "fixture-only".to_owned(),
        quant_recipe: "fixture-int8".to_owned(),
        packing_set_digest: digest,
        tokenizer_digest: digest,
        template_digest: digest,
        task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest,
        prompt_digest: digest,
        grammar_compiler_version: "none".to_owned(),
        schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 },
        kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None,
        calibration_digest: digest,
        decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None,
        compiler_identity: None,
    };
    let config = Int8SentimentMapHostConfig {
        identity,
        native,
        planning: SentimentLimits::default(),
        mapping: Int8SentimentMapLimits {
            chunks: ChunkLimits {
                max_input_bytes: 4096,
                max_chunk_bytes: 512,
                max_chunk_tokens: 128,
                context_tokens: native.context_tokens,
                reserved_tokens: 16,
                max_chunks: 64,
                max_tokenizer_calls: 1024,
            },
            max_model_work: Int8Work::for_sequence(0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
            max_retained_result_bytes: 8 * 1024 * 1024,
            max_result_bytes: 1024 * 1024,
            aggregation: SentimentAggregationPolicy::default(),
        },
        preparation_reserve_bytes: 128 * 1024 * 1024,
    };
    let request = SentimentRequest {
        document: "first section\r\n第二节🙂".to_owned(),
        axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal],
        budget,
    };
    (config, request)
}

#[test]
fn host_and_pre_io_callers_share_the_complete_declared_quote() {
    let (config, request) = fixture();
    for max_axes in 1..=4 {
        let hosted = config
            .requirements(request.budget, max_axes, &UntrustedEncoding::ByteFallback)
            .unwrap();
        let pure = Int8SentimentMapMemoryLimits {
            context_tokens: config.native.context_tokens,
            preparation_reserve_bytes: config.preparation_reserve_bytes,
            allocator_reserve_bytes: config.native.allocator_reserve_bytes,
            max_axes,
        }
        .requirements(request.budget, config.planning, config.mapping)
        .unwrap();
        assert_eq!(hosted, pure);
        assert_eq!(
            hosted.output_reservation_bytes,
            4 * hosted.max_output_bytes + 4096
        );
    }
}

#[test]
fn nonempty_unique_axis_selection_is_required_without_silent_repair() {
    let (config, mut request) = fixture();
    validate_request(&request, &config).unwrap();
    request.axes = SentimentAxis::ALL.to_vec();
    validate_request(&request, &config).unwrap();
    request.axes.push(SentimentAxis::Valence);
    assert!(validate_request(&request, &config).is_err());
    request.axes = vec![SentimentAxis::Valence, SentimentAxis::Valence];
    assert!(validate_request(&request, &config).is_err());
    assert_eq!(
        request.axes.len(),
        2,
        "validation must not deduplicate the caller request"
    );
    request.axes.clear();
    assert!(validate_request(&request, &config).is_err());
}

#[test]
fn source_limit_counts_original_utf8_bytes_and_rejects_empty_input() {
    let (mut config, mut request) = fixture();
    let original = request.document.clone();
    assert!(request.document.chars().count() < request.document.len());
    config.mapping.chunks.max_input_bytes = request.document.len();
    validate_request(&request, &config).unwrap();
    config.mapping.chunks.max_input_bytes -= 1;
    assert!(matches!(
        validate_request(&request, &config),
        Err(HostedError::SentimentMap(Int8SentimentMapError::Limits))
    ));
    assert_eq!(request.document, original);
    request.document.clear();
    assert!(matches!(
        validate_request(&request, &config),
        Err(HostedError::SentimentMap(
            Int8SentimentMapError::EmptySource
        ))
    ));
}

#[test]
fn owned_input_prices_capacity_above_declared_source_and_axis_bounds() {
    let (config, mut request) = fixture();
    let required = config
        .requirements(
            request.budget,
            request.axes.len(),
            &UntrustedEncoding::ByteFallback,
        )
        .unwrap();
    let baseline = owned_input_bytes(&request, &config, required).unwrap();
    assert_eq!(baseline, required.input_bytes);
    request
        .document
        .reserve(4 * config.mapping.chunks.max_input_bytes);
    request.axes.reserve(64);
    let source_capacity = request.document.capacity();
    let axes_capacity = request.axes.capacity();
    let source_excess = source_capacity - config.mapping.chunks.max_input_bytes;
    let axes_excess = (axes_capacity - 4) * std::mem::size_of::<SentimentAxis>();
    assert_eq!(
        owned_input_bytes(&request, &config, required).unwrap(),
        required.input_bytes + source_excess as u64 + axes_excess as u64
    );
    assert!(request.document.len() < config.mapping.chunks.max_input_bytes);
    validate_request(&request, &config).unwrap();
    assert_eq!(request.document.capacity(), source_capacity);
    assert_eq!(request.axes.capacity(), axes_capacity);
    let mut overflowing = required;
    overflowing.input_bytes = u64::MAX;
    assert!(owned_input_bytes(&request, &config, overflowing).is_err());
}

#[test]
fn every_run_limit_and_nonpayload_memory_floor_is_checked_before_dispatch() {
    for axis in 0..7 {
        let (mut config, request) = fixture();
        match axis {
            0 => config.native.run.max_elapsed = Duration::ZERO,
            1 => config.native.run.max_checkpoints = 1,
            2 => config.native.run.cleanup_reserve_bytes = 0,
            3 => config.preparation_reserve_bytes = 128 * 1024 * 1024 - 1,
            4 => config.native.allocator_reserve_bytes = 0,
            5 => config.native.context_tokens = config.mapping.chunks.context_tokens - 1,
            _ => config.preparation_reserve_bytes = u64::MAX,
        }
        assert!(
            config
                .requirements(request.budget, 4, &UntrustedEncoding::ByteFallback)
                .is_err(),
            "axis {axis}"
        );
    }
    let (config, mut request) = fixture();
    config
        .requirements(request.budget, 4, &UntrustedEncoding::ByteFallback)
        .unwrap();
    request.budget.max_kv_bytes -= 1;
    assert!(matches!(
        config.requirements(request.budget, 4, &UntrustedEncoding::ByteFallback),
        Err(HostedError::SentimentMap(Int8SentimentMapError::Admission))
    ));
}

#[test]
fn packed_prefill_adds_exact_reusable_scratch_without_discounting_ordinary_memory() {
    let (config, request) = fixture();
    let required = config
        .requirements(request.budget, 4, &UntrustedEncoding::ByteFallback)
        .unwrap();
    let scratch = crate::hosted::scored::scoring_scratch;
    assert_eq!(
        scratch(required.workspace_bytes, None).unwrap(),
        required.workspace_bytes
    );
    for rows in [1, 4, 64] {
        let extra = Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
        let prefill = Int8PrefillLimits {
            max_batch_rows: rows,
            max_extra_scratch_bytes: extra * 2,
        };
        assert_eq!(
            scratch(required.workspace_bytes, Some(prefill)).unwrap(),
            required.workspace_bytes + extra
        );
        assert!(
            scratch(
                required.workspace_bytes,
                Some(Int8PrefillLimits {
                    max_extra_scratch_bytes: extra - 1,
                    ..prefill
                })
            )
            .is_err()
        );
        assert!(scratch(u64::MAX, Some(prefill)).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        assert!(
            scratch(
                required.workspace_bytes,
                Some(Int8PrefillLimits {
                    max_batch_rows: rows,
                    max_extra_scratch_bytes: u64::MAX
                })
            )
            .is_err()
        );
    }
}

#[test]
fn compact_host_admission_adds_one_encoder_workspace_for_any_axis_selection() {
    let (config, request) = fixture();
    let policy = SafeBpeLimits {
        max_input_bytes: 8192,
        max_output_tokens: 4096,
        max_pair_checks: 100_000,
        max_pair_bytes: 1_000_000,
    };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    for max_axes in 1..=4 {
        let byte = config
            .requirements(request.budget, max_axes, &UntrustedEncoding::ByteFallback)
            .unwrap();
        let compact = config
            .requirements(request.budget, max_axes, &encoding)
            .unwrap();
        assert_eq!(compact, byte.with_input_encoding(&encoding).unwrap());
        assert_eq!(
            compact.input_bytes - byte.input_bytes,
            policy.requirements().unwrap().workspace_bytes
        );
        assert_eq!(compact.native, byte.native);
        assert_eq!(
            compact.output_reservation_bytes,
            byte.output_reservation_bytes
        );
    }
    let invalid = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_pair_checks: 0,
        ..policy
    });
    assert!(config.requirements(request.budget, 4, &invalid).is_err());
}

#[test]
fn every_cancellation_cause_survives_the_private_host_boundary() {
    struct Stop(DecodeCancellationKind, usize);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            panic!("the hosted document boundary must use the prefill hook");
        }
        fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
            assert_eq!(position, 0);
            self.1 += 1;
            Some(self.0)
        }
    }
    for cause in [
        DecodeCancellationKind::User,
        DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget,
        DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost,
        DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit,
    ] {
        let mut control = Stop(cause, 0);
        let error = checkpoint(&mut control).unwrap_err();
        assert_eq!(control.1, 1);
        let HostedError::SentimentMap(inner) = &error else {
            panic!("wrong task stage")
        };
        assert_eq!(inner.cancellation(), Some(cause));
        assert_eq!(format!("{error:?}"), error.to_string());
        assert!(error.source().is_some());
    }
}

#[test]
fn retained_input_and_guarded_output_can_cross_the_owned_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<MapInput>();
    send::<Int8SentimentMapHostConfig>();
    send::<Int8SentimentMapRun>();
    send::<HostedOutput<Int8SentimentMapRun>>();
    let _serial = NlpEngine::map_int8_sentiment;
    let _prefill = NlpEngine::map_int8_sentiment_layer_major;
}
