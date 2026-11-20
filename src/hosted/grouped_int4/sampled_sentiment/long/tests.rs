//! Admission and ownership fixtures. No native model or estimator-quality claim.
use super::*;
use crate::{
    native_engine::{constrained_int4::planned_work, decode::DecodeCancellationKind},
    tasks::mapreduce::ChunkLimits,
    tokenizer::untrusted::SafeBpeLimits,
};

const WIDTH: usize = 32;
const MIB: u64 = 1 << 20;

fn fixture() -> (SampledSentimentMapHostConfig, SentimentRequest) {
    let context = 2048;
    let task = TaskBudget { max_input_tokens: 1793, max_output_tokens: 256,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(context, WIDTH).unwrap().kv_bytes };
    let one = planned_work(WIDTH, task.max_input_tokens as usize, 32).unwrap();
    let work = (0..64).try_fold(Int4Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    let config = SampledSentimentMapHostConfig {
        native: NativeLimits { context_tokens: context, allocator_reserve_bytes: MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 1_000_000,
                cleanup_reserve_bytes: 65_536 } },
        planning: SampledSentimentLimits { max_input_bytes: 4096, max_result_bytes: 65_536,
            ..SampledSentimentLimits::default() },
        mapping: SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096,
            max_chunk_bytes: 512, max_chunk_tokens: 512, context_tokens: context,
            reserved_tokens: 1536, max_chunks: 8, max_tokenizer_calls: 256 },
            max_model_work: work, max_retained_result_bytes: 1 << 20, max_result_bytes: 2 << 20 },
        options: SampledSentimentOptions { effective_seed: [17; 32], samples: 2,
            temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
            max_sample_tokens: 32, max_justification_chars: 64 },
        preparation_reserve_bytes: 600 * MIB,
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        max_mask_node_visits: u64::MAX, prefill: None, reuse_prefill: false,
    };
    let request = SentimentRequest { document: "Helpful service. café 上海".to_owned(),
        axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget: task };
    (config, request)
}
fn prefill(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn document_chunks_share_one_native_sampler_and_optional_prefill_workspace() {
    let (mut config, request) = fixture();
    let plain = config.requirements(WIDTH, request.budget, 4).unwrap();
    for chunks in [1, 4, 8] { for reuse in [false, true] {
        config.mapping.chunks.max_chunks = chunks;
        config.reuse_prefill = reuse;
        config.prefill = None;
        let quote = config.requirements(WIDTH, request.budget, 4).unwrap();
        assert_eq!(quote.native, plain.native);
        assert_eq!(quote.workspace_bytes, plain.workspace_bytes);
        assert_eq!(quote.max_output_bytes, plain.max_output_bytes);
        assert!(quote.input_bytes <= plain.input_bytes);
        for rows in [1, 4, 64] {
            config.prefill = Some(prefill(rows));
            let selected = config.requirements(WIDTH, request.budget, 4).unwrap();
            assert_eq!(selected.workspace_bytes, quote.workspace_bytes + prefill(rows).max_extra_scratch_bytes);
            assert_eq!(selected.input_bytes, quote.input_bytes);
            assert_eq!(selected.native, quote.native);
            assert_eq!(selected.output_reservation_bytes, quote.output_reservation_bytes);
        }
    } }
}

#[test]
fn complete_encoder_workspace_and_output_token_frontier_are_separately_admitted() {
    let (config, request) = fixture();
    let plain = config.requirements(WIDTH, request.budget, 4).unwrap();
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096,
        max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 16 << 20 });
    let compact = config.requirements_with_encoding(WIDTH, request.budget, 4, &encoding).unwrap();
    let UntrustedEncoding::SafeBpeV1(limits) = encoding else { unreachable!() };
    assert!(compact.input_bytes >= plain.input_bytes + limits.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, plain.native);
    assert_eq!(compact.workspace_bytes, plain.workspace_bytes);
    let tokens = config.mapping.chunks.max_chunks as u64 * 4 * config.options.samples as u64
        * config.options.max_sample_tokens as u64;
    assert_eq!(plain.max_output_tokens, tokens);
    assert!(plain.output_reservation_bytes >= 4 * plain.max_output_bytes + 8 * tokens + 4096);
}

#[test]
fn invalid_masks_prefill_and_memory_cannot_reach_source_planning() {
    let (baseline, request) = fixture();
    for axis in 0..9 {
        let mut config = baseline.clone();
        let mut task = request.budget;
        match axis {
            0 => config.mask_limits.max_trie_node_visits = 0,
            1 => config.mask_limits.checkpoint_interval_nodes = 0,
            2 => config.max_mask_node_visits = 0,
            3 => config.max_sampler_bytes -= 1,
            4 => task.max_kv_bytes -= 1,
            5 => config.native.allocator_reserve_bytes = u64::MAX,
            6 => config.prefill = Some(Int4PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX }),
            7 => config.prefill = Some(Int4PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX }),
            _ => config.prefill = Some(Int4PrefillLimits { max_extra_scratch_bytes: prefill(4).max_extra_scratch_bytes - 1,
                ..prefill(4) }),
        }
        assert!(config.requirements(WIDTH, task, 4).is_err(), "axis {axis}");
    }
    for axes in [0, 5, usize::MAX] {
        assert!(baseline.requirements(WIDTH, request.budget, axes).is_err());
    }
}

#[test]
fn retained_request_capacity_is_charged_without_copying_or_truncating_the_document() {
    let (config, mut request) = fixture();
    let mut text = String::with_capacity(16_384); text.push_str(&request.document);
    request.document = text;
    let mut axes = Vec::with_capacity(64); axes.extend_from_slice(&request.axes); request.axes = axes;
    let quote = config.requirements(WIDTH, request.budget, request.axes.len()).unwrap();
    assert_eq!(owned_input_bytes(&request, &config, quote.input_bytes).unwrap(), quote.input_bytes
        + request.document.capacity().saturating_sub(config.mapping.chunks.max_input_bytes) as u64
        + (request.axes.capacity() - SentimentAxis::ALL.len()) as u64 * std::mem::size_of::<SentimentAxis>() as u64);
    assert!(owned_input_bytes(&request, &config, u64::MAX).is_err());
    assert!(validate_request(&request, &config).is_ok());
    request.axes.push(request.axes[0]);
    assert!(validate_request(&request, &config).is_err());
    request.axes.pop(); request.document.clear();
    assert!(validate_request(&request, &config).is_err());
}

#[test]
fn one_complete_mask_sum_is_checked_before_any_native_allocation() {
    use crate::native_engine::strict_int4::weights::projection_fixture;
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let (weights, _) = projection_fixture(WIDTH);
    let (mut config, request) = fixture();
    config.mapping.chunks.max_chunk_bytes = 8;
    let quote = config.requirements(WIDTH, request.budget, request.axes.len()).unwrap();
    let input = MapInput { request, config, encoding: UntrustedEncoding::ByteFallback };
    let planner = SampledSentimentPlanner::for_model(weights.identity(), input.config.options.clone(),
        input.request.budget, input.config.planning).unwrap();
    let prepared = planner.plan_document_with_control(&input.request, input.config.mapping, &mut Continue).unwrap();
    assert!(prepared.chunk_count() > 1);
    validate_prepared(&prepared, &input, quote, weights.identity()).unwrap();
    let vocabulary = SampledSentimentVocabulary::pinned().unwrap();
    let budget = document_budget(&prepared, &input.config, quote.native.kv_bytes);
    let charge = prepared.complete_mask_charge(&vocabulary, budget, input.config.max_sampler_bytes).unwrap();
    assert!(charge >= prepared.sample_count() as u64 * input.config.options.max_sample_tokens as u64);
    let mut short = budget; short.json.max_total_mask_node_visits = charge - 1;
    assert!(prepared.complete_mask_charge(&vocabulary, short, input.config.max_sampler_bytes).is_err());
    let mut exact = budget; exact.json.max_total_mask_node_visits = charge;
    assert_eq!(prepared.complete_mask_charge(&vocabulary, exact, input.config.max_sampler_bytes).unwrap(), charge);
    let mut wrong = input;
    wrong.config.mapping.max_model_work = prepared.planned_work();
    wrong.config.mapping.max_model_work.projections.multiply_accumulates -= 1;
    assert!(validate_prepared(&prepared, &wrong, quote, weights.identity()).is_err());
}

#[test]
fn owned_document_state_and_public_schedules_fit_the_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<MapInput>(); send::<MapExecution>(); send::<MapCompletion>();
    send::<HostedOutput<SampledSentimentMapRun>>();
    send::<HostedInt4TeamResult<SampledSentimentMapRun>>();
    let _serial = NlpEngine::map_int4_sampled_sentiment;
    let _compact = NlpEngine::map_int4_sampled_sentiment_with_encoding;
    let _team = NlpEngine::map_int4_sampled_sentiment_with_threads;
    let _compact_team = NlpEngine::map_int4_sampled_sentiment_with_encoding_and_threads;
    let (config, _) = fixture();
    let shown = format!("{config:?}");
    assert!(shown.contains("REDACTED"));
    assert!(!shown.contains("[17, 17"));
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights test cancellation/admission/drain, never model inference"]
fn every_map_schedule_preserves_terminal_cancellation_and_releases_owned_claims() {
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        native_engine::strict_int4::weights::projection_fixture};
    assert!(crate::installed_process_resources().is_none(), "requires a fresh process");
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16 * 1024 * MIB,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, MIB).unwrap(),
        || Ok(projection_fixture(WIDTH).0)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let baseline = host.resources().memory_snapshot();
    use DecodeCancellationKind::*;
    for (rows, reuse, threads) in [(None, false, None), (None, true, None),
        (Some(4), false, None), (Some(4), true, None), (None, false, Some(4)), (None, true, Some(4))] {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
            RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let (mut config, request) = fixture();
            config.prefill = rows.map(prefill); config.reuse_prefill = reuse;
            let stop = CancellationToken::default(); stop.cancel(cause);
            let error = match threads {
                None => host.map_int4_sampled_sentiment(&model, request, config,
                    PackedInt4Kernel::scalar(), stop).err().unwrap(),
                Some(width) => host.map_int4_sampled_sentiment_with_threads(&model, request, config,
                    PackedInt4Kernel::scalar(), width, stop).err().unwrap(),
            };
            let HostedError::Stopped { stop, .. } = error else {
                panic!("complete sampled map lost the original terminal cause");
            };
            assert_eq!(stop.kind, cause);
            assert_eq!(host.resources().memory_snapshot(), baseline);
            let closures = host.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0);
            assert_eq!(closures.wrapper_cancelled_closures, 0);
        }
    }
    let (mut config, request) = fixture(); config.prefill = Some(prefill(4));
    let error = host.map_int4_sampled_sentiment_with_threads(&model, request, config,
        PackedInt4Kernel::scalar(), 4, CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Limits(_)));
    assert_eq!(host.resources().memory_snapshot(), baseline);
    drop(model);
    assert_eq!(host.resources().memory_snapshot(), idle);
}
