//! Model-free admission checks; native parity remains an explicitly armed gate.
use super::*;
use crate::{
    native_engine::{constrained_int8::planned_work, decode::DecodeCancellationKind},
    tasks::mapreduce::ChunkLimits,
    tokenizer::untrusted::SafeBpeLimits,
};

const MIB: u64 = 1 << 20;

fn fixture() -> (Int8SampledSentimentMapHostConfig, SentimentRequest) {
    let context = 2048;
    let task = TaskBudget { max_input_tokens: 1793, max_output_tokens: 256,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(context).unwrap().kv_bytes };
    let one = planned_work(task.max_input_tokens as usize, 32).unwrap();
    let work = (0..64).try_fold(Int8Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    let config = Int8SampledSentimentMapHostConfig {
        native: NativeLimits { context_tokens: context, allocator_reserve_bytes: MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 1_000_000,
                cleanup_reserve_bytes: 65_536 } },
        planning: SampledSentimentLimits { max_input_bytes: 4096, max_result_bytes: 65_536,
            ..SampledSentimentLimits::default() },
        mapping: Int8SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096,
            max_chunk_bytes: 512, max_chunk_tokens: 512, context_tokens: context,
            reserved_tokens: 1536, max_chunks: 8, max_tokenizer_calls: 256 },
            max_model_work: work, max_retained_result_bytes: MIB, max_result_bytes: 2 * MIB },
        options: SampledSentimentOptions { effective_seed: [17; 32], samples: 2,
            temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
            max_sample_tokens: 32, max_justification_chars: 64 },
        preparation_reserve_bytes: 600 * MIB,
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        max_mask_node_visits: u64::MAX, prefill: None, selected_rows: None,
    };
    let request = SentimentRequest { document: "Helpful service. café 上海".to_owned(),
        axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget: task };
    (config, request)
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}

#[test]
fn chunks_share_native_sampler_capacity_and_add_only_one_prompt_workspace() {
    let (mut config, request) = fixture();
    let base = config.requirements(request.budget, 4).unwrap();
    for chunks in [1, 4, 8] {
        config.mapping.chunks.max_chunks = chunks;
        config.prefill = None;
        let quote = config.requirements(request.budget, 4).unwrap();
        assert_eq!(quote.native, base.native);
        assert_eq!(quote.workspace_bytes, base.workspace_bytes);
        assert!(quote.input_bytes <= base.input_bytes);
        for rows in [1, 4, 64] {
            config.prefill = Some(prefill(rows));
            let scheduled = config.requirements(request.budget, 4).unwrap();
            assert_eq!(scheduled.workspace_bytes, quote.workspace_bytes + prefill(rows).max_extra_scratch_bytes);
            assert_eq!(scheduled.input_bytes, quote.input_bytes);
            assert_eq!(scheduled.native, quote.native);
            assert_eq!(scheduled.output_reservation_bytes, quote.output_reservation_bytes);
            for width in [2, 4, 64] {
                assert!(dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap() > 0);
            }
        }
    }
}

#[test]
fn one_encoder_and_complete_output_token_frontier_are_priced_before_input() {
    let (config, request) = fixture();
    let byte = config.requirements(request.budget, 4).unwrap();
    let limits = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 16 << 20 };
    let compact = config.requirements_with_encoding(request.budget, 4,
        &UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    assert!(compact.input_bytes >= byte.input_bytes + limits.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, byte.native);
    assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
    let tokens = config.mapping.chunks.max_chunks as u64 * 4 * config.options.samples as u64
        * config.options.max_sample_tokens as u64;
    assert_eq!(byte.max_output_tokens, tokens);
    assert!(byte.output_reservation_bytes >= 4 * byte.max_output_bytes + 8 * tokens + 4096);
}

#[test]
fn invalid_masks_prefill_and_memory_refuse_before_source_planning() {
    let (base, request) = fixture();
    for axis in 0..10 {
        let mut config = base.clone();
        let mut task = request.budget;
        match axis {
            0 => config.mask_limits.max_trie_node_visits = 0,
            1 => config.mask_limits.checkpoint_interval_nodes = 0,
            2 => config.max_mask_node_visits = 0,
            3 => config.max_sampler_bytes -= 1,
            4 => task.max_kv_bytes -= 1,
            5 => config.native.allocator_reserve_bytes = u64::MAX,
            6 => config.prefill = Some(Int8PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX }),
            7 => config.prefill = Some(Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX }),
            8 => config.prefill = Some(Int8PrefillLimits { max_extra_scratch_bytes: prefill(4).max_extra_scratch_bytes - 1,
                ..prefill(4) }),
            _ => config.preparation_reserve_bytes = 1,
        }
        assert!(config.requirements(task, 4).is_err(), "axis {axis}");
    }
    for axes in [0, 5, usize::MAX] { assert!(base.requirements(request.budget, axes).is_err()); }
}

#[test]
fn transferred_source_and_axis_capacity_are_retained_in_admission() {
    let (config, mut request) = fixture();
    let mut text = String::with_capacity(16_384); text.push_str(&request.document);
    request.document = text;
    let mut axes = Vec::with_capacity(64); axes.extend_from_slice(&request.axes); request.axes = axes;
    let quote = config.requirements(request.budget, request.axes.len()).unwrap();
    assert_eq!(owned_input_bytes(&request, &config, quote.input_bytes).unwrap(), quote.input_bytes
        + request.document.capacity().saturating_sub(config.mapping.chunks.max_input_bytes) as u64
        + (request.axes.capacity() - SentimentAxis::ALL.len()) as u64 * std::mem::size_of::<SentimentAxis>() as u64);
    assert!(owned_input_bytes(&request, &config, u64::MAX).is_err());
    assert!(validate_request(&request, &config).is_ok());
    request.axes.push(request.axes[0]); assert!(validate_request(&request, &config).is_err());
    request.axes.pop(); request.document.clear(); assert!(validate_request(&request, &config).is_err());
}

#[test]
fn every_chunk_mask_and_exact_model_binding_precede_native_work() {
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let (mut config, request) = fixture();
    config.mapping.chunks.max_chunk_bytes = 8;
    let required = config.requirements(request.budget, request.axes.len()).unwrap();
    let input = MapInput { request, config, encoding: UntrustedEncoding::ByteFallback };
    let planner = Int8SampledSentimentPlanner::for_model(&facts(), input.config.options.clone(),
        input.request.budget, input.config.planning).unwrap();
    let prepared = planner.plan_document_with_control(&input.request, input.config.mapping, &mut Continue).unwrap();
    assert!(prepared.chunk_count() > 1);
    validate_prepared(&prepared, &input, required, &facts()).unwrap();
    let mut wrong_model = facts(); wrong_model.logical_model_sha256 = "ef".repeat(32);
    assert!(validate_prepared(&prepared, &input, required, &wrong_model).is_err());
    let vocabulary = SampledSentimentVocabulary::pinned().unwrap();
    let budget = document_budget(&prepared, &input.config, required.native.kv_bytes);
    let charge = prepared.complete_mask_charge(&vocabulary, budget, input.config.max_sampler_bytes).unwrap();
    assert!(charge >= prepared.sample_count() as u64 * input.config.options.max_sample_tokens as u64);
    let mut short = budget; short.json.max_total_mask_node_visits = charge - 1;
    assert!(prepared.complete_mask_charge(&vocabulary, short, input.config.max_sampler_bytes).is_err());
    let mut exact = budget; exact.json.max_total_mask_node_visits = charge;
    assert_eq!(prepared.complete_mask_charge(&vocabulary, exact, input.config.max_sampler_bytes).unwrap(), charge);
    let mut wrong_input = input;
    wrong_input.config.mapping.max_model_work = prepared.planned_work();
    wrong_input.config.mapping.max_model_work.projections.multiply_accumulates -= 1;
    assert!(validate_prepared(&prepared, &wrong_input, required, &facts()).is_err());
}

#[test]
fn cancellation_keeps_all_terminal_causes_and_private_diagnostics() {
    use DecodeCancellationKind::*;
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = checkpoint(&mut Stop(cause)).unwrap_err();
        assert_eq!(error.to_string(), "hosted native sampled sentiment document failed");
        assert_eq!(format!("{error:?}"), error.to_string());
        assert!(error.source().is_some());
        let HostedError::SampledSentimentMap(inner) = error else { unreachable!() };
        assert_eq!(inner.cancellation(), Some(cause));
    }
}

#[test]
fn document_owners_and_guarded_results_cross_the_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<MapAdmission>(); send::<Int8SampledSentimentMapRun>();
    send::<HostedInt8TeamResult<Int8SampledSentimentMapRun>>();
    let _serial = NlpEngine::map_int8_sampled_sentiment;
    let _compact = NlpEngine::map_int8_sampled_sentiment_with_encoding;
    let _team = NlpEngine::map_int8_sampled_sentiment_with_threads;
    let _compact_team = NlpEngine::map_int8_sampled_sentiment_with_encoding_and_threads;
}

#[test]
fn selected_document_plans_bind_the_cap_without_discounting_native_or_mask_storage() {
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let (mut config, request) = fixture();
    let dense_memory = config.requirements(request.budget, 4).unwrap();
    let planner = || Int8SampledSentimentPlanner::for_model(&facts(), config.options.clone(),
        request.budget, config.planning).unwrap();
    let dense = planner().plan_document_with_control(&request, config.mapping, &mut Continue).unwrap();
    let rows = Int8JsonSparseLimits { max_rows_per_step: 4096 };
    let selected = planner().with_selected_rows(rows).unwrap()
        .plan_document_with_control(&request, config.mapping, &mut Continue).unwrap();
    config.selected_rows = Some(rows);
    let selected_memory = config.requirements(request.budget, 4).unwrap();
    assert_eq!(selected_memory, dense_memory);
    assert_eq!(selected.planned_work().forward_positions, dense.planned_work().forward_positions);
    assert_eq!(selected.planned_work().attention_pairs, dense.planned_work().attention_pairs);
    assert!(selected.planned_work().projected_logits < dense.planned_work().projected_logits);
    let vocabulary = SampledSentimentVocabulary::pinned().unwrap();
    let mask = |plan: &PreparedInt8SampledSentimentMap|
        plan.complete_mask_charge(&vocabulary, document_budget(plan, &config, dense_memory.native.kv_bytes),
            config.max_sampler_bytes).unwrap();
    assert_eq!(mask(&selected), mask(&dense));
    let mut input = MapInput { request, config, encoding: UntrustedEncoding::ByteFallback };
    validate_prepared(&selected, &input, selected_memory, &facts()).unwrap();
    assert!(validate_prepared(&dense, &input, selected_memory, &facts()).is_err());
    input.config.selected_rows = Some(Int8JsonSparseLimits { max_rows_per_step: 8192 });
    assert!(validate_prepared(&selected, &input, selected_memory, &facts()).is_err());
    for cap in [0, 166145] {
        input.config.selected_rows = Some(Int8JsonSparseLimits { max_rows_per_step: cap });
        assert!(input.config.requirements(input.request.budget, 4).is_err());
    }
}

mod model;
