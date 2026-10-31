//! Hosted ownership and refusal contracts without loading a native model.
use super::*;
use crate::{native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringMode},
    tasks::{mapreduce::ChunkLimits, sentiment::{SentimentAggregationPolicy, SentimentPolicy}}};
use std::time::Duration;

fn fixture() -> (Int4SentimentMapHostConfig, SentimentRequest) {
    let native = NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } };
    let budget = TaskBudget { max_input_tokens: 128, max_output_tokens: 16,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(native.context_tokens, 32).unwrap().kv_bytes * 4 };
    let config = Int4SentimentMapHostConfig { native, planning: SentimentLimits::default(),
        mapping: Int4SentimentMapLimits { chunks: ChunkLimits {
            max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 128,
            context_tokens: native.context_tokens, reserved_tokens: 16,
            max_chunks: 64, max_tokenizer_calls: 1024 },
            max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
            max_retained_result_bytes: 8 * 1024 * 1024, max_result_bytes: 1024 * 1024,
            aggregation: SentimentAggregationPolicy::default() },
        options: SentimentOptions { mode: ScoringMode::FullVocabulary, eos_token_id: 166101,
            policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 } },
        preparation_reserve_bytes: 128 * 1024 * 1024 };
    (config, SentimentRequest { document: "first section\r\n第二节".to_owned(),
        axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget })
}

#[test]
fn host_and_pre_io_callers_use_the_same_complete_quote() {
    let (config, request) = fixture();
    for cohort_chunks in [None, Some(1), Some(4), Some(64)] {
        let hosted = config.requirements(32, request.budget, 4, cohort_chunks).unwrap();
        let pure = Int4SentimentMapMemoryLimits { context_tokens: config.native.context_tokens,
            preparation_reserve_bytes: config.preparation_reserve_bytes,
            allocator_reserve_bytes: config.native.allocator_reserve_bytes,
            max_axes: 4, cohort_chunks }.requirements(32, request.budget,
                config.planning, config.mapping).unwrap();
        assert_eq!(hosted, pure);
        assert_eq!(hosted.output_reservation_bytes, 4 * hosted.max_output_bytes + 4096);
    }
}

#[test]
fn actual_geometry_is_never_clamped_by_the_unknown_input_quote() {
    let (config, mut request) = fixture();
    config.requirements(32, request.budget, 4, Some(64)).unwrap();
    assert!(validate_request(&request, config, Some(64)).is_err());
    validate_request(&request, config, Some(32)).unwrap();
    request.axes.truncate(1);
    validate_request(&request, config, Some(64)).unwrap();
    assert!(validate_request(&request, config, Some(0)).is_err());
    request.axes.push(request.axes[0]);
    assert!(validate_request(&request, config, None).is_err());
    request.axes.clear();
    assert!(validate_request(&request, config, None).is_err());
}

#[test]
fn input_capacity_is_charged_even_when_logical_source_is_small() {
    let (config, mut request) = fixture();
    let required = config.requirements(32, request.budget, request.axes.len(), None).unwrap();
    request.document.reserve(4 * config.mapping.chunks.max_input_bytes);
    request.axes.reserve(64);
    let source_excess = request.document.capacity() - config.mapping.chunks.max_input_bytes;
    let axes_excess = (request.axes.capacity() - 4) * std::mem::size_of::<SentimentAxis>();
    assert_eq!(owned_input_bytes(&request, config, required).unwrap(),
        required.input_bytes + source_excess as u64 + axes_excess as u64);
    assert!(request.document.len() < config.mapping.chunks.max_input_bytes);
    validate_request(&request, config, None).unwrap();
}

#[test]
fn empty_or_oversized_sources_and_invalid_policies_refuse_before_planning() {
    let (mut config, mut request) = fixture();
    request.document.clear();
    assert!(matches!(validate_request(&request, config, None),
        Err(HostedError::Int4(Int4HostError::SentimentMap(Int4SentimentMapError::EmptySource)))));
    request.document = "x".repeat(config.mapping.chunks.max_input_bytes + 1);
    assert!(validate_request(&request, config, None).is_err());
    config.options.policy.minimum_peak_weight_ppm = 1_000_001;
    assert!(config.requirements(32, request.budget, 2, None).is_err());
    config.options.policy.minimum_peak_weight_ppm = 0;
    config.options.eos_token_id = NANBEIGE_VOCAB_SIZE as u32;
    assert!(config.requirements(32, request.budget, 2, None).is_err());
}

#[test]
fn all_five_native_counters_are_whole_document_allowances() {
    let (config, _) = fixture();
    let work = config.mapping.max_model_work;
    check_work(work, work).unwrap();
    for axis in 0..5 {
        let mut cap = work;
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        assert!(matches!(check_work(work, cap),
            Err(HostedError::Int4(Int4HostError::SentimentMap(Int4SentimentMapError::Work)))), "axis {axis}");
    }
}

#[test]
fn typed_native_cancellation_survives_redacted_host_diagnostics() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::CostBudget] {
        let error = map_error(Int4SentimentMapError::Sentiment(
            Int4SentimentError::Native(StrictInt4Error::Cancelled(cause))));
        let HostedError::Int4(Int4HostError::SentimentMap(inner)) = &error else { panic!("wrong stage") };
        assert_eq!(inner.cancellation(), Some(cause));
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
    }
    fn send<T: Send>() {}
    send::<Int4SentimentMapHostConfig>();
    send::<Int4SentimentMapRun>();
    send::<Int4SentimentMapCohortRun>();
    send::<HostedOutput<Int4SentimentMapRun>>();
    send::<HostedOutput<Int4SentimentMapCohortRun>>();
}
