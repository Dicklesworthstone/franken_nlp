//! Pure ownership/admission checks without resident model allocation.
use super::*;
use std::{error::Error, time::Duration};
use crate::{native_engine::decode::DecodeCancellationKind,
    tasks::{mapreduce::ChunkLimits, judge::{FaithfulnessAggregationPolicy, FaithfulnessPolicy, PairwisePolicy}}};

fn fixture() -> (Int4FaithfulnessMapHostConfig, JudgeRequest, TaskBudget) {
    let native = NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } };
    let budget = TaskBudget { max_input_tokens: 128, max_output_tokens: 16,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(native.context_tokens, 32).unwrap().kv_bytes * 32 };
    let planning = JudgeLimits { max_output_bytes: budget.max_output_bytes, ..JudgeLimits::default() };
    let config = Int4FaithfulnessMapHostConfig { native, planning,
        mapping: Int4FaithfulnessMapLimits { chunks: ChunkLimits {
            max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 128,
            context_tokens: native.context_tokens, reserved_tokens: 16,
            max_chunks: 64, max_tokenizer_calls: 1024 },
            max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
            max_retained_result_bytes: 8 * 1024 * 1024, max_result_bytes: 1024 * 1024,
            aggregation: FaithfulnessAggregationPolicy::default() },
        preparation_reserve_bytes: 128 * 1024 * 1024 };
    let request = JudgeRequest::Faithfulness { source: "A complete source.\r\n第二节".to_owned(),
        claim: "An exact claim.".to_owned(), policy: FaithfulnessPolicy {
            minimum_candidate_weight_ppm: 600_000, minimum_margin_milli: 100,
            evidence_window_bytes: 64, max_evidence_windows: 31, max_evidence_spans: 31 }, budget };
    (config, request, budget)
}

#[test]
fn host_and_cli_can_quote_the_same_complete_owner_set_before_io() {
    let (config, _, task) = fixture();
    let limits = config.memory_limits();
    let serial = config.requirements(32, task).unwrap();
    assert_eq!(serial, limits.requirements(32, task, config.planning, config.mapping).unwrap());
    assert_eq!(serial.native.kv_bytes * 32, task.max_kv_bytes);
    for chunks in [1, 2, 3, 16, 64] {
        let hosted = config.cohort_requirements(32, task, chunks).unwrap();
        let pure = limits.cohort_requirements(32, task, config.planning, config.mapping, 32, chunks).unwrap();
        assert_eq!(hosted, pure);
        assert_eq!(hosted.memory.native.kv_bytes, serial.native.kv_bytes * (32 * chunks).min(64) as u64);
        assert!(hosted.memory.input_bytes >= serial.input_bytes + hosted.task_workspace_bytes);
        assert!(hosted.memory.output_reservation_bytes >= serial.output_reservation_bytes);
    }
    for chunks in [0, 65, usize::MAX] { assert!(config.cohort_requirements(32, task, chunks).is_err()); }
}

#[test]
fn only_faithfulness_with_complete_bounded_source_and_claim_is_admitted() {
    let (config, mut request, task) = fixture();
    assert_eq!(validate_request(&request, config).unwrap(), task);
    let JudgeRequest::Faithfulness { source, claim, .. } = &mut request else { unreachable!() };
    // The cap covers both source and claim even if each separately fits.
    *source = "s".repeat(config.mapping.chunks.max_input_bytes / 2);
    *claim = "c".repeat(config.mapping.chunks.max_input_bytes / 2 + 1);
    assert!(validate_request(&request, config).is_err());
    let JudgeRequest::Faithfulness { claim, .. } = &mut request else { unreachable!() };
    claim.pop();
    validate_request(&request, config).unwrap();
    let JudgeRequest::Faithfulness { policy, .. } = &mut request else { unreachable!() };
    policy.minimum_candidate_weight_ppm = 1_000_001;
    assert!(validate_request(&request, config).is_err());
    let JudgeRequest::Faithfulness { source, policy, .. } = &mut request else { unreachable!() };
    policy.minimum_candidate_weight_ppm = 600_000;
    source.clear();
    assert!(validate_request(&request, config).is_err());
    let unrelated = JudgeRequest::Pairwise { criterion: "test".to_owned(), a: "a".to_owned(), b: "b".to_owned(),
        policy: PairwisePolicy { minimum_margin_milli: 0, maximum_order_disagreement_milli: 0 }, budget: task };
    assert!(validate_request(&unrelated, config).is_err());
    assert!(owned_input_bytes(&unrelated, config, config.requirements(32, task).unwrap()).is_err());
}

#[test]
fn caller_retained_source_and_claim_capacity_stays_charged_after_transfer() {
    let (config, mut request, task) = fixture();
    let required = config.requirements(32, task).unwrap();
    let JudgeRequest::Faithfulness { source, claim, .. } = &mut request else { unreachable!() };
    source.reserve(3 * config.mapping.chunks.max_input_bytes);
    claim.reserve(2 * config.mapping.chunks.max_input_bytes);
    let capacity = source.capacity() + claim.capacity();
    assert_eq!(owned_input_bytes(&request, config, required).unwrap(),
        required.input_bytes + (capacity - config.mapping.chunks.max_input_bytes) as u64);
    validate_request(&request, config).unwrap();
}

#[test]
fn all_five_work_axes_remain_one_nonrenewable_document_ceiling() {
    let (config, _, _) = fixture();
    let work = config.mapping.max_model_work;
    check_work(work, work).unwrap();
    for axis in 0..5 {
        let mut cap = work;
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        assert!(matches!(check_work(work, cap),
            Err(HostedError::Int4(Int4HostError::FaithfulnessMap(Int4FaithfulnessMapError::Work)))));
    }
}

#[test]
fn physical_projection_counts_use_prefix_evaluations_instead_of_vocabulary_rows() {
    let (config, _, _) = fixture();
    let work = config.mapping.max_model_work;
    let prefixes = work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
    assert!(valid_physical_counts(work, work.forward_positions, prefixes));
    assert!(!valid_physical_counts(work, work.forward_positions + 1, prefixes));
    assert!(!valid_physical_counts(work, 1, prefixes + 1));
    assert!(!valid_physical_counts(work, 0, prefixes));
    assert!(!valid_physical_counts(work, 1, 0));
    let mut invalid = work; invalid.projected_logits += 1;
    assert!(!valid_physical_counts(invalid, 1, 1));
}

#[test]
fn typed_cancellation_and_guarded_results_cross_the_host_boundary() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::CostBudget] {
        let error = map_error(Int4FaithfulnessMapError::Judge(Int4JudgeError::Native(StrictInt4Error::Cancelled(cause))));
        let HostedError::Int4(Int4HostError::FaithfulnessMap(inner)) = &error else { panic!("wrong stage") };
        assert_eq!(inner.cancellation(), Some(cause));
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
    }
    fn send<T: Send>() {}
    send::<Int4FaithfulnessMapHostConfig>(); send::<Int4FaithfulnessMapRun>();
    send::<Int4FaithfulnessMapCohortRun>(); send::<HostedOutput<Int4FaithfulnessMapRun>>();
    send::<HostedOutput<Int4FaithfulnessMapCohortRun>>();
}

#[test]
fn hosted_compact_quotes_add_one_encoder_without_changing_native_or_output_ownership() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let (config, mut request, task) = fixture();
    let limits = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let encoding = UntrustedEncoding::SafeBpeV1(limits);
    let workspace = limits.requirements().unwrap().workspace_bytes;
    let serial = config.requirements(32, task).unwrap();
    let compact = config.requirements_with_encoding(32, task, &encoding).unwrap();
    assert_eq!(compact, serial.with_input_encoding(&encoding).unwrap());
    assert_eq!(compact.input_bytes - serial.input_bytes, workspace);
    assert_eq!(compact.native, serial.native);
    assert_eq!(compact.output_reservation_bytes, serial.output_reservation_bytes);
    assert_eq!(config.requirements_with_encoding(32, task, &UntrustedEncoding::ByteFallback).unwrap(), serial);
    if let JudgeRequest::Faithfulness { source, claim, .. } = &mut request {
        source.reserve(config.mapping.chunks.max_input_bytes * 2);
        claim.reserve(config.mapping.chunks.max_input_bytes * 2);
    }
    assert_eq!(owned_input_bytes(&request, config, compact).unwrap()
        - owned_input_bytes(&request, config, serial).unwrap(), workspace);
    for chunks in [1, 2, 8] {
        let byte = config.cohort_requirements(32, task, chunks).unwrap();
        let compact = config.cohort_requirements_with_encoding(32, task, chunks, &encoding).unwrap();
        assert_eq!(compact.memory, byte.memory.with_input_encoding(&encoding).unwrap());
        assert_eq!(compact.memory.input_bytes - byte.memory.input_bytes, workspace);
        assert_eq!(compact.task_workspace_bytes, byte.task_workspace_bytes);
        assert_eq!(compact.max_cohort_output_bytes, byte.max_cohort_output_bytes);
    }
    let mut invalid = limits; invalid.max_output_tokens = 0;
    assert!(config.requirements_with_encoding(32, task, &UntrustedEncoding::SafeBpeV1(invalid)).is_err());
    assert!(config.cohort_requirements_with_encoding(32, task, 2, &UntrustedEncoding::SafeBpeV1(invalid)).is_err());
    let _serial = NlpEngine::map_int4_faithfulness_with_encoding;
    let _cohort = NlpEngine::map_int4_faithfulness_cohorts_with_encoding;
}
