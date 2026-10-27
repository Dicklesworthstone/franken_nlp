//! UNRUN admission/ownership fixtures. No native inference or quality evidence.
use super::*;
use crate::{
    execution_identity::NumericsProfile,
    native_engine::decode::DecodeCancellationKind,
    tasks::{corpus_keyphrases::CorpusKeyphraseLimits,
        source_planning::grouped_int4::Int4SourceError},
};

fn fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig, Int4KeyphraseLimits) {
    let (planner, config) = super::super::tests::fixture(SourceMapTask::Keyphrases(Default::default()), width);
    let limits = Int4KeyphraseLimits { aggregation: CorpusKeyphraseLimits {
        max_unique_phrases: 128, max_evidence_spans: 1024,
        max_value_bytes: 32_768, max_scan_work: 1 << 20 }, max_phrases: 16, max_result_bytes: 65_536 };
    (planner, config, limits)
}

#[test]
fn complete_ranking_retains_the_actual_group_pinned_assets_and_native_capacity() {
    for width in [1, 32, 127, 128] {
        let (planner, config, limits) = fixture(width);
        let need = keyphrase_requirements(&config, 16, width, &planner, limits).unwrap();
        assert_eq!(need, Int4MemoryRequirement::for_context(config.native.context_tokens, width).unwrap());
        assert_eq!(need.kv_bytes, config.budget.max_kv_bytes);
        assert!(keyphrase_requirements(&config, 16, if width == 32 { 128 } else { 32 }, &planner, limits).is_err());
    }
    for axis in 0..4 {
        let (planner, mut config, limits) = fixture(32);
        match axis { 0 => config.identity.task_spec = "ner-v1".to_owned(),
            1 => config.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => config.identity.tokenizer_digest = Sha256Digest::of_bytes(b"wrong-tokenizer"),
            _ => config.identity.template_digest = Sha256Digest::of_bytes(b"wrong-template") }
        assert!(keyphrase_requirements(&config, 16, 32, &planner, limits).is_err(), "axis {axis}");
    }
}

#[test]
fn other_native_tasks_cannot_be_relabeled_as_complete_keyphrase_ranking() {
    let (_, _, limits) = fixture(32);
    for task in [SourceMapTask::Ner(Default::default()), SourceMapTask::Summarize(Default::default())] {
        let (planner, config) = super::super::tests::fixture(task, 32);
        assert!(map_requirements(&config, 16, 32, &planner).is_ok());
        assert!(keyphrase_requirements(&config, 16, 32, &planner, limits).is_err());
    }
}

#[test]
fn a_small_final_top_k_cannot_shrink_the_full_candidate_frontier() {
    let (_, config, limits) = fixture(32);
    let all = config.keyphrase_workspace_bytes(4, limits).unwrap();
    let one = Int4KeyphraseLimits { max_phrases: 1, max_result_bytes: 1024, ..limits };
    assert_eq!(config.keyphrase_workspace_bytes(4, one).unwrap(), all);
    assert_eq!(all, config.workspace_bytes(4).unwrap()
        + 8 * limits.aggregation.max_value_bytes as u64
        + 512 * limits.aggregation.max_unique_phrases as u64
        + 2 * std::mem::size_of::<VerifiedSourceSpan>() as u64 * limits.aggregation.max_evidence_spans as u64);
}

#[test]
fn candidate_values_and_both_occurrence_copies_add_to_the_native_frontier() {
    let (_, config, limits) = fixture(32);
    let before = config.keyphrase_workspace_bytes(3, limits).unwrap();
    assert_eq!(config.keyphrase_workspace_bytes(4, limits).unwrap() - before,
        config.workspace_bytes(4).unwrap() - config.workspace_bytes(3).unwrap());
    for axis in 0..3 {
        let mut changed = limits;
        let expected = match axis {
            0 => { changed.aggregation.max_value_bytes += 1; 8 },
            1 => { changed.aggregation.max_unique_phrases += 1; 512 },
            _ => { changed.aggregation.max_evidence_spans += 1;
                2 * std::mem::size_of::<VerifiedSourceSpan>() as u64 },
        };
        assert_eq!(config.keyphrase_workspace_bytes(3, changed).unwrap() - before, expected, "axis {axis}");
    }
}

#[test]
fn every_union_scan_and_output_bound_is_admitted_before_native_execution() {
    for axis in 0..10 {
        let (planner, config, mut limits) = fixture(32);
        match axis {
            0 => limits.aggregation.max_unique_phrases = 0,
            1 => limits.aggregation.max_unique_phrases = 65_537,
            2 => limits.aggregation.max_evidence_spans = 0,
            3 => limits.aggregation.max_evidence_spans = 1_000_001,
            4 => limits.aggregation.max_value_bytes = 0,
            5 => limits.aggregation.max_value_bytes = config.mapping.reduction.max_value_bytes + 1,
            6 => limits.aggregation.max_scan_work = 0,
            7 => limits.max_phrases = 0,
            8 => limits.max_phrases = 4097,
            _ => limits.max_result_bytes = config.mapping.reduction.max_result_bytes + 1,
        }
        assert!(keyphrase_requirements(&config, 16, 32, &planner, limits).is_err(), "axis {axis}");
    }
}

#[test]
fn partition_overflow_zero_reserves_and_source_capacity_refuse_before_capture() {
    let (planner, config, limits) = fixture(32);
    for chunks in [0, 9, 257, usize::MAX] {
        assert!(config.keyphrase_workspace_bytes(chunks, limits).is_err());
    }
    for bytes in [0, config.mapping.chunks.max_input_bytes + 1] {
        assert!(keyphrase_requirements(&config, bytes, 32, &planner, limits).is_err());
    }
    for axis in 0..5 {
        let (planner, mut config, limits) = fixture(32);
        match axis { 0 => config.preparation_reserve_bytes = 0, 1 => config.reduction_reserve_bytes = 0,
            2 => config.native.allocator_reserve_bytes = 0, 3 => config.budget.max_kv_bytes -= 1,
            _ => config.mapping.chunks.max_chunks = 0 }
        assert!(keyphrase_requirements(&config, 16, 32, &planner, limits).is_err(), "axis {axis}");
    }
    let (_, mut config, limits) = fixture(32); config.budget.max_output_bytes = u64::MAX;
    assert!(config.keyphrase_workspace_bytes(2, limits).is_err());
}

#[test]
fn typed_native_cancellation_survives_redacted_host_diagnostics() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable] {
        let error = keyphrase_error(Int4CorpusKeyphraseError::Source(Int4SourceError::Cancelled(cause)));
        let HostedError::Int4(Int4HostError::SourceKeyphrases(source)) = &error else { panic!("wrong stage"); };
        assert_eq!(source.cancellation(), Some(cause));
        assert_eq!(error.to_string(), "hosted source-candidate INT4 loading or execution failed");
        assert!(!format!("{error:?}").contains("private"));
    }
}

#[test]
fn native_input_and_complete_union_remain_on_one_sendable_owned_host() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapConfig>(); send::<Charged<SourceMapInput>>();
    send::<Int4KeyphraseLimits>(); send::<Int4CorpusKeyphraseRun>();
    let _entry = NlpEngine::reduce_int4_keyphrases_document;
}
