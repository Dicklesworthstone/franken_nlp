//! Host admission and typed failures, without model weights or a fake runtime.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{corpus_keyphrases::CorpusKeyphraseLimits, keyphrases::{KeyphraseOptions, KeyphraseError},
        mapreduce::{ChunkLimits, ExecutionLimits, TaskStage}},
};
fn config() -> SourceMapConfig {
    let digest = Sha256Digest::of_bytes(b"host-keyphrase-model-free-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: digest, artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(),
        packing_set_digest: digest, tokenizer_digest: digest, template_digest: digest,
        task_spec: "keyphrases-v1".to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "fixture".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    SourceMapConfig { identity, task: SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
            max_grammar_states: 4096, max_kv_bytes: 1024 },
        planning: SourcePlanningLimits { max_context_tokens: 128, ..SourcePlanningLimits::default() },
        mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 16, max_chunk_tokens: 16,
                context_tokens: 128, reserved_tokens: 32, max_chunks: 16, ..ChunkLimits::default() },
            reduction: ExecutionLimits { max_value_bytes: 4096, max_live_value_bytes: 8192,
                max_result_bytes: 4096, ..ExecutionLimits::default() },
            max_model_work: Int8Work::for_sequence(0, 128, 128 * 166_144).unwrap(),
            mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 16_000,
        },
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 4096 } },
        preparation_reserve_bytes: 4096, reduction_reserve_bytes: 4096,
    }
}
fn limits() -> Int8KeyphraseLimits {
    Int8KeyphraseLimits { aggregation: CorpusKeyphraseLimits { max_value_bytes: 4096, ..Default::default() },
        max_phrases: 1, max_result_bytes: 4096 }
}
#[test]
fn complete_ranking_admits_only_keyphrase_maps() {
    let c = config(); validate_keyphrases(&c.task, c.mapping, limits()).unwrap();
    for task in [SourceMapTask::Ner(Default::default()), SourceMapTask::Summarize(Default::default())] {
        assert!(validate_keyphrases(&task, c.mapping, limits()).is_err());
    }
}
#[test]
fn ranking_does_not_bypass_complete_source_or_resident_capacity_admission() {
    let c = config(); validate(&c, 10, 1024).unwrap();
    assert!(validate(&c, 0, 1024).is_err()); assert!(validate(&c, 1025, 1024).is_err());
    assert!(validate(&c, 10, 1025).is_err());
    let mut c = config(); c.planning.max_context_tokens = 129;
    assert!(validate(&c, 10, 1024).is_err());
    let mut c = config(); c.identity.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(validate(&c, 10, 1024).is_err());
}
#[test]
fn complete_union_and_result_caps_are_admitted_before_forward() {
    let c = config();
    for axis in 0..5 {
        let mut l = limits();
        match axis { 0 => l.max_phrases = 0, 1 => l.aggregation.max_value_bytes = 4097,
            2 => l.max_result_bytes = 4097, 3 => l.aggregation.max_evidence_spans = 0,
            _ => l.aggregation.max_scan_work = 0 }
        assert!(validate_keyphrases(&c.task, c.mapping, l).is_err());
    }
}
#[test]
fn reservation_covers_all_native_and_evidence_frontiers_not_final_top_k() {
    let c = config(); let count = 3;
    let expected = (4096 * count as u64 + 8192) * 4 + 16 * count as u64 * 8 + 4096;
    assert_eq!(reduction_bytes(&c, count).unwrap(), expected);
    assert!(reduction_bytes(&c, 0).is_err()); assert!(reduction_bytes(&c, 17).is_err());
    let mut c = config(); c.budget.max_output_bytes = u64::MAX;
    assert!(reduction_bytes(&c, 3).is_err());
}
#[test]
fn native_cancellation_and_evidence_failures_keep_their_typed_categories() {
    let cause = DecodeCancellationKind::Deadline;
    let HostedError::Source(error) = execution_error(Int8CorpusKeyphraseError::Source(Int8SourceError::Cancelled(cause)))
        else { panic!("typed source category") };
    assert_eq!(error.cancellation(), Some(cause));
    let errors = [Int8CorpusKeyphraseError::Keyphrases(KeyphraseError::InvalidResult),
        Int8CorpusKeyphraseError::Execution(ExecutionError::Task { stage: TaskStage::Map { first_chunk: 3 },
            source: KeyphraseError::InvalidResult })];
    for error in errors {
        let failure = execution_error(error);
        assert!(matches!(&failure, HostedError::Source(_)));
        assert!(!format!("{failure:?}").contains("fixture"));
    }
}
