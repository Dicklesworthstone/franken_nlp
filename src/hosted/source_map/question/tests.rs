//! Model-free host admission and charge arithmetic. Not native execution evidence.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{answer::AnswerOptions, mapreduce::{ChunkLimits, ExecutionLimits}},
    validation::grounded_fields::GroundingBudget};

fn config() -> SourceMapConfig<SourceQuestion> {
    let d = Sha256Digest::of_bytes(b"host-question-model-free-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: ANSWER_TASK_VERSION.to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    SourceMapConfig { identity,
        task: SourceQuestion { question: "Who?".to_owned(), options: AnswerOptions::default(), verification: GroundingBudget::default() },
        budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
            max_grammar_states: 4096, max_kv_bytes: 1024 },
        planning: SourcePlanningLimits { max_context_tokens: 128, ..SourcePlanningLimits::default() },
        mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 16,
                max_chunk_tokens: 16, context_tokens: 128, reserved_tokens: 32, max_chunks: 16, ..ChunkLimits::default() },
            reduction: ExecutionLimits { max_value_bytes: 4096, max_live_value_bytes: 8192,
                max_result_bytes: 4096, ..ExecutionLimits::default() },
            max_model_work: Int8Work::for_sequence(0, 128, 128 * 166_144).unwrap(),
            mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 16_000 },
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 4096 } },
        preparation_reserve_bytes: 4096, reduction_reserve_bytes: 4096 }
}
#[test]
fn question_host_keeps_task_profile_and_complete_resident_capacity_admission() {
    validate_question(&config(), 10, 1024).unwrap();
    assert!(validate_question(&config(), 0, 1024).is_err());
    assert!(validate_question(&config(), 1025, 1024).is_err());
    assert!(validate_question(&config(), 10, 1025).is_err());
    let mut c = config(); c.identity.task_spec = "ner-v1".to_owned();
    assert!(validate_question(&c, 10, 1024).is_err());
    let mut c = config(); c.identity.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(validate_question(&c, 10, 1024).is_err());
    let mut c = config(); c.planning.max_context_tokens = 129;
    assert!(validate_question(&c, 10, 1024).is_err());
}
#[test]
fn invalid_question_options_and_whole_document_evidence_limits_are_refused() {
    for axis in 0..7 {
        let mut c = config();
        match axis { 0 => c.task.question = " \r\n".to_owned(), 1 => c.task.question = "x".repeat(65),
            2 => c.task.options.max_citations = 0, 3 => c.task.verification.max_fields = 0,
            4 => c.task.verification.max_matches = 1_000_001, 5 => c.task.verification.max_scan_steps = 0,
            _ => c.planning.max_input_bytes = 3 }
        assert!(validate_question(&c, 10, 1024).is_err(), "{axis}");
    }
}
#[test]
fn original_question_capacity_is_charged_not_just_length_or_source_capacity() {
    let mut c = config(); let mut question = String::with_capacity(8192); question.push_str("Who?");
    c.task.question = question;
    assert!(c.task.question.capacity() > c.task.question.len());
    assert_eq!(input_bytes(1024, &c).unwrap(), 1024 + c.task.question.capacity() as u64 + 4096);
    c.preparation_reserve_bytes = u64::MAX;
    assert!(input_bytes(1024, &c).is_err());
}
#[test]
fn complete_native_evidence_frontier_is_priced_even_for_blank_source_ranges() {
    let c = config(); let count = 3;
    assert_eq!(reduction_bytes(&c, count).unwrap(), (4096 * count as u64 + 8192) * 4 + 16 * count as u64 * 8 + 4096);
    assert!(reduction_bytes(&c, 0).is_err()); assert!(reduction_bytes(&c, 17).is_err());
    let mut c = config(); c.budget.max_output_bytes = u64::MAX;
    assert!(reduction_bytes(&c, 3).is_err());
}
#[test]
fn host_source_map_errors_preserve_exact_cancellation_without_question_diagnostics() {
    let cause = DecodeCancellationKind::Deadline;
    let error = HostedError::SourceMap(Int8SourceMapError::Source(Int8SourceError::Cancelled(cause)));
    let HostedError::SourceMap(source) = &error else { panic!("source map error") };
    assert_eq!(source.cancellation(), Some(cause));
    let diagnostic = format!("{error:?}");
    assert!(!diagnostic.contains("Who?")); assert!(!diagnostic.contains("host-question-model-free-fixture"));
}
