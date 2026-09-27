//! Model-free resource admission and typed-error tests, not native execution.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, quantized::long::Int8SourceMapLimits},
        redact::pipeline::LeakReport}};
fn config() -> RedactConfig<LongRedactionConfig> {
    let d = Sha256Digest::of_bytes(b"document-redaction-host-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: NER_TASK_VERSION.to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let mut request = RedactionRequest::default(); request.edit_budget.max_output_bytes = 4096;
    RedactConfig { ner_identity: identity, request,
        detector: LongRedactionConfig { ner: Default::default(),
            per_chunk: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
                max_grammar_states: 4096, max_kv_bytes: 1024 },
            planning: SourcePlanningLimits { max_context_tokens: 128, ..Default::default() },
            mapping: Int8SourceMapLimits {
                chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 16, max_chunk_tokens: 16,
                    context_tokens: 128, reserved_tokens: 32, max_chunks: 3, ..Default::default() },
                reduction: ExecutionLimits { max_value_bytes: 4096, max_live_value_bytes: 8192, max_result_bytes: 4096, ..Default::default() },
                max_model_work: Int8Work::for_sequence(0, 128, 128 * 166_144).unwrap(),
                mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 6000 }, max_result_bytes: 4096 },
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 4096 } },
        preparation_reserve_bytes: 4096, edit_reserve_bytes: 4096 }
}
fn check(c: &RedactConfig<LongRedactionConfig>, bytes: usize, kv: u64) -> Result<(), HostedError> {
    validate_document(c, bytes, c.ner_identity.tokenizer_digest, c.ner_identity.template_digest, kv)
}
#[test]
fn complete_source_full_kv_and_both_context_limits_are_admitted() {
    check(&config(), 10, 1024).unwrap();
    assert!(check(&config(), 0, 1024).is_err()); assert!(check(&config(), 1025, 1024).is_err());
    assert!(check(&config(), 10, 1025).is_err());
    for axis in 0..3 {
        let mut c = config();
        match axis { 0 => c.detector.planning.max_context_tokens = 129,
            1 => c.detector.mapping.chunks.context_tokens = 129, _ => c.request.rule_budget.max_input_bytes = 9 }
        assert!(check(&c, 10, 1024).is_err());
    }
}
#[test]
fn profile_recipe_and_complete_intermediate_output_cannot_be_bypassed() {
    for axis in 0..7 {
        let mut c = config();
        match axis { 0 => c.ner_identity.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => c.ner_identity.task_spec = "summarize-v1".to_owned(),
            2 => c.detector.per_chunk.max_output_bytes = 4097, 3 => c.detector.max_result_bytes = 4095,
            4 => c.preparation_reserve_bytes = 0, 5 => c.edit_reserve_bytes = 0,
            _ => c.detector.mapping.max_mask_visits = 1999 }
        assert!(check(&c, 10, 1024).is_err(), "{axis}");
    }
    let c = config();
    assert!(validate_document(&c, 10, Sha256Digest::of_bytes(b"foreign"), c.ner_identity.template_digest, 1024).is_err());
}
#[test]
fn entire_map_frontier_is_reserved_once_at_peak_not_only_final_redacted_text() {
    let c = config();
    assert_eq!(document_temporary_bytes(&c).unwrap(), (3 * 4096 + 8192) * 4 + 3 * 16 * 8 + 4096);
    let mut c = config(); c.detector.mapping.chunks.max_chunks = 257;
    assert!(document_temporary_bytes(&c).is_err());
    let mut c = config(); c.detector.per_chunk.max_output_bytes = u64::MAX;
    assert!(document_temporary_bytes(&c).is_err());
    let mut source = String::with_capacity(8192); source.push_str("Alice");
    assert_eq!(input_bytes(&source, &config(), None).unwrap(), source.capacity() as u64 + 4096);
}
#[test]
fn residual_vectors_are_discarded_under_their_guard_and_cancellation_stays_typed() {
    let c = config();
    let error = document_error(LongRedactionError::Redaction(Int8RedactionError::Residual(LeakReport {
        schema_version: 1, residuals: Vec::new(), rules: c.request.rules, model_types: Default::default() })));
    assert!(matches!(error, HostedError::Redaction(Int8RedactionError::Redaction(RedactError::VerificationResidual { count: 0 }))));
    let cause = DecodeCancellationKind::Deadline;
    let HostedError::Redaction(error) = document_error(LongRedactionError::Redaction(Int8RedactionError::Cancelled(cause)))
        else { panic!("typed cancellation") };
    assert_eq!(error.cancellation(), Some(cause));
}
#[test]
fn complete_owned_input_is_send_without_a_second_runtime_or_borrowed_key() {
    fn send<T: Send + 'static>() {}
    send::<RedactInput<LongRedactionConfig>>();
}
