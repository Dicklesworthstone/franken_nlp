//! Ledger arithmetic, key continuity and owned public API; no model inference.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int8, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::TaskBudget, ner::NerOptions, source_planning::SourcePlanningLimits,
        redact::{PiiKind, actions::RedactionAction, pseudonym::PseudonymKey}},
};
fn config() -> RedactionCorpusConfig {
    let d = Sha256Digest::of_bytes(b"hosted-redaction-corpus-fixture");
    let id = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "ner-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let work = constrained_int8::planned_work(128, 16).unwrap();
    RedactionCorpusConfig { edit_reserve_bytes: 4096, batch: Int8RedactionBatchConfig {
        ner_identity: id, detector: crate::tasks::redact::quantized::Int8RedactionConfig {
            ner: NerOptions::default(), per_pass: TaskBudget { max_input_tokens: 8192, max_output_tokens: 16,
                max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
            planning: SourcePlanningLimits::default(), max_model_work: work.checked_add(work).unwrap(),
            mask_limits: MaskWorkLimits::default(), mask_visits_per_pass: 1000, max_mask_visits: 2000,
            max_result_bytes: 4 << 20 }, request: RedactionRequest::default(),
        max_model_work: work.checked_add(work).unwrap().checked_add(work).unwrap(), max_mask_visits: 4000 } }
}
fn limits() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 4096,
        run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 100000, cleanup_reserve_bytes: 4096 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 8192, io_reserve_bytes: 4096 }
}
fn secret(byte: u8, namespace: &str) -> RedactionPseudonyms {
    RedactionPseudonyms { key: Arc::new(PseudonymKey::from_bytes(&[byte; 32], "same-public-id").unwrap()),
        namespace: namespace.to_owned() }
}
#[test]
fn native_context_and_complete_output_must_fit_hosted_reservations() {
    validate_limits(&config(), limits(), 4096).unwrap();
    for axis in 0..5 {
        let mut c = config(); let mut kv = 4096;
        match axis { 0 => c.edit_reserve_bytes = 0, 1 => kv = 0,
            2 => kv = c.batch.detector.per_pass.max_kv_bytes + 1,
            3 => c.batch.detector.planning.max_context_tokens = limits().native.context_tokens + 1,
            _ => c.batch.detector.max_result_bytes = limits().transport.max_output_line_bytes as u64 + 1 }
        assert!(validate_limits(&c, limits(), kv).is_err());
    }
}
#[test]
fn temporary_commitment_prices_ner_output_tokens_and_edit_heap() {
    let c = config();
    assert_eq!(temporary_bytes(&c).unwrap(), 4 * 65536 + 8 * 16 + 4096);
    let total = sum(&[limits().reservation_bytes().unwrap(), temporary_bytes(&c).unwrap(), secret_bytes(None).unwrap()]).unwrap();
    assert!(total > temporary_bytes(&c).unwrap());
}
#[test]
fn intermediate_and_total_memory_arithmetic_cannot_wrap() {
    let mut c = config(); c.batch.detector.per_pass.max_output_bytes = u64::MAX;
    assert!(temporary_bytes(&c).is_err());
    c.batch.detector.per_pass.max_output_bytes = 1; c.edit_reserve_bytes = u64::MAX;
    assert!(temporary_bytes(&c).is_err()); assert!(sum(&[u64::MAX, 1]).is_err());
}
#[test]
fn namespace_charge_uses_retained_capacity_not_length() {
    let mut s = secret(7, "corpus"); s.namespace.reserve(4096);
    assert_eq!(secret_bytes(Some(&s)).unwrap(), s.namespace.capacity() as u64 + 2048);
    assert!(secret_bytes(Some(&secret(7, ""))).is_err());
    assert!(secret_bytes(Some(&secret(7, &"x".repeat(257)))).is_err());
}
#[test]
fn missing_or_wrong_key_refuses_before_any_native_or_stream_io() {
    let s = secret(7, "corpus"); let other = secret(8, "corpus");
    let mut request = RedactionRequest::default(); request.actions.default_action = RedactionAction::Pseudonymize;
    assert!(pseudonym_context(None, &request).is_err());
    request.actions.expected_key_commitment = Some(s.key.commitment());
    assert!(pseudonym_context(Some(&other), &request).is_err());
    assert!(pseudonym_context(Some(&s), &request).is_ok());
}
#[test]
fn one_full_digest_scope_preserves_pseudonyms_between_documents() {
    let s = secret(7, "corpus"); let request = RedactionRequest::default();
    let context = pseudonym_context(Some(&s), &request).unwrap().unwrap();
    let first = context.pseudonym(PiiKind::Person, "Alice").unwrap();
    let _ = context.pseudonym(PiiKind::Person, "Bob").unwrap();
    assert_eq!(first, context.pseudonym(PiiKind::Person, "Alice").unwrap());
    let other = secret(7, "separate-job");
    assert_ne!(first, pseudonym_context(Some(&other), &request).unwrap().unwrap().pseudonym(PiiKind::Person, "Alice").unwrap());
}
#[test]
fn placeholder_only_run_needs_no_key_but_saved_commitment_is_not_ignored() {
    let mut request = RedactionRequest::default(); assert!(pseudonym_context(None, &request).unwrap().is_none());
    request.actions.expected_key_commitment = Some("0".repeat(64));
    assert!(pseudonym_context(None, &request).is_err());
}
#[test]
fn owned_configuration_and_io_use_the_public_hosted_entry_points() {
    fn send<T: Send + 'static>() {}
    send::<RedactionCorpusConfig>(); send::<RedactionPseudonyms>();
    send::<StreamInput<(RedactionCorpusConfig, Option<RedactionPseudonyms>), std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    let _corpus = NlpEngine::batch_int8_redact::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _single = NlpEngine::redact_int8;
}
