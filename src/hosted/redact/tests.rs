//! Admission geometry, owned secrets and result lifetime types; no model run.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits, native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{ir::TaskBudget, ner::NerOptions, source_planning::SourcePlanningLimits,
        redact::actions::{RedactionAction, VerificationStatus}}};
fn config() -> RedactConfig {
    let d = Sha256Digest::of_bytes(b"hosted-redaction-fixture");
    RedactConfig {
        ner_identity: ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
            tokenizer_digest: d, template_digest: d, task_spec: NER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
            grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None },
        request: RedactionRequest::default(),
        detector: Int8RedactionConfig { ner: NerOptions::default(),
            per_pass: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
                max_grammar_states: 4096, max_kv_bytes: 1 << 31 }, planning: SourcePlanningLimits::default(),
            max_model_work: Int8Work::for_sequence(0, 8192, 1_000_000).unwrap(),
            mask_limits: MaskWorkLimits::default(), mask_visits_per_pass: 1000, max_mask_visits: 2000, max_result_bytes: 4 << 20 },
        native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100000, cleanup_reserve_bytes: 65536 } },
        preparation_reserve_bytes: 16 << 20, edit_reserve_bytes: 4 << 20,
    }
}
fn check(c: &RedactConfig, bytes: usize, kv: u64) -> Result<(), HostedError> {
    validate(c, bytes, c.ner_identity.tokenizer_digest, c.ner_identity.template_digest, kv)
}
#[test]
fn host_context_and_full_resident_kv_cannot_exceed_the_task() {
    let mut c = config(); check(&c, 5, c.detector.per_pass.max_kv_bytes).unwrap();
    assert!(check(&c, 5, c.detector.per_pass.max_kv_bytes + 1).is_err());
    c.native.context_tokens -= 1; assert!(check(&c, 5, 4096).is_err());
}
#[test]
fn wrong_task_profile_backend_or_pinned_assets_refuses_before_native_allocation() {
    for axis in 0..4 {
        let mut c = config();
        match axis { 0 => c.ner_identity.task_spec = "redact-v1".to_owned(),
            1 => c.ner_identity.numerics_profile = NumericsProfile::HfBf16Eager,
            2 => c.ner_identity.backend_semantic_version = "other".to_owned(), _ => c.ner_identity.kv_dtype = "int8".to_owned() }
        assert!(matches!(check(&c, 5, 4096), Err(HostedError::ModelIdentity)));
    }
    let c = config(); let other = Sha256Digest::of_bytes(b"other");
    assert!(validate(&c, 5, other, c.ner_identity.template_digest, 4096).is_err());
    assert!(validate(&c, 5, c.ner_identity.tokenizer_digest, other, 4096).is_err());
}
#[test]
fn all_owned_limits_cover_both_passes_not_only_the_original_document() {
    for axis in 0..5 {
        let mut c = config();
        match axis { 0 => c.preparation_reserve_bytes = 0, 1 => c.edit_reserve_bytes = 0,
            2 => c.request.rule_budget.max_input_bytes = 4, 3 => c.detector.planning.max_input_bytes = 4,
            _ => c.detector.max_mask_visits = 1000 }
        assert!(check(&c, 5, 4096).is_err());
    }
    let mut c = config(); c.detector.max_mask_visits = 1000; c.request.verify = false;
    check(&c, 5, 4096).unwrap();
    c.request.verify = true; c.detector.mask_visits_per_pass = u64::MAX; c.detector.max_mask_visits = u64::MAX;
    assert!(check(&c, 5, 4096).is_err());
}
#[test]
fn original_and_namespace_spare_capacity_are_included_in_the_claim() {
    let c = config(); let mut source = String::with_capacity(10000); source.push_str("Alice");
    assert_eq!(input_bytes(&source, &c, None).unwrap(), source.capacity() as u64 + c.preparation_reserve_bytes);
    let mut namespace = String::with_capacity(4096); namespace.push_str("job");
    let secret = RedactionPseudonyms { key: Arc::new(PseudonymKey::from_bytes(&[7;32], "key").unwrap()), namespace };
    assert_eq!(input_bytes(&source, &c, Some(&secret)).unwrap(), source.capacity() as u64 + c.preparation_reserve_bytes
        + secret.namespace.capacity() as u64 + 2048);
}
#[test]
fn intermediate_ner_storage_is_priced_separately_and_arithmetic_cannot_wrap() {
    let mut c = config();
    assert_eq!(temporary_bytes(&c).unwrap(), 4 * c.detector.per_pass.max_output_bytes
        + 8 * u64::from(c.detector.per_pass.max_output_tokens) + c.edit_reserve_bytes);
    c.detector.per_pass.max_output_bytes = u64::MAX; assert!(temporary_bytes(&c).is_err());
    let mut c = config(); c.preparation_reserve_bytes = u64::MAX;
    assert!(input_bytes(&"x".to_owned(), &c, None).is_err());
}
#[test]
fn missing_or_wrong_pseudonym_key_never_silently_changes_the_action() {
    let mut c = config(); c.request.actions.default_action = RedactionAction::Pseudonymize;
    assert!(matches!(pseudonym_context(None, &c.request), Err(HostedError::Redaction(Int8RedactionError::Redaction(RedactError::MissingKey)))));
    let secret = RedactionPseudonyms { key: Arc::new(PseudonymKey::from_bytes(&[7;32], "key").unwrap()), namespace: "job".to_owned() };
    c.request.actions.expected_key_commitment = Some(secret.key.commitment());
    let context = pseudonym_context(Some(&secret), &c.request).unwrap().unwrap();
    assert_eq!(context.identity().encoding, crate::tasks::redact::pseudonym::PseudonymEncoding::Full256);
    c.request.actions.expected_key_commitment = Some("wrong".to_owned());
    assert!(matches!(pseudonym_context(Some(&secret), &c.request), Err(HostedError::Redaction(Int8RedactionError::Redaction(RedactError::KeyMismatch)))));
}
#[test]
fn complete_outer_output_cap_is_not_replaced_by_a_nested_text_limit() {
    let mut c = config(); c.detector.max_result_bytes = c.request.edit_budget.max_output_bytes as u64 - 1;
    assert!(check(&c, 5, 4096).is_err());
    c.detector.max_result_bytes = 64 * 1024 * 1024 + 1; assert!(check(&c, 5, 4096).is_err());
}
#[test]
fn typed_cancellation_and_secret_ownership_cross_the_completion_boundary() {
    fn send<T: Send + 'static>() {}
    send::<RedactInput>(); send::<RedactionPseudonyms>(); send::<Int8RedactionRun>(); send::<HostedOutput<Int8RedactionRun>>();
    let e = HostedError::Redaction(Int8RedactionError::Cancelled(DecodeCancellationKind::Deadline));
    let HostedError::Redaction(ref inner) = e else { unreachable!() };
    assert_eq!(inner.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert_eq!(format!("{e}"), "hosted native redaction failed");
    assert_eq!(format!("{e:?}"), "hosted native redaction failed");
    // Verification remains an explicit result status, never inferred from a
    // completed runtime wrapper or from the fact that memory was admitted.
    assert_ne!(VerificationStatus::CleanDeclaredUnion, VerificationStatus::NotRequested);
}

#[test]
fn residual_failure_does_not_return_an_uncharged_coordinate_vector() {
    use crate::{tasks::redact::{pipeline::LeakReport, union::RedactionRegion, PiiKind, Detector},
        validation::grounded_fields::VerifiedSourceSpan};
    let report = LeakReport { schema_version: 1, rules: Default::default(), model_types: Default::default(),
        residuals: vec![RedactionRegion {
            span: VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 },
            kinds: [PiiKind::Person].into_iter().collect(), detectors: [Detector::NerSourceV1].into_iter().collect(),
        }] };
    assert!(matches!(execution_error(Int8RedactionError::Residual(report)),
        HostedError::Redaction(Int8RedactionError::Redaction(RedactError::VerificationResidual { count: 1 }))));
}
