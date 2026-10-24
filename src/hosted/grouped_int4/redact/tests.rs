//! Full-capacity admission, retained secrets and failure ownership; no model run.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::TaskBudget, ner::NerOptions, source_planning::SourcePlanningLimits,
        redact::{actions::{RedactionAction, VerificationStatus}, pseudonym::{PseudonymKey, PseudonymEncoding}}},
};

fn config(width: usize) -> RedactConfig<Int4RedactionConfig> {
    let digest = Sha256Digest::of_bytes(b"int4-redaction-host-fixture");
    let context = 128;
    let mut request = RedactionRequest::default();
    request.edit_budget.max_output_bytes = 65_536;
    request.rule_budget.max_input_bytes = 65_536;
    RedactConfig {
        ner_identity: ExecutionIdentity {
            schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: digest,
            artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
            packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
                INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width,
            )).unwrap()),
            tokenizer_digest: digest, template_digest: digest, task_spec: NER_TASK_VERSION.to_owned(),
            taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(),
            schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
            kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: digest, decision_policy_digest: digest,
            backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
        },
        request,
        detector: Int4RedactionConfig {
            ner: NerOptions::default(),
            per_pass: TaskBudget { max_input_tokens: 64, max_output_tokens: 64, max_output_bytes: 65_536,
                max_grammar_states: 4096,
                max_kv_bytes: Int4MemoryRequirement::for_context(context, width).unwrap().kv_bytes },
            planning: SourcePlanningLimits { max_input_bytes: 65_536, max_context_tokens: context, ..Default::default() },
            max_model_work: constrained_int4::planned_work(width, 64, 64).unwrap()
                .checked_add(constrained_int4::planned_work(width, 64, 64).unwrap()).unwrap(),
            mask_limits: MaskWorkLimits::default(), mask_visits_per_pass: 1000,
            max_mask_visits: 2000, max_result_bytes: 65_536,
        },
        native: NativeLimits { context_tokens: context, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000, cleanup_reserve_bytes: 65_536 } },
        preparation_reserve_bytes: 1 << 20, edit_reserve_bytes: 1 << 20,
    }
}
fn check(c: &RedactConfig<Int4RedactionConfig>, bytes: usize, width: usize)
    -> Result<Int4MemoryRequirement, HostedError> {
    requirements(c, bytes, width, c.ner_identity.tokenizer_digest, c.ner_identity.template_digest)
}

#[test]
fn actual_group_and_entire_resident_kv_are_admitted_for_both_passes() {
    for width in [3, 16, 128, 10752] {
        let mut c = config(width);
        let needed = check(&c, 5, width).unwrap();
        assert_eq!(needed, Int4MemoryRequirement::for_context(c.native.context_tokens, width).unwrap());
        assert!(check(&c, 5, if width == 16 { 32 } else { 16 }).is_err());
        c.detector.per_pass.max_kv_bytes = Int4MemoryRequirement::for_context(64, width).unwrap().kv_bytes;
        assert!(check(&c, 5, width).is_err());
    }
}

#[test]
fn task_profile_backend_and_pinned_assets_cannot_be_relabelled() {
    for axis in 0..5 {
        let mut c = config(128);
        match axis {
            0 => c.ner_identity.task_spec = "redact-v1".to_owned(),
            1 => c.ner_identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => c.ner_identity.backend_semantic_version = "other".to_owned(),
            3 => c.ner_identity.kv_dtype = "int8".to_owned(),
            _ => c.ner_identity.packing_set_digest = Sha256Digest::of_bytes(b"other-partition"),
        }
        assert!(matches!(check(&c, 5, 128), Err(HostedError::ModelIdentity)));
    }
    let c = config(128); let other = Sha256Digest::of_bytes(b"other-assets");
    assert!(requirements(&c, 5, 128, other, c.ner_identity.template_digest).is_err());
    assert!(requirements(&c, 5, 128, c.ner_identity.tokenizer_digest, other).is_err());
}

#[test]
fn source_workspace_and_complete_mask_reservations_refuse_before_native_execution() {
    for axis in 0..8 {
        let mut c = config(32);
        match axis {
            0 => c.preparation_reserve_bytes = 0,
            1 => c.edit_reserve_bytes = 0,
            2 => c.native.allocator_reserve_bytes = 0,
            3 => c.request.rule_budget.max_input_bytes = 4,
            4 => c.detector.planning.max_input_bytes = 4,
            5 => c.detector.planning.max_context_tokens += 1,
            6 => c.detector.mask_limits.max_trie_node_visits = 0,
            _ => c.detector.max_mask_visits = 1000,
        }
        assert!(check(&c, 5, 32).is_err(), "axis {axis}");
    }
    let mut c = config(32); c.detector.max_mask_visits = 1000;
    c.request.verify = false; check(&c, 5, 32).unwrap();
    c.request.verify = true; assert!(check(&c, 5, 32).is_err());
    c.detector.mask_visits_per_pass = u64::MAX; c.detector.max_mask_visits = u64::MAX;
    assert!(check(&c, 5, 32).is_err());
}

#[test]
fn retained_source_and_namespace_capacity_are_charged_through_physical_completion() {
    let c = config(32); let mut source = String::with_capacity(10_000); source.push_str("Alice");
    assert_eq!(input_bytes(&source, &c, None).unwrap(), source.capacity() as u64 + c.preparation_reserve_bytes);
    let mut namespace = String::with_capacity(4096); namespace.push_str("job");
    let secret = RedactionPseudonyms { key: Arc::new(PseudonymKey::from_bytes(&[7; 32], "key").unwrap()), namespace };
    assert_eq!(input_bytes(&source, &c, Some(&secret)).unwrap(), source.capacity() as u64
        + c.preparation_reserve_bytes + secret.namespace.capacity() as u64 + 2048);
    let mut overflow = config(32); overflow.preparation_reserve_bytes = u64::MAX;
    assert!(input_bytes(&source, &overflow, Some(&secret)).is_err());
}

#[test]
fn transformed_text_ner_results_and_tokens_have_separate_checked_workspace() {
    let mut c = config(32);
    assert_eq!(temporary_bytes(&c).unwrap(), 4 * c.detector.per_pass.max_output_bytes
        + 8 * u64::from(c.detector.per_pass.max_output_tokens) + c.edit_reserve_bytes);
    c.detector.per_pass.max_output_bytes = u64::MAX;
    assert!(temporary_bytes(&c).is_err());
    let mut c = config(32); c.edit_reserve_bytes = u64::MAX;
    assert!(temporary_bytes(&c).is_err());
}

#[test]
fn full256_pseudonym_scope_requires_the_declared_key_commitment() {
    let mut c = config(32); c.request.actions.default_action = RedactionAction::Pseudonymize;
    assert!(matches!(pseudonym_context(None, &c.request),
        Err(HostedError::Int4(Int4HostError::Redaction(Int4RedactionError::Redaction(RedactError::MissingKey))))));
    let secret = RedactionPseudonyms {
        key: Arc::new(PseudonymKey::from_bytes(&[7; 32], "key").unwrap()), namespace: "job".to_owned(),
    };
    c.request.actions.expected_key_commitment = Some(secret.key.commitment());
    let context = pseudonym_context(Some(&secret), &c.request).unwrap().unwrap();
    assert_eq!(context.identity().encoding, PseudonymEncoding::Full256);
    c.request.actions.expected_key_commitment = Some("wrong".to_owned());
    assert!(matches!(pseudonym_context(Some(&secret), &c.request),
        Err(HostedError::Int4(Int4HostError::Redaction(Int4RedactionError::Redaction(RedactError::KeyMismatch))))));
}

#[test]
fn whole_envelope_limit_remains_separate_from_the_edited_text_bound() {
    let mut c = config(32); c.detector.max_result_bytes = c.request.edit_budget.max_output_bytes as u64 - 1;
    assert!(check(&c, 5, 32).is_err());
    c.detector.max_result_bytes = 64 * 1024 * 1024 + 1;
    assert!(check(&c, 5, 32).is_err());
}

#[test]
fn residual_failure_keeps_only_a_typed_count_outside_the_output_reservation() {
    use crate::{tasks::redact::{pipeline::LeakReport, union::RedactionRegion, PiiKind, Detector},
        validation::grounded_fields::VerifiedSourceSpan};
    let report = LeakReport { schema_version: 1, rules: Default::default(), model_types: Default::default(),
        residuals: vec![RedactionRegion {
            span: VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 5 },
            kinds: [PiiKind::Person].into_iter().collect(), detectors: [Detector::NerSourceV1].into_iter().collect(),
        }] };
    assert!(matches!(execution_error(Int4RedactionError::Residual(report)),
        HostedError::Int4(Int4HostError::Redaction(Int4RedactionError::Redaction(
            RedactError::VerificationResidual { count: 1 })))));
}

#[test]
fn typed_cancellation_and_secret_owners_cross_the_completion_boundary() {
    fn send<T: Send + 'static>() {}
    send::<RedactInput>(); send::<RedactionPseudonyms>(); send::<Int4RedactionRun>();
    send::<HostedOutput<Int4RedactionRun>>();
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::CostBudget] {
        let error = redaction_error(Int4RedactionError::Cancelled(cause));
        let HostedError::Int4(Int4HostError::Redaction(ref inner)) = error else { panic!("wrong stage"); };
        assert_eq!(inner.cancellation(), Some(cause));
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
    }
    assert_ne!(VerificationStatus::CleanDeclaredUnion, VerificationStatus::NotRequested);
    let _host = NlpEngine::redact_int4;
}
