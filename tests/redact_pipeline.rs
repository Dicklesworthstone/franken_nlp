//! Public, model-free redaction API. Native inference remains separately
//! model-gated; these tests do not substitute synthetic spans for model proof.
use franken_nlp::{canonjson, tasks::redact::{
    PiiKind, RedactionRequest, redact_rules,
    actions::{ActionPolicy, RedactionAction, VerificationStatus},
    pseudonym::{PseudonymKey, Pseudonyms, PseudonymBudget},
}};

#[test]
fn rule_redaction_is_a_complete_verified_result_without_weights() {
    let result = redact_rules("Contact a@example.org.", &RedactionRequest::default(), None).unwrap();
    assert_eq!(result.text(), "Contact [redacted:email].");
    assert_eq!(result.verification(), VerificationStatus::CleanDeclaredUnion);
    assert!(result.model_types().is_empty());
    assert!(result.edits().is_empty());
    let wire = canonjson::canonical_string(&result).unwrap();
    assert!(!wire.contains("a@example.org")); assert!(!wire.contains("confidence"));
}

#[test]
fn reusable_sealed_context_supports_multiple_documents_without_lazy_insertion() {
    let key = PseudonymKey::from_bytes(&[17;32], "test-only-key-v1").unwrap();
    let context = Pseudonyms::preflight128(&key, "test-job", &[(PiiKind::Email, "a@b.org"), (PiiKind::Email, "c@d.org")],
        PseudonymBudget::default(), Some(&key.commitment())).unwrap();
    let mut request = RedactionRequest::default(); request.actions.default_action = RedactionAction::Pseudonymize;
    let first = redact_rules("a@b.org", &request, Some(&context)).unwrap();
    let again = redact_rules("a@b.org", &request, Some(&context)).unwrap();
    assert_eq!(first.text(), again.text());
    assert!(redact_rules("c@d.org", &request, Some(&context)).is_ok());
    assert!(redact_rules("unlisted@example.org", &request, Some(&context)).is_err());
}

#[test]
fn mixed_kind_pseudonym_overlap_masks_the_entire_region_instead_of_inventing_a_type() {
    let key = PseudonymKey::from_bytes(&[17;32], "test-key").unwrap();
    let context = Pseudonyms::full256(&key, "test-job", None).unwrap();
    let mut request = RedactionRequest::default();
    request.actions = ActionPolicy { default_action: RedactionAction::Pseudonymize, include_map: true, ..ActionPolicy::default() };
    // Valid Luhn test card also lies in the declared plain-phone shape.
    let result = redact_rules("4222222222222", &request, Some(&context)).unwrap();
    assert_eq!(result.text(), "*************");
    assert_eq!(result.edits()[0].applied_action, RedactionAction::Mask);
    assert!(result.edits()[0].original.kinds.contains(&PiiKind::Phone));
    assert!(result.edits()[0].original.kinds.contains(&PiiKind::CreditCard));
}

#[test]
fn verification_choice_is_bound_into_the_public_policy_digest() {
    let verified = RedactionRequest::default();
    let unverified = RedactionRequest { verify: false, ..verified.clone() };
    let a = redact_rules("ordinary text", &verified, None).unwrap();
    let b = redact_rules("ordinary text", &unverified, None).unwrap();
    assert_eq!(a.text(), b.text()); assert_ne!(a.policy_digest(), b.policy_digest());
    assert_eq!(b.verification(), VerificationStatus::NotRequested);
}

#[test]
fn original_and_output_map_coordinates_are_utf8_safe() {
    let source = "上海 a@example.org é";
    let mut request = RedactionRequest::default(); request.actions.include_map = true;
    let result = redact_rules(source, &request, None).unwrap();
    for edit in result.edits() {
        let original = edit.original.span;
        assert_eq!(source[..original.byte_start].chars().count(), original.scalar_start);
        assert_eq!(source[..original.byte_end].chars().count(), original.scalar_end);
        assert_eq!(result.text()[..edit.output_byte_start].chars().count(), edit.output_scalar_start);
        assert_eq!(result.text()[..edit.output_byte_end].chars().count(), edit.output_scalar_end);
    }
}
