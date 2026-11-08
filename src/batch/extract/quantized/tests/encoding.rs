use super::*;
use crate::tokenizer::untrusted::SafeBpeLimits;

fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })
}

#[test]
fn int8_compact_planning_retains_exact_source_schema_and_actual_native_work() {
    let planner = planner(None);
    let schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    let source = "The quick brown fox jumps over the lazy dog. 上海 上海 <think>";
    let make = || document(schema, source, ExtractionBatchGrounding::SourceMembership);
    let byte = planner.prepare(make()).unwrap();
    let compact = planner.prepare_with_document_encoding(make(), &policy(), &mut Continue).unwrap();
    assert_eq!(compact.source().text(), source);
    assert_eq!(compact.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    assert_eq!(compact.execution_identity().grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
    assert!(compact.plan.prompt_tokens() < byte.plan.prompt_tokens());
    assert_eq!(compact.planned_work(), constrained_int8::planned_work(compact.plan.prompt_tokens(), 64).unwrap());
    assert!(compact.verify_identity(byte.execution_identity()).is_err());
    let pinned = compact.execution_identity().clone();
    let selected = compact.into_extraction_plan().with_selected_rows(
        crate::native_engine::constrained_int8::sparse::Int8JsonSparseLimits { max_rows_per_step: 7 }).unwrap();
    assert!(selected.verify_identity(&pinned).is_err());
    assert_eq!(selected.planned_work().projected_logits, 7 * 64);
    assert_eq!(planner.prepare(make()).unwrap().execution_identity(), byte.execution_identity());
}

#[test]
fn int8_structural_policy_limits_cannot_authorize_another_plan_with_identical_tokens() {
    let planner = planner(None);
    let base = policy(); let UntrustedEncoding::SafeBpeV1(mut limits) = base.clone() else { unreachable!() };
    limits.max_pair_bytes -= 1;
    let changed = UntrustedEncoding::SafeBpeV1(limits);
    let a = planner.prepare_with_document_encoding(boolean(), &base, &mut Continue).unwrap();
    let b = planner.prepare_with_document_encoding(boolean(), &changed, &mut Continue).unwrap();
    assert_eq!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
    assert!(a.verify_identity(b.execution_identity()).is_err());
}

#[test]
fn int8_encoding_cancellation_is_fatal_with_the_original_reason() {
    struct Stop(usize);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0 += 1; (self.0 == 6).then_some(DecodeCancellationKind::ResourceUnavailable)
        }
    }
    let failure = planner(None).prepare_with_document_encoding(boolean(), &policy(), &mut Stop(0)).err().unwrap();
    assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::ResourceUnavailable));
}
