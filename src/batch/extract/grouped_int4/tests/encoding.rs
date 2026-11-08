use super::*;
use crate::tokenizer::untrusted::SafeBpeLimits;

fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }

#[test]
fn int4_compact_source_and_selected_rows_are_independent_sealed_policies() {
    let dense = planner(127, None);
    let selected = planner(127, None).with_selected_rows(Int4JsonSparseLimits { max_rows_per_step: 7 }).unwrap();
    let schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    let text = "The quick brown fox jumps over the lazy dog. 上海 上海 <|im_start|>";
    let make = || document(schema, text, ExtractionBatchGrounding::SourceMembership);
    let byte = dense.prepare(make()).unwrap();
    let compact = dense.prepare_with_document_encoding(make(), &policy(), &mut Continue).unwrap();
    let sparse = selected.prepare_with_document_encoding(make(), &policy(), &mut Continue).unwrap();
    assert_eq!(compact.source().text(), text); assert_eq!(sparse.source().text(), text);
    assert_eq!(compact.request_digest(), byte.request_digest());
    assert_eq!(compact.request_digest(), sparse.request_digest());
    assert_eq!(compact.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    assert_eq!(compact.execution_identity().grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
    assert!(compact.plan.prompt_tokens() < byte.plan.prompt_tokens());
    assert_eq!(compact.planned_work(), crate::native_engine::constrained_int4::planned_work(127,
        compact.plan.prompt_tokens(), 64).unwrap());
    assert_eq!(sparse.planned_work().forward_positions, compact.planned_work().forward_positions);
    assert_eq!(sparse.planned_work().projected_logits, 7 * 64);
    assert!(sparse.verify_identity(compact.execution_identity()).is_err());
    assert!(compact.verify_identity(byte.execution_identity()).is_err());
    assert_ne!(compact.comparison_policy_digest(), byte.comparison_policy_digest());
    assert_ne!(sparse.comparison_policy_digest(), compact.comparison_policy_digest());
    assert_eq!(dense.prepare(make()).unwrap().execution_identity(), byte.execution_identity());
}

#[test]
fn int4_structural_encoding_limits_bind_even_if_source_and_prompt_do_not_change() {
    let planner = planner(127, None);
    let a = planner.prepare_with_document_encoding(boolean(), &policy(), &mut Continue).unwrap();
    let UntrustedEncoding::SafeBpeV1(mut limits) = policy() else { unreachable!() };
    limits.max_output_tokens -= 1;
    let b = planner.prepare_with_document_encoding(boolean(), &UntrustedEncoding::SafeBpeV1(limits), &mut Continue).unwrap();
    assert_eq!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
    assert_ne!(a.comparison_policy_digest(), b.comparison_policy_digest());
    assert!(a.verify_identity(b.execution_identity()).is_err());
}

#[test]
fn int4_shared_schema_rejection_and_original_cancellation_survive_explicit_encoding() {
    let compiler = planner(127, Some(args(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural)))
        .require_shared_defaults().unwrap();
    let failure = compiler.prepare_with_document_encoding(boolean(), &policy(), &mut Continue).err().unwrap();
    assert_eq!(failure.fault.code, BatchCode::InvalidEnvelope);
    struct Cancel(usize);
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0 += 1; (self.0 == 7).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let mut input = boolean(); input.task_args = None;
    let failure = compiler.prepare_with_document_encoding(input, &policy(), &mut Cancel(0)).err().unwrap();
    assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline));
}
