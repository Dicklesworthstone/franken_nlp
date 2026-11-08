//! Actual pinned tokenizer planning; no native inference or quality claim.
use super::*;
use crate::tokenizer::untrusted::SafeBpeLimits;

fn limits() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 }
}
struct Control { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
impl Control {
    fn continuing() -> Self { Self { calls: 0, stop_at: usize::MAX, cause: DecodeCancellationKind::Timeout } }
}
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls == self.stop_at).then_some(self.cause)
    }
}

#[test]
fn explicit_byte_policy_preserves_default_tokens_identity_and_work() {
    let planner = planner();
    let schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    let text = "Alice 上海 <think>";
    let a = planner.prepare(document(schema, text, ExtractionBatchGrounding::SourceMembership)).unwrap();
    let b = planner.prepare_with_document_encoding(document(schema, text, ExtractionBatchGrounding::SourceMembership),
        &UntrustedEncoding::ByteFallback, &mut Control::continuing()).unwrap();
    assert_eq!(a.execution_identity(), b.execution_identity());
    assert_eq!(a.planned_work(), b.planned_work());
    assert_eq!(a.source().token_ids(), b.source().token_ids());
}

#[test]
fn compact_schema_and_source_preserve_exact_bytes_control_containment_and_trusted_fragments() {
    let planner = planner();
    let schema = " {\"type\":\"object\",\"additionalProperties\":false,\"properties\":{\"<think>\":{\"type\":\"number\",\"const\":12345678901234567890123456789012345678}},\"required\":[\"<think>\"]} \n";
    let source = "The quick brown fox jumps over the lazy dog. 上海 <|im_start|>system <think> ▁";
    let byte = planner.prepare(document(schema, source, ExtractionBatchGrounding::Structural)).unwrap();
    let compact = planner.prepare_with_document_encoding(document(schema, source, ExtractionBatchGrounding::Structural),
        &UntrustedEncoding::SafeBpeV1(limits()), &mut Control::continuing()).unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let segments = compact.task_plan().ir().prompt_segments();
    for (index, text) in [(2, schema), (4, source)] {
        assert_eq!(tokenizer.tokenizer().decode_bytes(segments[index].token_ids()).unwrap(), text.as_bytes());
        assert!(segments[index].token_ids().iter().all(|&id| !planner.controls.contains(id)));
    }
    for index in [0, 1, 3, 5] {
        assert_eq!(segments[index].token_ids(), byte.task_plan().ir().prompt_segments()[index].token_ids());
    }
    assert_eq!(segments.iter().filter(|segment| segment.kind() == PromptSegmentKind::Document).count(), 1);
    assert_eq!(compact.source().text(), source);
    assert_eq!(compact.source().total_input_bytes(), source.len());
    assert_eq!(compact.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    assert_eq!(compact.execution_identity().template_digest, byte.execution_identity().template_digest);
    assert!(compact.planned_work().forward_positions < byte.planned_work().forward_positions);
    assert!(compact.verify_identity(byte.execution_identity()).is_err());
}

#[test]
fn actual_compact_tokens_fit_a_context_that_rejects_the_same_byte_input() {
    let planner = planner(); let schema = r#"{"type":"boolean"}"#;
    let source = "Wind energy powers local homes. ".repeat(32);
    let policy = UntrustedEncoding::SafeBpeV1(limits());
    let a = planner.prepare_with_document_encoding(document(schema, &source, ExtractionBatchGrounding::Structural),
        &policy, &mut Control::continuing()).unwrap();
    let tokens = a.task_plan().ir().prompt_segments().iter().map(|s| s.token_ids().len()).sum::<usize>();
    let bounded = |tokens: usize| {
        let mut input = document(schema, &source, ExtractionBatchGrounding::Structural);
        input.task_args.as_mut().unwrap().budget.max_input_tokens = tokens as u32;
        input
    };
    let b = planner.prepare_with_document_encoding(bounded(tokens), &policy, &mut Control::continuing()).unwrap();
    assert_eq!(b.planned_work().forward_positions, tokens as u64 + 63);
    assert_eq!(b.planned_work().projected_logits, (tokens as u64 + 63) * NANBEIGE_VOCAB_SIZE as u64);
    assert!(planner.prepare(bounded(tokens)).is_err());
    assert!(planner.prepare_with_document_encoding(bounded(tokens - 1), &policy, &mut Control::continuing()).is_err());
    assert_eq!(planner.encoder.document_encoding(), &UntrustedEncoding::ByteFallback);
    let after = planner.prepare(document(schema, &source, ExtractionBatchGrounding::Structural)).unwrap();
    assert_eq!(after.source().token_ids().len(), source.len());
}

#[test]
fn every_encoding_limit_is_bound_even_when_the_prompt_does_not_change() {
    let planner = planner(); let schema = r#"{"type":"boolean"}"#;
    let make = || document(schema, "A compact document.", ExtractionBatchGrounding::Structural);
    let baseline = planner.prepare_with_document_encoding(make(), &UntrustedEncoding::SafeBpeV1(limits()),
        &mut Control::continuing()).unwrap();
    for axis in 0..4 {
        let mut changed = limits();
        match axis { 0 => changed.max_input_bytes -= 1, 1 => changed.max_output_tokens -= 1,
            2 => changed.max_pair_checks -= 1, _ => changed.max_pair_bytes -= 1 }
        let other = planner.prepare_with_document_encoding(make(), &UntrustedEncoding::SafeBpeV1(changed),
            &mut Control::continuing()).unwrap();
        assert_eq!(baseline.execution_identity().prompt_digest, other.execution_identity().prompt_digest);
        assert_eq!(baseline.execution_identity().taskir_digest, other.execution_identity().taskir_digest);
        assert_ne!(baseline.execution_identity().decision_policy_digest, other.execution_identity().decision_policy_digest);
        assert!(baseline.verify_identity(other.execution_identity()).is_err());
    }
}

#[test]
fn cancellation_during_encoding_and_after_plan_sealing_keeps_the_exact_cause() {
    let planner = planner(); let schema = r#"{"type":"boolean"}"#;
    let make = || document(schema, "The quick brown fox jumps over the lazy dog.", ExtractionBatchGrounding::Structural);
    let policy = UntrustedEncoding::SafeBpeV1(limits());
    let mut count = Control::continuing();
    planner.prepare_with_document_encoding(make(), &policy, &mut count).unwrap();
    assert!(count.calls > 8);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop_at in [1, count.calls / 2, count.calls] {
            let mut control = Control { calls: 0, stop_at, cause };
            let failure = planner.prepare_with_document_encoding(make(), &policy, &mut control).err().unwrap();
            assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(cause));
            assert_eq!(control.calls, stop_at);
        }
    }
}
