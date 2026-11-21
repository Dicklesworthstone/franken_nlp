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

#[test]
fn configured_corpus_policy_matches_explicit_plans_and_cannot_be_replaced() {
    let byte = planner(None);
    let configured = planner(None).with_document_encoding(policy()).unwrap();
    let expected = byte.prepare_with_document_encoding(boolean(), &policy(), &mut Continue).unwrap();
    let actual = configured.prepare_with_control(boolean(), &mut Continue).unwrap();
    assert_eq!(actual.execution_identity(), expected.execution_identity());
    assert_eq!(actual.planned_work(), expected.planned_work());
    assert_eq!(actual.source().document_encoding(), configured.document_encoding());
    assert!(configured.prepare(boolean()).is_err(), "compact preparation needs caller control");
    assert!(configured.prepare_with_document_encoding(boolean(), &UntrustedEncoding::ByteFallback,
        &mut Continue).is_err());
    let UntrustedEncoding::SafeBpeV1(limits) = policy() else { unreachable!() };
    assert_eq!(configured.encoding_workspace_bytes().unwrap(), limits.requirements().unwrap().workspace_bytes);
    assert_eq!(byte.encoding_workspace_bytes().unwrap(), 0);
    for axis in 0..4 {
        let mut changed = limits;
        match axis { 0 => changed.max_input_bytes -= 1, 1 => changed.max_output_tokens -= 1,
            2 => changed.max_pair_checks -= 1, _ => changed.max_pair_bytes -= 1 }
        assert!(configured.prepare_with_document_encoding(boolean(), &UntrustedEncoding::SafeBpeV1(changed),
            &mut Continue).is_err());
    }
    assert!(configured.with_document_encoding(policy()).is_err());
}

#[test]
fn compact_corpus_schema_and_source_share_the_complete_record_allowance() {
    let schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    let text = "Alice 上海 <think>";
    let UntrustedEncoding::SafeBpeV1(mut limits) = policy() else { unreachable!() };
    limits.max_input_bytes = schema.len() + text.len();
    let exact = planner(None).with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    let make = || document(schema, text, ExtractionBatchGrounding::SourceMembership);
    let first = exact.prepare_with_control(make(), &mut Continue).unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    for (index, expected) in [(2, schema), (4, text)] {
        let tokens = first.task.ir().prompt_segments()[index].token_ids();
        assert_eq!(tokenizer.tokenizer().decode_bytes(tokens).unwrap(), expected.as_bytes());
        assert!(tokens.iter().all(|id| !exact.compiler.controls.contains(*id)));
    }
    limits.max_input_bytes -= 1; // Each segment fits alone, their sum does not.
    let short = planner(None).with_document_encoding(UntrustedEncoding::SafeBpeV1(limits)).unwrap();
    assert!(short.prepare_with_control(make(), &mut Continue).is_err());
    let second = exact.prepare_with_control(make(), &mut Continue).unwrap();
    assert_eq!(first.execution_identity(), second.execution_identity());
}

#[test]
fn retained_encoding_debits_rejected_schemas_and_never_renews_invocation_work() {
    let configured = planner(None).with_document_encoding(policy()).unwrap();
    let text = "The quick brown fox jumps over the lazy dog.";
    let schema = r#"{"type":"string","pattern":".*"}"#;
    let bytes = schema.len() + text.len();
    let total = SafeBpeWorkLimits { max_input_bytes: bytes as u64,
        max_output_tokens: 4096, max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 };
    let mut used = SafeBpeWork::default();
    let error = configured.prepare_with_invocation_encoding_work(
        document(schema, text, ExtractionBatchGrounding::Structural), &mut used, Some(total), &mut Continue)
        .err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Planning);
    assert_eq!(used.input_bytes(), bytes);
    assert!(used.output_tokens() > 0);
    let retained = used;
    let error = configured.prepare_with_invocation_encoding_work(boolean(), &mut used, Some(total), &mut Continue)
        .err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
    assert_eq!(used, retained);
}

#[test]
fn remaining_invocation_work_does_not_change_a_successful_plan_identity() {
    let configured = planner(None).with_document_encoding(policy()).unwrap();
    let total = SafeBpeWorkLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 };
    let mut used = SafeBpeWork::default();
    let first = configured.prepare_with_invocation_encoding_work(boolean(), &mut used, Some(total), &mut Continue).unwrap();
    let prior = used;
    let second = configured.prepare_with_invocation_encoding_work(boolean(), &mut used, Some(total), &mut Continue).unwrap();
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert_eq!(used.input_bytes(), prior.input_bytes() * 2);
    assert_eq!(used.output_tokens(), prior.output_tokens() * 2);
    assert_eq!(used.pair_checks(), prior.pair_checks() * 2);
    assert_eq!(used.pair_bytes(), prior.pair_bytes() * 2);
}

#[test]
fn controlled_corpus_compilation_retains_terminal_source_index_cancellation() {
    struct Count { polls: usize, stop: Option<usize>, cause: DecodeCancellationKind }
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.stop == Some(self.polls)).then_some(self.cause)
        }
    }
    let configured = planner(None).with_document_encoding(policy()).unwrap();
    let make = || document(r#"{"type":"string","x-fnlp-source":"verbatim"}"#,
        "Alice 上海 Alice", ExtractionBatchGrounding::SourceMembership);
    let mut count = Count { polls: 0, stop: None, cause: DecodeCancellationKind::User };
    configured.prepare_with_control(make(), &mut count).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut control = Count { polls: 0, stop: Some(count.polls - 1), cause };
        let error = configured.prepare_with_control(make(), &mut control).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.cancellation, Some(cause));
        assert_eq!(control.polls, count.polls - 1);
    }
}
