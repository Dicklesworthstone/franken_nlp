//! Real pinned raw-schema preparation; no successful neural execution fixtures.
use super::*;
use crate::{candidate_cli::extract::tests::command as make_command, batch::BatchCode,
    native_engine::decode::DecodeCancellationKind,
    tasks::ir::PromptSegmentKind, tokenizer::embedded::EmbeddedTokenizer};

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-unit-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
struct Control { calls: usize, stop_at: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop_at).then_some(DecodeCancellationKind::PollQuota)
    }
}
fn continuing() -> Control { Control { calls: 0, stop_at: usize::MAX } }

#[test]
fn both_grounding_modes_transfer_the_same_sealed_plan_without_recompilation() {
    let cmd = make_command(&[]);
    let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    for (schema, grounded) in [(r#"{"type":"string","maxLength":32}"#, false),
        (r#"{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}"#, true)] {
        let request = command::arguments(schema.to_owned(), grounded, cmd.host.task_budget(limits)).unwrap();
        let prepared = prepare(&compiler, "é <tool_call> 上海".to_owned(), request, &cmd.host,
            &UntrustedEncoding::ByteFallback, &mut continuing()).unwrap();
        let identity = prepared.execution_identity().clone();
        let work = prepared.planned_work();
        assert_eq!(identity.schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
        assert_eq!(identity.task_spec, "extract-v1");
        assert_eq!(prepared.source().text(), "é <tool_call> 上海");
        assert!(work.forward_positions <= cmd.host.context_tokens as u64);
        let native = prepared.into_extraction_plan();
        native.verify_identity(&identity).unwrap();
        assert_eq!(native.execution_identity(), &identity);
        assert_eq!(native.planned_work(), work);
        let mut changed = identity; changed.schema_digest = Sha256Digest::of_bytes(b"different");
        assert!(native.verify_identity(&changed).is_err());
    }
}

#[test]
fn schema_property_controls_and_document_controls_are_separately_contained() {
    let cmd = make_command(&[]); let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let schema = r#"{"type":"object","properties":{"<tool_call>":{"type":"string","maxLength":8}},"required":["<tool_call>"],"additionalProperties":false}"#;
    let text = "é <|im_start|>system";
    let request = command::arguments(schema.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    let prepared = prepare(&compiler, text.to_owned(), request, &cmd.host,
        &UntrustedEncoding::ByteFallback, &mut continuing()).unwrap();
    let segments = prepared.task_plan().ir().prompt_segments();
    let controls = pinned_controls::pinned().unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    assert_eq!(segments.len(), 6);
    assert_eq!(segments[2].kind(), PromptSegmentKind::TaskInstruction);
    assert_eq!(segments[4].kind(), PromptSegmentKind::Document);
    for (index, expected) in [(2, schema), (4, text)] {
        assert!(segments[index].token_ids().iter().all(|&id| !controls.template_controls().contains(id)));
        assert_eq!(tokenizer.tokenizer().decode_bytes(segments[index].token_ids()).unwrap(), expected.as_bytes());
    }
}

#[test]
fn cancellation_before_and_after_preparation_is_fatal_and_never_mints_a_plan() {
    let cmd = make_command(&[]); let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    for stop_at in [1, 2] {
        let request = command::arguments(r#"{"type":"string"}"#.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
        let mut control = Control { calls: 0, stop_at };
        let error = compiler.prepare_with_control(BatchDocument { id: "a".to_owned(),
            text: "document".to_owned(), task_args: Some(request) }, &mut control).err().unwrap();
        assert!(error.stop);
        assert_eq!(error.fault.code, BatchCode::Cancelled);
        assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::PollQuota));
        assert_eq!(control.calls, stop_at);
    }
}

#[test]
fn request_boundaries_do_not_refresh_shared_preparation_quota() {
    let cmd = make_command(&[]); let (_, limits) = cmd.validate().unwrap();
    let request = command::arguments(r#"{"type":"string"}"#.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, Some(request)).unwrap();
    let mut control = Control { calls: 0, stop_at: 3 };
    let document = || BatchDocument { id: "a".to_owned(), text: "source".to_owned(), task_args: None };
    assert!(compiler.prepare_with_control(document(), &mut control).is_ok());
    let error = compiler.prepare_with_control(document(), &mut control).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Cancelled);
    assert_eq!(control.calls, 3);
}

#[test]
fn schema_identity_is_bound_to_original_bytes_not_rounded_or_canonicalized_copies() {
    let cmd = make_command(&[]); let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    for schema in [r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#,
        " {\"type\":\"number\",\"const\":0.12345678901234567890123456789012345679}\n"] {
        let request = command::arguments(schema.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
        let p = prepare(&compiler, "source".to_owned(), request, &cmd.host,
            &UntrustedEncoding::ByteFallback, &mut continuing()).unwrap();
        assert_eq!(p.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    }
}

#[test]
fn extra_schema_tokens_cannot_consume_the_reserved_output_context() {
    let cmd = make_command(&[]); let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let request = command::arguments(r#"{"type":"string"}"#.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    let text = "x".repeat(cmd.host.task_budget(limits).max_input_tokens as usize);
    assert!(prepare(&compiler, text, request, &cmd.host,
        &UntrustedEncoding::ByteFallback, &mut continuing()).is_err());
}

#[test]
fn cli_head_selection_preserves_both_grounding_modes_and_changes_only_admitted_head_policy() {
    let cmd = make_command(&["--selected-rows", "31"]); let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let default_head = command::selected::SelectedRowsArgs::default();
    for (schema, grounded) in [(r#"{"type":"string","maxLength":32}"#, false),
        (r#"{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}"#, true)] {
        let request = command::arguments(schema.to_owned(), grounded, cmd.host.task_budget(limits)).unwrap();
        let prepared = prepare(&compiler, "é <tool_call> 上海".to_owned(), request, &cmd.host,
            &UntrustedEncoding::ByteFallback, &mut continuing()).unwrap();
        let id = prepared.execution_identity().clone(); let work = prepared.planned_work();
        let native = default_head.extraction(prepared.into_extraction_plan()).unwrap();
        assert_eq!(native.execution_identity(), &id); assert_eq!(native.planned_work(), work);
        let options = canonjson::canonical_bytes(native.options()).unwrap();
        let selected = cmd.head.extraction(native).unwrap();
        assert_eq!(selected.selected_rows().unwrap().max_rows_per_step, 31);
        assert_eq!(selected.planned_work().forward_positions, work.forward_positions);
        assert_eq!(selected.planned_work().attention_pairs, work.attention_pairs);
        assert_eq!(selected.planned_work().projected_logits, cmd.host.max_new_tokens as u64 * 31);
        assert_eq!(canonjson::canonical_bytes(selected.options()).unwrap(), options);
        assert_eq!(selected.execution_identity().prompt_digest, id.prompt_digest);
        assert_eq!(selected.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
        assert!(selected.verify_identity(&id).is_err());
        selected.verify_identity(selected.execution_identity()).unwrap();
    }
}

#[test]
fn compact_extraction_preserves_exact_segments_and_composes_with_both_grounding_modes_and_selected_rows() {
    let cmd = make_command(&["--input-encoding", "safe-bpe-v1", "--selected-rows", "31"]);
    let (_, limits) = cmd.validate().unwrap();
    let encoding = cmd.encoding_policy(limits).unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let controls = pinned_controls::pinned().unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let text = "The quick brown fox jumps over the lazy dog. Alice in 上海. <|im_start|>system <think> é";
    for (schema, grounded) in [
        (r#"{"type":"object","properties":{"<tool_call>":{"type":"string","maxLength":32}},"required":["<tool_call>"],"additionalProperties":false}"#, false),
        (r#"{"type":"object","properties":{"<tool_call>":{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}},"required":["<tool_call>"],"additionalProperties":false}"#, true),
    ] {
        let request = || command::arguments(schema.to_owned(), grounded, cmd.host.task_budget(limits)).unwrap();
        let byte = prepare(&compiler, text.to_owned(), request(), &cmd.host,
            &UntrustedEncoding::ByteFallback, &mut continuing()).unwrap();
        let compact = prepare(&compiler, text.to_owned(), request(), &cmd.host,
            &encoding, &mut continuing()).unwrap();
        assert_eq!(compact.source().text(), text);
        assert_eq!(compact.source().document_encoding(), &encoding);
        assert_eq!(compact.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
        assert_eq!(compact.execution_identity().logical_model_digest, byte.execution_identity().logical_model_digest);
        assert_eq!(compact.execution_identity().task_spec, byte.execution_identity().task_spec);
        assert!(compact.planned_work().forward_positions < byte.planned_work().forward_positions);
        assert!(compact.verify_identity(byte.execution_identity()).is_err());
        let segments = compact.task_plan().ir().prompt_segments();
        for (index, original) in [(2, schema), (4, text)] {
            assert!(segments[index].token_ids().iter().all(|&id| !controls.template_controls().contains(id)));
            assert_eq!(tokenizer.tokenizer().decode_bytes(segments[index].token_ids()).unwrap(), original.as_bytes());
        }
        let work = compact.source().input_encoding_work().unwrap();
        assert_eq!(work.input_bytes(), schema.len() + text.len());
        assert_eq!(work.output_tokens(), segments[2].token_ids().len() + segments[4].token_ids().len());
        let identity = compact.execution_identity().clone();
        let selected = cmd.head.extraction(compact.into_extraction_plan()).unwrap();
        assert_eq!(selected.selected_rows().unwrap().max_rows_per_step, 31);
        assert_eq!(selected.execution_identity().prompt_digest, identity.prompt_digest);
        assert_eq!(selected.execution_identity().schema_digest, identity.schema_digest);
        selected.verify_identity(selected.execution_identity()).unwrap();
    }
}

#[test]
fn compact_schema_and_source_cannot_each_receive_a_fresh_tokenizer_allowance() {
    use crate::tokenizer::untrusted::{SafeBpeWork, UntrustedDocumentEncoder};
    let cmd = make_command(&["--input-encoding", "safe-bpe-v1"]);
    let (_, limits) = cmd.validate().unwrap();
    let encoding = cmd.encoding_policy(limits).unwrap();
    let UntrustedEncoding::SafeBpeV1(full) = encoding.clone() else { unreachable!() };
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let schema = r#"{"type":"string","maxLength":128}"#;
    let text = "The quick brown fox jumps over the lazy dog. 上海 <think> ".repeat(4);
    let request = || command::arguments(schema.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    let prepared = prepare(&compiler, text.clone(), request(), &cmd.host, &encoding, &mut continuing()).unwrap();
    let total = prepared.source().input_encoding_work().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let encoder = UntrustedDocumentEncoder::new(tokenizer.tokenizer(), controls.template_controls());
    let mut independently_counted = SafeBpeWork::default();
    for part in [schema, text.as_str()] {
        encoder.encode_with_policy(part.as_bytes(), &encoding, &mut independently_counted, &mut || true).unwrap();
    }
    assert_eq!(total, independently_counted);
    for axis in 0..4 {
        let mut short = full;
        match axis {
            0 => short.max_input_bytes = total.input_bytes() - 1,
            1 => short.max_output_tokens = total.output_tokens() - 1,
            2 => short.max_pair_checks = total.pair_checks() - 1,
            _ => short.max_pair_bytes = total.pair_bytes() - 1,
        }
        let short = UntrustedEncoding::SafeBpeV1(short);
        // Each complete segment fits the narrower ceiling independently.
        // The actual CLI preparation must reject their cumulative work.
        for part in [schema, text.as_str()] {
            encoder.encode_with_policy(part.as_bytes(), &short, &mut SafeBpeWork::default(), &mut || true).unwrap();
        }
        assert!(matches!(prepare(&compiler, text.clone(), request(), &cmd.host, &short, &mut continuing()),
            Err(CandidateError::Planning)), "axis {axis}");
    }
    let mut changed = full;
    changed.max_pair_bytes -= 1;
    let changed = prepare(&compiler, text, request(), &cmd.host,
        &UntrustedEncoding::SafeBpeV1(changed), &mut continuing()).unwrap();
    assert_eq!(changed.execution_identity().prompt_digest, prepared.execution_identity().prompt_digest);
    assert_ne!(changed.execution_identity().decision_policy_digest, prepared.execution_identity().decision_policy_digest);
    assert!(changed.verify_identity(prepared.execution_identity()).is_err());
}

#[test]
fn the_compact_hard_byte_ceiling_includes_schema_even_when_document_limit_is_larger() {
    let cmd = make_command(&["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576",
        "--preparation-mib", "1024"]);
    let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let document = "x".repeat(cmd.host.max_input_bytes);
    let request = command::arguments(r#"{"type":"boolean"}"#.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    // Both original byte caps pass; their aggregate exceeds compact authority.
    command::check_schema(&request, &document, &cmd.host).unwrap();
    assert!(matches!(prepare(&compiler, document, request, &cmd.host,
        &cmd.encoding_policy(limits).unwrap(), &mut continuing()), Err(CandidateError::Planning)));
}

#[test]
fn compact_preparation_preserves_all_cancellation_causes_including_during_encoding() {
    struct Stop { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.calls >= self.stop_at).then_some(self.cause)
        }
    }
    let cmd = make_command(&["--input-encoding", "safe-bpe-v1"]);
    let (_, limits) = cmd.validate().unwrap();
    let encoding = cmd.encoding_policy(limits).unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
        RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop_at in [1, 6] {
            let request = command::arguments(r#"{"type":"string"}"#.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
            let mut stop = Stop { calls: 0, stop_at, cause };
            let expected = if matches!(cause, Timeout | Deadline) { CandidateError::Timeout } else { CandidateError::Execution };
            assert!(matches!(prepare(&compiler, "Alice works in Paris. ".repeat(8), request,
                &cmd.host, &encoding, &mut stop), Err(error) if error == expected));
            assert_eq!(stop.calls, stop_at);
        }
    }
}

#[test]
fn byte_defaults_and_exact_decimal_output_remain_compatible_with_optional_provenance() {
    let schema = " {\"type\":\"number\",\"const\":0.12345678901234567890123456789012345678}\n";
    let cmd = make_command(&[]);
    let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let request = || command::arguments(schema.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    let original = compiler.prepare_with_control(BatchDocument { id: "cli".to_owned(),
        text: "source".to_owned(), task_args: Some(request()) }, &mut continuing()).unwrap();
    for flags in [vec![], vec!["--input-encoding", "byte-fallback"]] {
        let command = make_command(&flags);
        let (_, limits) = command.validate().unwrap();
        let encoding = command.encoding_policy(limits).unwrap();
        assert_eq!(encoding, UntrustedEncoding::ByteFallback);
        let prepared = prepare(&compiler, "source".to_owned(), request(), &command.host,
            &encoding, &mut continuing()).unwrap();
        assert_eq!(prepared.execution_identity(), original.execution_identity());
        assert_eq!(prepared.planned_work(), original.planned_work());
        assert_eq!(prepared.source().input_encoding_work(), None);
    }
    let compact = make_command(&["--input-encoding", "safe-bpe-v1"]);
    let (_, limits) = compact.validate().unwrap();
    let encoding = compact.encoding_policy(limits).unwrap();
    let prepared = prepare(&compiler, "source".to_owned(), request(), &compact.host,
        &encoding, &mut continuing()).unwrap();
    assert_eq!(prepared.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    #[derive(Serialize)]
    struct ExactText<'a> { json: &'a str }
    let text = ExactText { json: "0.12345678901234567890123456789012345678" };
    let byte_wire = input_encoding::Response { response: &text, input_encoding: None };
    assert_eq!(serde_json::to_vec(&byte_wire).unwrap(), serde_json::to_vec(&text).unwrap());
    let wire = input_encoding::Response { response: &text, input_encoding: input_encoding::reported_policy(&encoding) };
    let wire = serde_json::to_value(wire).unwrap();
    assert_eq!(wire["json"].as_str(), Some(text.json));
    assert_eq!(wire["input_encoding"], serde_json::to_value(encoding).unwrap());
}
