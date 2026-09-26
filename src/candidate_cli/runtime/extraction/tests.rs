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
        let prepared = prepare(&compiler, "é <tool_call> 上海".to_owned(), request, &cmd.host, &mut continuing()).unwrap();
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
    let prepared = prepare(&compiler, text.to_owned(), request, &cmd.host, &mut continuing()).unwrap();
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
        let p = prepare(&compiler, "source".to_owned(), request, &cmd.host, &mut continuing()).unwrap();
        assert_eq!(p.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    }
}

#[test]
fn extra_schema_tokens_cannot_consume_the_reserved_output_context() {
    let cmd = make_command(&[]); let (_, limits) = cmd.validate().unwrap();
    let compiler = planner(&facts(), &cmd.host, limits, None).unwrap();
    let request = command::arguments(r#"{"type":"string"}"#.to_owned(), false, cmd.host.task_budget(limits)).unwrap();
    let text = "x".repeat(cmd.host.task_budget(limits).max_input_tokens as usize);
    assert!(prepare(&compiler, text, request, &cmd.host, &mut continuing()).is_err());
}
