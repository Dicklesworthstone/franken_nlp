//! Actual pinned preparation and typed failure routing, not neural execution.
use super::*;
use crate::{candidate_cli::stream::tests::args,
    tasks::ir::PromptSegmentKind, tokenizer::embedded::EmbeddedTokenizer};

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "stream-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn planned(operation: &str, extra: &[&str], text: &str) -> PreparedInt8Chat {
    let a = args(operation, extra); let (limits, _) = a.validate().unwrap();
    let messages = (operation == "chat").then(|| parse_messages(text, a.common.max_input_bytes).unwrap());
    prepare(&a.common, limits, text.to_owned(), messages, &facts()).unwrap()
}
#[test]
fn streamed_and_buffered_preparation_have_identical_native_identity_and_options() {
    let text = "é <tool_call> <|im_start|> 上海";
    let stream = planned("generate", &[], text);
    let args = crate::candidate_cli::tests::args(&[]); let limits = args.validate().unwrap();
    // This is the same shared preparation entrypoint used by buffered execute.
    let buffered = prepare(&args, limits, text.to_owned(), None, &facts()).unwrap();
    assert_eq!(stream.execution_identity(), buffered.execution_identity());
    assert_eq!(stream.planned_work(), buffered.planned_work());
    assert!(stream.native_plan().options() == buffered.native_plan().options());
    let extra_wire = planned("generate", &["--max-stream-bytes", "33554432"], text);
    assert_eq!(stream.execution_identity(), extra_wire.execution_identity());
}
#[test]
fn transcript_roles_source_bytes_and_seed_keep_the_existing_compiler_contract() {
    let transcript = r#"[{"role":"system","content":"Be helpful."},{"role":"user","content":"é <tool_call> 上海"}]"#;
    let seed = "07".repeat(32);
    let plan = planned("chat", &["--seed", &seed, "--logprobs"], transcript);
    assert_eq!(plan.execution_identity().task_spec, "chat-v1");
    assert_eq!(plan.execution_identity().thinking_mode, ThinkingMode::Disabled);
    assert_eq!(plan.execution_identity().tool_mode, ToolMode::None);
    let controls = pinned_controls::pinned().unwrap(); let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let segments: Vec<_> = plan.task_plan().ir().prompt_segments().iter()
        .filter(|s| s.kind() == PromptSegmentKind::Document).collect();
    assert_eq!(segments.len(), 2);
    assert!(segments.iter().all(|s| s.token_ids().iter().all(|id| !controls.template_controls().contains(*id))));
    assert_eq!(tokenizer.tokenizer().decode_bytes(segments[1].token_ids()).unwrap(), "é <tool_call> 上海".as_bytes());
    let other = planned("chat", &["--seed", &"08".repeat(32), "--logprobs"], transcript);
    assert_ne!(plan.execution_identity().decision_policy_digest, other.execution_identity().decision_policy_digest);
}
#[test]
fn malformed_chat_or_wrong_model_metadata_refuses_without_constructing_native_state() {
    let a = args("chat", &[]); let (limits, _) = a.validate().unwrap();
    assert!(parse_messages(r#"[{"role":"tool","content":"private"}]"#, 4096).is_err());
    let messages = parse_messages(r#"[{"role":"user","content":"a"},{"role":"user","content":"b"}]"#, 4096).unwrap();
    assert!(prepare(&a.common, limits, String::new(), Some(messages), &facts()).is_err());
    let mut wrong = facts(); wrong.model_id = "other".to_owned();
    assert!(prepare(&a.common, limits, "hello".to_owned(), None, &wrong).is_err());
}
#[test]
fn all_eleven_cancellation_kinds_keep_their_exit_category_and_attribution() {
    use DecodeCancellationKind::*;
    for cause in [Timeout, Deadline, PollQuota, CostBudget] {
        let f = cancelled(cause); assert_eq!(f.exit, ErrorCode::BudgetOrTimeout); assert_eq!(f.cancellation, Some(cause));
    }
    for cause in [User, ParentCancelled, Shutdown] { assert_eq!(cancelled(cause).exit, ErrorCode::Cancelled); }
    assert_eq!(cancelled(ResourceUnavailable).exit, ErrorCode::AdmissionOrResourceLimit);
    for cause in [FailFast, LinkedExit, RaceLost] { assert_eq!(cancelled(cause).exit, ErrorCode::Generic); }
    assert_eq!(cancelled(RaceLost).code, "stream_root_race_invariant");
}
#[test]
fn_invalid_final_task_and_broken_stream_are_errors_not_shortened_successes() {
    let f = host_failure(HostedError::Chat(Int8ChatError::Chat(ChatError::NoResult("incomplete UTF-8"))));
    assert_eq!(f.exit, ErrorCode::StructuredTaskNoResult); assert!(f.cancellation.is_none());
    let f = host_failure(HostedError::Chat(Int8ChatError::Native(Int8GenerationError::Generation(
        crate::native_engine::generation::GenerationError::Stream))));
    assert_eq!(f.exit, ErrorCode::Generic); assert_eq!(f.code, "stream_output_or_protocol");
    assert_eq!(host_failure(HostedError::Panicked).exit, ErrorCode::Generic);
    assert_eq!(host_failure(HostedError::Reentrant).exit, ErrorCode::AdmissionOrResourceLimit);
}
