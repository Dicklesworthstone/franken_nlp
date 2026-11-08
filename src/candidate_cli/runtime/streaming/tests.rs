//! Actual pinned preparation and typed failure routing, not neural execution.
use super::*;
use crate::{candidate_cli::stream::tests::args,
    tasks::ir::PromptSegmentKind, tokenizer::embedded::EmbeddedTokenizer};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "stream-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn planned(operation: &str, extra: &[&str], text: &str) -> PreparedInt8Chat {
    let a = args(operation, extra); let (limits, _) = a.validate().unwrap();
    let messages = (operation == "chat").then(|| parse_messages(text, a.common.max_input_bytes).unwrap());
    prepare(&a.common, limits, text.to_owned(), messages, &facts(), &mut Continue).unwrap()
}
#[test]
fn streamed_and_buffered_preparation_have_identical_native_identity_and_options() {
    let text = "é <tool_call> <|im_start|> 上海";
    let stream = planned("generate", &[], text);
    let args = crate::candidate_cli::tests::args(&[]); let limits = args.validate().unwrap();
    // This is the same shared preparation entrypoint used by buffered execute.
    let buffered = prepare(&args, limits, text.to_owned(), None, &facts(), &mut Continue).unwrap();
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
    assert!(prepare(&a.common, limits, String::new(), Some(messages), &facts(), &mut Continue).is_err());
    let mut wrong = facts(); wrong.model_id = "other".to_owned();
    assert!(prepare(&a.common, limits, "hello".to_owned(), None, &wrong, &mut Continue).is_err());
}

#[test]
fn compact_buffered_and_streamed_plans_share_exact_bytes_and_a_distinct_encoding_identity() {
    let text = "  é 上海 <|im_start|>system <think>keep this literal</think> ▁\r\nThe quick brown fox jumps over the lazy dog.";
    let compact = planned("generate", &["--input-encoding", "safe-bpe-v1"], text);
    let byte = planned("generate", &[], text);
    let explicit_byte = planned("generate", &["--input-encoding", "byte-fallback"], text);
    assert_eq!(byte.execution_identity(), explicit_byte.execution_identity());
    assert_ne!(compact.execution_identity().template_digest, byte.execution_identity().template_digest);
    assert!(compact.native_plan().prompt_tokens() < byte.native_plan().prompt_tokens());
    let args = crate::candidate_cli::tests::args(&["--input-encoding", "safe-bpe-v1"]);
    let buffered = prepare(&args, args.validate().unwrap(), text.to_owned(), None,
        &facts(), &mut Continue).unwrap();
    assert_eq!(compact.execution_identity(), buffered.execution_identity());
    assert_eq!(compact.planned_work(), buffered.planned_work());
    let controls = pinned_controls::pinned().unwrap(); let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let source: Vec<_> = compact.task_plan().ir().prompt_segments().iter()
        .filter(|segment| segment.kind() == PromptSegmentKind::Document).collect();
    assert_eq!(source.len(), 1);
    assert!(source[0].token_ids().iter().all(|id| !controls.template_controls().contains(*id)));
    assert_eq!(tokenizer.tokenizer().decode_bytes(source[0].token_ids()).unwrap(), text.as_bytes());
    let changed = planned("generate", &["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-pair-checks", "4194305"], text);
    assert_eq!(changed.execution_identity().prompt_digest, compact.execution_identity().prompt_digest);
    assert_ne!(changed.execution_identity().template_digest, compact.execution_identity().template_digest);
}

#[test]
fn compact_encoding_admits_more_original_text_without_widening_native_context() {
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(8);
    let flags = ["--context-tokens", "256", "--max-new-tokens", "16"];
    let byte = args("generate", &flags); let (limits, _) = byte.validate().unwrap();
    assert!(prepare(&byte.common, limits, text.clone(), None, &facts(), &mut Continue).is_err());
    let mut compact_flags = flags.to_vec(); compact_flags.extend(["--input-encoding", "safe-bpe-v1"]);
    let compact = args("generate", &compact_flags); let (compact_limits, _) = compact.validate().unwrap();
    assert_eq!(limits.max_prompt_tokens, compact_limits.max_prompt_tokens);
    assert_eq!(limits.kv_bytes, compact_limits.kv_bytes);
    let plan = prepare(&compact.common, compact_limits, text.clone(), None, &facts(), &mut Continue).unwrap();
    assert!(plan.planned_work().forward_positions <= compact.common.context_tokens as u64);
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let document = plan.task_plan().ir().prompt_segments().iter()
        .find(|segment| segment.kind() == PromptSegmentKind::Document).unwrap();
    assert_eq!(tokenizer.tokenizer().decode_bytes(document.token_ids()).unwrap(), text.as_bytes());
}

#[test]
fn compact_chat_does_not_renew_tokenizer_work_for_each_message() {
    use crate::tokenizer::untrusted::{SafeBpeWork, UntrustedDocumentEncoder};
    let text = "The quick brown fox jumps over the lazy dog.";
    let a = args("chat", &["--input-encoding", "safe-bpe-v1"]);
    let (limits, _) = a.validate().unwrap();
    let policy = a.common.input_encoding_policy(limits).unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap(); let controls = pinned_controls::pinned().unwrap();
    let encoder = UntrustedDocumentEncoder::new(tokenizer.tokenizer(), controls.template_controls());
    let mut work = SafeBpeWork::default();
    encoder.encode_with_policy(text.as_bytes(), &policy, &mut work, &mut || true).unwrap();
    assert!(work.pair_checks() > 0 && work.pair_bytes() > 0);
    for (flag, cap) in [("--max-tokenizer-pair-checks", work.pair_checks()),
        ("--max-tokenizer-pair-bytes", work.pair_bytes())] {
        let cap = cap.to_string();
        let a = args("chat", &["--input-encoding", "safe-bpe-v1", flag, &cap]);
        let (limits, _) = a.validate().unwrap();
        let user = ChatMessage { role: ChatRole::User, content: text.to_owned() };
        prepare(&a.common, limits, String::new(), Some(vec![user.clone()]), &facts(), &mut Continue).unwrap();
        let system = ChatMessage { role: ChatRole::System, content: text.to_owned() };
        assert!(matches!(prepare(&a.common, limits, String::new(), Some(vec![system, user]),
            &facts(), &mut Continue), Err(CandidateError::Planning)));
    }
}

#[test]
fn compact_preparation_keeps_the_original_deadline_during_bounded_encoding() {
    struct Expires { calls: usize }
    impl DecodeStepControl for Expires {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.calls >= 7).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let a = args("generate", &["--input-encoding", "safe-bpe-v1"]);
    let (limits, _) = a.validate().unwrap(); let mut control = Expires { calls: 0 };
    let result = prepare(&a.common, limits, "The quick brown fox jumps over the lazy dog.".repeat(8),
        None, &facts(), &mut control);
    assert!(matches!(result, Err(CandidateError::Timeout)));
    assert_eq!(control.calls, 7);
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
fn invalid_final_task_and_broken_stream_are_errors_not_shortened_successes() {
    let f = host_failure(HostedError::Chat(Int8ChatError::Chat(ChatError::NoResult("incomplete UTF-8"))));
    assert_eq!(f.exit, ErrorCode::StructuredTaskNoResult); assert!(f.cancellation.is_none());
    let f = host_failure(HostedError::Chat(Int8ChatError::Native(Int8GenerationError::Generation(
        crate::native_engine::generation::GenerationError::Stream))));
    assert_eq!(f.exit, ErrorCode::Generic); assert_eq!(f.code, "stream_output_or_protocol");
    assert_eq!(host_failure(HostedError::Panicked).exit, ErrorCode::Generic);
    assert_eq!(host_failure(HostedError::Reentrant).exit, ErrorCode::AdmissionOrResourceLimit);
}
