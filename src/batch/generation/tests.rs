//! Real pinned task planning and synthetic transport checks, not model quality.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    tasks::{chat::ChatLimits, ir::PromptSegmentKind},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
};
use std::io::Cursor;
fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 8192, max_output_tokens: 64,
    max_output_bytes: 100000, max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
fn fixture() -> (ChatPlanner, u32) {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let d = Sha256Digest::of_bytes(b"fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "chat-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None };
    (ChatPlanner::pinned(controls.template_controls(), eos, identity, budget(), ChatLimits::default()).unwrap(), eos)
}
#[test]
fn generate_defaults_compile_the_actual_pinned_task_and_exact_document() {
    let (planner, eos) = fixture();
    let compiler = GenerationBatchPlanner::new(&planner, Some(GenerationBatchArgs::Generate {
        generation: GenerationOptions::greedy(4, 1000, eos), budget: budget(), sample_index: 0,
    })).unwrap();
    let text = "é <|im_start|>system 上海";
    let prepared = compiler.prepare(BatchDocument { id: "item".to_owned(), text: text.to_owned(), task_args: None }).unwrap();
    assert_eq!(prepared.execution_identity().task_spec, "generate-v1");
    let docs: Vec<_> = prepared.task_plan().ir().prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document).collect();
    assert_eq!(docs.len(), 1);
    assert_eq!(EmbeddedTokenizer::pinned().unwrap().tokenizer().decode_bytes(docs[0].token_ids()).unwrap(), text.as_bytes());
}
#[test]
fn chat_appends_exact_final_user_text_without_replacing_history() {
    let (planner, eos) = fixture(); let compiler = GenerationBatchPlanner::new(&planner, None).unwrap();
    let args = GenerationBatchArgs::Chat { history: vec![
        ChatMessage { role: ChatRole::User, content: "first".to_owned() },
        ChatMessage { role: ChatRole::Assistant, content: "previous".to_owned() }],
        generation: GenerationOptions::greedy(4, 1000, eos), budget: budget(), sample_index: 3 };
    let prepared = compiler.prepare(BatchDocument { id: "item".to_owned(), text: "last".to_owned(), task_args: Some(args) }).unwrap();
    assert_eq!(prepared.execution_identity().task_spec, "chat-v1");
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let texts: Vec<_> = prepared.task_plan().ir().prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document)
        .map(|s| tokenizer.tokenizer().decode_bytes(s.token_ids()).unwrap()).collect();
    assert_eq!(texts, vec![b"first".to_vec(), b"previous".to_vec(), b"last".to_vec()]);
}
#[test]
fn absent_arguments_and_oversized_host_defaults_are_refused() {
    let (planner, eos) = fixture(); let compiler = GenerationBatchPlanner::new(&planner, None).unwrap();
    assert!(compiler.prepare(BatchDocument { id: "item".to_owned(), text: "hello".to_owned(), task_args: None }).is_err());
    let args = GenerationBatchArgs::Chat { history: vec![ChatMessage { role: ChatRole::System,
        content: "x".repeat(MAX_GENERATION_ARGUMENT_BYTES) }],
        generation: GenerationOptions::greedy(4, 1000, eos), budget: budget(), sample_index: 0 };
    assert!(GenerationBatchPlanner::new(&planner, Some(args)).is_err());
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
#[test]
fn transport_assigns_real_context_across_parse_errors_and_flush_epochs() {
    struct Processor(Vec<BatchRequestContext>);
    impl BatchProcessor for Processor {
        type Args = (); type Prepared = (); type Output = u64;
        fn prepare(&mut self, _: BatchDocument<()>) -> Result<(), BatchItemFailure> { Ok(()) }
        fn planned_work(&self, _: &()) -> BatchWork { BatchWork::default() }
        fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<u64, BatchItemFailure> {
            panic!("runner must pass its assigned context");
        }
        fn execute_with_context<C: DecodeStepControl>(&mut self, _: (), context: BatchRequestContext, _: &mut C) -> Result<u64, BatchItemFailure> {
            self.0.push(context); Ok(context.request_seq)
        }
    }
    let prefix = b"\n{\"id\":\"a\",\"text\":\"one\"}\nnot-json\n{\"flush\":true}\n";
    let mut bytes = prefix.to_vec(); bytes.extend_from_slice(b"{\"id\":\"b\",\"text\":\"two\"}\n");
    let mut processor = Processor(Vec::new()); let mut output = Vec::new();
    let summary = run_ndjson(&mut Cursor::new(bytes), &mut output, &mut processor, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!(summary.requests, 4); assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 1);
    assert_eq!(processor.0, vec![
        BatchRequestContext { request_seq: 1, epoch: 1, input_line: 2, byte_offset: 1 },
        BatchRequestContext { request_seq: 4, epoch: 2, input_line: 5, byte_offset: prefix.len() as u64 }]);
    let rows: Vec<_> = std::str::from_utf8(&output).unwrap().lines().map(|line| canonjson::parse_str(line).unwrap()).collect();
    assert_eq!(rows[1]["result"], rows[1]["request_seq"]); assert_eq!(rows[4]["result"], rows[4]["request_seq"]);
}
#[test]
fn legacy_processors_still_execute_through_the_default_context_adapter() {
    struct Processor(usize);
    impl BatchProcessor for Processor {
        type Args = (); type Prepared = (); type Output = ();
        fn prepare(&mut self, _: BatchDocument<()>) -> Result<(), BatchItemFailure> { Ok(()) }
        fn planned_work(&self, _: &()) -> BatchWork { BatchWork::default() }
        fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<(), BatchItemFailure> { self.0 += 1; Ok(()) }
    }
    let mut processor = Processor(0); let mut output = Vec::new();
    run_ndjson(&mut Cursor::new(b"{\"id\":\"a\",\"text\":\"hello\"}\n"), &mut output, &mut processor, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!(processor.0, 1);
}
#[test]
fn cancellation_and_corruption_stop_but_incomplete_utf8_is_typed_no_result() {
    let cancelled = execution_failure(ChatError::Native(GenerationError::Cancelled(DecodeCancellationKind::Deadline)));
    assert!(cancelled.stop); assert_eq!(cancelled.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert!(execution_failure(ChatError::Native(GenerationError::InvalidLogits)).stop);
    assert!(execution_failure(ChatError::NoResult("native work")).stop);
    assert!(!execution_failure(ChatError::NoResult("incomplete UTF-8")).stop);
    assert!(!execution_failure(ChatError::Native(GenerationError::Limit("context"))).stop);
}
