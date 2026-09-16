//! Pinned planning/decoding with synthetic outputs; no native-model execution.
use super::*;
use crate::tokenizer::specials::ArchivedControlRegistries;
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
fn plans(planner: &ChatPlanner, eos: u32) -> [PreparedChat; 2] {
    [planner.plan_generate(&GenerateRequest { item_id: "a".to_owned(), sample_index: 0, prompt: "hello".to_owned(),
        generation: GenerationOptions::greedy(4, 1000, eos), budget: budget() }).unwrap(),
     planner.plan_chat(&ChatRequest { item_id: "b".to_owned(), sample_index: 0,
        messages: vec![ChatMessage { role: ChatRole::User, content: "another slightly longer prompt".to_owned() }],
        generation: GenerationOptions::greedy(4, 1000, eos), budget: budget() }).unwrap()]
}
fn requests(plans: &[PreparedChat; 2]) -> [BatchChatRequest<'_>; 2] {
    [BatchChatRequest { prepared: &plans[0], admitted_identity: plans[0].execution_identity(), slot: 0, request_seq: 1 },
     BatchChatRequest { prepared: &plans[1], admitted_identity: plans[1].execution_identity(), slot: 1, request_seq: 2 }]
}
fn outputs(requests: &[BatchChatRequest<'_>], bytes: [u8; 2], eos: u32) -> (BatchGenerationOutput, BatchGenerationRequirements) {
    let mut planned = GenerationWork::default(); let mut actual = GenerationWork::default(); let mut sequences = Vec::new(); let mut steps = 0;
    for (request, byte) in requests.iter().zip(bytes) {
        let plan = request.prepared;
        let token = plan.tokenizer.tokenizer().encode_byte_fallback_only(&[byte]).unwrap()[0];
        let positions = plan.native.prompt_tokens() as u64 + 1;
        let work = GenerationWork { forward_positions: positions, projected_logits: 2 * NANBEIGE_VOCAB_SIZE as u64, sampled_steps: 0 };
        actual = sum(actual, work).unwrap(); steps = steps.max(positions);
        planned = sum(planned, GenerationWork { forward_positions: plan.native.planned_work().forward_positions,
            projected_logits: plan.native.options().max_new_tokens as u64 * NANBEIGE_VOCAB_SIZE as u64, sampled_steps: 0 }).unwrap();
        sequences.push(GeneratedSequence { schema_version: 1, execution: BATCH_GENERATION_VERSION.to_owned(),
            numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), request_seq: request.request_seq, sample_index: 0,
            token_ids: vec![token, eos], content_bytes: vec![byte], finish_reason: GenerationFinish::Eos,
            effective_seed: None, token_logprobs: None, logprob_score_space: None, native_work: work });
    }
    (BatchGenerationOutput { sequences, group_steps: steps, planned_work: planned, actual_work: actual },
     BatchGenerationRequirements { planned_work: planned, native_payload_bytes: 0, sampler_payload_bytes: 0, output_payload_upper_bytes: 0 })
}
#[test]
fn prepared_chat_and_generate_finalize_selected_only_projections_without_relabeling() {
    let (planner, eos) = fixture(); let plans = plans(&planner, eos); let reqs = requests(&plans);
    let (raw, required) = outputs(&reqs, [b'A', b'B'], eos);
    let result = finalize(&reqs, raw, required, 100000).unwrap();
    for (index, expected_task) in ["generate-v1", "chat-v1"].iter().enumerate() {
        let BatchChatItem::Completed { result } = &result.results[index] else { panic!("valid typed result expected"); };
        assert_eq!(result.task, *expected_task); assert_eq!(result.execution, BATCH_GENERATION_VERSION);
        assert_eq!(result.request_seq, index as u64 + 1);
        assert_eq!(result.native_work.projected_logits, 2 * NANBEIGE_VOCAB_SIZE as u64);
        assert!(result.native_work.forward_positions > 2, "the full pinned prompt was still forwarded");
    }
}
#[test]
fn invalid_utf8_is_per_row_no_result_and_valid_siblings_are_retained() {
    let (planner, eos) = fixture(); let plans = plans(&planner, eos); let reqs = requests(&plans);
    let (raw, required) = outputs(&reqs, [0xff, b'B'], eos);
    let result = finalize(&reqs, raw, required, 100000).unwrap();
    assert!(matches!(&result.results[0], BatchChatItem::NoResult { request_seq: 1, reason: BatchChatNoResult::IncompleteUtf8, .. }));
    let BatchChatItem::Completed { result } = &result.results[1] else { panic!("valid sibling must survive"); };
    assert_eq!(result.content, "B");
}
#[test]
fn forged_row_routing_denominators_and_group_work_are_fatal() {
    let (planner, eos) = fixture(); let plans = plans(&planner, eos); let reqs = requests(&plans);
    let (mut raw, required) = outputs(&reqs, [b'A', b'B'], eos); raw.sequences.swap(0, 1);
    assert!(finalize(&reqs, raw, required, 100000).is_err());
    let (mut raw, required) = outputs(&reqs, [b'A', b'B'], eos);
    raw.sequences[0].native_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64;
    raw.actual_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64;
    assert!(finalize(&reqs, raw, required, 100000).is_err(), "aggregate allowance cannot excuse a wrong row denominator count");
    let (mut raw, required) = outputs(&reqs, [b'A', b'B'], eos); raw.group_steps += 1;
    assert!(finalize(&reqs, raw, required, 100000).is_err());
}
#[test]
fn complete_cohort_envelope_is_bounded_not_only_individual_content() {
    let (planner, eos) = fixture(); let plans = plans(&planner, eos); let reqs = requests(&plans);
    let (raw, required) = outputs(&reqs, [b'A', b'B'], eos);
    let result = finalize(&reqs, raw, required, 100000).unwrap();
    let exact = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    let (raw, required) = outputs(&reqs, [b'A', b'B'], eos); finalize(&reqs, raw, required, exact).unwrap();
    let (raw, required) = outputs(&reqs, [b'A', b'B'], eos);
    assert!(finalize(&reqs, raw, required, exact - 1).is_err());
}
#[test]
fn task_kv_limit_prices_entire_selected_slot_before_native_admission() {
    let (planner, eos) = fixture(); let plans = plans(&planner, eos); let reqs = requests(&plans);
    check_task_kv_limits(&[2, 3], &reqs).unwrap();
    let over = (budget().max_kv_bytes / KV_BYTES_PER_TOKEN as u64 + 1) as usize;
    assert!(check_task_kv_limits(&[over, 3], &reqs).is_err());
    assert!(check_task_kv_limits(&[2], &reqs).is_err());
}
#[test]
fn complete_row_result_limit_is_reported_without_discarding_other_rows() {
    let (planner, eos) = fixture(); let mut plans = plans(&planner, eos);
    plans[0] = planner.plan_generate(&GenerateRequest { item_id: "small".to_owned(), sample_index: 0, prompt: "hello".to_owned(),
        generation: GenerationOptions::greedy(4, 16, eos), budget: TaskBudget { max_output_bytes: 32, ..budget() } }).unwrap();
    let reqs = requests(&plans); let (raw, required) = outputs(&reqs, [b'A', b'B'], eos);
    let result = finalize(&reqs, raw, required, 100000).unwrap();
    assert!(matches!(&result.results[0], BatchChatItem::NoResult { reason: BatchChatNoResult::ResultByteLimit, .. }));
    assert!(matches!(&result.results[1], BatchChatItem::Completed { .. }));
}
