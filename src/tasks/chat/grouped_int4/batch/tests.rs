//! Synthetic completion/real pinned-finalizer tests; no model executes here.
use super::*;
use crate::{artifact::converter::PINNED_REVISION, native_engine::{
    generation::grouped_int4::{INT4_SOURCE_FORMAT, INT4_GENERATION_VERSION},
    portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
    strict_int4::{Int4RunBudget, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE},
}, tokenizer::pinned_controls};

fn fixture() -> (Vec<PreparedInt4Chat>, u32) {
    let budget = TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
    let d = Sha256Digest::of_bytes(b"batch-chat-fixture-not-model-evidence");
    let id = ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=128"),
        packing_set_digest: hash(&(INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 128_usize)).unwrap(),
        tokenizer_digest: d, template_digest: d, task_spec: "chat-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = Int4ChatPlanner::pinned(controls.template_controls(), eos, id, 128, budget, ChatLimits::default()).unwrap();
    let generate = planner.plan_generate(&GenerateRequest { item_id: "first".to_owned(), sample_index: 0,
        prompt: "Summarize this sentence.".to_owned(), generation: GenerationOptions::greedy(4, 64, eos), budget }).unwrap();
    let chat = planner.plan_chat(&ChatRequest { item_id: "second".to_owned(), sample_index: 1,
        messages: vec![ChatMessage { role: ChatRole::User, content: "Explain a longer second sentence.".to_owned() }],
        generation: GenerationOptions::greedy(4, 64, eos), budget }).unwrap();
    (vec![generate, chat], eos)
}
fn requests(plans: &[PreparedInt4Chat]) -> Vec<Int4ChatBatchRequest<'_>> {
    plans.iter().enumerate().map(|(i, prepared)| Int4ChatBatchRequest { prepared,
        admitted_identity: prepared.execution_identity(), request_seq: i as u64 + 21,
        budget: Int4GenerationBudget { native: Int4RunBudget::exact(prepared.planned_work()),
            max_kv_bytes: u64::MAX, max_sampler_bytes: prepared.native.sampler_bytes() } }).collect()
}
fn limits() -> Int4ChatBatchLimits {
    Int4ChatBatchLimits { generation: Int4BatchGenerationBudget {
        native: Int4RunBudget { max_forward_positions: u64::MAX, max_attention_pairs: u64::MAX,
            max_projection_work: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        max_native_payload_bytes: u64::MAX, max_sampler_payload_bytes: u64::MAX, max_output_payload_bytes: u64::MAX },
        max_result_bytes: 216_384 }
}
fn raw(plans: &[PreparedInt4Chat], eos: u32) -> Int4BatchGenerationOutput {
    let mut actual = Int4Work::default(); let mut planned = Int4Work::default(); let mut steps = 0;
    let mut runs = Vec::new();
    for (i, plan) in plans.iter().enumerate() {
        let token = plan.tokenizer.tokenizer().encode_byte_fallback_only(b"A").unwrap()[0];
        let positions = plan.native.prompt_tokens() + 1;
        let work = Int4Work::for_sequence(128, 0, positions, 2 * NANBEIGE_VOCAB_SIZE).unwrap();
        actual = actual.checked_add(work).unwrap(); planned = planned.checked_add(plan.planned_work()).unwrap();
        steps = steps.max(positions as u64);
        runs.push(Int4GenerationRun { schema_version: 1, sequence: GeneratedSequence { schema_version: 1,
            execution: INT4_GENERATION_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            request_seq: i as u64 + 21, sample_index: plan.sample_index, token_ids: vec![token, eos], content_bytes: b"A".to_vec(),
            finish_reason: GenerationFinish::Eos, effective_seed: None, token_logprobs: None, logprob_score_space: None,
            native_work: GenerationWork { forward_positions: positions as u64,
                projected_logits: (2 * NANBEIGE_VOCAB_SIZE) as u64, sampled_steps: 0 } }, model_work: work });
    }
    Int4BatchGenerationOutput { schema_version: 1, execution: INT4_BATCH_GENERATION_VERSION,
        runs, group_steps: steps, planned_work: planned, actual_work: actual }
}

#[test]
fn mixed_generate_and_chat_results_reuse_the_independent_single_row_finalizer() {
    let (plans, eos) = fixture(); let requests = requests(&plans);
    let expected: Vec<_> = plans.iter().zip(raw(&plans, eos).runs)
        .map(|(plan, run)| plan.finish(run).unwrap()).collect();
    let output = finish(&requests, raw(&plans, eos), limits().max_result_bytes).unwrap();
    assert!(output.results == expected);
    assert_eq!(output.execution, INT4_BATCH_GENERATION_VERSION);
    assert!(serde_json::to_vec(&output).unwrap().len() < limits().max_result_bytes as usize);
}

#[test]
fn incomplete_misrouted_or_forged_batch_work_never_yields_a_successful_prefix() {
    let (plans, eos) = fixture(); let requests = requests(&plans);
    for axis in 0..7 {
        let mut output = raw(&plans, eos);
        match axis { 0 => { output.runs.pop(); }, 1 => output.runs.swap(0, 1),
            2 => output.actual_work.projections.dot_products += 1,
            3 => output.planned_work.forward_positions += 1, 4 => output.group_steps += 1,
            5 => output.runs[1].sequence.content_bytes = b"not decoded A".to_vec(),
            _ => output.runs[1].model_work.projections.multiply_accumulates += 1 }
        assert!(finish(&requests, output, limits().max_result_bytes).is_err());
    }
}

#[test]
fn aggregate_wrapper_and_per_task_kv_caps_are_enforced_before_native_execution() {
    let (plans, _) = fixture(); let requests = requests(&plans);
    let contexts: Vec<_> = plans.iter().map(|p| p.planned_work().forward_positions as usize).collect();
    requirements(&requests, &contexts, limits()).unwrap();
    let mut too_small = limits(); too_small.max_result_bytes -= 1;
    assert!(requirements(&requests, &contexts, too_small).is_err());
    // The callers supplied unlimited KV, but the tasks' 1 GiB ceiling remains
    // effective for an 8192-token allocated cache (not merely its live prefix).
    assert!(requirements(&requests, &[8192, 8192], limits()).is_err());
    assert!(requirements(&[], &[], limits()).is_err());
    assert!(wrapper_bound([u64::MAX].into_iter()).is_err());
}

#[test]
fn changed_last_admission_or_duplicate_event_address_is_not_silently_rebound() {
    let (plans, _) = fixture(); let mut requests = requests(&plans);
    let contexts: Vec<_> = plans.iter().map(|p| p.planned_work().forward_positions as usize).collect();
    requests[1].request_seq = requests[0].request_seq;
    assert!(requirements(&requests, &contexts, limits()).is_err());
    requests[1].request_seq = 22;
    let mut other = plans[1].execution_identity().clone(); other.prompt_digest = Sha256Digest::of_bytes(b"other");
    requests[1].admitted_identity = &other;
    assert!(requirements(&requests, &contexts, limits()).is_err());
}
