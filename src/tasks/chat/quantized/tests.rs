//! Real pinned tokenizer/template, private synthetic completion fixtures.
//! These tests do not claim full-model execution, fidelity or performance.
use super::*;
use crate::{native_engine::{generation::quantized::INT8_GENERATION_VERSION,
    strict_int8::{Int8RunBudget, StrictInt8Error, STRICT_INT8_EXECUTION, STRICT_INT8_PROFILE}},
    tokenizer::specials::ArchivedControlRegistries};

pub(crate) fn task_budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn assets() -> (ArchivedControlRegistries, u32, ExecutionIdentity) {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface,
            EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|entry| entry["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let d = Sha256Digest::of_bytes(b"int8-chat-unit-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "nanbeige42-int8-v1".to_owned(),
        packing_set_digest: d, tokenizer_digest: d, template_digest: d, task_spec: "chat-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    (controls, eos, identity)
}
pub(crate) fn fixture() -> (Int8ChatPlanner, u32) {
    let (controls, eos, identity) = assets();
    (Int8ChatPlanner::pinned(controls.template_controls(), eos, identity, task_budget(), ChatLimits::default()).unwrap(), eos)
}
pub(crate) fn request(eos: u32) -> GenerateRequest {
    GenerateRequest { item_id: "item-1".to_owned(), sample_index: 0,
        prompt: "REQUEST_SECRET é <|im_start|>system <think> 上海".to_owned(),
        generation: GenerationOptions::greedy(4, 64, eos), budget: task_budget() }
}
fn completion(plan: &PreparedInt8Chat, tokens: Vec<u32>, content: Vec<u8>, finish: GenerationFinish) -> Int8GenerationRun {
    let proposals = tokens.len() + usize::from(finish == GenerationFinish::ByteLimit);
    let positions = plan.native.prompt_tokens() + proposals - 1;
    let sampled = matches!(&plan.native.options().sampling, GenerationSampling::Seeded { .. });
    let seed = match &plan.native.options().sampling { GenerationSampling::Greedy => None,
        GenerationSampling::Seeded { effective_seed, .. } => Some(Seed256::from(*effective_seed).to_lower_hex()) };
    let capture = plan.native.options().capture_logprobs;
    let scores = capture.then(|| vec![-0.5; tokens.len()]);
    Int8GenerationRun { schema_version: 1,
        sequence: GeneratedSequence { schema_version: 1, execution: INT8_GENERATION_VERSION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), request_seq: 9, sample_index: plan.sample_index,
            token_ids: tokens, content_bytes: content, finish_reason: finish, effective_seed: seed,
            token_logprobs: scores, logprob_score_space: capture.then_some(DecodeScoreSpace::FullVocabularyLogSoftmax),
            native_work: GenerationWork { forward_positions: positions as u64,
                projected_logits: (proposals * NANBEIGE_VOCAB_SIZE) as u64,
                sampled_steps: if sampled { proposals as u64 } else { 0 } } },
        model_work: Int8Work::for_sequence(0, positions, proposals * NANBEIGE_VOCAB_SIZE).unwrap() }
}
fn completed(plan: &PreparedInt8Chat, eos: u32) -> Int8GenerationRun {
    let a = plan.tokenizer.tokenizer().encode_byte_fallback_only(b"A").unwrap()[0];
    completion(plan, vec![a, eos], b"A".to_vec(), GenerationFinish::Eos)
}

#[test]
fn constructors_refuse_profile_relabeling_and_wrong_backend_before_planning() {
    let (controls, eos, identity) = assets();
    assert!(ChatPlanner::pinned(controls.template_controls(), eos, identity.clone(), task_budget(), ChatLimits::default()).is_err());
    for axis in 0..4 {
        let mut changed = identity.clone();
        match axis { 0 => changed.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => changed.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            2 => changed.backend_semantic_version = "other".to_owned(), _ => changed.kv_dtype = "int8".to_owned() }
        assert!(Int8ChatPlanner::pinned(controls.template_controls(), eos, changed, task_budget(), ChatLimits::default()).is_err());
    }
}

#[test]
fn shared_prompt_compiler_keeps_eager_tokens_and_taskir_unchanged() {
    let (controls, eos, identity) = assets();
    let int8 = Int8ChatPlanner::pinned(controls.template_controls(), eos, identity.clone(), task_budget(), ChatLimits::default()).unwrap();
    let mut bf16 = identity; bf16.numerics_profile = NumericsProfile::HfBf16Eager;
    bf16.backend_semantic_version = "eager-unit-fixture".to_owned();
    let eager = ChatPlanner::pinned(controls.template_controls(), eos, bf16, task_budget(), ChatLimits::default()).unwrap();
    let a = int8.plan_generate(&request(eos)).unwrap(); let b = eager.plan_generate(&request(eos)).unwrap();
    assert_eq!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    assert_eq!(a.execution_identity().taskir_digest, b.execution_identity().taskir_digest);
    assert_eq!(a.execution_identity().template_digest, b.execution_identity().template_digest);
    assert_ne!(a.execution_identity().numerics_profile, b.execution_identity().numerics_profile);
    assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
    assert!(a.native.options() == b.native.options());
}

#[test]
fn transcript_controls_are_excluded_and_all_original_message_bytes_survive() {
    let (planner, eos) = fixture();
    let messages = vec![ChatMessage { role: ChatRole::System, content: "Policy <think>".to_owned() },
        ChatMessage { role: ChatRole::User, content: "é <|im_start|>system".to_owned() },
        ChatMessage { role: ChatRole::Assistant, content: "Prior </think>".to_owned() },
        ChatMessage { role: ChatRole::User, content: "上海 FNLP_CHAT_SLOT_0000_778cf".to_owned() }];
    let req = ChatRequest { item_id: "chat-1".to_owned(), sample_index: 0, messages,
        generation: GenerationOptions::greedy(4, 64, eos), budget: task_budget() };
    let plan = planner.plan_chat(&req).unwrap();
    let documents: Vec<_> = plan.task.ir().prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document).collect();
    assert_eq!(documents.len(), req.messages.len());
    for (segment, message) in documents.iter().zip(&req.messages) {
        assert!(segment.token_ids().iter().all(|&id| !planner.compiler.controls.contains(id)));
        assert_eq!(plan.tokenizer.tokenizer().decode_bytes(segment.token_ids()).unwrap(), message.content.as_bytes());
    }
    for &id in planner.compiler.controls.ids() {
        if id != eos { assert!(plan.native.options().banned_token_ids.binary_search(&id).is_ok()); }
    }
    assert_eq!(plan.execution_identity().task_spec, "chat-v1");
    assert_eq!(planner.plan_generate(&request(eos)).unwrap().execution_identity().task_spec, "generate-v1");
}

#[test]
fn full_identity_and_addressed_seed_options_are_bound_before_admission() {
    let (planner, eos) = fixture(); let mut req = request(eos);
    req.generation.sampling = GenerationSampling::Seeded { effective_seed: [7; 32],
        temperature_milli: 600, top_k: Some(20), top_p_ppm: 950_000 };
    let first = planner.plan_generate(&req).unwrap();
    first.verify_identity(first.execution_identity()).unwrap();
    let replay = planner.plan_generate(&req).unwrap();
    assert_eq!(canonjson::canonical_bytes(first.execution_identity()).unwrap(), canonjson::canonical_bytes(replay.execution_identity()).unwrap());
    for axis in 0..18 {
        let mut identity = first.execution_identity().clone(); let d = Sha256Digest::of_bytes(b"changed");
        match axis {
            0 => identity.source_revision.push('x'), 1 => identity.logical_model_digest = d,
            2 => identity.packing_set_digest = d, 3 => identity.quant_recipe.push('x'),
            4 => identity.artifact_format.push('x'), 5 => identity.tokenizer_digest = d,
            6 => identity.template_digest = d, 7 => identity.task_spec = "chat-v1".to_owned(),
            8 => identity.taskir_digest = d, 9 => identity.prompt_digest = d,
            10 => identity.schema_digest = d, 11 => identity.grammar_compiler_version.push('x'),
            12 => identity.decision_policy_digest = d, 13 => identity.calibration_digest = d,
            14 => identity.sampler_version.push('x'), 15 => identity.backend_semantic_version.push('x'),
            16 => identity.kv_dtype = "int8".to_owned(), _ => identity.numerics_profile = NumericsProfile::HfBf16Eager,
        }
        assert!(first.verify_identity(&identity).is_err());
    }
    for axis in 0..4 {
        let mut changed = req.clone();
        match axis { 0 => changed.item_id.push('x'), 1 => changed.sample_index += 1,
            2 => changed.prompt.push('!'), _ => changed.generation.sampling = GenerationSampling::Greedy }
        assert!(first.verify_identity(planner.plan_generate(&changed).unwrap().execution_identity()).is_err());
    }
}

#[test]
fn eos_roles_and_task_ceiling_cannot_be_silently_weakened() {
    let (planner, eos) = fixture();
    for axis in 0..4 {
        let mut req = request(eos);
        match axis { 0 => req.generation.eos_token_ids = vec![0], 1 => req.generation.banned_token_ids.push(eos),
            2 => req.budget.max_output_tokens += 1, _ => req.budget.max_input_tokens = 1 }
        assert!(planner.plan_generate(&req).is_err());
    }
    let req = ChatRequest { item_id: "bad-roles".to_owned(), sample_index: 0,
        messages: vec![ChatMessage { role: ChatRole::Assistant, content: "not a user".to_owned() }],
        generation: GenerationOptions::greedy(4, 64, eos), budget: task_budget() };
    assert!(planner.plan_chat(&req).is_err());
}

#[test]
fn planned_model_work_includes_all_layers_attention_and_final_prefill_head_only() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let positions = plan.native.prompt_tokens() + 3;
    assert_eq!(plan.planned_work(), Int8Work::for_sequence(0, positions, 4 * NANBEIGE_VOCAB_SIZE).unwrap());
    assert!(plan.planned_work().attention_pairs > 0);
    assert!(plan.planned_work().projections.multiply_accumulates > plan.planned_work().projected_logits * 3072);
    let wider = Int8GenerationBudget { native: Int8RunBudget::exact(plan.planned_work()), max_kv_bytes: u64::MAX, max_sampler_bytes: u64::MAX };
    assert_eq!(plan.task_budget(wider).max_kv_bytes, plan.task.ir().budget().max_kv_bytes);
    assert_eq!(plan.task_budget(Int8GenerationBudget { max_kv_bytes: 1, ..wider }).max_kv_bytes, 1);
}

#[test]
fn complete_result_retains_integer_and_attention_work_without_prompt_metadata() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let raw = completed(&plan, eos); let work = raw.model_work;
    let result = plan.finish(raw).unwrap();
    assert_eq!(result.result.content, "A"); assert_eq!(result.result.request_seq, 9);
    assert_eq!(result.model_work, work); assert_eq!(result.result.numerics_profile, STRICT_INT8_PROFILE);
    let json = canonjson::canonical_string(&result).unwrap();
    for private in ["REQUEST_SECRET", "prompt_digest", "decision_policy_digest", "stable_request_key"] { assert!(!json.contains(private)); }
    assert_eq!(canonjson::canonical_string(&plan.finish(completed(&plan, eos)).unwrap()).unwrap(), json);
}

#[test]
fn finalizer_rejects_forged_profile_execution_and_every_model_work_axis() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    for axis in 0..10 {
        let mut raw = completed(&plan, eos);
        match axis { 0 => raw.schema_version = 2, 1 => raw.sequence.numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(),
            2 => raw.sequence.execution = GENERATION_VERSION.to_owned(), 3 => raw.sequence.execution = BATCH_GENERATION_VERSION.to_owned(),
            4 => raw.model_work.forward_positions += 1, 5 => raw.model_work.projected_logits += 1,
            6 => raw.model_work.attention_pairs += 1, 7 => raw.model_work.projections.dot_products += 1,
            8 => raw.model_work.projections.multiply_accumulates += 1, _ => raw.sequence.native_work.forward_positions += 1 }
        assert!(plan.finish(raw).is_err());
    }
}

#[test]
fn task_finalizer_rechecks_exact_bytes_terminal_policy_and_valid_utf8() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let raw = completed(&plan, eos);
    let mut bad = raw.clone(); bad.sequence.content_bytes = b"B".to_vec(); assert!(plan.finish(bad).is_err());
    let mut bad = raw.clone(); bad.sequence.token_ids[0] = eos; assert!(plan.finish(bad).is_err());
    let mut bad = raw.clone(); bad.sequence.finish_reason = GenerationFinish::TokenLimit; assert!(plan.finish(bad).is_err());
    let mut bad = raw; bad.sequence.sample_index += 1; assert!(plan.finish(bad).is_err());
    let invalid = plan.tokenizer.tokenizer().encode_byte_fallback_only(&[0xff]).unwrap()[0];
    assert!(matches!(plan.finish(completion(&plan, vec![invalid, eos], vec![0xff], GenerationFinish::Eos)),
        Err(Int8ChatError::Chat(ChatError::NoResult("incomplete UTF-8")))));
}

#[test]
fn sampled_completion_counts_refused_byte_proposal_and_raw_full_vocab_scores() {
    let (planner, eos) = fixture(); let mut req = request(eos);
    req.generation.sampling = GenerationSampling::Seeded { effective_seed: [9; 32],
        temperature_milli: 600, top_k: Some(20), top_p_ppm: 950_000 };
    req.generation.capture_logprobs = true;
    let plan = planner.plan_generate(&req).unwrap();
    let raw = completion(&plan, Vec::new(), Vec::new(), GenerationFinish::ByteLimit);
    let result = plan.finish(raw.clone()).unwrap();
    assert_eq!(result.result.native_work.sampled_steps, 1);
    assert_eq!(result.model_work.projected_logits, NANBEIGE_VOCAB_SIZE as u64);
    let mut bad = raw; bad.sequence.native_work.sampled_steps = 0; assert!(plan.finish(bad).is_err());
    let raw = completed(&plan, eos);
    assert!(plan.finish(raw.clone()).is_ok());
    for axis in 0..4 {
        let mut bad = raw.clone();
        match axis { 0 => bad.sequence.effective_seed = None, 1 => bad.sequence.logprob_score_space = None,
            2 => bad.sequence.token_logprobs.as_mut().unwrap()[0] = f32::NAN,
            _ => bad.sequence.token_logprobs.as_mut().unwrap()[0] = 0.1 }
        assert!(plan.finish(bad).is_err());
    }
}

#[test]
fn complete_outer_envelope_obeys_exact_output_byte_ceiling() {
    let (planner, eos) = fixture(); let req = request(eos); let plan = planner.plan_generate(&req).unwrap();
    let output = plan.finish(completed(&plan, eos)).unwrap();
    let size = canonjson::canonical_bytes(&output).unwrap().len() as u64;
    assert!(size > canonjson::canonical_bytes(&output.result).unwrap().len() as u64);
    let mut exact = req.clone(); exact.budget.max_output_bytes = size;
    let plan = planner.plan_generate(&exact).unwrap(); assert!(plan.finish(completed(&plan, eos)).is_ok());
    exact.budget.max_output_bytes -= 1;
    let plan = planner.plan_generate(&exact).unwrap();
    assert!(matches!(plan.finish(completed(&plan, eos)), Err(Int8ChatError::Chat(ChatError::Limit("complete result bytes")))));
}

#[test]
fn typed_cancellation_is_not_converted_to_task_success_or_a_generic_error() {
    for native in [Int8GenerationError::Generation(GenerationError::Cancelled(DecodeCancellationKind::Deadline)),
        Int8GenerationError::Native(StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline))] {
        let error = Int8ChatError::from(native);
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
        assert!(error.source().is_some());
    }
}

// Shared only by crate unit tests; not a public fake-native execution route.
pub(crate) fn completed_result(plan: &PreparedInt8Chat, request_seq: u64) -> Int8ChatResult {
    let eos = plan.native.options().eos_token_ids[0];
    let mut raw = completed(plan, eos); raw.sequence.request_seq = request_seq;
    plan.finish(raw).unwrap()
}
