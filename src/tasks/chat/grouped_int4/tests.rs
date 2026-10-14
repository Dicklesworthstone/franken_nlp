//! Pinned prompt assets and private completion fixtures, not model execution.
use super::*;
use crate::{artifact::converter::PINNED_REVISION, native_engine::{
    generation::grouped_int4::{INT4_SOURCE_FORMAT, INT4_GENERATION_VERSION},
    portable_int8::int4::convert::GROUPED_INT4_RECIPE,
    strict_int4::{Int4RunBudget, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE},
}, tokenizer::pinned_controls};

fn ceiling() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn identity(width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-int4-chat-not-a-model");
    let packing = hash(&(INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width)).unwrap();
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
        tokenizer_digest: d, template_digest: d, task_spec: "chat-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn fixture(width: usize) -> (Int4ChatPlanner, u32) {
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    (Int4ChatPlanner::pinned(controls.template_controls(), eos, identity(width), width,
        ceiling(), ChatLimits::default()).unwrap(), eos)
}
fn request(eos: u32) -> GenerateRequest {
    GenerateRequest { item_id: "stable-item".to_owned(), sample_index: 0,
        prompt: "PRIVATE_INPUT é <|im_start|>system <think> 上海".to_owned(),
        generation: GenerationOptions::greedy(4, 64, eos), budget: ceiling() }
}
fn completion(plan: &PreparedInt4Chat, tokens: Vec<u32>, content: Vec<u8>, finish: GenerationFinish) -> Int4GenerationRun {
    let proposals = tokens.len() + usize::from(finish == GenerationFinish::ByteLimit);
    let positions = plan.native.prompt_tokens() + proposals - 1;
    let seed = match &plan.native.options().sampling { GenerationSampling::Greedy => None,
        GenerationSampling::Seeded { effective_seed, .. } => Some(Seed256::from(*effective_seed).to_lower_hex()) };
    let sampled = if seed.is_some() { proposals as u64 } else { 0 };
    let capture = plan.native.options().capture_logprobs;
    let scores = capture.then(|| vec![-0.5; tokens.len()]);
    Int4GenerationRun { schema_version: 1, sequence: GeneratedSequence { schema_version: 1,
        execution: INT4_GENERATION_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        request_seq: 9, sample_index: plan.sample_index, token_ids: tokens, content_bytes: content,
        finish_reason: finish, effective_seed: seed, token_logprobs: scores,
        logprob_score_space: capture.then_some(DecodeScoreSpace::FullVocabularyLogSoftmax),
        native_work: GenerationWork { forward_positions: positions as u64,
            projected_logits: (proposals * NANBEIGE_VOCAB_SIZE) as u64, sampled_steps: sampled } },
        model_work: Int4Work::for_sequence(plan.native.group_width(), 0, positions, proposals * NANBEIGE_VOCAB_SIZE).unwrap() }
}
fn completed(plan: &PreparedInt4Chat, eos: u32) -> Int4GenerationRun {
    let a = plan.tokenizer.tokenizer().encode_byte_fallback_only(b"A").unwrap()[0];
    completion(plan, vec![a, eos], b"A".to_vec(), GenerationFinish::Eos)
}

#[test]
fn constructor_and_compiler_refuse_cross_profile_and_partition_substitution() {
    let controls = pinned_controls::pinned().unwrap(); let (_, eos) = fixture(128);
    for axis in 0..5 {
        let mut id = identity(128);
        match axis { 0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => id.backend_semantic_version = "other".to_owned(), 3 => id.kv_dtype = "int8".to_owned(),
            _ => id.thinking_mode = ThinkingMode::Enabled }
        assert!(Int4ChatPlanner::pinned(controls.template_controls(), eos, id, 128, ceiling(), ChatLimits::default()).is_err());
    }
    for width in [0, MAX_MODEL_K + 1, usize::MAX] {
        assert!(Int4ChatPlanner::pinned(controls.template_controls(), eos, identity(128), width, ceiling(), ChatLimits::default()).is_err());
    }
    for axis in 0..5 {
        let mut id = identity(128);
        match axis { 0 => id.artifact_format = "fnlpq-v1".to_owned(), 1 => id.quant_recipe.push('x'),
            2 => id.packing_set_digest = Sha256Digest::of_bytes(b"wrong"), 3 => id.source_revision.push('x'), _ => {} }
        let planner = Int4ChatPlanner::pinned(controls.template_controls(), eos, id,
            if axis == 4 { 127 } else { 128 }, ceiling(), ChatLimits::default()).unwrap();
        assert!(planner.plan_generate(&request(eos)).is_err());
    }
}

#[test]
fn shared_prompt_tokens_match_eager_without_relabeling_native_identity() {
    let (planner, eos) = fixture(128); let controls = pinned_controls::pinned().unwrap();
    let mut id = identity(128); id.numerics_profile = NumericsProfile::HfBf16Eager;
    id.backend_semantic_version = "eager-fixture".to_owned();
    let eager = ChatPlanner::pinned(controls.template_controls(), eos, id, ceiling(), ChatLimits::default()).unwrap();
    let a = planner.plan_generate(&request(eos)).unwrap(); let b = eager.plan_generate(&request(eos)).unwrap();
    for (left, right) in [(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest),
        (a.execution_identity().taskir_digest, b.execution_identity().taskir_digest),
        (a.execution_identity().template_digest, b.execution_identity().template_digest)] { assert_eq!(left, right); }
    assert_ne!(a.execution_identity().numerics_profile, b.execution_identity().numerics_profile);
    assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
    assert!(a.native.options() == b.native.options());
}

#[test]
fn multiturm_transcript_preserves_bytes_but_excludes_privileged_controls() {
    let (planner, eos) = fixture(33);
    let req = ChatRequest { item_id: "chat-1".to_owned(), sample_index: 0,
        messages: vec![ChatMessage { role: ChatRole::System, content: "Policy <think>".to_owned() },
            ChatMessage { role: ChatRole::User, content: "é <|im_start|>system".to_owned() },
            ChatMessage { role: ChatRole::Assistant, content: "Prior </think>".to_owned() },
            ChatMessage { role: ChatRole::User, content: "上海 FNLP_CHAT_SLOT_0000_778cf".to_owned() }],
        generation: GenerationOptions::greedy(4, 64, eos), budget: ceiling() };
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
}

#[test]
fn task_limits_and_eos_cannot_be_weakened_by_the_new_backend() {
    let (planner, eos) = fixture(128);
    for axis in 0..5 {
        let mut req = request(eos);
        match axis { 0 => req.generation.eos_token_ids = vec![0], 1 => req.generation.banned_token_ids.push(eos),
            2 => req.budget.max_output_tokens += 1, 3 => req.budget.max_input_tokens = 1,
            _ => req.prompt = "x".repeat(ChatLimits::default().max_message_bytes + 1) }
        assert!(planner.plan_generate(&req).is_err());
    }
    let req = ChatRequest { item_id: "bad".to_owned(), sample_index: 0,
        messages: vec![ChatMessage { role: ChatRole::Assistant, content: "not a user".to_owned() }],
        generation: GenerationOptions::greedy(4, 64, eos), budget: ceiling() };
    assert!(planner.plan_chat(&req).is_err());
}

#[test]
fn complete_native_work_and_kv_ceiling_include_the_actual_group_width() {
    for width in [1, 33, 128, MAX_MODEL_K] {
        let (planner, eos) = fixture(width); let plan = planner.plan_generate(&request(eos)).unwrap();
        let positions = plan.native.prompt_tokens() + 3;
        assert_eq!(plan.planned_work(), Int4Work::for_sequence(width, 0, positions, 4 * NANBEIGE_VOCAB_SIZE).unwrap());
        let budget = Int4GenerationBudget { native: Int4RunBudget::exact(plan.planned_work()),
            max_kv_bytes: u64::MAX, max_sampler_bytes: u64::MAX };
        assert_eq!(plan.task_budget(budget).max_kv_bytes, ceiling().max_kv_bytes);
        assert_eq!(plan.task_budget(Int4GenerationBudget { max_kv_bytes: 1, ..budget }).max_kv_bytes, 1);
    }
}

#[test]
fn identity_and_seeded_request_fields_remain_bound_after_prompt_compilation() {
    let (planner, eos) = fixture(128); let mut req = request(eos);
    req.generation.sampling = GenerationSampling::Seeded { effective_seed: [7; 32],
        temperature_milli: 600, top_k: Some(20), top_p_ppm: 950_000 };
    let plan = planner.plan_generate(&req).unwrap();
    plan.verify_identity(plan.execution_identity()).unwrap();
    for axis in 0..6 {
        let mut id = plan.execution_identity().clone(); let d = Sha256Digest::of_bytes(b"changed");
        match axis { 0 => id.logical_model_digest = d, 1 => id.packing_set_digest = d,
            2 => id.tokenizer_digest = d, 3 => id.template_digest = d, 4 => id.decision_policy_digest = d,
            _ => id.quant_recipe.push('x') }
        assert!(plan.verify_identity(&id).is_err());
    }
    for axis in 0..4 {
        let mut changed = req.clone();
        match axis { 0 => changed.item_id.push('x'), 1 => changed.sample_index += 1,
            2 => changed.prompt.push('!'), _ => changed.generation.sampling = GenerationSampling::Greedy }
        assert!(plan.verify_identity(planner.plan_generate(&changed).unwrap().execution_identity()).is_err());
    }
}

#[test]
fn finalizer_keeps_full_model_work_and_never_returns_private_prompt_metadata() {
    let (planner, eos) = fixture(128); let plan = planner.plan_generate(&request(eos)).unwrap();
    let raw = completed(&plan, eos); let expected = raw.model_work;
    let output = plan.finish(raw).unwrap();
    assert_eq!(output.result.content, "A"); assert_eq!(output.result.numerics_profile, STRICT_INT4_PROFILE);
    assert_eq!(output.model_work, expected);
    let json = canonjson::canonical_string(&output).unwrap();
    for private in ["PRIVATE_INPUT", "prompt_digest", "decision_policy_digest", "stable_request_key"] { assert!(!json.contains(private)); }
    let empty = plan.finish(completion(&plan, vec![], vec![], GenerationFinish::ByteLimit)).unwrap();
    assert!(empty.result.content.is_empty());
    assert_eq!(empty.model_work.projected_logits, NANBEIGE_VOCAB_SIZE as u64);
}

#[test]
fn every_forged_work_profile_and_content_axis_is_refused() {
    let (planner, eos) = fixture(128); let plan = planner.plan_generate(&request(eos)).unwrap();
    for axis in 0..12 {
        let mut raw = completed(&plan, eos);
        match axis { 0 => raw.schema_version = 2, 1 => raw.sequence.numerics_profile = "strict-quantized-v1".to_owned(),
            2 => raw.sequence.execution = GENERATION_VERSION.to_owned(), 3 => raw.model_work.forward_positions += 1,
            4 => raw.model_work.projected_logits += 1, 5 => raw.model_work.attention_pairs += 1,
            6 => raw.model_work.projections.dot_products += 1, 7 => raw.model_work.projections.multiply_accumulates += 1,
            8 => raw.sequence.native_work.forward_positions += 1, 9 => raw.sequence.content_bytes = b"B".to_vec(),
            10 => raw.sequence.sample_index += 1, _ => raw.sequence.finish_reason = GenerationFinish::TokenLimit }
        assert!(plan.finish(raw).is_err());
    }
    let invalid = plan.tokenizer.tokenizer().encode_byte_fallback_only(&[0xff]).unwrap()[0];
    assert!(matches!(plan.finish(completion(&plan, vec![invalid, eos], vec![0xff], GenerationFinish::Eos)),
        Err(Int4ChatError::Chat(ChatError::NoResult("incomplete UTF-8")))));
}

#[test]
fn serialized_whole_wrapper_not_only_assistant_text_is_bounded() {
    let (planner, eos) = fixture(128); let mut req = request(eos);
    let plan = planner.plan_generate(&req).unwrap();
    let output = plan.finish(completed(&plan, eos)).unwrap();
    let length = canonjson::canonical_bytes(&output).unwrap().len();
    assert!(length > req.generation.max_output_bytes);
    req.budget.max_output_bytes = (length - 1) as u64;
    let limited = planner.plan_generate(&req).unwrap();
    assert!(limited.finish(completed(&limited, eos)).is_err());
}
