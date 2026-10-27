//! UNRUN pinned planning/commitment fixtures, not native inference or recovery
//! proof. Synthetic model identities remain confined to this private module.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobId, JobInput, JobSecret},
    native_engine::{decode::DecodeCancellationKind,
        generation::{GenerationLimits, GenerationSampling, grouped_int4::INT4_SOURCE_FORMAT},
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{Int4RunBudget, STRICT_INT4_EXECUTION}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};
const WIDTH: usize = 127;
const CONTEXT: usize = 256;
const INPUT: &[u8] = br#"{"id":"item","text":"Alice"}"#;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn eos() -> u32 { EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap() }
fn model_work() -> Int4Work { Int4Work::for_sequence(WIDTH, 0, 2048, 64 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn limits() -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: 8192, max_snapshot_bytes: 65_536,
        max_result_bytes: 65_536, max_spool_bytes: 1 << 20, max_materialized_bytes: 1 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4, max_work: JobWork::from_int4(model_work(), 0) }
}
fn config(task: GenerationJobTask) -> Int4GenerationJobConfig {
    let mut generation = GenerationOptions::greedy(8, 4096, eos());
    generation.sampling = GenerationSampling::Seeded { effective_seed: [7; 32], temperature_milli: 1000,
        top_k: Some(32), top_p_ppm: 900_000 };
    let kv = Int4MemoryRequirement::for_context(CONTEXT, WIDTH).unwrap().kv_bytes;
    Int4GenerationJobConfig { task, generation, context_tokens: CONTEXT,
        budget: TaskBudget { max_input_tokens: 128, max_output_tokens: 16, max_output_bytes: 65_536,
            max_grammar_states: 1, max_kv_bytes: kv },
        planning: ChatLimits { max_messages: 16, max_message_bytes: 4096, max_total_message_bytes: 8192,
            generation: GenerationLimits { max_prompt_tokens: 128, max_new_tokens: 16,
                max_output_bytes: 65_536, max_sampler_bytes: 32 << 20 } },
        native: Int4GenerationBudget { native: Int4RunBudget::exact(model_work()), max_kv_bytes: kv,
            max_sampler_bytes: 32 << 20 } }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-int4-durable-text-planning-only");
    let packing = canonjson::canonical_bytes(&(INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH)).unwrap();
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: d, template_digest: d,
        task_spec: "uncompiled-source-int4-text".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn configured(config: Int4GenerationJobConfig, job: JobLimits) -> Result<Int4GenerationJobPlanner, BatchFault> {
    let controls = pinned_controls::pinned().unwrap();
    let compiler = Int4ChatPlanner::pinned(controls.template_controls(), eos(), identity(), WIDTH,
        config.budget, config.planning).unwrap();
    Int4GenerationJobPlanner::configured(compiler, config, job)
}
fn planner(task: GenerationJobTask) -> Int4GenerationJobPlanner { configured(config(task), limits()).unwrap() }
fn document(id: &str, sample_index: u64, history: Vec<ChatMessage>) -> BatchDocument<GenerationJobArgs> {
    BatchDocument { id: id.to_owned(), text: " Original café 上海 <tool_call> \n".to_owned(),
        task_args: Some(GenerationJobArgs { sample_index, history }) }
}
fn history() -> Vec<ChatMessage> {
    vec![ChatMessage { role: ChatRole::System, content: "Be helpful.".to_owned() },
        ChatMessage { role: ChatRole::User, content: "Earlier question".to_owned() },
        ChatMessage { role: ChatRole::Assistant, content: "Earlier answer".to_owned() }]
}
fn frozen(planner: &Int4GenerationJobPlanner, input: &[u8], secret: u8) -> FrozenManifest {
    FrozenManifest::freeze(&JobSecret::from_bytes([secret; 32]), JobContract { job_id: JobId([9; 16]),
        execution: planner.execution_identity(), recipe: planner.job_recipe(), limits: planner.job_limits() },
        [JobInput { id: "item", original: input, normalized: input }], &mut Continue).unwrap()
}

#[test]
fn both_tasks_use_pinned_int4_plans_and_real_group_work() {
    for task in [GenerationJobTask::Generate, GenerationJobTask::Chat] {
        let planner = planner(task);
        let history = if task == GenerationJobTask::Chat { history() } else { vec![] };
        let prepared = planner.prepare_with_control(document("item", 7, history), &mut Continue).unwrap();
        planner.check(&prepared).unwrap();
        assert_eq!(planner.execution_identity().task_spec, task.identity());
        assert_eq!(planner.execution_identity().sampler_version, crate::native_engine::sampler::SAMPLER_VERSION);
        assert_eq!(prepared.execution_identity().task_spec, task.identity());
        assert_eq!(prepared.execution_identity().logical_model_digest, planner.execution_identity().logical_model_digest);
        assert_ne!(prepared.execution_identity().prompt_digest, planner.execution_identity().prompt_digest);
        let work = prepared.model_work();
        assert_eq!(work.projected_logits, 8 * NANBEIGE_VOCAB_SIZE as u64);
        assert_eq!(work, Int4Work::for_sequence(WIDTH, 0, work.forward_positions as usize, work.projected_logits as usize).unwrap());
        assert_eq!(prepared.work, JobWork::from_int4(work, 0));
        assert!(prepared.native.native_plan().options().sampling == config(task).generation.sampling);
        assert_eq!(planner.native_limits().max_kv_bytes, planner.task_budget().max_kv_bytes);
    }
}
#[test]
fn reopen_and_skipped_items_preserve_addressed_identity_but_not_factory_seal() {
    let first = planner(GenerationJobTask::Generate);
    let before = first.prepare_with_control(document("item", 9, vec![]), &mut Continue).unwrap();
    let reopened = planner(GenerationJobTask::Generate);
    let _other = reopened.prepare_with_control(document("another", 0, vec![]), &mut Continue).unwrap();
    let after = reopened.prepare_with_control(document("item", 9, vec![]), &mut Continue).unwrap();
    assert_eq!(before.execution_identity(), after.execution_identity());
    assert_eq!(before.work, after.work);
    assert!(first.check(&after).is_err());
    assert!(frozen(&first, INPUT, 7).binding.compare(&frozen(&reopened, INPUT, 7).binding).is_ok());
    for (id, sample) in [("another", 9), ("item", 10)] {
        let changed = reopened.prepare_with_control(document(id, sample, vec![]), &mut Continue).unwrap();
        assert_ne!(after.execution_identity().decision_policy_digest, changed.execution_identity().decision_policy_digest);
    }
}
#[test]
fn every_policy_axis_changes_authenticated_resume_contract() {
    let original = planner(GenerationJobTask::Generate);
    let manifest = frozen(&original, INPUT, 7);
    for axis in 0..15 {
        let mut config = config(GenerationJobTask::Generate);
        match axis {
            0 => config.generation.max_new_tokens -= 1,
            1 => config.generation.min_new_tokens = 1,
            2 => config.generation.max_output_bytes -= 1,
            3 => config.generation.capture_logprobs = true,
            4 => config.generation.stop_suffixes.push(b"END".to_vec()),
            5 => config.generation.banned_token_ids.push(7),
            6 => config.generation.repetition_penalty_milli = 1100,
            7 => config.generation.presence_penalty_milli = -100,
            8 => config.generation.frequency_penalty_milli = 100,
            9 => { config.generation.logit_bias_milli.insert(8, -500); },
            10 => config.generation.sampling = GenerationSampling::Greedy,
            _ => config.generation.sampling = GenerationSampling::Seeded {
                effective_seed: if axis == 11 { [8; 32] } else { [7; 32] },
                temperature_milli: if axis == 12 { 500 } else { 1000 },
                top_k: if axis == 13 { None } else { Some(32) },
                top_p_ppm: if axis == 14 { 1_000_000 } else { 900_000 } },
        }
        let changed = configured(config, limits()).unwrap();
        assert!(manifest.binding.compare(&frozen(&changed, INPUT, 7).binding).is_err(), "policy axis {axis}");
    }
}
#[test]
fn planning_task_native_and_every_job_limit_are_frozen() {
    let original = planner(GenerationJobTask::Generate);
    let before = canonjson::canonical_bytes(original.job_recipe()).unwrap();
    for axis in 0..17 {
        let mut config = config(GenerationJobTask::Generate);
        match axis {
            0 => config.planning.max_messages += 1,
            1 => config.planning.max_message_bytes += 1,
            2 => config.planning.max_total_message_bytes += 1,
            3 => config.planning.generation.max_prompt_tokens += 1,
            4 => config.planning.generation.max_new_tokens += 1,
            5 => config.planning.generation.max_output_bytes += 1,
            6 => config.planning.generation.max_sampler_bytes += 1,
            7 => config.budget.max_input_tokens += 1,
            8 => config.budget.max_output_tokens += 1,
            9 => config.budget.max_output_bytes -= 1,
            10 => config.budget.max_grammar_states += 1,
            11 => config.native.max_sampler_bytes += 1,
            12 => config.native.native.max_forward_positions -= 1,
            13 => config.native.native.max_attention_pairs -= 1,
            14 => config.native.native.max_projection_work.dot_products -= 1,
            15 => config.native.native.max_projection_work.multiply_accumulates -= 1,
            _ => {
                config.context_tokens += 1;
                let kv = Int4MemoryRequirement::for_context(config.context_tokens, WIDTH).unwrap().kv_bytes;
                config.budget.max_kv_bytes = kv; config.native.max_kv_bytes = kv;
            }
        }
        let changed = configured(config, limits()).unwrap();
        assert_ne!(before, canonjson::canonical_bytes(changed.job_recipe()).unwrap(), "recipe axis {axis}");
    }
    for axis in 0..15 {
        let mut job = limits();
        match axis {
            0 => job.max_items += 1, 1 => job.max_id_bytes += 1, 2 => job.max_input_bytes_per_item += 1,
            3 => job.max_snapshot_bytes += 1, 4 => job.max_result_bytes += 1, 5 => job.max_spool_bytes += 1,
            6 => job.max_materialized_bytes += 1, 7 => job.max_journal_bytes += 4096, 8 => job.max_attempts += 1,
            9 => job.max_work.model.forward_positions += 1, 10 => job.max_work.model.projected_logits += 1,
            11 => job.max_work.model.attention_pairs += 1, 12 => job.max_work.model.projections.dot_products += 1,
            13 => job.max_work.model.projections.multiply_accumulates += 1, _ => job.max_work.mask_node_visits += 1,
        }
        assert!(original.check_job_limits(job).is_err(), "job axis {axis}");
        let changed = configured(config(GenerationJobTask::Generate), job).unwrap();
        assert_ne!(before, canonjson::canonical_bytes(changed.job_recipe()).unwrap(), "job recipe axis {axis}");
    }
}
#[test]
fn private_input_history_sample_and_secret_changes_reject_resume() {
    let planner = planner(GenerationJobTask::Chat); let manifest = frozen(&planner, INPUT, 7);
    for bytes in [br#"{"id":"item","text":"Bob"}"#.as_slice(),
        br#"{"id":"item","text":"Alice","task_args":{"sample_index":1}}"#,
        br#"{"id":"item","text":"Alice","task_args":{"history":[{"role":"system","content":"Different"}]}}"#] {
        assert!(manifest.binding.compare(&frozen(&planner, bytes, 7).binding).is_err());
    }
    assert!(manifest.binding.compare(&frozen(&planner, INPUT, 8).binding).is_err());
}
#[test]
fn complete_context_kv_sampler_eos_and_projected_logit_ceilings_fail_closed() {
    for axis in 0..7 {
        let mut config = config(GenerationJobTask::Generate);
        match axis {
            0 => config.native.max_kv_bytes -= 1, 1 => config.budget.max_kv_bytes += 1,
            2 => config.context_tokens -= 1, 3 => config.native.max_sampler_bytes = 1,
            4 => config.generation.eos_token_ids = vec![7],
            5 => config.generation.banned_token_ids.push(eos()),
            _ => config.budget.max_input_tokens = CONTEXT as u32,
        }
        // The context-overflow case also raises the independent prompt cap.
        if axis == 6 { config.planning.generation.max_prompt_tokens = CONTEXT; }
        assert!(configured(config, limits()).is_err(), "invalid axis {axis}");
    }
    let mut job = limits(); job.max_work.model.projected_logits = 8 * NANBEIGE_VOCAB_SIZE as u64 - 1;
    let planner = configured(config(GenerationJobTask::Generate), job).unwrap();
    let error = planner.prepare_with_control(document("item", 0, vec![]), &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(error.stop);
}
#[test]
fn record_overrides_invalid_history_and_cancellation_are_no_result() {
    for json in [r#"{"generation":{}}"#, r#"{"budget":{}}"#, r#"{"task":"chat"}"#,
        r#"{"seed":"x"}"#, r#"{"identity":{}}"#, r#"{"sample_index":-1}"#] {
        assert!(serde_json::from_str::<GenerationJobArgs>(json).is_err());
    }
    let generate = planner(GenerationJobTask::Generate);
    assert!(generate.prepare_with_control(document("item", 0, history()), &mut Continue).is_err());
    let chat = planner(GenerationJobTask::Chat);
    for messages in [vec![ChatMessage { role: ChatRole::User, content: "Unanswered".to_owned() }],
        vec![ChatMessage { role: ChatRole::Assistant, content: "No prior user".to_owned() }]] {
        assert!(chat.prepare_with_control(document("item", 0, messages), &mut Continue).is_err());
    }
    struct Stop { at: usize, seen: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.seen += 1; (self.seen == self.at).then_some(DecodeCancellationKind::Deadline)
        }
    }
    for at in [1, 2] {
        let error = generate.prepare_with_control(document("item", 0, vec![]), &mut Stop { at, seen: 0 }).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline)); assert!(error.stop);
    }
}
