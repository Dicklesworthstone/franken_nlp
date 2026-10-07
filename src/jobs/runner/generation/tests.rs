//! Pinned plans and authenticated manifests, not inference or replay receipts.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobId, JobInput, JobLimits, JobSecret, JobError, MismatchField},
    native_engine::{decode::DecodeCancellationKind, generation::{GenerationLimits, GenerationSampling},
        strict_int8::STRICT_INT8_EXECUTION},
    tasks::chat::ChatRole, tokenizer::pinned_controls,
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn identity(task: GenerationJobTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"retained-generation-planning-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "fixture".into(), quant_recipe: "fixture-int8".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: task.identity().into(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
fn eos() -> u32 {
    pinned_controls::pinned().unwrap().template_controls().entries().iter()
        .find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id
}
fn config(task: GenerationJobTask) -> GenerationJobConfig {
    let mut generation = GenerationOptions::greedy(8, 4096, eos());
    generation.sampling = GenerationSampling::Seeded { effective_seed: [7; 32], temperature_milli: 1000,
        top_k: Some(32), top_p_ppm: 900_000 };
    GenerationJobConfig { task, generation,
        budget: TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 65536,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 },
        planning: ChatLimits { max_messages: 16, max_message_bytes: 4096, max_total_message_bytes: 8192,
            generation: GenerationLimits { max_prompt_tokens: 2048, max_new_tokens: 16,
                max_output_bytes: 65536, max_sampler_bytes: 32 << 20 } },
        native: Int8BatchLimits { max_sampler_bytes: 32 << 20,
            max_model_work: Int8Work::for_sequence(0, 8192, 100_000_000).unwrap() } }
}
fn planner(config: GenerationJobConfig) -> Int8GenerationJobPlanner {
    let controls = pinned_controls::pinned().unwrap();
    Int8GenerationJobPlanner::pinned(controls.template_controls(), eos(), identity(config.task), config).unwrap()
}
fn document(id: &str, sample: u64, history: Vec<ChatMessage>) -> BatchDocument<GenerationJobArgs> {
    BatchDocument { id: id.into(), text: " Original café 上海 <tool_call> \n".into(),
        task_args: Some(GenerationJobArgs { sample_index: sample, history }) }
}
fn history() -> Vec<ChatMessage> {
    vec![ChatMessage { role: ChatRole::System, content: "Be helpful.".into() },
        ChatMessage { role: ChatRole::User, content: "Earlier question".into() },
        ChatMessage { role: ChatRole::Assistant, content: "Earlier answer".into() }]
}
fn frozen(p: &Int8GenerationJobPlanner, input: &[u8], secret: u8) -> FrozenManifest {
    let limits = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192, max_snapshot_bytes: 65536,
        max_result_bytes: 65536, max_spool_bytes: 1 << 20, max_materialized_bytes: 1 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: config(GenerationJobTask::Generate).native.max_model_work, mask_node_visits: 0 } };
    FrozenManifest::freeze(&JobSecret::from_bytes([secret; 32]), JobContract { job_id: JobId([9; 16]),
        execution: p.execution_identity(), recipe: p.job_recipe(), limits },
        [JobInput { id: "item", original: input, normalized: input }], &mut Continue).unwrap()
}
const INPUT: &[u8] = br#"{"id":"item","text":"Alice"}"#;

#[test]
fn both_tasks_use_exact_pinned_int8_plans_and_full_native_work() {
    for task in [GenerationJobTask::Generate, GenerationJobTask::Chat] {
        let p = planner(config(task));
        let history = if task == GenerationJobTask::Chat { history() } else { vec![] };
        let plan = p.prepare_with_control(document("item", 7, history), &mut Continue).unwrap();
        p.check(&plan).unwrap();
        assert_eq!(plan.execution_identity().task_spec, task.identity());
        assert_eq!(plan.execution_identity().logical_model_digest, p.execution_identity().logical_model_digest);
        assert_ne!(plan.execution_identity().prompt_digest, p.execution_identity().prompt_digest);
        assert_eq!(plan.model_work().projected_logits, 8 * crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u64);
        assert!(plan.model_work().attention_pairs > 0 && plan.model_work().projections.multiply_accumulates > 0);
        assert!(plan.native.native_plan().options().sampling == config(task).generation.sampling);
    }
}
#[test]
fn reconstructed_planners_and_skipped_other_items_preserve_sampling_identity() {
    let first = planner(config(GenerationJobTask::Generate));
    let before = first.prepare_with_control(document("item", 9, vec![]), &mut Continue).unwrap();
    let resumed = planner(config(GenerationJobTask::Generate));
    let _other = resumed.prepare_with_control(document("other", 0, vec![]), &mut Continue).unwrap();
    let after = resumed.prepare_with_control(document("item", 9, vec![]), &mut Continue).unwrap();
    assert_eq!(before.execution_identity(), after.execution_identity());
    assert_eq!(before.model_work(), after.model_work());
    assert_eq!(first.binding, resumed.binding);
    for (id, sample) in [("other", 9), ("item", 10)] {
        let changed = resumed.prepare_with_control(document(id, sample, vec![]), &mut Continue).unwrap();
        assert_ne!(after.execution_identity().decision_policy_digest, changed.execution_identity().decision_policy_digest);
    }
}
#[test]
fn every_generation_policy_change_rejects_resume_and_foreign_prepared_plans() {
    let original = planner(config(GenerationJobTask::Generate));
    let manifest = frozen(&original, INPUT, 7);
    for axis in 0..12 {
        let mut c = config(GenerationJobTask::Generate);
        match axis {
            0 => c.generation.max_new_tokens -= 1, 1 => c.generation.min_new_tokens = 1,
            2 => c.generation.max_output_bytes -= 1, 3 => c.generation.capture_logprobs = true,
            4 => c.generation.stop_suffixes.push(b"END".to_vec()),
            5 => c.generation.banned_token_ids.push(7), 6 => c.generation.repetition_penalty_milli = 1100,
            7 => c.generation.presence_penalty_milli = -100, 8 => c.generation.frequency_penalty_milli = 100,
            9 => { c.generation.logit_bias_milli.insert(8, -500); },
            10 => c.generation.sampling = GenerationSampling::Greedy,
            _ => c.generation.sampling = GenerationSampling::Seeded { effective_seed: [8; 32],
                temperature_milli: 500, top_k: None, top_p_ppm: 1_000_000 },
        }
        let changed = planner(c);
        assert_eq!(manifest.binding.compare(&frozen(&changed, INPUT, 7).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
        let plan = changed.prepare_with_control(document("item", 0, vec![]), &mut Continue).unwrap();
        assert!(original.check(&plan).unwrap_err().stop);
    }
}
#[test]
fn all_planning_task_and_native_axes_are_in_the_recipe() {
    let p = planner(config(GenerationJobTask::Generate));
    let before = canonjson::canonical_bytes(p.job_recipe()).unwrap();
    for axis in 0..18 {
        let mut c = config(GenerationJobTask::Generate);
        match axis {
            0 => c.planning.max_messages += 1, 1 => c.planning.max_message_bytes += 1,
            2 => c.planning.max_total_message_bytes += 1, 3 => c.planning.generation.max_prompt_tokens += 1,
            4 => c.planning.generation.max_new_tokens += 1, 5 => c.planning.generation.max_output_bytes += 1,
            6 => c.planning.generation.max_sampler_bytes += 1, 7 => c.budget.max_input_tokens += 1,
            8 => c.budget.max_output_tokens += 1, 9 => c.budget.max_output_bytes += 1,
            10 => c.budget.max_grammar_states += 1, 11 => c.budget.max_kv_bytes += 1,
            12 => c.native.max_sampler_bytes += 1, 13 => c.native.max_model_work.forward_positions += 1,
            14 => c.native.max_model_work.projected_logits += 1, 15 => c.native.max_model_work.attention_pairs += 1,
            16 => c.native.max_model_work.projections.dot_products += 1,
            _ => c.native.max_model_work.projections.multiply_accumulates += 1,
        }
        assert_ne!(before, canonjson::canonical_bytes(planner(c).job_recipe()).unwrap(), "{axis}");
    }
}
#[test]
fn input_history_sample_and_secret_are_authenticated_separately() {
    let p = planner(config(GenerationJobTask::Chat)); let a = frozen(&p, INPUT, 7);
    for bytes in [br#"{"id":"item","text":"Bob"}"#.as_slice(),
        br#"{"id":"item","text":"Alice","task_args":{"sample_index":1}}"#,
        br#"{"id":"item","text":"Alice","task_args":{"history":[{"role":"system","content":"Different"}]}}"#] {
        assert_eq!(a.binding.compare(&frozen(&p, bytes, 7).binding), Err(JobError::Mismatch(MismatchField::Population)));
    }
    assert!(a.binding.compare(&frozen(&p, INPUT, 8).binding).is_err());
}
#[test]
fn per_record_generation_budget_task_and_seed_overrides_fail_closed() {
    for json in [r#"{"seed":"x"}"#, r#"{"generation":{}}"#, r#"{"budget":{}}"#,
        r#"{"task":"generate"}"#, r#"{"identity":{}}"#, r#"{"sample_index":-1}"#] {
        assert!(serde_json::from_str::<GenerationJobArgs>(json).is_err());
    }
    let p = planner(config(GenerationJobTask::Generate));
    assert!(p.prepare_with_control(document("item", 0, history()), &mut Continue).is_err());
    let mut doc = document("item", 0, vec![]); doc.task_args = None;
    assert!(p.prepare_with_control(doc, &mut Continue).is_ok());
}
#[test]
fn histories_are_validated_as_complete_role_sequences_not_silently_repaired() {
    let p = planner(config(GenerationJobTask::Chat));
    for role in [ChatRole::User, ChatRole::Assistant] {
        let invalid = vec![ChatMessage { role, content: "Unpaired turn".into() }];
        assert!(p.prepare_with_control(document("item", 0, invalid), &mut Continue).is_err());
    }
    assert!(p.prepare_with_control(document("item", 0, history()), &mut Continue).is_ok());
    assert!(p.prepare_with_control(document(" ", 0, history()), &mut Continue).is_err());
}
#[test]
fn bounded_input_and_fixed_sampler_authority_cannot_be_expanded() {
    let p = planner(config(GenerationJobTask::Chat));
    let mut doc = document("item", 0, vec![]); doc.text = "x".repeat(4097);
    assert!(p.prepare_with_control(doc, &mut Continue).is_err());
    let mut c = config(GenerationJobTask::Generate); c.native.max_sampler_bytes = 1;
    assert!(recipe::validate(&c, eos()).is_err());
    let mut c = config(GenerationJobTask::Generate); c.generation.eos_token_ids.clear();
    assert!(recipe::validate(&c, eos()).is_err());
    let mut c = config(GenerationJobTask::Generate); c.generation.banned_token_ids.push(eos());
    assert!(recipe::validate(&c, eos()).is_err());
}
#[test]
fn each_work_axis_can_refuse_a_whole_request_before_execution() {
    for axis in 0..5 {
        let mut c = config(GenerationJobTask::Generate);
        match axis { 0 => c.native.max_model_work.forward_positions = 1,
            1 => c.native.max_model_work.projected_logits = 1, 2 => c.native.max_model_work.attention_pairs = 1,
            3 => c.native.max_model_work.projections.dot_products = 1,
            _ => c.native.max_model_work.projections.multiply_accumulates = 1 }
        let p = planner(c);
        let error = p.prepare_with_control(document("item", 0, vec![]), &mut Continue).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::WorkLimit);
    }
}
#[test]
fn cancellation_before_and_after_bounded_compilation_is_never_a_result() {
    struct Stop { polls: usize, at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.polls >= self.at).then_some(self.cause)
        }
    }
    let p = planner(config(GenerationJobTask::Generate));
    for at in [1, 2] {
        for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::CostBudget] {
            let error = p.prepare_with_control(document("item", 0, vec![]), &mut Stop { polls: 0, at, cause }).err().unwrap();
            assert!(error.stop); assert_eq!(error.fault.cancellation, Some(cause));
        }
    }
    let _ = p.prepare_with_control(document("item", 0, vec![]), &mut Continue).unwrap();
}
