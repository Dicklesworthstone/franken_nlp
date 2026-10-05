//! Pinned preparation plus synthetic terminal data; no model execution.
use super::*;
use crate::native_engine::{generation::quantized::Int8GenerationRun,
    strict_int8::{Int8RunBudget, STRICT_INT8_PROFILE}};
use crate::tasks::chat::quantized::tests::{fixture, request};
fn row(plan: &PreparedInt8Chat, seq: u64) -> Int8ChatCohortRequest<'_> {
    Int8ChatCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(), request_seq: seq,
        budget: Int8GenerationBudget { native: Int8RunBudget::exact(plan.planned_work()),
            max_kv_bytes: u64::MAX, max_sampler_bytes: plan.native.sampler_bytes() } }
}
fn raw(plan: &PreparedInt8Chat, seq: u64, eos: u32) -> Int8GenerationRun {
    let position = plan.native.prompt_tokens();
    Int8GenerationRun { schema_version: 1,
        sequence: GeneratedSequence { schema_version: 1, execution: INT8_GENERATION_VERSION.into(),
            numerics_profile: STRICT_INT8_PROFILE.into(), request_seq: seq, sample_index: plan.sample_index,
            token_ids: vec![eos], content_bytes: Vec::new(), finish_reason: GenerationFinish::Eos,
            effective_seed: None, token_logprobs: None, logprob_score_space: None,
            native_work: GenerationWork { forward_positions: position as u64,
                projected_logits: NANBEIGE_VOCAB_SIZE as u64, sampled_steps: 0 } },
        model_work: Int8Work::for_sequence(0, position, NANBEIGE_VOCAB_SIZE).unwrap() }
}
fn envelope(plan: &PreparedInt8Chat, eos: u32) -> (Int8GenerationCohortRun, Int8CohortRequirements) {
    let sequence = raw(plan, 9, eos); let work = sequence.model_work;
    (Int8GenerationCohortRun { schema_version: 1, execution: INT8_COHORT_EXECUTION.into(), sequences: vec![sequence],
        group_steps: work.forward_positions, planned_work: plan.planned_work(), model_work: work },
        Int8CohortRequirements { planned_work: plan.planned_work(), kv_bytes: 0, sampler_bytes: plan.native.sampler_bytes() })
}
#[test]
fn pinned_task_kv_limit_is_not_weakened_by_cohort_admission() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let rows = native_requests(&[row(&plan, 9)]).unwrap();
    assert_eq!(rows[0].budget.max_kv_bytes, plan.task.ir().budget().max_kv_bytes);
    assert_eq!(rows[0].plan.execution_identity(), plan.execution_identity());
    assert!(std::ptr::eq(rows[0].decoder, plan.tokenizer.tokenizer()));
}
#[test]
fn finalizer_keeps_ordinary_task_semantics_and_checks_whole_envelope() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let (raw, required) = envelope(&plan, eos);
    let result = finalize(&[row(&plan, 9)], raw, required, 1_000_000).unwrap();
    assert_eq!(result.results.len(), 1); assert_eq!(result.group_steps, plan.native.prompt_tokens() as u64);
    let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    let (raw, required) = envelope(&plan, eos);
    assert!(finalize(&[row(&plan, 9)], raw, required, bytes - 1).is_err());
}
#[test]
fn routing_execution_and_work_tampering_cannot_publish_a_cohort() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    for axis in 0..7 {
        let (mut raw, required) = envelope(&plan, eos);
        match axis { 0 => raw.schema_version = 2, 1 => raw.execution.push('x'),
            2 => raw.sequences[0].sequence.request_seq = 8, 3 => raw.sequences[0].sequence.sample_index += 1,
            4 => raw.model_work.attention_pairs += 1, 5 => raw.group_steps += 1,
            _ => raw.planned_work.forward_positions += 1 }
        assert!(finalize(&[row(&plan, 9)], raw, required, 1_000_000).is_err());
    }
}
