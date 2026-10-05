//! Real pinned preparation; synthetic complete terminal data, not model output.
use super::*;
use crate::native_engine::{generation::quantized::Int8GenerationRun, strict_int8::{Int8RunBudget, STRICT_INT8_PROFILE}};
use crate::tasks::chat::quantized::tests::{fixture, request};
fn limits() -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap() }
}
fn row(plan: &PreparedInt8Chat, seq: u64) -> Int8ChatCohortRequest<'_> {
    Int8ChatCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(), request_seq: seq,
        budget: Int8GenerationBudget { native: Int8RunBudget::exact(plan.planned_work()),
            max_kv_bytes: u64::MAX, max_sampler_bytes: plan.native.sampler_bytes() } }
}
fn envelope(plans: &[PreparedInt8Chat], eos: u32) -> (Int8GenerationCohortRun, Int8CohortRequirements) {
    let mut sequences = Vec::new(); let mut actual = Int8Work::default(); let mut planned = Int8Work::default();
    let mut prompts = Vec::new(); let mut positions = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        let prompt = plan.native.prompt_tokens();
        let work = Int8Work::for_sequence(0, prompt, NANBEIGE_VOCAB_SIZE).unwrap();
        sequences.push(Int8GenerationRun { schema_version: 1,
            sequence: GeneratedSequence { schema_version: 1, execution: INT8_GENERATION_VERSION.into(),
                numerics_profile: STRICT_INT8_PROFILE.into(), request_seq: 90 - index as u64, sample_index: plan.sample_index,
                token_ids: vec![eos], content_bytes: Vec::new(), finish_reason: GenerationFinish::Eos,
                effective_seed: None, token_logprobs: None, logprob_score_space: None,
                native_work: GenerationWork { forward_positions: prompt as u64,
                    projected_logits: NANBEIGE_VOCAB_SIZE as u64, sampled_steps: 0 } }, model_work: work });
        actual = actual.checked_add(work).unwrap(); planned = planned.checked_add(plan.planned_work()).unwrap();
        prompts.push(prompt); positions.push(prompt as u64);
    }
    (Int8GenerationCohortRun { schema_version: 1, execution: INT8_REFILL_EXECUTION.into(), sequences,
        group_steps: native::packed::refill::expected_steps(&prompts, &positions, 1, 4).unwrap(),
        planned_work: planned, model_work: actual },
        Int8CohortRequirements { planned_work: planned, kv_bytes: 0, sampler_bytes: 0 })
}
#[test]
fn finalization_preserves_input_order_not_delivery_number_or_slot_order() {
    let (planner, eos) = fixture();
    let plans = [planner.plan_generate(&request(eos)).unwrap(), planner.plan_generate(&request(eos)).unwrap()];
    let rows = [row(&plans[0], 90), row(&plans[1], 89)];
    let (raw, required) = envelope(&plans, eos);
    let result = finalize(&rows, raw, required, 1_000_000, 1, limits()).unwrap();
    assert_eq!(result.execution, INT8_REFILL_EXECUTION);
    assert_eq!(result.results.iter().map(|row| row.result.request_seq).collect::<Vec<_>>(), [90, 89]);
    let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    let (raw, required) = envelope(&plans, eos);
    assert!(finalize(&rows, raw, required, bytes - 1, 1, limits()).is_err());
}
#[test]
fn changed_execution_routing_work_and_partial_epochs_never_finalize() {
    let (planner, eos) = fixture(); let plans = [planner.plan_generate(&request(eos)).unwrap()];
    for axis in 0..9 {
        let (mut raw, required) = envelope(&plans, eos);
        match axis { 0 => raw.schema_version = 2, 1 => raw.execution = INT8_PACKED_COHORT_EXECUTION.into(),
            2 => raw.sequences[0].sequence.request_seq = 1, 3 => raw.sequences[0].sequence.sample_index += 1,
            4 => raw.model_work.attention_pairs += 1, 5 => raw.group_steps += 1,
            6 => raw.planned_work.forward_positions += 1, 7 => { raw.sequences.clear(); },
            _ => raw.sequences[0].sequence.content_bytes = vec![0xff] }
        assert!(finalize(&[row(&plans[0], 90)], raw, required, 1_000_000, 1, limits()).is_err());
    }
    let (raw, required) = envelope(&plans, eos);
    assert!(finalize(&[row(&plans[0], 90)], raw, required, 1_000_000, 0, limits()).is_err());
}
#[test]
fn native_request_conversion_retains_pinned_task_limits_and_decoder() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let rows = native_requests(&[row(&plan, 90)]).unwrap();
    assert_eq!(rows[0].budget.max_kv_bytes, plan.task.ir().budget().max_kv_bytes);
    assert!(std::ptr::eq(rows[0].decoder, plan.tokenizer.tokenizer()));
}
