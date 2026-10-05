//! Pinned tokenizer/task fixtures with synthetic completions, not model parity.
use super::*;
use crate::native_engine::{generation::quantized::Int8GenerationRun,
    strict_int8::{Int8RunBudget, STRICT_INT8_PROFILE}};
use crate::tasks::chat::quantized::tests::{fixture, request};
fn limits(width: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: width,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(width).unwrap() }
}
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
fn envelope(plan: &PreparedInt8Chat, eos: u32, width: usize) -> (Int8GenerationCohortRun, Int8CohortRequirements) {
    let sequence = raw(plan, 9, eos); let work = sequence.model_work;
    (Int8GenerationCohortRun { schema_version: 1, execution: INT8_PACKED_COHORT_EXECUTION.into(), sequences: vec![sequence],
        group_steps: plan.native.prompt_tokens().div_ceil(width) as u64,
        planned_work: plan.planned_work(), model_work: work },
        Int8CohortRequirements { planned_work: plan.planned_work(), kv_bytes: 0, sampler_bytes: plan.native.sampler_bytes() })
}
#[test]
fn physical_pack_count_does_not_replace_semantic_forward_count() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let (raw, required) = envelope(&plan, eos, 8);
    let result = finalize(&[row(&plan, 9)], raw, required, 1_000_000, limits(8)).unwrap();
    assert_eq!(result.group_steps, plan.native.prompt_tokens().div_ceil(8) as u64);
    assert_eq!(result.model_work.forward_positions, plan.native.prompt_tokens() as u64);
    assert_eq!(result.results[0].result.execution, INT8_GENERATION_VERSION);
    assert_eq!(result.execution, INT8_PACKED_COHORT_EXECUTION);
    let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    let (raw, required) = envelope(&plan, eos, 8);
    assert!(finalize(&[row(&plan, 9)], raw, required, bytes - 1, limits(8)).is_err());
}
#[test]
fn strategy_width_routing_and_work_are_all_checked_before_publication() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    for axis in 0..8 {
        let (mut raw, required) = envelope(&plan, eos, 8);
        match axis { 0 => raw.schema_version = 2, 1 => raw.execution = INT8_COHORT_EXECUTION.into(),
            2 => raw.sequences[0].sequence.request_seq = 8, 3 => raw.sequences[0].sequence.sample_index += 1,
            4 => raw.model_work.attention_pairs += 1, 5 => raw.group_steps += 1,
            6 => raw.planned_work.forward_positions += 1, _ => raw.sequences[0].sequence.content_bytes.push(b'x') }
        assert!(finalize(&[row(&plan, 9)], raw, required, 1_000_000, limits(8)).is_err());
    }
    let (raw, required) = envelope(&plan, eos, 8);
    assert!(finalize(&[row(&plan, 9)], raw, required, 1_000_000, limits(1)).is_err());
}
#[test]
fn ordinary_cohort_finalizer_does_not_silently_accept_packed_execution() {
    let (planner, eos) = fixture(); let plan = planner.plan_generate(&request(eos)).unwrap();
    let (raw, required) = envelope(&plan, eos, 8);
    assert!(super::super::finalize(&[row(&plan, 9)], raw, required, 1_000_000).is_err());
}
#[test]
fn independently_finalized_documents_preserve_input_order_and_full_work() {
    let (planner, eos) = fixture(); let first = planner.plan_generate(&request(eos)).unwrap();
    let mut other = request(eos); other.item_id = "other".into(); other.prompt.push_str(" another document");
    let second = planner.plan_generate(&other).unwrap();
    let a = raw(&first, 9, eos); let b = raw(&second, 2, eos);
    let actual = a.model_work.checked_add(b.model_work).unwrap();
    let planned = first.planned_work().checked_add(second.planned_work()).unwrap();
    let steps = native::packed::expected_group_steps(&[first.native.prompt_tokens(), second.native.prompt_tokens()],
        &[a.model_work.forward_positions, b.model_work.forward_positions], 3).unwrap();
    let raw = Int8GenerationCohortRun { schema_version: 1, execution: INT8_PACKED_COHORT_EXECUTION.into(),
        sequences: vec![a, b], group_steps: steps, planned_work: planned, model_work: actual };
    let required = Int8CohortRequirements { planned_work: planned, kv_bytes: 0, sampler_bytes: 0 };
    let result = finalize(&[row(&first, 9), row(&second, 2)], raw, required, 1_000_000, limits(3)).unwrap();
    assert_eq!(result.results.iter().map(|row| row.result.request_seq).collect::<Vec<_>>(), [9, 2]);
    assert_eq!(result.model_work, actual); assert_eq!(result.group_steps, steps);
}
