use super::*;
use crate::tasks::chat::quantized::tests::{fixture, request};
fn limits() -> ChatCohortLimits {
    ChatCohortLimits { native: NativeLimits { context_tokens: crate::native_engine::rope::DEFAULT_ADMITTED_CONTEXT_CAP,
        allocator_reserve_bytes: 64 << 20, run: RunLimits { max_elapsed: Duration::from_secs(1),
            max_checkpoints: 10_000, cleanup_reserve_bytes: 4096 } }, max_sampler_bytes: 256 << 20,
        preparation_reserve_bytes: 256 << 20, max_result_bytes: 1 << 20 }
}
#[test]
fn queued_requests_do_not_allocate_simultaneous_kv_or_sampler_slots() {
    let (planner, eos) = fixture();
    let plans: Vec<_> = (0..4).map(|_| planner.plan_generate(&request(eos)).unwrap()).collect();
    let row = plans[0].planned_work(); let sampler = plans[0].native_plan().sampler_bytes();
    let mut bounds = limits(); bounds.max_sampler_bytes = 2 * sampler;
    let input = build_input(plans, bounds, 2).unwrap();
    assert_eq!(input.prepared.len(), 4); assert_eq!(input.contexts.len(), 2);
    assert_eq!(input.sampler_bytes, 2 * sampler); assert_eq!(input.work.forward_positions, 4 * row.forward_positions);
    assert_eq!(Int8MemoryRequirement::for_cohort(&input.contexts).unwrap().kv_bytes,
        2 * row.forward_positions * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(input.output_tokens, 4 * u64::from(request(eos).budget.max_output_tokens));
}
#[test]
fn every_reusable_slot_reserves_the_longest_queued_context() {
    let (planner, eos) = fixture(); let a = planner.plan_generate(&request(eos)).unwrap();
    let mut long = request(eos); long.prompt.push_str(&" longer".repeat(30));
    let b = planner.plan_generate(&long).unwrap(); let longest = b.planned_work().forward_positions as usize;
    let input = build_input(vec![a, b], limits(), 1).unwrap(); assert_eq!(input.contexts, [longest]);
}
#[test]
fn a_short_requests_task_kv_ceiling_still_bounds_the_physical_slot() {
    let (planner, eos) = fixture(); let ordinary = planner.plan_generate(&request(eos)).unwrap();
    let mut short = request(eos); short.budget.max_kv_bytes = ordinary.planned_work().forward_positions * KV_BYTES_PER_TOKEN as u64;
    let a = planner.plan_generate(&short).unwrap();
    let mut long = request(eos); long.prompt.push_str(&" longer".repeat(30));
    let b = planner.plan_generate(&long).unwrap();
    assert!(build_input(vec![a, b], limits(), 1).is_err());
}
#[test]
fn smaller_live_limit_does_not_discard_the_whole_epoch_output_charge() {
    let (planner, eos) = fixture();
    for axis in 0..3 {
        let plans: Vec<_> = (0..4).map(|_| planner.plan_generate(&request(eos)).unwrap()).collect();
        let mut bounds = limits();
        match axis { 0 => bounds.max_result_bytes = 2 * request(eos).budget.max_output_bytes + 4096,
            1 => bounds.max_sampler_bytes = 2 * plans[0].native_plan().sampler_bytes() - 1,
            _ => bounds.native.context_tokens = plans[0].native_plan().prompt_tokens() - 1 }
        assert!(build_input(plans, bounds, 2).is_err());
    }
}
#[test]
fn invalid_slot_counts_and_delivery_overflow_refuse_before_dispatch() {
    for (active, count) in [(0, 4), (5, 4), (1, 65), (1, 0)] { assert!(validate_slots(active, count).is_err()); }
    validate_slots(1, 64).unwrap(); validate_slots(64, 64).unwrap();
    assert!(validate_limits(limits(), 2, u64::MAX).is_err());
    let mut short = Int8PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap() };
    short.max_extra_scratch_bytes -= 1; assert!(packed_scratch(Some(short)).is_err());
}
