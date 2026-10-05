use super::*;
use crate::tasks::chat::quantized::tests::{fixture, request};
fn limits() -> ChatCohortLimits {
    ChatCohortLimits { native: NativeLimits { context_tokens: crate::native_engine::rope::DEFAULT_ADMITTED_CONTEXT_CAP,
        allocator_reserve_bytes: 64 << 20, run: RunLimits { max_elapsed: Duration::from_secs(1),
            max_checkpoints: 10_000, cleanup_reserve_bytes: 4096 } }, max_sampler_bytes: 256 << 20,
        preparation_reserve_bytes: 256 << 20, max_result_bytes: 1 << 20 }
}
#[test]
fn admission_rejects_invalid_cohorts_and_delivery_overflow() {
    validate_limits(limits(), 1, u64::MAX).unwrap();
    validate_limits(limits(), 64, 1).unwrap();
    for (count, first) in [(0, 1), (65, 1), (1, 0), (2, u64::MAX)] {
        assert!(validate_limits(limits(), count, first).is_err());
    }
    for axis in 0..6 { let mut bad = limits();
        match axis { 0 => bad.max_sampler_bytes = 0, 1 => bad.preparation_reserve_bytes = 0,
            2 => bad.max_result_bytes = 0, 3 => bad.native.allocator_reserve_bytes = 0,
            4 => bad.native.context_tokens = 0, _ => bad.native.run.max_elapsed = Duration::ZERO }
        assert!(validate_limits(bad, 2, 1).is_err());
    }
}
#[test]
fn per_row_context_and_simultaneous_storage_are_derived_from_real_plans() {
    let (planner, eos) = fixture();
    let a = planner.plan_generate(&request(eos)).unwrap();
    let mut long = request(eos); long.item_id = "other".into(); long.prompt.push_str(&" longer".repeat(30));
    let b = planner.plan_generate(&long).unwrap();
    let expected = a.planned_work().checked_add(b.planned_work()).unwrap();
    let sampler = a.native_plan().sampler_bytes() + b.native_plan().sampler_bytes();
    let input = build_input(vec![a, b], limits()).unwrap();
    assert!(input.contexts[0] < input.contexts[1]); assert_eq!(input.work, expected); assert_eq!(input.sampler_bytes, sampler);
    assert_eq!(Int8MemoryRequirement::for_cohort(&input.contexts).unwrap().kv_bytes,
        expected.forward_positions * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64);
}
#[test]
fn one_rows_budget_cannot_cover_two_samplers_or_two_outputs() {
    let (planner, eos) = fixture();
    for axis in 0..3 {
        let a = planner.plan_generate(&request(eos)).unwrap(); let b = planner.plan_generate(&request(eos)).unwrap();
        let mut short = limits();
        match axis { 0 => short.max_sampler_bytes = a.native_plan().sampler_bytes(),
            1 => short.max_result_bytes = a.task_plan().ir().budget().max_output_bytes + 4096,
            _ => short.native.context_tokens = a.native_plan().prompt_tokens() - 1 }
        assert!(build_input(vec![a, b], short).is_err());
    }
}
#[test]
fn owned_cohort_types_can_cross_the_existing_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Input>(); send::<HostedOutput<Int8ChatCohortResult>>();
}
