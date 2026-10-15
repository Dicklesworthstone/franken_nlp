use super::*;
use crate::native_engine::strict_int4::Int4Work;

fn need(contexts: &[usize]) -> Int4BatchGenerationRequirements {
    let native = Int4MemoryRequirement::for_cohort(contexts, 128).unwrap();
    Int4BatchGenerationRequirements { planned_work: Int4Work::default(), group_width: 128,
        native, native_payload_bytes: native.kv_bytes + native.rope_bytes + native.scratch_payload_bound,
        sampler_payload_bytes: contexts.len() as u64 * 1024, output_payload_upper_bytes: contexts.len() as u64 * 4096 }
}
#[test]
fn host_reserves_shared_native_state_all_samplers_and_raw_outputs_separately_from_final_output() {
    for contexts in [vec![1], vec![2, 7, 3], vec![16; 64]] {
        let required = need(&contexts); let allocator = 64 * 1024;
        assert_eq!(workspace_bytes(required, allocator).unwrap(), required.native.rope_bytes
            + required.native.scratch_payload_bound + required.sampler_payload_bytes
            + required.output_payload_upper_bytes + allocator);
        assert_eq!(required.native.kv_bytes,
            contexts.iter().sum::<usize>() as u64 * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64);
    }
}
#[test]
fn aggregate_overflow_is_refused_instead_of_wrapping_the_reservation() {
    let original = need(&[2, 7]);
    assert!(workspace_bytes(original, u64::MAX).is_err());
    for axis in 0..4 {
        let mut required = original;
        match axis { 0 => required.native.rope_bytes = u64::MAX,
            1 => required.native.scratch_payload_bound = u64::MAX,
            2 => required.sampler_payload_bytes = u64::MAX, _ => required.output_payload_upper_bytes = u64::MAX }
        assert!(workspace_bytes(required, 1).is_err());
    }
}
