//! Exact simultaneous sampler/address payload admission; no native execution.
use super::*;
use crate::native_engine::constrained_int8::{
    json_sampling_reuse_workspace_bytes, json_sampling_workspace_bytes,
};

#[test]
fn reuse_adds_only_the_simultaneous_address_payload_to_cold_scratch() {
    let ordinary = json_sampling_workspace_bytes().unwrap();
    let cold = ordinary.checked_add(1_048_576).unwrap();
    assert_eq!(schedule_scratch(cold, ordinary, None).unwrap(), cold);
    assert_eq!(schedule_scratch(cold, ordinary, Some(ordinary)).unwrap(), cold);
    for samples in [1_usize, 2, 32] {
        let needed = json_sampling_reuse_workspace_bytes(samples).unwrap();
        let delta = needed.checked_sub(ordinary).unwrap();
        assert!(delta > 0);
        assert_eq!(schedule_scratch(cold, needed, Some(needed)).unwrap(),
            cold.checked_add(delta).unwrap());
        assert_eq!(schedule_scratch(cold, u64::MAX, Some(needed)).unwrap(),
            cold.checked_add(delta).unwrap());
    }
}

#[test]
fn sampler_caps_and_inconsistent_reuse_quotes_refuse_without_credit() {
    let ordinary = json_sampling_workspace_bytes().unwrap();
    let cold = ordinary.checked_add(1_048_576).unwrap();
    // The existing validate() gate checks the ordinary ceiling first; this
    // helper preserves its cold quote and admits only the reuse delta.
    assert_eq!(schedule_scratch(cold, ordinary - 1, None).unwrap(), cold);
    assert!(schedule_scratch(cold, u64::MAX, Some(ordinary - 1)).is_err());
    for samples in [1_usize, 2, 32] {
        let needed = json_sampling_reuse_workspace_bytes(samples).unwrap();
        assert!(schedule_scratch(cold, needed - 1, Some(needed)).is_err());
        assert!(schedule_scratch(cold, ordinary, Some(needed)).is_err());
    }
}

#[test]
fn reuse_scratch_addition_checks_the_exact_u64_boundary() {
    let ordinary = json_sampling_workspace_bytes().unwrap();
    let needed = json_sampling_reuse_workspace_bytes(32).unwrap();
    let delta = needed.checked_sub(ordinary).unwrap();
    assert!(delta > 0);
    assert_eq!(schedule_scratch(u64::MAX, ordinary, None).unwrap(), u64::MAX);
    assert_eq!(schedule_scratch(u64::MAX - delta, needed, Some(needed)).unwrap(),
        u64::MAX);
    assert!(schedule_scratch(u64::MAX - delta + 1, needed, Some(needed)).is_err());
    assert!(schedule_scratch(u64::MAX, needed, Some(needed)).is_err());
}
