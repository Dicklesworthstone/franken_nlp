use super::*;

#[test]
fn wrapper_bound_is_still_whole_epoch_not_live_slot_count() {
    let values = [17_u64, 23, 31, 47];
    let expected = 16_384 + values.iter().sum::<u64>();
    assert_eq!(wrapper_bound(values.into_iter()).unwrap(), expected);
}

#[test]
fn refill_execution_label_is_distinct_from_fixed_and_packed_cohorts() {
    assert_ne!(INT4_REFILL_EXECUTION, INT4_BATCH_GENERATION_VERSION);
    assert_ne!(INT4_REFILL_EXECUTION, INT4_PACKED_COHORT_EXECUTION);
}
