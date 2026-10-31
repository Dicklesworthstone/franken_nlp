//! Independent admission boundaries; these tests need no model weights.
use super::*;
use super::super::tests::config;

#[test]
fn simultaneous_contexts_are_charged_without_widening_chunk_task_authority() {
    let mut config = config();
    let task_budget = config.request.budget;
    config.limits.mapping.reduction.map_batch_chunks = 3;
    let required = config.cohort_requirements(32).unwrap();
    let serial = Int4MemoryRequirement::for_context(config.native.context_tokens, 32).unwrap();
    assert_eq!(required.native.kv_bytes, 3 * serial.kv_bytes);
    assert_eq!(required.native.rope_bytes, serial.rope_bytes);
    assert!(required.native.scratch_payload_bound > serial.scratch_payload_bound);
    assert!(required.workspace_bytes > 3 * config.request.budget.max_output_bytes);
    assert!(required.max_result_bytes > 3 * config.request.budget.max_output_bytes);
    assert_eq!(config.request.budget, task_budget);
    assert_eq!(task_budget.max_kv_bytes, serial.kv_bytes);
}

#[test]
fn complete_peak_must_fit_every_resource_axis() {
    let mut config = config();
    config.limits.mapping.reduction.map_batch_chunks = 4;
    let ceiling = config.cohort_requirements(32).unwrap();
    check_peak(ceiling, ceiling).unwrap();
    for axis in 0..5 {
        let mut too_large = ceiling;
        match axis {
            0 => too_large.native.kv_bytes += 1,
            1 => too_large.native.rope_bytes += 1,
            2 => too_large.native.scratch_payload_bound += 1,
            3 => too_large.workspace_bytes += 1,
            _ => too_large.max_result_bytes += 1,
        }
        assert!(check_peak(too_large, ceiling).is_err(), "resource axis {axis}");
    }
    config.limits.mapping.reduction.map_batch_chunks = 1;
    check_peak(config.cohort_requirements(32).unwrap(), ceiling).unwrap();
}

#[test]
fn invalid_window_sizes_and_overflow_are_refused_before_allocation() {
    let mut config = config();
    for count in [0, 17, 65, usize::MAX] {
        config.limits.mapping.reduction.map_batch_chunks = count;
        assert!(config.cohort_requirements(32).is_err(), "window {count}");
    }
    config.limits.mapping.reduction.map_batch_chunks = 2;
    config.request.budget.max_output_bytes = u64::MAX;
    assert!(config.cohort_requirements(32).is_err());
    config.request.budget.max_output_bytes = 4096;
    config.native.context_tokens = 0;
    assert!(config.cohort_requirements(32).is_err());
}

#[test]
fn cohort_window_does_not_reduce_document_frontier_or_output_reservations() {
    let mut config = config();
    let reduction = config.workspace_bytes(11).unwrap();
    let output_bytes = config.limits.mapping.reduction.max_result_bytes;
    config.limits.mapping.reduction.map_batch_chunks = 4;
    assert_eq!(config.workspace_bytes(11).unwrap(), reduction);
    assert_eq!(config.limits.mapping.reduction.max_result_bytes, output_bytes);
    assert!(config.cohort_requirements(32).unwrap().workspace_bytes > 0);
}

#[test]
fn owned_cohort_document_can_cross_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4ExtractionMapCohortRun>();
    send::<HostedOutput<Int4ExtractionMapCohortRun>>();
}
