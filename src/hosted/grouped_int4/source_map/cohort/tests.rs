//! Model-free source-map resource admission for all supported raw task kinds.
use super::*;
use super::super::tests::fixture;

#[test]
fn all_source_task_windows_keep_per_chunk_authority_and_complete_frontier() {
    for task in [SourceMapTask::Ner(Default::default()), SourceMapTask::Keyphrases(Default::default()),
        SourceMapTask::Summarize(Default::default())] {
        let (planner, mut config) = fixture(task, 32);
        let row = map_requirements(&config, 16, 32, &planner).unwrap();
        let task_budget = config.budget;
        let frontier = config.workspace_bytes(7).unwrap();
        config.mapping.reduction.map_batch_chunks = 4;
        let required = config.cohort_requirements(32).unwrap();
        assert_eq!(required.native.kv_bytes, row.kv_bytes * 4);
        assert_eq!(required.native.rope_bytes, row.rope_bytes);
        assert!(required.native.scratch_payload_bound > row.scratch_payload_bound);
        assert!(required.workspace_bytes > config.budget.max_output_bytes * 4);
        assert_eq!(config.budget, task_budget);
        assert_eq!(config.budget.max_kv_bytes, row.kv_bytes);
        assert_eq!(config.workspace_bytes(7).unwrap(), frontier);
        map_requirements(&config, 16, 32, &planner).unwrap();
    }
}

#[test]
fn every_cohort_peak_component_is_independently_bounded() {
    let (_, mut config) = fixture(SourceMapTask::Ner(Default::default()), 32);
    config.mapping.reduction.map_batch_chunks = 3;
    let ceiling = config.cohort_requirements(32).unwrap();
    check_peak(ceiling, ceiling).unwrap();
    for axis in 0..5 {
        let mut required = ceiling;
        match axis {
            0 => required.native.kv_bytes += 1,
            1 => required.native.rope_bytes += 1,
            2 => required.native.scratch_payload_bound += 1,
            3 => required.workspace_bytes += 1,
            _ => required.max_result_bytes += 1,
        }
        assert!(check_peak(required, ceiling).is_err(), "resource axis {axis}");
    }
    config.mapping.reduction.map_batch_chunks = 1;
    check_peak(config.cohort_requirements(32).unwrap(), ceiling).unwrap();
}

#[test]
fn invalid_source_windows_and_result_overflow_fail_before_native_allocation() {
    let (_, mut config) = fixture(SourceMapTask::Keyphrases(Default::default()), 32);
    for chunks in [0, 9, 65, usize::MAX] {
        config.mapping.reduction.map_batch_chunks = chunks;
        assert!(config.cohort_requirements(32).is_err(), "window {chunks}");
    }
    config.mapping.reduction.map_batch_chunks = 2;
    config.budget.max_output_bytes = u64::MAX;
    assert!(config.cohort_requirements(32).is_err());
    config.budget.max_output_bytes = 65_536;
    config.native.context_tokens = 0;
    assert!(config.cohort_requirements(32).is_err());
}

#[test]
fn owned_source_cohort_result_can_cross_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapCohortRun>();
    send::<HostedOutput<Int4SourceMapCohortRun>>();
}
