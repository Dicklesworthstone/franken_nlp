//! Source-only static admission and phase-ledger fixtures; no native execution.
use super::*;
use super::super::tests as hierarchy;
use crate::tasks::source_planning::grouped_int4::long::{tests as maps,
    cohort::Int4SourceMapCohortMemoryLimits};

#[test]
fn every_reduction_window_reserves_full_context_without_changing_semantic_authority() {
    let planner = maps::planner();
    for width in [16, 32, 127] { for groups in [1, 2, 4] { for discovery_cohorts in [false, true] {
        let mut mapping = maps::limits(); mapping.reduction.map_batch_chunks = 4;
        let prepared = hierarchy::prepare(&planner, "éabc----βdef", mapping, hierarchy::limits(), width).unwrap();
        let expected = prepared.preflight_metadata();
        let identities = prepared.execution_identities().cloned().collect::<Vec<_>>();
        let context = prepared.base.planning.max_context_tokens;
        let task = prepared.base.budget;
        let schedule = Int4HierarchyCohortSchedule { max_groups: groups, discovery_cohorts };
        let full = Int4SourceMapCohortMemoryLimits { max_chunks: groups, context_tokens: context,
            max_output_tokens_per_chunk: u64::from(task.max_output_tokens),
            max_result_bytes_per_chunk: task.max_output_bytes }.requirements(width).unwrap();
        let peak = prepared.hierarchy_cohort_requirements(context, schedule).unwrap();
        if discovery_cohorts {
            let discovery = prepared.base.map.cohort_requirements(context).unwrap();
            assert_eq!(peak.native.kv_bytes, full.native.kv_bytes.max(discovery.native.kv_bytes));
            assert_eq!(peak.native.rope_bytes, full.native.rope_bytes.max(discovery.native.rope_bytes));
            assert_eq!(peak.native.scratch_payload_bound,
                full.native.scratch_payload_bound.max(discovery.native.scratch_payload_bound));
            assert_eq!(peak.workspace_bytes, full.workspace_bytes.max(discovery.workspace_bytes));
            assert_eq!(peak.max_result_bytes, full.max_result_bytes.max(discovery.max_result_bytes));
        } else { assert_eq!(peak, full); }
        assert_eq!(prepared.preflight_metadata(), expected);
        assert_eq!(prepared.execution_identities().cloned().collect::<Vec<_>>(), identities);
        assert_eq!(prepared.base.budget, task);
        let mut work = prepared.base.map.planned_work();
        for _ in 0..expected.limits.max_passes {
            work = add_work(work, expected.discovery.synthesis_reserve).unwrap();
        }
        assert_eq!(expected.reserved_model_work(), work);
        assert_eq!(expected.reserved_mask_visits(), 6000);
    } } }
}

#[test]
fn invalid_schedule_context_and_original_row_kv_refuse_before_discovery() {
    let planner = maps::planner();
    for discovery_cohorts in [false, true] {
        let mut prepared = hierarchy::prepare(&planner, "source", maps::limits(), hierarchy::limits(), 32).unwrap();
        let context = prepared.base.planning.max_context_tokens;
        for max_groups in [0, 5, 65, usize::MAX] {
            let schedule = Int4HierarchyCohortSchedule { max_groups, discovery_cohorts };
            assert!(matches!(prepared.hierarchy_cohort_requirements(context, schedule), Err(Int4SourceMapError::InvalidLimits)));
        }
        let schedule = Int4HierarchyCohortSchedule { max_groups: 2, discovery_cohorts };
        assert!(prepared.base.map.cohort_requirements(context - 1).is_ok());
        assert!(matches!(prepared.hierarchy_cohort_requirements(context - 1, schedule), Err(Int4SourceMapError::Admission)));
        prepared.base.budget.max_kv_bytes = Int4MemoryRequirement::for_context(context, 32).unwrap().kv_bytes - 1;
        assert!(prepared.base.map.cohort_requirements(context).is_ok());
        assert!(matches!(prepared.hierarchy_cohort_requirements(context, schedule), Err(Int4SourceMapError::Admission)));
    }
}

#[test]
fn physical_pass_ledger_counts_windows_per_level_and_excludes_discovery() {
    let (expected, document) = hierarchy::complete(false);
    expected.verify_completed(&document).unwrap();
    assert_eq!(document.levels.len(), 2); assert_eq!(document.passes.len(), 3);
    let mut planned = Int4Work::default(); let mut actual = Int4Work::default();
    for pass in &document.passes {
        planned = add_work(planned, pass.planned_model_work).unwrap();
        actual = add_work(actual, pass.native.model_work).unwrap();
    }
    // These are validator fixtures, not measured native execution counters.
    for max_groups in [1, 2, 4] {
        let schedule = Int4HierarchyCohortSchedule { max_groups, discovery_cohorts: false };
        let stats = Int4HierarchyCohortStats { cohorts: if max_groups == 1 { 3 } else { 2 }, passes: 3,
            group_steps: actual.forward_positions, projection_groups: 3, planned_work: planned, model_work: actual };
        check_completed(&document, schedule, stats).unwrap();
        for axis in 0..6 {
            let mut wrong = stats;
            match axis { 0 => wrong.cohorts = 1, 1 => wrong.passes += 1,
                2 => wrong.planned_work = document.planned_model_work,
                3 => wrong.model_work = document.model_work,
                4 => wrong.group_steps = 0, _ => wrong.projection_groups = 0 }
            assert!(check_completed(&document, schedule, wrong).is_err(), "{axis}");
        }
    }
}

#[test]
fn no_initial_evidence_cannot_fabricate_a_hierarchy_native_window() {
    let (_, document) = hierarchy::complete(true);
    let schedule = Int4HierarchyCohortSchedule { max_groups: 4, discovery_cohorts: false };
    check_completed(&document, schedule, Int4HierarchyCohortStats::default()).unwrap();
    for axis in 0..4 {
        let mut wrong = Int4HierarchyCohortStats::default();
        match axis { 0 => wrong.cohorts = 1, 1 => wrong.passes = 1,
            2 => wrong.group_steps = 1, _ => wrong.projection_groups = 1 }
        assert!(check_completed(&document, schedule, wrong).is_err());
    }
    assert_eq!(document.model_work, document.discovery.model_work);
    assert!(document.passes.is_empty());
}
