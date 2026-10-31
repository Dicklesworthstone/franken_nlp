use super::*;

fn task(context: usize) -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 64, max_output_bytes: 16384,
        max_grammar_states: 16384, max_kv_bytes: Int4MemoryRequirement::for_context(context, 32).unwrap().kv_bytes }
}

#[test]
fn schedule_bounds_and_concurrent_preparation_are_checked_before_planning() {
    for groups in [0, 65, usize::MAX] {
        assert!(Int4HierarchyCohortSchedule { max_groups: groups, discovery_cohorts: false }.validate(256).is_err());
    }
    let one = Int4HierarchyCohortSchedule { max_groups: 1, discovery_cohorts: false };
    assert!(one.validate(0).is_err());
    assert!(one.validate(257).is_err());
    assert_eq!(one.additional_preparation_bytes(4096).unwrap(), 0);
    assert!(one.additional_preparation_bytes(0).is_err());
    let three = Int4HierarchyCohortSchedule { max_groups: 3, ..one };
    assert!(three.validate(2).is_err());
    three.validate(3).unwrap();
    assert_eq!(three.additional_preparation_bytes(4096).unwrap(), 8192);
    assert!(three.additional_preparation_bytes(u64::MAX).is_err());
}

#[test]
fn phase_peak_keeps_full_group_context_and_original_task_authority() {
    let context = 4096;
    let budget = task(context);
    let group_only = Int4HierarchyCohortSchedule { max_groups: 3, discovery_cohorts: false };
    let grouped = group_only.requirements(None, context, 32, budget).unwrap();
    let expected = Int4SourceMapCohortMemoryLimits { max_chunks: 3, context_tokens: context,
        max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: 16384 }.requirements(32).unwrap();
    assert_eq!(grouped.native.kv_bytes, expected.native.kv_bytes);
    assert_eq!(grouped.workspace_bytes, expected.workspace_bytes);
    assert_eq!(grouped.max_result_bytes, expected.max_result_bytes);
    let discovery = Int4SourceMapCohortMemoryLimits { max_chunks: 5, context_tokens: 2048,
        max_output_tokens_per_chunk: 64, max_result_bytes_per_chunk: 16384 }.requirements(32).unwrap();
    let both = Int4HierarchyCohortSchedule { discovery_cohorts: true, ..group_only };
    let combined = both.requirements(Some(discovery), context, 32, budget).unwrap();
    assert_eq!(combined, maximum(discovery, grouped));
    assert!(combined.native.kv_bytes < discovery.native.kv_bytes + grouped.native.kv_bytes);
    assert!(group_only.requirements(Some(discovery), context, 32, budget).is_err());
    assert!(both.requirements(None, context, 32, budget).is_err());
    let too_small = TaskBudget { max_kv_bytes: budget.max_kv_bytes - 1, ..budget };
    assert!(group_only.requirements(None, context, 32, too_small).is_err());
}

#[test]
fn hierarchy_ledger_separates_passes_from_discovery_and_rejects_tampering() {
    let empty = Int4HierarchyCohortStats::default();
    check_stats(empty, 0, 0, Int4Work::default(), Int4Work::default()).unwrap();
    assert!(check_stats(empty, 1, 0, Int4Work::default(), Int4Work::default()).is_err());
    let row = constrained_int4::planned_work(32, 32, 2).unwrap();
    let work = add_work(row, row).unwrap();
    let stats = Int4HierarchyCohortStats { cohorts: 1, passes: 2,
        group_steps: row.forward_positions, projection_groups: 2, planned_work: work, model_work: work };
    check_stats(stats, 1, 2, work, work).unwrap();
    for axis in 0..7 {
        let mut wrong = stats;
        match axis { 0 => wrong.cohorts = 0, 1 => wrong.passes += 1,
            2 => wrong.planned_work.forward_positions += 1,
            3 => wrong.model_work.projections.multiply_accumulates += 1,
            4 => wrong.group_steps = work.forward_positions + 1,
            5 => wrong.projection_groups = 0, _ => wrong.projection_groups = 5 }
        assert!(check_stats(wrong, 1, 2, work, work).is_err());
    }
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn empty_evidence_retains_no_fabricated_native_pass_or_discovery_ledger() {
    let result = finish("empty evidence", None, Int4HierarchyCohortStats::default(), 1024, &mut Continue).unwrap();
    let json = serde_json::to_value(result).unwrap();
    assert!(json.get("discovery_cohorts").is_none());
    assert_eq!(json["hierarchy_cohorts"]["passes"], 0);
    assert!(finish("empty evidence", None, Int4HierarchyCohortStats::default(), u64::MAX, &mut Continue).is_err());
}
