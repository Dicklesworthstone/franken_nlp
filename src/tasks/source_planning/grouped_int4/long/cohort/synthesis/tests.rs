//! Phase peaks are independent of final evidence size or early map completion.
use super::*;

fn discovery(contexts: &[usize], width: usize) -> Int4SourceMapCohortRequirement {
    Int4SourceMapCohortRequirement {
        native: Int4MemoryRequirement::for_cohort(contexts, width).unwrap(),
        workspace_bytes: 1 << 20, max_result_bytes: 1 << 20,
    }
}

#[test]
fn full_synthesis_context_is_reserved_even_when_discovery_has_one_short_row() {
    for width in [16, 32, 128] {
        let map = discovery(&[64], width);
        let scalar = Int4MemoryRequirement::for_context(8192, width).unwrap();
        let peak = map.with_synthesis_context(8192, width, scalar.kv_bytes).unwrap();
        assert_eq!(peak.native.kv_bytes, scalar.kv_bytes.max(map.native.kv_bytes));
        assert_eq!(peak.native.rope_bytes, scalar.rope_bytes.max(map.native.rope_bytes));
        assert_eq!(peak.native.scratch_payload_bound,
            scalar.scratch_payload_bound.max(map.native.scratch_payload_bound));
        assert_eq!(peak.workspace_bytes, map.workspace_bytes);
        assert_eq!(peak.max_result_bytes, map.max_result_bytes);
        assert!(peak.native.kv_bytes > map.native.kv_bytes);
        assert!(map.with_synthesis_context(8192, width, scalar.kv_bytes - 1).is_err());
    }
}

#[test]
fn separate_physical_phases_price_the_componentwise_peak_instead_of_simultaneous_engines() {
    let map = discovery(&[4096, 4096, 4096], 32);
    let scalar = Int4MemoryRequirement::for_context(8192, 32).unwrap();
    let peak = map.with_synthesis_context(8192, 32, scalar.kv_bytes).unwrap();
    assert_eq!(peak.native.kv_bytes, map.native.kv_bytes);
    assert_eq!(peak.native.rope_bytes, scalar.rope_bytes.max(map.native.rope_bytes));
    assert!(peak.native.kv_bytes < map.native.kv_bytes + scalar.kv_bytes);
    assert_eq!(peak, peak.with_synthesis_context(8192, 32, scalar.kv_bytes).unwrap());
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn stats() -> Int4SourceMapCohortStats {
    let work = constrained_int4::planned_work(32, 32, 2).unwrap();
    Int4SourceMapCohortStats { chunks: 1, cohorts: 1, group_steps: work.forward_positions,
        projection_groups: 2, planned_work: work, model_work: work }
}

#[test]
fn wrapper_reconciles_discovery_without_charging_blank_source_windows_or_synthesis_as_maps() {
    let stats = stats();
    let run = finish("complete document", stats, 1, 3, stats.planned_work, stats.model_work,
        4096, &mut Continue).unwrap();
    assert_eq!(run.discovery_cohorts, stats);
    assert_eq!(run.document, "complete document");
    for axis in 0..7 {
        let mut wrong = stats;
        match axis { 0 => wrong.chunks += 1, 1 => wrong.cohorts = 4,
            2 => wrong.planned_work.forward_positions += 1,
            3 => wrong.model_work.projections.multiply_accumulates += 1,
            4 => wrong.group_steps += 1, 5 => wrong.projection_groups = 0,
            _ => wrong.projection_groups = 3 }
        assert!(finish("complete document", wrong, 1, 3, stats.planned_work, stats.model_work,
            4096, &mut Continue).is_err());
    }
    assert!(finish("complete document", stats, 1, 3, stats.planned_work, stats.model_work,
        u64::MAX, &mut Continue).is_err());
}
