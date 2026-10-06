//! Model-free admission arithmetic, not runtime or neural-model evidence.
use super::*;
use crate::grammar::mask::MaskWorkLimits;
fn limits() -> SourceLimits {
    SourceLimits { native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 4096,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 1000, cleanup_reserve_bytes: 4096 } },
        preparation_reserve_bytes: 4096, mask_limits: MaskWorkLimits::default(), max_mask_node_visits: 10000 }
}
fn quote(context: usize, output: u64) -> Quote {
    Quote { work: Int8Work::for_sequence(0, context, 6).unwrap(), output_bytes: output,
        output_tokens: 3, max_kv_bytes: Int8MemoryRequirement::for_context(context).unwrap().kv_bytes }
}
#[test]
fn every_admission_axis_is_required_before_native_allocation() {
    validate(limits(), 4096, 2).unwrap();
    for axis in 0..8 {
        let mut l = limits(); let mut cap = 4096; let mut count = 2;
        match axis { 0 => count = 0, 1 => count = MAX_BATCH_ROWS + 1, 2 => cap = 0,
            3 => l.preparation_reserve_bytes = 0, 4 => l.native.allocator_reserve_bytes = 0,
            5 => l.mask_limits.max_trie_node_visits = 0, 6 => l.mask_limits.checkpoint_interval_nodes = 0,
            _ => l.max_mask_node_visits = 0 }
        assert!(validate(l, cap, count).is_err());
    }
    assert!(require_selected(false).is_err()); require_selected(true).unwrap();
}
#[test]
fn aggregate_work_sums_independent_attention_triangles_and_retains_all_results() {
    let expected = quote(17, 8000).work.checked_add(quote(8, 10000).work).unwrap();
    let cap = 4096 + 2 + 8000 + 10000;
    let priced = price([Ok(quote(17, 8000)), Ok(quote(8, 10000))].into_iter(), 2, limits(), cap).unwrap();
    assert_eq!(priced.contexts, [17,8]); assert_eq!(priced.work, expected); assert_eq!(priced.output_tokens, 6);
    assert_ne!(priced.work.attention_pairs, Int8Work::for_sequence(0, 25, 12).unwrap().attention_pairs);
    assert!(price([Ok(quote(17, 8000)), Ok(quote(8, 10000))].into_iter(), 2, limits(), cap - 1).is_err());
}
#[test]
fn missing_extra_and_overflowing_row_prices_cannot_change_cohort_membership() {
    assert!(price([Ok(quote(1, 10))].into_iter(), 2, limits(), 10000).is_err());
    assert!(price([Ok(quote(1, 10)), Ok(quote(1, 10))].into_iter(), 1, limits(), 10000).is_err());
    assert!(price([Ok(quote(1, u64::MAX))].into_iter(), 1, limits(), u64::MAX).is_err());
}
#[test]
fn mask_limits_and_each_row_budget_are_not_aggregate_allowances_in_disguise() {
    assert_eq!(aggregate_masks(11, 3).unwrap(), 33);
    assert!(aggregate_masks(u64::MAX, 2).is_err()); assert!(aggregate_masks(0, 2).is_err());
    let q = quote(17, 1024); let b = row_budget(q.work, 17, limits()).unwrap();
    assert_eq!(b.native.max_forward_positions, 17);
    assert_eq!(b.native.max_projection_work, q.work.projections);
    assert_eq!(b.json.max_kv_bytes, q.max_kv_bytes);
    assert_eq!(b.json.max_projected_logits, 6);
    assert_eq!(b.json.max_total_mask_node_visits, limits().max_mask_node_visits);
}
#[test]
fn complete_resident_kv_capacity_must_fit_each_tasks_exact_authority() {
    let mut q = quote(17, 1024); q.max_kv_bytes -= 1;
    assert!(price([Ok(q)].into_iter(), 1, limits(), 10000).is_err());
    let mut l = limits(); l.native.context_tokens = 16;
    assert!(price([Ok(quote(17, 1024))].into_iter(), 1, l, 10000).is_err());
}
#[test]
fn both_closed_input_and_result_families_cross_the_owned_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Input<Int8ExtractPlan>>(); send::<Input<PreparedInt8SourceTask>>();
    send::<Int8ExtractCohortRun>(); send::<Int8SourceCohortRun>();
}
