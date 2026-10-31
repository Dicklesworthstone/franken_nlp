//! Pure finite-plan admission fixtures, without native inference claims.
use super::*;
use crate::grammar::mask::MaskWorkLimits;

fn limits() -> SourceLimits {
    SourceLimits { native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 65536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10_000,
            cleanup_reserve_bytes: 65536 } }, preparation_reserve_bytes: 16 << 20,
        mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_mask_node_visits: 4000 }
}
fn quote(prompt: usize, tokens: usize) -> Quote {
    let work = constrained_int4::planned_work(32, prompt, tokens).unwrap();
    Quote { work, output_bytes: 8192, output_tokens: tokens as u64,
        max_kv_bytes: work.forward_positions * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64,
        group_width: 32 }
}
#[test]
fn independent_attention_triangles_and_contexts_survive_finite_collection_pricing() {
    let a = quote(8, 4); let b = quote(21, 7);
    let work = a.work.checked_add(b.work).unwrap();
    let result = price([Ok(a), Ok(b)].into_iter(), 2, 32, limits(), 32768).unwrap();
    assert_eq!(result.contexts, vec![11, 27]);
    assert_eq!(result.work, work); assert_eq!(result.output_tokens, 11);
    assert!(result.work.attention_pairs < Int4Work::for_sequence(32, 0, 38,
        result.work.projected_logits as usize).unwrap().attention_pairs);
    assert!(required_workspace_bytes(2, result.output_tokens, 32768).unwrap() > 8 * 32768);
}
#[test]
fn the_last_documents_kv_output_or_group_mismatch_prevents_all_native_allocation() {
    for axis in 0..6 {
        let mut late = quote(21, 7);
        match axis { 0 => late.max_kv_bytes -= 1, 1 => late.group_width = 16,
            2 => late.output_tokens = 0, 3 => late.output_tokens = 100,
            4 => late.output_bytes = 0, _ => late.output_bytes = 32768 }
        assert!(price([Ok(quote(8, 4)), Ok(late)].into_iter(), 2, 32, limits(), 32768).is_err(), "axis {axis}");
    }
    assert!(price([Ok(quote(8, 4))].into_iter(), 2, 32, limits(), 32768).is_err());
    assert!(price([Ok(quote(8, 4)), Ok(quote(8, 4))].into_iter(), 1, 32, limits(), 32768).is_err());
}
#[test]
fn output_envelopes_and_mask_allowances_are_explicit_checked_resources() {
    let cap = 4096 + 2 + 2 * 8192;
    price([Ok(quote(8, 4)), Ok(quote(21, 7))].into_iter(), 2, 32, limits(), cap).unwrap();
    assert!(price([Ok(quote(8, 4)), Ok(quote(21, 7))].into_iter(), 2, 32, limits(), cap - 1).is_err());
    assert_eq!(aggregate_masks(4000, 2).unwrap(), 8000);
    assert!(aggregate_masks(0, 2).is_err()); assert!(aggregate_masks(u64::MAX, 2).is_err());
    for count in [0, 65, usize::MAX] { assert!(validate(limits(), cap, count).is_err()); }
}
#[test]
fn public_collection_owners_cross_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Vec<Int4ExtractPlan>>(); send::<Int4ExtractCohortRun>();
    send::<Vec<PreparedInt4SourceTask>>(); send::<Int4SourceCohortRun>();
}
