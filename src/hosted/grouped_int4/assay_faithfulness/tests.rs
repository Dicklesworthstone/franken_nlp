//! Pure reservation and nonrenewable work contracts; no model required.
use super::*;
use crate::native_engine::portable_int8::ProjectionWork;
fn host() -> Int4FaithfulnessAssayHostLimits {
    Int4FaithfulnessAssayHostLimits { native: NativeLimits { context_tokens: 512, allocator_reserve_bytes: 65536,
        run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } },
        task: TaskBudget { max_input_tokens: 511, max_output_tokens: 2, max_output_bytes: 100_000,
            max_grammar_states: 7, max_kv_bytes: Int4MemoryRequirement::for_context(512, 128).unwrap().kv_bytes },
        max_heads: 32, preparation_reserve_bytes: 512 * 1024 * 1024,
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } } }
}
fn row() -> Int4Work { Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
    projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } } }
fn axis(work: &mut Int4Work, index: usize) -> &mut u64 {
    match index { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        4 => &mut work.projections.multiply_accumulates, _ => panic!("unknown axis") }
}
#[test]
fn paired_residents_reuse_one_native_envelope_but_keep_both_private_replays() {
    let host = host(); let limits = metrics::Limits::default();
    let left = host.requirements(&[3], limits).unwrap(); let right = host.requirements(&[10752], limits).unwrap();
    let pair = host.requirements(&[3, 10752], limits).unwrap();
    assert_eq!(pair, host.requirements(&[10752, 3], limits).unwrap());
    assert_eq!(pair.native, right.native); assert_eq!(pair.workspace_bytes, right.workspace_bytes);
    assert_eq!(pair.native.kv_bytes, left.native.kv_bytes); assert_eq!(pair.output_reservation_bytes, left.output_reservation_bytes);
    assert_eq!(pair.input_bytes - left.input_bytes, native_faithfulness::required_replay_bytes(limits, 2).unwrap()
        - native_faithfulness::required_replay_bytes(limits, 1).unwrap());
    assert_eq!(left.output_reservation_bytes, 4 * limits.max_output_bytes as u64 + 4096);
    let mut more = host; more.max_heads += 1; assert!(more.requirements(&[128], limits).is_err());
    let mut fewer = host; fewer.max_heads -= 1;
    assert_eq!(left.input_bytes - fewer.requirements(&[3], limits).unwrap().input_bytes,
        host.native.context_tokens as u64 * 256 + 65536);
}
#[test]
fn every_head_claim_and_arm_spends_one_atomic_five_axis_work_ledger() {
    let complete = row().checked_add(row()).unwrap();
    for field in 0..5 {
        let mut ceiling = complete; *axis(&mut ceiling, field) -= 1;
        let mut spent = Int4Work::default(); charge_work(&mut spent, row(), ceiling).unwrap();
        let before = spent;
        assert!(matches!(charge_work(&mut spent, row(), ceiling),
            Err(HostedError::Int4(Int4HostError::Judge(Int4JudgeError::WorkBudget)))));
        assert_eq!(spent, before);
        *axis(&mut spent, field) = u64::MAX; let before = spent;
        assert!(charge_work(&mut spent, row(), host().max_model_work).is_err()); assert_eq!(spent, before);
    }
}
#[test]
fn zero_or_undersized_complete_resource_axes_refuse_without_model_io() {
    for field in 0..12 {
        let mut value = host();
        match field { 0 => value.task.max_kv_bytes -= 1, 1 => value.native.context_tokens = 0,
            2 => value.preparation_reserve_bytes = 0, 3 => value.native.allocator_reserve_bytes = 0,
            4 => value.max_heads = 0, 5 => value.task.max_output_tokens = 1, 6 => value.native.run.max_checkpoints = 1,
            _ => *axis(&mut value.max_model_work, field - 7) = 0 }
        assert!(value.requirements(&[128], metrics::Limits::default()).is_err(), "field {field}");
    }
    for widths in [&[][..], &[0][..], &[10753][..], &[1, 2, 3][..]] {
        assert!(host().requirements(widths, metrics::Limits::default()).is_err());
    }
}
