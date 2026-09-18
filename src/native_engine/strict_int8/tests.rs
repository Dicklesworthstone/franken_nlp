use super::*;

#[test]
fn every_linear_of_both_loops_and_untied_head_is_priced() {
    let work = Int8Work::for_sequence(0, 1, V).unwrap();
    assert_eq!(work.forward_positions, 1);
    assert_eq!(work.projections.dot_products, 1_743_104);
    assert_eq!(work.projections.multiply_accumulates, 6_808_141_824);
    assert_eq!(work.attention_pairs, 44 * 48);
    assert_eq!(work.projected_logits, V as u64);
}
#[test]
fn prefill_projects_only_the_final_head_but_every_decoder_position() {
    let one = Int8Work::for_sequence(0, 1, 0).unwrap();
    let prefill = Int8Work::for_sequence(0, 4, 3).unwrap();
    assert_eq!(prefill.projections.dot_products, one.projections.dot_products * 4 + 3);
    assert_eq!(prefill.projections.multiply_accumulates, one.projections.multiply_accumulates * 4 + 3 * H as u64);
    assert_eq!(prefill.attention_pairs, 44 * 48 * 10);
}
#[test]
fn attention_work_includes_existing_context_and_quadratic_prefill() {
    let a = Int8Work::for_sequence(0, 3, 0).unwrap();
    let b = Int8Work::for_sequence(3, 2, 0).unwrap();
    let all = Int8Work::for_sequence(0, 5, 0).unwrap();
    assert_eq!(a.attention_pairs + b.attention_pairs, all.attention_pairs);
    assert_eq!(b.attention_pairs, 44 * 48 * (4 + 5));
}
#[test]
fn zero_forward_head_projection_has_no_attention_or_decoder_work() {
    let work = Int8Work::for_sequence(123, 0, 7).unwrap();
    assert_eq!(work.attention_pairs, 0); assert_eq!(work.forward_positions, 0);
    assert_eq!(work.projections, ProjectionWork::for_shape(7, H).unwrap());
}
#[test]
fn work_arithmetic_overflow_refuses() {
    assert!(Int8Work::for_sequence(usize::MAX, 1, 0).is_err());
    assert!(Int8Work::for_sequence(0, usize::MAX, 0).is_err());
    assert!(Int8Work::for_sequence(0, 0, usize::MAX).is_err());
}
#[test]
fn memory_checks_all_44_slots_and_two_rope_tables_before_allocation() {
    let need = Int8MemoryRequirement::for_context(2).unwrap();
    assert_eq!(need.kv_bytes, 2 * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(need.rope_bytes, (2 * 128 + 64) * 4);
    let exact = Int8MemoryBudget { max_kv_bytes: need.kv_bytes, max_rope_bytes: need.rope_bytes,
        max_scratch_payload_bytes: need.scratch_payload_bound };
    assert!(need.check(exact).is_ok());
    assert!(need.check(Int8MemoryBudget { max_kv_bytes: exact.max_kv_bytes - 1, ..exact }).is_err());
    assert!(need.check(Int8MemoryBudget { max_rope_bytes: exact.max_rope_bytes - 1, ..exact }).is_err());
    assert!(need.check(Int8MemoryBudget { max_scratch_payload_bytes: exact.max_scratch_payload_bytes - 1, ..exact }).is_err());
}
#[test]
fn source_config_maximum_is_not_silently_admitted() {
    assert!(Int8MemoryRequirement::for_context(0).is_err());
    assert!(Int8MemoryRequirement::for_context(DEFAULT_ADMITTED_CONTEXT_CAP + 1).is_err());
}
#[test]
fn geometry_refuses_transposes_even_if_element_counts_agree() {
    assert!(weights::geometry(Q, H, Q, H).is_ok());
    assert!(weights::geometry(H, Q, Q, H).is_err());
    assert!(weights::geometry(H, H, Q, H).is_err());
}
#[test]
fn successful_session_cleanup_retains_capacity_and_clears_every_slot() {
    let mut cache = KvCache::try_with_capacity(1).unwrap();
    let bits = vec![0; K];
    for slot in 0..KV_SLOT_COUNT { cache.append(slot, 0, &bits, &bits).unwrap(); }
    let mut state = RunState::default(); state.open().unwrap();
    end_session(&mut cache, &mut state);
    assert!(cache.all_slots_have_len(0)); assert_eq!(cache.capacity_positions(), 1);
    assert!(state.open().is_ok());
}
#[test]
fn partial_forward_cleanup_does_not_unpoison_the_engine() {
    let mut cache = KvCache::try_with_capacity(1).unwrap();
    cache.append(0, 0, &vec![0; K], &vec![0; K]).unwrap();
    let mut state = RunState::default(); state.open().unwrap(); state.poisoned = true;
    end_session(&mut cache, &mut state);
    assert!(cache.all_slots_have_len(0)); assert_eq!(state.open(), Err(StrictInt8Error::EngineUnavailable));
}
#[test]
fn forgotten_or_reentrant_session_cannot_reopen_the_engine() {
    let mut state = RunState::default(); state.open().unwrap();
    assert_eq!(state.open(), Err(StrictInt8Error::EngineUnavailable));
    assert!(state.check().is_ok()); state.poisoned = true;
    assert_eq!(state.check(), Err(StrictInt8Error::EngineUnavailable));
}
#[test]
fn norm_rejects_variance_overflow_instead_of_returning_zero() {
    assert_eq!(norm(&[Bf16::from_f32(1.0e30)], &[Bf16::from_f32(1.0)]), Err(StrictInt8Error::Primitive));
    let values = [Bf16::from_f32(1.0), Bf16::from_f32(-2.0)];
    let scale = [Bf16::from_f32(1.0); 2];
    assert_eq!(norm(&values, &scale).unwrap(), rms_norm_f32_reduce_cast_back(&values, &scale, RMS_NORM_EPSILON).unwrap());
}
#[test]
fn projection_cancellation_keeps_its_typed_cause() {
    assert_eq!(StrictInt8Error::from(LinearError::Cancelled(DecodeCancellationKind::Shutdown)),
        StrictInt8Error::Cancelled(DecodeCancellationKind::Shutdown));
}
