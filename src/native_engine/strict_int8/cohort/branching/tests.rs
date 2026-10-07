//! Real KV suffix handling and checked work arithmetic; no model-weight fixture.
use super::*;
fn row(positions: usize) -> Sequence {
    let mut cache = KvCache::try_with_capacity(positions + 1).unwrap();
    for slot in 0..KV_SLOT_COUNT {
        for position in 0..positions {
            cache.append(slot, position, &vec![0; K], &vec![Bf16::from_f32((position + 1) as f32).to_bits(); K]).unwrap();
        }
    }
    Sequence { cache, hidden: Some(vec![Bf16::from_f32(3.0); H]) }
}
#[test]
fn rewind_discards_all_loops_and_invalidates_only_the_target_hidden() {
    let mut rows = [row(5), row(3)];
    truncate(&mut rows[0], 5, 2).unwrap();
    assert!(rows[0].cache.all_slots_have_len(2)); assert!(rows[0].hidden.is_none());
    assert!(rows[1].cache.all_slots_have_len(3)); assert!(rows[1].hidden.is_some());
    let discarded = quote(5, 2, 0, 5).unwrap();
    assert_eq!(discarded, 3); verify_position(2, discarded, 5).unwrap();
    assert!(verify_position(2, 0, 5).is_err());
}
#[test]
fn unchanged_prefix_keeps_its_hidden_but_cannot_extend_without_a_forward() {
    let mut row = row(3); truncate(&mut row, 3, 3).unwrap();
    assert!(row.hidden.is_some()); assert_eq!(quote(3, 3, 2, 5).unwrap(), 2);
    assert!(truncate(&mut row, 3, 4).is_err());
    assert!(row.cache.all_slots_have_len(3)); assert!(row.hidden.is_some());
    assert!(quote(3, 4, 2, 5).is_err());
}
#[test]
fn partial_loop_state_refuses_before_touching_hidden_or_any_other_slot() {
    let mut row = row(3);
    row.cache.append(0, 3, &vec![0; K], &vec![0; K]).unwrap();
    assert!(truncate(&mut row, 3, 1).is_err()); assert!(row.hidden.is_some());
    assert_eq!(row.cache.len_for_slot(0).unwrap(), 4);
    for slot in 1..KV_SLOT_COUNT { assert_eq!(row.cache.len_for_slot(slot).unwrap(), 3); }
}
#[test]
fn branch_depth_prices_attention_without_refunding_decoder_or_projection_work() {
    let prompt = Int8Work::for_sequence(0, 3, V).unwrap();
    let deep = Int8Work::for_sequence(3, 3, V).unwrap();
    let sibling = Int8Work::for_sequence(4, 1, V).unwrap();
    let expected = prompt.checked_add(deep).unwrap().checked_add(sibling).unwrap();
    let mut account = Account { budget: Int8RunBudget::exact(expected), work: prompt };
    let delta = account.quote(3, 8, 3, V).unwrap(); account.work = account.work.checked_add(delta).unwrap();
    let discarded = quote(6, 4, 0, account.work.forward_positions).unwrap();
    let delta = account.quote(4, 8, 1, V).unwrap(); account.work = account.work.checked_add(delta).unwrap();
    verify_position(5, discarded, account.work.forward_positions).unwrap();
    assert_eq!(account.work, expected); assert_eq!(account.work.forward_positions, 7);
    assert!(account.quote(5, 8, 1, 0).is_err()); assert!(account.quote(5, 8, 0, 1).is_err());
    assert!(expected.attention_pairs < Int8Work::for_sequence(0, 7, 3 * V).unwrap().attention_pairs);
}
#[test]
fn repeated_rewinds_preserve_an_exact_nonrenewable_position_equation() {
    let mut discarded = 0_u64; let mut position = 7_usize; let mut forwards = 7_u64;
    for retain in [3, 4, 1, 0] {
        discarded = quote(position, retain, discarded, forwards).unwrap(); position = retain;
        verify_position(position, discarded, forwards).unwrap();
        position += 2; forwards += 2; verify_position(position, discarded, forwards).unwrap();
    }
    assert!(verify_position(position, discarded + 1, forwards).is_err());
    assert!(verify_position(1, u64::MAX, 0).is_err());
}
#[test]
fn existing_cleanup_clears_branched_kv_and_preserves_failure_poison() {
    let mut rows = [row(4), row(2)]; truncate(&mut rows[0], 4, 1).unwrap();
    let mut state = RunState { active: true, poisoned: true }; clear(&mut rows, &mut state);
    assert!(rows.iter().all(|row| row.hidden.is_none() && row.cache.all_slots_have_len(0)));
    assert!(!state.active); assert!(state.poisoned); assert!(state.open().is_err());
}
