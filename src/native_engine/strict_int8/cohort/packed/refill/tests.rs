//! Model-free ownership/accounting fixtures, not native-model qualification.
use super::*;
fn budget(tokens: usize) -> Int8RunBudget {
    Int8RunBudget::exact(Int8Work::for_sequence(0, tokens, V).unwrap())
}
#[test]
fn finished_slot_accepts_next_request_while_another_slot_remains_live() {
    let mut queue = Tickets::new(&[budget(2), budget(9), budget(3), budget(1)], 2).unwrap();
    let a = queue.next_admission().unwrap().unwrap(); queue.install(a).unwrap();
    let b = queue.next_admission().unwrap().unwrap(); queue.install(b).unwrap();
    assert_eq!(a, RefillAdmission { sequence: 0, request: 0 });
    assert_eq!(b, RefillAdmission { sequence: 1, request: 1 });
    assert!(queue.next_admission().is_err());
    queue.retire(0).unwrap();
    let c = queue.next_admission().unwrap().unwrap(); queue.install(c).unwrap();
    assert_eq!(c, RefillAdmission { sequence: 0, request: 2 });
    assert_eq!(queue.request(1).unwrap(), 1);
    assert_eq!(queue.budgets[c.request].max_forward_positions, 3);
    queue.retire(0).unwrap();
    let d = queue.next_admission().unwrap().unwrap(); queue.install(d).unwrap();
    assert_eq!(d.request, 3); assert!(queue.next_admission().unwrap().is_none());
    queue.retire(1).unwrap(); assert!(!queue.complete());
    queue.retire(0).unwrap(); assert!(queue.complete());
    assert!(queue.next_admission().unwrap().is_none());
}
#[test]
fn stale_tickets_and_double_retirement_cannot_renew_allowances() {
    let mut queue = Tickets::new(&[budget(2), budget(3)], 1).unwrap();
    let first = queue.next_admission().unwrap().unwrap(); queue.install(first).unwrap();
    assert!(queue.install(first).is_err());
    queue.retire(0).unwrap(); assert!(queue.retire(0).is_err());
    assert!(queue.install(first).is_err());
    assert!(queue.install(RefillAdmission { sequence: 1, request: 1 }).is_err());
    assert_eq!(queue.next, 1); assert_eq!(queue.completed, 1);
    let second = queue.next_admission().unwrap().unwrap(); queue.install(second).unwrap();
    assert!(queue.request(1).is_err()); assert!(queue.request(usize::MAX).is_err());
}
#[test]
fn reset_clears_both_loops_and_preserves_physical_capacity() {
    let mut row = Sequence { cache: KvCache::try_with_capacity(3).unwrap(), hidden: Some(vec![Bf16::from_f32(9.0); H]) };
    for slot in 0..KV_SLOT_COUNT {
        row.cache.append(slot, 0, &vec![7; K], &vec![11; K]).unwrap();
    }
    let retired = Int8Work::for_sequence(0, 1, V).unwrap();
    let mut account = Account { budget: budget(3), work: retired };
    reset_storage(&mut row, &mut account);
    assert!(row.hidden.is_none()); assert!(row.cache.all_slots_have_len(0));
    assert_eq!(row.cache.capacity_positions(), 3);
    assert_eq!(account.work, Int8Work::default());
    assert!(account.quote(0, 3, 1, 0).is_err());
    // A new ticket gets its own quota, not the previous ticket's remainder.
    account.budget = budget(1);
    let delta = account.quote(0, 3, 1, V).unwrap(); account.work = delta;
    assert!(account.quote(1, 3, 1, 0).is_err());
    assert_eq!(retired.checked_add(account.work).unwrap().forward_positions, 2);
}
#[test]
fn bounded_epoch_and_all_budget_axes_refuse_invalid_arithmetic() {
    for (count, slots) in [(0, 1), (1, 0), (2, 3), (65, 2)] {
        assert!(Tickets::new(&vec![budget(1); count], slots).is_err());
    }
    assert!(Tickets::new(&[Int8RunBudget::exact(Int8Work::default())], 1).is_err());
    let budgets = [budget(2), budget(4)]; let total = epoch_budget(&budgets).unwrap();
    assert_eq!(total.max_forward_positions, 6);
    assert_eq!(total.max_attention_pairs, budgets[0].max_attention_pairs + budgets[1].max_attention_pairs);
    for axis in 0..4 {
        let mut large = budget(1);
        match axis { 0 => large.max_forward_positions = u64::MAX, 1 => large.max_attention_pairs = u64::MAX,
            2 => large.max_projection_work.dot_products = u64::MAX,
            _ => large.max_projection_work.multiply_accumulates = u64::MAX }
        assert!(epoch_budget(&[large, budget(1)]).is_err());
    }
}
#[test]
fn retired_work_remains_in_the_epoch_ledger_after_slot_reuse() {
    let first = Int8Work::for_sequence(0, 2, V).unwrap();
    let next = Int8Work::for_sequence(0, 1, V).unwrap();
    let maximum = epoch_budget(&[budget(5), budget(3)]).unwrap();
    let mut ledger = ProjectionLedger::new(maximum.max_projection_work);
    // Check the exact cumulative arithmetic without a synthetic native receipt.
    let total = first.checked_add(next).unwrap();
    assert!(total.projections.fits(maximum.max_projection_work));
    assert!(total.attention_pairs < Int8Work::for_sequence(0, 3, 2 * V).unwrap().attention_pairs);
    assert!(ledger.preflight(total.projections).is_ok());
    assert_eq!(ledger.reserved(), ProjectionWork::default());
}
