//! Model-free arithmetic, independent-KV and lifecycle regression definitions.
//! These fixtures do not certify full-model grouped/scalar parity.
use super::*;
#[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
#[test]
fn memory_prices_ragged_kv_shared_rope_and_all_head_outputs() {
    let contexts = [2, 7, 3];
    let grouped = Int8MemoryRequirement::for_cohort(&contexts).unwrap();
    assert_eq!(grouped.kv_bytes, 12 * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(grouped.rope_bytes, Int8MemoryRequirement::for_context(7).unwrap().rope_bytes);
    assert!(grouped.scratch_payload_bound >= (contexts.len() * V * 4) as u64);
    let budget = Int8MemoryBudget { max_kv_bytes: grouped.kv_bytes, max_rope_bytes: grouped.rope_bytes,
        max_scratch_payload_bytes: grouped.scratch_payload_bound };
    grouped.check(budget).unwrap();
    for axis in 0..3 { let mut short = budget;
        match axis { 0 => short.max_kv_bytes -= 1, 1 => short.max_rope_bytes -= 1, _ => short.max_scratch_payload_bytes -= 1 }
        assert!(grouped.check(short).is_err());
    }
    for bad in [vec![], vec![0], vec![1; MAX_BATCH_ROWS + 1], vec![DEFAULT_ADMITTED_CONTEXT_CAP + 1]] {
        assert!(Int8MemoryRequirement::for_cohort(&bad).is_err());
    }
}
#[test]
fn active_rows_are_unique_bounded_and_not_implicitly_reordered() {
    check_indices([0, 2, 4].into_iter(), 5).unwrap();
    for bad in [vec![], vec![0, 0], vec![2, 1], vec![5], vec![usize::MAX]] {
        assert!(check_indices(bad.into_iter(), 5).is_err());
    }
}
#[test]
fn one_sequence_cannot_spend_another_sequences_unused_work() {
    let work = Int8Work::for_sequence(0, 2, V).unwrap();
    let mut account = Account { budget: Int8RunBudget::exact(work), work: Int8Work::default() };
    for position in 0..2 {
        let delta = account.quote(position, 8, 1, 0).unwrap(); account.work = account.work.checked_add(delta).unwrap();
    }
    assert!(account.quote(2, 8, 1, 0).is_err());
    account.work = account.work.checked_add(account.quote(2, 8, 0, V).unwrap()).unwrap();
    assert_eq!(account.work, work); assert!(account.quote(2, 8, 0, 1).is_err());
    assert!(account.quote(usize::MAX, usize::MAX, 1, 0).is_err());
}
#[test]
fn ragged_work_sums_independent_triangles_not_a_concatenated_sequence() {
    let a = Int8Work::for_sequence(0, 2, V).unwrap();
    let b = Int8Work::for_sequence(0, 5, V).unwrap();
    let total = a.checked_add(b).unwrap();
    assert_eq!(total.forward_positions, 7);
    assert_eq!(total.attention_pairs, (3 + 15) * (KV_SLOT_COUNT * QUERY_HEAD_COUNT) as u64);
    assert!(total.attention_pairs < Int8Work::for_sequence(0, 7, 2 * V).unwrap().attention_pairs);
}
fn sequences(starts: &[usize], slot: usize) -> Vec<Sequence> {
    starts.iter().enumerate().map(|(row, &start)| {
        let mut cache = KvCache::try_with_capacity(start + 1).unwrap();
        for position in 0..start {
            cache.append(slot, position, &vec![0; K],
                &vec![Bf16::from_f32((10 * row + position + 1) as f32).to_bits(); K]).unwrap();
        }
        Sequence { cache, hidden: None }
    }).collect()
}
fn ragged_case(slot: usize) {
    let starts = [0, 3, 1];
    let mut grouped = sequences(&starts, slot); let mut reference = sequences(&starts, slot);
    let rope = RopeTablesF32::nanbeige(4).unwrap();
    let steps: Vec<_> = (0..3).map(|sequence| CohortToken { sequence, token: 1 }).collect();
    let mut q: Vec<_> = (0..3 * Q).map(|i| Bf16::from_f32((i % 17) as f32 / 32.0)).collect();
    let mut k: Vec<_> = (0..3 * K).map(|i| Bf16::from_f32((i % 11) as f32 / 16.0)).collect();
    let v: Vec<_> = [2.0, 50.0, -4.0].into_iter().flat_map(|x| vec![Bf16::from_f32(x); K]).collect();
    let original_q = q.clone(); let original_k = k.clone();
    let out = execution::attend_sequences(&mut q, &mut k, &v, &steps, &starts,
        slot, &rope, &mut grouped, &mut Control::default()).unwrap();
    for row in 0..3 {
        let mut query = original_q[row * Q..(row + 1) * Q].to_vec();
        let mut key = original_k[row * K..(row + 1) * K].to_vec();
        for head in query.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(starts[row], head).unwrap(); }
        for head in key.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(starts[row], head).unwrap(); }
        reference[row].cache.append(slot, starts[row], &key.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            &v[row * K..(row + 1) * K].iter().map(|x| x.to_bits()).collect::<Vec<_>>()).unwrap();
        let expected = eager_gqa_attention_from_cache(&query, &reference[row].cache, slot).unwrap();
        assert_eq!(out[row * Q..(row + 1) * Q].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert_eq!(grouped[row].cache.len_for_slot(slot).unwrap(), starts[row] + 1);
    }
    assert!(out[..Q].iter().all(|x| x.to_f32() == 2.0));
}
#[test] fn independent_documents_never_attend_to_other_rows() { ragged_case(0); }
#[test] fn second_loop_uses_each_documents_own_position() { ragged_case(22); }
#[test]
fn invalid_later_row_is_refused_before_the_first_kv_write() {
    let mut rows = sequences(&[0, 0], 0); let rope = RopeTablesF32::nanbeige(2).unwrap();
    let steps = [CohortToken { sequence: 0, token: 1 }, CohortToken { sequence: 1, token: 1 }];
    assert!(execution::attend_sequences(&mut vec![Bf16::from_bits(0); 2 * Q],
        &mut vec![Bf16::from_bits(0); 2 * K], &vec![Bf16::from_bits(0); 2 * K],
        &steps, &[0, 1], 0, &rope, &mut rows, &mut Control::default()).is_err());
    assert!(rows.iter().all(|row| row.cache.all_slots_have_len(0)));
}
#[test]
fn cancellation_keeps_cause_and_cleanup_clears_every_row() {
    let mut rows = sequences(&[0, 0], 0); let rope = RopeTablesF32::nanbeige(1).unwrap();
    let steps = [CohortToken { sequence: 0, token: 1 }, CohortToken { sequence: 1, token: 1 }];
    let result = execution::attend_sequences(&mut vec![Bf16::from_bits(0); 2 * Q],
        &mut vec![Bf16::from_bits(0); 2 * K], &vec![Bf16::from_bits(0); 2 * K],
        &steps, &[0, 0], 0, &rope, &mut rows, &mut Control { polls: 0, cancel: Some(3) });
    assert!(matches!(result, Err(StrictInt8Error::Cancelled(DecodeCancellationKind::User))));
    assert_eq!(rows[0].cache.len_for_slot(0).unwrap(), 1);
    assert_eq!(rows[1].cache.len_for_slot(0).unwrap(), 0);
    let mut state = RunState { active: true, poisoned: true };
    clear(&mut rows, &mut state);
    assert!(rows.iter().all(|row| row.cache.all_slots_have_len(0) && row.hidden.is_none()));
    assert!(!state.active); assert!(state.open().is_err());
}
