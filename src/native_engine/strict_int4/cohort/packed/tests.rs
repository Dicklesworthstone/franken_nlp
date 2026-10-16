//! Real address/attention/work tests for token packs, not full-model parity.
use super::*;
use crate::native_engine::layer::NANBEIGE_Q_PROJECTION_SIZE as Q;

fn limits(rows: usize) -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
#[derive(Default)]
struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
fn sequences() -> Vec<Sequence> {
    (0..3).map(|_| Sequence { cache: KvCache::try_with_capacity(4).unwrap(), hidden: None }).collect()
}
fn bits(values: &[Bf16]) -> Vec<u16> { values.iter().map(|v| v.to_bits()).collect() }
fn pattern(length: usize, salt: usize) -> Vec<Bf16> {
    (0..length).map(|i| Bf16::from_f32(((i * 13 + salt * 7) % 31) as f32 / 32.0)).collect()
}

#[test]
fn total_tokens_not_document_count_controls_pack_admission() {
    let runs = [Int4TokenRun { sequence: 0, tokens: &[1, 2, 3] }, Int4TokenRun { sequence: 2, tokens: &[4] }];
    assert_eq!(check_runs(&runs, 3, limits(4)), Ok(4));
    assert_eq!(check_runs(&runs, 3, limits(3)), Err(StrictInt4Error::Memory));
    let mut short = limits(4); short.max_extra_scratch_bytes -= 1;
    assert_eq!(check_runs(&runs, 3, short), Err(StrictInt4Error::Memory));
    assert_eq!(check_runs(&[Int4TokenRun { sequence: 0, tokens: &[1; 64] }], 1, limits(64)), Ok(64));
}
#[test]
fn every_run_and_token_is_checked_including_the_last_one() {
    for runs in [vec![], vec![Int4TokenRun { sequence: 0, tokens: &[] }],
        vec![Int4TokenRun { sequence: 0, tokens: &[1] }, Int4TokenRun { sequence: 0, tokens: &[2] }],
        vec![Int4TokenRun { sequence: 1, tokens: &[1] }, Int4TokenRun { sequence: 0, tokens: &[2] }],
        vec![Int4TokenRun { sequence: 0, tokens: &[1] }, Int4TokenRun { sequence: 3, tokens: &[2] }],
        vec![Int4TokenRun { sequence: 0, tokens: &[1] }, Int4TokenRun { sequence: 2, tokens: &[V as u32] }]] {
        assert!(check_runs(&runs, 3, limits(64)).is_err());
    }
}
#[test]
fn per_document_packed_work_equals_sequential_work_without_cross_attention() {
    for width in [1, 3, 128, MAX_MODEL_K] {
        let mut aggregate = Int4Work::default();
        for (start, count) in [(0, 3), (2, 4), (7, 1)] {
            let quote = Int4Work::for_sequence(width, start, count, 0).unwrap();
            let mut serial = Int4Work::default();
            for position in start..start + count {
                serial = serial.checked_add(Int4Work::for_sequence(width, position, 1, 0).unwrap()).unwrap();
            }
            assert_eq!(quote, serial); aggregate = aggregate.checked_add(quote).unwrap();
            let account = Account { budget: Int4RunBudget::exact(quote), work: Int4Work::default() };
            assert_eq!(account.quote(width, start, start + count, count, 0).unwrap(), quote);
            assert!(account.quote(width, start, start + count + 1, count + 1, 0).is_err());
        }
        assert_eq!(aggregate.forward_positions, 8);
        assert_eq!(aggregate.attention_pairs, (6 + 18 + 8) * (KV_SLOT_COUNT * QUERY_HEAD_COUNT) as u64);
    }
}
#[test]
fn consecutive_positions_replay_exact_causal_attention_across_both_loop_slots() {
    for slot in [0, 21, 22, 43] {
        let mut actual = sequences(); let mut reference = sequences();
        let rope = RopeTablesF32::nanbeige(4).unwrap();
        let steps = [Int4CohortToken { sequence: 0, token: 1 }, Int4CohortToken { sequence: 0, token: 2 },
            Int4CohortToken { sequence: 2, token: 3 }, Int4CohortToken { sequence: 2, token: 4 }];
        let positions = [0, 1, 0, 1];
        let mut q = pattern(4 * Q, 1); let mut k = pattern(4 * K, 2); let v = pattern(4 * K, 3);
        let mut expected = Vec::new();
        for (index, step) in steps.iter().enumerate() {
            let mut qr = q[index * Q..(index + 1) * Q].to_vec();
            let mut kr = k[index * K..(index + 1) * K].to_vec();
            for head in qr.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(positions[index], head).unwrap(); }
            for head in kr.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(positions[index], head).unwrap(); }
            let cache = &mut reference[step.sequence].cache;
            cache.append(slot, positions[index], &bits(&kr), &bits(&v[index * K..(index + 1) * K])).unwrap();
            expected.extend(eager_gqa_attention_from_cache(&qr, cache, slot).unwrap());
        }
        let result = execution::attend_sequences(&mut q, &mut k, &v, &steps, &positions, slot,
            &rope, &mut actual, &mut Control::default()).unwrap();
        assert_eq!(bits(&result), bits(&expected));
        assert_eq!(actual[0].cache.len_for_slot(slot).unwrap(), 2);
        assert_eq!(actual[2].cache.len_for_slot(slot).unwrap(), 2);
        assert!(actual[1].cache.all_slots_have_len(0));
    }
}
#[test]
fn future_tokens_and_other_documents_cannot_change_earlier_attention() {
    let rope = RopeTablesF32::nanbeige(4).unwrap();
    let steps = [Int4CohortToken { sequence: 0, token: 1 }, Int4CohortToken { sequence: 0, token: 2 },
        Int4CohortToken { sequence: 2, token: 3 }];
    let mut baseline = None;
    for salt in [1, 9, 31] {
        let mut q = pattern(3 * Q, salt); let mut k = pattern(3 * K, salt); let mut v = pattern(3 * K, salt);
        q[..Q].fill(Bf16::from_f32(0.25)); k[..K].fill(Bf16::from_f32(0.5)); v[..K].fill(Bf16::from_f32(3.0));
        let output = execution::attend_sequences(&mut q, &mut k, &v, &steps, &[0, 1, 0], 22,
            &rope, &mut sequences(), &mut Control::default()).unwrap();
        let first = bits(&output[..Q]);
        if let Some(expected) = &baseline { assert_eq!(&first, expected); } else { baseline = Some(first); }
    }
}
#[test]
fn late_duplicate_gap_and_descending_address_refuse_before_cache_or_rotary_mutation() {
    let rope = RopeTablesF32::nanbeige(4).unwrap();
    for (slots, positions) in [([0, 0, 2], [0, 0, 0]), ([0, 0, 2], [0, 2, 0]),
        ([0, 0, 2], [0, 1, 4]), ([0, 2, 0], [0, 0, 1]), ([0, 0, 3], [0, 1, 0])] {
        let steps = slots.map(|sequence| Int4CohortToken { sequence, token: 1 });
        let mut q = pattern(3 * Q, 1); let mut k = pattern(3 * K, 2); let v = pattern(3 * K, 3);
        let before_q = bits(&q); let before_k = bits(&k);
        let mut caches = sequences(); let mut control = Control::default();
        assert!(execution::attend_sequences(&mut q, &mut k, &v, &steps, &positions, 0,
            &rope, &mut caches, &mut control).is_err());
        assert_eq!(control.polls, 0); assert_eq!(bits(&q), before_q); assert_eq!(bits(&k), before_k);
        assert!(caches.iter().all(|s| s.cache.all_slots_have_len(0)));
    }
}
#[test]
fn cancellation_between_tokens_uses_the_existing_whole_cohort_cleanup() {
    let steps = [Int4CohortToken { sequence: 0, token: 1 }, Int4CohortToken { sequence: 0, token: 2 }];
    let mut q = pattern(2 * Q, 1); let mut k = pattern(2 * K, 2); let v = pattern(2 * K, 3);
    let mut caches = sequences(); let mut control = Control { polls: 0, cancel: Some(3) };
    let result = execution::attend_sequences(&mut q, &mut k, &v, &steps, &[0, 1], 43,
        &RopeTablesF32::nanbeige(4).unwrap(), &mut caches, &mut control);
    assert!(matches!(result, Err(StrictInt4Error::Cancelled(DecodeCancellationKind::User))));
    assert_eq!(caches[0].cache.len_for_slot(43).unwrap(), 1);
    let mut state = RunState { active: true, poisoned: true };
    clear(&mut caches, &mut state);
    assert!(caches.iter().all(|s| s.cache.all_slots_have_len(0)));
    assert!(state.poisoned); assert!(!state.active);
}
