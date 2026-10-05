//! Model-free packed attention and admission regressions; not full-model parity.
use super::*;
#[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1;
        (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
fn limits(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
#[test]
fn total_tokens_not_document_count_bound_the_pack() {
    let runs = [CohortTokenRun { sequence: 0, tokens: &[1, 2, 3] },
        CohortTokenRun { sequence: 2, tokens: &[4, 5] }];
    assert_eq!(check_runs(&runs, 3, limits(5)).unwrap(), 5);
    assert!(check_runs(&runs, 3, limits(4)).is_err());
    let mut short = limits(5); short.max_extra_scratch_bytes -= 1;
    assert!(check_runs(&runs, 3, short).is_err());
    assert!(check_runs(&[], 3, limits(5)).is_err());
}
#[test]
fn malformed_runs_cannot_hide_in_a_valid_later_morsel() {
    for bad in [vec![CohortTokenRun { sequence: 0, tokens: &[] }],
        vec![CohortTokenRun { sequence: 0, tokens: &[1, V as u32] }],
        vec![CohortTokenRun { sequence: 3, tokens: &[1] }],
        vec![CohortTokenRun { sequence: 0, tokens: &[1] }, CohortTokenRun { sequence: 0, tokens: &[2] }],
        vec![CohortTokenRun { sequence: 2, tokens: &[1] }, CohortTokenRun { sequence: 1, tokens: &[2] }]] {
        assert!(check_runs(&bad, 3, limits(8)).is_err());
    }
}
#[test]
fn packed_quotes_equal_serial_quotes_without_cross_document_attention() {
    let starts = [0, 3, 9]; let counts = [4, 1, 2];
    for (&start, &count) in starts.iter().zip(&counts) {
        let budget = Int8RunBudget::exact(Int8Work::for_sequence(0, start + count, V).unwrap());
        let account = Account { budget, work: Int8Work::for_sequence(0, start, 0).unwrap() };
        let packed = account.quote(start, start + count, count, V).unwrap();
        let mut serial = Int8Work::for_sequence(start + count, 0, V).unwrap();
        for position in start..start + count { serial = serial.checked_add(Int8Work::for_sequence(position, 1, 0).unwrap()).unwrap(); }
        assert_eq!(packed, serial);
        assert!(account.quote(start, start + count, count + 1, 0).is_err());
    }
}
fn caches(starts: &[usize], counts: &[usize], slot: usize) -> Vec<Sequence> {
    starts.iter().zip(counts).enumerate().map(|(sequence, (&start, &count))| {
        let mut cache = KvCache::try_with_capacity(start + count).unwrap();
        for position in 0..start {
            let value = Bf16::from_f32((sequence * 10 + position + 1) as f32).to_bits();
            cache.append(slot, position, &vec![0; K], &vec![value; K]).unwrap();
        }
        Sequence { cache, hidden: None }
    }).collect()
}
fn causal_case(slot: usize, starts: [usize; 2]) {
    let counts = [3, 2];
    let mut packed = caches(&starts, &counts, slot); let mut serial = caches(&starts, &counts, slot);
    let rope = RopeTablesF32::nanbeige(8).unwrap();
    let steps: Vec<_> = [0, 0, 0, 1, 1].into_iter().map(|sequence| CohortToken { sequence, token: 1 }).collect();
    let positions = [starts[0], starts[0] + 1, starts[0] + 2, starts[1], starts[1] + 1];
    let mut q: Vec<_> = (0..5 * Q).map(|i| Bf16::from_f32((i % 17) as f32 / 32.0)).collect();
    let mut k: Vec<_> = (0..5 * K).map(|i| Bf16::from_f32((i % 11) as f32 / 16.0)).collect();
    let v: Vec<_> = [1.0, 3.0, 9.0, -20.0, -50.0].into_iter()
        .flat_map(|value| vec![Bf16::from_f32(value); K]).collect();
    let original_q = q.clone(); let original_k = k.clone();
    let actual = execution::attend_sequences(&mut q, &mut k, &v, &steps, &positions,
        slot, &rope, &mut packed, &mut Control::default()).unwrap();
    for (row, (step, &position)) in steps.iter().zip(&positions).enumerate() {
        let mut q = original_q[row * Q..(row + 1) * Q].to_vec();
        let mut k = original_k[row * K..(row + 1) * K].to_vec();
        for head in q.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).unwrap(); }
        for head in k.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).unwrap(); }
        let kb: Vec<_> = k.iter().map(|v| v.to_bits()).collect();
        let vb: Vec<_> = v[row * K..(row + 1) * K].iter().map(|v| v.to_bits()).collect();
        serial[step.sequence].cache.append(slot, position, &kb, &vb).unwrap();
        let expected = eager_gqa_attention_from_cache(&q, &serial[step.sequence].cache, slot).unwrap();
        assert_eq!(actual[row * Q..(row + 1) * Q].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    if starts == [0, 0] {
        assert!(actual[..Q].iter().all(|v| v.to_f32() == 1.0));
        assert!(actual[3 * Q..4 * Q].iter().all(|v| v.to_f32() == -20.0));
    }
    for (sequence, count) in packed.iter().zip(starts.iter().zip(counts).map(|(&s, n)| s + n)) {
        assert_eq!(sequence.cache.len_for_slot(slot).unwrap(), count);
    }
}
#[test] fn multiple_prompt_rows_never_see_future_or_foreign_tokens() { causal_case(0, [0, 0]); }
#[test] fn ragged_existing_prefixes_and_second_loop_keep_independent_coordinates() { causal_case(22, [2, 1]); }
#[test]
fn late_bad_address_refuses_before_any_token_is_appended() {
    let rope = RopeTablesF32::nanbeige(4).unwrap();
    let steps = [CohortToken { sequence: 0, token: 1 }, CohortToken { sequence: 0, token: 2 },
        CohortToken { sequence: 1, token: 3 }];
    for positions in [[0, 0, 0], [0, 2, 0], [0, 1, 1]] {
        let mut sequences = caches(&[0, 0], &[2, 1], 0);
        assert!(execution::attend_sequences(&mut vec![Bf16::from_bits(0); 3 * Q],
            &mut vec![Bf16::from_bits(0); 3 * K], &vec![Bf16::from_bits(0); 3 * K],
            &steps, &positions, 0, &rope, &mut sequences, &mut Control::default()).is_err());
        assert!(sequences.iter().all(|row| row.cache.all_slots_have_len(0)));
    }
}
#[test]
fn cancellation_between_same_document_tokens_preserves_partial_boundary() {
    let rope = RopeTablesF32::nanbeige(2).unwrap(); let mut sequences = caches(&[0], &[2], 0);
    let steps = [CohortToken { sequence: 0, token: 1 }, CohortToken { sequence: 0, token: 2 }];
    let error = execution::attend_sequences(&mut vec![Bf16::from_bits(0); 2 * Q],
        &mut vec![Bf16::from_bits(0); 2 * K], &vec![Bf16::from_bits(0); 2 * K],
        &steps, &[0, 1], 0, &rope, &mut sequences, &mut Control { polls: 0, cancel: Some(3) }).unwrap_err();
    assert!(matches!(error, StrictInt8Error::Cancelled(DecodeCancellationKind::User)));
    assert_eq!(sequences[0].cache.len_for_slot(0).unwrap(), 1);
    let mut state = RunState { active: true, poisoned: true };
    clear(&mut sequences, &mut state);
    assert!(sequences[0].cache.all_slots_have_len(0)); assert!(state.open().is_err());
}
