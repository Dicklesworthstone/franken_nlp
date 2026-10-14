//! Real work/admission and causal-attention seams; no fabricated full model.
use super::*;
use crate::native_engine::layer::NANBEIGE_Q_PROJECTION_SIZE as Q;
#[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
#[test]
fn ragged_memory_sums_kv_but_shares_rope_and_unpacking() {
    for width in [1, 3, 128, MAX_MODEL_K] {
        for contexts in [vec![1], vec![1, 7, 3], vec![1; MAX_INT4_SEQUENCES]] {
            let actual = Int4MemoryRequirement::for_cohort(&contexts, width).unwrap();
            let base = Int4MemoryRequirement::for_context(*contexts.iter().max().unwrap(), width).unwrap();
            assert_eq!(actual.kv_bytes, contexts.iter().sum::<usize>() as u64 * KV_BYTES_PER_TOKEN as u64);
            assert_eq!(actual.rope_bytes, base.rope_bytes);
            let rows = contexts.len() as u64;
            assert_eq!(actual.scratch_payload_bound, base.scratch_payload_bound
                + prefill::Int4PrefillLimits::required_extra_scratch_bytes(contexts.len()).unwrap()
                + rows * (V * std::mem::size_of::<f32>() + 8192) as u64);
            let exact = Int4MemoryBudget { max_kv_bytes: actual.kv_bytes, max_rope_bytes: actual.rope_bytes,
                max_scratch_payload_bytes: actual.scratch_payload_bound };
            assert!(actual.check(exact).is_ok());
            for axis in 0..3 {
                let mut short = exact;
                match axis { 0 => short.max_kv_bytes -= 1, 1 => short.max_rope_bytes -= 1, _ => short.max_scratch_payload_bytes -= 1 }
                assert!(actual.check(short).is_err());
            }
        }
    }
}
#[test]
fn invalid_membership_context_and_partition_refuse_before_allocation() {
    for contexts in [vec![], vec![1; MAX_INT4_SEQUENCES + 1], vec![1, 0],
        vec![1, DEFAULT_ADMITTED_CONTEXT_CAP + 1], vec![usize::MAX]] {
        assert!(Int4MemoryRequirement::for_cohort(&contexts, 128).is_err());
    }
    for width in [0, MAX_MODEL_K + 1, usize::MAX] { assert!(Int4MemoryRequirement::for_cohort(&[1], width).is_err()); }
    for indices in [vec![], vec![0, 0], vec![2, 1], vec![3], vec![usize::MAX]] {
        assert!(check_indices(indices.into_iter(), 3).is_err());
    }
    assert!(check_indices([0, 2].into_iter(), 3).is_ok());
}
#[test]
fn every_work_axis_checks_overflow_and_independent_attention_triangles() {
    let one = Int4Work { forward_positions: 1, projected_logits: 1, attention_pairs: 1,
        projections: ProjectionWork { dot_products: 1, multiply_accumulates: 1 } };
    for axis in 0..5 {
        let mut huge = Int4Work::default();
        match axis { 0 => huge.forward_positions = u64::MAX, 1 => huge.projected_logits = u64::MAX,
            2 => huge.attention_pairs = u64::MAX, 3 => huge.projections.dot_products = u64::MAX,
            _ => huge.projections.multiply_accumulates = u64::MAX }
        assert!(huge.checked_add(one).is_err());
    }
    for width in [1, 3, 128, MAX_MODEL_K] {
        let a = Int4Work::for_sequence(width, 0, 3, V).unwrap();
        let b = Int4Work::for_sequence(width, 0, 7, 2 * V).unwrap();
        let combined = a.checked_add(b).unwrap();
        let concatenated = Int4Work::for_sequence(width, 0, 10, 3 * V).unwrap();
        assert_eq!(combined.projections, concatenated.projections);
        assert!(combined.attention_pairs < concatenated.attention_pairs);
    }
}
#[test]
fn per_sequence_accounts_cannot_borrow_neighbors_unused_ceiling() {
    for width in [1, 3, 128, MAX_MODEL_K] {
        let full = Int4Work::for_sequence(width, 0, 3, 17).unwrap();
        let exact = Int4RunBudget::exact(full);
        let mut account = Account { budget: exact, work: Int4Work::default() };
        for start in 0..3 {
            let delta = account.quote(width, start, 3, 1, 0).unwrap();
            account.work = account.work.checked_add(delta).unwrap();
        }
        account.work = account.work.checked_add(account.quote(width, 3, 3, 0, 17).unwrap()).unwrap();
        assert_eq!(account.work, full);
        assert!(account.quote(width, 3, 4, 1, 0).is_err());
        assert!(account.quote(width, 3, 3, 0, 1).is_err());
        for axis in 0..4 {
            let mut budget = exact;
            match axis { 0 => budget.max_forward_positions -= 1, 1 => budget.max_attention_pairs -= 1,
                2 => budget.max_projection_work.dot_products -= 1, _ => budget.max_projection_work.multiply_accumulates -= 1 }
            let small = Account { budget, work: Int4Work::default() };
            let neighbor = Account { budget: exact, work: Int4Work::default() };
            assert!(neighbor.quote(width, 0, 3, 3, 17).is_ok());
            assert!(small.quote(width, 0, 3, 3, 17).is_err());
            assert_eq!(small.work, Int4Work::default()); assert_eq!(neighbor.work, Int4Work::default());
        }
    }
}
fn fixture(prefixes: &[usize], slot: usize) -> Vec<Sequence> {
    prefixes.iter().enumerate().map(|(index, &prefix)| {
        let mut cache = KvCache::try_with_capacity(prefix + 1).unwrap();
        for position in 0..prefix {
            let key = vec![Bf16::from_f32((position + 1) as f32 / 16.0).to_bits(); K];
            let value = vec![Bf16::from_f32((index * 8 + position + 1) as f32).to_bits(); K];
            cache.append(slot, position, &key, &value).unwrap();
        }
        Sequence { cache, hidden: None }
    }).collect()
}
fn bits(values: &[Bf16]) -> Vec<u16> { values.iter().map(|v| v.to_bits()).collect() }
#[test]
fn ragged_attention_matches_independent_replay_and_leaves_inactive_documents_untouched() {
    for slot in [0, 22, 43] {
        let mut sequences = fixture(&[0, 1, 3], slot);
        let mut reference = fixture(&[0, 1, 3], slot);
        let rope = RopeTablesF32::nanbeige(4).unwrap();
        let steps = [Int4CohortToken { sequence: 0, token: 1 }, Int4CohortToken { sequence: 2, token: 2 }];
        let positions = [0, 3];
        let mut query: Vec<_> = (0..2 * Q).map(|i| Bf16::from_f32((i % 17) as f32 / 32.0)).collect();
        let mut key: Vec<_> = (0..2 * K).map(|i| Bf16::from_f32((i % 13) as f32 / 16.0)).collect();
        let value: Vec<_> = (0..2 * K).map(|i| Bf16::from_f32((i / K * 32 + 5) as f32)).collect();
        let mut expected = Vec::new();
        for (lane, step) in steps.iter().enumerate() {
            let mut q = query[lane * Q..(lane + 1) * Q].to_vec();
            let mut k = key[lane * K..(lane + 1) * K].to_vec();
            for head in q.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(positions[lane], head).unwrap(); }
            for head in k.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(positions[lane], head).unwrap(); }
            reference[step.sequence].cache.append(slot, positions[lane], &bits(&k), &bits(&value[lane * K..(lane + 1) * K])).unwrap();
            expected.extend(eager_gqa_attention_from_cache(&q, &reference[step.sequence].cache, slot).unwrap());
        }
        let actual = execution::attend_sequences(&mut query, &mut key, &value, &steps, &positions, slot,
            &rope, &mut sequences, &mut Control::default()).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        assert!(actual[..Q].iter().all(|v| v.to_f32() == 5.0));
        for (index, length) in [1, 1, 4].into_iter().enumerate() {
            assert_eq!(sequences[index].cache.len_for_slot(slot).unwrap(), length);
            for other in 0..KV_SLOT_COUNT { if other != slot { assert_eq!(sequences[index].cache.len_for_slot(other).unwrap(), 0); } }
        }
    }
}
#[test]
fn late_address_or_geometry_refusal_does_not_mutate_earlier_caches_or_poll() {
    let steps = [Int4CohortToken { sequence: 0, token: 1 }, Int4CohortToken { sequence: 1, token: 2 }];
    let rope = RopeTablesF32::nanbeige(2).unwrap();
    for bad_geometry in [false, true] {
        let mut sequences = fixture(&[0, 1], 22); let mut control = Control::default();
        let mut q = vec![Bf16::from_bits(0); if bad_geometry { Q } else { 2 * Q }];
        let mut k = vec![Bf16::from_bits(0); 2 * K]; let v = k.clone();
        let positions = if bad_geometry { [0, 1] } else { [0, 0] };
        assert!(execution::attend_sequences(&mut q, &mut k, &v, &steps, &positions, 22, &rope,
            &mut sequences, &mut control).is_err());
        assert_eq!(control.polls, 0);
        assert_eq!(sequences[0].cache.len_for_slot(22).unwrap(), 0);
        assert_eq!(sequences[1].cache.len_for_slot(22).unwrap(), 1);
    }
}
#[test]
fn partial_epoch_cancellation_clears_every_row_and_keeps_the_cohort_poisoned() {
    let steps = [Int4CohortToken { sequence: 0, token: 1 }, Int4CohortToken { sequence: 1, token: 2 }];
    let rope = RopeTablesF32::nanbeige(1).unwrap();
    let mut sequences = fixture(&[0, 0], 0);
    let mut q = vec![Bf16::from_bits(0); 2 * Q]; let mut k = vec![Bf16::from_bits(0); 2 * K]; let v = k.clone();
    assert!(matches!(execution::attend_sequences(&mut q, &mut k, &v, &steps, &[0, 0], 0, &rope,
        &mut sequences, &mut Control { polls: 0, cancel: Some(3) }),
        Err(StrictInt4Error::Cancelled(DecodeCancellationKind::User))));
    assert_eq!(sequences[0].cache.len_for_slot(0).unwrap(), 1);
    assert_eq!(sequences[1].cache.len_for_slot(0).unwrap(), 0);
    sequences[0].hidden = Some(vec![Bf16::from_bits(0); H]);
    let mut state = RunState { active: true, poisoned: true };
    clear(&mut sequences, &mut state);
    assert!(sequences.iter().all(|row| row.hidden.is_none() && row.cache.all_slots_have_len(0)));
    assert!(!state.active); assert!(state.poisoned); assert!(state.open().is_err());
}
