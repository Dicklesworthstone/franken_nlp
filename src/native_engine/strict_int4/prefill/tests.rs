use super::*;

#[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
    }
}
#[test]
fn workspace_admission_checks_all_row_limits_in_bytes() {
    for rows in [1, 3, 4, 5, MAX_INT4_BATCH_ROWS] {
        let bytes = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
        assert!(Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes }.validate().is_ok());
        assert!(Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes - 1 }.validate().is_err());
    }
    for rows in [0, 65, usize::MAX] { assert!(Int4PrefillLimits::required_extra_scratch_bytes(rows).is_err()); }
}
#[test]
fn every_group_partition_prices_morsels_as_one_sequence_with_only_one_head() {
    for group in [1, 3, 128, MAX_MODEL_K] { for start in [0, 7, 63] { for size in [1, 3, 4, 8, 64] {
        let tokens = vec![0_u32; 67]; let mut at = start; let mut sum = Int4Work::default();
        for chunk in tokens.chunks(size) {
            let work = Int4Work::for_sequence(group, at, chunk.len(), 0).unwrap();
            sum.forward_positions += work.forward_positions; sum.attention_pairs += work.attention_pairs;
            sum.projections = sum.projections.checked_add(work.projections).unwrap(); at += chunk.len();
        }
        let head = Int4Work::for_sequence(group, at, 0, V).unwrap();
        sum.projected_logits += head.projected_logits;
        sum.projections = sum.projections.checked_add(head.projections).unwrap();
        assert_eq!(sum, Int4Work::for_sequence(group, start, tokens.len(), V).unwrap());
    } } }
}
fn attention_case(start: usize, slot: usize) {
    let rows = 3; let capacity = start + rows;
    let rope = RopeTablesF32::nanbeige(capacity).unwrap();
    let mut grouped = KvCache::try_with_capacity(capacity).unwrap();
    let mut serial = KvCache::try_with_capacity(capacity).unwrap();
    for pos in 0..start {
        let key = vec![0; K]; let value = vec![Bf16::from_f32((pos + 1) as f32).to_bits(); K];
        grouped.append(slot, pos, &key, &value).unwrap(); serial.append(slot, pos, &key, &value).unwrap();
    }
    let mut query: Vec<_> = (0..rows * Q).map(|i| Bf16::from_f32(((i * 3) % 11) as f32 * 0.03125)).collect();
    let mut key: Vec<_> = (0..rows * K).map(|i| Bf16::from_f32(((i * 7) % 13) as f32 * 0.0625)).collect();
    let values: Vec<_> = [1.0, 3.0, 9.0].into_iter().flat_map(|x| vec![Bf16::from_f32(x); K]).collect();
    let original_query = query.clone(); let original_key = key.clone();
    let mut scratch = vec![f32::from_bits(0x7fc0_1234); capacity + 7];
    let pointer = scratch.as_ptr();
    attend_rows(&mut query, &mut key, &values, rows, start, slot, &rope, &mut grouped,
        &mut scratch, &mut Control::default()).unwrap();
    assert_eq!(scratch.as_ptr(), pointer);
    assert!(scratch[capacity..].iter().all(|value| value.to_bits() == 0x7fc0_1234));
    let output = query;
    let mut reference = Vec::new();
    for i in 0..rows {
        let mut q = original_query[i * Q..(i + 1) * Q].to_vec();
        let mut k = original_key[i * K..(i + 1) * K].to_vec();
        for h in q.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(start + i, h).unwrap(); }
        for h in k.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(start + i, h).unwrap(); }
        let kb: Vec<_> = k.iter().map(|v| v.to_bits()).collect();
        let vb: Vec<_> = values[i * K..(i + 1) * K].iter().map(|v| v.to_bits()).collect();
        serial.append(slot, start + i, &kb, &vb).unwrap();
        reference.extend(eager_gqa_attention_from_cache(&q, &serial, slot).unwrap());
    }
    assert_eq!(output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert_eq!(grouped.len_for_slot(slot).unwrap(), capacity);
    if start == 0 { assert!(output[..Q].iter().all(|v| v.to_f32() == 1.0)); }
}
#[test] fn prompt_attention_never_observes_future_rows() { attention_case(0, 0); }
#[test] fn continuation_and_second_pass_preserve_exact_rope_coordinates() { attention_case(2, 22); }
#[test] fn retained_prompt_scratch_crosses_attention_checkpoint_blocks() { attention_case(32, 43); }
#[test]
fn malformed_attention_geometry_is_refused_without_mutating_cache() {
    let mut cache = KvCache::try_with_capacity(2).unwrap(); let rope = RopeTablesF32::nanbeige(2).unwrap();
    for rows in [0, 1, 65, usize::MAX] {
        assert!(attend_rows(&mut [Bf16::from_bits(0)], &mut [], &[], rows, 0, 0, &rope, &mut cache, &mut [0.0; 2], &mut Control::default()).is_err());
        assert!(cache.all_slots_have_len(0));
    }
}
#[test]
fn cancellation_before_a_row_cannot_append_its_kv() {
    let mut cache = KvCache::try_with_capacity(2).unwrap(); let rope = RopeTablesF32::nanbeige(2).unwrap();
    let mut q = vec![Bf16::from_bits(0); Q]; let mut k = vec![Bf16::from_bits(0); K]; let v = k.clone();
    assert!(matches!(attend_rows(&mut q, &mut k, &v, 1, 0, 0, &rope, &mut cache, &mut [0.0; 2],
        &mut Control { polls: 0, cancel: Some(1) }), Err(StrictInt4Error::Cancelled(DecodeCancellationKind::User))));
    assert!(cache.all_slots_have_len(0));
}
#[test]
fn partial_causal_failure_leaves_only_a_private_prefix_for_session_cleanup() {
    let mut cache = KvCache::try_with_capacity(3).unwrap(); let rope = RopeTablesF32::nanbeige(3).unwrap();
    let mut q = vec![Bf16::from_bits(0); 3 * Q]; let mut k = vec![Bf16::from_bits(0); 3 * K]; let v = k.clone();
    assert!(attend_rows(&mut q, &mut k, &v, 3, 0, 0, &rope, &mut cache, &mut [0.0; 3],
        &mut Control { polls: 0, cancel: Some(3) }).is_err());
    assert_eq!(cache.len_for_slot(0).unwrap(), 1);
    let mut state = RunState { active: true, poisoned: true };
    end_session(&mut cache, &mut state);
    assert!(cache.all_slots_have_len(0)); assert!(state.poisoned); assert!(!state.active);
    assert!(state.open().is_err());
}
#[test]
fn cancellation_inside_attention_keeps_its_cause_and_requires_poisoned_cleanup() {
    struct DeadlineControl { polls: usize }
    impl DecodeStepControl for DeadlineControl {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1;
            // Two causal-driver checks, attention entry, then score/max/exp/
            // sum/probability checks. Stop at the first weighted-value block.
            (self.polls == 9).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let mut cache = KvCache::try_with_capacity(2).unwrap();
    let rope = RopeTablesF32::nanbeige(2).unwrap();
    let mut query = vec![Bf16::from_bits(0); 2 * Q];
    let mut key = vec![Bf16::from_bits(0); 2 * K];
    let value = vec![Bf16::from_f32(3.0); 2 * K];
    let error = attend_rows(&mut query, &mut key, &value, 2, 0, 22, &rope,
        &mut cache, &mut [0.0; 2], &mut DeadlineControl { polls: 0 }).unwrap_err();
    assert_eq!(error, StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline));
    assert_eq!(cache.len_for_slot(22).unwrap(), 1);
    let mut state = RunState { active: true, poisoned: true };
    end_session(&mut cache, &mut state);
    assert!(cache.all_slots_have_len(0));
    assert!(state.poisoned);
    assert!(state.open().is_err());
}
#[test]
fn insufficient_prompt_scratch_refuses_before_any_row_or_cache_mutation() {
    let mut cache = KvCache::try_with_capacity(2).unwrap();
    let rope = RopeTablesF32::nanbeige(2).unwrap();
    let mut query = vec![Bf16::from_f32(0.5); 2 * Q];
    let mut key = vec![Bf16::from_f32(0.25); 2 * K]; let value = key.clone();
    let before_query = query.clone(); let before_key = key.clone();
    let mut scratch = [123.0]; let mut control = Control::default();
    assert_eq!(attend_rows(&mut query, &mut key, &value, 2, 0, 0, &rope,
        &mut cache, &mut scratch, &mut control), Err(StrictInt4Error::Attention));
    assert_eq!(query, before_query); assert_eq!(key, before_key);
    assert_eq!(scratch, [123.0]); assert_eq!(control.polls, 0);
    assert!(cache.all_slots_have_len(0));
}

#[test]
fn prompt_norm_rows_reuse_storage_and_preserve_encoded_activations() {
    let scale: Vec<_> = (0..H).map(|i| Bf16::from_f32(0.5 + (i % 7) as f32 * 0.125)).collect();
    let mut hidden: Vec<Vec<_>> = (0..3).map(|row| (0..H).map(|i|
        Bf16::from_f32(((i * 11 + row * 17) % 63) as f32 * 0.0625 - 1.5)).collect()).collect();
    let mut activations: Vec<_> = (0..3).map(|_| ActivationBuffer::try_new(I).unwrap()).collect();
    let mut workspace = DecoderWorkspace::try_new(1).unwrap();
    let pointer = workspace.normalized.as_ptr(); let mut control = Control::default();
    for round in 0..2 {
        workspace.normalized.fill(Bf16::from_bits(0x7fc1));
        encode_norms(&hidden, &scale, &mut activations, &mut workspace.normalized, &mut control).unwrap();
        for (row, actual) in hidden.iter().zip(&activations) {
            let mut expected = ActivationBuffer::try_new(I).unwrap();
            expected.encode_bf16(&norm(row, &scale).unwrap()).unwrap();
            assert_eq!(actual.values().unwrap(), expected.values().unwrap());
            assert_eq!(actual.scale().unwrap().to_bits(), expected.scale().unwrap().to_bits());
        }
        assert_eq!(workspace.normalized.as_ptr(), pointer);
        assert_eq!(control.polls, (round + 1) * hidden.len());
        hidden.reverse();
    }
    let mut stopped: Vec<_> = (0..3).map(|_| ActivationBuffer::try_new(I).unwrap()).collect();
    let mut control = Control { polls: 0, cancel: Some(2) };
    assert_eq!(encode_norms(&hidden, &scale, &mut stopped, &mut workspace.normalized, &mut control),
        Err(StrictInt4Error::Cancelled(DecodeCancellationKind::User)));
    assert!(stopped[0].values().is_ok()); assert!(stopped[1].values().is_err());
    assert!(stopped[2].values().is_err()); assert_eq!(control.polls, 2);
}

#[test]
fn both_prompt_loop_norms_keep_each_hidden_row_in_its_own_allocation() {
    let scale = vec![Bf16::from_f32(1.125); H];
    let mut hidden: Vec<Vec<_>> = (0..3).map(|row| (0..H).map(|i|
        Bf16::from_f32(((i + row * 7) % 31) as f32 * 0.125 - 1.0)).collect()).collect();
    let pointers: Vec<_> = hidden.iter().map(|row| row.as_ptr()).collect();
    let mut expected = hidden.clone(); let rope = RopeTablesF32::nanbeige(1).unwrap();
    let mut cache = KvCache::try_with_capacity(1).unwrap();
    let mut activations: Vec<_> = (0..3).map(|_| ActivationBuffer::try_new(I).unwrap()).collect();
    let mut workspace = DecoderWorkspace::try_new(1).unwrap();
    let mut ledger = ProjectionLedger::new(ProjectionWork::default());
    let mut control = Control::default(); let mut projection = None; let mut unpack = [0; 32];
    let mut executor = PrefillExecutor { final_norm: &scale, rope: &rope, cache: &mut cache,
        activations: &mut activations, workspace: &mut workspace, ledger: &mut ledger,
        control: &mut control, projection: &mut projection, unpack: &mut unpack,
        kernel: PackedInt4Kernel::scalar(), start: 0, rows: 3,
        completed: PHYSICAL_LAYER_COUNT, norms: 0 };
    for pass in 1..=2 {
        for row in &mut expected { *row = norm(row, &scale).unwrap(); }
        executor.completed = pass * PHYSICAL_LAYER_COUNT;
        executor.workspace.normalized.fill(Bf16::from_bits(0x7fc1));
        executor.final_norm_group(&mut hidden).unwrap();
        assert_eq!(hidden, expected);
        assert_eq!(hidden.iter().map(|row| row.as_ptr()).collect::<Vec<_>>(), pointers);
    }
    assert_eq!(executor.final_norm_group(&mut hidden), Err(StrictInt4Error::Boundary));
    assert_eq!(executor.control.polls, 6);
}

#[test]
fn invalid_prompt_tokens_are_checked_as_one_complete_input() {
    assert!(check_tokens(&[]).is_err()); assert!(check_tokens(&[0, V as u32]).is_err());
    assert!(check_tokens(&[0, V as u32 - 1]).is_ok());
}
