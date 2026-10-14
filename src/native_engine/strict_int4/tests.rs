use super::*;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn work_counts_every_group_and_both_passes_without_padded_tail_macs() {
    for width in [1, 3, 32, 128, 1000, MAX_MODEL_K] {
        for (start, count, logits) in [(0, 1, V), (3, 7, 13), (10, 0, 7), (0, 0, 0)] {
            let work = Int4Work::for_sequence(width, start, count, logits).unwrap();
            let mut dots = 0_u64; let mut macs = 0_u64; let mut pairs = 0_u64;
            for position in start..start + count {
                for _ in 0..KV_SLOT_COUNT {
                    for (n, k) in PROJECTION_SHAPES {
                        dots += (n * k.div_ceil(width)) as u64;
                        macs += (n * k) as u64;
                    }
                    pairs += ((position + 1) * QUERY_HEAD_COUNT) as u64;
                }
            }
            dots += (logits * H.div_ceil(width)) as u64; macs += (logits * H) as u64;
            assert_eq!(work.projections, ProjectionWork { dot_products: dots, multiply_accumulates: macs });
            assert_eq!(work.attention_pairs, pairs);
            assert_eq!(work.forward_positions, count as u64); assert_eq!(work.projected_logits, logits as u64);
        }
    }
}

#[test]
fn prefix_and_continuation_work_compose_without_resetting_position() {
    for start in 0..8 {
        for a in 0..8 {
            for b in 0..8 {
                let first = Int4Work::for_sequence(127, start, a, 3).unwrap();
                let second = Int4Work::for_sequence(127, start + a, b, V).unwrap();
                let together = Int4Work::for_sequence(127, start, a + b, V + 3).unwrap();
                assert_eq!(first.projections.checked_add(second.projections).unwrap(), together.projections);
                assert_eq!(first.attention_pairs + second.attention_pairs, together.attention_pairs);
            }
        }
    }
}

#[test]
fn invalid_partition_and_overflow_refuse_without_wrapping() {
    for width in [0, MAX_MODEL_K + 1, usize::MAX] {
        assert!(Int4Work::for_sequence(width, 0, 1, V).is_err());
    }
    assert!(Int4Work::for_sequence(128, usize::MAX, usize::MAX, V).is_err());
    assert!(Int4Work::for_sequence(1, 0, usize::MAX, usize::MAX).is_err());
    // With no positions, a large start alone must not overflow a triangle.
    assert_eq!(Int4Work::for_sequence(128, usize::MAX, 0, 0).unwrap(), Int4Work::default());
}

#[test]
fn engine_memory_includes_unpacking_and_every_budget_axis_is_checked() {
    let need = Int4MemoryRequirement::for_context(3, 127).unwrap();
    let int8 = super::super::strict_int8::Int8MemoryRequirement::for_context(3).unwrap();
    assert_eq!(need.kv_bytes, int8.kv_bytes); assert_eq!(need.rope_bytes, int8.rope_bytes);
    assert_eq!(need.scratch_payload_bound, int8.scratch_payload_bound + 127);
    let budget = Int4MemoryBudget { max_kv_bytes: need.kv_bytes, max_rope_bytes: need.rope_bytes,
        max_scratch_payload_bytes: need.scratch_payload_bound };
    assert!(need.check(budget).is_ok());
    for axis in 0..3 {
        let mut bad = budget;
        match axis { 0 => bad.max_kv_bytes -= 1, 1 => bad.max_rope_bytes -= 1, _ => bad.max_scratch_payload_bytes -= 1 }
        assert_eq!(need.check(bad), Err(StrictInt4Error::Memory));
    }
    for context in [0, DEFAULT_ADMITTED_CONTEXT_CAP + 1, usize::MAX] {
        assert!(Int4MemoryRequirement::for_context(context, 128).is_err());
    }
}

#[test]
fn cleanup_clears_all_slots_but_never_unpoisons_native_failure() {
    let mut cache = KvCache::try_with_capacity(2).unwrap();
    let mut state = RunState::default();
    state.open().unwrap(); assert!(state.open().is_err());
    for slot in 0..KV_SLOT_COUNT { cache.append(slot, 0, &[0; K], &[0; K]).unwrap(); }
    end_session(&mut cache, &mut state);
    assert!(cache.all_slots_have_len(0)); assert_eq!(cache.capacity_positions(), 2);
    state.open().unwrap(); state.poisoned = true;
    end_session(&mut cache, &mut state);
    assert!(state.poisoned); assert!(state.open().is_err()); assert!(state.check().is_err());
}

#[test]
fn unwind_between_forwards_keeps_engine_poisoned_after_cleanup() {
    struct Guard<'a>(&'a mut KvCache, &'a mut RunState);
    impl Drop for Guard<'_> { fn drop(&mut self) { end_session(self.0, self.1); } }
    let mut cache = KvCache::try_with_capacity(1).unwrap();
    let mut state = RunState::default(); state.open().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = Guard(&mut cache, &mut state);
        panic!("injected task unwind");
    }));
    assert!(result.is_err()); assert!(!state.active); assert!(state.poisoned);
    assert!(cache.all_slots_have_len(0)); assert!(state.open().is_err());
}

#[test]
fn native_projection_preserves_group_scaling_and_charges_actual_group_dots() {
    use super::super::portable_int8::int4::convert::{Int4ConversionBudget, Int4ConversionRequirement};
    let source = [7.0, 7.0, 70.0, 70.0].map(Bf16::from_f32);
    let need = Int4ConversionRequirement::for_shape(1, 4, 2).unwrap();
    let matrix = GroupedInt4Matrix::from_bf16(1, 4, 2, &source, Int4ConversionBudget {
        max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: need.source_values,
        max_groups: need.groups }, &mut Continue).unwrap();
    let mut activation = ActivationBuffer::try_new(4).unwrap(); activation.encode_f32(&[127.0; 4]).unwrap();
    let mut ledger = ProjectionLedger::new(matrix.as_linear().planned_work(LinearRows::All).unwrap());
    let rope = RopeTablesF32::nanbeige(1).unwrap(); let mut scratch = [0; 2]; let mut control = Continue;
    let mut executor = Executor { final_norm: &[], rope: &rope, activation: &mut activation,
        unpack: &mut scratch, kernel: PackedInt4Kernel::scalar(), ledger: &mut ledger,
        control: &mut control, completed: 0, norms: 0 };
    assert_eq!(executor.project(&matrix).unwrap(), [Bf16::from_f32(19_558.0)]);
    assert_eq!(ledger.reserved(), ProjectionWork { dot_products: 2, multiply_accumulates: 4 });
}

#[test]
fn cancellation_cause_and_norm_overflow_are_not_hidden() {
    let error = GroupedInt4Error::Linear(LinearError::Cancelled(DecodeCancellationKind::Deadline));
    assert_eq!(StrictInt4Error::from(error), StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline));
    assert!(norm(&[Bf16::from_f32(1.0e30)], &[Bf16::from_f32(1.0)]).is_err());
    assert!(finite(&[Bf16::from_f32(f32::NAN)]).is_err());
}

/// The controller must explicitly arm this on a model-present DSR host. This
/// is a same-recipe consistency check, not BF16 parity or task qualification.
#[test]
#[ignore = "requires admitted pinned full source closure and multi-GB model memory"]
fn model_present_prefill_append_sparse_full_and_session_cleanup_agree() {
    use crate::artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input};
    let root = std::env::var_os("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var_os("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest path required");
    let source = prepare_nanbeige42_input(std::path::PathBuf::from(root),
        ConversionSourceManifest::load_pinned(std::path::PathBuf::from(manifest)).unwrap(), false, 64 * 1024).unwrap();
    let need = Int4ModelLoadRequirement::for_group_width(128).unwrap();
    let weights = Int4ModelWeights::from_prepared(&source, 128, Int4ModelLoadBudget {
        max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: need.source_values,
        max_groups: need.groups, max_shard_revalidation_bytes: need.shard_revalidation_bytes }, &mut Continue).unwrap();
    let memory = Int4MemoryRequirement::for_context(2, 128).unwrap();
    let mut engine = StrictInt4Engine::new(&weights, 2, Int4MemoryBudget { max_kv_bytes: memory.kv_bytes,
        max_rope_bytes: memory.rope_bytes, max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
    assert_eq!(engine.profile(), STRICT_INT4_PROFILE);
    let rows = LinearRows::Selected(&[0, 42, (V - 1) as u32]);
    let mut control = Continue;
    let expected;
    {
        let work = Int4Work::for_sequence(128, 0, 2, V + 3).unwrap();
        let mut session = engine.session(Int4RunBudget::exact(work), &mut control).unwrap();
        expected = session.prefill(&[1, 2], rows).unwrap();
        let full = session.logits(LinearRows::All).unwrap();
        assert_eq!(expected, [full[0], full[42], full[V - 1]]);
        assert_eq!(session.position().unwrap(), 2); assert_eq!(session.work(), work);
    }
    assert!(engine.kv_cache().all_slots_have_len(0)); assert!(!engine.is_poisoned());
    {
        let work = Int4Work::for_sequence(128, 0, 2, 3).unwrap();
        let mut session = engine.session(Int4RunBudget::exact(work), &mut control).unwrap();
        session.append(1).unwrap(); session.append(2).unwrap();
        assert_eq!(session.logits(rows).unwrap(), expected); assert_eq!(session.work(), work);
    }
    assert!(engine.kv_cache().all_slots_have_len(0));
}
