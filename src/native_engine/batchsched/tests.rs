//! Executable synthetic source regressions; not a native-model parity award.
use super::*;
use super::super::{
    kv::{KV_ELEMENTS_PER_POSITION, PHYSICAL_LAYER_COUNT},
    weights::Bf16Matrix,
    looprun::{GroupLayerExecutor, LayerBinding, PositionContext, StructuralLayerExecutor},
};
struct Continue;
impl BatchControl for Continue { fn checkpoint(&mut self) -> Option<DecodeCancellationKind> { None } }
fn matrix() -> Bf16Matrix {
    Bf16Matrix::new(4, 3, [1.0, -1.0, 0.5, 0.125, 100.0, -0.25, -2.0, 0.0, 2.0, 3.0, 4.0, 5.0]
        .into_iter().map(Bf16::from_f32).collect()).unwrap()
}
fn token(sequence: BatchSequence) -> BatchToken { BatchToken { sequence, token_id: 1, projection: BatchProjection::None } }
fn append(cache: &mut KvCache, position: usize) {
    let values = vec![0_u16; KV_ELEMENTS_PER_POSITION];
    for slot in 0..KV_SLOT_COUNT { cache.append(slot, position, &values, &values).unwrap(); }
}
#[test]
fn projection_matches_existing_bf16_and_lm_head_cast_paths_per_row() {
    let matrix = matrix();
    let inputs: Vec<Vec<Bf16>> = [[0.1, -0.2, 0.3], [2.0, 1.0, -1.0], [-0.0, 0.0, 0.0]]
        .iter().map(|row| row.iter().copied().map(Bf16::from_f32).collect()).collect();
    let views: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
    let activations = linear::activation(&matrix, &inputs, &mut Continue).unwrap();
    let logits = linear::project(&matrix, &views, &mut Continue, |x| Bf16::from_f32(x).to_f32()).unwrap();
    for index in 0..inputs.len() {
        assert_eq!(activations[index], matrix.project_f32_accumulate_cast_back(&inputs[index]).unwrap());
        assert_eq!(logits[index].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            matrix.project_f32_accumulate_bf16_then_export(&inputs[index]).unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
}
#[test]
fn projection_is_composition_and_permutation_independent() {
    let matrix = matrix();
    let rows: Vec<Vec<Bf16>> = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1e5, 1e-5, -1e5]]
        .iter().map(|r| r.iter().copied().map(Bf16::from_f32).collect()).collect();
    let all = linear::activation(&matrix, &rows, &mut Continue).unwrap();
    let reordered = linear::activation(&matrix, &[rows[2].clone(), rows[0].clone()], &mut Continue).unwrap();
    assert_eq!(all[2], reordered[0]); assert_eq!(all[0], reordered[1]);
    for index in 0..rows.len() {
        assert_eq!(linear::activation(&matrix, &[rows[index].clone()], &mut Continue).unwrap()[0], all[index]);
    }
}
#[test]
fn cancellation_stops_projection_between_weight_rows() {
    struct Stop(usize);
    impl BatchControl for Stop {
        fn checkpoint(&mut self) -> Option<DecodeCancellationKind> {
            self.0 += 1; (self.0 == 2).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let input = vec![Bf16::from_f32(1.0); 3]; let mut control = Stop(0);
    assert!(matches!(linear::activation(&matrix(), &[input], &mut control),
        Err(BatchError::Cancelled(DecodeCancellationKind::Deadline))));
    assert_eq!(control.0, 2);
}
#[test]
fn payload_prices_all_ragged_kv_slots_and_one_shared_rope() {
    let estimate = BatchEnvelope::estimate(&[2, 5]).unwrap();
    assert_eq!(estimate.kv_bytes, 7 * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(estimate.rope_bytes, (5 * 128 + 64) * 4);
    assert_eq!(estimate.full_logit_bytes, 2 * NANBEIGE_VOCAB_SIZE as u64 * 4);
    assert!(BatchEnvelope::compile(&[2, 5], estimate.total_bytes - 1).is_err());
    assert_eq!(BatchEnvelope::compile(&[2, 5], estimate.total_bytes).unwrap().payload(), estimate);
    for bad in [vec![], vec![0], vec![MAX_BATCH_CONTEXT + 1], vec![1; MAX_BATCH_ROWS + 1]] {
        assert!(BatchEnvelope::estimate(&bad).is_err());
    }
}
#[test]
fn ragged_positions_follow_sequence_handles_not_input_order() {
    let mut pool = pool::SequencePool::new(&[3, 4]).unwrap();
    let a = pool.open(0).unwrap(); let b = pool.open(1).unwrap();
    append(&mut pool.rows[0].cache, 0); append(&mut pool.rows[1].cache, 0); append(&mut pool.rows[1].cache, 1);
    let (slots, positions) = pool.preflight(&[token(b), token(a)]).unwrap();
    assert_eq!(slots, [1, 0]); assert_eq!(positions, [2, 1]);
    assert!(matches!(pool.preflight(&[token(a), token(a)]), Err(BatchError::DuplicateSequence)));
    assert_eq!(pool.len(a).unwrap(), 1); assert_eq!(pool.len(b).unwrap(), 2);
}
#[test]
fn stale_and_foreign_handles_cannot_alias_recycled_cache_slots() {
    let mut pool = pool::SequencePool::new(&[2]).unwrap();
    let old = pool.open(0).unwrap(); append(&mut pool.rows[0].cache, 0); pool.close(old).unwrap();
    let new = pool.open(0).unwrap(); assert_ne!(old, new); assert!(pool.len(old).is_err()); assert_eq!(pool.len(new).unwrap(), 0);
    let mut other = pool::SequencePool::new(&[2]).unwrap(); let foreign = other.open(0).unwrap();
    assert!(pool.len(foreign).is_err()); assert!(matches!(pool.open(0), Err(BatchError::SequenceBusy)));
}
#[test]
fn partial_step_retirement_preserves_untouched_sequences() {
    let mut pool = pool::SequencePool::new(&[2, 2, 2]).unwrap();
    let a = pool.open(0).unwrap(); let b = pool.open(1).unwrap(); let c = pool.open(2).unwrap();
    append(&mut pool.rows[2].cache, 0);
    {
        let transaction = pool::StepTransaction::new(&mut pool, &[0, 1]);
        let values = vec![0; KV_ELEMENTS_PER_POSITION];
        transaction.pool.rows[0].cache.append(0, 0, &values, &values).unwrap();
    }
    assert!(pool.len(a).is_err()); assert!(pool.len(b).is_err()); assert_eq!(pool.len(c).unwrap(), 1);
    assert!(pool.rows[0].cache.all_slots_have_len(0)); assert!(pool.rows[1].cache.all_slots_have_len(0));
}
#[test]
fn unwinding_retires_group_and_complete_commit_preserves_new_lengths() {
    let mut pool = pool::SequencePool::new(&[2]).unwrap(); let old = pool.open(0).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let transaction = pool::StepTransaction::new(&mut pool, &[0]);
        append(&mut transaction.pool.rows[0].cache, 0); panic!("synthetic worker failure");
    }));
    assert!(panic.is_err()); assert!(pool.len(old).is_err());
    let new = pool.open(0).unwrap();
    {
        let mut transaction = pool::StepTransaction::new(&mut pool, &[0]);
        append(&mut transaction.pool.rows[0].cache, 0); transaction.commit(&[0]).unwrap();
    }
    assert_eq!(pool.len(new).unwrap(), 1);
}
#[test]
fn capacity_or_divergence_is_refused_before_advancing_any_row() {
    let mut pool = pool::SequencePool::new(&[1, 2]).unwrap(); let a = pool.open(0).unwrap(); let b = pool.open(1).unwrap();
    append(&mut pool.rows[0].cache, 0);
    assert!(matches!(pool.preflight(&[token(b), token(a)]), Err(BatchError::ContextFull)));
    assert_eq!(pool.len(b).unwrap(), 0);
    let values = vec![0; KV_ELEMENTS_PER_POSITION]; pool.rows[1].cache.append(0, 0, &values, &values).unwrap();
    assert!(pool.preflight(&[token(b)]).is_err());
}
#[test]
fn group_and_scalar_use_identical_shared_loop_boundaries() {
    struct Scalar(Vec<usize>);
    impl StructuralLayerExecutor<usize> for Scalar {
        type Hidden = i64; type Error = ();
        fn layer_forward(&mut self, binding: &LayerBinding<'_, usize>, h: &mut i64, _: PositionContext) -> Result<(), ()> {
            self.0.push(binding.kv_slot()); *h += *binding.weights() as i64; Ok(())
        }
        fn final_rms_norm(&mut self, h: &mut i64, _: PositionContext) -> Result<(), ()> { self.0.push(100); *h *= 2; Ok(()) }
    }
    struct Group(Vec<usize>);
    impl GroupLayerExecutor<usize> for Group {
        type HiddenGroup = Vec<i64>; type Error = ();
        fn layer_group(&mut self, binding: &LayerBinding<'_, usize>, hidden: &mut Vec<i64>) -> Result<(), ()> {
            self.0.push(binding.kv_slot()); for h in hidden { *h += *binding.weights() as i64; } Ok(())
        }
        fn final_norm_group(&mut self, hidden: &mut Vec<i64>) -> Result<(), ()> {
            self.0.push(100); for h in hidden { *h *= 2; } Ok(())
        }
    }
    let weights: [usize; PHYSICAL_LAYER_COUNT] = std::array::from_fn(|i| i + 1);
    let runner = LoopRunner::from_layer_weights(&weights); let mut scalar = Scalar(Vec::new()); let mut single = 7;
    runner.run_token_structural(&mut scalar, &mut single, PositionContext::at(3)).unwrap();
    let mut group = Group(Vec::new()); let mut many = vec![7, 11]; runner.run_group(&mut group, &mut many).unwrap();
    assert_eq!(scalar.0, group.0); assert_eq!(many[0], single);
    assert_eq!(group.0[22], 100); assert_eq!(group.0[45], 100); assert_eq!(many[1] - many[0], 16);
    assert!(std::ptr::eq(runner.binding(0, 0).unwrap().weights(), runner.binding(1, 0).unwrap().weights()));
}
