//! Operator-major INT4 linear work with independently addressed causal KV.
use super::*;
use crate::native_engine::{layer::NANBEIGE_Q_PROJECTION_SIZE as Q, looprun::GroupLayerExecutor};

pub(super) struct Executor<'a, C> {
    pub(super) final_norm: &'a [Bf16], pub(super) rope: &'a RopeTablesF32,
    pub(super) sequences: &'a mut [Sequence], pub(super) activations: &'a mut [ActivationBuffer],
    pub(super) unpack: &'a mut [i8], pub(super) kernel: PackedInt4Kernel,
    pub(super) steps: &'a [Int4CohortToken], pub(super) positions: &'a [usize],
    pub(super) ledger: &'a mut ProjectionLedger, pub(super) control: &'a mut C,
    pub(super) completed: usize, pub(super) norms: usize,
}
impl<C: DecodeStepControl> GroupLayerExecutor<Int4Layer> for Executor<'_, C> {
    type HiddenGroup = Vec<Vec<Bf16>>;
    type Error = StrictInt4Error;
    fn layer_group(&mut self, binding: &LayerBinding<'_, Int4Layer>, hidden: &mut Self::HiddenGroup)
        -> Result<(), Self::Error> {
        if slot_for(binding.loop_index(), binding.layer_index()) != Some(binding.kv_slot())
            || binding.kv_slot() != self.completed || hidden.len() != self.steps.len()
            || hidden.iter().any(|row| row.len() != H) { return Err(StrictInt4Error::Boundary); }
        let layer = binding.weights();
        encode_norms(hidden, &layer.norm1, self.activations, self.control)?;
        let mut query = self.project(&layer.q)?;
        let mut key = self.project(&layer.k)?;
        let value = self.project(&layer.v)?;
        let attention = attend_sequences(&mut query, &mut key, &value, self.steps, self.positions,
            binding.kv_slot(), self.rope, self.sequences, self.control)?;
        drop(query); drop(key); drop(value);
        for (input, row) in self.activations.iter_mut().zip(attention.chunks_exact(Q)) { input.encode_bf16(row)?; }
        let update = self.project(&layer.o)?;
        add_rows(hidden, &update, self.control)?;
        drop(attention); drop(update);
        encode_norms(hidden, &layer.norm2, self.activations, self.control)?;
        let gate = self.project(&layer.gate)?;
        let up = self.project(&layer.up)?;
        for ((input, gate), up) in self.activations.iter_mut().zip(gate.chunks_exact(I)).zip(up.chunks_exact(I)) {
            poll(self.control)?;
            input.encode_bf16(&swiglu_f32_cast_back(gate, up).map_err(|_| StrictInt4Error::Primitive)?)?;
        }
        drop(gate); drop(up);
        let update = self.project(&layer.down)?;
        add_rows(hidden, &update, self.control)?;
        self.completed += 1; Ok(())
    }
    fn final_norm_group(&mut self, hidden: &mut Self::HiddenGroup) -> Result<(), Self::Error> {
        if self.norms >= 2 || self.completed != (self.norms + 1) * PHYSICAL_LAYER_COUNT
            || hidden.len() != self.steps.len() { return Err(StrictInt4Error::Boundary); }
        for row in hidden { poll(self.control)?; *row = norm(row, self.final_norm)?; }
        self.norms += 1; Ok(())
    }
}
impl<C: DecodeStepControl> Executor<'_, C> {
    fn project(&mut self, matrix: &GroupedInt4Matrix) -> Result<Vec<Bf16>, StrictInt4Error> {
        let length = matrix.rows().checked_mul(self.activations.len()).ok_or(StrictInt4Error::Memory)?;
        let mut output = filled(length, Bf16::from_bits(0))?;
        matrix.as_linear().with_kernel(self.kernel).project_batch_bf16_into(self.activations, LinearRows::All,
            &mut output, self.unpack, self.ledger, self.control)?;
        Ok(output)
    }
}
fn add_rows<C: DecodeStepControl>(hidden: &mut [Vec<Bf16>], update: &[Bf16], control: &mut C)
    -> Result<(), StrictInt4Error> {
    if update.len() != hidden.len() * H { return Err(StrictInt4Error::Boundary); }
    for (row, update) in hidden.iter_mut().zip(update.chunks_exact(H)) {
        poll(control)?; *row = residual_add_f32_cast_back(row, update).map_err(|_| StrictInt4Error::Primitive)?;
        finite(row)?;
    }
    Ok(())
}
fn encode_norms<C: DecodeStepControl>(hidden: &[Vec<Bf16>], scale: &[Bf16],
    activations: &mut [ActivationBuffer], control: &mut C) -> Result<(), StrictInt4Error> {
    if hidden.len() != activations.len() { return Err(StrictInt4Error::Boundary); }
    for (row, input) in hidden.iter().zip(activations) { poll(control)?; input.encode_bf16(&norm(row, scale)?)?; }
    Ok(())
}
/// Validate even the LAST sequence address before touching any cache. Unlike a
/// prompt morsel, this step has at most one token per independent sequence.
fn check_addresses(steps: &[Int4CohortToken], positions: &[usize], slot: usize,
    sequences: &[Sequence]) -> Result<(), StrictInt4Error> {
    check_indices(steps.iter().map(|step| step.sequence), sequences.len())?;
    if positions.len() != steps.len() { return Err(StrictInt4Error::Input); }
    for (step, &position) in steps.iter().zip(positions) {
        let cache = &sequences[step.sequence].cache;
        if cache.len_for_slot(slot).map_err(|_| StrictInt4Error::Cache)? != position { return Err(StrictInt4Error::Cache); }
        if position >= cache.capacity_positions() { return Err(StrictInt4Error::Context); }
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn attend_sequences<C: DecodeStepControl>(query: &mut [Bf16], key: &mut [Bf16], value: &[Bf16],
    steps: &[Int4CohortToken], positions: &[usize], slot: usize, rope: &RopeTablesF32,
    sequences: &mut [Sequence], control: &mut C) -> Result<Vec<Bf16>, StrictInt4Error> {
    check_addresses(steps, positions, slot, sequences)?;
    if query.len() != steps.len() * Q || key.len() != steps.len() * K
        || value.len() != steps.len() * K { return Err(StrictInt4Error::Input); }
    let mut output = filled(query.len(), Bf16::from_bits(0))?;
    let mut key_bits = filled(K, 0_u16)?; let mut value_bits = filled(K, 0_u16)?;
    for (index, (step, &position)) in steps.iter().zip(positions).enumerate() {
        poll(control)?;
        let q = &mut query[index * Q..(index + 1) * Q];
        let k = &mut key[index * K..(index + 1) * K];
        let v = &value[index * K..(index + 1) * K];
        for head in q.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt4Error::Rope)?; }
        for head in k.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt4Error::Rope)?; }
        finite(q)?; finite(k)?; finite(v)?;
        for (out, value) in key_bits.iter_mut().zip(k.iter()) { *out = value.to_bits(); }
        for (out, value) in value_bits.iter_mut().zip(v) { *out = value.to_bits(); }
        let cache = &mut sequences[step.sequence].cache;
        cache.append(slot, position, &key_bits, &value_bits).map_err(|_| StrictInt4Error::Cache)?;
        poll(control)?;
        let attention = eager_gqa_attention_from_cache(q, cache, slot).map_err(|_| StrictInt4Error::Attention)?;
        finite(&attention)?; output[index * Q..(index + 1) * Q].copy_from_slice(&attention);
    }
    poll(control)?; Ok(output)
}
