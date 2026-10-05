//! Operator-major linear work with independently addressed causal attention.
use super::*;
use crate::native_engine::looprun::GroupLayerExecutor;

pub(super) struct Executor<'a, C> {
    pub(super) final_norm: &'a [Bf16], pub(super) rope: &'a RopeTablesF32,
    pub(super) sequences: &'a mut [Sequence], pub(super) activations: &'a mut [ActivationBuffer],
    pub(super) steps: &'a [CohortToken], pub(super) positions: &'a [usize],
    pub(super) ledger: &'a mut ProjectionLedger, pub(super) control: &'a mut C,
    pub(super) completed: usize, pub(super) norms: usize,
}
impl<'w, C: DecodeStepControl> GroupLayerExecutor<Int8Layer<'w>> for Executor<'_, C> {
    type HiddenGroup = Vec<Vec<Bf16>>;
    type Error = StrictInt8Error;
    fn layer_group(&mut self, binding: &LayerBinding<'_, Int8Layer<'w>>, hidden: &mut Self::HiddenGroup)
        -> Result<(), Self::Error> {
        if slot_for(binding.loop_index(), binding.layer_index()) != Some(binding.kv_slot())
            || binding.kv_slot() != self.completed || hidden.len() != self.steps.len()
            || hidden.iter().any(|row| row.len() != H) { return Err(StrictInt8Error::Boundary); }
        let layer = binding.weights();
        encode_norms(hidden, layer.norm1, self.activations, self.control)?;
        let mut query = project_group(layer.q, self.activations, self.ledger, self.control)?;
        let mut key = project_group(layer.k, self.activations, self.ledger, self.control)?;
        let value = project_group(layer.v, self.activations, self.ledger, self.control)?;
        let attention = attend_sequences(&mut query, &mut key, &value, self.steps, self.positions,
            binding.kv_slot(), self.rope, self.sequences, self.control)?;
        drop(query); drop(key); drop(value);
        for (input, row) in self.activations.iter_mut().zip(attention.chunks_exact(Q)) { input.encode_bf16(row)?; }
        let update = project_group(layer.o, self.activations, self.ledger, self.control)?;
        add_rows(hidden, &update, self.control)?;
        drop(attention); drop(update);
        encode_norms(hidden, layer.norm2, self.activations, self.control)?;
        let gate = project_group(layer.gate, self.activations, self.ledger, self.control)?;
        let up = project_group(layer.up, self.activations, self.ledger, self.control)?;
        for ((input, gate), up) in self.activations.iter_mut().zip(gate.chunks_exact(I)).zip(up.chunks_exact(I)) {
            poll(self.control)?;
            input.encode_bf16(&swiglu_f32_cast_back(gate, up).map_err(|_| StrictInt8Error::Primitive)?)?;
        }
        drop(gate); drop(up);
        let update = project_group(layer.down, self.activations, self.ledger, self.control)?;
        add_rows(hidden, &update, self.control)?;
        self.completed += 1; Ok(())
    }
    fn final_norm_group(&mut self, hidden: &mut Self::HiddenGroup) -> Result<(), Self::Error> {
        if self.norms >= 2 || self.completed != (self.norms + 1) * PHYSICAL_LAYER_COUNT
            || hidden.len() != self.steps.len() { return Err(StrictInt8Error::Boundary); }
        for row in hidden { poll(self.control)?; *row = norm(row, self.final_norm)?; }
        self.norms += 1; Ok(())
    }
}
fn add_rows<C: DecodeStepControl>(hidden: &mut [Vec<Bf16>], update: &[Bf16], control: &mut C)
    -> Result<(), StrictInt8Error> {
    if update.len() != hidden.len() * H { return Err(StrictInt8Error::Boundary); }
    for (row, update) in hidden.iter_mut().zip(update.chunks_exact(H)) {
        poll(control)?; *row = residual_add_f32_cast_back(row, update).map_err(|_| StrictInt8Error::Primitive)?;
        finite(row)?;
    }
    Ok(())
}
fn encode_norms<C: DecodeStepControl>(hidden: &[Vec<Bf16>], scale: &[Bf16],
    activations: &mut [ActivationBuffer], control: &mut C) -> Result<(), StrictInt8Error> {
    if hidden.len() != activations.len() { return Err(StrictInt8Error::Boundary); }
    for (row, input) in hidden.iter().zip(activations) { poll(control)?; input.encode_bf16(&norm(row, scale)?)?; }
    Ok(())
}
fn project_group<C: DecodeStepControl>(matrix: QuantizedLinear<'_>, inputs: &[ActivationBuffer],
    ledger: &mut ProjectionLedger, control: &mut C) -> Result<Vec<Bf16>, StrictInt8Error> {
    let length = matrix.rows().checked_mul(inputs.len()).ok_or(StrictInt8Error::Memory)?;
    let mut output = filled(length, Bf16::from_bits(0))?;
    matrix.project_batch_bf16_into(inputs, LinearRows::All, &mut output, ledger, control)?; Ok(output)
}

/// No padded or broadcast attention. Validate the complete ragged address set
/// before the first KV write; each row then uses the existing scalar primitive.
#[allow(clippy::too_many_arguments)]
pub(super) fn attend_sequences<C: DecodeStepControl>(query: &mut [Bf16], key: &mut [Bf16], value: &[Bf16],
    steps: &[CohortToken], positions: &[usize], slot: usize, rope: &RopeTablesF32,
    sequences: &mut [Sequence], control: &mut C) -> Result<Vec<Bf16>, StrictInt8Error> {
    check_indices(steps.iter().map(|step| step.sequence), sequences.len())?;
    if positions.len() != steps.len() || query.len() != steps.len() * Q
        || key.len() != steps.len() * K || value.len() != steps.len() * K { return Err(StrictInt8Error::Input); }
    for (step, &position) in steps.iter().zip(positions) {
        let cache = &sequences[step.sequence].cache;
        if position >= cache.capacity_positions() { return Err(StrictInt8Error::Context); }
        if cache.len_for_slot(slot).map_err(|_| StrictInt8Error::Cache)? != position { return Err(StrictInt8Error::Cache); }
    }
    let mut output = filled(query.len(), Bf16::from_bits(0))?;
    let mut key_bits = filled(K, 0_u16)?; let mut value_bits = filled(K, 0_u16)?;
    for (index, (step, &position)) in steps.iter().zip(positions).enumerate() {
        poll(control)?;
        let q = &mut query[index * Q..(index + 1) * Q];
        let k = &mut key[index * K..(index + 1) * K];
        let v = &value[index * K..(index + 1) * K];
        for head in q.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt8Error::Rope)?; }
        for head in k.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt8Error::Rope)?; }
        finite(q)?; finite(k)?; finite(v)?;
        for (out, value) in key_bits.iter_mut().zip(k.iter()) { *out = value.to_bits(); }
        for (out, value) in value_bits.iter_mut().zip(v) { *out = value.to_bits(); }
        let cache = &mut sequences[step.sequence].cache;
        cache.append(slot, position, &key_bits, &value_bits).map_err(|_| StrictInt8Error::Cache)?;
        poll(control)?;
        let attention = eager_gqa_attention_from_cache(q, cache, slot).map_err(|_| StrictInt8Error::Attention)?;
        finite(&attention)?; output[index * Q..(index + 1) * Q].copy_from_slice(&attention);
    }
    poll(control)?; Ok(output)
}
