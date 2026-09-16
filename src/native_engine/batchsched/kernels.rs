//! Concrete 48:8 GQA, shared Q/K/V/O and MLP groups, retaining the eager casts.
use super::*;
use super::super::{
    attention::eager_gqa_attention_from_cache,
    layer::HfBf16LayerError,
    looprun::{GroupLayerExecutor, LayerBinding},
    nn::{ReferencePrimitiveError, RMS_NORM_EPSILON, residual_add_f32_cast_back,
        rms_norm_f32_reduce_cast_back, swiglu_f32_cast_back},
};

pub(super) struct GroupExecutor<'a, C> {
    pub rows: &'a mut [pool::Row], pub slots: &'a [usize], pub positions: &'a [usize],
    pub rope: &'a RopeTablesF32, pub final_norm: &'a [Bf16], pub control: &'a mut C,
}
impl<C: BatchControl> GroupLayerExecutor<HfBf16EagerLayerWeights> for GroupExecutor<'_, C> {
    type HiddenGroup = Vec<Vec<Bf16>>;
    type Error = BatchError;
    fn layer_group(&mut self, binding: &LayerBinding<'_, HfBf16EagerLayerWeights>, hidden: &mut Self::HiddenGroup)
        -> Result<(), BatchError> {
        let layer = binding.weights();
        let norm = map_rows(hidden, self.control, |row| layer.input_rms_norm(row).map_err(HfBf16EagerError::from).map_err(BatchError::from))?;
        let mut query = linear::activation(&layer.q_proj, &norm, self.control)?;
        let mut key = linear::activation(&layer.k_proj, &norm, self.control)?;
        let value = linear::activation(&layer.v_proj, &norm, self.control)?;
        drop(norm);
        let mut attention = reserve(hidden.len())?;
        for index in 0..hidden.len() {
            checkpoint(self.control)?;
            let position = self.positions[index];
            for head in query[index].chunks_exact_mut(NANBEIGE_HEAD_DIM) {
                self.rope.apply_split_half(position, head).map_err(HfBf16EagerError::from)?;
            }
            for head in key[index].chunks_exact_mut(NANBEIGE_HEAD_DIM) {
                self.rope.apply_split_half(position, head).map_err(HfBf16EagerError::from)?;
            }
            let mut key_bits = reserve(key[index].len())?; key_bits.extend(key[index].iter().map(|v| v.to_bits()));
            let mut value_bits = reserve(value[index].len())?; value_bits.extend(value[index].iter().map(|v| v.to_bits()));
            let cache = &mut self.rows[self.slots[index]].cache;
            cache.append(binding.kv_slot(), position, &key_bits, &value_bits).map_err(HfBf16EagerError::from)?;
            // Only this sequence's cache and position can enter its attention.
            // Different lengths do not split otherwise-compatible linear groups.
            attention.push(eager_gqa_attention_from_cache(&query[index], cache, binding.kv_slot())
                .map_err(HfBf16EagerError::from)?);
        }
        drop(query); drop(key); drop(value);
        let projected = linear::activation(&layer.o_proj, &attention, self.control)?;
        drop(attention);
        let mut after_attention = reserve(hidden.len())?;
        for (original, projected) in hidden.iter().zip(&projected) {
            checkpoint(self.control)?;
            after_attention.push(residual_add_f32_cast_back(original, projected).map_err(primitive)?);
        }
        drop(projected);
        let normalized = map_rows(&after_attention, self.control, |row| {
            rms_norm_f32_reduce_cast_back(row, &layer.post_attention_norm, RMS_NORM_EPSILON).map_err(primitive)
        })?;
        let gate = linear::activation(&layer.gate_proj, &normalized, self.control)?;
        let up = linear::activation(&layer.up_proj, &normalized, self.control)?;
        drop(normalized);
        let mut activated = reserve(hidden.len())?;
        for (gate, up) in gate.iter().zip(&up) {
            checkpoint(self.control)?; activated.push(swiglu_f32_cast_back(gate, up).map_err(primitive)?);
        }
        drop(gate); drop(up);
        let down = linear::activation(&layer.down_proj, &activated, self.control)?;
        drop(activated);
        for ((destination, residual), down) in hidden.iter_mut().zip(&after_attention).zip(&down) {
            checkpoint(self.control)?;
            *destination = residual_add_f32_cast_back(residual, down).map_err(primitive)?;
        }
        check_hidden(hidden)
    }
    fn final_norm_group(&mut self, hidden: &mut Self::HiddenGroup) -> Result<(), BatchError> {
        for row in hidden.iter_mut() {
            checkpoint(self.control)?;
            *row = rms_norm_f32_reduce_cast_back(row, self.final_norm, RMS_NORM_EPSILON).map_err(primitive)?;
        }
        check_hidden(hidden)
    }
}
fn map_rows<C: BatchControl>(rows: &[Vec<Bf16>], control: &mut C,
    mut operation: impl FnMut(&[Bf16]) -> Result<Vec<Bf16>, BatchError>) -> Result<Vec<Vec<Bf16>>, BatchError> {
    let mut result = reserve(rows.len())?;
    for row in rows { checkpoint(control)?; result.push(operation(row)?); }
    Ok(result)
}
fn primitive(error: ReferencePrimitiveError) -> BatchError { HfBf16EagerError::from(HfBf16LayerError::from(error)).into() }
fn check_hidden(rows: &[Vec<Bf16>]) -> Result<(), BatchError> {
    if rows.iter().flatten().any(|x| !x.to_f32().is_finite()) { return Err(BatchError::InvalidNumerics); }
    Ok(())
}
