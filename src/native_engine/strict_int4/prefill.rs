//! Explicit layer-major prompt execution on the existing grouped-INT4 session.
//!
//! Bounded prompt morsels share packed group reductions. Attention appends
//! ONE token's K/V before attending to precisely that prefix. Both loop passes
//! and post-pass norms use LoopRunner; no cache is prefilled with future rows.
//! Limits declare additional workspace, not a host memory permit or speed claim.
use super::*;
use crate::native_engine::{layer::NANBEIGE_Q_PROJECTION_SIZE as Q,
    looprun::GroupLayerExecutor, portable_int8::int4::batch::MAX_INT4_BATCH_ROWS};

pub const INT4_LAYER_MAJOR_PREFILL_VERSION: &str = "grouped-int4-layer-major-causal-prefill-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int4PrefillLimits {
    /// Simultaneous prompt rows, 1..=64; longer prompts use finite morsels.
    pub max_batch_rows: usize,
    /// Additional payload, on top of Int4MemoryRequirement. The existing
    /// engine owns one group-unpacking buffer, independent of prompt rows.
    pub max_extra_scratch_bytes: u64,
}
impl Int4PrefillLimits {
    pub fn required_extra_scratch_bytes(rows: usize) -> Result<u64, StrictInt4Error> {
        if rows == 0 || rows > MAX_INT4_BATCH_ROWS { return Err(StrictInt4Error::Memory); }
        // Conservative bound for simultaneously live BF16 rails, activation
        // bytes, descriptors and finite projection tiles. Allocator slack,
        // model, KV, RoPE and attention scratch remain separately priced.
        (rows as u64).checked_mul((32 * I * size_of::<f32>() + 4096) as u64)
            .ok_or(StrictInt4Error::Memory)
    }
    pub fn validate(self) -> Result<u64, StrictInt4Error> {
        let bytes = Self::required_extra_scratch_bytes(self.max_batch_rows)?;
        if bytes > self.max_extra_scratch_bytes { return Err(StrictInt4Error::Memory); }
        Ok(bytes)
    }
}

impl<C: DecodeStepControl> Int4Session<'_, '_, C> {
    /// Admit the ENTIRE exact prompt before mutation, including continuation
    /// from a complete prefix. Once execution starts, errors poison the normal
    /// session; Drop clears all 44 slots without refunding performed work.
    pub fn append_layer_major(&mut self, tokens: &[u32], limits: Int4PrefillLimits)
        -> Result<(), StrictInt4Error> {
        check_tokens(tokens)?;
        limits.validate()?;
        self.preflight(tokens.len(), 0)?;
        let capacity = tokens.len().min(limits.max_batch_rows);
        let mut activations = Vec::new();
        activations.try_reserve_exact(capacity).map_err(|_| StrictInt4Error::Allocation)?;
        for _ in 0..capacity { activations.push(ActivationBuffer::try_new(I)?); }
        for chunk in tokens.chunks(limits.max_batch_rows) {
            let start = self.position()?;
            let bound = self.preflight(chunk.len(), 0)?;
            self.remaining_positions -= bound.forward_positions;
            self.remaining_attention -= bound.attention_pairs;
            self.engine.state.poisoned = true;
            self.hidden = None;
            poll(self.control)?;
            let mut hidden = Vec::new();
            hidden.try_reserve_exact(chunk.len()).map_err(|_| StrictInt4Error::Allocation)?;
            for &token in chunk {
                poll(self.control)?;
                let source = self.engine.weights.embeddings.row(token as usize).map_err(|_| StrictInt4Error::Input)?;
                let mut row = filled(H, Bf16::from_bits(0))?;
                row.copy_from_slice(source); finite(&row)?; hidden.push(row);
            }
            let runner = LoopRunner::from_layer_weights(&self.engine.weights.layers);
            let mut executor = PrefillExecutor {
                final_norm: &self.engine.weights.final_norm, rope: &self.engine.rope,
                cache: &mut self.engine.cache, activations: &mut activations[..chunk.len()],
                unpack: &mut self.engine.unpack, kernel: self.engine.kernel,
                ledger: &mut self.ledger, control: &mut *self.control,
                start, rows: chunk.len(), completed: 0, norms: 0,
            };
            runner.run_group(&mut executor, &mut hidden)?;
            if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt4Error::Boundary); }
            if !executor.cache.all_slots_have_len(start + chunk.len()) { return Err(StrictInt4Error::Cache); }
            let last = hidden.pop().ok_or(StrictInt4Error::EmptyHidden)?;
            finite(&last)?; poll(self.control)?;
            self.hidden = Some(last);
            self.work.forward_positions = self.work.forward_positions.checked_add(bound.forward_positions).ok_or(StrictInt4Error::Work)?;
            self.work.attention_pairs = self.work.attention_pairs.checked_add(bound.attention_pairs).ok_or(StrictInt4Error::Work)?;
            self.engine.state.poisoned = false;
        }
        Ok(())
    }
    /// Project only the final prompt position, including at morsel boundaries.
    /// Selected rows stay a real lm-head slice; no intermediate head is run.
    pub fn prefill_layer_major(&mut self, tokens: &[u32], rows: LinearRows<'_>, limits: Int4PrefillLimits)
        -> Result<Vec<f32>, StrictInt4Error> {
        check_tokens(tokens)?;
        limits.validate()?;
        self.preflight(tokens.len(), rows.checked_count(V)?)?;
        self.append_layer_major(tokens, limits)?;
        self.logits(rows)
    }
}
fn check_tokens(tokens: &[u32]) -> Result<(), StrictInt4Error> {
    if tokens.is_empty() || tokens.iter().any(|&id| id as usize >= V) { Err(StrictInt4Error::Input) } else { Ok(()) }
}

struct PrefillExecutor<'a, C> {
    final_norm: &'a [Bf16], rope: &'a RopeTablesF32, cache: &'a mut KvCache,
    activations: &'a mut [ActivationBuffer], unpack: &'a mut [i8], kernel: PackedInt4Kernel,
    ledger: &'a mut ProjectionLedger, control: &'a mut C,
    start: usize, rows: usize, completed: usize, norms: usize,
}
impl<C: DecodeStepControl> GroupLayerExecutor<Int4Layer> for PrefillExecutor<'_, C> {
    type HiddenGroup = Vec<Vec<Bf16>>;
    type Error = StrictInt4Error;
    fn layer_group(&mut self, binding: &LayerBinding<'_, Int4Layer>, hidden: &mut Self::HiddenGroup)
        -> Result<(), Self::Error> {
        if slot_for(binding.loop_index(), binding.layer_index()) != Some(binding.kv_slot())
            || binding.kv_slot() != self.completed || hidden.len() != self.rows
            || hidden.iter().any(|row| row.len() != H) { return Err(StrictInt4Error::Boundary); }
        let layer = binding.weights();
        encode_norms(hidden, &layer.norm1, self.activations, self.control)?;
        let mut query = self.project(&layer.q)?;
        let mut key = self.project(&layer.k)?;
        let value = self.project(&layer.v)?;
        let attention = attend_rows(&mut query, &mut key, &value, self.rows, self.start,
            binding.kv_slot(), self.rope, self.cache, self.control)?;
        drop(query); drop(key); drop(value);
        for (input, row) in self.activations.iter_mut().zip(attention.chunks_exact(Q)) { input.encode_bf16(row)?; }
        let update = self.project(&layer.o)?;
        for (row, update) in hidden.iter_mut().zip(update.chunks_exact(H)) {
            poll(self.control)?;
            *row = residual_add_f32_cast_back(row, update).map_err(|_| StrictInt4Error::Primitive)?;
            finite(row)?;
        }
        drop(attention); drop(update);
        encode_norms(hidden, &layer.norm2, self.activations, self.control)?;
        let gate = self.project(&layer.gate)?;
        let up = self.project(&layer.up)?;
        for ((input, gate), up) in self.activations.iter_mut().zip(gate.chunks_exact(I)).zip(up.chunks_exact(I)) {
            poll(self.control)?;
            let product = swiglu_f32_cast_back(gate, up).map_err(|_| StrictInt4Error::Primitive)?;
            input.encode_bf16(&product)?;
        }
        drop(gate); drop(up);
        let update = self.project(&layer.down)?;
        for (row, update) in hidden.iter_mut().zip(update.chunks_exact(H)) {
            poll(self.control)?;
            *row = residual_add_f32_cast_back(row, update).map_err(|_| StrictInt4Error::Primitive)?;
            finite(row)?;
        }
        self.completed += 1; Ok(())
    }
    fn final_norm_group(&mut self, hidden: &mut Self::HiddenGroup) -> Result<(), Self::Error> {
        if self.norms >= 2 || self.completed != (self.norms + 1) * PHYSICAL_LAYER_COUNT || hidden.len() != self.rows {
            return Err(StrictInt4Error::Boundary);
        }
        for row in hidden { poll(self.control)?; *row = norm(row, self.final_norm)?; }
        self.norms += 1; Ok(())
    }
}
impl<C: DecodeStepControl> PrefillExecutor<'_, C> {
    fn project(&mut self, matrix: &GroupedInt4Matrix) -> Result<Vec<Bf16>, StrictInt4Error> {
        let length = matrix.rows().checked_mul(self.activations.len()).ok_or(StrictInt4Error::Memory)?;
        let mut output = filled(length, Bf16::from_bits(0))?;
        matrix.as_linear().with_kernel(self.kernel).project_batch_bf16_into(self.activations, LinearRows::All,
            &mut output, self.unpack, self.ledger, self.control)?;
        Ok(output)
    }
}
fn encode_norms<C: DecodeStepControl>(hidden: &[Vec<Bf16>], scale: &[Bf16],
    activations: &mut [ActivationBuffer], control: &mut C) -> Result<(), StrictInt4Error> {
    if hidden.len() != activations.len() { return Err(StrictInt4Error::Boundary); }
    for (row, activation) in hidden.iter().zip(activations) {
        poll(control)?; activation.encode_bf16(&norm(row, scale)?)?;
    }
    Ok(())
}

/// The causal seam is fixture-tested against the existing single-row primitive.
#[allow(clippy::too_many_arguments)]
fn attend_rows<C: DecodeStepControl>(query: &mut [Bf16], key: &mut [Bf16], value: &[Bf16],
    rows: usize, start: usize, slot: usize, rope: &RopeTablesF32, cache: &mut KvCache, control: &mut C)
    -> Result<Vec<Bf16>, StrictInt4Error> {
    if rows == 0 || rows > MAX_INT4_BATCH_ROWS || query.len() != rows * Q || key.len() != rows * K || value.len() != rows * K
        || start.checked_add(rows).is_none_or(|end| end > cache.capacity_positions()) {
        return Err(StrictInt4Error::Input);
    }
    if cache.len_for_slot(slot).map_err(|_| StrictInt4Error::Cache)? != start { return Err(StrictInt4Error::Cache); }
    let mut output = filled(rows * Q, Bf16::from_bits(0))?;
    let mut key_bits = filled(K, 0_u16)?; let mut value_bits = filled(K, 0_u16)?;
    for index in 0..rows {
        poll(control)?;
        let position = start + index;
        let q = &mut query[index * Q..(index + 1) * Q];
        let k = &mut key[index * K..(index + 1) * K];
        let v = &value[index * K..(index + 1) * K];
        for head in q.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt4Error::Rope)?; }
        for head in k.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt4Error::Rope)?; }
        finite(q)?; finite(k)?; finite(v)?;
        for (out, value) in key_bits.iter_mut().zip(k.iter()) { *out = value.to_bits(); }
        for (out, value) in value_bits.iter_mut().zip(v) { *out = value.to_bits(); }
        cache.append(slot, position, &key_bits, &value_bits).map_err(|_| StrictInt4Error::Cache)?;
        poll(control)?;
        let attention = eager_gqa_attention_from_cache(q, cache, slot).map_err(|_| StrictInt4Error::Attention)?;
        finite(&attention)?;
        output[index * Q..(index + 1) * Q].copy_from_slice(&attention);
    }
    poll(control)?; Ok(output)
}

#[cfg(test)] mod tests;
