//! Opt-in bounded layer-major prefill on the existing strict INT8 session.
//!
//! Linear operators share weight traversals across prompt rows. Attention
//! remains token-ordered: append ONE row's K/V, attend to exactly that prefix,
//! then append the next. Never preappend the whole chunk to a causal cache.
//! Both passes use LoopRunner, including both post-pass norms. Sequential
//! append/prefill remain unchanged; callers must separately admit this extra
//! workspace. This candidate carries no model-parity or performance award.
use super::*;
use crate::native_engine::{looprun::GroupLayerExecutor, portable_int8::batch::MAX_BATCH_ROWS};

pub const LAYER_MAJOR_PREFILL_VERSION: &str = "portable-int8-layer-major-causal-prefill-v1";

#[derive(Clone, Copy, Debug)]
pub struct Int8PrefillLimits {
    /// Maximum simultaneous prompt rows; 1..=64. Long prompts use morsels.
    pub max_batch_rows: usize,
    /// ADDITIONAL payload allowance, on top of Int8MemoryRequirement. This is
    /// not a memory permit; the host owns admission, allocator slack and RSS.
    pub max_extra_scratch_bytes: u64,
}
impl Int8PrefillLimits {
    /// Conservative simultaneous payload bound including hidden/activation
    /// rows, Q/K/V, attention, residuals, MLP rails and vector descriptors.
    pub fn required_extra_scratch_bytes(rows: usize) -> Result<u64, StrictInt8Error> {
        if rows == 0 || rows > MAX_BATCH_ROWS { return Err(StrictInt8Error::Memory); }
        (rows as u64).checked_mul((32 * I * size_of::<f32>() + 4096) as u64)
            .ok_or(StrictInt8Error::Memory)
    }
    pub fn validate(self) -> Result<u64, StrictInt8Error> {
        let bytes = Self::required_extra_scratch_bytes(self.max_batch_rows)?;
        if bytes > self.max_extra_scratch_bytes { return Err(StrictInt8Error::Memory); }
        Ok(bytes)
    }
}

impl<C: DecodeStepControl> Int8Session<'_, '_, C> {
    /// Append an entire exact prompt in finite layer-major morsels, retaining
    /// only the final hidden state. Works from an existing complete prefix.
    /// Whole input/context/work and declared workspace are preflighted BEFORE
    /// mutation. Once native work starts, failure poisons the normal RAII
    /// session; no caller may continue from partly updated layer slots.
    pub fn append_layer_major(&mut self, tokens: &[u32], limits: Int8PrefillLimits)
        -> Result<(), StrictInt8Error> {
        check_tokens(tokens)?;
        limits.validate()?;
        self.preflight(tokens.len(), 0)?;
        let capacity = tokens.len().min(limits.max_batch_rows);
        let mut activations = Vec::new();
        activations.try_reserve_exact(capacity).map_err(|_| StrictInt8Error::Allocation)?;
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
            hidden.try_reserve_exact(chunk.len()).map_err(|_| StrictInt8Error::Allocation)?;
            for &token in chunk {
                poll(self.control)?;
                let source = self.engine.weights.embeddings.row(token as usize).map_err(|_| StrictInt8Error::Input)?;
                let mut row = filled(H, Bf16::from_bits(0))?;
                row.copy_from_slice(source); finite(&row)?; hidden.push(row);
            }
            let runner = LoopRunner::from_layer_weights(&self.engine.weights.layers);
            let mut executor = PrefillExecutor {
                final_norm: self.engine.weights.final_norm, rope: &self.engine.rope,
                cache: &mut self.engine.cache, activations: &mut activations[..chunk.len()],
                ledger: &mut self.ledger, control: &mut *self.control,
                start, rows: chunk.len(), completed: 0, norms: 0,
            };
            runner.run_group(&mut executor, &mut hidden)?;
            if executor.completed != KV_SLOT_COUNT || executor.norms != 2 { return Err(StrictInt8Error::Boundary); }
            if !executor.cache.all_slots_have_len(start + chunk.len()) { return Err(StrictInt8Error::Cache); }
            let last = hidden.pop().ok_or(StrictInt8Error::EmptyHidden)?;
            finite(&last)?; poll(self.control)?;
            self.hidden = Some(last);
            self.work.forward_positions = self.work.forward_positions.checked_add(bound.forward_positions).ok_or(StrictInt8Error::Work)?;
            self.work.attention_pairs = self.work.attention_pairs.checked_add(bound.attention_pairs).ok_or(StrictInt8Error::Work)?;
            self.engine.state.poisoned = false;
        }
        Ok(())
    }

    /// Full or genuinely selected final-position projection. No intermediate
    /// prompt position computes an lm-head row, including morsel boundaries.
    pub fn prefill_layer_major(&mut self, tokens: &[u32], rows: LinearRows<'_>, limits: Int8PrefillLimits)
        -> Result<Vec<f32>, StrictInt8Error> {
        check_tokens(tokens)?;
        limits.validate()?;
        self.preflight(tokens.len(), rows.checked_count(V)?)?;
        self.append_layer_major(tokens, limits)?;
        self.logits(rows)
    }
}
fn check_tokens(tokens: &[u32]) -> Result<(), StrictInt8Error> {
    if tokens.is_empty() || tokens.iter().any(|&id| id as usize >= V) { Err(StrictInt8Error::Input) } else { Ok(()) }
}

struct PrefillExecutor<'a, C> {
    final_norm: &'a [Bf16], rope: &'a RopeTablesF32, cache: &'a mut KvCache,
    activations: &'a mut [ActivationBuffer], ledger: &'a mut ProjectionLedger, control: &'a mut C,
    start: usize, rows: usize, completed: usize, norms: usize,
}
impl<'w, C: DecodeStepControl> GroupLayerExecutor<Int8Layer<'w>> for PrefillExecutor<'_, C> {
    type HiddenGroup = Vec<Vec<Bf16>>;
    type Error = StrictInt8Error;
    fn layer_group(&mut self, binding: &LayerBinding<'_, Int8Layer<'w>>, hidden: &mut Self::HiddenGroup)
        -> Result<(), Self::Error> {
        if slot_for(binding.loop_index(), binding.layer_index()) != Some(binding.kv_slot())
            || binding.kv_slot() != self.completed || hidden.len() != self.rows
            || hidden.iter().any(|row| row.len() != H) { return Err(StrictInt8Error::Boundary); }
        let layer = binding.weights();
        encode_norms(hidden, layer.norm1, self.activations, self.control)?;
        let mut query = project_group(layer.q, self.activations, self.ledger, self.control)?;
        let mut key = project_group(layer.k, self.activations, self.ledger, self.control)?;
        let value = project_group(layer.v, self.activations, self.ledger, self.control)?;
        let attention = attend_rows(&mut query, &mut key, &value, self.rows, self.start,
            binding.kv_slot(), self.rope, self.cache, self.control)?;
        drop(query); drop(key); drop(value);
        for (input, row) in self.activations.iter_mut().zip(attention.chunks_exact(Q)) { input.encode_bf16(row)?; }
        let update = project_group(layer.o, self.activations, self.ledger, self.control)?;
        for (row, update) in hidden.iter_mut().zip(update.chunks_exact(H)) {
            poll(self.control)?;
            *row = residual_add_f32_cast_back(row, update).map_err(|_| StrictInt8Error::Primitive)?;
            finite(row)?;
        }
        drop(attention); drop(update);
        encode_norms(hidden, layer.norm2, self.activations, self.control)?;
        let gate = project_group(layer.gate, self.activations, self.ledger, self.control)?;
        let up = project_group(layer.up, self.activations, self.ledger, self.control)?;
        for ((input, gate), up) in self.activations.iter_mut().zip(gate.chunks_exact(I)).zip(up.chunks_exact(I)) {
            poll(self.control)?;
            let product = swiglu_f32_cast_back(gate, up).map_err(|_| StrictInt8Error::Primitive)?;
            input.encode_bf16(&product)?;
        }
        drop(gate); drop(up);
        let update = project_group(layer.down, self.activations, self.ledger, self.control)?;
        for (row, update) in hidden.iter_mut().zip(update.chunks_exact(H)) {
            poll(self.control)?;
            *row = residual_add_f32_cast_back(row, update).map_err(|_| StrictInt8Error::Primitive)?;
            finite(row)?;
        }
        self.completed += 1; Ok(())
    }
    fn final_norm_group(&mut self, hidden: &mut Self::HiddenGroup) -> Result<(), Self::Error> {
        if self.norms >= 2 || self.completed != (self.norms + 1) * PHYSICAL_LAYER_COUNT || hidden.len() != self.rows {
            return Err(StrictInt8Error::Boundary);
        }
        for row in hidden { poll(self.control)?; *row = norm(row, self.final_norm)?; }
        self.norms += 1; Ok(())
    }
}
fn encode_norms<C: DecodeStepControl>(hidden: &[Vec<Bf16>], scale: &[Bf16],
    activations: &mut [ActivationBuffer], control: &mut C) -> Result<(), StrictInt8Error> {
    if hidden.len() != activations.len() { return Err(StrictInt8Error::Boundary); }
    for (row, activation) in hidden.iter().zip(activations) {
        poll(control)?; activation.encode_bf16(&norm(row, scale)?)?;
    }
    Ok(())
}
fn project_group<C: DecodeStepControl>(matrix: QuantizedLinear<'_>, inputs: &[ActivationBuffer],
    ledger: &mut ProjectionLedger, control: &mut C) -> Result<Vec<Bf16>, StrictInt8Error> {
    let length = matrix.rows().checked_mul(inputs.len()).ok_or(StrictInt8Error::Memory)?;
    let mut output = filled(length, Bf16::from_bits(0))?;
    matrix.project_batch_bf16_into(inputs, LinearRows::All, &mut output, ledger, control)?;
    Ok(output)
}

/// Separate causal seam exercised by model-free fixtures against the SAME
/// single-token attention primitive. Query/key mutate only their RoPE rails.
#[allow(clippy::too_many_arguments)]
fn attend_rows<C: DecodeStepControl>(query: &mut [Bf16], key: &mut [Bf16], value: &[Bf16],
    rows: usize, start: usize, slot: usize, rope: &RopeTablesF32, cache: &mut KvCache, control: &mut C)
    -> Result<Vec<Bf16>, StrictInt8Error> {
    if rows == 0 || rows > MAX_BATCH_ROWS || query.len() != rows * Q || key.len() != rows * K || value.len() != rows * K
        || start.checked_add(rows).is_none_or(|end| end > cache.capacity_positions()) {
        return Err(StrictInt8Error::Input);
    }
    if cache.len_for_slot(slot).map_err(|_| StrictInt8Error::Cache)? != start { return Err(StrictInt8Error::Cache); }
    let mut output = filled(rows * Q, Bf16::from_bits(0))?;
    let mut key_bits = filled(K, 0_u16)?; let mut value_bits = filled(K, 0_u16)?;
    for index in 0..rows {
        poll(control)?;
        let position = start + index;
        let q = &mut query[index * Q..(index + 1) * Q];
        let k = &mut key[index * K..(index + 1) * K];
        let v = &value[index * K..(index + 1) * K];
        for head in q.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt8Error::Rope)?; }
        for head in k.chunks_exact_mut(NANBEIGE_HEAD_DIM) { rope.apply_split_half(position, head).map_err(|_| StrictInt8Error::Rope)?; }
        finite(q)?; finite(k)?; finite(v)?;
        for (out, value) in key_bits.iter_mut().zip(k.iter()) { *out = value.to_bits(); }
        for (out, value) in value_bits.iter_mut().zip(v) { *out = value.to_bits(); }
        cache.append(slot, position, &key_bits, &value_bits).map_err(|_| StrictInt8Error::Cache)?;
        poll(control)?;
        let attention = eager_gqa_attention_from_cache(q, cache, slot).map_err(|_| StrictInt8Error::Attention)?;
        finite(&attention)?;
        output[index * Q..(index + 1) * Q].copy_from_slice(&attention);
    }
    poll(control)?; Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)] struct Control { polls: usize, cancel: Option<usize> }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.cancel == Some(self.polls)).then_some(DecodeCancellationKind::User)
        }
    }
    #[test] fn workspace_limits_are_finite_and_checked_in_bytes() {
        for rows in [1, 3, 4, 5, MAX_BATCH_ROWS] {
            let bytes = Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            assert!(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes }.validate().is_ok());
            assert!(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: bytes - 1 }.validate().is_err());
        }
        for rows in [0, 65, usize::MAX] { assert!(Int8PrefillLimits::required_extra_scratch_bytes(rows).is_err()); }
    }
    #[test] fn morsel_work_is_exactly_the_whole_sequence_without_intermediate_heads() {
        for start in [0, 7, 63] { for size in [1, 3, 4, 8, 64] {
            let tokens = vec![0_u32; 67]; let mut at = start; let mut sum = Int8Work::default();
            for chunk in tokens.chunks(size) {
                sum = sum.checked_add(Int8Work::for_sequence(at, chunk.len(), 0).unwrap()).unwrap(); at += chunk.len();
            }
            sum = sum.checked_add(Int8Work::for_sequence(at, 0, V).unwrap()).unwrap();
            assert_eq!(sum, Int8Work::for_sequence(start, tokens.len(), V).unwrap());
        } }
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
        let output = attend_rows(&mut query, &mut key, &values, rows, start, slot, &rope, &mut grouped, &mut Control::default()).unwrap();
        let mut reference = Vec::new();
        for i in 0..rows {
            let mut q = original_query[i * Q..(i + 1) * Q].to_vec();
            let mut k = original_key[i * K..(i + 1) * K].to_vec();
            // Independent token-at-a-time replay, not the grouped helper.
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
    #[test] fn batched_prompt_attention_never_sees_future_tokens() { attention_case(0, 0); }
    #[test] fn existing_prefix_and_second_loop_keep_exact_causal_coordinates() { attention_case(2, 22); }
    #[test] fn malformed_attention_geometry_does_not_append_any_cache_row() {
        let mut cache = KvCache::try_with_capacity(2).unwrap(); let rope = RopeTablesF32::nanbeige(2).unwrap();
        assert!(attend_rows(&mut [Bf16::from_bits(0)], &mut [], &[], 1, 0, 0, &rope, &mut cache, &mut Control::default()).is_err());
        assert!(cache.all_slots_have_len(0));
    }
    #[test] fn cancellation_before_a_row_cannot_append_its_kv() {
        let mut cache = KvCache::try_with_capacity(2).unwrap(); let rope = RopeTablesF32::nanbeige(2).unwrap();
        let mut q = vec![Bf16::from_bits(0); Q]; let mut k = vec![Bf16::from_bits(0); K]; let v = k.clone();
        assert!(matches!(attend_rows(&mut q, &mut k, &v, 1, 0, 0, &rope, &mut cache,
            &mut Control { polls: 0, cancel: Some(1) }), Err(StrictInt8Error::Cancelled(DecodeCancellationKind::User))));
        assert!(cache.all_slots_have_len(0));
    }
    #[test] fn invalid_prompt_tokens_are_rejected_as_one_complete_input() {
        assert!(check_tokens(&[]).is_err()); assert!(check_tokens(&[0, V as u32]).is_err());
        assert!(check_tokens(&[0, V as u32 - 1]).is_ok());
    }
}
