//! Concrete 48:8 GQA, shared Q/K/V/O and MLP groups, retaining the eager casts.
use super::*;
use super::super::{
    attention::eager_gqa_attention_from_cache_with_control,
    decode::DecodeStepControl,
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
            attention.push(attend(&query[index], cache, binding.kv_slot(), self.control)?);
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
/// Borrow the batch's one checkpoint authority; never copy or renew its
/// allowance for a head, cache position or document.
struct AttentionControl<'a, C>(&'a mut C);
impl<C: BatchControl> DecodeStepControl for AttentionControl<'_, C> {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.0.checkpoint()
    }
}
fn attend<C: BatchControl>(
    query: &[Bf16],
    cache: &KvCache,
    slot: usize,
    control: &mut C,
) -> Result<Vec<Bf16>, BatchError> {
    eager_gqa_attention_from_cache_with_control(query, cache, slot, &mut AttentionControl(control))
        .map_err(BatchError::from)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::attention::{
        ATTENTION_POSITIONS_PER_CHECKPOINT, eager_gqa_attention_from_cache,
    };

    struct Control {
        polls: usize,
        cancel_at: Option<usize>,
        cause: DecodeCancellationKind,
    }
    impl Control {
        fn running() -> Self {
            Self { polls: 0, cancel_at: None, cause: DecodeCancellationKind::User }
        }
    }
    impl BatchControl for Control {
        fn checkpoint(&mut self) -> Option<DecodeCancellationKind> {
            self.polls += 1;
            (self.cancel_at == Some(self.polls)).then_some(self.cause)
        }
    }
    fn causes() -> [DecodeCancellationKind; 11] {
        use DecodeCancellationKind::*;
        [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit]
    }
    fn fixture(positions: usize, slot: usize) -> (KvCache, Vec<Bf16>) {
        let mut cache = KvCache::try_with_capacity(positions).unwrap();
        for position in 0..positions {
            let key = (0..NANBEIGE_KV_PROJECTION_SIZE)
                .map(|index| Bf16::from_f32(((index * 7 + position * 3) % 41) as f32 / 32.0 - 0.5).to_bits())
                .collect::<Vec<_>>();
            let value = (0..NANBEIGE_KV_PROJECTION_SIZE)
                .map(|index| Bf16::from_f32(((index * 13 + position * 11) % 97) as f32 / 8.0 - 6.0).to_bits())
                .collect::<Vec<_>>();
            cache.append(slot, position, &key, &value).unwrap();
        }
        let query = (0..NANBEIGE_Q_PROJECTION_SIZE)
            .map(|index| Bf16::from_f32((index % 19) as f32 / 16.0 - 0.5)).collect();
        (cache, query)
    }

    #[test]
    fn batch_attention_matches_reference_bits_across_context_and_loop_edges() {
        for slot in [0, 22, 43] {
            for positions in [1, 31, 32, 33, 65] {
                let (cache, query) = fixture(positions, slot);
                let expected = eager_gqa_attention_from_cache(&query, &cache, slot).unwrap();
                let mut control = Control::running();
                let actual = attend(&query, &cache, slot, &mut control).unwrap();
                assert_eq!(actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
                assert!(control.polls > 1, "the owning batch is polled within attention");
            }
        }
    }

    #[test]
    fn batch_attention_uses_one_allowance_and_preserves_every_cause_in_every_phase() {
        let (cache, query) = fixture(65, 22);
        let blocks = 65_usize.div_ceil(ATTENTION_POSITIONS_PER_CHECKPOINT);
        for cause in causes() {
            for phase in 0..6 {
                let cancel_at = 1 + phase * blocks + 2;
                let mut control = Control { polls: 0, cancel_at: Some(cancel_at), cause };
                assert!(matches!(attend(&query, &cache, 22, &mut control),
                    Err(BatchError::Cancelled(actual)) if actual == cause));
                assert_eq!(control.polls, cancel_at);
            }
        }
    }

    #[test]
    fn batch_attention_refuses_cancelled_entry_and_completed_output_without_renewing_control() {
        let (cache, query) = fixture(2, 0);
        let mut completed = Control::running();
        attend(&query, &cache, 0, &mut completed).unwrap();
        for cause in causes() {
            for cancel_at in [1, completed.polls] {
                let mut control = Control { polls: 0, cancel_at: Some(cancel_at), cause };
                assert!(matches!(attend(&query, &cache, 0, &mut control),
                    Err(BatchError::Cancelled(actual)) if actual == cause));
                assert_eq!(control.polls, cancel_at);
            }
        }
    }

    #[test]
    fn batch_attention_preserves_shape_refusals_without_polling() {
        let (cache, _) = fixture(1, 0);
        let mut control = Control { polls: 0, cancel_at: Some(1), cause: DecodeCancellationKind::User };
        assert!(matches!(attend(&[], &cache, 0, &mut control),
            Err(BatchError::Native(HfBf16EagerError::Attention(AttentionError::QueryVectorLength { .. })))));
        assert_eq!(control.polls, 0);
    }

    #[test]
    fn batch_error_adapters_keep_attention_cancellation_and_allocation_typed() {
        for cause in causes() {
            assert!(matches!(BatchError::from(AttentionError::Cancelled(cause)),
                BatchError::Cancelled(actual) if actual == cause));
            assert!(matches!(BatchError::from(HfBf16EagerError::Attention(AttentionError::Cancelled(cause))),
                BatchError::Cancelled(actual) if actual == cause));
        }
        assert!(matches!(BatchError::from(AttentionError::Allocation), BatchError::Allocation));
        assert!(matches!(BatchError::from(HfBf16EagerError::Attention(AttentionError::Allocation)),
            BatchError::Allocation));
        assert!(matches!(BatchError::from(AttentionError::EmptySequence),
            BatchError::Native(HfBf16EagerError::Attention(AttentionError::EmptySequence))));
    }
}
