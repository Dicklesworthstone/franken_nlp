//! Profile-agnostic two-pass decoder loop runner.
//!
//! There is intentionally one implementation of Nanbeige's loop architecture:
//! 22 physical layers run twice, each pass is followed by the same final
//! RMSNorm, and loop two consumes that normalized hidden state directly. The
//! runner owns only straight-line control flow; selected numerics profiles own
//! the math behind [`LayerExecutor`].

use super::kv::{KV_SLOT_COUNT, KvCache, LOOP_COUNT, PHYSICAL_LAYER_COUNT, slot_for};

/// Logical position identities shared unchanged by both decoder passes.
///
/// The current model has no loop-specific mask, cache, or RoPE coordinate.
/// Keeping the identities explicit makes a future divergence a type/API change
/// instead of an accidental boundary adjustment in a layer backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionContext {
    /// Token position in the sequence.
    pub position: usize,
    /// KV-cache position used by every one of the 44 slots.
    pub cache_position: usize,
    /// RoPE position used by both loop passes.
    pub rope_position: usize,
    /// Causal-mask position used by both loop passes.
    pub mask_position: usize,
}

impl PositionContext {
    /// Builds the model's common coordinate set for one token position.
    #[must_use]
    pub const fn at(position: usize) -> Self {
        Self {
            position,
            cache_position: position,
            rope_position: position,
            mask_position: position,
        }
    }
}

/// Prebuilt hot-loop binding of one physical weight reference to one logical KV
/// slot. The same physical weights appear once in each of the two passes.
#[derive(Clone, Copy)]
pub struct LayerBinding<'weights, W> {
    loop_index: usize,
    layer_index: usize,
    kv_slot: usize,
    weights: &'weights W,
}

impl<W> LayerBinding<'_, W> {
    /// The zero-based logical loop pass.
    #[must_use]
    pub const fn loop_index(&self) -> usize {
        self.loop_index
    }

    /// The zero-based physical layer index.
    #[must_use]
    pub const fn layer_index(&self) -> usize {
        self.layer_index
    }

    /// The precomputed 44-slot K/V destination.
    #[must_use]
    pub const fn kv_slot(&self) -> usize {
        self.kv_slot
    }

    /// The physical layer's resolved weight reference.
    #[must_use]
    pub const fn weights(&self) -> &W {
        self.weights
    }
}

/// Numerics-profile-specific execution behind the single loop architecture.
///
/// `layer_forward` receives a resolved binding rather than a logical layer key,
/// so it performs neither map lookups nor slot arithmetic in the hot loop. It
/// must append one K and V vector at `positions.cache_position` in the supplied
/// binding's slot. `final_rms_norm` is deliberately invoked after *each* pass.
pub trait LayerExecutor<W> {
    /// Profile-specific hidden-state representation.
    type Hidden;
    /// Profile-specific failure returned without erasing its diagnostics.
    type Error;

    /// Executes one physical layer at its already-resolved logical slot.
    fn layer_forward(
        &mut self,
        binding: &LayerBinding<'_, W>,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
        kv_cache: &mut KvCache,
    ) -> Result<(), Self::Error>;

    /// Applies the model's shared final RMSNorm at one loop boundary.
    fn final_rms_norm(
        &mut self,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
    ) -> Result<(), Self::Error>;
}

/// An executor whose numerics profile owns its cache representation.
///
/// The diagnostic-f32 profile uses this entry point so its K/V activations stay
/// f32 after the one-time weight widening.  It still consumes this module's
/// single resolved 44-binding schedule; it is not a second loop implementation.
pub trait StructuralLayerExecutor<W> {
    /// Profile-specific hidden-state representation.
    type Hidden;
    /// Profile-specific failure returned without erasing its diagnostics.
    type Error;

    /// Executes a binding in the precomputed two-pass schedule.
    fn layer_forward(
        &mut self,
        binding: &LayerBinding<'_, W>,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
    ) -> Result<(), Self::Error>;

    /// Applies the shared post-pass final RMSNorm.
    fn final_rms_norm(
        &mut self,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
    ) -> Result<(), Self::Error>;
}

/// Prebuilt, fixed-size binding table for the two-pass decoder.
pub struct LoopRunner<'weights, W> {
    bindings: [LayerBinding<'weights, W>; KV_SLOT_COUNT],
}

impl<'weights, W> LoopRunner<'weights, W> {
    /// Resolves every `(loop, layer)` binding once at engine construction.
    #[must_use]
    pub fn from_layer_weights(layer_weights: &'weights [W; PHYSICAL_LAYER_COUNT]) -> Self {
        let bindings = std::array::from_fn(|index| {
            let loop_index = index / PHYSICAL_LAYER_COUNT;
            let layer_index = index % PHYSICAL_LAYER_COUNT;
            let kv_slot = slot_for(loop_index, layer_index)
                .expect("binding table indexes are constrained to two loops of 22 layers");
            LayerBinding {
                loop_index,
                layer_index,
                kv_slot,
                weights: &layer_weights[layer_index],
            }
        });
        Self { bindings }
    }

    /// The complete fixed binding table in runner execution order.
    #[must_use]
    pub const fn bindings(&self) -> &[LayerBinding<'weights, W>; KV_SLOT_COUNT] {
        &self.bindings
    }

    /// Looks up a resolved binding outside the hot loop for diagnostics/tests.
    #[must_use]
    pub fn binding(
        &self,
        loop_index: usize,
        layer_index: usize,
    ) -> Option<&LayerBinding<'weights, W>> {
        slot_for(loop_index, layer_index).and_then(|slot| self.bindings.get(slot))
    }

    /// Executes both 22-layer passes for one token position.
    ///
    /// `hidden` is passed continuously through all 44 layers. In particular,
    /// the output written by the first `final_rms_norm` call is the exact same
    /// mutable hidden state presented to loop 2 layer 0; there is no embedding
    /// re-injection or loop-boundary projection surface in this API.
    pub fn run_token<E>(
        &self,
        executor: &mut E,
        hidden: &mut E::Hidden,
        positions: PositionContext,
        kv_cache: &mut KvCache,
    ) -> Result<(), E::Error>
    where
        E: LayerExecutor<W>,
    {
        for loop_index in 0..LOOP_COUNT {
            let offset = loop_index * PHYSICAL_LAYER_COUNT;
            for binding in &self.bindings[offset..offset + PHYSICAL_LAYER_COUNT] {
                executor.layer_forward(binding, hidden, positions, kv_cache)?;
            }
            executor.final_rms_norm(hidden, positions)?;
        }
        Ok(())
    }

    /// Executes the same two-pass schedule for a profile with non-bf16 K/V
    /// storage.  The sole distinction from [`Self::run_token`] is that the
    /// executor owns its typed cache; binding order and boundary norms remain
    /// exactly the same shared implementation contract.
    pub fn run_token_structural<E>(
        &self,
        executor: &mut E,
        hidden: &mut E::Hidden,
        positions: PositionContext,
    ) -> Result<(), E::Error>
    where
        E: StructuralLayerExecutor<W>,
    {
        for loop_index in 0..LOOP_COUNT {
            let offset = loop_index * PHYSICAL_LAYER_COUNT;
            for binding in &self.bindings[offset..offset + PHYSICAL_LAYER_COUNT] {
                executor.layer_forward(binding, hidden, positions)?;
            }
            executor.final_rms_norm(hidden, positions)?;
        }
        Ok(())
    }
}
