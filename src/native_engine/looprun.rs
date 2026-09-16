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
    pub const fn loop_index(&self) -> usize { self.loop_index }
    /// The zero-based physical layer index.
    #[must_use]
    pub const fn layer_index(&self) -> usize { self.layer_index }
    /// The precomputed 44-slot K/V destination.
    #[must_use]
    pub const fn kv_slot(&self) -> usize { self.kv_slot }
    /// The physical layer's resolved weight reference.
    #[must_use]
    pub const fn weights(&self) -> &W { self.weights }
}

/// Numerics-profile-specific execution behind the single loop architecture.
/// `layer_forward` appends one K/V vector to the resolved slot and position.
pub trait LayerExecutor<W> {
    type Hidden;
    type Error;
    fn layer_forward(&mut self, binding: &LayerBinding<'_, W>, hidden: &mut Self::Hidden,
        positions: PositionContext, kv_cache: &mut KvCache) -> Result<(), Self::Error>;
    fn final_rms_norm(&mut self, hidden: &mut Self::Hidden, positions: PositionContext) -> Result<(), Self::Error>;
}

/// An executor whose numerics profile owns its cache representation.
/// Diagnostic-f32 retains f32 K/V, while consuming the same binding schedule.
pub trait StructuralLayerExecutor<W> {
    type Hidden;
    type Error;
    fn layer_forward(&mut self, binding: &LayerBinding<'_, W>, hidden: &mut Self::Hidden,
        positions: PositionContext) -> Result<(), Self::Error>;
    fn final_rms_norm(&mut self, hidden: &mut Self::Hidden, positions: PositionContext) -> Result<(), Self::Error>;
}

/// A group owns a separate position and cache for every participating sequence.
/// There is intentionally NO scalar PositionContext here: broadcasting one
/// row's position over a ragged group would silently corrupt attention/RoPE.
/// Implementations batch compatible linear operators and retain per-row causal
/// state. A final norm transforms every row before the next pass can begin.
pub trait GroupLayerExecutor<W> {
    type HiddenGroup;
    type Error;
    fn layer_group(&mut self, binding: &LayerBinding<'_, W>, hidden: &mut Self::HiddenGroup) -> Result<(), Self::Error>;
    fn final_norm_group(&mut self, hidden: &mut Self::HiddenGroup) -> Result<(), Self::Error>;
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
            LayerBinding { loop_index, layer_index, kv_slot, weights: &layer_weights[layer_index] }
        });
        Self { bindings }
    }
    /// The complete fixed binding table in runner execution order.
    #[must_use]
    pub const fn bindings(&self) -> &[LayerBinding<'weights, W>; KV_SLOT_COUNT] { &self.bindings }
    /// Looks up a resolved binding outside the hot loop for diagnostics/tests.
    #[must_use]
    pub fn binding(&self, loop_index: usize, layer_index: usize) -> Option<&LayerBinding<'weights, W>> {
        slot_for(loop_index, layer_index).and_then(|slot| self.bindings.get(slot))
    }

    /// The sole schedule walker. None is exactly one post-pass final norm;
    /// no executor gets a boundary at which it could re-inject an embedding.
    fn walk<E>(&self, mut visit: impl FnMut(Option<&LayerBinding<'weights, W>>) -> Result<(), E>) -> Result<(), E> {
        for loop_index in 0..LOOP_COUNT {
            let offset = loop_index * PHYSICAL_LAYER_COUNT;
            for binding in &self.bindings[offset..offset + PHYSICAL_LAYER_COUNT] { visit(Some(binding))?; }
            visit(None)?;
        }
        Ok(())
    }
    /// Executes both passes for one token position. The first final norm's
    /// exact mutable hidden state is presented directly to loop two layer zero.
    pub fn run_token<E>(&self, executor: &mut E, hidden: &mut E::Hidden,
        positions: PositionContext, kv_cache: &mut KvCache) -> Result<(), E::Error>
    where E: LayerExecutor<W> {
        self.walk(|binding| match binding {
            Some(binding) => executor.layer_forward(binding, hidden, positions, kv_cache),
            None => executor.final_rms_norm(hidden, positions),
        })
    }
    /// Executes the identical schedule with profile-owned non-bf16 cache state.
    pub fn run_token_structural<E>(&self, executor: &mut E, hidden: &mut E::Hidden,
        positions: PositionContext) -> Result<(), E::Error>
    where E: StructuralLayerExecutor<W> {
        self.walk(|binding| match binding {
            Some(binding) => executor.layer_forward(binding, hidden, positions),
            None => executor.final_rms_norm(hidden, positions),
        })
    }
    /// One logical layer operation per group, then one norm per group/pass.
    /// The executor owns independent row positions; linear batching does not
    /// require equal context lengths and never implies padded attention.
    pub fn run_group<E>(&self, executor: &mut E, hidden: &mut E::HiddenGroup) -> Result<(), E::Error>
    where E: GroupLayerExecutor<W> {
        self.walk(|binding| match binding {
            Some(binding) => executor.layer_group(binding, hidden),
            None => executor.final_norm_group(hidden),
        })
    }
}
