//! Forty-four-slot KV cache for Nanbeige's two-pass decoder loop.
//!
//! The cache deliberately models logical layer executions, not physical weight
//! layers: a token owns one K and one V vector in each of 44 independent slots.
//! Storage uses raw bf16 bit-patterns (`u16`) so numerics profiles decide their
//! own conversion policy without changing addressing or append semantics.

use std::array;

/// Physical decoder layers shared by the two logical passes.
pub const PHYSICAL_LAYER_COUNT: usize = 22;
/// Logical decoder passes per token.
pub const LOOP_COUNT: usize = 2;
/// Independent logical K/V slots (`22 * 2`).
pub const KV_SLOT_COUNT: usize = PHYSICAL_LAYER_COUNT * LOOP_COUNT;
/// Key/value attention heads, not query heads.
pub const KV_HEAD_COUNT: usize = 8;
/// Explicit configured head width. Never derive this from `hidden / q_heads`.
pub const KV_HEAD_DIM: usize = 128;
/// Raw bf16 scalars in one key or value vector at one token position.
pub const KV_ELEMENTS_PER_POSITION: usize = KV_HEAD_COUNT * KV_HEAD_DIM;
/// Bytes for K and V at one logical slot and one token position.
pub const KV_BYTES_PER_SLOT_POSITION: usize = 2 * KV_ELEMENTS_PER_POSITION * size_of::<u16>();
/// Nanbeige's 44-slot bf16 K/V footprint per token: 180,224 bytes / 176 KiB.
pub const KV_BYTES_PER_TOKEN: usize = KV_SLOT_COUNT * KV_BYTES_PER_SLOT_POSITION;

/// Logical token range used by the initial paged-KV candidate.
///
/// A logical page is an addressing/admission unit, never an instruction to
/// allocate one monolithic 44-slot buffer. Physical slabs remain independently
/// owned by a [`KvSlabKey`].
pub const KV_LOGICAL_PAGE_TOKENS: usize = 16;
/// Separate K and V slabs for every logical layer slot in the baseline layout.
pub const KV_SLABS_PER_LOGICAL_PAGE: usize = KV_SLOT_COUNT * 2;
/// Bytes in one bf16 K-or-V slab for one layer slot and 16 token positions.
pub const KV_BF16_SLAB_BYTES: usize =
    KV_LOGICAL_PAGE_TOKENS * KV_ELEMENTS_PER_POSITION * size_of::<u16>();
/// Payload bytes in one 16-token logical page across all 44 K/V slots.
pub const KV_BF16_LOGICAL_PAGE_BYTES: usize = KV_BYTES_PER_TOKEN * KV_LOGICAL_PAGE_TOKENS;
/// Int8 K/V payload bytes per token before scale and page-table accounting.
pub const KV_INT8_PAYLOAD_BYTES_PER_TOKEN: usize = KV_BYTES_PER_TOKEN / 2;
/// Number of K/V scales per token in an int8 KV representation.
pub const KV_INT8_SCALE_COUNT_PER_TOKEN: usize = KV_SLOT_COUNT * KV_HEAD_COUNT * 2;
/// f32 scale bytes per int8 K/V token.
pub const KV_INT8_F32_SCALE_BYTES_PER_TOKEN: usize =
    KV_INT8_SCALE_COUNT_PER_TOKEN * size_of::<f32>();
/// f16 scale bytes per int8 K/V token.
pub const KV_INT8_F16_SCALE_BYTES_PER_TOKEN: usize =
    KV_INT8_SCALE_COUNT_PER_TOKEN * size_of::<u16>();

/// Storage representation selected for a physical K/V slab.
///
/// The current slab cache admits bf16 data only; the int8 variants exist now
/// so their payload and scale charges remain structural rather than becoming a
/// late unpriced execution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvSlabDtype {
    /// Raw bf16 bit patterns, two bytes per element.
    Bf16,
    /// One-byte K/V payload with f32 scales.
    Int8F32Scale,
    /// One-byte K/V payload with f16 scales.
    Int8F16Scale,
}

impl KvSlabDtype {
    /// Payload and per-token scale bytes for the fixed 44-slot layout.
    #[must_use]
    pub const fn bytes_per_token(self) -> usize {
        match self {
            Self::Bf16 => KV_BYTES_PER_TOKEN,
            Self::Int8F32Scale => {
                KV_INT8_PAYLOAD_BYTES_PER_TOKEN + KV_INT8_F32_SCALE_BYTES_PER_TOKEN
            }
            Self::Int8F16Scale => {
                KV_INT8_PAYLOAD_BYTES_PER_TOKEN + KV_INT8_F16_SCALE_BYTES_PER_TOKEN
            }
        }
    }
}

/// Address and ownership identity for one independently reference-counted
/// physical K/V slab.
///
/// `loop_layer_start` and `loop_layer_count` make grouping an explicit
/// measured choice. The baseline allocator uses one logical layer and keeps K
/// and V separate, so a 16-token group has 88 bf16 slabs rather than one
/// all-44-slot allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvSlabKey {
    /// First logical token position covered by this slab.
    pub token_start: usize,
    /// Number of logical token positions represented by the slab.
    pub token_count: usize,
    /// First logical loop-layer slot covered by this slab.
    pub loop_layer_start: usize,
    /// Number of consecutive logical loop-layer slots in the grouping.
    pub loop_layer_count: usize,
    /// Whether this slab carries K or V values.
    pub vector: KvVector,
    /// Physical payload/scales representation.
    pub dtype: KvSlabDtype,
}

impl KvSlabKey {
    /// Builds and bounds-checks an independently charged slab identity.
    pub fn new(
        token_start: usize,
        token_count: usize,
        loop_layer_start: usize,
        loop_layer_count: usize,
        vector: KvVector,
        dtype: KvSlabDtype,
    ) -> Result<Self, KvSlabError> {
        if token_count == 0 || loop_layer_count == 0 {
            return Err(KvSlabError::InvalidSlabKey {
                token_count,
                loop_layer_start,
                loop_layer_count,
            });
        }
        token_start
            .checked_add(token_count)
            .ok_or(KvSlabError::AdmissionArithmeticOverflow {
                positions: token_start,
            })?;
        let end = loop_layer_start.checked_add(loop_layer_count).ok_or(
            KvSlabError::InvalidSlabKey {
                token_count,
                loop_layer_start,
                loop_layer_count,
            },
        )?;
        if end > KV_SLOT_COUNT {
            return Err(KvSlabError::InvalidSlabKey {
                token_count,
                loop_layer_start,
                loop_layer_count,
            });
        }
        Ok(Self {
            token_start,
            token_count,
            loop_layer_start,
            loop_layer_count,
            vector,
            dtype,
        })
    }

    /// Payload-plus-scale bytes charged by this slab, excluding page-table
    /// metadata and allocator padding.
    pub fn payload_bytes(self) -> Result<usize, KvSlabError> {
        let bytes_per_slot_vector = self.dtype.bytes_per_token() / KV_SLABS_PER_LOGICAL_PAGE;
        self.token_count
            .checked_mul(self.loop_layer_count)
            .and_then(|positions| positions.checked_mul(bytes_per_slot_vector))
            .ok_or(KvSlabError::AdmissionArithmeticOverflow {
                positions: self.token_count,
            })
    }
}

/// Typed failures from the refcounted slab protocol.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvSlabError {
    /// A requested slab has an empty or out-of-range logical slot grouping.
    InvalidSlabKey {
        /// Requested token-range length.
        token_count: usize,
        /// Requested first loop-layer slot.
        loop_layer_start: usize,
        /// Requested loop-layer grouping width.
        loop_layer_count: usize,
    },
    /// A page or slab size cannot be represented in the fixed byte ledger.
    AdmissionArithmeticOverflow {
        /// Token position or range involved in the failed calculation.
        positions: usize,
    },
}

/// Returns Nanbeige's logical K/V slot for a physical layer and loop pass.
#[must_use]
pub const fn slot_for(loop_index: usize, layer_index: usize) -> Option<usize> {
    if loop_index < LOOP_COUNT && layer_index < PHYSICAL_LAYER_COUNT {
        Some(layer_index + loop_index * PHYSICAL_LAYER_COUNT)
    } else {
        None
    }
}

/// Identifies the vector whose shape failed validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvVector {
    /// A key vector.
    Key,
    /// A value vector.
    Value,
}

/// Typed refusal from the fixed-shape, append-only cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCacheError {
    /// The requested slot is outside the 44 logical decoder executions.
    InvalidSlot { slot: usize },
    /// A key or value vector did not contain all 8 × 128 bf16 values.
    InvalidVectorLength {
        vector: KvVector,
        expected: usize,
        actual: usize,
    },
    /// A caller tried to overwrite, skip, or reorder a slot position.
    NonAppendPosition {
        slot: usize,
        expected_position: usize,
        received_position: usize,
    },
    /// The engine did not reserve enough positions before entering its hot loop.
    CapacityExceeded {
        slot: usize,
        capacity_positions: usize,
    },
    /// The requested reserve arithmetic overflowed `usize`.
    CapacityOverflow { positions: usize },
    /// The allocator could not reserve the fixed engine buffer before execution.
    AllocationRefused { positions: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KvSlot {
    keys: Vec<u16>,
    values: Vec<u16>,
}

impl KvSlot {
    fn try_with_capacity(element_capacity: usize, positions: usize) -> Result<Self, KvCacheError> {
        let mut keys = Vec::new();
        keys.try_reserve_exact(element_capacity)
            .map_err(|_| KvCacheError::AllocationRefused { positions })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(element_capacity)
            .map_err(|_| KvCacheError::AllocationRefused { positions })?;
        Ok(Self { keys, values })
    }

    fn len_positions(&self) -> usize {
        self.keys.len() / KV_ELEMENTS_PER_POSITION
    }
}

/// Preallocated, append-only K/V storage for every logical decoder execution.
///
/// Construct this at engine build time with [`KvCache::try_with_capacity`].
/// Once provisioned, [`KvCache::append`] only copies into already-reserved
/// vectors and therefore performs no general allocation in the token/layer loop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCache {
    slots: [KvSlot; KV_SLOT_COUNT],
    capacity_positions: usize,
}

impl KvCache {
    /// Preallocates K and V vectors for every slot through `max_positions`.
    pub fn try_with_capacity(max_positions: usize) -> Result<Self, KvCacheError> {
        let element_capacity = max_positions.checked_mul(KV_ELEMENTS_PER_POSITION).ok_or(
            KvCacheError::CapacityOverflow {
                positions: max_positions,
            },
        )?;
        let mut reservation_failed = None;
        let slots =
            array::from_fn(
                |_| match KvSlot::try_with_capacity(element_capacity, max_positions) {
                    Ok(slot) => slot,
                    Err(error) => {
                        reservation_failed = Some(error);
                        KvSlot {
                            keys: Vec::new(),
                            values: Vec::new(),
                        }
                    }
                },
            );
        if let Some(error) = reservation_failed {
            return Err(error);
        }
        Ok(Self {
            slots,
            capacity_positions: max_positions,
        })
    }

    /// The fixed maximum token positions provisioned at engine build time.
    #[must_use]
    pub const fn capacity_positions(&self) -> usize {
        self.capacity_positions
    }

    /// Appends one key/value pair at `position` in the selected logical slot.
    ///
    /// Every slot must receive monotonically increasing positions. This catches
    /// accidental 22-slot reuse, skipped prefill entries, and decode overwrites
    /// before a numerics backend can corrupt a later attention read.
    pub fn append(
        &mut self,
        slot: usize,
        position: usize,
        key: &[u16],
        value: &[u16],
    ) -> Result<(), KvCacheError> {
        if key.len() != KV_ELEMENTS_PER_POSITION {
            return Err(KvCacheError::InvalidVectorLength {
                vector: KvVector::Key,
                expected: KV_ELEMENTS_PER_POSITION,
                actual: key.len(),
            });
        }
        if value.len() != KV_ELEMENTS_PER_POSITION {
            return Err(KvCacheError::InvalidVectorLength {
                vector: KvVector::Value,
                expected: KV_ELEMENTS_PER_POSITION,
                actual: value.len(),
            });
        }
        if slot >= KV_SLOT_COUNT {
            return Err(KvCacheError::InvalidSlot { slot });
        }
        if position >= self.capacity_positions {
            return Err(KvCacheError::CapacityExceeded {
                slot,
                capacity_positions: self.capacity_positions,
            });
        }

        let target = &mut self.slots[slot];
        let expected_position = target.len_positions();
        if position != expected_position {
            return Err(KvCacheError::NonAppendPosition {
                slot,
                expected_position,
                received_position: position,
            });
        }
        target.keys.extend_from_slice(key);
        target.values.extend_from_slice(value);
        Ok(())
    }

    /// Removes all logical positions while retaining the engine-build buffers.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.keys.clear();
            slot.values.clear();
        }
    }

    /// Number of positions written to one logical slot.
    pub fn len_for_slot(&self, slot: usize) -> Result<usize, KvCacheError> {
        self.slots
            .get(slot)
            .map(KvSlot::len_positions)
            .ok_or(KvCacheError::InvalidSlot { slot })
    }

    /// Total `(slot, position)` pairs populated across all 44 slots.
    #[must_use]
    pub fn occupied_slot_positions(&self) -> usize {
        self.slots.iter().map(KvSlot::len_positions).sum()
    }

    /// Whether every logical slot contains exactly `positions` entries.
    #[must_use]
    pub fn all_slots_have_len(&self, positions: usize) -> bool {
        self.slots
            .iter()
            .all(|slot| slot.len_positions() == positions)
    }

    /// Reads the K vector at one logical slot and token position.
    pub fn key_at(&self, slot: usize, position: usize) -> Result<&[u16], KvCacheError> {
        self.vector_at(slot, position, KvVector::Key)
    }

    /// Reads the V vector at one logical slot and token position.
    pub fn value_at(&self, slot: usize, position: usize) -> Result<&[u16], KvCacheError> {
        self.vector_at(slot, position, KvVector::Value)
    }

    fn vector_at(
        &self,
        slot: usize,
        position: usize,
        vector: KvVector,
    ) -> Result<&[u16], KvCacheError> {
        let source = self
            .slots
            .get(slot)
            .ok_or(KvCacheError::InvalidSlot { slot })?;
        let available_positions = source.len_positions();
        if position >= available_positions {
            return Err(KvCacheError::NonAppendPosition {
                slot,
                expected_position: available_positions,
                received_position: position,
            });
        }
        let start = position * KV_ELEMENTS_PER_POSITION;
        let end = start + KV_ELEMENTS_PER_POSITION;
        Ok(match vector {
            KvVector::Key => &source.keys[start..end],
            KvVector::Value => &source.values[start..end],
        })
    }
}
