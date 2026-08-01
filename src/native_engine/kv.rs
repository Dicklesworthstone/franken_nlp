//! Forty-four-slot KV cache for Nanbeige's two-pass decoder loop.
//!
//! The cache deliberately models logical layer executions, not physical weight
//! layers: a token owns one K and one V vector in each of 44 independent slots.
//! Storage uses raw bf16 bit-patterns (`u16`) so numerics profiles decide their
//! own conversion policy without changing addressing or append semantics.

use std::{array, cell::RefCell, rc::Rc};

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
/// Required alignment for every physical K/V vector in a bf16 slab.
pub const KV_SLAB_VECTOR_ALIGNMENT_BYTES: usize = 64;
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
    /// The fixed allocator cannot represent an empty physical page.
    InvalidPageTokens {
        /// Requested positions per logical page.
        page_tokens: usize,
    },
    /// A baseline bf16 slab request did not match the pool's fixed page width.
    UnsupportedPageLength {
        /// Pool page width selected at construction.
        expected: usize,
        /// Requested slab token-range width.
        actual: usize,
    },
    /// The initial aligned slab pool supports only bf16 payloads.
    UnsupportedSlabDtype {
        /// Requested representation.
        dtype: KvSlabDtype,
    },
    /// The pre-reserved physical slab pool cannot satisfy a boundary request.
    PoolExhausted {
        /// Slabs needed atomically for the requested operation.
        requested: usize,
        /// Free slabs available before the operation.
        available: usize,
    },
    /// Preallocating a physical slab or page-table entry failed.
    PoolAllocationRefused {
        /// Number of physical slabs requested at pool construction.
        slab_capacity: usize,
        /// Logical positions held by each physical slab.
        page_tokens: usize,
    },
    /// A token position was not admitted/prepared at its page boundary.
    PositionNotPrepared {
        /// The next token position that may be appended.
        expected: usize,
        /// Position supplied by the caller.
        received: usize,
    },
    /// A sequence fork was requested while a 44-slot append was incomplete.
    ForkDuringAppend {
        /// The partially appended token position.
        position: usize,
    },
    /// A hot-loop append received a vector with the wrong fixed width.
    InvalidSlabVectorLength {
        /// K or V vector that failed validation.
        vector: KvVector,
        /// Required 8 × 128 bf16 elements.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// A vector write would overwrite or skip a prepared slab position.
    NonAppendSlabPosition {
        /// Requested logical slot.
        slot: usize,
        /// Expected position within that slab's token range.
        expected_position: usize,
        /// Position supplied by the caller.
        received_position: usize,
    },
    /// The caller supplied a destination slice of the wrong fixed width.
    InvalidSlabOutputLength {
        /// Required 8 × 128 bf16 elements.
        expected: usize,
        /// Supplied output capacity.
        actual: usize,
    },
    /// A requested logical page exceeds the cache's admitted context cap.
    PageCapacityExceeded {
        /// Fixed cache capacity selected before execution.
        capacity_positions: usize,
    },
    /// A physical slab id was not live in the pool.
    UnknownSlab {
        /// Unrecognized pool-local slab id.
        slab_id: usize,
    },
    /// A physical slab's deterministic reference count would overflow.
    SlabRefcountOverflow {
        /// Pool-local slab id.
        slab_id: usize,
    },
    /// A physical slab release did not have a matching retained reference.
    SlabRefcountUnderflow {
        /// Pool-local slab id.
        slab_id: usize,
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

#[repr(C, align(64))]
#[derive(Clone, Debug, Eq, PartialEq)]
struct AlignedKvVector {
    values: [u16; KV_ELEMENTS_PER_POSITION],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KvSlabId(usize);

#[derive(Debug)]
struct KvPhysicalSlab {
    key: Option<KvSlabKey>,
    vectors: Vec<AlignedKvVector>,
    references: usize,
    sealed: bool,
}

impl KvPhysicalSlab {
    fn try_preallocated(page_tokens: usize) -> Result<Self, KvSlabError> {
        let mut vectors = Vec::new();
        vectors
            .try_reserve_exact(page_tokens)
            .map_err(|_| KvSlabError::PoolAllocationRefused {
                slab_capacity: 1,
                page_tokens,
            })?;
        Ok(Self {
            key: None,
            vectors,
            references: 0,
            sealed: false,
        })
    }

    fn append(
        &mut self,
        slot: usize,
        position: usize,
        position_in_slab: usize,
        vector: KvVector,
        values: &[u16],
    ) -> Result<(), KvSlabError> {
        if values.len() != KV_ELEMENTS_PER_POSITION {
            return Err(KvSlabError::InvalidSlabVectorLength {
                vector,
                expected: KV_ELEMENTS_PER_POSITION,
                actual: values.len(),
            });
        }
        if self.vectors.len() != position_in_slab {
            return Err(KvSlabError::NonAppendSlabPosition {
                slot,
                expected_position: self.vectors.len(),
                received_position: position_in_slab,
            });
        }
        let values: &[u16; KV_ELEMENTS_PER_POSITION] = values
            .try_into()
            .expect("fixed vector length was checked before conversion");
        self.vectors.push(AlignedKvVector { values: *values });
        let _ = position;
        Ok(())
    }

    fn copy_at(
        &self,
        slot: usize,
        position_in_slab: usize,
        vector: KvVector,
        output: &mut [u16],
    ) -> Result<(), KvSlabError> {
        if output.len() != KV_ELEMENTS_PER_POSITION {
            return Err(KvSlabError::InvalidSlabOutputLength {
                expected: KV_ELEMENTS_PER_POSITION,
                actual: output.len(),
            });
        }
        let source = self.vectors.get(position_in_slab).ok_or(
            KvSlabError::NonAppendSlabPosition {
                slot,
                expected_position: self.vectors.len(),
                received_position: position_in_slab,
            },
        )?;
        output.copy_from_slice(&source.values);
        let _ = vector;
        Ok(())
    }
}

/// Pool-wide byte/refcount evidence for a bf16 paged K/V cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvSlabPoolStats {
    /// Physical slabs currently retained by one or more page tables.
    pub live_slab_count: usize,
    /// Preallocated physical slabs not currently assigned to a logical page.
    pub free_slab_count: usize,
    /// Charged bf16 slab bytes, excluding per-sequence page-table metadata.
    pub live_payload_bytes: usize,
    /// Logical slab acquisitions; this must not change during prepared appends.
    pub allocation_events: usize,
}

/// Pre-reserved, 64-byte-aligned physical bf16 K/V slab pool.
///
/// This type deliberately has no public hot-path acquisition method. A
/// [`KvSlabCache`] prepares an admitted token position at a page boundary, then
/// its 44 append calls only copy into vectors whose capacity was allocated here.
#[derive(Debug)]
pub struct KvSlabPool {
    page_tokens: usize,
    slab_payload_bytes: usize,
    slabs: Vec<KvPhysicalSlab>,
    free: Vec<KvSlabId>,
    live_slab_count: usize,
    live_payload_bytes: usize,
    allocation_events: usize,
}

impl KvSlabPool {
    /// Allocates a fixed number of aligned bf16 slabs before token/layer work.
    pub fn try_with_capacity(
        slab_capacity: usize,
        page_tokens: usize,
    ) -> Result<Self, KvSlabError> {
        if page_tokens == 0 {
            return Err(KvSlabError::InvalidPageTokens { page_tokens });
        }
        let slab_payload_bytes = page_tokens
            .checked_mul(KV_ELEMENTS_PER_POSITION)
            .and_then(|elements| elements.checked_mul(size_of::<u16>()))
            .ok_or(KvSlabError::AdmissionArithmeticOverflow {
                positions: page_tokens,
            })?;
        slab_capacity.checked_mul(slab_payload_bytes).ok_or(
            KvSlabError::AdmissionArithmeticOverflow {
                positions: slab_capacity,
            },
        )?;
        let mut slabs = Vec::new();
        slabs
            .try_reserve_exact(slab_capacity)
            .map_err(|_| KvSlabError::PoolAllocationRefused {
                slab_capacity,
                page_tokens,
            })?;
        let mut free = Vec::new();
        free.try_reserve_exact(slab_capacity)
            .map_err(|_| KvSlabError::PoolAllocationRefused {
                slab_capacity,
                page_tokens,
            })?;
        for slab_id in 0..slab_capacity {
            slabs.push(KvPhysicalSlab::try_preallocated(page_tokens)?);
            free.push(KvSlabId(slab_id));
        }
        Ok(Self {
            page_tokens,
            slab_payload_bytes,
            slabs,
            free,
            live_slab_count: 0,
            live_payload_bytes: 0,
            allocation_events: 0,
        })
    }

    /// Fixed logical positions available in every physical slab.
    #[must_use]
    pub const fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    /// Returns current retained slab accounting without allocating.
    #[must_use]
    pub fn stats(&self) -> KvSlabPoolStats {
        KvSlabPoolStats {
            live_slab_count: self.live_slab_count,
            free_slab_count: self.free.len(),
            live_payload_bytes: self.live_payload_bytes,
            allocation_events: self.allocation_events,
        }
    }

    fn require_free(&self, requested: usize) -> Result<(), KvSlabError> {
        if self.free.len() < requested {
            return Err(KvSlabError::PoolExhausted {
                requested,
                available: self.free.len(),
            });
        }
        Ok(())
    }

    fn slab(&self, slab_id: KvSlabId) -> Result<&KvPhysicalSlab, KvSlabError> {
        self.slabs.get(slab_id.0).ok_or(KvSlabError::UnknownSlab {
            slab_id: slab_id.0,
        })
    }

    fn slab_mut(&mut self, slab_id: KvSlabId) -> Result<&mut KvPhysicalSlab, KvSlabError> {
        self.slabs
            .get_mut(slab_id.0)
            .ok_or(KvSlabError::UnknownSlab {
                slab_id: slab_id.0,
            })
    }

    fn acquire(&mut self, key: KvSlabKey) -> Result<KvSlabId, KvSlabError> {
        if key.dtype != KvSlabDtype::Bf16 {
            return Err(KvSlabError::UnsupportedSlabDtype { dtype: key.dtype });
        }
        if key.token_count != self.page_tokens {
            return Err(KvSlabError::UnsupportedPageLength {
                expected: self.page_tokens,
                actual: key.token_count,
            });
        }
        if key.loop_layer_count != 1 {
            return Err(KvSlabError::InvalidSlabKey {
                token_count: key.token_count,
                loop_layer_start: key.loop_layer_start,
                loop_layer_count: key.loop_layer_count,
            });
        }
        self.require_free(1)?;
        let live_payload_bytes = self
            .live_payload_bytes
            .checked_add(self.slab_payload_bytes)
            .ok_or(KvSlabError::AdmissionArithmeticOverflow {
                positions: self.live_slab_count,
            })?;
        let allocation_events = self
            .allocation_events
            .checked_add(1)
            .ok_or(KvSlabError::AdmissionArithmeticOverflow {
                positions: self.allocation_events,
            })?;
        let slab_id = self
            .free
            .pop()
            .expect("require_free guarantees one pre-reserved slab id");
        let slab = self.slab_mut(slab_id)?;
        debug_assert!(slab.key.is_none());
        debug_assert!(slab.vectors.is_empty());
        debug_assert_eq!(slab.references, 0);
        slab.key = Some(key);
        slab.references = 1;
        slab.sealed = false;
        self.live_slab_count += 1;
        self.live_payload_bytes = live_payload_bytes;
        self.allocation_events = allocation_events;
        Ok(slab_id)
    }

    fn retain_for_fork(&mut self, slab_id: KvSlabId) -> Result<(), KvSlabError> {
        let slab = self.slab_mut(slab_id)?;
        slab.references = slab
            .references
            .checked_add(1)
            .ok_or(KvSlabError::SlabRefcountOverflow {
                slab_id: slab_id.0,
            })?;
        slab.sealed = true;
        Ok(())
    }

    fn release(&mut self, slab_id: KvSlabId) -> Result<(), KvSlabError> {
        let slab = self.slab_mut(slab_id)?;
        if slab.references == 0 {
            return Err(KvSlabError::SlabRefcountUnderflow {
                slab_id: slab_id.0,
            });
        }
        slab.references -= 1;
        if slab.references == 0 {
            slab.key = None;
            slab.vectors.clear();
            slab.sealed = false;
            self.live_slab_count -= 1;
            self.live_payload_bytes -= self.slab_payload_bytes;
            self.free.push(slab_id);
        }
        Ok(())
    }

    fn is_writable(&self, slab_id: KvSlabId) -> Result<bool, KvSlabError> {
        let slab = self.slab(slab_id)?;
        Ok(slab.references == 1 && !slab.sealed)
    }

    fn key(&self, slab_id: KvSlabId) -> Result<KvSlabKey, KvSlabError> {
        self.slab(slab_id)?
            .key
            .ok_or(KvSlabError::UnknownSlab {
                slab_id: slab_id.0,
            })
    }

    fn copy_contents(
        &mut self,
        source_id: KvSlabId,
        target_id: KvSlabId,
    ) -> Result<(), KvSlabError> {
        if source_id == target_id {
            return Ok(());
        }
        let source_index = source_id.0;
        let target_index = target_id.0;
        if source_index >= self.slabs.len() {
            return Err(KvSlabError::UnknownSlab {
                slab_id: source_index,
            });
        }
        if target_index >= self.slabs.len() {
            return Err(KvSlabError::UnknownSlab {
                slab_id: target_index,
            });
        }
        let (source, target) = if source_index < target_index {
            let (before_target, from_target) = self.slabs.split_at_mut(target_index);
            (&before_target[source_index], &mut from_target[0])
        } else {
            let (before_source, from_source) = self.slabs.split_at_mut(source_index);
            (&from_source[0], &mut before_source[target_index])
        };
        target.vectors.extend_from_slice(&source.vectors);
        Ok(())
    }

    fn append_vector(
        &mut self,
        slab_id: KvSlabId,
        slot: usize,
        position: usize,
        position_in_slab: usize,
        vector: KvVector,
        values: &[u16],
    ) -> Result<(), KvSlabError> {
        let slab = self.slab_mut(slab_id)?;
        if slab.references != 1 || slab.sealed {
            return Err(KvSlabError::PositionNotPrepared {
                expected: position,
                received: position,
            });
        }
        slab.append(slot, position, position_in_slab, vector, values)
    }

    fn copy_vector(
        &self,
        slab_id: KvSlabId,
        slot: usize,
        position_in_slab: usize,
        vector: KvVector,
        output: &mut [u16],
    ) -> Result<(), KvSlabError> {
        self.slab(slab_id)?
            .copy_at(slot, position_in_slab, vector, output)
    }

    fn refcount(&self, slab_id: KvSlabId) -> Result<usize, KvSlabError> {
        Ok(self.slab(slab_id)?.references)
    }

    fn vector_alignment_offset(
        &self,
        slab_id: KvSlabId,
        position_in_slab: usize,
    ) -> Result<usize, KvSlabError> {
        let vector = self
            .slab(slab_id)?
            .vectors
            .get(position_in_slab)
            .ok_or(KvSlabError::UnknownSlab {
                slab_id: slab_id.0,
            })?;
        Ok((vector.values.as_ptr() as usize) % KV_SLAB_VECTOR_ALIGNMENT_BYTES)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KvSlabPage {
    token_start: usize,
    slabs: [[KvSlabId; 2]; KV_SLOT_COUNT],
}

const fn slab_vector_index(vector: KvVector) -> usize {
    match vector {
        KvVector::Key => 0,
        KvVector::Value => 1,
    }
}

/// Paged 44-slot bf16 K/V cache backed by independently refcounted slabs.
///
/// Call [`KvSlabCache::prepare_append`] at an admission/page boundary before
/// the 44 per-layer appends for one token. [`KvSlabCache::append`] is then a
/// copy-only hot path: page-table growth, slab acquisition, and COW all happen
/// before it is entered.
#[derive(Debug)]
pub struct KvSlabCache {
    pool: Rc<RefCell<KvSlabPool>>,
    pages: Vec<KvSlabPage>,
    max_positions: usize,
    page_tokens: usize,
    completed_positions: usize,
    prepared_position: Option<usize>,
    prepared_slot_writes: usize,
}

impl KvSlabCache {
    /// Builds a cache using the baseline 16-token logical page size.
    pub fn try_with_capacity(
        max_positions: usize,
        max_live_slabs: usize,
    ) -> Result<Self, KvSlabError> {
        Self::try_with_page_tokens(max_positions, KV_LOGICAL_PAGE_TOKENS, max_live_slabs)
    }

    /// Builds a cache with an explicit, measured logical page width.
    pub fn try_with_page_tokens(
        max_positions: usize,
        page_tokens: usize,
        max_live_slabs: usize,
    ) -> Result<Self, KvSlabError> {
        if page_tokens == 0 {
            return Err(KvSlabError::InvalidPageTokens { page_tokens });
        }
        let max_pages = if max_positions == 0 {
            0
        } else {
            1 + (max_positions - 1) / page_tokens
        };
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(max_pages)
            .map_err(|_| KvSlabError::PoolAllocationRefused {
                slab_capacity: max_live_slabs,
                page_tokens,
            })?;
        Ok(Self {
            pool: Rc::new(RefCell::new(KvSlabPool::try_with_capacity(
                max_live_slabs,
                page_tokens,
            )?)),
            pages,
            max_positions,
            page_tokens,
            completed_positions: 0,
            prepared_position: None,
            prepared_slot_writes: 0,
        })
    }

    /// Logical positions fully populated in all 44 K/V slots.
    #[must_use]
    pub const fn len_positions(&self) -> usize {
        self.completed_positions
    }

    /// Fixed maximum positions admitted when this cache was constructed.
    #[must_use]
    pub const fn capacity_positions(&self) -> usize {
        self.max_positions
    }

    /// Retained page-table metadata bytes for this sequence only.
    #[must_use]
    pub fn page_table_bytes(&self) -> usize {
        self.pages.len() * size_of::<KvSlabPage>()
    }

    /// Pool-wide slab retention/refcount evidence.
    #[must_use]
    pub fn pool_stats(&self) -> KvSlabPoolStats {
        self.pool.borrow().stats()
    }

    /// Prepares a token position at a page boundary before its 44 K/V writes.
    ///
    /// A forked/sealed tail is copied here, never by [`Self::append`].
    pub fn prepare_append(&mut self, position: usize) -> Result<(), KvSlabError> {
        if position >= self.max_positions {
            return Err(KvSlabError::PageCapacityExceeded {
                capacity_positions: self.max_positions,
            });
        }
        if let Some(active) = self.prepared_position {
            if active == position {
                return Ok(());
            }
            return Err(KvSlabError::PositionNotPrepared {
                expected: active,
                received: position,
            });
        }
        if position != self.completed_positions {
            return Err(KvSlabError::PositionNotPrepared {
                expected: self.completed_positions,
                received: position,
            });
        }
        let page_index = position / self.page_tokens;
        if page_index == self.pages.len() {
            self.allocate_page(page_index)?;
        } else {
            self.copy_page_tail_if_shared(page_index)?;
        }
        self.prepared_position = Some(position);
        self.prepared_slot_writes = 0;
        Ok(())
    }

    /// Appends one K/V pair after [`Self::prepare_append`] has prepared it.
    ///
    /// This method performs no page-table reserve, slab acquisition, or COW.
    pub fn append(
        &mut self,
        slot: usize,
        position: usize,
        key: &[u16],
        value: &[u16],
    ) -> Result<(), KvSlabError> {
        if slot >= KV_SLOT_COUNT {
            return Err(KvSlabError::NonAppendSlabPosition {
                slot,
                expected_position: 0,
                received_position: position,
            });
        }
        if key.len() != KV_ELEMENTS_PER_POSITION {
            return Err(KvSlabError::InvalidSlabVectorLength {
                vector: KvVector::Key,
                expected: KV_ELEMENTS_PER_POSITION,
                actual: key.len(),
            });
        }
        if value.len() != KV_ELEMENTS_PER_POSITION {
            return Err(KvSlabError::InvalidSlabVectorLength {
                vector: KvVector::Value,
                expected: KV_ELEMENTS_PER_POSITION,
                actual: value.len(),
            });
        }
        if self.prepared_position != Some(position) {
            return Err(KvSlabError::PositionNotPrepared {
                expected: self.prepared_position.unwrap_or(self.completed_positions),
                received: position,
            });
        }
        let page_index = position / self.page_tokens;
        let page = self.pages.get(page_index).ok_or(KvSlabError::PageCapacityExceeded {
            capacity_positions: self.max_positions,
        })?;
        let position_in_slab = position - page.token_start;
        let key_slab = page.slabs[slot][slab_vector_index(KvVector::Key)];
        let value_slab = page.slabs[slot][slab_vector_index(KvVector::Value)];
        {
            let mut pool = self.pool.borrow_mut();
            pool.append_vector(
                key_slab,
                slot,
                position,
                position_in_slab,
                KvVector::Key,
                key,
            )?;
            pool.append_vector(
                value_slab,
                slot,
                position,
                position_in_slab,
                KvVector::Value,
                value,
            )?;
        }
        self.prepared_slot_writes += 1;
        if self.prepared_slot_writes == KV_SLOT_COUNT {
            self.completed_positions += 1;
            self.prepared_position = None;
            self.prepared_slot_writes = 0;
        }
        Ok(())
    }

    /// Copies a completed K vector into a caller-owned fixed-width buffer.
    pub fn copy_key_at(
        &self,
        slot: usize,
        position: usize,
        output: &mut [u16],
    ) -> Result<(), KvSlabError> {
        self.copy_vector_at(slot, position, KvVector::Key, output)
    }

    /// Copies a completed V vector into a caller-owned fixed-width buffer.
    pub fn copy_value_at(
        &self,
        slot: usize,
        position: usize,
        output: &mut [u16],
    ) -> Result<(), KvSlabError> {
        self.copy_vector_at(slot, position, KvVector::Value, output)
    }

    /// Forks a completed prefix, sealing the shared slabs deterministically.
    pub fn try_fork(&self) -> Result<Self, KvSlabError> {
        if let Some(position) = self.prepared_position {
            return Err(KvSlabError::ForkDuringAppend { position });
        }
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(self.pages.len())
            .map_err(|_| KvSlabError::PoolAllocationRefused {
                slab_capacity: self.pages.len() * KV_SLABS_PER_LOGICAL_PAGE,
                page_tokens: self.page_tokens,
            })?;
        pages.extend_from_slice(&self.pages);
        let mut pool = self.pool.borrow_mut();
        for page in &pages {
            for slabs in &page.slabs {
                for slab_id in slabs {
                    if pool.refcount(*slab_id)? == usize::MAX {
                        return Err(KvSlabError::SlabRefcountOverflow {
                            slab_id: slab_id.0,
                        });
                    }
                }
            }
        }
        for page in &pages {
            for slabs in &page.slabs {
                for slab_id in slabs {
                    pool.retain_for_fork(*slab_id)?;
                }
            }
        }
        drop(pool);
        Ok(Self {
            pool: Rc::clone(&self.pool),
            pages,
            max_positions: self.max_positions,
            page_tokens: self.page_tokens,
            completed_positions: self.completed_positions,
            prepared_position: None,
            prepared_slot_writes: 0,
        })
    }

    /// Returns the deterministic reference count for one completed logical K/V vector.
    pub fn refcount_at(
        &self,
        slot: usize,
        position: usize,
        vector: KvVector,
    ) -> Result<usize, KvSlabError> {
        let slab_id = self.slab_id_at(slot, position, vector)?;
        self.pool.borrow().refcount(slab_id)
    }

    /// Returns a completed vector's byte-address modulo the required alignment.
    ///
    /// A result of zero is the 64-byte alignment invariant consumed by the
    /// K/V scan kernels; this method exposes an auditable value rather than a
    /// raw pointer or mutable slab storage.
    pub fn vector_alignment_offset_at(
        &self,
        slot: usize,
        position: usize,
        vector: KvVector,
    ) -> Result<usize, KvSlabError> {
        let slab_id = self.slab_id_at(slot, position, vector)?;
        let page = &self.pages[position / self.page_tokens];
        self.pool
            .borrow()
            .vector_alignment_offset(slab_id, position - page.token_start)
    }

    fn allocate_page(&mut self, page_index: usize) -> Result<(), KvSlabError> {
        if page_index != self.pages.len() {
            return Err(KvSlabError::PositionNotPrepared {
                expected: self.pages.len() * self.page_tokens,
                received: page_index * self.page_tokens,
            });
        }
        let token_start = page_index
            .checked_mul(self.page_tokens)
            .ok_or(KvSlabError::AdmissionArithmeticOverflow {
                positions: page_index,
            })?;
        let mut slabs = [[KvSlabId(usize::MAX); 2]; KV_SLOT_COUNT];
        let mut pool = self.pool.borrow_mut();
        pool.require_free(KV_SLABS_PER_LOGICAL_PAGE)?;
        for slot in 0..KV_SLOT_COUNT {
            for vector in [KvVector::Key, KvVector::Value] {
                let key = KvSlabKey::new(
                    token_start,
                    self.page_tokens,
                    slot,
                    1,
                    vector,
                    KvSlabDtype::Bf16,
                )?;
                slabs[slot][slab_vector_index(vector)] = pool.acquire(key)?;
            }
        }
        drop(pool);
        self.pages.push(KvSlabPage { token_start, slabs });
        Ok(())
    }

    fn copy_page_tail_if_shared(&mut self, page_index: usize) -> Result<(), KvSlabError> {
        let page = self.pages.get_mut(page_index).ok_or(KvSlabError::PageCapacityExceeded {
            capacity_positions: self.max_positions,
        })?;
        let old_slabs = page.slabs;
        let mut pool = self.pool.borrow_mut();
        let mut needs_copy = false;
        for slabs in &old_slabs {
            for slab_id in slabs {
                if !pool.is_writable(*slab_id)? {
                    needs_copy = true;
                }
            }
        }
        if !needs_copy {
            return Ok(());
        }
        pool.require_free(KV_SLABS_PER_LOGICAL_PAGE)?;
        let mut replacement = old_slabs;
        for slot in 0..KV_SLOT_COUNT {
            for vector in [KvVector::Key, KvVector::Value] {
                let vector_index = slab_vector_index(vector);
                let key = pool.key(old_slabs[slot][vector_index])?;
                replacement[slot][vector_index] = pool.acquire(key)?;
            }
        }
        for slot in 0..KV_SLOT_COUNT {
            for vector in [KvVector::Key, KvVector::Value] {
                let vector_index = slab_vector_index(vector);
                pool.copy_contents(
                    old_slabs[slot][vector_index],
                    replacement[slot][vector_index],
                )?;
            }
        }
        for slabs in &old_slabs {
            for slab_id in slabs {
                pool.release(*slab_id)?;
            }
        }
        page.slabs = replacement;
        Ok(())
    }

    fn copy_vector_at(
        &self,
        slot: usize,
        position: usize,
        vector: KvVector,
        output: &mut [u16],
    ) -> Result<(), KvSlabError> {
        if slot >= KV_SLOT_COUNT || position >= self.completed_positions {
            return Err(KvSlabError::NonAppendSlabPosition {
                slot,
                expected_position: self.completed_positions,
                received_position: position,
            });
        }
        let slab_id = self.slab_id_at(slot, position, vector)?;
        let page = &self.pages[position / self.page_tokens];
        self.pool.borrow().copy_vector(
            slab_id,
            slot,
            position - page.token_start,
            vector,
            output,
        )
    }

    fn slab_id_at(
        &self,
        slot: usize,
        position: usize,
        vector: KvVector,
    ) -> Result<KvSlabId, KvSlabError> {
        if slot >= KV_SLOT_COUNT || position >= self.completed_positions {
            return Err(KvSlabError::NonAppendSlabPosition {
                slot,
                expected_position: self.completed_positions,
                received_position: position,
            });
        }
        self.pages
            .get(position / self.page_tokens)
            .map(|page| page.slabs[slot][slab_vector_index(vector)])
            .ok_or(KvSlabError::PageCapacityExceeded {
                capacity_positions: self.max_positions,
            })
    }
}

impl Drop for KvSlabCache {
    fn drop(&mut self) {
        let mut pool = self.pool.borrow_mut();
        for page in &self.pages {
            for slabs in &page.slabs {
                for slab_id in slabs {
                    pool.release(*slab_id)
                        .expect("every page-table slab retains one pool reference");
                }
            }
        }
    }
}
