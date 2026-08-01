#![deny(unsafe_code)]

use franken_nlp::native_engine::kv::{
    KV_BF16_LOGICAL_PAGE_BYTES, KV_BF16_SLAB_BYTES, KV_BYTES_PER_TOKEN,
    KV_INT8_F16_SCALE_BYTES_PER_TOKEN, KV_INT8_F32_SCALE_BYTES_PER_TOKEN,
    KV_INT8_PAYLOAD_BYTES_PER_TOKEN, KV_LOGICAL_PAGE_TOKENS, KV_SLOT_COUNT,
    KV_SLABS_PER_LOGICAL_PAGE, KV_SLAB_VECTOR_ALIGNMENT_BYTES, KvSlabAdmission, KvSlabCache,
    KvSlabDtype, KvSlabError, KvSlabKey, KvSlabPoolStats, KvVector,
};

fn append_prepared_position(cache: &mut KvSlabCache, position: usize, marker: u16) {
    cache
        .prepare_append(position)
        .expect("the position is admitted at its page boundary");
    for slot in 0..KV_SLOT_COUNT {
        let key = [marker + slot as u16; 1_024];
        let value = [marker + 1_000 + slot as u16; 1_024];
        cache
            .append(slot, position, &key, &value)
            .expect("prepared hot-loop K/V append must not allocate or fail");
    }
}

fn assert_pool_ledger_reconciles(stats: KvSlabPoolStats) {
    assert_eq!(
        stats.reserved_payload_bytes,
        stats.slab_capacity * KV_BF16_SLAB_BYTES,
        "pre-reserved pool bytes are the exact sum of physical slabs"
    );
    assert_eq!(
        stats.live_payload_bytes,
        stats.live_slab_count * KV_BF16_SLAB_BYTES,
        "live ledger bytes are the exact sum of retained physical slabs"
    );
    assert_eq!(
        stats.free_slab_count + stats.live_slab_count,
        stats.slab_capacity,
        "every pre-reserved slab is either free or reachable from a page table"
    );
    let references = stats
        .retained_reference_count
        .expect("bounded lifecycle test must retain a representable refcount total");
    assert!(
        references >= stats.live_slab_count,
        "each live slab retains at least one deterministic page-table reference"
    );
}

#[test]
fn baseline_slab_geometry_is_the_44_slot_byte_certificate() {
    assert_eq!(KV_SLOT_COUNT, 44);
    assert_eq!(KV_BYTES_PER_TOKEN, 180_224);
    assert_eq!(KV_LOGICAL_PAGE_TOKENS, 16);
    assert_eq!(KV_SLABS_PER_LOGICAL_PAGE, 88);
    assert_eq!(KV_BF16_SLAB_BYTES, 32 * 1024);
    assert_eq!(KV_SLAB_VECTOR_ALIGNMENT_BYTES, 64);
    assert_eq!(KV_BF16_LOGICAL_PAGE_BYTES, 2_883_584);
    assert_eq!(KV_BF16_LOGICAL_PAGE_BYTES * 4, 11 * 1024 * 1024);

    assert_eq!(KV_INT8_PAYLOAD_BYTES_PER_TOKEN, 88 * 1024);
    assert_eq!(KV_INT8_F32_SCALE_BYTES_PER_TOKEN, 2_816);
    assert_eq!(KV_INT8_F16_SCALE_BYTES_PER_TOKEN, 1_408);
    assert_eq!(KvSlabDtype::Bf16.bytes_per_token(), 180_224);
    assert_eq!(KvSlabDtype::Int8F32Scale.bytes_per_token(), 92_928);
    assert_eq!(KvSlabDtype::Int8F16Scale.bytes_per_token(), 91_520);
}

#[test]
fn typed_admission_separates_token_charge_from_page_reservation() {
    let one = KvSlabAdmission::try_for_positions(1, KV_LOGICAL_PAGE_TOKENS)
        .expect("one token has a checked bf16 admission price");
    assert_eq!(one.logical_bf16_bytes(), KV_BYTES_PER_TOKEN);
    assert_eq!(one.reserved_page_count(), 1);
    assert_eq!(one.reserved_slab_count(), KV_SLABS_PER_LOGICAL_PAGE);
    assert_eq!(one.reserved_bf16_bytes(), KV_BF16_LOGICAL_PAGE_BYTES);

    let full_page = KvSlabAdmission::try_for_positions(16, KV_LOGICAL_PAGE_TOKENS)
        .expect("one logical page has no rounding slack");
    assert_eq!(full_page.logical_bf16_bytes(), KV_BF16_LOGICAL_PAGE_BYTES);
    assert_eq!(full_page.reserved_bf16_bytes(), KV_BF16_LOGICAL_PAGE_BYTES);

    let seventeen = KvSlabAdmission::try_for_positions(17, KV_LOGICAL_PAGE_TOKENS)
        .expect("a page boundary rounds upward deterministically");
    assert_eq!(seventeen.positions(), 17);
    assert_eq!(seventeen.page_tokens(), KV_LOGICAL_PAGE_TOKENS);
    assert_eq!(seventeen.logical_bf16_bytes(), 17 * KV_BYTES_PER_TOKEN);
    assert_eq!(seventeen.reserved_page_count(), 2);
    assert_eq!(
        seventeen.reserved_slab_count(),
        2 * KV_SLABS_PER_LOGICAL_PAGE
    );
    assert_eq!(
        seventeen.reserved_bf16_bytes(),
        2 * KV_BF16_LOGICAL_PAGE_BYTES
    );

    assert!(matches!(
        KvSlabAdmission::try_for_positions(1, 0),
        Err(KvSlabError::InvalidPageTokens { page_tokens: 0 })
    ));
    assert!(matches!(
        KvSlabAdmission::try_for_positions(usize::MAX, KV_LOGICAL_PAGE_TOKENS),
        Err(KvSlabError::AdmissionArithmeticOverflow { .. })
    ));
}

#[test]
fn typed_admission_constructs_an_exactly_priced_pre_reserved_pool() {
    let admission = KvSlabAdmission::try_for_positions(17, KV_LOGICAL_PAGE_TOKENS)
        .expect("the two-page context has a checked reservation");
    let cache = KvSlabCache::try_with_bf16_admission(admission)
        .expect("cache construction consumes only the typed reservation");
    let stats = cache.pool_stats();
    assert_eq!(cache.capacity_positions(), 17);
    assert_eq!(stats.slab_capacity, admission.reserved_slab_count());
    assert_eq!(stats.reserved_payload_bytes, admission.reserved_bf16_bytes());
    assert_eq!(stats.live_slab_count, 0);
    assert_eq!(stats.free_slab_count, admission.reserved_slab_count());
    assert_eq!(stats.retained_reference_count, Some(0));
    assert_pool_ledger_reconciles(stats);
}

#[test]
fn slab_key_prices_one_vector_in_one_logical_layer_group() {
    let key = KvSlabKey::new(32, 16, 43, 1, KvVector::Key, KvSlabDtype::Bf16)
        .expect("the final logical slot is addressable");
    assert_eq!(key.payload_bytes(), Ok(KV_BF16_SLAB_BYTES));

    let int8 = KvSlabKey::new(32, 16, 0, 1, KvVector::Value, KvSlabDtype::Int8F32Scale)
        .expect("one int8 vector slab is addressable");
    assert_eq!(int8.payload_bytes(), Ok(16_896));

    assert!(matches!(
        KvSlabKey::new(0, 16, KV_SLOT_COUNT, 1, KvVector::Key, KvSlabDtype::Bf16),
        Err(KvSlabError::InvalidSlabKey { .. })
    ));
    assert!(matches!(
        KvSlabKey::new(usize::MAX, 1, 0, 1, KvVector::Key, KvSlabDtype::Bf16),
        Err(KvSlabError::AdmissionArithmeticOverflow { .. })
    ));
}

#[test]
fn prepared_append_uses_only_preallocated_aligned_slabs() {
    let mut cache = KvSlabCache::try_with_capacity(32, KV_SLABS_PER_LOGICAL_PAGE)
        .expect("one logical page worth of slabs pre-reserves successfully");
    cache
        .prepare_append(0)
        .expect("page admission happens before the hot loop");
    let before_append = cache.pool_stats();
    for slot in 0..KV_SLOT_COUNT {
        let key = [slot as u16; 1_024];
        let value = [1_000 + slot as u16; 1_024];
        cache
            .append(slot, 0, &key, &value)
            .expect("prepared append writes only existing slabs");
    }
    let after_append = cache.pool_stats();

    assert_eq!(cache.len_positions(), 1);
    assert_eq!(before_append.allocation_events, after_append.allocation_events);
    assert_eq!(after_append.live_slab_count, KV_SLABS_PER_LOGICAL_PAGE);
    assert_eq!(after_append.live_payload_bytes, KV_BF16_LOGICAL_PAGE_BYTES);
    assert_eq!(
        after_append.retained_reference_count,
        Some(KV_SLABS_PER_LOGICAL_PAGE)
    );
    assert_pool_ledger_reconciles(after_append);

    let mut key = [0_u16; 1_024];
    let mut value = [0_u16; 1_024];
    cache
        .copy_key_at(43, 0, &mut key)
        .expect("completed K data is addressable by logical slot and position");
    cache
        .copy_value_at(43, 0, &mut value)
        .expect("completed V data is addressable by logical slot and position");
    assert_eq!(key, [43; 1_024]);
    assert_eq!(value, [1_043; 1_024]);
    assert_eq!(
        cache.vector_alignment_offset_at(43, 0, KvVector::Key),
        Ok(0),
        "safe aligned-vector storage keeps every K address on a 64-byte boundary"
    );
}

#[test]
fn page_admission_refuses_atomically_before_any_slab_is_assigned() {
    let mut cache = KvSlabCache::try_with_capacity(16, KV_SLABS_PER_LOGICAL_PAGE - 1)
        .expect("the intentionally undersized pool itself remains constructible");
    assert!(matches!(
        cache.prepare_append(0),
        Err(KvSlabError::PoolExhausted {
            requested: KV_SLABS_PER_LOGICAL_PAGE,
            available,
        }) if available == KV_SLABS_PER_LOGICAL_PAGE - 1
    ));
    assert_eq!(cache.pool_stats().live_slab_count, 0);
    assert_eq!(cache.pool_stats().live_payload_bytes, 0);
    assert_eq!(cache.pool_stats().allocation_events, 0);
    assert_eq!(cache.pool_stats().retained_reference_count, Some(0));
    assert_pool_ledger_reconciles(cache.pool_stats());
}

#[test]
fn forked_tail_cow_releases_only_the_cancelled_fork_slabs() {
    let mut parent = KvSlabCache::try_with_capacity(32, KV_SLABS_PER_LOGICAL_PAGE * 2)
        .expect("the pool reserves a parent page plus one COW tail");
    for position in 0..3 {
        append_prepared_position(&mut parent, position, 10 + position as u16);
    }

    let mut fork = parent
        .try_fork()
        .expect("a completed prefix can retain sealed slab references");
    assert_eq!(parent.refcount_at(0, 0, KvVector::Key), Ok(2));
    assert_eq!(fork.pool_stats().live_slab_count, KV_SLABS_PER_LOGICAL_PAGE);
    assert_eq!(
        fork.pool_stats().retained_reference_count,
        Some(KV_SLABS_PER_LOGICAL_PAGE * 2)
    );
    assert_pool_ledger_reconciles(fork.pool_stats());

    fork.prepare_append(3)
        .expect("fork-tail COW is prepared outside the append loop");
    assert_eq!(fork.pool_stats().live_slab_count, KV_SLABS_PER_LOGICAL_PAGE * 2);
    assert_eq!(
        fork.pool_stats().retained_reference_count,
        Some(KV_SLABS_PER_LOGICAL_PAGE * 2)
    );
    assert_pool_ledger_reconciles(fork.pool_stats());
    append_prepared_position(&mut fork, 3, 99);

    let mut parent_key = [0_u16; 1_024];
    parent
        .copy_key_at(0, 2, &mut parent_key)
        .expect("the parent retains its sealed prefix");
    assert_eq!(parent_key, [12; 1_024]);
    assert_eq!(fork.refcount_at(0, 3, KvVector::Key), Ok(1));

    drop(fork);
    let retained = parent.pool_stats();
    assert_eq!(retained.live_slab_count, KV_SLABS_PER_LOGICAL_PAGE);
    assert_eq!(retained.live_payload_bytes, KV_BF16_LOGICAL_PAGE_BYTES);
    assert_eq!(
        retained.retained_reference_count,
        Some(KV_SLABS_PER_LOGICAL_PAGE)
    );
    assert_pool_ledger_reconciles(retained);
    assert_eq!(parent.refcount_at(0, 0, KvVector::Key), Ok(1));
}

#[test]
fn parent_append_after_fork_cows_the_sealed_page_before_writing() {
    let mut parent = KvSlabCache::try_with_capacity(32, KV_SLABS_PER_LOGICAL_PAGE * 2)
        .expect("the pool reserves a shared page plus one parent COW page");
    for position in 0..3 {
        append_prepared_position(&mut parent, position, 40 + position as u16);
    }

    let fork = parent
        .try_fork()
        .expect("forking seals the completed parent page");
    assert_eq!(parent.refcount_at(7, 2, KvVector::Value), Ok(2));

    append_prepared_position(&mut parent, 3, 90);
    let after_parent_cow = parent.pool_stats();
    assert_eq!(
        after_parent_cow.live_slab_count,
        KV_SLABS_PER_LOGICAL_PAGE * 2
    );
    assert_eq!(
        after_parent_cow.retained_reference_count,
        Some(KV_SLABS_PER_LOGICAL_PAGE * 2)
    );
    assert_pool_ledger_reconciles(after_parent_cow);
    assert_eq!(parent.refcount_at(7, 2, KvVector::Value), Ok(1));
    assert_eq!(fork.refcount_at(7, 2, KvVector::Value), Ok(1));

    let mut parent_value = [0_u16; 1_024];
    let mut fork_value = [0_u16; 1_024];
    parent
        .copy_value_at(7, 2, &mut parent_value)
        .expect("parent COW retains the shared prefix values");
    fork.copy_value_at(7, 2, &mut fork_value)
        .expect("fork retains its own immutable prefix values");
    assert_eq!(parent_value, [1_049; 1_024]);
    assert_eq!(fork_value, [1_049; 1_024]);

    drop(fork);
    let parent_only = parent.pool_stats();
    assert_eq!(parent_only.live_slab_count, KV_SLABS_PER_LOGICAL_PAGE);
    assert_eq!(
        parent_only.retained_reference_count,
        Some(KV_SLABS_PER_LOGICAL_PAGE)
    );
    assert_pool_ledger_reconciles(parent_only);
}

#[test]
fn fork_sealed_page_refuses_direct_append_until_cow_is_prepared() {
    let mut parent = KvSlabCache::try_with_capacity(16, KV_SLABS_PER_LOGICAL_PAGE * 2)
        .expect("the pool has room for one sealed page and one COW page");
    append_prepared_position(&mut parent, 0, 12);
    let fork = parent
        .try_fork()
        .expect("forking seals the completed page before either branch can write it");

    let before = parent.pool_stats();
    let key = [7_u16; 1_024];
    let value = [8_u16; 1_024];
    assert_eq!(
        parent.append(0, 1, &key, &value),
        Err(KvSlabError::PositionNotPrepared {
            expected: 1,
            received: 1,
        })
    );
    let after = parent.pool_stats();
    assert_eq!(after, before);
    assert_pool_ledger_reconciles(after);

    let mut fork_key = [0_u16; 1_024];
    fork.copy_key_at(0, 0, &mut fork_key)
        .expect("failed direct parent write cannot alter the fork's sealed prefix");
    assert_eq!(fork_key, [12; 1_024]);
}
