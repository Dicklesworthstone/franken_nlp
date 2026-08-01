#![deny(unsafe_code)]

use franken_nlp::native_engine::kv::{
    KV_BF16_LOGICAL_PAGE_BYTES, KV_BF16_SLAB_BYTES, KV_BYTES_PER_TOKEN,
    KV_INT8_F16_SCALE_BYTES_PER_TOKEN, KV_INT8_F32_SCALE_BYTES_PER_TOKEN,
    KV_INT8_PAYLOAD_BYTES_PER_TOKEN, KV_LOGICAL_PAGE_TOKENS, KV_SLOT_COUNT,
    KV_SLABS_PER_LOGICAL_PAGE, KvSlabDtype, KvSlabError, KvSlabKey, KvVector,
};

#[test]
fn baseline_slab_geometry_is_the_44_slot_byte_certificate() {
    assert_eq!(KV_SLOT_COUNT, 44);
    assert_eq!(KV_BYTES_PER_TOKEN, 180_224);
    assert_eq!(KV_LOGICAL_PAGE_TOKENS, 16);
    assert_eq!(KV_SLABS_PER_LOGICAL_PAGE, 88);
    assert_eq!(KV_BF16_SLAB_BYTES, 32 * 1024);
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
