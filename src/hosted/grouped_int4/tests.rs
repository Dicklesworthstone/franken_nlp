use super::*;

fn limits(width: usize) -> Int4LoadLimits {
    let need = Int4ModelLoadRequirement::for_group_width(width).unwrap();
    Int4LoadLimits { model: Int4ModelLoadBudget {
        max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: need.source_values,
        max_groups: need.groups, max_shard_revalidation_bytes: need.shard_revalidation_bytes,
    }, source_preparation_bytes: 128 * 1024 * 1024, allocator_reserve_bytes: 64 * 1024 * 1024,
        run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 100, cleanup_reserve_bytes: 65536 } }
}

#[test]
fn load_admission_prices_resident_and_transient_storage_without_double_charging_rows() {
    for width in [1, 3, 33, 128, MAX_MODEL_K] {
        let limits = limits(width); let need = Int4ModelLoadRequirement::for_group_width(width).unwrap();
        let claims = limits.requirements(width).unwrap();
        assert_eq!(claims.resident_bytes, need.weight_payload_bytes + limits.allocator_reserve_bytes);
        assert_eq!(claims.staging_bytes, limits.source_preparation_bytes + need.scratch_payload_bytes);
        assert_eq!(claims.resident_bytes + claims.staging_bytes,
            need.peak_payload_bytes().unwrap() + limits.source_preparation_bytes + limits.allocator_reserve_bytes);
    }
}

#[test]
fn every_load_budget_axis_refuses_before_source_preparation() {
    let baseline = limits(128);
    for axis in 0..8 {
        let mut limits = baseline;
        match axis { 0 => limits.model.max_payload_bytes -= 1, 1 => limits.model.max_source_values -= 1,
            2 => limits.model.max_groups -= 1, 3 => limits.model.max_shard_revalidation_bytes -= 1,
            4 => limits.source_preparation_bytes -= 1, 5 => limits.allocator_reserve_bytes = 0,
            6 => limits.run.max_elapsed = Duration::ZERO, _ => limits.run.max_checkpoints = 1 }
        assert!(limits.requirements(128).is_err());
    }
    for width in [0, MAX_MODEL_K + 1, usize::MAX] { assert!(baseline.requirements(width).is_err()); }
    assert!(Int4LoadLimits { source_preparation_bytes: u64::MAX, ..baseline }.requirements(128).is_err());
    assert!(Int4LoadLimits { allocator_reserve_bytes: u64::MAX, ..baseline }.requirements(128).is_err());
}

#[test]
fn retained_source_tokenizer_and_configuration_must_match_binary_assets_exactly() {
    check_tokenizer(PINNED_TOKENIZER_MODEL_BYTES, PINNED_TOKENIZER_CONFIG_BYTES).unwrap();
    for (model, config) in [(&[][..], PINNED_TOKENIZER_CONFIG_BYTES), (PINNED_TOKENIZER_MODEL_BYTES, &[][..]),
        (&PINNED_TOKENIZER_MODEL_BYTES[1..], PINNED_TOKENIZER_CONFIG_BYTES),
        (PINNED_TOKENIZER_MODEL_BYTES, &PINNED_TOKENIZER_CONFIG_BYTES[1..])] {
        assert!(matches!(check_tokenizer(model, config), Err(HostedError::ModelIdentity)));
    }
}

#[test]
fn new_host_errors_preserve_typed_causes_without_logging_nested_private_text() {
    let error = chat_error(Int4ChatError::Chat(crate::tasks::chat::ChatError::Contract("PRIVATE_PROMPT")));
    assert!(!format!("{error:?} {error}").contains("PRIVATE_PROMPT"));
    assert!(error.source().is_some());
    let error = model_error(Int4ModelError::Cancelled(crate::native_engine::decode::DecodeCancellationKind::Deadline));
    assert!(matches!(error, HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(
        crate::native_engine::decode::DecodeCancellationKind::Deadline)))));
}

#[test]
fn all_supported_partitions_use_the_same_aggregate_memory_formula() {
    for width in 1..=MAX_MODEL_K {
        let limits = limits(width); let need = Int4ModelLoadRequirement::for_group_width(width).unwrap();
        let claims = limits.requirements(width).unwrap();
        assert!(claims.resident_bytes >= need.weight_payload_bytes);
        assert!(claims.staging_bytes >= need.scratch_payload_bytes);
        assert_eq!(claims.resident_bytes + claims.staging_bytes,
            need.peak_payload_bytes().unwrap() + limits.source_preparation_bytes + limits.allocator_reserve_bytes);
    }
}
