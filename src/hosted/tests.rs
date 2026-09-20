use super::*;
use crate::execution_identity::{NumericsProfile, ThinkingMode, ToolMode};

#[test]
fn finite_limits_and_all_memory_sums_reject_zero_or_overflow() {
    assert!(RunLimits { max_elapsed: Duration::ZERO, max_checkpoints: 10, cleanup_reserve_bytes: 65536 }.validate().is_err());
    assert!(RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1, cleanup_reserve_bytes: 65536 }.validate().is_err());
    assert!(RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 2, cleanup_reserve_bytes: 65536 }.validate().is_ok());
    assert_eq!(sum(&[1, 2, 3]).unwrap(), 6);
    assert!(sum(&[u64::MAX, 1]).is_err());
    let native = NativeLimits { context_tokens: 0, allocator_reserve_bytes: 1,
        run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 10, cleanup_reserve_bytes: 65536 } };
    assert!(requirements(native).is_err());
    assert!(requirements(NativeLimits { context_tokens: 1, allocator_reserve_bytes: 0, ..native }).is_err());
}

fn identities() -> (ArtifactIdentity, ExecutionIdentity) {
    let d = Sha256Digest::of_bytes(b"fixture");
    let model = ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "fixture".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "0".repeat(64),
        logical_model_sha256: "0".repeat(64) };
    let identity = ExecutionIdentity { schema_version: 1, source_revision: model.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&model.logical_model_sha256).unwrap(),
        artifact_format: "fixture".to_owned(), quant_recipe: model.recipe_id.clone(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "chat-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: crate::native_engine::strict_int8::STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    (model, identity)
}

#[test]
fn actual_model_binding_rejects_every_materialized_identity_mismatch() {
    let (model, identity) = identities();
    check_model_identity(&model, &identity).unwrap();
    for axis in 0..5 {
        let mut changed = model.clone();
        match axis {
            0 => changed.model_id.push('x'), 1 => changed.revision.push('x'),
            2 => changed.recipe_id.push('x'), 3 => changed.logical_model_sha256 = "1".repeat(64),
            _ => changed.logical_model_sha256 = "malformed".to_owned(),
        }
        assert!(matches!(check_model_identity(&changed, &identity), Err(HostedError::ModelIdentity)));
    }
}

#[test]
fn payload_requirements_keep_the_full_44_slot_kv_and_explicit_scratch() {
    let native = NativeLimits { context_tokens: 32, allocator_reserve_bytes: 4096,
        run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 65536 } };
    let required = requirements(native).unwrap();
    assert_eq!(required.kv_bytes, 32 * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64);
    let budget = memory_budget(required);
    assert_eq!(budget.max_kv_bytes, required.kv_bytes);
    assert_eq!(budget.max_rope_bytes, required.rope_bytes);
    assert_eq!(budget.max_scratch_payload_bytes, required.scratch_payload_bound);
    assert!(required.rope_bytes > 0 && required.scratch_payload_bound > 0);
}

#[test]
fn diagnostic_formatting_does_not_export_private_nested_error_text() {
    let error = HostedError::Limits("PRIVATE_INPUT_SHOULD_NOT_BE_PRINTED");
    assert!(!format!("{error:?} {error}").contains("PRIVATE_INPUT"));
    let error = HostedError::Chat(Int8ChatError::Chat(crate::tasks::chat::ChatError::Contract("PRIVATE_INPUT")));
    assert!(!format!("{error:?} {error}").contains("PRIVATE_INPUT"));
    assert!(error.source().is_some(), "typed explicit diagnostic access is retained");
}
