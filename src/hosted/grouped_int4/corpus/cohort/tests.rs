//! Admission fixtures only; no synthetic value is native parity evidence.
use super::*;
use crate::{artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::{mask::MaskWorkLimits, runtime::JSON_RUNTIME_VERSION},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION}};

fn masks() -> ExtractionMaskBudget {
    ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_visits_per_item: 2000, max_visits_per_run: 8000 }
}
fn identity(width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"cohort-host-admission-only");
    let packing = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
    )).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "constrained-json".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn request(identity: &ExecutionIdentity, width: usize, prompt: usize) -> Int4ExtractionAdmission<'_> {
    let work = constrained_int4::planned_work(width, prompt, 4).unwrap();
    Int4ExtractionAdmission { identity, model_work: work, mask_node_visits: masks().max_visits_per_item,
        mask_limits: masks().per_mask, kv_reservation_bytes: work.forward_positions * KV_BYTES_PER_TOKEN as u64,
        max_result_bytes: 8192 }
}
#[test]
fn different_document_contexts_keep_exact_independent_kv_and_dense_work() {
    for width in [1, 3, 32, 127, 128, MAX_MODEL_K] {
        let identity = identity(width);
        for prompt in [8, 21] {
            let request = request(&identity, width, prompt);
            check_cohort_request(width, 32, 8192, masks(), &request).unwrap();
            assert_eq!(request.kv_reservation_bytes, (prompt + 3) as u64 * KV_BYTES_PER_TOKEN as u64);
        }
    }
}
#[test]
fn neither_neighbor_contexts_nor_a_serial_reusable_cache_can_replace_the_exact_row_charge() {
    let identity = identity(128); let original = request(&identity, 128, 8);
    for bytes in [0, original.kv_reservation_bytes - 1, 32 * KV_BYTES_PER_TOKEN as u64,
        2 * original.kv_reservation_bytes, u64::MAX] {
        let mut request = request(&identity, 128, 8); request.kv_reservation_bytes = bytes;
        let error = check_cohort_request(128, 32, 8192, masks(), &request).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
    assert!(check_cohort_request(128, 10, 8192, masks(), &original).is_err());
}
#[test]
fn changing_any_native_work_axis_or_mask_identity_is_a_fatal_admission_mismatch() {
    let identity = identity(128);
    for axis in 0..8 {
        let mut request = request(&identity, 128, 8);
        match axis {
            0 => request.model_work.forward_positions += 1,
            1 => request.model_work.projected_logits -= 1,
            2 => request.model_work.attention_pairs -= 1,
            3 => request.model_work.projections.dot_products -= 1,
            4 => request.model_work.projections.multiply_accumulates -= 1,
            5 => request.mask_node_visits += 1,
            6 => request.mask_limits.max_trie_node_visits += 1,
            _ => request.mask_limits.checkpoint_interval_nodes += 1,
        }
        assert!(check_cohort_request(128, 32, 8192, masks(), &request).unwrap_err().stop, "axis {axis}");
    }
}
#[test]
fn complete_output_bounds_remain_typed_document_rejections() {
    let identity = identity(128); let request = request(&identity, 128, 8);
    let error = check_cohort_request(128, 32, 8191, masks(), &request).unwrap_err();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
}
