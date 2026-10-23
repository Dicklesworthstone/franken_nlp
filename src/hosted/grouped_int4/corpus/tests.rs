//! Admission checks over real declared work/identity types, without pretending
//! a synthetic model or writer transcript establishes native inference parity.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::BatchLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::{mask::MaskWorkLimits, runtime::JSON_RUNTIME_VERSION},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
};

fn limits() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1024 * 1024,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10_000,
            cleanup_reserve_bytes: 65_536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 128 * 1024 * 1024,
        io_reserve_bytes: 64 * 1024 }
}
fn masks() -> ExtractionMaskBudget {
    ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        max_visits_per_item: 2000, max_visits_per_run: 8000 }
}
fn work(width: usize) -> Int4Work { constrained_int4::planned_work(width, 8, 4).unwrap() }
fn native_work(width: usize) -> Int4ExtractionBatchLimits {
    Int4ExtractionBatchLimits { max_model_work: work(width), masks: masks() }
}
fn identity(width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-host-admission-only");
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
fn request(identity: &ExecutionIdentity, width: usize) -> Int4ExtractionAdmission<'_> {
    Int4ExtractionAdmission { identity, model_work: work(width),
        mask_node_visits: masks().max_visits_per_item, mask_limits: masks().per_mask,
        kv_reservation_bytes: 32 * KV_BYTES_PER_TOKEN as u64, max_result_bytes: 8192 }
}
fn check(width: usize, request: &Int4ExtractionAdmission<'_>) -> Result<(), BatchItemFailure> {
    check_request(width, 32 * KV_BYTES_PER_TOKEN as u64, 8192, masks(), request)
}

#[test]
fn complete_native_workspace_and_owned_transport_are_charged_once() {
    for width in [1, 3, 32, 127, 128, MAX_MODEL_K] {
        let limits = limits();
        let need = requirements(width, native_work(width), limits).unwrap();
        let expected = Int4MemoryRequirement::for_context(32, width).unwrap();
        assert_eq!(need.native, expected);
        assert_eq!(need.native.kv_bytes, 32 * KV_BYTES_PER_TOKEN as u64);
        assert_eq!(need.workspace_bytes,
            expected.rope_bytes + expected.scratch_payload_bound + limits.native.allocator_reserve_bytes);
        assert_eq!(need.input_bytes, limits.reservation_bytes().unwrap());
    }
}

#[test]
fn whole_corpus_work_may_exceed_one_document_context_without_renewing_it() {
    let mut budget = native_work(128);
    let work = &mut budget.max_model_work;
    work.forward_positions *= 10; work.projected_logits *= 10; work.attention_pairs *= 10;
    work.projections.dot_products *= 10; work.projections.multiply_accumulates *= 10;
    let need = requirements(128, budget, limits()).unwrap();
    assert!(budget.max_model_work.forward_positions > limits().native.context_tokens as u64);
    assert_eq!(need.native.kv_bytes, 32 * KV_BYTES_PER_TOKEN as u64);
}

#[test]
fn every_missing_native_or_mask_budget_axis_refuses_before_allocation() {
    for axis in 0..9 {
        let mut budget = native_work(128);
        match axis {
            0 => budget.max_model_work.forward_positions = 0,
            1 => budget.max_model_work.projected_logits = 0,
            2 => budget.max_model_work.attention_pairs = 0,
            3 => budget.max_model_work.projections.dot_products = 0,
            4 => budget.max_model_work.projections.multiply_accumulates = 0,
            5 => budget.masks.per_mask.max_trie_node_visits = 0,
            6 => budget.masks.per_mask.checkpoint_interval_nodes = 0,
            7 => budget.masks.max_visits_per_item = 0,
            _ => budget.masks.max_visits_per_run = budget.masks.max_visits_per_item - 1,
        }
        assert!(requirements(128, budget, limits()).is_err(), "axis {axis}");
    }
}

#[test]
fn invalid_context_or_unpriced_and_overflowing_storage_refuses_before_io() {
    for width in [0, MAX_MODEL_K + 1, usize::MAX] {
        assert!(requirements(width, native_work(128), limits()).is_err());
    }
    for axis in 0..7 {
        let mut limits = limits();
        match axis {
            0 => limits.native.context_tokens = 0,
            1 => limits.native.allocator_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = u64::MAX,
            3 => limits.preparation_reserve_bytes = 0,
            4 => limits.io_reserve_bytes = 0,
            5 => limits.preparation_reserve_bytes = u64::MAX,
            _ => limits.transport.max_json_depth = 65,
        }
        assert!(requirements(128, native_work(128), limits).is_err(), "axis {axis}");
    }
}

#[test]
fn request_profile_group_task_and_thinking_modes_are_not_repaired() {
    for axis in 0..8 {
        let mut id = identity(128);
        match axis {
            0 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            1 => id.backend_semantic_version = "other".to_owned(),
            2 => id.kv_dtype = "int8".to_owned(),
            3 => id.artifact_format = "fnlpq-v1".to_owned(),
            4 => id.thinking_mode = ThinkingMode::Enabled,
            5 => id.tool_mode = ToolMode::Json,
            6 => id.task_spec = "chat-v1".to_owned(),
            _ => id.packing_set_digest = Sha256Digest::of_bytes(b"different-groups"),
        }
        let failure = check(128, &request(&id, 128)).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
    }
    let id = identity(127);
    assert!(check(128, &request(&id, 127)).unwrap_err().stop);
}

#[test]
fn resident_kv_must_equal_the_full_physical_charge_and_fit_the_item() {
    let id = identity(128);
    let mut request = request(&id, 128);
    check(128, &request).unwrap();
    for bytes in [0, request.kv_reservation_bytes - 1, request.kv_reservation_bytes + 1, u64::MAX] {
        request.kv_reservation_bytes = bytes;
        let failure = check(128, &request).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
    }
    request.kv_reservation_bytes = (request.model_work.forward_positions - 1) * KV_BYTES_PER_TOKEN as u64;
    let failure = check_request(128, request.kv_reservation_bytes, 8192, masks(), &request).unwrap_err();
    assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
}

#[test]
fn mask_allowances_cannot_drift_between_setup_and_per_document_admission() {
    let id = identity(128);
    for axis in 0..7 {
        let mut request = request(&id, 128);
        match axis {
            0 => request.mask_node_visits = 0,
            1 => request.mask_node_visits -= 1,
            2 => request.mask_node_visits += 1,
            3 => request.mask_limits.max_trie_node_visits -= 1,
            4 => request.mask_limits.max_trie_node_visits += 1,
            5 => request.mask_limits.checkpoint_interval_nodes -= 1,
            _ => request.mask_limits.checkpoint_interval_nodes += 1,
        }
        let failure = check(128, &request).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn all_group_work_axes_and_full_vocabulary_projection_are_rechecked() {
    let id = identity(128);
    for axis in 0..7 {
        let mut request = request(&id, 128);
        match axis {
            0 => request.model_work.forward_positions = 0,
            1 => request.model_work.projected_logits = 0,
            2 => request.model_work.attention_pairs -= 1,
            3 => request.model_work.projections.dot_products -= 1,
            4 => request.model_work.projections.multiply_accumulates -= 1,
            5 => request.model_work.projected_logits -= 1,
            _ => request.model_work = work(32),
        }
        let failure = check(128, &request).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn result_limit_is_a_document_refusal_without_fabricating_completion() {
    let id = identity(128);
    let mut request = request(&id, 128);
    for bytes in [0, 8193, u64::MAX] {
        request.max_result_bytes = bytes;
        let failure = check(128, &request).unwrap_err();
        assert!(!failure.stop); assert_eq!(failure.fault.code, BatchCode::OutputLineLimit);
    }
    request.max_result_bytes = 1;
    check(128, &request).unwrap();
}

#[test]
fn owned_corpus_inputs_can_cross_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<StreamInput<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
    send::<Int4ExtractionBatchPlanner>();
    send::<Arc<ExtractionVocabulary>>();
}
