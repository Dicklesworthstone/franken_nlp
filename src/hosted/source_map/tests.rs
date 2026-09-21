//! Host geometry, ownership and error contracts, not model execution evidence.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits}, ner::NerOptions,
        source_planning::quantized::Int8SourceError},
};
fn config() -> SourceMapConfig {
    let d = Sha256Digest::of_bytes(b"host source-map fixture");
    SourceMapConfig {
        identity: ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
            tokenizer_digest: d, template_digest: d, task_spec: "ner-v1".to_owned(), taskir_digest: d, prompt_digest: d,
            grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None },
        task: SourceMapTask::Ner(NerOptions::default()),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65536,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 },
        planning: SourcePlanningLimits::default(),
        mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_chunks: 8, ..ChunkLimits::default() },
            reduction: ExecutionLimits::default(),
            max_model_work: Int8Work::for_sequence(0, 8192, 512 * NANBEIGE_VOCAB_SIZE).unwrap(),
            mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 10000, max_mask_visits: 80000,
        },
        native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100000, cleanup_reserve_bytes: 65536 } },
        preparation_reserve_bytes: 16 << 20, reduction_reserve_bytes: 1 << 20,
    }
}
#[test]
fn complete_context_and_resident_kv_must_fit_before_owned_planning() {
    let mut c = config(); validate(&c, 10, c.budget.max_kv_bytes).unwrap();
    assert!(validate(&c, 10, c.budget.max_kv_bytes + 1).is_err());
    c.native.context_tokens -= 1;
    assert!(validate(&c, 10, 4096).is_err());
}
#[test]
fn configured_task_and_native_profile_cannot_be_repaired() {
    for axis in 0..6 {
        let mut c = config();
        match axis {
            0 => c.identity.task_spec = "answer-v1".to_owned(),
            1 => c.identity.numerics_profile = NumericsProfile::HfBf16Eager,
            2 => c.identity.backend_semantic_version = "other".to_owned(),
            3 => c.identity.kv_dtype = "int8".to_owned(),
            4 => c.identity.thinking_mode = ThinkingMode::Enabled,
            _ => c.identity.tool_mode = ToolMode::Json,
        }
        assert!(matches!(validate(&c, 10, 4096), Err(HostedError::ModelIdentity)));
    }
}
#[test]
fn source_and_both_nonpayload_reserves_are_explicit() {
    let c = config();
    assert!(validate(&c, 0, 4096).is_err());
    assert!(validate(&c, c.mapping.chunks.max_input_bytes + 1, 4096).is_err());
    for axis in 0..2 {
        let mut c = config();
        if axis == 0 { c.preparation_reserve_bytes = 0; } else { c.reduction_reserve_bytes = 0; }
        assert!(validate(&c, 10, 4096).is_err());
    }
}
#[test]
fn pinned_planner_assets_must_match_both_identity_fields() {
    let c = config(); let d = c.identity.tokenizer_digest; let other = Sha256Digest::of_bytes(b"other");
    check_assets(&c.identity, d, d).unwrap();
    assert!(matches!(check_assets(&c.identity, other, d), Err(HostedError::ModelIdentity)));
    assert!(matches!(check_assets(&c.identity, d, other), Err(HostedError::ModelIdentity)));
}
#[test]
fn all_native_selections_require_full_vocabulary_projection_work() {
    let mut c = config(); c.mapping.max_model_work.projected_logits = NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert!(validate(&c, 10, 4096).is_err());
}
#[test]
fn reduction_reservation_prices_actual_chunks_plus_the_retained_frontier() {
    let mut c = config(); let chunks = 3;
    let expected = 4 * (c.budget.max_output_bytes * chunks as u64 + c.mapping.reduction.max_live_value_bytes as u64)
        + 8 * u64::from(c.budget.max_output_tokens) * chunks as u64 + c.reduction_reserve_bytes;
    assert_eq!(reduction_bytes(&c, chunks).unwrap(), expected);
    c.mapping.chunks.max_chunks = 256;
    assert_eq!(reduction_bytes(&c, chunks).unwrap(), expected);
    c.reduction_reserve_bytes += 17;
    assert_eq!(reduction_bytes(&c, chunks).unwrap(), expected + 17);
}
#[test]
fn impossible_partition_cardinality_is_not_a_resource_receipt() {
    let mut c = config();
    for n in [0, 9, usize::MAX] { assert!(reduction_bytes(&c, n).is_err()); }
    c.mapping.chunks.max_chunks = 1024;
    assert!(reduction_bytes(&c, 257).is_err());
}
#[test]
fn every_modeled_result_and_staging_overflow_refuses() {
    let mut c = config(); c.budget.max_output_bytes = u64::MAX;
    assert!(reduction_bytes(&c, 2).is_err());
    c.budget.max_output_bytes = u64::MAX / 4;
    assert!(reduction_bytes(&c, 1).is_err());
    let mut c = config(); c.reduction_reserve_bytes = u64::MAX;
    assert!(reduction_bytes(&c, 1).is_err());
    if usize::BITS == 64 {
        let mut c = config(); c.mapping.reduction.max_live_value_bytes = usize::MAX;
        assert!(reduction_bytes(&c, 1).is_err());
    }
}
#[test]
fn run_duration_checkpoints_and_cleanup_are_whole_document_limits() {
    for axis in 0..3 {
        let mut c = config();
        match axis { 0 => c.native.run.max_elapsed = Duration::ZERO,
            1 => c.native.run.max_checkpoints = 1, _ => c.native.run.cleanup_reserve_bytes = 0 }
        assert!(validate(&c, 10, 4096).is_err());
    }
}
#[test]
fn native_source_map_errors_keep_typed_cancellation_without_source_text() {
    let error = HostedError::SourceMap(Int8SourceMapError::Source(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)));
    let HostedError::SourceMap(ref inner) = error else { unreachable!() };
    assert_eq!(inner.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(format!("{error}"), "hosted native source map failed");
    assert_eq!(format!("{error:?}"), "hosted native source map failed");
}
#[test]
fn owned_inputs_and_guarded_results_cross_the_physical_completion_boundary() {
    fn send<T: Send + 'static>() {}
    send::<SourceMapInput>(); send::<Int8SourceMapRun>(); send::<HostedOutput<Int8SourceMapRun>>();
}
