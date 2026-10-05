//! Model-free admission/ownership arithmetic, not model or execution evidence.
use super::*;
use crate::{
    batch::extract::ExtractionBatchGrounding,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::DecodeCancellationKind, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::quantized::long::Int8SourceMapLimits},
    validation::grounded_fields::GroundingBudget,
};

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"host-extraction-map-model-free-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn config() -> ExtractionMapConfig {
    ExtractionMapConfig {
        request: ExtractionBatchArgs { schema: r#"{"type":"string","maxLength":16}"#.to_owned(),
            grounding: ExtractionBatchGrounding::SourceMembership,
            budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
                max_grammar_states: 4096, max_kv_bytes: 1024 } },
        limits: Int8ExtractionMapLimits {
            mapping: Int8SourceMapLimits {
                chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 16, max_chunk_tokens: 16,
                    context_tokens: 128, reserved_tokens: 32, max_chunks: 16, ..ChunkLimits::default() },
                reduction: ExecutionLimits { max_value_bytes: 4096, max_live_value_bytes: 8192,
                    max_result_bytes: 4096, ..ExecutionLimits::default() },
                max_model_work: Int8Work::for_sequence(0, 128, 128 * 166_144).unwrap(),
                mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 16_000 },
            verification: GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 10_000 } },
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 4096 } },
        preparation_reserve_bytes: 4096, reduction_reserve_bytes: 4096,
    }
}
#[test]
fn task_profile_and_full_resident_capacity_are_required() {
    validate(&config(), &identity(), 10, 1024).unwrap();
    assert!(validate(&config(), &identity(), 0, 1024).is_err());
    assert!(validate(&config(), &identity(), 1025, 1024).is_err());
    assert!(validate(&config(), &identity(), 10, 1025).is_err());
    let mut id = identity(); id.task_spec = "ner-v1".to_owned();
    assert!(validate(&config(), &id, 10, 1024).is_err());
    let mut id = identity(); id.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(validate(&config(), &id, 10, 1024).is_err());
    let mut id = identity(); id.backend_semantic_version = "other".to_owned();
    assert!(validate(&config(), &id, 10, 1024).is_err());
}
#[test]
fn context_and_nonrenewable_work_limits_fail_before_native_allocation() {
    for axis in 0..13 {
        let mut c = config();
        match axis {
            0 => c.native.context_tokens = 127,
            1 => c.limits.mapping.chunks.max_chunks = 257,
            2 => c.limits.mapping.chunks.max_chunk_tokens = 0,
            3 => c.limits.mapping.max_model_work.forward_positions = 0,
            4 => c.limits.mapping.max_model_work.projected_logits = 0,
            5 => c.limits.mapping.max_model_work.attention_pairs = 0,
            6 => c.limits.mapping.max_model_work.projections.dot_products = 0,
            7 => c.limits.mapping.max_model_work.projections.multiply_accumulates = 0,
            8 => c.limits.mapping.mask_visits_per_chunk = 0,
            9 => c.limits.mapping.max_mask_visits = 999,
            10 => c.limits.mapping.mask_limits.max_trie_node_visits = 0,
            11 => c.limits.mapping.mask_limits.checkpoint_interval_nodes = 0,
            _ => c.limits.mapping.chunks.max_chunks = 0,
        }
        assert!(validate(&c, &identity(), 10, 1024).is_err(), "axis {axis}");
    }
}
#[test]
fn schema_preparation_output_and_verification_limits_are_not_optional() {
    for axis in 0..12 {
        let mut c = config();
        match axis {
            0 => c.request.schema.clear(),
            1 => c.request.budget.max_output_tokens = 0,
            2 => c.preparation_reserve_bytes = 0,
            3 => c.reduction_reserve_bytes = 0,
            4 => c.limits.mapping.reduction.max_result_bytes = 0,
            5 => c.limits.mapping.reduction.max_live_value_bytes = 0,
            6 => c.limits.verification.max_fields = 0,
            7 => c.limits.verification.max_fields = 1_000_001,
            8 => c.limits.verification.max_matches = 0,
            9 => c.limits.verification.max_matches = 1_000_001,
            10 => c.limits.verification.max_scan_steps = 0,
            _ => c.native.run.max_checkpoints = 1,
        }
        assert!(validate(&c, &identity(), 10, 1024).is_err(), "axis {axis}");
    }
}
#[test]
fn schema_capacity_not_only_schema_length_is_charged() {
    let mut c = config();
    let mut schema = String::with_capacity(8192); schema.push_str(&c.request.schema);
    c.request.schema = schema;
    assert!(c.request.schema.capacity() > c.request.schema.len());
    assert_eq!(input_bytes(1024, &c).unwrap(), 1024 + c.request.schema.capacity() as u64 + 4096);
    c.preparation_reserve_bytes = u64::MAX;
    assert!(input_bytes(1024, &c).is_err());
}
#[test]
fn reduction_charges_native_results_and_temporary_coordinate_scan_storage() {
    let c = config(); let count = 3;
    let fields = 32 * std::mem::size_of::<SourceFieldEvidence>() as u64;
    let spans = 128 * std::mem::size_of::<SourceOccurrence>() as u64 * 2;
    assert_eq!(reduction_bytes(&c, count).unwrap(),
        (4096 * count as u64 + 8192) * 4 + 16 * count as u64 * 8 + fields + spans + 4096);
    assert!(reduction_bytes(&c, 0).is_err());
    assert!(reduction_bytes(&c, 17).is_err());
    let mut more = config(); more.limits.verification.max_matches += 1;
    assert_eq!(reduction_bytes(&more, count).unwrap() - reduction_bytes(&c, count).unwrap(),
        2 * std::mem::size_of::<SourceOccurrence>() as u64);
}
#[test]
fn reduction_arithmetic_never_wraps_into_a_smaller_reservation() {
    let mut c = config(); c.request.budget.max_output_bytes = u64::MAX;
    assert!(reduction_bytes(&c, 3).is_err());
    let mut c = config(); c.reduction_reserve_bytes = u64::MAX;
    assert!(reduction_bytes(&c, 1).is_err());
    let mut c = config(); c.request.budget.max_output_bytes = u64::MAX / 2;
    assert!(reduction_bytes(&c, 1).is_err());
}
#[test]
fn structural_mode_does_not_bypass_whole_document_admission() {
    let mut c = config(); c.request.grounding = ExtractionBatchGrounding::Structural;
    validate(&c, &identity(), 10, 1024).unwrap();
    c.limits.verification.max_scan_steps = 0;
    assert!(validate(&c, &identity(), 10, 1024).is_err());
}
#[test]
fn cancellation_is_typed_and_does_not_echo_schema_or_source() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
        fn prefill_checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let error = poll(&mut Stop).unwrap_err();
    let HostedError::SourceMap(ref source) = error else { panic!("source map error") };
    assert_eq!(source.cancellation(), Some(DecodeCancellationKind::Deadline));
    let diagnostic = format!("{error:?}");
    assert!(!diagnostic.contains(&config().request.schema));
    assert!(!diagnostic.contains("host-extraction-map-model-free-fixture"));
}
