//! Host admission and reservation arithmetic only; no native execution fixture.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding, quantized::long::Int8ExtractionMapLimits},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::quantized::long::Int8SourceMapLimits},
    validation::grounded_fields::GroundingBudget,
};
fn model() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "fixture".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(), source_root_sha256: "0".repeat(64), logical_model_sha256: "1".repeat(64) }
}
fn identity() -> ExecutionIdentity {
    let m = model(); let d = Sha256Digest::of_bytes(b"document-corpus-host-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: m.revision,
        logical_model_digest: Sha256Digest::from_hex(&m.logical_model_sha256).unwrap(), artifact_format: "fixture".to_owned(),
        quant_recipe: m.recipe_id, packing_set_digest: d, tokenizer_digest: d, template_digest: d,
        task_spec: "extract-v1".to_owned(), taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn request(ids: &[ExecutionIdentity]) -> ExtractionDocumentAdmission<'_> {
    ExtractionDocumentAdmission { identities: ids,
        model_work: Int8Work::for_sequence(0, 16, 16 * NANBEIGE_VOCAB_SIZE).unwrap(),
        mask_node_visits: 4000, mask_limits: Default::default(), kv_reservation_bytes: 4096, max_result_bytes: 8192 }
}
fn config() -> (ExtractionDocumentCorpusConfig, CorpusLimits) {
    let native = NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
        run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: 4096 } };
    let work = Int8Work::for_sequence(0, 128, 128 * NANBEIGE_VOCAB_SIZE).unwrap();
    let verification = GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 10000 };
    let config = ExtractionDocumentCorpusConfig { batch: ExtractionDocumentBatchConfig {
        request: ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(), grounding: ExtractionBatchGrounding::Structural,
            budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 1024,
                max_grammar_states: 4096, max_kv_bytes: 4096 } },
        limits: Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 16, max_chunk_tokens: 16,
                context_tokens: 128, reserved_tokens: 32, max_chunks: 8, ..ChunkLimits::default() },
            reduction: ExecutionLimits { max_value_bytes: 4096, max_live_value_bytes: 8192,
                max_result_bytes: 4096, ..ExecutionLimits::default() }, max_model_work: work,
            mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 8000 }, verification },
        max_model_work: work, max_mask_visits: 8000, max_verification: verification }, reduction_reserve_bytes: 4096 };
    let limits = CorpusLimits { native, transport: BatchLimits { max_output_line_bytes: 8192, ..BatchLimits::default() },
        preparation_reserve_bytes: 4096, io_reserve_bytes: 4096 };
    (config, limits)
}
#[test]
fn every_child_including_the_last_must_match_the_resident_model_and_task() {
    let mut ids = vec![identity(), identity()];
    ids[1].prompt_digest = Sha256Digest::of_bytes(b"legitimate-second-chunk");
    check_document_request(&model(), 4096, 8192, &request(&ids)).unwrap();
    for axis in 0..5 {
        let mut bad = ids.clone();
        match axis { 0 => bad[1].source_revision.push('x'), 1 => bad[1].quant_recipe.push('x'),
            2 => bad[1].logical_model_digest = Sha256Digest::of_bytes(b"foreign"),
            3 => bad[1].task_spec = "ner-v1".to_owned(), _ => bad[1].numerics_profile = NumericsProfile::HfBf16Eager }
        let error = check_document_request(&model(), 4096, 8192, &request(&bad)).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
}
#[test]
fn a_document_output_claim_is_not_a_per_chunk_claim_or_a_smaller_kv_reservation() {
    let ids = vec![identity(), identity()];
    let mut r = request(&ids); r.max_result_bytes = 8193;
    assert_eq!(check_document_request(&model(), 4096, 8192, &r).unwrap_err().fault.code, BatchCode::OutputLineLimit);
    r.max_result_bytes = 8192; r.kv_reservation_bytes = 4095;
    assert!(check_document_request(&model(), 4096, 8192, &r).unwrap_err().stop);
}
#[test]
fn empty_or_unbounded_partitions_and_missing_masks_refuse_before_output_admission() {
    for count in [0, 257] {
        let ids = vec![identity(); count];
        assert!(check_document_request(&model(), 4096, 8192, &request(&ids)).is_err());
    }
    let ids = vec![identity()];
    for axis in 0..3 { let mut r = request(&ids);
        match axis { 0 => r.mask_node_visits = 0, 1 => r.mask_limits.max_trie_node_visits = 0,
            _ => r.mask_limits.checkpoint_interval_nodes = 0 }
        assert!(check_document_request(&model(), 4096, 8192, &r).is_err());
    }
}
#[test]
fn host_context_output_and_reduction_reservations_cannot_be_bypassed() {
    let (c, limits) = config(); c.batch.validate().unwrap(); validate(&c, limits, 4096).unwrap();
    for axis in 0..4 { let (mut c, mut limits) = config();
        match axis { 0 => c.reduction_reserve_bytes = 0, 1 => c.batch.request.budget.max_kv_bytes = 4095,
            2 => limits.native.context_tokens += 1, _ => limits.transport.max_output_line_bytes = 4095 }
        assert!(validate(&c, limits, 4096).is_err());
    }
}
#[test]
fn corpus_workspace_prices_the_maximum_document_frontier_and_full_coordinates() {
    let (c, _) = config(); let b = &c.batch;
    let bytes = super::super::super::extraction_map::document_workspace_bytes(b.request.budget, b.limits, 8, 4096).unwrap();
    let fields = 32 * std::mem::size_of::<crate::validation::grounded_fields::SourceFieldEvidence>() as u64;
    let spans = 128 * 2 * 4 * std::mem::size_of::<usize>() as u64;
    assert!(bytes >= (8 * 1024 + 8192) * 4 + 8 * 16 * 8 + fields + spans + 4096);
    assert!(bytes > b.limits.mapping.reduction.max_result_bytes as u64);
    assert!(super::super::super::extraction_map::document_workspace_bytes(b.request.budget, b.limits, 9, 4096).is_err());
}
