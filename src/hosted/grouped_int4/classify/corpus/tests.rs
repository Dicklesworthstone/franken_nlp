//! Pure declared-memory/admission fixtures. Synthetic identities never grant
//! model execution or task-quality authority.
use super::*;
use crate::{artifact::converter::PINNED_REVISION, batch::BatchLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{Int4Work, STRICT_INT4_EXECUTION}},
    tasks::{classify::ClassificationAggregationPolicy, mapreduce::ChunkLimits}};

fn fixture() -> (TaskBudget, ClassificationLimits, Int4ClassificationMapLimits, CorpusLimits) {
    let task = TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes };
    let planning = ClassificationLimits { max_input_bytes: 4096, max_labels: 4,
        max_total_label_bytes: 1024, ..Default::default() };
    let mapping = Int4ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
        max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 2048, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: ClassificationAggregationPolicy::default() };
    let limits = CorpusLimits { native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65536,
        run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 128 * 1024 * 1024, io_reserve_bytes: 65536 };
    (task, planning, mapping, limits)
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"classification-document-host-admission-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize)).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "classify-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "complete-candidate-eos".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn request(identities: &[ExecutionIdentity]) -> Int4ClassificationDocumentAdmission<'_> {
    let (task, _, mapping, _) = fixture();
    Int4ClassificationDocumentAdmission { identities, model_work: mapping.max_model_work, group_width: 32,
        required_context: 768, task_budget: task, kv_reservation_bytes: task.max_kv_bytes,
        max_result_bytes: mapping.max_result_bytes }
}
fn check(request: &Int4ClassificationDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    let (task, _, mapping, limits) = fixture();
    check_request(32, limits.native.context_tokens, task.max_kv_bytes, mapping, request)
}

#[test]
fn corpus_quote_keeps_one_native_cache_and_adds_all_stream_and_frozen_identity_owners() {
    let (task, planning, mapping, limits) = fixture();
    for width in [1, 3, 32, 128, 10752] {
        let base = Int4ClassificationMapMemoryLimits { context_tokens: limits.native.context_tokens,
            preparation_reserve_bytes: limits.preparation_reserve_bytes,
            allocator_reserve_bytes: limits.native.allocator_reserve_bytes }
            .requirements(width, task, planning, mapping).unwrap();
        let actual = int4_classification_document_corpus_requirements(width, task, planning, mapping, limits).unwrap();
        assert_eq!(actual.native, base.native); assert_eq!(actual.workspace_bytes, base.workspace_bytes);
        assert_eq!(actual.max_output_bytes, mapping.max_result_bytes);
        assert_eq!(actual.output_reservation_bytes, base.output_reservation_bytes);
        assert_eq!(actual.input_bytes, base.input_bytes + limits.reservation_bytes().unwrap()
            + mapping.chunks.max_chunks as u64 * CLASSIFICATION_DOCUMENT_CHUNK_METADATA_BYTES);
    }
}

#[test]
fn missing_host_memory_axes_and_checked_overflows_fail_before_any_input_or_model_io() {
    for field in 0..10 {
        let (mut task, planning, mut mapping, mut limits) = fixture();
        match field { 0 => limits.preparation_reserve_bytes = 0, 1 => limits.io_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = 0, 3 => limits.native.run.max_checkpoints = 1,
            4 => limits.preparation_reserve_bytes = u64::MAX, 5 => limits.io_reserve_bytes = u64::MAX,
            6 => limits.native.allocator_reserve_bytes = u64::MAX, 7 => task.max_kv_bytes -= 1,
            8 => mapping.max_result_bytes = limits.transport.max_output_line_bytes as u64 + 1,
            _ => limits.transport.max_json_depth = 65 }
        assert!(int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).is_err(),
            "field {field}");
    }
    let (task, planning, mapping, limits) = fixture();
    for width in [0, 10753, usize::MAX] {
        assert!(int4_classification_document_corpus_requirements(width, task, planning, mapping, limits).is_err());
    }
}

#[test]
fn buffers_and_preparation_grow_exactly_without_allocating_another_native_cache() {
    let (task, planning, mapping, mut limits) = fixture();
    let base = int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).unwrap();
    limits.io_reserve_bytes += 4096; limits.preparation_reserve_bytes += 8192;
    let larger = int4_classification_document_corpus_requirements(32, task, planning, mapping, limits).unwrap();
    // The explicitly conservative map and corpus preparation quotes each
    // retain this owner allowance; the public helper and CLI share that fact.
    assert_eq!(larger.input_bytes - base.input_bytes, 4096 + 2 * 8192);
    assert_eq!(larger.native, base.native); assert_eq!(larger.workspace_bytes, base.workspace_bytes);
}

#[test]
fn complete_map_output_is_not_mistaken_for_one_chunk_output_or_one_context_work() {
    let identities = [identity(), identity()]; let request = request(&identities);
    assert!(request.max_result_bytes > request.task_budget.max_output_bytes);
    assert!(request.model_work.forward_positions > fixture().3.native.context_tokens as u64);
    check(&request).unwrap();
}

#[test]
fn every_chunk_identity_must_keep_the_model_profile_and_classification_task() {
    for field in 0..6 {
        let mut identities = [identity(), identity()];
        match field { 0 => identities[1].task_spec = "judge-v1".to_owned(),
            1 => identities[1].thinking_mode = ThinkingMode::Enabled,
            2 => identities[1].tool_mode = ToolMode::Json,
            3 => identities[1].numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            4 => identities[1].kv_dtype = "int8".to_owned(),
            _ => identities[1].packing_set_digest = Sha256Digest::of_bytes(b"wrong grouping") }
        let error = check(&request(&identities)).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
    let empty = request(&[]); assert!(check(&empty).is_err());
    let too_many = vec![identity(); fixture().2.chunks.max_chunks + 1];
    assert!(check(&request(&too_many)).is_err());
}

#[test]
fn actual_full_kv_and_all_document_work_axes_are_independently_authorized() {
    let identities = [identity(), identity()];
    for axis in 0..5 {
        let mut item = request(&identities);
        match axis { 0 => item.model_work.forward_positions += 1, 1 => item.model_work.projected_logits += 1,
            2 => item.model_work.attention_pairs += 1, 3 => item.model_work.projections.dot_products += 1,
            _ => item.model_work.projections.multiply_accumulates += 1 }
        let failure = check(&item).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::InvalidExecution);
    }
    for field in 0..6 {
        let mut item = request(&identities);
        match field { 0 => item.kv_reservation_bytes -= 1, 1 => item.task_budget.max_kv_bytes -= 1,
            2 => item.required_context = 0, 3 => item.required_context = 1025,
            4 => item.group_width = 128, _ => item.max_result_bytes -= 1 }
        assert!(check(&item).unwrap_err().stop, "field {field}");
    }
}

#[test]
fn owned_stream_crosses_only_the_existing_blocking_host_surface() {
    let _host = NlpEngine::batch_int4_classification_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _quote = crate::hosted::int4_classification_document_corpus_requirements;
}
