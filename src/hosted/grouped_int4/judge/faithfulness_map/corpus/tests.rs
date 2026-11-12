//! Pure reservation and admission regressions. Synthetic identities below
//! cannot construct a native model or attest to faithfulness quality.
#[cfg(feature = "asupersync-runtime")]
mod team;

use super::*;
use std::time::Duration;
use crate::{artifact::converter::PINNED_REVISION, batch::BatchLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{mapreduce::ChunkLimits, judge::FaithfulnessAggregationPolicy}};

fn fixture() -> (TaskBudget, JudgeLimits, Int4FaithfulnessMapLimits, CorpusLimits) {
    let task = TaskBudget { max_input_tokens: 128, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(256, 32).unwrap().kv_bytes * 32 };
    let mapping = Int4FaithfulnessMapLimits { chunks: ChunkLimits {
        max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 128, context_tokens: 256,
        reserved_tokens: 16, max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 8 * 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: FaithfulnessAggregationPolicy::default() };
    let limits = CorpusLimits { native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000, cleanup_reserve_bytes: 65_536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 128 * 1024 * 1024, io_reserve_bytes: 65_536 };
    (task, JudgeLimits::default(), mapping, limits)
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"faithfulness-document-host-admission-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize)).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "judge-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "faithfulness-complete-candidate-eos-v1".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn capacity(cohort_chunks: Option<usize>) -> DocumentCapacity {
    let (task, planning, mapping, limits) = fixture();
    let quote = int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, cohort_chunks).unwrap();
    DocumentCapacity { context: limits.native.context_tokens, kv_bytes: quote.native.kv_bytes, mapping,
        cohort_chunks, max_heads: MAX_FAITHFULNESS_MAP_HEADS,
        max_result_bytes: document_output_bytes(mapping, cohort_chunks).unwrap(), task_budget: task }
}
fn request(identities: &[ExecutionIdentity], capacity: DocumentCapacity) -> Int4FaithfulnessDocumentAdmission<'_> {
    Int4FaithfulnessDocumentAdmission { identities, model_work: capacity.mapping.max_model_work, group_width: 32,
        required_context: 128, head_count: 2 * identities.len(), max_head_count: 2,
        task_budget: capacity.task_budget, kv_reservation_bytes: capacity.kv_bytes,
        max_result_bytes: capacity.max_result_bytes, cohort_chunks: capacity.cohort_chunks }
}

#[test]
fn complete_quote_adds_owned_transport_and_frozen_identity_storage_once() {
    let (task, planning, mapping, limits) = fixture();
    for chunks in [None, Some(1), Some(2), Some(3), Some(16), Some(64)] {
        let base = match chunks {
            None => map_memory(limits).requirements(32, task, planning, mapping).unwrap(),
            Some(chunks) => map_memory(limits).cohort_requirements(32, task, planning, mapping,
                MAX_FAITHFULNESS_MAP_HEADS, chunks).unwrap().memory,
        };
        let actual = int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, chunks).unwrap();
        assert_eq!(actual.native, base.native);
        assert_eq!(actual.workspace_bytes, base.workspace_bytes);
        assert_eq!(actual.max_output_bytes, base.max_output_bytes);
        assert_eq!(actual.output_reservation_bytes, base.output_reservation_bytes);
        assert_eq!(actual.input_bytes, base.input_bytes + limits.reservation_bytes().unwrap()
            + mapping.chunks.max_chunks as u64 * FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES);
    }
}

#[test]
fn ragged_window_quote_does_not_refuse_chunks_times_theoretical_maximum() {
    let (task, planning, mapping, limits) = fixture();
    let row = Int4MemoryRequirement::for_context(limits.native.context_tokens, 32).unwrap();
    for chunks in [3, 16, 64] {
        let quote = int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, Some(chunks)).unwrap();
        assert_eq!(quote.native.kv_bytes, row.kv_bytes * 64);
    }
    for chunks in [0, 65, usize::MAX] {
        assert!(int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, Some(chunks)).is_err());
    }
    let mut mapping = mapping; mapping.chunks.max_chunks = 2;
    assert!(int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, Some(3)).is_err());
}

#[test]
fn total_document_heads_are_distinct_from_simultaneous_window_capacity() {
    let identities = [identity(), identity(), identity(), identity()];
    let capacity = capacity(Some(3));
    let mut request = request(&identities, capacity);
    // Four ragged chunks can have 17 + 16 + 16 + 16 heads. Both requested
    // three-chunk windows fit although the document has more than 64 heads.
    request.head_count = 65; request.max_head_count = 17;
    check_request(32, capacity, &request).unwrap();
    request.head_count = 69;
    assert_eq!(check_request(32, capacity, &request).unwrap_err().fault.code, BatchCode::Admission);
}

#[test]
fn serial_execution_only_prices_one_physical_cache_in_its_task_allowance() {
    let (mut task, planning, mapping, limits) = fixture();
    task.max_kv_bytes = Int4MemoryRequirement::for_context(limits.native.context_tokens, 32).unwrap().kv_bytes;
    let quote = int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, None).unwrap();
    let capacity = DocumentCapacity { task_budget: task, kv_bytes: quote.native.kv_bytes, ..capacity(None) };
    let identities = [identity(), identity()];
    let mut request = request(&identities, capacity);
    request.head_count = 64; request.max_head_count = 32;
    check_request(32, capacity, &request).unwrap();
    assert!(int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, Some(1)).is_err());
}

#[test]
fn capped_corpus_quote_preserves_legacy_default_and_enables_small_evidence_sets() {
    let (task, planning, mapping, limits) = fixture();
    let row = Int4MemoryRequirement::for_context(limits.native.context_tokens, 32).unwrap().kv_bytes;
    for chunks in [None, Some(1), Some(3), Some(64)] {
        let legacy = int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, chunks).unwrap();
        let unchanged = int4_faithfulness_document_corpus_requirements_with_max_heads(
            32, task, planning, mapping, limits, 32, chunks).unwrap();
        assert_eq!(unchanged, legacy);
        let mut one_row = task; one_row.max_kv_bytes = row;
        let capped = int4_faithfulness_document_corpus_requirements_with_max_heads(
            32, one_row, planning, mapping, limits, 1, chunks).unwrap();
        assert_eq!(capped.native.kv_bytes, row * chunks.unwrap_or(1) as u64);
        assert!(capped.input_bytes < legacy.input_bytes);
        assert_eq!(capped.output_reservation_bytes, legacy.output_reservation_bytes);
        assert_eq!(capped.max_output_bytes, legacy.max_output_bytes);
    }
    for cap in [0, 33, usize::MAX] {
        assert!(int4_faithfulness_document_corpus_requirements_with_max_heads(
            32, task, planning, mapping, limits, cap, None).is_err());
    }
}

#[test]
fn admission_requires_actual_heads_and_physical_peak_to_fit_the_declared_cap() {
    let (mut task, planning, mapping, limits) = fixture();
    task.max_kv_bytes = Int4MemoryRequirement::for_context(limits.native.context_tokens, 32).unwrap().kv_bytes * 3;
    let quote = int4_faithfulness_document_corpus_requirements_with_max_heads(
        32, task, planning, mapping, limits, 3, Some(3)).unwrap();
    let capacity = DocumentCapacity { task_budget: task, kv_bytes: quote.native.kv_bytes,
        max_heads: 3, ..capacity(Some(3)) };
    let identities = [identity(), identity(), identity(), identity()];
    let mut request = request(&identities, capacity);
    request.head_count = 11; request.max_head_count = 3;
    check_request(32, capacity, &request).unwrap();
    request.max_head_count = 4;
    assert_eq!(check_request(32, capacity, &request).unwrap_err().fault.code, BatchCode::Admission);
    request.max_head_count = 3;
    request.kv_reservation_bytes += 1;
    assert_eq!(check_request(32, capacity, &request).unwrap_err().fault.code, BatchCode::Admission);
    request.kv_reservation_bytes -= 1;
    for cap in [0, 33, usize::MAX] {
        assert_eq!(check_request(32, DocumentCapacity { max_heads: cap, ..capacity }, &request)
            .unwrap_err().fault.code, BatchCode::Admission);
    }
}

#[test]
fn every_native_axis_and_fixed_execution_cap_is_checked_at_admission() {
    let identities = [identity(), identity()];
    let capacity = capacity(Some(3));
    for axis in 0..5 {
        let mut request = request(&identities, capacity);
        match axis { 0 => request.model_work.forward_positions += 1, 1 => request.model_work.projected_logits += 1,
            2 => request.model_work.attention_pairs += 1, 3 => request.model_work.projections.dot_products += 1,
            _ => request.model_work.projections.multiply_accumulates += 1 }
        let error = check_request(32, capacity, &request).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
    for fault in 0..12 {
        let mut request = request(&identities, capacity);
        match fault { 0 => request.cohort_chunks = None, 1 => request.cohort_chunks = Some(2),
            2 => request.head_count = 1, 3 => request.max_head_count = 0, 4 => request.max_head_count = 33,
            5 => request.kv_reservation_bytes -= 1, 6 => request.required_context = 257,
            7 => request.max_result_bytes = capacity.mapping.max_result_bytes,
            8 => request.task_budget.max_output_bytes += 1, 9 => request.group_width = 128,
            10 => request.required_context = 0, _ => request.head_count = 5 }
        assert!(check_request(32, capacity, &request).unwrap_err().stop, "fault {fault}");
    }
    let mut foreign = identities; foreign[0].task_spec = "sentiment-v1".to_owned();
    assert_eq!(check_request(32, capacity, &request(&foreign, capacity)).unwrap_err().fault.code, BatchCode::Admission);
}

#[test]
fn missing_reservations_output_capacity_and_arithmetic_fail_before_io() {
    for field in 0..11 {
        let (mut task, planning, mut mapping, mut limits) = fixture();
        match field { 0 => limits.preparation_reserve_bytes = 0, 1 => limits.io_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = 0, 3 => limits.preparation_reserve_bytes = u64::MAX,
            4 => limits.io_reserve_bytes = u64::MAX, 5 => limits.native.allocator_reserve_bytes = u64::MAX,
            6 => task.max_kv_bytes = 0, 7 => mapping.max_result_bytes = limits.transport.max_output_line_bytes as u64 + 1,
            8 => limits.transport.max_json_depth = 65, 9 => limits.native.context_tokens = 0,
            _ => limits.native.run.max_checkpoints = 1 }
        assert!(int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, None).is_err(),
            "field {field}");
    }
    let (task, planning, mapping, mut limits) = fixture();
    limits.transport.max_output_line_bytes = mapping.max_result_bytes as usize;
    int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, None).unwrap();
    assert!(int4_faithfulness_document_corpus_requirements(32, task, planning, mapping, limits, Some(1)).is_err());
}

#[test]
fn owned_stream_and_quote_are_available_on_the_existing_host_surface() {
    let _host = NlpEngine::batch_int4_faithfulness_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int4_faithfulness_documents_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _quote = crate::hosted::int4_faithfulness_document_corpus_requirements;
    let _capped_quote = crate::hosted::int4_faithfulness_document_corpus_requirements_with_max_heads;
    fn send<T: Send + 'static>() {}
    send::<StreamInput<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
}


#[test]
fn compact_population_quote_has_one_encoder_owner_across_all_chunk_and_head_bounds() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let policy = SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 32768,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    let encoder = policy.requirements().unwrap().workspace_bytes;
    let (task, planning, mapping, limits) = fixture();
    for heads in [1, 3, 32] {
        for chunks in [None, Some(1), Some(3), Some(64)] {
            let base = int4_faithfulness_document_corpus_requirements_with_max_heads(
                32, task, planning, mapping, limits, heads, chunks).unwrap();
            assert_eq!(base.with_input_encoding(&UntrustedEncoding::ByteFallback).unwrap(), base);
            let compact = base.with_input_encoding(&encoding).unwrap();
            assert_eq!(compact.input_bytes, base.input_bytes + encoder);
            assert_eq!(compact.native, base.native);
            assert_eq!(compact.workspace_bytes, base.workspace_bytes);
            assert_eq!(compact.output_reservation_bytes, base.output_reservation_bytes);
            assert_eq!(compact.max_output_bytes, base.max_output_bytes);
            let mut invalid = policy; invalid.max_pair_checks = 0;
            assert!(base.with_input_encoding(&UntrustedEncoding::SafeBpeV1(invalid)).is_err());
            let mut full = base; full.input_bytes = u64::MAX - encoder + 1;
            assert!(full.with_input_encoding(&encoding).is_err());
        }
    }
}
