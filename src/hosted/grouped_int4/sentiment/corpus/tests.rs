//! Pure reservation and admission contracts. Synthetic profile identities
//! exercise refusal only; they cannot construct a native model or task plan.
use super::*;
use std::time::Duration;
use crate::{artifact::converter::PINNED_REVISION, batch::BatchLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy}};

fn fixture() -> (TaskBudget, SentimentLimits, Int4SentimentMapLimits, CorpusLimits) {
    let task = TaskBudget { max_input_tokens: 128, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(256, 32).unwrap().kv_bytes * 4 };
    let mapping = Int4SentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
        max_chunk_tokens: 128, context_tokens: 256, reserved_tokens: 16, max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 8 * 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: SentimentAggregationPolicy::default() };
    let limits = CorpusLimits { native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000, cleanup_reserve_bytes: 65_536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 128 * 1024 * 1024, io_reserve_bytes: 65_536 };
    (task, SentimentLimits::default(), mapping, limits)
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"sentiment-document-host-admission-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize)).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "sentiment-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "sentiment-complete-candidate-eos-v1".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn capacity(max_axes: usize, cohort_chunks: Option<usize>) -> DocumentCapacity {
    let (task, planning, mapping, limits) = fixture();
    let quote = int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, max_axes, cohort_chunks).unwrap();
    DocumentCapacity { context: limits.native.context_tokens, kv_bytes: quote.native.kv_bytes, mapping,
        max_axes, cohort_chunks, max_result_bytes: document_output_bytes(mapping, cohort_chunks).unwrap() }
}
fn request(identities: &[ExecutionIdentity], capacity: DocumentCapacity) -> Int4SentimentDocumentAdmission<'_> {
    let (mut task, _, mapping, _) = fixture();
    task.max_kv_bytes = Int4MemoryRequirement::for_context(capacity.context, 32).unwrap().kv_bytes * capacity.max_axes as u64;
    Int4SentimentDocumentAdmission { identities, model_work: mapping.max_model_work, group_width: 32,
        required_context: 128, head_count: capacity.max_axes, task_budget: task,
        kv_reservation_bytes: capacity.kv_bytes, max_result_bytes: capacity.max_result_bytes,
        cohort_chunks: capacity.cohort_chunks }
}

#[test]
fn serial_and_cohort_quotes_add_only_the_owned_stream_and_frozen_metadata() {
    let (task, planning, mapping, limits) = fixture();
    for width in [1, 3, 32, 128, 10752] {
        for (axes, chunks) in [(1, None), (4, None), (4, Some(1)), (4, Some(16)), (2, Some(32)), (1, Some(64))] {
            let base = map_memory(limits, axes, chunks).requirements(width, task, planning, mapping).unwrap();
            let actual = int4_sentiment_document_corpus_requirements(width, task, planning, mapping, limits, axes, chunks).unwrap();
            assert_eq!(actual.native, base.native);
            assert_eq!(actual.workspace_bytes, base.workspace_bytes);
            assert_eq!(actual.task_workspace_bytes, base.task_workspace_bytes);
            assert_eq!(actual.max_output_bytes, base.max_output_bytes);
            assert_eq!(actual.max_cohort_output_bytes, base.max_cohort_output_bytes);
            assert_eq!(actual.output_reservation_bytes, base.output_reservation_bytes);
            assert_eq!(actual.input_bytes, base.input_bytes + limits.reservation_bytes().unwrap()
                + mapping.chunks.max_chunks as u64 * SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
        }
    }
}

#[test]
fn fixed_axis_authority_refuses_oversized_cohorts_before_input_io() {
    let (task, planning, mapping, limits) = fixture();
    for (axes, chunks) in [(0, None), (5, None), (4, Some(0)), (4, Some(17)),
        (2, Some(33)), (1, Some(65)), (4, Some(usize::MAX))] {
        assert!(int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, axes, chunks).is_err());
    }
    // A caller that explicitly restricts overrides to one axis may use all
    // 64 native sequences; four unspecified future axes may not be clamped.
    int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 1, Some(64)).unwrap();
    assert!(int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 4, Some(64)).is_err());
    let mut mapping = mapping; mapping.chunks.max_chunks = 2;
    assert!(int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 1, Some(3)).is_err());
}

#[test]
fn missing_memory_authority_and_overflows_fail_before_model_or_source_io() {
    for field in 0..11 {
        let (mut task, planning, mut mapping, mut limits) = fixture();
        match field { 0 => limits.preparation_reserve_bytes = 0, 1 => limits.io_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = 0, 3 => limits.native.run.max_checkpoints = 1,
            4 => limits.preparation_reserve_bytes = u64::MAX, 5 => limits.io_reserve_bytes = u64::MAX,
            6 => limits.native.allocator_reserve_bytes = u64::MAX, 7 => task.max_kv_bytes -= 1,
            8 => mapping.max_result_bytes = limits.transport.max_output_line_bytes as u64 + 1,
            9 => limits.transport.max_json_depth = 65, _ => limits.native.context_tokens = 0 }
        assert!(int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 4, None).is_err(),
            "field {field}");
    }
    let (task, planning, mapping, limits) = fixture();
    for width in [0, 10753, usize::MAX] {
        assert!(int4_sentiment_document_corpus_requirements(width, task, planning, mapping, limits, 4, None).is_err());
    }
}

#[test]
fn larger_caller_buffers_preserve_the_same_native_peak() {
    let (task, planning, mapping, mut limits) = fixture();
    for cohorts in [None, Some(4)] {
        let base = int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 4, cohorts).unwrap();
        limits.io_reserve_bytes += 4096; limits.preparation_reserve_bytes += 8192;
        let larger = int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 4, cohorts).unwrap();
        assert_eq!(larger.input_bytes - base.input_bytes, 4096 + 2 * 8192);
        assert_eq!(larger.native, base.native); assert_eq!(larger.workspace_bytes, base.workspace_bytes);
    }
}

#[test]
fn complete_results_have_distinct_task_map_and_cohort_output_caps() {
    let identities = [identity(), identity()];
    let serial = capacity(4, None); let cohort = capacity(4, Some(2));
    let serial_request = request(&identities, serial); let cohort_request = request(&identities, cohort);
    assert!(serial_request.max_result_bytes > serial_request.task_budget.max_output_bytes);
    assert!(serial_request.model_work.forward_positions > serial.context as u64);
    assert_eq!(cohort_request.max_result_bytes, serial_request.max_result_bytes + 4096);
    check_request(32, serial, &serial_request).unwrap();
    check_request(32, cohort, &cohort_request).unwrap();
    let (task, planning, mapping, mut limits) = fixture();
    limits.transport.max_output_line_bytes = mapping.max_result_bytes as usize;
    int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 4, None).unwrap();
    assert!(int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, 4, Some(2)).is_err());
}

#[test]
fn per_chunk_axis_kv_and_simultaneous_window_kv_are_independent() {
    let identities = [identity(), identity()];
    let capacity = capacity(2, Some(3)); let mut request = request(&identities, capacity);
    assert_eq!(request.kv_reservation_bytes, request.task_budget.max_kv_bytes * 3);
    check_request(32, capacity, &request).unwrap();
    request.task_budget.max_kv_bytes -= 1;
    assert_eq!(check_request(32, capacity, &request).unwrap_err().fault.code, BatchCode::Admission);
    request.head_count = 1;
    check_request(32, capacity, &request).unwrap();
}

#[test]
fn all_original_chunk_identities_keep_the_sentiment_and_native_profile() {
    let capacity = capacity(4, None);
    for field in 0..6 {
        let mut identities = [identity(), identity()];
        match field { 0 => identities[1].task_spec = "classify-v1".to_owned(),
            1 => identities[1].thinking_mode = ThinkingMode::Enabled,
            2 => identities[1].tool_mode = ToolMode::Json,
            3 => identities[1].numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            4 => identities[1].kv_dtype = "int8".to_owned(),
            _ => identities[1].packing_set_digest = Sha256Digest::of_bytes(b"wrong grouping") }
        let error = check_request(32, capacity, &request(&identities, capacity)).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
    assert!(check_request(32, capacity, &request(&[], capacity)).is_err());
    let too_many = vec![identity(); capacity.mapping.chunks.max_chunks + 1];
    assert!(check_request(32, capacity, &request(&too_many, capacity)).is_err());
}

#[test]
fn every_native_work_axis_and_schedule_owner_is_checked_at_admission() {
    let identities = [identity(), identity()]; let capacity = capacity(2, Some(3));
    for axis in 0..5 {
        let mut request = request(&identities, capacity);
        match axis { 0 => request.model_work.forward_positions += 1, 1 => request.model_work.projected_logits += 1,
            2 => request.model_work.attention_pairs += 1, 3 => request.model_work.projections.dot_products += 1,
            _ => request.model_work.projections.multiply_accumulates += 1 }
        let error = check_request(32, capacity, &request).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
    for fault in 0..10 {
        let mut request = request(&identities, capacity);
        match fault { 0 => request.cohort_chunks = None, 1 => request.cohort_chunks = Some(2),
            2 => request.head_count = 0, 3 => request.head_count = 3, 4 => request.kv_reservation_bytes -= 1,
            5 => request.required_context = 257, 6 => request.max_result_bytes = capacity.mapping.max_result_bytes,
            7 => request.max_result_bytes += 1, 8 => request.group_width = 128, _ => request.required_context = 0 }
        assert!(check_request(32, capacity, &request).unwrap_err().stop, "fault {fault}");
    }
}

#[test]
fn owned_stream_and_pure_quote_use_the_existing_host_surface() {
    let _host = NlpEngine::batch_int4_sentiment_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _quote = crate::hosted::int4_sentiment_document_corpus_requirements;
    fn send<T: Send + 'static>() {}
    send::<StreamInput<std::io::Cursor<Vec<u8>>, Vec<u8>>>();
}

#[test]
fn compact_corpus_workspace_is_charged_once_independently_of_axes_and_chunk_cohorts() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let policy = SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 131_072,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    let workspace = policy.requirements().unwrap().workspace_bytes;
    let (task, planning, mapping, limits) = fixture();
    for (axes, chunks) in [(1, None), (4, None), (4, Some(2)), (1, Some(64))] {
        let byte = int4_sentiment_document_corpus_requirements(32, task, planning, mapping, limits, axes, chunks).unwrap();
        let compact = byte.with_input_encoding(&encoding).unwrap();
        assert_eq!(compact.input_bytes - byte.input_bytes, workspace);
        assert_eq!(compact.native, byte.native);
        assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
        assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
        assert_eq!(byte.with_input_encoding(&UntrustedEncoding::ByteFallback).unwrap(), byte);
        let mut overflow = byte; overflow.input_bytes = u64::MAX;
        assert!(overflow.with_input_encoding(&encoding).is_err());
    }
}
