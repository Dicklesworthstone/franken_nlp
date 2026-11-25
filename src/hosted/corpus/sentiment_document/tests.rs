//! Pure corpus quote/admission tests. Synthetic identities exercise refusal
//! and ownership contracts, never native numerical or task-quality evidence.
use super::*;
use std::time::Duration;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{portable_int8::ProjectionWork, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy},
    tokenizer::untrusted::SafeBpeLimits,
};

fn fixture() -> (TaskBudget, SentimentLimits, Int8SentimentMapLimits, CorpusLimits) {
    let native = Int8MemoryRequirement::for_context(256).unwrap();
    let task = TaskBudget { max_input_tokens: 128, max_output_tokens: 16,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes * 4 };
    let mapping = Int8SentimentMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
            max_chunk_tokens: 128, context_tokens: 256, reserved_tokens: 16,
            max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int8Work { forward_positions: 1024, projected_logits: 1_000_000,
            attention_pairs: 1_000_000,
            projections: ProjectionWork { dot_products: 1_000_000, multiply_accumulates: 1_000_000 } },
        max_retained_result_bytes: 8 * 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: SentimentAggregationPolicy::default(),
    };
    let limits = CorpusLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(10),
                max_checkpoints: 1000, cleanup_reserve_bytes: 65_536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 128 * 1024 * 1024,
        io_reserve_bytes: 65_536,
    };
    (task, SentimentLimits::default(), mapping, limits)
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"int8 sentiment document corpus host admission fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: digest, artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(),
        packing_set_digest: digest, tokenizer_digest: digest, template_digest: digest,
        task_spec: "sentiment-v1".to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "none".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "sentiment-complete-candidate-eos-v1".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn model() -> ArtifactIdentity {
    let identity = identity();
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: identity.source_revision,
        recipe_id: identity.quant_recipe, source_root_sha256: identity.logical_model_digest.to_hex(),
        logical_model_sha256: identity.logical_model_digest.to_hex() }
}
fn request(identities: &[ExecutionIdentity], heads: usize) -> Int8SentimentDocumentAdmission<'_> {
    let (mut task, _, mapping, limits) = fixture();
    let kv = Int8MemoryRequirement::for_context(limits.native.context_tokens).unwrap().kv_bytes;
    task.max_kv_bytes = kv * heads as u64;
    Int8SentimentDocumentAdmission { identities, model_work: mapping.max_model_work,
        required_context: 128, head_count: heads, task_budget: task,
        kv_reservation_bytes: kv, max_result_bytes: mapping.max_result_bytes }
}
fn check(request: &Int8SentimentDocumentAdmission<'_>) -> Result<(), BatchItemFailure> {
    validate_admission(&model(), Int8MemoryRequirement::for_context(256).unwrap().kv_bytes,
        fixture().3.transport.max_output_line_bytes as u64, request)
}

#[test]
fn quote_adds_all_transport_and_frozen_metadata_beside_complete_map_ownership() {
    let (task, planning, mapping, limits) = fixture();
    let base = Int8SentimentMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes, max_axes: 4 }
        .requirements(task, planning, mapping).unwrap();
    let actual = int8_sentiment_document_corpus_requirements(
        task, planning, mapping, &UntrustedEncoding::ByteFallback, limits).unwrap();
    assert_eq!(actual.native, base.native);
    assert_eq!(actual.workspace_bytes, base.workspace_bytes);
    assert_eq!(actual.task_workspace_bytes, base.task_workspace_bytes);
    assert_eq!(actual.max_output_bytes, base.max_output_bytes);
    assert_eq!(actual.output_reservation_bytes, base.output_reservation_bytes);
    assert_eq!(actual.input_bytes, base.input_bytes + limits.reservation_bytes().unwrap()
        + mapping.chunks.max_chunks as u64 * INT8_SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
    assert_eq!(actual.native.kv_bytes * 4, task.max_kv_bytes,
        "the logical axis allowance must not multiply the physical native owner");
}

#[test]
fn compact_workspace_is_admitted_once_beside_all_complete_chunk_plans() {
    let (task, planning, mapping, limits) = fixture();
    let policy = SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 131_072,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let byte = int8_sentiment_document_corpus_requirements(
        task, planning, mapping, &UntrustedEncoding::ByteFallback, limits).unwrap();
    let compact = int8_sentiment_document_corpus_requirements(
        task, planning, mapping, &UntrustedEncoding::SafeBpeV1(policy), limits).unwrap();
    assert_eq!(compact.input_bytes - byte.input_bytes, policy.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, byte.native);
    assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
    assert_eq!(compact.task_workspace_bytes, byte.task_workspace_bytes);
    assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
}

#[test]
fn missing_capacity_authority_and_overflow_fail_before_private_io() {
    for field in 0..15 {
        let (mut task, mut planning, mut mapping, mut limits) = fixture();
        match field {
            0 => limits.preparation_reserve_bytes = 0,
            1 => limits.io_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = 0,
            3 => limits.native.run.max_checkpoints = 1,
            4 => limits.preparation_reserve_bytes = u64::MAX,
            5 => limits.io_reserve_bytes = u64::MAX,
            6 => limits.native.allocator_reserve_bytes = u64::MAX,
            7 => task.max_kv_bytes -= 1,
            8 => mapping.max_result_bytes = limits.transport.max_output_line_bytes as u64 + 1,
            9 => limits.transport.max_json_depth = 65,
            10 => limits.native.context_tokens = 0,
            11 => limits.native.context_tokens = 255,
            12 => mapping.chunks.max_chunks = 257,
            13 => mapping.max_model_work.forward_positions = u64::MAX,
            _ => planning.max_total_prompt_tokens = 0,
        }
        assert!(int8_sentiment_document_corpus_requirements(task, planning, mapping,
            &UntrustedEncoding::ByteFallback, limits).is_err(), "field {field}");
    }
}

#[test]
fn caller_buffers_change_only_the_corresponding_non_native_owners() {
    let (task, planning, mapping, mut limits) = fixture();
    let before = int8_sentiment_document_corpus_requirements(
        task, planning, mapping, &UntrustedEncoding::ByteFallback, limits).unwrap();
    limits.preparation_reserve_bytes += 8192;
    limits.io_reserve_bytes += 4096;
    let after = int8_sentiment_document_corpus_requirements(
        task, planning, mapping, &UntrustedEncoding::ByteFallback, limits).unwrap();
    assert_eq!(after.input_bytes - before.input_bytes, 2 * 8192 + 4096);
    assert_eq!(after.native, before.native);
    assert_eq!(after.workspace_bytes, before.workspace_bytes);
    assert_eq!(after.output_reservation_bytes, before.output_reservation_bytes);
}

#[test]
fn complete_outputs_and_selected_head_kv_have_separate_correct_bounds() {
    let identities = [identity(), identity()];
    for heads in 1..=4 {
        let request = request(&identities, heads);
        assert!(request.max_result_bytes > request.task_budget.max_output_bytes);
        assert!(request.model_work.forward_positions > request.required_context as u64);
        check(&request).unwrap();
        let mut too_little = request;
        too_little.task_budget.max_kv_bytes -= 1;
        assert_eq!(check(&too_little).unwrap_err().fault.code, BatchCode::Admission);
    }
}

#[test]
fn all_chunk_identities_are_checked_against_model_profile_and_task() {
    for index in 0..2 {
        for field in 0..7 {
            let mut identities = [identity(), identity()];
            match field {
                0 => identities[index].task_spec = "classification-v1".to_owned(),
                1 => identities[index].source_revision.push('x'),
                2 => identities[index].quant_recipe.push('x'),
                3 => identities[index].logical_model_digest = Sha256Digest::of_bytes(b"other model"),
                4 => identities[index].numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
                5 => identities[index].thinking_mode = ThinkingMode::Enabled,
                _ => identities[index].kv_dtype = "f32".to_owned(),
            }
            let error = check(&request(&identities, 4)).unwrap_err();
            assert!(error.stop);
            assert_eq!(error.fault.code, BatchCode::Admission);
        }
    }
    assert!(check(&request(&[], 4)).is_err());
    let too_many = vec![identity(); 257];
    assert!(check(&request(&too_many, 4)).is_err());
}

#[test]
fn zero_native_counters_and_invalid_geometry_never_reach_output_allocation() {
    let identities = [identity(), identity()];
    for axis in 0..5 {
        let mut request = request(&identities, 4);
        match axis {
            0 => request.model_work.forward_positions = 0,
            1 => request.model_work.projected_logits = 0,
            2 => request.model_work.attention_pairs = 0,
            3 => request.model_work.projections.dot_products = 0,
            _ => request.model_work.projections.multiply_accumulates = 0,
        }
        let error = check(&request).unwrap_err();
        assert!(error.stop);
        assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
    for field in 0..6 {
        let mut request = request(&identities, 4);
        match field {
            0 => request.head_count = 0,
            1 => request.head_count = 5,
            2 => request.required_context = 0,
            3 => request.required_context = 257,
            4 => request.kv_reservation_bytes -= 1,
            _ => request.task_budget.max_output_tokens = 0,
        }
        assert_eq!(check(&request).unwrap_err().fault.code, BatchCode::Admission);
    }
    for bytes in [0, fixture().3.transport.max_output_line_bytes as u64 + 1] {
        let mut request = request(&identities, 4);
        request.max_result_bytes = bytes;
        let error = check(&request).unwrap_err();
        assert!(!error.stop);
        assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
    }
}

#[test]
fn projected_rows_are_not_reinterpreted_as_generated_tokens_or_score_space() {
    let identities = [identity()];
    for rows in [1, 3, 17, 166_105, 332_210] {
        let mut request = request(&identities, 1);
        request.model_work.projected_logits = rows;
        check(&request).unwrap();
    }
}

#[test]
fn owned_stream_and_all_execution_schedules_have_the_public_host_surface() {
    let _serial = NlpEngine::batch_int8_sentiment_documents::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _prefill = NlpEngine::batch_int8_sentiment_documents_layer_major::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _team = NlpEngine::batch_int8_sentiment_documents_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _combined = NlpEngine::batch_int8_sentiment_documents_layer_major_with_threads::<std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _quote = crate::hosted::corpus::int8_sentiment_document_corpus_requirements;
    fn send<T: Send + 'static>() {}
    send::<SentimentDocumentCorpusConfig>();
    send::<SentimentDocumentStream>();
    send::<StreamInput<SentimentDocumentStream, std::io::Cursor<Vec<u8>>, Vec<u8>>>();
}
