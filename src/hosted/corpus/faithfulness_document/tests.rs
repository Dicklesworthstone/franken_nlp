//! Pure simultaneous-owner quote and complete-document admission regressions.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::corpus::scored::CorpusTask,
    native_engine::{portable_int8::ProjectionWork, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{judge::FaithfulnessAggregationPolicy, mapreduce::ChunkLimits},
    tokenizer::{pinned_controls, embedded::EmbeddedTokenizer, untrusted::SafeBpeLimits},
};
use std::time::Duration;

fn fixture() -> (TaskBudget, JudgeLimits, Int8FaithfulnessMapLimits, CorpusLimits) {
    let task = TaskBudget { max_input_tokens: 4096, max_output_tokens: 8, max_output_bytes: 1_000_000,
        max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(1024).unwrap().kv_bytes * 32 };
    let mapping = Int8FaithfulnessMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 128, max_chunk_tokens: 1024,
            context_tokens: 1024, reserved_tokens: 1, max_chunks: 8, max_tokenizer_calls: 1024 },
        max_model_work: Int8Work { forward_positions: 1_000_000, projected_logits: 1_000_000_000_000,
            attention_pairs: 1_000_000_000_000,
            projections: ProjectionWork { dot_products: 1_000_000_000_000,
                multiply_accumulates: 1_000_000_000_000_000_000 } },
        max_retained_result_bytes: 4 << 20, max_result_bytes: 8 << 20,
        aggregation: FaithfulnessAggregationPolicy::default(),
    };
    let limits = CorpusLimits {
        native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000,
                cleanup_reserve_bytes: 65536 } },
        transport: BatchLimits { max_document_bytes: 4096, max_output_line_bytes: 16 << 20,
            ..Default::default() },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 65536,
    };
    (task, JudgeLimits::default(), mapping, limits)
}
fn planner() -> Arc<JudgePlanner> {
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    Arc::new(JudgePlanner::pinned(controls.template_controls(), eos).unwrap())
}
fn config(planner: &JudgePlanner) -> FaithfulnessDocumentCorpusConfig {
    let (task, planning, mapping, _) = fixture();
    let absent = Sha256Digest::of_bytes(b"synthetic-int8-faithfulness-corpus");
    let identity = ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1, source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: absent, artifact_format: "synthetic-only".to_owned(),
        quant_recipe: "fixture-int8".to_owned(), packing_set_digest: absent,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "judge-v1".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }).unwrap();
    FaithfulnessDocumentCorpusConfig { identity, task_ceiling: task, planning, mapping,
        policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_milli: 100,
            evidence_window_bytes: 8, max_evidence_windows: 31, max_evidence_spans: 31 },
        max_model_work: mapping.max_model_work, max_tokenizer_work: None }
}

#[test]
fn complete_quote_adds_transport_original_pairs_and_retained_chunk_identities_once() {
    let (task, planning, mapping, limits) = fixture();
    let base = Int8FaithfulnessMapHostConfig { native: limits.native, planning, mapping,
        preparation_reserve_bytes: limits.preparation_reserve_bytes }.requirements(task).unwrap();
    let corpus = int8_faithfulness_document_corpus_requirements(task, planning, mapping,
        &UntrustedEncoding::ByteFallback, limits).unwrap();
    assert_eq!(corpus.input_bytes, base.input_bytes + limits.reservation_bytes().unwrap()
        + mapping.chunks.max_chunks as u64 * FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES);
    assert_eq!(corpus.native, base.native); assert_eq!(corpus.workspace_bytes, base.workspace_bytes);
    assert_eq!(corpus.output_reservation_bytes, mapping.max_result_bytes * 8 + 4096);
    assert_eq!(corpus.output_reservation_bytes, base.output_reservation_bytes);
    assert!(corpus.output_reservation_bytes > task.max_output_bytes * 8);
}

#[test]
fn source_chunks_and_evidence_heads_share_one_physical_cache_and_coordinator() {
    let (task, planning, mut mapping, limits) = fixture();
    let mut previous = 0;
    for chunks in [1, 8, 64] {
        mapping.chunks.max_chunks = chunks;
        let required = int8_faithfulness_document_corpus_requirements(task, planning, mapping,
            &UntrustedEncoding::ByteFallback, limits).unwrap();
        assert_eq!(required.native, Int8MemoryRequirement::for_context(1024).unwrap());
        assert_eq!(required.workspace_bytes, required.native.rope_bytes
            + required.native.scratch_payload_bound + limits.native.allocator_reserve_bytes);
        assert!(required.input_bytes > previous); previous = required.input_bytes;
    }
}

#[test]
fn compact_and_prefill_workspace_add_to_every_retained_source_and_evidence_owner() {
    let (task, planning, mapping, limits) = fixture();
    let policy = SafeBpeLimits { max_input_bytes: 262144, max_output_tokens: 131072,
        max_pair_checks: 1 << 25, max_pair_bytes: 1 << 28 };
    let plain = int8_faithfulness_document_corpus_requirements(task, planning, mapping,
        &UntrustedEncoding::ByteFallback, limits).unwrap();
    let compact = int8_faithfulness_document_corpus_requirements(task, planning, mapping,
        &UntrustedEncoding::SafeBpeV1(policy), limits).unwrap();
    assert_eq!(compact.input_bytes, plain.input_bytes + policy.requirements().unwrap().workspace_bytes);
    assert_eq!(compact.native, plain.native); assert_eq!(compact.output_reservation_bytes, plain.output_reservation_bytes);
    let prefill = Int8PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap() };
    let packed = compact.with_prefill(prefill).unwrap();
    assert_eq!(packed.workspace_bytes, compact.workspace_bytes + prefill.validate().unwrap());
    assert_eq!(packed.input_bytes, compact.input_bytes); assert_eq!(packed.native, compact.native);
}

#[test]
fn invalid_map_geometry_transport_and_overflow_refuse_before_io() {
    let (task, planning, mapping, limits) = fixture();
    for axis in 0..7 {
        let mut selected = limits;
        match axis {
            0 => selected.transport.max_output_line_bytes = mapping.max_result_bytes as usize - 1,
            1 => selected.io_reserve_bytes = 0,
            2 => selected.preparation_reserve_bytes = u64::MAX,
            3 => selected.native.context_tokens = mapping.chunks.context_tokens - 1,
            4 => selected.transport.max_epoch_ids = usize::MAX,
            5 => selected.transport.max_line_bytes = usize::MAX,
            _ => selected.native.allocator_reserve_bytes = 0,
        }
        assert!(int8_faithfulness_document_corpus_requirements(task, planning, mapping,
            &UntrustedEncoding::ByteFallback, selected).is_err(), "axis {axis}");
    }
}

#[test]
fn fixed_policy_refuses_early_and_actual_compiler_owner_survives_the_consumed_callback() {
    let planner = planner(); let mut settings = config(&planner);
    settings.validate(&planner).unwrap();
    settings.policy.max_evidence_windows = 32;
    assert!(settings.validate(&planner).is_err());
    let settings = config(&planner);
    let compiler = Int8FaithfulnessDocumentBatchPlanner::new(Arc::clone(&planner), settings.identity,
        settings.task_ceiling, settings.planning, settings.mapping, settings.policy,
        settings.max_model_work, settings.max_tokenizer_work).unwrap();
    let stream = FaithfulnessDocumentStream { compiler: Arc::new(compiler) };
    let weak = Arc::downgrade(&stream.compiler);
    let retained = stream.retain_for_join().unwrap();
    drop(stream);
    assert!(weak.upgrade().is_some());
    drop(retained);
    assert!(weak.upgrade().is_none());
}

#[test]
fn complete_document_admission_checks_fixed_policy_geometry_and_all_five_native_axes() {
    let (task, _, mapping, limits) = fixture();
    let planner = planner(); let identity = config(&planner).identity;
    let identities = [identity];
    let kv = Int8MemoryRequirement::for_context(limits.native.context_tokens).unwrap().kv_bytes;
    for axis in 0..16 {
        let mut request = Int8FaithfulnessDocumentAdmission { identities: &identities,
            model_work: mapping.max_model_work, required_context: 768, task_budget: task,
            max_head_count: 3, kv_reservation_bytes: kv, max_result_bytes: mapping.max_result_bytes };
        check_capacity(mapping, task, kv, &request).unwrap();
        match axis {
            0 => request.identities = &[],
            1 => request.required_context = mapping.chunks.context_tokens + 1,
            2 => request.max_head_count = MAX_FAITHFULNESS_MAP_HEADS + 1,
            3 => request.kv_reservation_bytes -= 1,
            4 => request.max_result_bytes -= 1,
            5 => request.task_budget.max_kv_bytes -= 1,
            6 => request.task_budget.max_input_tokens += 1,
            7 => request.task_budget.max_output_bytes += 1,
            8 => request.task_budget.max_output_tokens += 1,
            9 => request.model_work.forward_positions += 1,
            10 => request.model_work.projected_logits += 1,
            11 => request.model_work.attention_pairs += 1,
            12 => request.model_work.projections.dot_products += 1,
            13 => request.model_work.projections.multiply_accumulates += 1,
            14 => request.max_head_count = 0,
            _ => request.required_context = 0,
        }
        assert_eq!(check_capacity(mapping, task, kv, &request).unwrap_err().fault.code,
            BatchCode::Admission, "axis {axis}");
    }
}
