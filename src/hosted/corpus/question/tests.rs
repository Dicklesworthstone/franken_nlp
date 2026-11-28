//! Pure admission and simultaneous-ownership checks; no native model is loaded.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{answer::AnswerOptions, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, quantized::long::Int8SourceMapLimits}},
    validation::grounded_fields::GroundingBudget,
};

fn config() -> Int8QuestionDocumentBatchConfig {
    let work = constrained_int8::planned_work(8128, 64).unwrap();
    let item = work.checked_add(work).unwrap().checked_add(work).unwrap().checked_add(work).unwrap();
    let verification = GroundingBudget { max_fields: 32, max_matches: 64, max_scan_steps: 100_000 };
    Int8QuestionDocumentBatchConfig { options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(8192).unwrap().kv_bytes },
        planning: SourcePlanningLimits::default(),
        mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 1024,
                max_chunk_tokens: 1024, context_tokens: 8192, reserved_tokens: 64,
                max_chunks: 4, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits::default(), max_model_work: item,
            mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 4000 },
        verification, max_question_bytes: 1024, max_model_work: item.checked_add(item).unwrap(), max_mask_visits: 8000,
        max_verification: GroundingBudget { max_fields: 64, max_matches: 128, max_scan_steps: 200_000 } }
}
fn limits(config: Int8QuestionDocumentBatchConfig) -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 10_000, cleanup_reserve_bytes: 65_536 } },
        transport: BatchLimits { max_line_bytes: 8192, max_document_bytes: 4096,
            max_output_line_bytes: config.mapping.reduction.max_result_bytes + 4096,
            max_output_bytes: 128 << 20, ..BatchLimits::default() },
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 65_536 }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-host-question-admission");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: ANSWER_TASK_VERSION.to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn capacity() -> QuestionCapacity {
    let config = config();
    QuestionCapacity { kv_bytes: config.budget.max_kv_bytes,
        max_result_bytes: config.mapping.reduction.max_result_bytes as u64, max_output_tokens: 0, config }
}
fn request(identities: &[ExecutionIdentity]) -> Int8QuestionDocumentAdmission<'_> {
    let capacity = capacity(); let one = constrained_int8::planned_work(64, 64).unwrap();
    Int8QuestionDocumentAdmission { identities, model_work: one.checked_add(one).unwrap(),
        mask_node_visits: 2000, mask_limits: capacity.config.mapping.mask_limits,
        kv_reservation_bytes: capacity.kv_bytes, max_result_bytes: capacity.max_result_bytes,
        max_output_tokens: 0, chunk_count: 3, native_chunks: 2 }
}

#[test]
fn quote_prices_complete_document_and_transport_once_and_schedules_add_only_storage() {
    let config = config(); let limits = limits(config);
    let raw = Int8QuestionDocumentMemoryLimits { context_tokens: 8192, allocator_reserve_bytes: 4096,
        preparation_reserve_bytes: limits.preparation_reserve_bytes, io_reserve_bytes: limits.io_reserve_bytes,
        reduction_reserve_bytes: 65_536 }.requirements(config, limits.transport).unwrap();
    let base = int8_question_document_corpus_requirements(config, limits, 65_536, None, None).unwrap();
    assert_eq!(base.document, raw); assert_eq!(base.team_workspace_bytes, 0);
    let transport = limits.reservation_bytes().unwrap();
    assert!(raw.input_bytes > transport);
    let extra = raw.input_bytes - transport + raw.document_workspace_bytes;
    assert_eq!(transport + extra, raw.input_bytes + raw.document_workspace_bytes);
    assert_eq!(raw.output_reservation_bytes, raw.max_output_bytes * 4 + 4096);
    assert_eq!(raw.max_output_tokens, 0);
    let prefill = Int8PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap() };
    for width in [None, Some(2), Some(4), Some(64)] {
        let scheduled = int8_question_document_corpus_requirements(config, limits, 65_536, Some(prefill), width).unwrap();
        assert_eq!(scheduled.document.workspace_bytes, raw.workspace_bytes + prefill.validate().unwrap());
        assert_eq!(scheduled.document.input_bytes, raw.input_bytes);
        assert_eq!(scheduled.document.document_workspace_bytes, raw.document_workspace_bytes);
        assert_eq!(scheduled.document.output_reservation_bytes, raw.output_reservation_bytes);
        let team = width.map(|w| dispatch::int8_prefill_team_workspace_bytes(w, 4).unwrap()).unwrap_or(0);
        assert_eq!(scheduled.team_workspace_bytes, team);
        assert_eq!(scheduled.total_bytes().unwrap(), base.total_bytes().unwrap() + prefill.validate().unwrap() + team);
    }
}

#[test]
fn quote_refuses_missing_owners_wrong_capacity_and_underpriced_schedule() {
    for axis in 0..7 {
        let config = config(); let mut limits = limits(config);
        match axis { 0 => limits.preparation_reserve_bytes = 0, 1 => limits.io_reserve_bytes = 0,
            2 => limits.native.context_tokens -= 1, 3 => limits.native.allocator_reserve_bytes = 0,
            4 => limits.transport.max_document_bytes = config.mapping.chunks.max_input_bytes - 1,
            5 => limits.transport.max_output_line_bytes = config.mapping.reduction.max_result_bytes - 1,
            _ => limits.io_reserve_bytes = u64::MAX }
        assert!(int8_question_document_corpus_requirements(config, limits, 65_536, None, None).is_err(), "axis {axis}");
    }
    let config = config(); let limits = limits(config);
    assert!(int8_question_document_corpus_requirements(config, limits, 0, None, None).is_err());
    let short = Int8PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: 1 };
    assert!(int8_question_document_corpus_requirements(config, limits, 65_536, Some(short), None).is_err());
    for width in [0, 1, 65] {
        assert!(int8_question_document_corpus_requirements(config, limits, 65_536, None, Some(width)).is_err());
    }
}

#[test]
fn admission_keeps_blank_ranges_but_requires_every_native_child_and_exact_mask_geometry() {
    let base = identity(); let identities = [base.clone(), base.clone()];
    check_question_request(&base, capacity(), &request(&identities)).unwrap();
    for axis in 0..13 {
        let mut req = request(&identities);
        match axis { 0 => req.chunk_count = 0, 1 => req.chunk_count = 5,
            2 => req.native_chunks = 0, 3 => req.native_chunks = 3, 4 => req.chunk_count = 1,
            5 => req.kv_reservation_bytes -= 1, 6 => req.mask_node_visits -= 1,
            7 => req.mask_limits.max_trie_node_visits -= 1,
            8 => req.mask_limits.checkpoint_interval_nodes = 0,
            9 => req.model_work.projected_logits -= 1, 10 => req.model_work.forward_positions = 0,
            11 => req.max_result_bytes -= 1, _ => req.max_output_tokens = 1 }
        assert!(check_question_request(&base, capacity(), &req).is_err(), "axis {axis}");
    }
}

#[test]
fn every_native_work_axis_and_both_document_run_envelopes_are_authoritative() {
    let base = identity(); let identities = [base.clone(), base.clone()];
    for scope in [false, true] {
        for axis in 0..6 {
            let request = request(&identities); let mut c = capacity();
            let work = if scope { &mut c.config.max_model_work } else { &mut c.config.mapping.max_model_work };
            match axis { 0 => work.forward_positions = request.model_work.forward_positions - 1,
                1 => work.projected_logits = request.model_work.projected_logits - 1,
                2 => work.attention_pairs = request.model_work.attention_pairs - 1,
                3 => work.projections.dot_products = request.model_work.projections.dot_products - 1,
                4 => work.projections.multiply_accumulates = request.model_work.projections.multiply_accumulates - 1,
                _ => if scope { c.config.max_mask_visits = request.mask_node_visits - 1; }
                    else { c.config.mapping.max_mask_visits = request.mask_node_visits - 1; } }
            assert!(check_question_request(&base, c, &request).is_err(), "scope={scope} axis={axis}");
        }
    }
}

#[test]
fn altered_task_tokenizer_template_or_profile_cannot_enter_question_admission() {
    let base = identity();
    for axis in 0..4 {
        let mut other = base.clone();
        match axis { 0 => other.task_spec = "summarize-v1".to_owned(),
            1 => other.tokenizer_digest = Sha256Digest::of_bytes(b"other tokenizer"),
            2 => other.template_digest = Sha256Digest::of_bytes(b"other template"),
            _ => other.numerics_profile = NumericsProfile::StrictQuantized { version: 2 } }
        let identities = [base.clone(), other];
        assert!(check_question_request(&base, capacity(), &request(&identities)).is_err());
    }
    let mut wrong = base.clone(); wrong.task_spec = "summarize-v1".to_owned();
    let identities = [base.clone(), base];
    assert!(check_question_request(&wrong, capacity(), &request(&identities)).is_err());
}
