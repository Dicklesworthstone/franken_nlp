//! Metadata-only ownership/admission fixtures; no model execution claims.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::grouped_int4::long::question::synthesis::QuestionSynthesisLimits},
    validation::grounded_fields::GroundingBudget,
};
use std::{io::Cursor, time::Duration};

fn identity(width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"int4-question-corpus-host-fixture");
    ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap()), tokenizer_digest: d, template_digest: d, task_spec: ANSWER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }
}
fn config(width: usize, synthesis: bool) -> Int4QuestionDocumentBatchConfig {
    let row = constrained_int4::planned_work(width, 64, 64).unwrap();
    let mut item = row;
    for _ in 1..4 { item = item.checked_add(row).unwrap(); }
    let mut run = item;
    for _ in 1..4 { run = run.checked_add(item).unwrap(); }
    let grounding = GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 1_000_000 };
    let run_grounding = GroundingBudget { max_fields: 128, max_matches: 512, max_scan_steps: 4_000_000 };
    Int4QuestionDocumentBatchConfig {
        options: Default::default(),
        budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 64, max_output_bytes: 65_536,
            max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(128, width).unwrap().kv_bytes },
        planning: SourcePlanningLimits { max_input_bytes: 4096, max_context_tokens: 128,
            max_passages: 4, ..Default::default() },
        mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 128, max_chunk_tokens: 64,
                context_tokens: 128, reserved_tokens: 64, max_chunks: 3, ..Default::default() },
            reduction: ExecutionLimits { map_batch_chunks: 1, max_value_bytes: 65_536,
                max_live_value_bytes: 131_072, max_result_bytes: 262_144, ..Default::default() },
            max_model_work: item, mask_limits: Default::default(), mask_visits_per_chunk: 1000,
            max_mask_visits: 4000, verification: grounding,
        },
        verification: grounding,
        synthesis: synthesis.then_some(QuestionSynthesisLimits { max_evidence_passages: 4, max_evidence_bytes: 1024 }),
        max_question_bytes: 32, max_model_work: run, max_mask_visits: 16_000,
        max_map_verification: run_grounding, max_verification: run_grounding,
    }
}
fn limits() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } }, transport: crate::batch::BatchLimits::default(),
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 1 << 20 }
}
fn capacity(config: Int4QuestionDocumentBatchConfig, cohort: Option<usize>, width: usize) -> QuestionCapacity {
    let quote = int4_question_document_corpus_requirements(width, config, limits(), 1 << 20, cohort).unwrap();
    QuestionCapacity { config, kv_bytes: quote.native.kv_bytes, max_result_bytes: quote.max_output_bytes }
}
fn request<'a>(identities: &'a [ExecutionIdentity], capacity: QuestionCapacity,
    logical_chunks: usize) -> Int4QuestionDocumentAdmission<'a> {
    let contexts = identities.len() + usize::from(capacity.config.synthesis.is_some());
    let row = constrained_int4::planned_work(127, 64, 64).unwrap();
    let mut work = Int4Work::default();
    for _ in 0..contexts { work = work.checked_add(row).unwrap(); }
    Int4QuestionDocumentAdmission { identities, model_work: work,
        mask_node_visits: capacity.config.mapping.mask_visits_per_chunk * contexts as u64,
        mask_limits: capacity.config.mapping.mask_limits, kv_reservation_bytes: capacity.kv_bytes,
        max_result_bytes: capacity.max_result_bytes, chunk_count: logical_chunks, native_chunks: identities.len() }
}

#[test]
fn interior_whitespace_retains_logical_chunks_without_fabricating_native_identities() {
    let c = capacity(config(127, false), None, 127); let base = identity(127);
    let identities = [base.clone(), base.clone()];
    let exact = request(&identities, c, 3);
    check_request(127, &base, c, &exact).unwrap();
    for axis in 0..5 {
        let mut changed = request(&identities, c, 3);
        match axis { 0 => changed.native_chunks = 3, 1 => changed.native_chunks = 0,
            2 => changed.chunk_count = 1, 3 => changed.chunk_count = 4, _ => changed.chunk_count = 0 }
        let error = check_request(127, &base, c, &changed).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    }
}

#[test]
fn optional_synthesis_reserves_an_extra_independent_context_before_discovery() {
    let base = identity(127); let identities = [base.clone(), base.clone()];
    for synthesis in [false, true] {
        let c = capacity(config(127, synthesis), None, 127);
        let exact = request(&identities, c, 3);
        check_request(127, &base, c, &exact).unwrap();
        assert_eq!(exact.mask_node_visits, if synthesis { 3000 } else { 2000 });
        assert!(exact.model_work.forward_positions > limits().native.context_tokens as u64);
        let concatenated = Int4Work::for_sequence(127, 0, exact.model_work.forward_positions as usize,
            exact.model_work.projected_logits as usize).unwrap();
        assert_ne!(exact.model_work.attention_pairs, concatenated.attention_pairs);
        let mut missing = request(&identities, c, 3); missing.mask_node_visits -= 1000;
        assert_eq!(check_request(127, &base, c, &missing).unwrap_err().fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn every_chunk_must_have_the_fixed_task_profile_tokenizer_and_template() {
    let c = capacity(config(127, true), None, 127); let base = identity(127);
    for axis in 0..5 {
        let mut other = base.clone();
        match axis { 0 => other.task_spec = "ner-v1".to_owned(),
            1 => other.tokenizer_digest = Sha256Digest::of_bytes(b"other-tokenizer"),
            2 => other.template_digest = Sha256Digest::of_bytes(b"other-template"),
            3 => other.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            _ => other.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=128") }
        let identities = [base.clone(), other];
        assert_eq!(check_request(127, &base, c, &request(&identities, c, 3)).unwrap_err().fault.code,
            BatchCode::Admission);
    }
}

#[test]
fn full_native_peak_is_distinct_from_per_chunk_authority_and_semantic_result_size() {
    let mut config = config(127, true); config.mapping.reduction.map_batch_chunks = 3;
    let c = capacity(config, Some(3), 127); let base = identity(127); let identities = [base.clone()];
    assert_eq!(c.kv_bytes, config.budget.max_kv_bytes * 3);
    check_request(127, &base, c, &request(&identities, c, 1)).unwrap();
    let mut one_row = request(&identities, c, 1); one_row.kv_reservation_bytes = config.budget.max_kv_bytes;
    assert_eq!(check_request(127, &base, c, &one_row).unwrap_err().fault.code, BatchCode::Admission);
    let mut expanded = request(&identities, c, 1); expanded.max_result_bytes += 4096;
    assert_eq!(check_request(127, &base, c, &expanded).unwrap_err().fault.code, BatchCode::OutputLineLimit);
}

#[test]
fn all_native_work_and_mask_axes_are_independently_bounded() {
    let c = capacity(config(127, true), None, 127); let base = identity(127); let identities = [base.clone()];
    for axis in 0..8 {
        let mut changed = request(&identities, c, 1); let cap = c.config.mapping.max_model_work;
        match axis { 0 => changed.model_work.forward_positions = cap.forward_positions + 1,
            1 => changed.model_work.projected_logits = cap.projected_logits + NANBEIGE_VOCAB_SIZE as u64,
            2 => changed.model_work.attention_pairs = cap.attention_pairs + 1,
            3 => changed.model_work.projections.dot_products = cap.projections.dot_products + 1,
            4 => changed.model_work.projections.multiply_accumulates = cap.projections.multiply_accumulates + 1,
            5 => changed.mask_limits.max_trie_node_visits += 1,
            6 => changed.mask_limits.checkpoint_interval_nodes += 1,
            _ => changed.model_work.projected_logits += 1 }
        let error = check_request(127, &base, c, &changed).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn host_quote_uses_the_shared_memory_contract_and_refuses_invalid_context_before_io() {
    for synthesis in [false, true] {
        for cohort in [None, Some(1), Some(3)] {
            let mut config = config(127, synthesis); config.mapping.reduction.map_batch_chunks = cohort.unwrap_or(1);
            let host = limits();
            let expected = memory_limits(host, 1 << 20, cohort).requirements(127, config, host.transport).unwrap();
            assert_eq!(int4_question_document_corpus_requirements(127, config, host, 1 << 20, cohort).unwrap(), expected);
            assert_eq!(expected.max_output_bytes, config.mapping.reduction.max_result_bytes as u64);
            assert_eq!(expected.max_output_tokens, 0);
            let mut wrong = host; wrong.native.context_tokens -= 1;
            assert!(int4_question_document_corpus_requirements(127, config, wrong, 1 << 20, cohort).is_err());
            assert!(int4_question_document_corpus_requirements(127, config, host, 0, cohort).is_err());
        }
    }
}

#[test]
fn question_factory_vocabulary_and_both_stream_handles_cross_one_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4QuestionDocumentBatchPlanner>();
    send::<QuestionStreamInput<Cursor<Vec<u8>>, Vec<u8>>>();
    send::<Charged<QuestionStreamInput<Cursor<Vec<u8>>, Vec<u8>>>>();
    let _host = NlpEngine::batch_int4_question_documents::<Cursor<Vec<u8>>, Vec<u8>>;
}
