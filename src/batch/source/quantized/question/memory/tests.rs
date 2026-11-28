//! Pure complete-ownership arithmetic, not a measured process RSS claim.
use super::*;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::Int8Work},
    tasks::{answer::AnswerOptions, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, quantized::long::Int8SourceMapLimits}},
    validation::grounded_fields::GroundingBudget};

fn fixture() -> (Int8QuestionDocumentMemoryLimits, Int8QuestionDocumentBatchConfig, BatchLimits) {
    let memory = Int8QuestionDocumentMemoryLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
        preparation_reserve_bytes: 32 * 1024 * 1024, io_reserve_bytes: 65_536, reduction_reserve_bytes: 65_536 };
    let work = Int8Work::for_sequence(0, 1024, 128 * NANBEIGE_VOCAB_SIZE).unwrap();
    let verification = GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 100_000 };
    let config = Int8QuestionDocumentBatchConfig { options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
            max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(128).unwrap().kv_bytes },
        planning: SourcePlanningLimits { max_input_bytes: 4096, max_context_tokens: 128, ..SourcePlanningLimits::default() },
        mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
                max_chunk_tokens: 64, context_tokens: 128, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 8, max_reduction_levels: 8,
                max_task_calls: 1024, max_value_bytes: 65_536, max_live_value_bytes: 131_072,
                max_total_value_bytes: 1_048_576, max_result_bytes: 131_072 },
            max_model_work: work, mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 8000 },
        verification, max_question_bytes: 32, max_model_work: work.checked_add(work).unwrap(), max_mask_visits: 16000,
        max_verification: GroundingBudget { max_fields: 64, max_matches: 256, max_scan_steps: 200_000 } };
    (memory, config, BatchLimits::default())
}

#[test]
fn complete_source_questions_and_all_logical_chunk_owners_coexist() {
    let (memory, mut config, transport) = fixture();
    let base = memory.requirements(config, transport).unwrap();
    config.mapping.chunks.max_chunks += 1;
    let more = memory.requirements(config, transport).unwrap();
    assert_eq!(more.native, base.native);
    assert_eq!(more.input_bytes - base.input_bytes,
        QUESTION_DOCUMENT_CHUNK_METADATA_BYTES + config.max_question_bytes as u64 * 64);
    assert_eq!(more.document_workspace_bytes - base.document_workspace_bytes,
        config.budget.max_output_bytes * 4 + u64::from(config.budget.max_output_tokens) * 8);
    config.mapping.chunks.max_input_bytes += 1;
    let source = memory.requirements(config, transport).unwrap(); assert_eq!(source.input_bytes - more.input_bytes, 64);
    config.max_question_bytes += 1;
    let question = memory.requirements(config, transport).unwrap();
    assert_eq!(question.input_bytes - source.input_bytes, (config.mapping.chunks.max_chunks as u64 + 1) * 64);
    assert_eq!(question.output_reservation_bytes, base.output_reservation_bytes);
    assert_eq!(question.max_output_tokens, 0);
}

#[test]
fn native_source_verification_and_lifted_question_citations_have_separate_payloads() {
    let (memory, mut config, transport) = fixture();
    let base = memory.requirements(config, transport).unwrap();
    config.planning.source.verification.max_fields += 1; config.planning.source.verification.max_matches += 1;
    let source = memory.requirements(config, transport).unwrap();
    assert_eq!(source.document_workspace_bytes - base.document_workspace_bytes,
        (std::mem::size_of::<SourceFieldEvidence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64);
    config.verification.max_fields += 1; config.verification.max_matches += 1;
    let qa = memory.requirements(config, transport).unwrap();
    assert_eq!(qa.document_workspace_bytes - source.document_workspace_bytes,
        (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()
            + std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64);
    assert_eq!(qa.output_reservation_bytes, 4 * qa.max_output_bytes + 4096);
    assert_eq!(qa.native, base.native);
}

#[test]
fn quote_refuses_every_missing_owner_context_mismatch_and_arithmetic_overflow() {
    let (memory, config, transport) = fixture();
    for axis in 0..5 {
        let mut m = memory;
        match axis { 0 => m.allocator_reserve_bytes = 0, 1 => m.preparation_reserve_bytes = 0,
            2 => m.io_reserve_bytes = 0, 3 => m.reduction_reserve_bytes = 0, _ => m.context_tokens += 1 }
        assert!(m.requirements(config, transport).is_err());
    }
    let mut too_large = memory; too_large.preparation_reserve_bytes = u64::MAX;
    assert!(too_large.requirements(config, transport).is_err());
    let mut wrong = config; wrong.budget.max_kv_bytes += 1;
    assert!(memory.requirements(wrong, transport).is_err());
    let mut insufficient = transport; insufficient.max_output_line_bytes = config.mapping.reduction.max_result_bytes - 1;
    assert!(memory.requirements(config, insufficient).is_err());
    let mut exhausted = config; exhausted.max_verification.max_fields = config.verification.max_fields - 1;
    assert!(memory.requirements(exhausted, transport).is_err());
}
