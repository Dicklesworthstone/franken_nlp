//! Pure ownership/admission fixtures; no model execution or answer quality.
use super::*;
use crate::{
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{answer::AnswerOptions, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, grouped_int4::long::{Int4SourceMapLimits,
            question::synthesis::QuestionSynthesisLimits}}},
    validation::grounded_fields::GroundingBudget,
};

fn fixture() -> (Int4QuestionDocumentMemoryLimits, Int4QuestionDocumentBatchConfig, BatchLimits) {
    let memory = Int4QuestionDocumentMemoryLimits { context_tokens: 128,
        allocator_reserve_bytes: 65_536, preparation_reserve_bytes: 32 * 1024 * 1024,
        io_reserve_bytes: 65_536, reduction_reserve_bytes: 65_536, cohort_chunks: None };
    let work = Int4Work::for_sequence(32, 0, 1024, 128 * NANBEIGE_VOCAB_SIZE).unwrap();
    let verification = GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 100_000 };
    let run_verification = GroundingBudget { max_fields: 64, max_matches: 256, max_scan_steps: 200_000 };
    let config = Int4QuestionDocumentBatchConfig {
        options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16,
            max_output_bytes: 4096, max_grammar_states: 4096,
            max_kv_bytes: Int4MemoryRequirement::for_context(128, 32).unwrap().kv_bytes },
        planning: SourcePlanningLimits { max_input_bytes: 4096, max_context_tokens: 128,
            ..SourcePlanningLimits::default() },
        mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 64,
                context_tokens: 128, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 8, max_reduction_levels: 8,
                max_task_calls: 1024, max_value_bytes: 65_536, max_live_value_bytes: 131_072,
                max_total_value_bytes: 1_048_576, max_result_bytes: 131_072 },
            max_model_work: work, mask_limits: Default::default(), mask_visits_per_chunk: 1000,
            max_mask_visits: 8000, verification },
        verification, synthesis: None, max_question_bytes: 32,
        max_model_work: work.checked_add(work).unwrap(), max_mask_visits: 16_000,
        max_map_verification: run_verification, max_verification: run_verification,
    };
    (memory, config, BatchLimits::default())
}
fn synthesize(config: &mut Int4QuestionDocumentBatchConfig) {
    config.synthesis = Some(QuestionSynthesisLimits { max_evidence_passages: 4, max_evidence_bytes: 512 });
}

#[test]
fn complete_source_questions_and_ordered_identities_remain_charged_beyond_the_native_window() {
    let (mut memory, mut config, transport) = fixture(); memory.cohort_chunks = Some(3);
    let original = memory.requirements(32, config, transport).unwrap();
    config.mapping.chunks.max_chunks += 1;
    let longer = memory.requirements(32, config, transport).unwrap();
    assert_eq!(longer.native, original.native);
    assert_eq!(longer.input_bytes - original.input_bytes,
        QUESTION_DOCUMENT_CHUNK_METADATA_BYTES + config.max_question_bytes as u64 * 64);
    assert_eq!(longer.document_workspace_bytes - original.document_workspace_bytes,
        config.budget.max_output_bytes * 4 + u64::from(config.budget.max_output_tokens) * 8);
    assert_eq!(longer.output_reservation_bytes, original.output_reservation_bytes);
    config.mapping.chunks.max_input_bytes += 1;
    let more_source = memory.requirements(32, config, transport).unwrap();
    assert_eq!(more_source.input_bytes - longer.input_bytes, 64);
    config.max_question_bytes += 1;
    let more_question = memory.requirements(32, config, transport).unwrap();
    assert_eq!(more_question.input_bytes - more_source.input_bytes,
        (config.mapping.chunks.max_chunks as u64 + 1) * 64);
    assert_eq!(more_question.native, original.native);
}

#[test]
fn discovery_and_flat_synthesis_reuse_the_native_peak_without_expanding_result_authority() {
    for width in [3, 32, 127, 128, 10752] {
        for chunks in [1, 3] {
            let (mut memory, mut config, transport) = fixture();
            memory.cohort_chunks = Some(chunks); config.mapping.reduction.map_batch_chunks = chunks;
            let work = Int4Work::for_sequence(width, 0, 1024, 128 * NANBEIGE_VOCAB_SIZE).unwrap();
            config.mapping.max_model_work = work; config.max_model_work = work.checked_add(work).unwrap();
            let independent = memory.requirements(width, config, transport).unwrap();
            synthesize(&mut config);
            let synthesized = memory.requirements(width, config, transport).unwrap();
            let row = Int4MemoryRequirement::for_context(memory.context_tokens, width).unwrap();
            assert_eq!(synthesized.native,
                Int4MemoryRequirement::for_cohort(&vec![memory.context_tokens; chunks], width).unwrap());
            assert_eq!(synthesized.native, independent.native);
            assert_eq!(config.budget.max_kv_bytes, row.kv_bytes);
            assert_eq!(synthesized.max_output_bytes, independent.max_output_bytes);
            assert_eq!(synthesized.output_reservation_bytes, independent.output_reservation_bytes);
            assert_eq!(synthesized.max_output_tokens, 0);
            assert_eq!(synthesized.input_bytes - independent.input_bytes,
                QUESTION_DOCUMENT_CHUNK_METADATA_BYTES + config.max_question_bytes as u64 * 64);
            assert!(synthesized.document_workspace_bytes > independent.document_workspace_bytes);
            let quote = memory.cohort_requirements(width, config).unwrap().unwrap();
            assert_eq!(quote.native, synthesized.native);
            let serial = Int4QuestionDocumentMemoryLimits { cohort_chunks: None, ..memory }
                .requirements(width, config, transport).unwrap();
            assert_eq!(serial.native, row);
            assert_eq!(synthesized.document_workspace_bytes - serial.document_workspace_bytes,
                quote.workspace_bytes + 4 * COHORT_ENVELOPE_BYTES);
            if chunks > 1 {
                let mut widened = config; widened.budget.max_kv_bytes = synthesized.native.kv_bytes;
                assert!(memory.requirements(width, widened, transport).is_err());
            }
        }
    }
}

#[test]
fn all_evidence_passages_origin_fanout_and_temporary_transcripts_are_priced() {
    let (memory, mut config, transport) = fixture(); synthesize(&mut config);
    let first = memory.requirements(32, config, transport).unwrap();
    config.synthesis.as_mut().unwrap().max_evidence_bytes += 1;
    let bytes = memory.requirements(32, config, transport).unwrap();
    assert_eq!(bytes.document_workspace_bytes - first.document_workspace_bytes, 64);
    config.synthesis.as_mut().unwrap().max_evidence_passages += 1;
    let passages = memory.requirements(32, config, transport).unwrap();
    assert_eq!(passages.document_workspace_bytes - bytes.document_workspace_bytes, 1024);
    config.verification.max_fields += 1; config.verification.max_matches += 1;
    let evidence = memory.requirements(32, config, transport).unwrap();
    assert_eq!(evidence.document_workspace_bytes - passages.document_workspace_bytes,
        (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()
            + std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64
            + 128 + 128);
    config.budget.max_output_tokens += 1; config.mapping.chunks.reserved_tokens += 1;
    let transcript = memory.requirements(32, config, transport).unwrap();
    assert_eq!(transcript.document_workspace_bytes - evidence.document_workspace_bytes,
        8 * (config.mapping.chunks.max_chunks as u64 + 1));
    // Final semantic QA contains no token arrays, even though native phases do.
    assert_eq!(transcript.output_reservation_bytes, first.output_reservation_bytes);
    assert_eq!(transcript.output_reservation_bytes, 4 * transcript.max_output_bytes + 4096);
    assert_eq!(transcript.max_output_tokens, 0);
}

#[test]
fn map_and_question_verification_pools_have_independent_ownership_and_run_limits() {
    let (memory, mut config, transport) = fixture();
    let first = memory.requirements(32, config, transport).unwrap();
    config.mapping.verification.max_fields += 1; config.mapping.verification.max_matches += 1;
    let map = memory.requirements(32, config, transport).unwrap();
    assert_eq!(map.document_workspace_bytes - first.document_workspace_bytes,
        (std::mem::size_of::<SourceFieldEvidence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64);
    assert_eq!(map.native, first.native); assert_eq!(map.input_bytes, first.input_bytes);
    let mut map_short = config; map_short.max_map_verification.max_fields = config.mapping.verification.max_fields - 1;
    assert!(memory.requirements(32, map_short, transport).is_err());
    let mut qa_short = config; qa_short.max_verification.max_matches = config.verification.max_matches - 1;
    assert!(memory.requirements(32, qa_short, transport).is_err());
    config.max_verification.max_scan_steps += 1;
    config.max_map_verification.max_scan_steps += 1;
    assert_eq!(memory.requirements(32, config, transport).unwrap(), map);
}

#[test]
fn valid_fixed_answer_options_change_policy_without_inventing_a_new_memory_permit() {
    let (memory, mut config, transport) = fixture();
    let original = memory.requirements(32, config, transport).unwrap();
    config.options = AnswerOptions { max_answer_scalars: 64, max_citations: 4, max_quote_scalars: 32 };
    assert_eq!(memory.requirements(32, config, transport).unwrap(), original);
    config.options.max_citations = 0;
    assert!(memory.requirements(32, config, transport).is_err());
}

#[test]
fn complete_schedule_question_context_transport_and_evidence_limits_refuse_before_io() {
    for fault in 0..21 {
        let (mut memory, mut config, mut transport) = fixture();
        match fault {
            0 => memory.context_tokens += 1,
            1 => memory.preparation_reserve_bytes = 0,
            2 => memory.io_reserve_bytes = 0,
            3 => memory.reduction_reserve_bytes = 0,
            4 => memory.allocator_reserve_bytes = 0,
            5 => config.budget.max_kv_bytes -= 1,
            6 => config.mapping.chunks.max_chunks = 0,
            7 => config.mapping.chunks.max_chunks = 257,
            8 => config.max_question_bytes = 0,
            9 => config.max_question_bytes = config.planning.max_input_bytes + 1,
            10 => config.mapping.max_mask_visits = config.mapping.mask_visits_per_chunk - 1,
            11 => config.verification.max_fields = 0,
            12 => config.verification.max_matches = 0,
            13 => config.verification.max_scan_steps = 0,
            14 => transport.max_output_line_bytes = config.mapping.reduction.max_result_bytes - 1,
            15 => transport.max_epoch_id_bytes = transport.max_id_bytes - 1,
            16 => memory.cohort_chunks = Some(2),
            17 => { memory.cohort_chunks = Some(9); config.mapping.reduction.map_batch_chunks = 9; },
            18 => { synthesize(&mut config); config.synthesis.as_mut().unwrap().max_evidence_passages = 33; },
            19 => { synthesize(&mut config); config.synthesis.as_mut().unwrap().max_evidence_bytes = 4097; },
            _ => config.mapping.chunks.context_tokens -= 1,
        }
        assert!(memory.requirements(32, config, transport).is_err(), "fault {fault}");
    }
    for chunks in [0, 65, usize::MAX] {
        let (mut memory, mut config, transport) = fixture();
        memory.cohort_chunks = Some(chunks); config.mapping.reduction.map_batch_chunks = chunks;
        assert!(memory.requirements(32, config, transport).is_err(), "chunks {chunks}");
    }
}

#[test]
fn synthesis_cannot_spend_the_entire_allowance_before_discovery_or_reset_native_axes() {
    let (memory, mut config, transport) = fixture(); synthesize(&mut config);
    config.mapping.max_mask_visits = config.mapping.mask_visits_per_chunk;
    assert!(memory.requirements(32, config, transport).is_err());
    for axis in 0..5 {
        let (memory, mut config, transport) = fixture(); synthesize(&mut config);
        let work = &mut config.mapping.max_model_work;
        match axis { 0 => work.forward_positions = 1, 1 => work.projected_logits = 1,
            2 => work.attention_pairs = 1, 3 => work.projections.dot_products = 1,
            _ => work.projections.multiply_accumulates = 1 }
        assert!(memory.requirements(32, config, transport).is_err(), "axis {axis}");
    }
}

#[test]
fn input_document_and_native_reservations_must_coexist_without_overflow() {
    for owner in 0..5 {
        let (mut memory, mut config, transport) = fixture();
        match owner { 0 => memory.preparation_reserve_bytes = u64::MAX,
            1 => memory.io_reserve_bytes = u64::MAX, 2 => memory.allocator_reserve_bytes = u64::MAX,
            3 => memory.reduction_reserve_bytes = u64::MAX,
            _ => config.budget.max_output_bytes = u64::MAX }
        assert!(memory.requirements(32, config, transport).is_err(), "owner {owner}");
    }
    let (mut memory, config, transport) = fixture();
    let ordinary = memory.requirements(32, config, transport).unwrap();
    let other_input = ordinary.input_bytes - memory.preparation_reserve_bytes;
    memory.preparation_reserve_bytes = u64::MAX - other_input - 1024;
    assert!(memory.requirements(32, config, transport).is_err(),
        "representable input alone must not authorize an overflowing simultaneous peak");
}
