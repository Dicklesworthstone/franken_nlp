//! Pure complete-population accounting; no native allocation or execution.
use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int4, strict_int4::Int4MemoryRequirement},
    tasks::{answer::AnswerOptions, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, grouped_int4::long::{Int4SourceMapLimits,
            question::synthesis::{QuestionSynthesisLimits, hierarchy::QuestionHierarchyLimits}}}},
};

const WIDTH: usize = 127;
fn config(mode: usize) -> Int4QuestionDocumentBatchConfig {
    let mut item = constrained_int4::planned_work(WIDTH, 8128, 64).unwrap();
    for _ in 0..8 { item = item.checked_add(item).unwrap(); }
    let total = item.checked_add(item).unwrap();
    let verification = GroundingBudget::default();
    let lifetime = GroundingBudget { max_fields: verification.max_fields * 4,
        max_matches: verification.max_matches * 4, max_scan_steps: verification.max_scan_steps * 4 };
    Int4QuestionDocumentBatchConfig { options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65_536,
            max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes },
        planning: SourcePlanningLimits::default(), mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 12, max_chunk_tokens: 12,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2, ..ExecutionLimits::default() },
            max_model_work: item, mask_limits: MaskWorkLimits::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 256000, verification },
        verification, synthesis: (mode > 0).then_some(QuestionSynthesisLimits {
            max_evidence_passages: 32, max_evidence_bytes: 4096 }),
        hierarchy: (mode > 1).then_some(QuestionHierarchyLimits {
            max_levels: 3, max_passes: 4, ..QuestionHierarchyLimits::default() }),
        max_question_bytes: 1024, max_model_work: total.checked_add(total).unwrap(),
        max_mask_visits: 1_024_000, max_map_verification: lifetime, max_verification: lifetime }
}
fn limits(config: Int4QuestionDocumentBatchConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 8 << 20, max_spool_bytes: 32 << 20,
        max_materialized_bytes: 32 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(config.max_model_work, config.max_mask_visits) }
}

fn memory(chunks: Option<usize>, groups: Option<usize>) -> Int4QuestionDocumentJobMemoryLimits {
    Int4QuestionDocumentJobMemoryLimits {
        document: Int4QuestionDocumentMemoryLimits { context_tokens: 8192,
            allocator_reserve_bytes: 65_536, preparation_reserve_bytes: 128 << 20,
            io_reserve_bytes: 131_072, reduction_reserve_bytes: 1 << 20,
            cohort_chunks: chunks, hierarchy_groups: groups },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    }
}

#[test]
fn complete_document_modes_and_independent_schedules_use_one_phase_peak() {
    let row = Int4MemoryRequirement::for_context(8192, WIDTH).unwrap();
    for mode in 0..3 {
        for chunks in [None, Some(1), Some(3)] {
            for groups in [None, Some(1), Some(2), Some(4)] {
                if mode != 2 && groups.is_some() { continue; }
                let mut config = config(mode); config.mapping.reduction.map_batch_chunks = chunks.unwrap_or(1);
                let quote = memory(chunks, groups).requirements(WIDTH, config, limits(config), JobInputMemory::CompletePopulation).unwrap();
                let rows = chunks.unwrap_or(1).max(groups.unwrap_or(1));
                assert_eq!(quote.document.native.kv_bytes, row.kv_bytes * rows as u64);
                assert_eq!(config.budget.max_kv_bytes, row.kv_bytes);
                assert_eq!(quote.document.max_output_bytes, config.mapping.reduction.max_result_bytes as u64);
                let tokens = if mode == 2 { 4 * 64 } else { 0 };
                assert_eq!(quote.document.max_output_tokens, tokens);
                assert_eq!(quote.document.output_reservation_bytes,
                    quote.document.max_output_bytes * 4 + tokens * 8 + 4096);
            }
        }
    }
    for mode in 0..2 {
        let config = config(mode);
        assert!(memory(None, Some(1)).requirements(WIDTH, config, limits(config), JobInputMemory::CompletePopulation).is_err());
    }
}

#[test]
fn whole_population_and_host_declarations_coexist_without_double_charging_preparation() {
    let config = config(1); let job = limits(config); let original = memory(None, None);
    let quote = original.requirements(WIDTH, config, job, JobInputMemory::CompletePopulation).unwrap();
    assert!(quote.document.input_bytes > quote.population_buffer_bytes);
    for field in 0..4 {
        let mut changed = original;
        match field { 0 => changed.document.preparation_reserve_bytes += 4096,
            1 => changed.document.io_reserve_bytes += 4096,
            2 => changed.journal_reserve_bytes += 4096, _ => changed.serialization_reserve_bytes += 4096 }
        let changed = changed.requirements(WIDTH, config, job, JobInputMemory::CompletePopulation).unwrap();
        assert_eq!(changed.document.input_bytes - quote.document.input_bytes, 4096);
        assert_eq!(changed.population_buffer_bytes - quote.population_buffer_bytes, 4096);
        assert_eq!(changed.document.native, quote.document.native);
        assert_eq!(changed.document.document_workspace_bytes, quote.document.document_workspace_bytes);
    }
    let mut larger = job; larger.max_snapshot_bytes += 4096;
    let larger = original.requirements(WIDTH, config, larger, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(larger.document.input_bytes - quote.document.input_bytes, 8192);
    assert_eq!(larger.population_buffer_bytes - quote.population_buffer_bytes, 8192);
    let mut larger = job; larger.max_items += 1;
    let larger = original.requirements(WIDTH, config, larger, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(larger.document.input_bytes - quote.document.input_bytes, 1024);
    assert_eq!(larger.population_buffer_bytes - quote.population_buffer_bytes, 1024);
}

#[test]
fn durable_native_mask_and_verification_authority_is_checked_before_quoting_memory() {
    let original = config(2); let job = limits(original);
    for field in 0..6 {
        let mut changed = job;
        match field { 0 => changed.max_work.model.forward_positions -= 1,
            1 => changed.max_work.model.projected_logits -= 1,
            2 => changed.max_work.model.attention_pairs -= 1,
            3 => changed.max_work.model.projections.dot_products -= 1,
            4 => changed.max_work.model.projections.multiply_accumulates -= 1,
            _ => changed.max_work.mask_node_visits -= 1 }
        assert!(memory(None, None).requirements(WIDTH, original, changed, JobInputMemory::CompletePopulation).is_err(), "native {field}");
    }
    for field in 0..6 {
        let mut changed = original;
        let pool = if field < 3 { &mut changed.max_map_verification } else { &mut changed.max_verification };
        match field % 3 { 0 => pool.max_fields -= 1, 1 => pool.max_matches -= 1, _ => pool.max_scan_steps -= 1 }
        assert!(memory(None, None).requirements(WIDTH, changed, job, JobInputMemory::CompletePopulation).is_err(), "verification {field}");
    }
    let mut attempts = job; attempts.max_attempts += 1;
    assert!(memory(None, None).requirements(WIDTH, original, attempts, JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn independent_source_question_and_encoded_record_caps_preserve_all_declared_ownership() {
    let mut config = config(2); let job = limits(config);
    config.mapping.chunks.max_input_bytes = job.max_input_bytes_per_item;
    config.max_question_bytes = job.max_input_bytes_per_item;
    let both = memory(None, None).requirements(WIDTH, config, job, JobInputMemory::CompletePopulation).unwrap();
    let mut short_question = config; short_question.max_question_bytes = 1;
    let short = memory(None, None).requirements(WIDTH, short_question, job, JobInputMemory::CompletePopulation).unwrap();
    assert!(both.document.input_bytes > short.document.input_bytes);
    assert_eq!(both.population_buffer_bytes, short.population_buffer_bytes);
    assert_eq!(both.document.native, short.document.native);
    for field in 0..3 {
        let mut refused = config;
        match field { 0 => refused.mapping.chunks.max_input_bytes += 1,
            1 => refused.max_question_bytes += 1,
            _ => refused.mapping.reduction.max_result_bytes = job.max_result_bytes + 1 }
        assert!(memory(None, None).requirements(WIDTH, refused, job, JobInputMemory::CompletePopulation).is_err(), "field {field}");
    }
}

#[test]
fn invalid_schedule_host_reserves_and_simultaneous_ownership_overflow_refuse() {
    let config = config(2); let job = limits(config);
    for field in 0..12 {
        let mut changed = memory(None, None);
        match field {
            0 => changed.document.context_tokens -= 1,
            1 => changed.document.allocator_reserve_bytes = 0,
            2 => changed.document.preparation_reserve_bytes = 0,
            3 => changed.document.io_reserve_bytes = 0,
            4 => changed.document.reduction_reserve_bytes = 0,
            5 => changed.journal_reserve_bytes = 0,
            6 => changed.serialization_reserve_bytes = 0,
            7 => changed.document.preparation_reserve_bytes = u64::MAX,
            8 => changed.journal_reserve_bytes = u64::MAX,
            9 => changed.serialization_reserve_bytes = u64::MAX,
            10 => changed.transport.max_stream_bytes = 0,
            _ => changed.document.allocator_reserve_bytes = u64::MAX,
        }
        assert!(changed.requirements(WIDTH, config, job, JobInputMemory::CompletePopulation).is_err(), "field {field}");
    }
    for groups in [0, 5, 65, usize::MAX] {
        assert!(memory(None, Some(groups)).requirements(WIDTH, config, job, JobInputMemory::CompletePopulation).is_err());
    }
    let mut maximal_population = job; maximal_population.max_snapshot_bytes = i64::MAX as u64;
    assert!(memory(None, None).requirements(WIDTH, config, maximal_population, JobInputMemory::CompletePopulation).is_err());
    for width in [0, 10753, usize::MAX] {
        assert!(memory(None, None).requirements(width, config, job, JobInputMemory::CompletePopulation).is_err());
    }
}

#[test]
fn retained_qa_quotes_remove_only_snapshot_copies_across_map_and_hierarchy_schedules() {
    for (mode, chunks, groups) in [(0, None, None), (1, Some(3), None),
        (2, None, Some(2)), (2, Some(3), Some(2))] {
        let mut config = config(mode); config.mapping.reduction.map_batch_chunks = chunks.unwrap_or(1);
        let job = limits(config); let memory = memory(chunks, groups);
        let complete = memory.requirements(WIDTH, config, job, JobInputMemory::CompletePopulation).unwrap();
        let retained = memory.requirements(WIDTH, config, job, JobInputMemory::RetainedRecord).unwrap();
        let mut expected = complete;
        expected.population_buffer_bytes -= 2 * job.max_snapshot_bytes;
        expected.document.input_bytes -= 2 * job.max_snapshot_bytes;
        // Every native, decoder, reduction, output and metadata field stays
        // identical, including the selected map/hierarchy phase peak.
        assert_eq!(retained, expected);
        let mut larger = job; larger.max_snapshot_bytes += 4096;
        assert_eq!(memory.requirements(WIDTH, config, larger, JobInputMemory::RetainedRecord).unwrap(), retained);
        for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
            let mut short = job; short.max_work.model.attention_pairs -= 1;
            assert!(memory.requirements(WIDTH, config, short, input).is_err());
            let mut missing = memory; missing.document.reduction_reserve_bytes = 0;
            assert!(missing.requirements(WIDTH, config, job, input).is_err());
            let mut missing = memory; missing.document.allocator_reserve_bytes = 0;
            assert!(missing.requirements(WIDTH, config, job, input).is_err());
        }
    }
}
