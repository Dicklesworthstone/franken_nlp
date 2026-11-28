//! Pure lifetime and simultaneous-ownership bounds, not native execution.
use super::*;
use crate::{grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int8, strict_int8::{Int8MemoryRequirement, Int8Work}},
    tasks::{answer::AnswerOptions, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, quantized::long::Int8SourceMapLimits}}};

pub(crate) fn config() -> Int8QuestionDocumentBatchConfig {
    let one = constrained_int8::planned_work(8128, 64).unwrap();
    let item = (0..64).try_fold(Int8Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    let total = item.checked_add(item).unwrap(); let verification = GroundingBudget::default();
    Int8QuestionDocumentBatchConfig { options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65_536,
            max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(8192).unwrap().kv_bytes },
        planning: SourcePlanningLimits::default(), mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 12, max_chunk_tokens: 12,
                context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2, ..ExecutionLimits::default() },
            max_model_work: item, mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000 },
        verification, max_question_bytes: 1024, max_model_work: total.checked_add(total).unwrap(), max_mask_visits: 256_000,
        max_verification: GroundingBudget { max_fields: verification.max_fields * 4,
            max_matches: verification.max_matches * 4, max_scan_steps: verification.max_scan_steps * 4 } }
}
pub(crate) fn limits(config: Int8QuestionDocumentBatchConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 8 << 20, max_spool_bytes: 32 << 20,
        max_materialized_bytes: 32 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: config.max_model_work, mask_node_visits: config.max_mask_visits } }
}
fn memory() -> Int8QuestionDocumentJobMemoryLimits {
    Int8QuestionDocumentJobMemoryLimits {
        document: Int8QuestionDocumentMemoryLimits { context_tokens: 8192, allocator_reserve_bytes: 65_536,
            preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 131_072, reduction_reserve_bytes: 1 << 20 },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20 }
}

#[test]
fn retained_input_removes_only_complete_payload_not_the_document_question_or_index() {
    let config = config(); let job = limits(config);
    let complete = memory().requirements(config, job, JobInputMemory::CompletePopulation).unwrap();
    let retained = memory().requirements(config, job, JobInputMemory::RetainedRecord).unwrap();
    assert_eq!(complete.population_buffer_bytes - retained.population_buffer_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(complete.document.input_bytes - retained.document.input_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(complete.document.native, retained.document.native);
    assert_eq!(complete.document.document_workspace_bytes, retained.document.document_workspace_bytes);
    assert_eq!(complete.document.output_reservation_bytes, retained.document.output_reservation_bytes);
    assert_eq!(retained.document.max_output_tokens, 0);
    assert!(retained.document.input_bytes > retained.population_buffer_bytes);
    assert_eq!(retained.document.output_reservation_bytes, 4 * retained.document.max_output_bytes + 4096);
}

#[test]
fn preparation_io_journal_and_serialization_are_counted_exactly_once() {
    let config = config(); let job = limits(config); let baseline = memory();
    let quote = baseline.requirements(config, job, JobInputMemory::CompletePopulation).unwrap();
    for field in 0..4 {
        let mut changed = baseline;
        match field { 0 => changed.document.preparation_reserve_bytes += 4096,
            1 => changed.document.io_reserve_bytes += 4096,
            2 => changed.journal_reserve_bytes += 4096, _ => changed.serialization_reserve_bytes += 4096 }
        let more = changed.requirements(config, job, JobInputMemory::CompletePopulation).unwrap();
        assert_eq!(more.document.input_bytes - quote.document.input_bytes, 4096);
        assert_eq!(more.population_buffer_bytes - quote.population_buffer_bytes, 4096);
        assert_eq!(more.document.document_workspace_bytes, quote.document.document_workspace_bytes);
    }
}

#[test]
fn verification_covers_every_failed_or_resumed_attempt_without_overflow() {
    let config = config(); let job = limits(config); check_limits(config, job).unwrap();
    for field in 0..3 {
        let mut short = config;
        match field { 0 => short.max_verification.max_fields -= 1, 1 => short.max_verification.max_matches -= 1,
            _ => short.max_verification.max_scan_steps -= 1 }
        assert_eq!(check_limits(short, job), Err(BatchCode::InvalidLimits.into()));
    }
    let maximum = GroundingBudget { max_fields: usize::MAX, max_matches: usize::MAX, max_scan_steps: u64::MAX };
    for field in 0..3 {
        let mut per = GroundingBudget { max_fields: 1, max_matches: 1, max_scan_steps: 1 };
        match field { 0 => per.max_fields = usize::MAX, 1 => per.max_matches = usize::MAX, _ => per.max_scan_steps = u64::MAX }
        assert!(verification_lifetime(per, maximum, 2).is_err());
    }
    assert!(verification_lifetime(GroundingBudget::default(), maximum, 0).is_err());
}

#[test]
fn every_native_mask_and_whole_owner_mismatch_refuses_before_private_io() {
    let config = config(); let original = limits(config);
    for axis in 0..6 {
        let mut job = original;
        match axis { 0 => job.max_work.model.forward_positions -= 1, 1 => job.max_work.model.projected_logits -= 1,
            2 => job.max_work.model.attention_pairs -= 1, 3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1, _ => job.max_work.mask_node_visits -= 1 }
        assert!(check_limits(config, job).is_err());
    }
    for axis in 0..4 {
        let mut m = memory();
        match axis { 0 => m.journal_reserve_bytes = 0, 1 => m.serialization_reserve_bytes = 0,
            2 => m.document.preparation_reserve_bytes = u64::MAX, _ => m.document.reduction_reserve_bytes = u64::MAX }
        assert!(m.requirements(config, original, JobInputMemory::CompletePopulation).is_err());
    }
    let mut huge = original; huge.max_snapshot_bytes = i64::MAX as u64;
    assert!(memory().requirements(config, huge, JobInputMemory::CompletePopulation).is_err());
    assert!(memory().requirements(config, huge, JobInputMemory::RetainedRecord).is_ok());
    let mut too_big = config; too_big.max_question_bytes = original.max_input_bytes_per_item + 1;
    assert!(check_limits(too_big, original).is_err());
}
