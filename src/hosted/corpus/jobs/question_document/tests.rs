//! Pure admission fixtures; no model outputs or runtime qualification claims.
use super::*;
use crate::{
    grammar::mask::MaskWorkLimits,
    jobs::JobWork,
    native_engine::constrained_int8,
    tasks::{answer::AnswerOptions, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::{SourcePlanningLimits, quantized::long::Int8SourceMapLimits}},
    validation::grounded_fields::GroundingBudget,
};
use std::time::Duration;

fn config() -> Int8QuestionDocumentBatchConfig {
    let one = constrained_int8::planned_work(8128, 64).unwrap();
    let document = (0..64).try_fold(Int8Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    let verification = GroundingBudget { max_fields: 4096, max_matches: 16384, max_scan_steps: 67_108_864 };
    Int8QuestionDocumentBatchConfig {
        options: AnswerOptions::default(),
        budget: TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(8192).unwrap().kv_bytes },
        planning: SourcePlanningLimits::default(),
        mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6,
                max_chunk_tokens: 6, context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 2, ..ExecutionLimits::default() },
            max_model_work: document, mask_limits: MaskWorkLimits::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 64_000 },
        verification, max_question_bytes: 1024, max_model_work: document.checked_add(document).unwrap(),
        max_mask_visits: 128_000,
        max_verification: GroundingBudget { max_fields: verification.max_fields * 4,
            max_matches: verification.max_matches * 4, max_scan_steps: verification.max_scan_steps * 4 },
    }
}
fn limits() -> JobLimits {
    let config = config();
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 8 << 20, max_result_bytes: config.mapping.reduction.max_result_bytes,
        max_spool_bytes: 64 << 20, max_materialized_bytes: 64 << 20,
        max_journal_bytes: 8 << 20, max_attempts: 4,
        max_work: JobWork { model: config.max_model_work, mask_node_visits: config.max_mask_visits } }
}
fn host() -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 1_000_000,
            cleanup_reserve_bytes: 1 << 20 } },
        transport: PopulationReadLimits { max_stream_bytes: 8 << 20, max_lines: 2 },
        preparation_reserve_bytes: 1 << 30, io_reserve_bytes: 128 << 10,
        journal_reserve_bytes: 64 << 20, serialization_reserve_bytes: 16 << 20 }
}
fn quote(input: JobInputMemory, prefill: Option<Int8PrefillLimits>, threads: Option<usize>)
    -> Int8QuestionDocumentJobRequirements {
    int8_question_document_job_requirements(config(), limits(), host(), input, 64 << 20, prefill, threads).unwrap()
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn retained_replay_changes_only_the_authenticated_population_owner() {
    let complete = quote(JobInputMemory::CompletePopulation, None, None);
    let retained = quote(JobInputMemory::RetainedRecord, None, None);
    for (required, input) in [(complete, JobInputMemory::CompletePopulation), (retained, JobInputMemory::RetainedRecord)] {
        assert_eq!(required.population_buffer_bytes, host().required_buffer_bytes_for_input(limits(), input).unwrap());
        assert!(required.document.input_bytes > required.population_buffer_bytes);
        assert_eq!(required.document.max_output_tokens, 0);
        assert_eq!(required.document.max_output_bytes, config().mapping.reduction.max_result_bytes as u64);
        assert_eq!(required.document.output_reservation_bytes, required.document.max_output_bytes * 4 + 4096);
    }
    assert_eq!(complete.document.input_bytes - complete.population_buffer_bytes,
        retained.document.input_bytes - retained.population_buffer_bytes);
    assert_eq!(complete.document.native, retained.document.native);
    assert_eq!(complete.document.workspace_bytes, retained.document.workspace_bytes);
    assert_eq!(complete.document.document_workspace_bytes, retained.document.document_workspace_bytes);
    assert_eq!(complete.document.output_reservation_bytes, retained.document.output_reservation_bytes);
}

#[test]
fn prompt_morsels_and_persistent_workers_add_their_simultaneous_storage_once() {
    let cold = quote(JobInputMemory::RetainedRecord, None, None);
    for rows in [1, 4, 64] {
        let layered = quote(JobInputMemory::RetainedRecord, Some(prefill(rows)), None);
        assert_eq!(layered.document.workspace_bytes, cold.document.workspace_bytes + prefill(rows).max_extra_scratch_bytes);
        assert_eq!(layered.document.input_bytes, cold.document.input_bytes);
        assert_eq!(layered.document.document_workspace_bytes, cold.document.document_workspace_bytes);
        for width in [2, 4, 64] {
            let team = quote(JobInputMemory::RetainedRecord, Some(prefill(rows)), Some(width));
            assert_eq!(team.document, layered.document);
            assert_eq!(team.team_workspace_bytes, dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert_eq!(team.total_bytes().unwrap(), layered.total_bytes().unwrap() + team.team_workspace_bytes);
        }
    }
}

#[test]
fn six_journal_axes_and_three_attempt_bounded_verification_axes_are_not_renewable() {
    for axis in 0..6 {
        let mut job = limits();
        match axis {
            0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1,
            2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            _ => job.max_work.mask_node_visits -= 1,
        }
        assert!(int8_question_document_job_requirements(config(), job, host(),
            JobInputMemory::RetainedRecord, 64 << 20, None, None).is_err());
    }
    for axis in 0..3 {
        let mut config = config();
        match axis { 0 => config.max_verification.max_fields -= 1,
            1 => config.max_verification.max_matches -= 1, _ => config.max_verification.max_scan_steps -= 1 }
        assert!(int8_question_document_job_requirements(config, limits(), host(),
            JobInputMemory::RetainedRecord, 64 << 20, None, None).is_err());
    }
    let mut extra_attempt = limits(); extra_attempt.max_attempts += 1;
    assert!(int8_question_document_job_requirements(config(), extra_attempt, host(),
        JobInputMemory::RetainedRecord, 64 << 20, None, None).is_err());
}

#[test]
fn storage_context_frontier_and_schedule_underquotes_refuse_without_io() {
    for axis in 0..9 {
        let mut h = host(); let mut job = limits(); let mut config = config();
        let mut reduction = 64 << 20; let mut schedule = None; let mut threads = None;
        match axis {
            0 => h.native.context_tokens -= 1,
            1 => h.preparation_reserve_bytes = 0,
            2 => h.journal_reserve_bytes = 0,
            3 => h.serialization_reserve_bytes = u64::MAX,
            4 => job.max_result_bytes = config.mapping.reduction.max_result_bytes - 1,
            5 => config.max_question_bytes = job.max_input_bytes_per_item + 1,
            6 => reduction = 0,
            7 => { let mut short = prefill(4); short.max_extra_scratch_bytes -= 1; schedule = Some(short); }
            _ => threads = Some(65),
        }
        assert!(int8_question_document_job_requirements(config, job, h,
            JobInputMemory::CompletePopulation, reduction, schedule, threads).is_err(), "axis {axis}");
    }
}
