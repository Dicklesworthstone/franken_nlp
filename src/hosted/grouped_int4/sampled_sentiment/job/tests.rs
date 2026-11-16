//! Pure host admission and controller-only physical-lifecycle fixtures.
//! These checks do not establish real-model parity or durability certification.

use super::*;
use crate::{
    batch::extract::ExtractionMaskBudget,
    jobs::population::PopulationReadLimits,
    tasks::sentiment::sampled::SampledSentimentLimits,
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding},
};
use std::time::Duration;

const PREPARATION: u64 = 600 * 1024 * 1024;

fn fixture(width: usize, samples: usize)
    -> (TaskBudget, SampledSentimentBatchLimits, JobLimits, JobHostLimits) {
    let native = Int4MemoryRequirement::for_context(128, width).unwrap();
    let one = constrained_int4::planned_work(width, 32, 16).unwrap();
    let item = (0..samples).try_fold(Int4Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    let work = SampledSentimentBatchLimits { max_item_work: item, max_model_work: item.checked_add(item).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 2_000_000,
            checkpoint_interval_nodes: 256 }, max_visits_per_item: 32_000_000_000,
            max_visits_per_run: 64_000_000_000 }, max_sampler_bytes: json_sampling_workspace_bytes().unwrap() };
    let task = TaskBudget { max_input_tokens: 32, max_output_tokens: 2048, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
    let job = JobLimits { max_items: 2, max_id_bytes: 256, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 65_536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
                cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: PREPARATION, io_reserve_bytes: 131_072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (task, work, job, host)
}

#[test]
fn all_samples_share_one_kv_context_and_sampler_but_keep_complete_work_charges() {
    for width in [1, 3, 32, 127, 128, MAX_MODEL_K] {
        let mut previous = None;
        for samples in [1, 4, 32, 128] {
            let (task, work, job, host) = fixture(width, samples);
            let required = requirements(width, task, work, PREPARATION, task.max_output_bytes,
                job, host, JobInputMemory::CompletePopulation, None).unwrap();
            assert_eq!(required.native, Int4MemoryRequirement::for_context(128, width).unwrap());
            assert_eq!(required.workspace_bytes, required.native.rope_bytes + required.native.scratch_payload_bound
                + host.native.allocator_reserve_bytes + json_sampling_workspace_bytes().unwrap());
            assert_eq!(required.input_bytes, host.required_buffer_bytes(job).unwrap());
            if let Some(expected) = previous { assert_eq!(required, expected); }
            previous = Some(required);
            assert_eq!(job.max_work, JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run));
        }
    }
}

#[test]
fn every_frozen_native_and_mask_counter_is_required_for_both_input_owners() {
    let (task, work, job, host) = fixture(32, 8);
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        for axis in 0..6 {
            let mut wrong = job;
            match axis {
                0 => wrong.max_work.model.forward_positions += 1,
                1 => wrong.max_work.model.projected_logits += 1,
                2 => wrong.max_work.model.attention_pairs += 1,
                3 => wrong.max_work.model.projections.dot_products += 1,
                4 => wrong.max_work.model.projections.multiply_accumulates += 1,
                _ => wrong.max_work.mask_node_visits += 1,
            }
            assert!(requirements(32, task, work, PREPARATION, task.max_output_bytes,
                wrong, host, input, None).is_err(), "axis {axis}");
        }
    }
}

#[test]
fn retained_replay_removes_only_population_copies_and_keeps_sampler_native_and_preparation() {
    let (task, work, job, host) = fixture(128, 8);
    let quote = |input| requirements(128, task, work, PREPARATION, task.max_output_bytes,
        job, host, input, None).unwrap();
    let original = quote(JobInputMemory::CompletePopulation);
    let retained = quote(JobInputMemory::RetainedRecord);
    assert_eq!(original.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(retained.native, original.native);
    assert_eq!(retained.workspace_bytes, original.workspace_bytes);
    assert_eq!(retained.input_bytes,
        host.required_buffer_bytes_for_input(job, JobInputMemory::RetainedRecord).unwrap());
}

#[test]
fn compact_workspace_is_in_the_planner_quote_and_does_not_renew_per_sample_or_attempt() {
    use crate::batch::sentiment::sampled::{sampled_sentiment_batch_preparation_bytes,
        sampled_sentiment_job_preparation_bytes, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES};
    let (task, work, job, host) = fixture(32, 8);
    let limits = SampledSentimentLimits::default();
    let encoding = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let base = sampled_sentiment_job_preparation_bytes(task, limits, &UntrustedEncoding::ByteFallback).unwrap();
    assert_eq!(base, sampled_sentiment_batch_preparation_bytes(task, limits,
        &UntrustedEncoding::ByteFallback).unwrap() + SAMPLED_SENTIMENT_JOB_RECIPE_BYTES);
    let compact = sampled_sentiment_job_preparation_bytes(task, limits,
        &UntrustedEncoding::SafeBpeV1(encoding)).unwrap();
    assert_eq!(compact - base, encoding.requirements().unwrap().workspace_bytes);
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let base_host = JobHostLimits { preparation_reserve_bytes: base, ..host };
        let compact_host = JobHostLimits { preparation_reserve_bytes: compact, ..host };
        let byte = requirements(32, task, work, base, task.max_output_bytes, job, base_host, input, None).unwrap();
        let wide = requirements(32, task, work, compact, task.max_output_bytes, job, compact_host, input, None).unwrap();
        assert_eq!(wide.input_bytes - byte.input_bytes, compact - base);
        assert_eq!(wide.workspace_bytes, byte.workspace_bytes);
        assert!(requirements(32, task, work, compact, task.max_output_bytes, job,
            JobHostLimits { preparation_reserve_bytes: compact - 1, ..host }, input, None).is_err());
        let mut larger = job; larger.max_attempts = 1000;
        let again = requirements(32, task, work, compact, task.max_output_bytes,
            larger, compact_host, input, None).unwrap();
        assert_eq!(again.native, wide.native);
        assert_eq!(again.workspace_bytes, wide.workspace_bytes);
    }
}

#[test]
fn effective_task_output_can_be_tighter_than_the_task_ceiling_without_changing_storage() {
    let (mut task, work, job, host) = fixture(32, 8);
    let effective = task.max_output_bytes;
    task.max_output_bytes *= 2;
    requirements(32, task, work, PREPARATION, effective, job, host,
        JobInputMemory::RetainedRecord, None).unwrap();
    for output in [0, effective + 1, task.max_output_bytes + 1] {
        assert!(requirements(32, task, work, PREPARATION, output, job, host,
            JobInputMemory::RetainedRecord, None).is_err());
    }
}

#[test]
fn missing_or_overflowing_resource_axes_refuse_before_population_or_native_allocation() {
    for axis in 0..12 {
        let (mut task, mut work, job, mut host) = fixture(32, 8);
        match axis {
            0 => host.native.context_tokens = 0,
            1 => task.max_kv_bytes -= 1,
            2 => work.max_sampler_bytes -= 1,
            3 => work.masks.max_visits_per_run = work.masks.max_visits_per_item - 1,
            4 => host.preparation_reserve_bytes -= 1,
            5 => host.io_reserve_bytes = 0,
            6 => host.journal_reserve_bytes = 0,
            7 => host.serialization_reserve_bytes = 0,
            8 => host.native.allocator_reserve_bytes = 0,
            9 => host.native.allocator_reserve_bytes = u64::MAX,
            10 => host.preparation_reserve_bytes = u64::MAX,
            _ => host.native.run.max_elapsed = Duration::ZERO,
        }
        assert!(requirements(32, task, work, PREPARATION, task.max_output_bytes,
            job, host, JobInputMemory::CompletePopulation, None).is_err(), "axis {axis}");
    }
}

#[test]
fn full_declared_prefill_storage_is_reserved_once_for_the_whole_job() {
    for width in [1, 3, 32, 128] {
        let (task, work, job, host) = fixture(width, 8);
        let base = requirements(width, task, work, PREPARATION, task.max_output_bytes,
            job, host, JobInputMemory::RetainedRecord, None).unwrap();
        for rows in [1, 4, 64] {
            let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            let packed = requirements(width, task, work, PREPARATION, task.max_output_bytes,
                job, host, JobInputMemory::RetainedRecord, Some(prefill)).unwrap();
            assert_eq!(packed.native, base.native);
            assert_eq!(packed.input_bytes, base.input_bytes);
            assert_eq!(packed.workspace_bytes, base.workspace_bytes + extra);
            check_schedule(None, Some(prefill)).unwrap();
            assert!(check_schedule(Some(2), Some(prefill)).is_err());
            assert!(requirements(width, task, work, PREPARATION, task.max_output_bytes,
                job, host, JobInputMemory::RetainedRecord,
                Some(Int4PrefillLimits { max_extra_scratch_bytes: extra - 1, ..prefill })).is_err());
        }
        for rows in [0, 65, usize::MAX] {
            let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
            assert!(check_schedule(None, Some(prefill)).is_err());
        }
    }
}

#[test]
fn owned_types_cross_serial_team_and_prefill_host_boundaries() {
    fn send<T: Send + 'static>() {}
    send::<SampledSentimentJobPlanner>();
    send::<JobInput<std::io::Cursor<Vec<u8>>>>();
    send::<JobExecution<std::io::Cursor<Vec<u8>>>>();
    send::<DeferredSampledJob<std::io::Cursor<Vec<u8>>>>();
    let _serial = NlpEngine::job_int4_sampled_sentiment::<std::io::Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_sampled_sentiment_with_threads::<std::io::Cursor<Vec<u8>>>;
    let _prefill = NlpEngine::job_int4_sampled_sentiment_layer_major::<std::io::Cursor<Vec<u8>>>;
    for width in [0, 1, 65, usize::MAX] { assert!(int4_task_team_workspace_bytes(width, 32).is_err()); }
    for width in [2, 4, 64] { assert!(int4_task_team_workspace_bytes(width, 32).unwrap() > 0); }
}

#[test]
fn public_pre_io_quote_prices_the_durable_output_claim_and_one_physical_team() {
    let (task, work, job, host) = fixture(32, 8);
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let serial = int4_sampled_sentiment_job_requirements(32, task, work, PREPARATION,
            job, host, input, None, None).unwrap();
        assert_eq!(serial.input_bytes, host.required_buffer_bytes_for_input(job, input).unwrap());
        assert_eq!(serial.output_reservation_bytes, 4 * task.max_output_bytes + 4096);
        assert_eq!(serial.team_workspace_bytes, 0);
        assert_eq!(serial.total_bytes().unwrap(), serial.input_bytes + serial.native.kv_bytes
            + serial.workspace_bytes + serial.output_reservation_bytes);
        for threads in [2, 4, 64] {
            let team = int4_sampled_sentiment_job_requirements(32, task, work, PREPARATION,
                job, host, input, None, Some(threads)).unwrap();
            assert_eq!(team.team_workspace_bytes, int4_task_team_workspace_bytes(threads, 32).unwrap());
            assert_eq!(team.total_bytes().unwrap() - serial.total_bytes().unwrap(),
                team.team_workspace_bytes);
            assert_eq!(team.native, serial.native);
            assert_eq!(team.input_bytes, serial.input_bytes);
            assert_eq!(team.workspace_bytes, serial.workspace_bytes);
            assert_eq!(team.output_reservation_bytes, serial.output_reservation_bytes);
        }
        for threads in [0, 1, 65, usize::MAX] {
            assert!(int4_sampled_sentiment_job_requirements(32, task, work, PREPARATION,
                job, host, input, None, Some(threads)).is_err());
        }
    }
}

#[test]
fn public_quote_keeps_tighter_stored_results_and_prefill_recipe_capacity_explicit() {
    let (mut task, work, job, host) = fixture(32, 8);
    task.max_output_bytes *= 2;
    let serial = int4_sampled_sentiment_job_requirements(32, task, work, PREPARATION,
        job, host, JobInputMemory::RetainedRecord, None, None).unwrap();
    assert_eq!(serial.output_reservation_bytes, 4 * job.max_result_bytes as u64 + 4096);
    let prefill = Int4PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() };
    let packed = int4_sampled_sentiment_job_requirements(32, task, work, PREPARATION,
        job, host, JobInputMemory::RetainedRecord, Some(prefill), None).unwrap();
    assert_eq!(packed.total_bytes().unwrap() - serial.total_bytes().unwrap(),
        prefill.max_extra_scratch_bytes);
    assert_eq!(packed.output_reservation_bytes, serial.output_reservation_bytes);
    assert!(int4_sampled_sentiment_job_requirements(32, task, work, PREPARATION,
        job, host, JobInputMemory::RetainedRecord, Some(prefill), Some(2)).is_err());
    let overflow = SampledSentimentJobRequirements { output_reservation_bytes: u64::MAX, ..serial };
    assert!(overflow.total_bytes().is_err());
}

mod lifecycle;
mod model;
