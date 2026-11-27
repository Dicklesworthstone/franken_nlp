//! Model-free refusal and simultaneous-ownership checks; no native-output claim.
use super::*;
use crate::{batch::extract::ExtractionMaskBudget, grammar::mask::MaskWorkLimits};

fn host() -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 1 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 16 << 20, max_lines: 4 },
        preparation_reserve_bytes: 64 << 20, io_reserve_bytes: 65536,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 2 << 20 }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 32, max_output_tokens: 64, max_output_bytes: 65536,
        max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(host().native.context_tokens).unwrap().kv_bytes }
}
fn native() -> Int8SampledSentimentJobLimits {
    let max_item_work = Int8Work::for_sequence(0, 4096, 64 * NANBEIGE_VOCAB_SIZE).unwrap();
    let mut max_model_work = Int8Work::default();
    for _ in 0..4 { max_model_work = max_model_work.checked_add(max_item_work).unwrap(); }
    Int8SampledSentimentJobLimits { max_item_work, max_model_work,
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 1000,
            checkpoint_interval_nodes: 64 }, max_visits_per_item: 1_000_000,
            max_visits_per_run: 4_000_000 },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn job() -> JobLimits {
    let work = native();
    JobLimits { max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 16 << 20, max_result_bytes: 65536, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20, max_attempts: 8,
        max_work: JobWork { model: work.max_model_work,
            mask_node_visits: work.masks.max_visits_per_run } }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn quote(host: JobHostLimits, task: TaskBudget, work: Int8SampledSentimentJobLimits,
    job: JobLimits, input: JobInputMemory, prefill: Option<Int8PrefillLimits>, width: Option<usize>)
    -> Result<Int8SampledSentimentJobRequirements, HostedJobError> {
    int8_sampled_sentiment_job_requirements(task, work, 32 << 20, job, host, input, prefill, width)
}

#[test]
fn original_and_retained_quotes_keep_bundle_sampler_prompt_and_children_together() {
    let host = host(); let task = task(); let work = native(); let job = job();
    let sampler = json_sampling_workspace_bytes().unwrap();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let serial = quote(host, task, work, job, input, None, None).unwrap();
        assert_eq!(serial.input_bytes, host.required_buffer_bytes_for_input(job, input).unwrap());
        assert_eq!(serial.workspace_bytes, serial.native.rope_bytes
            + serial.native.scratch_payload_bound + host.native.allocator_reserve_bytes + sampler);
        assert_eq!(serial.output_reservation_bytes,
            4 * task.max_output_bytes + 8 * u64::from(task.max_output_tokens) + 4096);
        assert_eq!(serial.team_workspace_bytes, 0);
        for width in [2, 3, 64] { for rows in [1, 4, 64] {
            let prompt = prefill(rows);
            let scheduled = quote(host, task, work, job, input, Some(prompt), Some(width)).unwrap();
            assert_eq!(scheduled.native, serial.native);
            assert_eq!(scheduled.input_bytes, serial.input_bytes);
            assert_eq!(scheduled.output_reservation_bytes, serial.output_reservation_bytes);
            assert_eq!(scheduled.workspace_bytes, serial.workspace_bytes + prompt.validate().unwrap());
            assert_eq!(scheduled.team_workspace_bytes,
                dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert_eq!(scheduled.total_bytes().unwrap(), serial.total_bytes().unwrap()
                + prompt.validate().unwrap() + scheduled.team_workspace_bytes);
        } }
    }
    let original = quote(host, task, work, job, JobInputMemory::CompletePopulation, None, None).unwrap();
    let retained = quote(host, task, work, job, JobInputMemory::RetainedRecord, None, None).unwrap();
    assert!(retained.input_bytes < original.input_bytes);
    assert_eq!(retained.workspace_bytes, original.workspace_bytes);
    assert_eq!(retained.output_reservation_bytes, original.output_reservation_bytes);
}

#[test]
fn sampler_cap_does_not_invent_an_allocation_or_discount_the_actual_payload() {
    let actual = native();
    let mut larger = actual; larger.max_sampler_bytes = u64::MAX;
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let baseline = quote(host(), task(), actual, job(), input, Some(prefill(4)), Some(3)).unwrap();
        assert_eq!(baseline,
            quote(host(), task(), larger, job(), input, Some(prefill(4)), Some(3)).unwrap());
        let mut short = actual; short.max_sampler_bytes -= 1;
        assert!(quote(host(), task(), short, job(), input, None, None).is_err());
    }
}

#[test]
fn lifetime_work_preparation_full_kv_and_output_are_checked_before_input() {
    let original = host(); let task = task(); let work = native(); let job = job();
    let input = JobInputMemory::CompletePopulation;
    let mut wrong = job; wrong.max_work.mask_node_visits += 1;
    assert!(quote(original, task, work, wrong, input, None, None).is_err());
    let mut wrong = job; wrong.max_work.model.forward_positions += 1;
    assert!(quote(original, task, work, wrong, input, None, None).is_err());
    let mut short = task; short.max_kv_bytes -= 1;
    assert!(quote(original, short, work, job, input, None, None).is_err());
    let mut short = original; short.preparation_reserve_bytes = (32 << 20) - 1;
    assert!(quote(short, task, work, job, input, None, None).is_err());
    assert!(int8_sampled_sentiment_job_requirements(task, work, 0, job, original,
        input, None, None).is_err());
    let mut empty = task; empty.max_output_bytes = 0;
    assert!(quote(original, empty, work, job, input, None, None).is_err());
    let mut overflow = original; overflow.native.allocator_reserve_bytes = u64::MAX;
    assert!(quote(overflow, task, work, job, input, None, None).is_err());
    let mut overflow = original; overflow.serialization_reserve_bytes = u64::MAX;
    assert!(quote(overflow, task, work, job, input, None, None).is_err());
}

#[test]
fn bad_worker_or_prompt_capacity_refuses_before_any_job_directory_is_opened() {
    let input = JobInputMemory::RetainedRecord;
    for width in [0, 1, 65, usize::MAX] {
        assert!(quote(host(), task(), native(), job(), input, Some(prefill(4)), Some(width)).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        assert!(quote(host(), task(), native(), job(), input,
            Some(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX }),
            Some(3)).is_err());
    }
    let mut short = prefill(4); short.max_extra_scratch_bytes -= 1;
    assert!(quote(host(), task(), native(), job(), input, Some(short), Some(3)).is_err());
}

#[test]
fn durable_factory_is_a_shared_physical_owner() {
    fn send<T: Send + 'static>() {}
    send::<Arc<Int8SampledSentimentJobPlanner>>();
}
