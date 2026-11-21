//! Parser, pre-IO refusal and complete-document durable resource contracts.
use super::*;
use crate::{batch::sentiment::sampled::long::sampled_sentiment_document_preparation_bytes,
    jobs::{JobInputMemory, JobWork}};

const SEED: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
fn argv(operation: &str, snapshot: bool) -> Vec<String> {
    let mut words = ["sentiment-sampled-document-job", operation, "--memory-mib", "131072",
        "--seed", SEED, "--job-dir", "private-job", "--key-file", "private-key",
        "--limits", "private-limits", "--job-id", "0123456789abcdef0123456789abcdef",
        "--store-results"].into_iter().map(str::to_owned).collect::<Vec<_>>();
    let model = if snapshot { vec!["--snapshot", "private-snapshot", "--expected-model-sha256", SEED] }
        else { vec!["--source-dir", "private-source", "--source-manifest", "private-manifest"] };
    words.extend(model.into_iter().map(str::to_owned));
    words
}
pub(super) fn command(operation: &str, snapshot: bool, flags: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation, snapshot);
    words.extend(flags.iter().map(|word| (*word).to_owned()));
    JobCommand::from_matches(&definition().try_get_matches_from(words).unwrap()).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, command: &JobCommand, admitted: MapLimits) -> JobLimits {
    let sampled = command.args.document.sampled;
    let work = sampled.work_ceiling();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: admitted.mapping.max_result_bytes as usize,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork::from_int4(work.checked_add(work).unwrap(), 2 * sampled.max_mask_node_visits) }
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused job read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused job flushed output"); }
}

#[test]
fn complete_document_command_preserves_sampling_chunk_and_owned_dispatch_options() {
    for snapshot in [false, true] { for operation in ["start", "resume"] {
        let (c, args) = command(operation, snapshot, &["population.ndjson", "--samples", "7",
            "--temperature-milli", "800", "--top-p-ppm", "950000", "--top-k", "9",
            "--max-chunks", "11", "--max-chunk-bytes", "2048", "--max-tokenizer-calls", "1234",
            "--input-encoding", "safe-bpe-v1", "--reuse-prefill", "--compute-threads", "4"]);
        assert_eq!(c.mode.name(), operation);
        assert_eq!(args.input, PathBuf::from("population.ndjson"));
        let sampled = c.args.document.sampled;
        let options = sampled.options(&args).unwrap();
        assert_eq!((options.samples, options.temperature_milli, options.top_p_ppm, options.top_k),
            (7, 800, 950000, Some(9)));
        let admitted = c.args.validate(&args).unwrap();
        assert_eq!(admitted.common.compute.threads, 4);
        assert_eq!(admitted.common.snapshot_pin.is_some(), snapshot);
        assert_eq!(admitted.mapping.chunks.max_chunks, 11);
        assert_eq!(admitted.mapping.chunks.max_chunk_bytes, 2048);
        assert_eq!(admitted.mapping.chunks.max_tokenizer_calls, 1234);
        assert!(!args.input_encoding.is_default());
        assert!(sampled.reuse_prefill);
        let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()];
        words.extend(argv(operation, snapshot));
        let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
        let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong candidate dispatcher"); };
        assert_eq!(c.task, Int4Task::SampledSentimentDocumentJob);
        assert!(matches!(&c.job, Some(Int4JobCommand::SampledSentimentDocument(_))));
        assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    } }
}

#[test]
fn lifetime_work_and_masks_never_renew_at_chunks_samples_items_or_attempts() {
    let (mut c, args) = command("resume", false, &[]);
    let admitted = c.args.validate(&args).unwrap();
    let job = lifetime(&args, &c, admitted);
    let a = c.args.lifetime_limits(&args, admitted, job).unwrap();
    assert_eq!(a.native.max_model_work, job.max_work.int4_model_work());
    assert_eq!(a.native.masks.max_visits_per_run, job.max_work.mask_node_visits);
    assert_eq!(a.native.max_item_work, a.mapping.max_model_work);
    let mut more = job; more.max_items += 20; more.max_attempts += 40;
    c.args.storage.max_stream_mib = 1024;
    c.args.storage.max_input_lines = 1_000_000;
    let b = c.args.lifetime_limits(&args, admitted, more).unwrap();
    assert_eq!(a.native.max_model_work, b.native.max_model_work);
    assert_eq!(a.native.max_item_work, b.native.max_item_work);
    assert_eq!(a.native.masks.max_visits_per_run, b.native.masks.max_visits_per_run);
    assert_eq!(a.native.max_sampler_bytes, b.native.max_sampler_bytes);
    assert!(b.buffer_bytes > a.buffer_bytes);
    let mut exact = job;
    exact.max_work = JobWork::from_int4(a.native.max_item_work, a.native.masks.max_visits_per_item);
    c.args.lifetime_limits(&args, admitted, exact).unwrap();
    for axis in 0..6 {
        let mut short = exact;
        match axis {
            0 => short.max_work.model.forward_positions -= 1,
            1 => short.max_work.model.projected_logits -= 1,
            2 => short.max_work.model.attention_pairs -= 1,
            3 => short.max_work.model.projections.dot_products -= 1,
            4 => short.max_work.model.projections.multiply_accumulates -= 1,
            _ => short.max_work.mask_node_visits -= 1,
        }
        assert_eq!(c.args.lifetime_limits(&args, admitted, short).unwrap_err().code,
            "sampled_document_job_lifetime_work");
    }
}

#[test]
fn complete_map_preparation_and_output_are_charged_beyond_the_leaf_budget() {
    for snapshot in [false, true] { for flags in [
        &[][..], &["--compute-threads", "4"][..], &["--prefill-rows", "4"][..],
        &["--input-encoding", "safe-bpe-v1", "--reuse-prefill"][..],
    ] {
        let (c, args) = command("start", snapshot, flags);
        let admitted = c.args.validate(&args).unwrap();
        let job = lifetime(&args, &c, admitted);
        let limits = c.args.lifetime_limits(&args, admitted, job).unwrap();
        let common = limits.common;
        let prepared = sampled_sentiment_document_preparation_bytes(args.group_width,
            limits.task, limits.planning, limits.mapping,
            &c.args.document.sampled.options(&args).unwrap(),
            &c.args.document.input_encoding(&args).unwrap()).unwrap();
        assert_eq!(limits.preparation_reserve_bytes, prepared + SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64);
        assert!(prepared > common.preparation_bytes);
        assert_eq!(limits.memory.max_output_bytes, limits.mapping.max_result_bytes);
        assert!(limits.memory.max_output_bytes > limits.task.max_output_bytes);
        assert_eq!(limits.execution_bytes, limits.buffer_bytes + limits.memory.native.kv_bytes
            + limits.memory.workspace_bytes + limits.memory.output_reservation_bytes
            + common.compute.workspace_bytes);
        let loading = if snapshot { common.snapshot_need.scratch_bytes }
            else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
        assert_eq!(common.minimum_memory_bytes, common.preparation_bytes
            + common.load.weight_payload_bytes + ALLOCATOR_BYTES
            + loading.max(limits.execution_bytes) + CLEANUP_BYTES);
        let mut short = admitted; short.common.memory_bytes = common.minimum_memory_bytes - 1;
        assert_eq!(c.args.lifetime_limits(&args, short, job).unwrap_err().exit,
            ErrorCode::AdmissionOrResourceLimit);
        short.common.memory_bytes += 1;
        c.args.lifetime_limits(&args, short, job).unwrap();
    } }
}

#[test]
fn stored_input_and_both_leaf_and_complete_result_limits_are_authoritative() {
    let (c, args) = command("start", false, &[]);
    let admitted = c.args.validate(&args).unwrap();
    let job = lifetime(&args, &c, admitted);
    let mut narrow = job; narrow.max_input_bytes_per_item /= 2;
    let accepted = c.args.lifetime_limits(&args, admitted, narrow).unwrap();
    assert_eq!(accepted.mapping.chunks.max_input_bytes, narrow.max_input_bytes_per_item);
    assert_eq!(accepted.planning.max_input_bytes,
        admitted.planning.max_input_bytes.min(narrow.max_input_bytes_per_item));
    let mut smaller_chunk = admitted; smaller_chunk.planning.max_input_bytes = 1024;
    let preserved = c.args.lifetime_limits(&args, smaller_chunk, job).unwrap();
    assert_eq!(preserved.planning.max_input_bytes, 1024);
    for kind in 0..3 {
        let mut wrong = job;
        match kind { 0 => wrong.max_input_bytes_per_item += 1,
            1 => wrong.max_id_bytes = 257, _ => wrong.max_result_bytes -= 1 }
        assert_eq!(c.args.lifetime_limits(&args, admitted, wrong).unwrap_err().code,
            if kind == 2 { "sampled_document_job_result_ceiling" }
            else { "sampled_document_job_input_ceiling" });
    }
    let mut small_map = admitted;
    small_map.mapping.max_result_bytes = 1;
    let mut short_leaf = job; short_leaf.max_result_bytes = admitted.common.result_bytes - 1;
    assert_eq!(c.args.lifetime_limits(&args, small_map, short_leaf).unwrap_err().code,
        "sampled_document_job_result_ceiling");
}

#[test]
fn reuse_changes_no_cold_work_or_memory_allowance() {
    for snapshot in [false, true] { for operation in ["start", "resume"] {
        for flags in [&[][..], &["--compute-threads", "4"][..], &["--prefill-rows", "4"][..]] {
            let (cold, cold_args) = command(operation, snapshot, flags);
            let mut reused_flags = flags.to_vec(); reused_flags.push("--reuse-prefill");
            let (reused, args) = command(operation, snapshot, &reused_flags);
            let cold_admitted = cold.args.validate(&cold_args).unwrap();
            let admitted = reused.args.validate(&args).unwrap();
            let job = lifetime(&args, &reused, admitted);
            let a = cold.args.lifetime_limits(&cold_args, cold_admitted, job).unwrap();
            let b = reused.args.lifetime_limits(&args, admitted, job).unwrap();
            assert_eq!(a.native.max_item_work, b.native.max_item_work);
            assert_eq!(a.native.max_model_work, b.native.max_model_work);
            assert_eq!(a.native.masks.max_visits_per_item, b.native.masks.max_visits_per_item);
            assert_eq!(a.native.masks.max_visits_per_run, b.native.masks.max_visits_per_run);
            assert_eq!(a.task, b.task);
            assert_eq!(a.memory, b.memory);
            assert_eq!(a.preparation_reserve_bytes, b.preparation_reserve_bytes);
            assert_eq!(a.execution_bytes, b.execution_bytes);
            assert_eq!(a.common.minimum_memory_bytes, b.common.minimum_memory_bytes);
        }
    } }
}

#[test]
fn explicit_retained_replay_prices_one_record_and_does_not_read_supplied_input() {
    let (original, args) = command("resume", false, &[]);
    let admitted = original.args.validate(&args).unwrap();
    let job = lifetime(&args, &original, admitted);
    let ordinary = original.args.lifetime_limits(&args, admitted, job).unwrap();
    for (operation, flags) in [
        ("start", &["--spool-input", "--max-retained-input-mib", "8"][..]),
        ("resume", &["--use-retained-input", "--max-retained-input-mib", "8"][..]),
    ] {
        let (retained, retained_args) = command(operation, false, flags);
        let streamed = retained.args.lifetime_limits(&retained_args, admitted, job).unwrap();
        assert_eq!(ordinary.buffer_bytes - streamed.buffer_bytes, 2 * job.max_snapshot_bytes);
        assert_eq!(ordinary.native.max_model_work, streamed.native.max_model_work);
        assert_eq!(retained.args.storage.input_retention.input_memory().unwrap(), JobInputMemory::RetainedRecord);
        let mut larger = job; larger.max_snapshot_bytes *= 100;
        let large = retained.args.lifetime_limits(&retained_args, admitted, larger).unwrap();
        assert_eq!(streamed.buffer_bytes, large.buffer_bytes);
        if operation == "resume" {
            let mut reader = job_cli::population_reader(retained.mode, retained.args.storage.input_retention,
                &retained_args.input, NoIo, IO_BUFFER_BYTES).unwrap();
            assert_eq!(reader.read(&mut [0]).unwrap(), 0);
        }
    }
}

#[test]
fn invalid_storage_mapping_schedule_encoding_and_snapshot_save_refuse_before_io() {
    for fault in 0..9 {
        let (mut c, args) = command("start", false, &[]);
        match fault {
            0 => c.args.storage.store_results = false,
            1 => c.args.storage.job_dir = PathBuf::new(),
            2 => c.args.storage.key_file = PathBuf::from("-"),
            3 => c.args.storage.limits_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0,
            5 => c.args.storage.max_input_lines = 0,
            6 => c.args.storage.journal_memory_mib = 0,
            7 => c.args.storage.serialization_memory_mib = 0,
            _ => c.args.storage.max_stream_mib = u64::MAX,
        }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_sampled_document_job_storage");
    }
    for flags in [
        &["--save-snapshot", "private-new-snapshot"][..], &["--samples", "0"][..],
        &["--max-chunks", "0"][..], &["--max-chunks", "257"][..],
        &["--max-map-result-bytes", "0"][..], &["--max-tokenizer-calls", "0"][..],
        &["--max-tokenizer-input-bytes", "8"][..],
        &["--compute-threads", "4", "--prefill-rows", "4"][..],
        &["--max-mask-node-visits", "0"][..], &["--max-forward-positions", "0"][..],
    ] {
        let (c, args) = command("start", false, flags);
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
    let (c, args) = command("start", false, &["--use-retained-input", "--max-retained-input-mib", "8"]);
    assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "job_input_retention_options");
}
