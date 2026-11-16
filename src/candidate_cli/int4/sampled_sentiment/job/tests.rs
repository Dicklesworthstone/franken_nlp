//! Pure job admission and parser regressions; none requires model weights.
use super::*;
use crate::jobs::{JobInputMemory, JobWork};

const SEED: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
fn argv(operation: &str, route: usize) -> Vec<String> {
    let mut words = ["sentiment-sampled-job", operation, "--memory-mib", "131072", "--seed", SEED,
        "--job-dir", "private-job", "--key-file", "private-key", "--limits", "private-limits",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect::<Vec<_>>();
    let model: &[&str] = match route {
        0 => &["--source-dir", "private-source", "--source-manifest", "private-manifest"],
        1 => &["--source-dir", "private-source", "--source-manifest", "private-manifest",
            "--save-snapshot", "private-new-snapshot"],
        2 => &["--snapshot", "private-snapshot", "--expected-model-sha256", SEED],
        _ => panic!("unknown fixture model route"),
    };
    words.extend(model.iter().map(|word| (*word).to_owned()));
    words
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
pub(super) fn command(operation: &str, route: usize, flags: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation, route);
    words.extend(flags.iter().map(|word| (*word).to_owned()));
    parse(words).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, command: &JobCommand, common: Int4Limits) -> JobLimits {
    let work = command.args.sampled.work_ceiling();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork::from_int4(work.checked_add(work).unwrap(),
            2 * command.args.sampled.max_mask_node_visits) }
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
fn one_off_prefill_reuse_is_refused_before_job_limits_key_or_population_io() {
    for route in 0..3 { for operation in ["start", "resume"] {
        let (c, args) = command(operation, route, &["--reuse-prefill"]);
        assert!(c.args.sampled.reuse_prefill);
        assert_eq!(c.args.validate(&args).unwrap_err().code,
            "sampled_sentiment_job_prefill_reuse_unavailable");
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "sampled_sentiment_job_prefill_reuse_unavailable");
    } }
}

#[test]
fn source_snapshot_and_save_keep_exact_sampler_and_owned_dispatch() {
    for route in 0..3 {
        for operation in ["start", "resume"] {
            let (c, args) = command(operation, route, &["population.ndjson", "--samples", "7",
                "--temperature-milli", "800", "--top-p-ppm", "950000", "--top-k", "9",
                "--input-encoding", "safe-bpe-v1", "--compute-threads", "4"]);
            assert_eq!(c.mode.name(), operation);
            assert_eq!(args.input, PathBuf::from("population.ndjson"));
            let options = c.args.sampled.options(&args).unwrap();
            assert_eq!((options.samples, options.temperature_milli, options.top_p_ppm, options.top_k),
                (7, 800, 950000, Some(9)));
            let common = c.args.validate(&args).unwrap();
            assert_eq!(common.compute.threads, 4);
            assert!(common.prefill.is_none());
            assert_eq!(common.snapshot_pin.is_some(), route == 2);
            assert_eq!(args.save_snapshot.is_some(), route == 1);
            assert!(!args.input_encoding.is_default());
        }
    }
    for operation in ["start", "resume"] {
        let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()];
        words.extend(argv(operation, 0));
        let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
        let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong candidate dispatcher"); };
        assert_eq!(c.task, Int4Task::SampledSentimentJob);
        assert!(matches!(&c.job, Some(Int4JobCommand::SampledSentiment(_))));
        assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn lifetime_work_and_masks_never_scale_with_items_attempts_or_restart() {
    let (mut c, args) = command("resume", 0, &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, &c, common);
    let a = c.args.lifetime_limits(&args, common, job).unwrap();
    assert_eq!(a.native.max_model_work, job.max_work.int4_model_work());
    assert_eq!(a.native.masks.max_visits_per_run, job.max_work.mask_node_visits);
    assert_eq!(a.native.max_item_work, c.args.sampled.work_ceiling());
    let mut more = job; more.max_items += 20; more.max_attempts += 40;
    c.args.storage.max_stream_mib = 1024;
    c.args.storage.max_input_lines = 1_000_000;
    let b = c.args.lifetime_limits(&args, common, more).unwrap();
    assert_eq!(a.native.max_model_work, b.native.max_model_work);
    assert_eq!(a.native.max_item_work, b.native.max_item_work);
    assert_eq!(a.native.masks.max_visits_per_run, b.native.masks.max_visits_per_run);
    assert_eq!(a.native.max_sampler_bytes, b.native.max_sampler_bytes);
    assert!(b.buffer_bytes > a.buffer_bytes);
}

#[test]
fn every_persisted_native_and_mask_axis_must_fit_one_complete_item_allowance() {
    let (c, args) = command("start", 0, &[]);
    let common = c.args.validate(&args).unwrap();
    let mut exact = lifetime(&args, &c, common);
    exact.max_work = JobWork::from_int4(c.args.sampled.work_ceiling(), c.args.sampled.max_mask_node_visits);
    c.args.lifetime_limits(&args, common, exact).unwrap();
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
        assert_eq!(c.args.lifetime_limits(&args, common, short).unwrap_err().code,
            "sampled_sentiment_job_lifetime_work");
    }
}

#[test]
fn stored_source_result_and_sample_address_limits_are_authoritative() {
    let (c, args) = command("start", 0, &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, &c, common);
    let mut narrow = job; narrow.max_input_bytes_per_item /= 2;
    let accepted = c.args.lifetime_limits(&args, common, narrow).unwrap();
    assert_eq!(accepted.planning.max_input_bytes, narrow.max_input_bytes_per_item);
    assert_eq!(accepted.task, c.args.sampled.budget(&args, common, 4).unwrap());
    assert_eq!(accepted.native.max_model_work, job.max_work.int4_model_work());
    for kind in 0..3 {
        let mut wrong = job;
        match kind { 0 => wrong.max_input_bytes_per_item += 1,
            1 => wrong.max_id_bytes = 257, _ => wrong.max_result_bytes -= 1 }
        let failure = c.args.lifetime_limits(&args, common, wrong).unwrap_err();
        assert_eq!(failure.code, if kind == 2 { "sampled_sentiment_job_result_ceiling" }
            else { "sampled_sentiment_job_input_ceiling" });
    }
}

#[test]
fn exact_peak_prices_population_native_sampler_output_and_one_physical_schedule() {
    for route in 0..3 {
        for flags in [vec![], vec!["--compute-threads", "4"], vec!["--prefill-rows", "4"]] {
            let (c, args) = command("start", route, &flags);
            let common = c.args.validate(&args).unwrap();
            let job = lifetime(&args, &c, common);
            let limits = c.args.lifetime_limits(&args, common, job).unwrap();
            let output = 4 * limits.task.max_output_bytes + 4096;
            assert_eq!(limits.execution_bytes, limits.buffer_bytes + common.native.kv_bytes
                + common.native.rope_bytes + common.native.scratch_payload_bound + ALLOCATOR_BYTES
                + limits.native.max_sampler_bytes + output
                + common.prefill.map_or(0, |prefill| prefill.max_extra_scratch_bytes)
                + common.compute.workspace_bytes);
            let load = if route == 2 { common.snapshot_need.scratch_bytes }
                else { common.source_preparation_bytes + common.load.scratch_payload_bytes };
            let save = if route == 1 { common.snapshot_need.scratch_bytes } else { 0 };
            assert_eq!(limits.common.minimum_memory_bytes, common.preparation_bytes
                + common.load.weight_payload_bytes + ALLOCATOR_BYTES
                + load.max(save).max(limits.execution_bytes) + CLEANUP_BYTES);
            let mut short = common; short.memory_bytes = limits.common.minimum_memory_bytes - 1;
            assert_eq!(c.args.lifetime_limits(&args, short, job).unwrap_err().exit,
                ErrorCode::AdmissionOrResourceLimit);
            short.memory_bytes += 1;
            c.args.lifetime_limits(&args, short, job).unwrap();
        }
    }
}

#[test]
fn retained_replay_uses_one_record_and_never_reads_supplied_input() {
    let (original, args) = command("resume", 0, &[]);
    let common = original.args.validate(&args).unwrap();
    let job = lifetime(&args, &original, common);
    let ordinary = original.args.lifetime_limits(&args, common, job).unwrap();
    let (retained, retained_args) = command("resume", 0,
        &["--use-retained-input", "--max-retained-input-mib", "8"]);
    let streamed = retained.args.lifetime_limits(&retained_args, common, job).unwrap();
    assert_eq!(ordinary.buffer_bytes - streamed.buffer_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(ordinary.native.max_model_work, streamed.native.max_model_work);
    assert_eq!(ordinary.native.masks.max_visits_per_run, streamed.native.masks.max_visits_per_run);
    assert_eq!(retained.args.storage.input_retention.input_memory().unwrap(), JobInputMemory::RetainedRecord);
    let mut reader = job_cli::population_reader(retained.mode, retained.args.storage.input_retention,
        &retained_args.input, NoIo, IO_BUFFER_BYTES).unwrap();
    let mut byte = [0]; assert_eq!(reader.read(&mut byte).unwrap(), 0);
    let mut larger = job; larger.max_snapshot_bytes *= 100;
    let large = retained.args.lifetime_limits(&retained_args, common, larger).unwrap();
    assert_eq!(streamed.buffer_bytes, large.buffer_bytes);
    let (creating, create_args) = command("start", 0, &["--spool-input", "--max-retained-input-mib", "8"]);
    let create = creating.args.lifetime_limits(&create_args, common, job).unwrap();
    assert_eq!(streamed.buffer_bytes, create.buffer_bytes);
}

#[test]
fn storage_consent_and_invalid_configuration_refuse_before_private_io() {
    for fault in 0..11 {
        let (mut c, args) = command("start", 0, &[]);
        match fault {
            0 => c.args.storage.store_results = false,
            1 => c.args.storage.job_dir = PathBuf::new(),
            2 => c.args.storage.key_file = PathBuf::from("-"),
            3 => c.args.storage.limits_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0,
            5 => c.args.storage.max_input_lines = 0,
            6 => c.args.storage.journal_memory_mib = 0,
            7 => c.args.storage.serialization_memory_mib = 0,
            8 => c.args.storage.max_stream_mib = u64::MAX,
            9 => c.args.storage.job_dir = PathBuf::from("-"),
            _ => c.args.storage.key_file = PathBuf::new(),
        }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_sampled_sentiment_job_storage");
    }
    for flags in [vec!["--samples", "0"], vec!["--samples", "33"],
        vec!["--prefill-rows", "0"], vec!["--prefill-rows", "65"],
        vec!["--compute-threads", "4", "--prefill-rows", "4"],
        vec!["--max-mask-node-visits", "0"], vec!["--max-forward-positions", "0"],
        vec!["--max-tokenizer-pair-checks", "10"]] {
        let (c, args) = command("start", 0, &flags);
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
    for journal in [false, true] {
        let (mut c, args) = command("start", 0, &[]);
        if journal { c.args.storage.journal_memory_mib = u64::MAX; }
        else { c.args.storage.serialization_memory_mib = u64::MAX; }
        assert_eq!(c.args.validate(&args).unwrap_err().code, "sampled_sentiment_job_memory_options");
    }
}

#[test]
fn sampling_controls_and_materialization_preserve_exact_selected_options() {
    for operation in ["start", "resume"] {
        let (selected, args) = command(operation, 0, &["--materialize", "--samples", "3",
            "--max-new-tokens", "96", "--max-justification-chars", "100",
            "--temperature-milli", "750", "--top-k", "10", "--top-p-ppm", "900000"]);
        selected.args.validate(&args).unwrap();
        assert!(selected.args.storage.store_results && selected.args.storage.materialize);
        let options = selected.args.sampled.options(&args).unwrap();
        assert_eq!(options.effective_seed, parse_seed(SEED).unwrap());
        assert_eq!((options.samples, options.max_sample_tokens, options.max_justification_chars), (3, 96, 100));
        assert_eq!((options.temperature_milli, options.top_k, options.top_p_ppm), (750, Some(10), 900000));
    }
    for flags in [vec!["--max-new-tokens", "0"], vec!["--temperature-milli", "0"]] {
        let (selected, args) = command("start", 0, &flags);
        assert!(selected.args.validate(&args).is_err());
    }
}

#[test]
fn compact_encoding_quotes_both_the_exact_workspace_and_private_durable_recipe() {
    use crate::batch::sentiment::sampled::{sampled_sentiment_batch_preparation_bytes,
        SAMPLED_SENTIMENT_JOB_RECIPE_BYTES};
    let (selected, args) = command("start", 0, &["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-pair-checks", "1048576", "--max-tokenizer-pair-bytes", "16777216"]);
    let common = selected.args.validate(&args).unwrap();
    let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens).unwrap();
    let task = selected.args.sampled.budget(&args, common, SentimentAxis::ALL.len()).unwrap();
    let planning = selected.args.sampled.task_limits(&args);
    let compact = sampled_sentiment_job_preparation_bytes(task, planning, &encoding).unwrap();
    let byte = sampled_sentiment_job_preparation_bytes(task, planning,
        &crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback).unwrap();
    assert!(compact > byte && compact <= common.preparation_bytes);
    assert_eq!(compact, sampled_sentiment_batch_preparation_bytes(task, planning, &encoding).unwrap()
        + SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64);
    let fixed = lifetime(&args, &selected, common);
    let admitted = selected.args.lifetime_limits(&args, common, fixed).unwrap();
    assert!(admitted.buffer_bytes > 2 * fixed.max_snapshot_bytes + common.preparation_bytes);
    assert_eq!(JobWork::from_int4(admitted.native.max_model_work,
        admitted.native.masks.max_visits_per_run), fixed.max_work);
}

#[test]
fn parser_rejects_missing_seed_unsupported_processors_and_ambiguous_retention() {
    assert!(definition().try_get_matches_from(["sentiment-sampled-job"]).is_err());
    let mut no_seed = argv("start", 0);
    let at = no_seed.iter().position(|word| word == "--seed").unwrap(); no_seed.drain(at..at + 2);
    assert!(parse(no_seed).is_err());
    for flags in [vec!["--logprobs"], vec!["--stop", "text"], vec!["--max-output-bytes", "10"],
        vec!["--max-output-bytes", "65536"], vec!["--min-new-tokens", "0"],
        vec!["--max-requests", "3"], vec!["--max-total-output-bytes", "65536"],
        vec!["--defaults", "private.json"], vec!["--task", "sentiment"], vec!["--cohort-heads", "2"],
        vec!["--discard-uncommitted-tail"], vec!["--use-retained-input"],
        vec!["--spool-input", "--use-retained-input", "--max-retained-input-mib", "8"]] {
        let mut words = argv("start", 0); words.extend(flags.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
    for (mode, flags) in [("start", vec!["--use-retained-input", "--max-retained-input-mib", "8"]),
        ("resume", vec!["--spool-input", "--max-retained-input-mib", "8"])] {
        let (c, args) = command(mode, 0, &flags);
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "job_input_retention_options");
    }
    let (resume, _) = command("resume", 0, &["--discard-uncommitted-tail"]);
    assert!(matches!(resume.mode, RunMode::Resume { discard_uncommitted: true }));
    assert!(matches!(command("resume", 0, &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    let mut supplied = argv("resume", 0);
    supplied.extend(["private/population.ndjson", "--use-retained-input", "--max-retained-input-mib", "8"]
        .into_iter().map(str::to_owned));
    assert!(parse(supplied).is_err());
    assert!(definition().try_get_matches_from(["sentiment-sampled-job", "verify"]).is_err());
}

#[test]
fn error_receipt_is_metadata_only_and_keeps_the_exact_cancellation_kind() {
    let report = ErrorReport { protocol: "fnlp-candidate-int4-sentiment-sampled-job-v1",
        schema_version: 1, operation: "resume", status: "error", code: "sampled_sentiment_budget",
        exit_code: ErrorCode::BudgetOrTimeout,
        cancellation: Some(crate::native_engine::decode::DecodeCancellationKind::CostBudget),
        durable_progress_may_exist: true,
        recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
    let mut output = Vec::new(); publish(&report, job_cli::REPORT_BYTES, &mut output).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(value["durable_progress_may_exist"], true);
    assert_eq!(value["cancellation"], serde_json::to_value(
        crate::native_engine::decode::DecodeCancellationKind::CostBudget).unwrap());
    let text = String::from_utf8(output).unwrap();
    for private in [SEED, "effective_seed", "private-key", "private-job", "prompt_digest"] {
        assert!(!text.contains(private));
    }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unsupported_build_refuses_without_config_key_model_or_population_io() {
    for operation in ["start", "resume"] {
        let (c, args) = command(operation, 0, &[]);
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "sampled_sentiment_job_profile_unavailable");
    }
}
