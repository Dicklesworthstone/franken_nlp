use super::*;
use crate::jobs::{JobInputMemory, JobWork};

fn argv(operation: &str) -> Vec<String> {
    let mut words = ["sentiment-sampled-job", operation, "--source-dir", "sources",
        "--source-manifest", "manifest.json", "--memory-mib", "65536",
        "--job-dir", "private/job", "--key-file", "private/key", "--limits", "limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results", "--seed"]
        .into_iter().map(str::to_owned).collect::<Vec<_>>();
    words.push("ab".repeat(32));
    words
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
fn command(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation);
    words.extend(extra.iter().map(|value| (*value).to_owned()));
    parse(words).unwrap()
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(job.sampled.work_ceiling(), job.sampled.max_mask_node_visits) }
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused job read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused job flushed output"); }
}

#[test]
fn sampled_start_and_resume_require_explicit_seed_and_support_pinned_snapshots() {
    for operation in ["start", "resume"] {
        let (command, args) = command(operation, &["--materialize"]);
        assert_eq!(command.mode.name(), operation);
        assert!(command.args.storage.store_results && command.args.storage.materialize);
        assert!(command.args.validate(&args).unwrap().snapshot_pin.is_none());
        assert_eq!(command.args.sampled.options(&args).unwrap().effective_seed, [0xab; 32]);
        let mut words = argv(operation);
        let at = words.iter().position(|s| s == "--source-dir").unwrap();
        words.drain(at..at + 4);
        words.extend(["--snapshot", "candidate.int4", "--expected-model-sha256"]
            .into_iter().map(str::to_owned));
        words.push("cd".repeat(32));
        let (command, args) = parse(words).unwrap();
        assert!(command.args.validate(&args).unwrap().snapshot_pin.is_some());
        let mut missing = argv(operation);
        let at = missing.iter().position(|s| s == "--seed").unwrap();
        missing.drain(at..at + 2);
        assert!(parse(missing).is_err());
    }
}

#[test]
fn job_is_routed_through_owned_dispatch_and_not_the_single_request_handler() {
    for operation in ["start", "resume"] {
        let mut words = vec!["source-int4".to_owned()];
        words.extend(argv(operation));
        let matches = crate::candidate_cli::int4::definition().try_get_matches_from(words).unwrap();
        let command = Int4Command::from_matches(&matches).unwrap();
        assert!(matches!(command.task, Int4Task::SampledSentimentJob));
        assert!(matches!(command.job, Some(Int4JobCommand::SampledSentiment(_))));
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn consent_paths_and_storage_options_are_checked_before_any_private_io() {
    for case in 0..7 {
        let (mut command, args) = command("start", &[]);
        match case {
            0 => command.args.storage.store_results = false,
            1 => command.args.storage.job_dir = PathBuf::from("-"),
            2 => command.args.storage.key_file = PathBuf::new(),
            3 => command.args.storage.limits_file = PathBuf::from("-"),
            4 => command.args.storage.max_stream_mib = 0,
            5 => command.args.storage.max_input_lines = 0,
            _ => command.args.storage.serialization_memory_mib = 0,
        }
        assert_eq!(command.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_sampled_sentiment_job_storage");
    }
    let (mut command, args) = command("start", &[]);
    command.args.storage.journal_memory_mib = u64::MAX;
    assert_eq!(command.args.validate(&args).unwrap_err().code, "sampled_sentiment_job_memory");
}

#[test]
fn sampling_controls_are_explicit_and_unrelated_generation_stream_controls_refuse() {
    let (selected, args) = command("start", &["--samples", "3", "--max-new-tokens", "96",
        "--max-justification-chars", "100", "--temperature-milli", "750",
        "--top-k", "10", "--top-p-ppm", "900000"]);
    selected.args.validate(&args).unwrap();
    let options = selected.args.sampled.options(&args).unwrap();
    assert_eq!((options.samples, options.max_sample_tokens, options.max_justification_chars), (3, 96, 100));
    assert_eq!((options.temperature_milli, options.top_k, options.top_p_ppm), (750, Some(10), 900000));
    for extra in [vec!["--logprobs"], vec!["--max-output-bytes", "65536"],
        vec!["--min-new-tokens", "0"], vec!["--stop", "END"],
        vec!["--max-requests", "3"], vec!["--max-total-output-bytes", "65536"],
        vec!["--defaults", "private.json"], vec!["--task", "sentiment"],
        vec!["--cohort-heads", "2"]] {
        let mut words = argv("start"); words.extend(extra.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
    for extra in [&["--samples", "0"][..], &["--samples", "33"][..],
        &["--max-new-tokens", "0"][..], &["--temperature-milli", "0"][..]] {
        let (command, args) = command("start", extra);
        assert!(command.args.validate(&args).is_err());
    }
}

#[test]
fn all_six_lifetime_work_axes_are_exact_and_do_not_scale_with_attempts() {
    let (mut command, args) = command("start", &[]);
    let common = command.args.validate(&args).unwrap();
    let fixed = lifetime(&command.args, &args, common);
    let first = command.args.lifetime_limits(&args, common, fixed).unwrap();
    assert_eq!(first.native.max_item_work, command.args.sampled.work_ceiling());
    assert_eq!(JobWork::from_int4(first.native.max_model_work, first.native.masks.max_visits_per_run), fixed.max_work);
    let mut larger = fixed; larger.max_items = 101; larger.max_attempts = 1000;
    command.args.storage.max_stream_mib = 1024;
    command.args.storage.max_input_lines = 1_000_000;
    let second = command.args.lifetime_limits(&args, common, larger).unwrap();
    assert_eq!(JobWork::from_int4(second.native.max_model_work, second.native.masks.max_visits_per_run), fixed.max_work);
    assert!(second.buffer_bytes > first.buffer_bytes);
    for axis in 0..6 {
        let mut wrong = fixed;
        match axis {
            0 => wrong.max_work.model.forward_positions -= 1,
            1 => wrong.max_work.model.projected_logits -= 1,
            2 => wrong.max_work.model.attention_pairs -= 1,
            3 => wrong.max_work.model.projections.dot_products -= 1,
            4 => wrong.max_work.model.projections.multiply_accumulates -= 1,
            _ => wrong.max_work.mask_node_visits -= 1,
        }
        assert_eq!(command.args.lifetime_limits(&args, common, wrong).unwrap_err().code,
            "lifetime_sampled_sentiment_work_limits");
    }
}

#[test]
fn each_legal_physical_schedule_keeps_the_semantic_job_limits() {
    let (serial, args) = command("start", &[]);
    let common = serial.args.validate(&args).unwrap();
    let fixed = lifetime(&serial.args, &args, common);
    for extra in [&["--compute-threads", "1"][..], &["--compute-threads", "4"][..],
        &["--prefill-rows", "3"][..]] {
        let (resume, args) = command("resume", extra);
        let quote = resume.args.validate(&args).unwrap();
        let limits = resume.args.lifetime_limits(&args, quote, fixed).unwrap();
        assert_eq!(JobWork::from_int4(limits.native.max_model_work, limits.native.masks.max_visits_per_run), fixed.max_work);
        assert_eq!(resume.args.sampled.options(&args).unwrap(), serial.args.sampled.options(&args).unwrap());
        assert!(quote.minimum_memory_bytes >= common.minimum_memory_bytes);
        if extra[0] == "--prefill-rows" {
            assert!(quote.prefill.unwrap().max_extra_scratch_bytes > 0);
        } else if extra[1] != "1" { assert!(quote.compute.workspace_bytes > 0); }
    }
    let (command, args) = command("start", &["--compute-threads", "2", "--prefill-rows", "2"]);
    assert!(command.args.validate(&args).is_err());
}

#[test]
fn compact_encoding_and_owned_population_have_complete_pre_io_memory_quotes() {
    let (byte, args) = command("start", &[]);
    byte.args.validate(&args).unwrap();
    let (command, args) = command("start", &["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-pair-checks", "1048576", "--max-tokenizer-pair-bytes", "16777216"]);
    let common = command.args.validate(&args).unwrap();
    let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens).unwrap();
    let task = command.args.sampled.budget(&args, common, SentimentAxis::ALL.len()).unwrap();
    let planning = command.args.sampled.task_limits(&args);
    let compact = sampled_sentiment_batch_preparation_bytes(task, planning, &encoding).unwrap();
    let byte = sampled_sentiment_batch_preparation_bytes(task, planning,
        &crate::tokenizer::untrusted::UntrustedEncoding::ByteFallback).unwrap();
    assert!(compact > byte && compact <= common.preparation_bytes);
    let fixed = lifetime(&command.args, &args, common);
    let all = command.args.lifetime_limits(&args, common, fixed).unwrap();
    assert!(all.buffer_bytes > 2 * fixed.max_snapshot_bytes + common.preparation_bytes);
    assert_eq!(all.common.minimum_memory_bytes,
        common.minimum_memory_bytes - common.preparation_bytes + all.buffer_bytes);
    let mut tight = common; tight.memory_bytes = all.common.minimum_memory_bytes - 1;
    assert_eq!(command.args.lifetime_limits(&args, tight, fixed).unwrap_err().code, "job_population_memory");
    tight.memory_bytes += 1;
    assert!(command.args.lifetime_limits(&args, tight, fixed).is_ok());
    let mut wrong = fixed; wrong.max_input_bytes_per_item += 1;
    assert_eq!(command.args.lifetime_limits(&args, common, wrong).unwrap_err().code, "stored_input_ceiling");
    let mut wrong = fixed; wrong.max_result_bytes -= 1;
    assert_eq!(command.args.lifetime_limits(&args, common, wrong).unwrap_err().code, "stored_result_ceiling");
}

#[test]
fn retained_inputs_are_separate_consent_and_resume_never_reads_an_external_population() {
    let (start, args) = command("start", &["--spool-input", "--max-retained-input-mib", "8"]);
    assert!(start.args.storage.input_retention.validate(start.mode, &args.input).is_ok());
    assert_eq!(start.args.storage.input_retention.input_memory().unwrap(), JobInputMemory::RetainedRecord);
    let (resume, args) = command("resume", &["--use-retained-input", "--max-retained-input-mib", "8"]);
    let mut reader = job_cli::population_reader(resume.mode, resume.args.storage.input_retention,
        &args.input, NoIo, IO_BUFFER_BYTES).unwrap();
    let mut byte = [0_u8; 1];
    assert_eq!(reader.read(&mut byte).unwrap(), 0);
    for (operation, extra) in [("start", vec!["--use-retained-input", "--max-retained-input-mib", "8"]),
        ("resume", vec!["--spool-input", "--max-retained-input-mib", "8"])] {
        let (command, args) = command(operation, &extra);
        assert!(command.execute_owned(args, NoIo, &mut NoIo).is_err());
    }
    let mut words = argv("resume");
    words.extend(["private/population.ndjson", "--use-retained-input", "--max-retained-input-mib", "8"]
        .into_iter().map(str::to_owned));
    assert!(parse(words).is_err());
}

#[test]
fn spool_repair_remains_explicit_and_resume_only() {
    assert!(matches!(command("resume", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    let mut words = argv("start"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
    assert!(definition().try_get_matches_from(["sentiment-sampled-job"]).is_err());
    assert!(definition().try_get_matches_from(["sentiment-sampled-job", "verify"]).is_err());
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_storage_profile_refuses_without_reading_or_writing_private_data() {
    let (command, args) = command("start", &[]);
    assert_eq!(command.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
        "owned_sampled_sentiment_job_profile_unavailable");
}
