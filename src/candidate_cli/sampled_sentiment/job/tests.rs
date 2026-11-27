//! CLI consent and admission regressions; no model, key or population IO.
use super::*;
use crate::jobs::{JobInputMemory, JobWork};

fn parsed(operation: &str, extra: &[&str]) -> Result<JobCommand, clap::Error> {
    let seed = "4a".repeat(32);
    let mut argv = vec!["sentiment-sampled-job", operation,
        "--model", "missing-private-model.fnlpq", "--memory-mib", "16384",
        "--store-results", "--job-dir", "missing-private-job",
        "--job-id", "00112233445566778899aabbccddeeff00",
        "--key-file", "missing-private-key", "--limits", "missing-private-limits.json"];
    if !extra.contains(&"--seed") { argv.extend(["--seed", seed.as_str()]); }
    argv.extend_from_slice(extra);
    JobCommand::from_matches(&definition().try_get_matches_from(argv)?)
}
fn command(operation: &str, extra: &[&str]) -> JobCommand { parsed(operation, extra).unwrap() }
pub(in crate::candidate_cli) fn arguments(operation: &str, extra: &[&str]) -> (RunMode, JobArgs) {
    let command = command(operation, extra); (command.mode, command.args)
}
pub(in crate::candidate_cli) fn lifetime(args: &JobArgs, limits: Limits) -> JobLimits {
    let one = args.sampled.work_ceiling();
    JobLimits { max_items: 10, max_id_bytes: 128, max_input_bytes_per_item: 65_536,
        max_snapshot_bytes: 8 << 20, max_result_bytes: limits.result_bytes,
        max_spool_bytes: 32 << 20, max_materialized_bytes: 32 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 20, max_work: JobWork {
            model: one.checked_add(one).unwrap(),
            mask_node_visits: args.sampled.max_mask_node_visits * 2,
        } }
}
pub(in crate::candidate_cli) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read before admission"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("private output write before admission"); }
    fn flush(&mut self) -> io::Result<()> { panic!("private output flush before admission"); }
}

#[test]
fn both_operations_reach_owned_dispatch_with_explicit_sampler_and_storage() {
    let seed = "4a".repeat(32);
    for operation in ["start", "resume"] {
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "sentiment-sampled-job", operation, "--model", "m",
            "--memory-mib", "16384", "--seed", &seed, "--store-results",
            "--job-dir", "job", "--job-id", "00112233445566778899aabbccddeeff00",
            "--key-file", "key", "--limits", "limits.json",
            "--samples", "7", "--top-k", "9", "--top-p-ppm", "950000",
            "--compute-threads", "4", "--prefill-rows", "4",
        ]).unwrap();
        let CandidateCommand::SampledSentimentJob(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("owned sampled route required"); };
        let (common, limits) = command.args.validate().unwrap();
        let options = command.args.sampled.options().unwrap();
        assert_eq!(options.effective_seed, [0x4a; 32]);
        assert_eq!((options.samples, options.top_k, options.top_p_ppm), (7, Some(9), 950_000));
        assert_eq!(limits.compute.threads, 4);
        assert_eq!(limits.compute, common.compute.admit_prefill(command.args.sampled.prefill().unwrap()).unwrap());
        assert!(!command.args.storage.input_retention.requested());
    }
    assert!(parsed("start", &["--discard-uncommitted-tail"]).is_err());
    assert!(parsed("start", &["--seed", "00"]).is_ok());
    let mut missing = command("start", &[]); missing.args.sampled.seed.clear();
    assert_eq!(missing.execute_owned(NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn missing_consent_invalid_caps_and_overflow_refuse_before_private_files_or_streams() {
    for case in 0..14 {
        let mut command = command("start", &[]);
        match case {
            0 => command.args.storage.store_results = false,
            1 => command.args.storage.job_dir = PathBuf::from("-"),
            2 => command.args.storage.key_file = PathBuf::from("-"),
            3 => command.args.storage.limits_file = PathBuf::from("-"),
            4 => command.args.storage.max_stream_mib = u64::MAX,
            5 => command.args.storage.max_input_lines = 0,
            6 => command.args.storage.journal_memory_mib = u64::MAX,
            7 => command.args.storage.serialization_memory_mib = u64::MAX,
            8 => command.args.sampled.memory_mib = 1,
            9 => command.args.sampled.prefill_rows = Some(0),
            10 => command.args.sampled.samples = usize::MAX,
            11 => command.args.storage.input_retention.max_retained_input_mib = Some(1),
            12 => {
                command.args.storage.input_retention.spool_input = true;
                command.args.storage.input_retention.max_retained_input_mib = Some(u64::MAX);
            }
            _ => command.args.sampled.max_mask_node_visits = 0,
        }
        assert_eq!(command.execute_owned(NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
    let command = command("start", &["--max-total-tokenizer-pair-checks", "1"]);
    assert_eq!(command.execute_owned(NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
}

#[test]
fn lifetime_native_mask_input_and_result_caps_are_never_multiplied_by_item_or_attempt_count() {
    let (_, args) = arguments("start", &[]);
    let (_, limits) = args.validate().unwrap();
    let job = lifetime(&args, limits);
    let envelope = args.lifetime_limits(limits, job).unwrap();
    assert_eq!(envelope.max_item_work, args.sampled.work_ceiling());
    assert_eq!(envelope.masks.max_visits_per_run, job.max_work.mask_node_visits);
    for axis in 0..9 {
        let mut short = job;
        short.max_work.model = envelope.max_item_work;
        short.max_work.mask_node_visits = envelope.masks.max_visits_per_item;
        short.max_items = 100_000; short.max_attempts = 1_000_000;
        match axis {
            0 => short.max_work.model.forward_positions -= 1,
            1 => short.max_work.model.projected_logits -= 1,
            2 => short.max_work.model.attention_pairs -= 1,
            3 => short.max_work.model.projections.dot_products -= 1,
            4 => short.max_work.model.projections.multiply_accumulates -= 1,
            5 => short.max_work.mask_node_visits -= 1,
            6 => short.max_input_bytes_per_item = args.sampled.max_input_bytes + 1,
            7 => short.max_result_bytes -= 1,
            _ => short.max_id_bytes = 257,
        }
        assert!(args.lifetime_limits(limits, short).is_err());
    }
}

#[test]
fn input_retention_requires_separate_consent_and_replay_never_reads_a_fallback() {
    for flags in [
        vec!["--spool-input"],
        vec!["--use-retained-input"],
        vec!["--spool-input", "--use-retained-input", "--max-retained-input-mib", "8"],
        vec!["private-population.ndjson", "--use-retained-input", "--max-retained-input-mib", "8"],
    ] { assert!(parsed("resume", &flags).is_err()); }
    for (mode, flags) in [
        ("start", vec!["--use-retained-input", "--max-retained-input-mib", "8"]),
        ("resume", vec!["--spool-input", "--max-retained-input-mib", "8"]),
    ] {
        assert_eq!(command(mode, &flags).execute_owned(NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
    let (mode, args) = arguments("resume", &["--use-retained-input", "--max-retained-input-mib", "8"]);
    assert_eq!(args.storage.input_retention.input_memory().unwrap(), JobInputMemory::RetainedRecord);
    let mut reader = job_cli::population_reader(mode, args.storage.input_retention,
        &args.sampled.input, NoIo, 64).unwrap();
    assert_eq!(reader.read(&mut [0_u8; 1]).unwrap(), 0);
    let (_, normal) = arguments("resume", &[]);
    assert_eq!(normal.storage.input_retention.input_memory().unwrap(), JobInputMemory::CompletePopulation);
}

#[test]
fn compact_tokenizer_totals_are_bounded_and_add_the_private_invocation_owner() {
    let (_, args) = arguments("start", &["--input-encoding", "safe-bpe-v1",
        "--max-total-tokenizer-input-bytes", "1000", "--max-total-tokenizer-output-tokens", "1000",
        "--max-total-tokenizer-pair-checks", "2000", "--max-total-tokenizer-pair-bytes", "3000"]);
    let (common, limits) = args.validate().unwrap();
    let envelope = args.lifetime_limits(limits, lifetime(&args, limits)).unwrap();
    assert_eq!(args.encoding_work_limits().unwrap(), Some(SafeBpeWorkLimits {
        max_input_bytes: 1000, max_output_tokens: 1000, max_pair_checks: 2000, max_pair_bytes: 3000 }));
    let base = sampled_sentiment_job_preparation_bytes(envelope.task, envelope.planning,
        &common.input_encoding_policy(limits).unwrap()).unwrap();
    assert_eq!(envelope.preparation_bytes, base + 1024);
    let (_, byte) = arguments("start", &[]);
    assert_eq!(byte.encoding_work_limits().unwrap(), None);
}
