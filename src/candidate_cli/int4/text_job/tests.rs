//! UNRUN parser, immutable-authority and pre-IO regressions. No model results
//! or durable-success receipts are fabricated by these fixtures.
use super::*;
use crate::{jobs::JobWork, native_engine::{constrained_int4, strict_int8::Int8Work}};

pub(super) fn argv(operation: &str, task: &str) -> Vec<String> {
    ["text-job", operation, task, "--source-dir", "private-source", "--source-manifest", "private-manifest.json",
        "--memory-mib", "65536", "--job-dir", "private-job", "--key-file", "private-key",
        "--limits", "private-limits.json", "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
pub(super) fn command(operation: &str, task: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation, task);
    words.extend(extra.iter().map(|value| (*value).to_owned()));
    parse(words).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, common: Int4Limits) -> JobLimits {
    let work = constrained_int4::planned_work(args.group_width,
        common.max_prompt_tokens, args.max_new_tokens).unwrap();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5, max_work: JobWork::from_int4(work, 0) }
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read before admission"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected result publication"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected result flush"); }
}

#[test]
fn both_tasks_and_operations_preserve_source_storage_and_explicit_snapshot_pins() {
    for operation in ["start", "resume"] {
        for task in ["generate", "chat"] {
            let (c, args) = command(operation, task, &["--materialize"]);
            assert_eq!(c.mode.name(), operation);
            assert_eq!(c.args.task_name(), task);
            assert_eq!(args.input, PathBuf::from("-"));
            assert!(c.args.storage.store_results && c.args.storage.materialize);
            assert_eq!(job_cli::job_id_hex(c.args.storage.job_id), "0123456789abcdef0123456789abcdef");
            assert!(c.args.validate(&args).is_ok());
        }
    }
    let mut words = argv("resume", "chat");
    let source = words.iter().position(|s| s == "--source-dir").unwrap();
    words.drain(source..source + 4);
    words.extend(["--snapshot", "model.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
    words.push("ab".repeat(32));
    let (c, args) = parse(words).unwrap();
    assert_eq!(c.args.validate(&args).unwrap().snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
}

#[test]
fn operations_require_an_explicit_task_and_repair_is_resume_only() {
    for words in [vec!["text-job"], vec!["text-job", "start"], vec!["text-job", "resume"],
        vec!["text-job", "verify"], vec!["text-job", "start", "extract"]] {
        assert!(definition().try_get_matches_from(words).is_err());
    }
    assert!(matches!(command("resume", "generate", &[]).0.mode,
        RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", "chat", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    let mut words = argv("start", "generate"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
}

#[test]
fn storage_consent_paths_and_transport_bounds_refuse_before_any_private_io() {
    for case in 0..8 {
        let (mut c, args) = command("start", "generate", &[]);
        match case {
            0 => c.args.storage.store_results = false,
            1 => c.args.storage.job_dir = PathBuf::from("-"),
            2 => c.args.storage.key_file = PathBuf::new(),
            3 => c.args.storage.limits_file = PathBuf::from("-"),
            4 => c.args.storage.max_stream_mib = 0,
            5 => c.args.storage.max_input_lines = 0,
            6 => c.args.storage.journal_memory_mib = 0,
            _ => c.args.storage.serialization_memory_mib = 0,
        }
        assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "explicit_text_job_storage");
    }
    for bad in ["AB".repeat(16), "ab".repeat(15), "x".repeat(32)] {
        let mut words = argv("start", "generate");
        let at = words.iter().position(|s| s == "--job-id").unwrap() + 1;
        words[at] = bad;
        assert!(parse(words).is_err());
    }
}

#[test]
fn retained_route_never_accepts_ignored_prefill_cohort_or_per_record_policy_flags() {
    for task in ["generate", "chat"] {
        let (c, args) = command("start", task, &["--prefill-rows", "1"]);
        assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "text_job_serial_schedule");
    }
    for extra in [vec!["--cohort-rows", "2"], vec!["--active-rows", "1"], vec!["--task", "generate"],
        vec!["--defaults", "private.json"], vec!["--schema", "private.json"], vec!["--thinking"], vec!["--tools", "tools.json"]] {
        let mut words = argv("start", "generate"); words.extend(extra.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
}

#[test]
fn seeded_generation_options_preserve_stops_penalties_and_logprobs_without_reseeding() {
    let seed = "ab".repeat(32);
    let (c, args) = command("resume", "chat", &["--seed", &seed, "--temperature-milli", "600",
        "--top-k", "20", "--top-p-ppm", "950000", "--logprobs", "--min-new-tokens", "2",
        "--stop", " END\n", "--stop", "終わり", "--ban-token", "7",
        "--repetition-penalty-milli", "1200", "--presence-penalty-milli", "-250",
        "--frequency-penalty-milli", "750", "--logit-bias", "42=-1500"]);
    c.args.validate(&args).unwrap();
    let options = args.options(166_101).unwrap();
    assert!(matches!(options.sampling, GenerationSampling::Seeded { effective_seed, temperature_milli: 600,
        top_k: Some(20), top_p_ppm: 950_000 } if effective_seed == [0xab; 32]));
    assert!(options.capture_logprobs);
    assert_eq!(options.min_new_tokens, 2);
    assert_eq!(options.stop_suffixes, vec![b" END\n".to_vec(), "終わり".as_bytes().to_vec()]);
    assert_eq!(options.banned_token_ids, vec![7]);
    assert_eq!(options.repetition_penalty_milli, 1200);
    assert_eq!(options.presence_penalty_milli, -250);
    assert_eq!(options.frequency_penalty_milli, 750);
    assert_eq!(options.logit_bias_milli.get(&42), Some(&-1500));
    let (_, greedy) = command("start", "generate", &[]);
    assert!(greedy.options(166_101).unwrap() == GenerationOptions::greedy(128, 65_536, 166_101));
}

#[test]
fn invalid_seed_or_generation_controls_refuse_before_loading() {
    for extra in [vec!["--seed", "abc"], vec!["--ban-token", "166101"], vec!["--max-new-tokens", "0"],
        vec!["--context-tokens", "1"], vec!["--min-new-tokens", "129"], vec!["--stop", ""]] {
        let (c, args) = command("start", "generate", &extra);
        assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
    for flag in ["--temperature-milli", "--top-k", "--top-p-ppm"] {
        let mut words = argv("start", "generate"); words.extend([flag.to_owned(), "1".to_owned()]);
        assert!(parse(words).is_err());
    }
}

#[test]
fn native_lifetime_work_is_exact_and_never_multiplied_by_items_attempts_or_transport() {
    let (mut c, args) = command("start", "generate", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, common);
    let a = c.args.lifetime_limits(&args, common, job).unwrap();
    let work = job.max_work.int4_model_work();
    assert_eq!(a.native.native.max_forward_positions, work.forward_positions);
    assert_eq!(a.native.native.max_attention_pairs, work.attention_pairs);
    assert_eq!(a.native.native.max_projection_work, work.projections);
    assert_eq!(a.native.max_sampler_bytes, SAMPLER_BYTES);
    assert_eq!(a.native.max_kv_bytes, common.native.kv_bytes);
    let int8 = Int8Work::for_sequence(0, work.forward_positions as usize, work.projected_logits as usize).unwrap();
    assert!(work.projections.dot_products > int8.projections.dot_products);
    let mut larger = job; larger.max_items = 101; larger.max_attempts = 1000;
    c.args.storage.max_input_lines = 1_000_000; c.args.storage.max_stream_mib = 1024;
    let b = c.args.lifetime_limits(&args, common, larger).unwrap();
    assert_eq!(b.native.native.max_forward_positions, a.native.native.max_forward_positions);
    assert_eq!(b.native.native.max_attention_pairs, a.native.native.max_attention_pairs);
    assert_eq!(b.native.native.max_projection_work, a.native.native.max_projection_work);
    assert_eq!(b.native.max_sampler_bytes, a.native.max_sampler_bytes);
    assert!(b.buffer_bytes > a.buffer_bytes);
}

#[test]
fn every_native_axis_is_positive_and_text_jobs_refuse_grammar_work() {
    let (c, args) = command("start", "generate", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, common);
    for axis in 0..6 {
        let mut invalid = job;
        match axis {
            0 => invalid.max_work.model.forward_positions = 0,
            1 => invalid.max_work.model.projected_logits = 0,
            2 => invalid.max_work.model.attention_pairs = 0,
            3 => invalid.max_work.model.projections.dot_products = 0,
            4 => invalid.max_work.model.projections.multiply_accumulates = 0,
            _ => invalid.max_work.mask_node_visits = 1,
        }
        assert_eq!(c.args.lifetime_limits(&args, common, invalid).unwrap_err().code, "lifetime_text_work_limits");
    }
}

#[test]
fn stored_input_result_and_attempt_authority_cannot_outrun_admission() {
    let (c, args) = command("start", "generate", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, common);
    let mut invalid = job; invalid.max_input_bytes_per_item += 1;
    assert_eq!(c.args.lifetime_limits(&args, common, invalid).unwrap_err().code, "stored_input_ceiling");
    let mut invalid = job; invalid.max_result_bytes -= 1;
    assert_eq!(c.args.lifetime_limits(&args, common, invalid).unwrap_err().code, "stored_result_ceiling");
    let mut invalid = job; invalid.max_attempts = 0;
    assert_eq!(c.args.lifetime_limits(&args, common, invalid).unwrap_err().code, "job_limits");
}

#[test]
fn complete_population_and_both_preparation_lifetimes_are_priced_before_model_io() {
    let (c, args) = command("start", "chat", &[]);
    let common = c.args.validate(&args).unwrap();
    let job = lifetime(&args, common);
    let limits = c.args.lifetime_limits(&args, common, job).unwrap();
    assert_eq!(limits.preparation_reserve_bytes, common.preparation_bytes);
    assert!(limits.buffer_bytes > common.preparation_bytes + 2 * job.max_snapshot_bytes);
    assert_eq!(limits.common.minimum_memory_bytes, common.minimum_memory_bytes + limits.buffer_bytes);
    let mut too_small = common; too_small.memory_bytes = limits.common.minimum_memory_bytes - 1;
    assert_eq!(c.args.lifetime_limits(&args, too_small, job).unwrap_err().code, "job_population_memory");
    too_small.memory_bytes += 1;
    assert!(c.args.lifetime_limits(&args, too_small, job).is_ok());
}

#[test]
fn population_and_staging_arithmetic_fail_closed() {
    let (mut c, args) = command("start", "generate", &[]);
    let common = c.args.validate(&args).unwrap();
    let mut job = lifetime(&args, common); job.max_snapshot_bytes = i64::MAX as u64;
    assert_eq!(c.args.lifetime_limits(&args, common, job).unwrap_err().code, "job_memory_arithmetic");
    c.args.storage.journal_memory_mib = u64::MAX;
    assert_eq!(c.args.validate(&args).unwrap_err().code, "text_job_memory");
}

#[test]
fn root_dispatch_keeps_text_jobs_owned_and_reports_no_private_content() {
    let root = |retain: bool| {
        let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()];
        words.extend(argv("resume", "chat"));
        if !retain { words.retain(|word| word != "--store-results"); }
        let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong root dispatcher"); };
        assert_eq!(command.task, Int4Task::TextJob);
        command
    };
    assert_eq!(root(true).execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    let mut diagnostics = Vec::new();
    assert_eq!(root(false).run_owned(NoRead, NoWrite, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume");
    assert_eq!(report["task"], "chat");
    assert_eq!(report["code"], "explicit_text_job_storage");
    assert_eq!(report["durable_progress_may_exist"], true);
    let text = std::str::from_utf8(&diagnostics).unwrap();
    for private in ["private-", "0123456789abcdef", "history", "seed"] { assert!(!text.contains(private)); }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_owned_profile_refuses_before_private_input_or_storage() {
    for task in ["generate", "chat"] {
        let (c, args) = command("start", task, &[]);
        assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "owned_text_job_profile_unavailable");
    }
}
