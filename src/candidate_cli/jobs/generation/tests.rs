//! CLI, consent and pre-IO boundaries; no model or filesystem execution.
use super::*;
use crate::native_engine::strict_int8::Int8Work;
use crate::jobs::JobWork;

fn parse(operation: &str, task: &str, extra: &[&str]) -> Result<TextJobCommand, clap::Error> {
    let mut argv = vec!["candidate", "text-job", operation, "--task", task, "--model", "local.fnlpq",
        "--memory-mib", "8192", "--store-results", "--job-dir", "protected-job",
        "--job-id", "0123456789abcdef0123456789abcdef", "--key-file", "job.key", "--limits", "limits.json"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::TextJob(command) = CandidateCommand::from_matches(&matches)? else { panic!("text job route") };
    Ok(command)
}
pub(in crate::candidate_cli) fn args(task: &str, extra: &[&str]) -> TextJobArgs {
    parse("start", task, extra).unwrap().into_parts().1
}
pub(in crate::candidate_cli) fn lifetime() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192, max_snapshot_bytes: 65536,
        max_result_bytes: 1 << 20, max_spool_bytes: 4 << 20, max_materialized_bytes: 4 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: Int8Work::for_sequence(0, 8192, 100_000_000).unwrap(), mask_node_visits: 0 } }
}
#[test]
fn generation_and_chat_have_distinct_start_and_authenticated_resume_routes() {
    for task in ["generate", "chat"] {
        let (mode, a) = parse("start", task, &[]).unwrap().into_parts();
        assert!(matches!(mode, RunMode::Start)); a.validate().unwrap(); assert_eq!(a.task_name(), task);
        let (mode, a) = parse("resume", task, &["--discard-uncommitted-tail"]).unwrap().into_parts();
        assert!(matches!(mode, RunMode::Resume { discard_uncommitted: true })); a.validate().unwrap();
    }
    assert!(parse("start", "generate", &["--discard-uncommitted-tail"]).is_err());
    assert!(parse("start", "judge", &[]).is_err());
}
#[test]
fn all_generation_controls_survive_the_retained_parser_and_explicit_resume() {
    let seed = "07".repeat(32);
    let switches = ["--seed", seed.as_str(), "--temperature-milli", "850", "--top-k", "32", "--top-p-ppm", "900000",
        "--stop", " END\n", "--min-new-tokens", "2", "--ban-token", "7", "--logit-bias", "8=-500",
        "--repetition-penalty-milli", "1100", "--presence-penalty-milli", "-200", "--frequency-penalty-milli", "250", "--logprobs"];
    for task in ["generate", "chat"] {
        let a = args(task, &switches);
        let (_, b) = parse("resume", task, &switches).unwrap().into_parts();
        a.validate().unwrap(); b.validate().unwrap();
        let options = a.common.options(166_101).unwrap();
        assert!(options == b.common.options(166_101).unwrap());
        assert_eq!(options.stop_suffixes, vec![b" END\n".to_vec()]);
        assert_eq!(options.min_new_tokens, 2); assert!(options.capture_logprobs);
        assert_eq!(options.logit_bias_milli.get(&8), Some(&-500));
        assert!(matches!(options.sampling, GenerationSampling::Seeded { effective_seed, .. } if effective_seed == [7; 32]));
    }
}
#[test]
fn omitted_seed_is_greedy_never_a_new_random_seed_on_resume() {
    for operation in ["start", "resume"] {
        let (_, a) = parse(operation, "generate", &[]).unwrap().into_parts();
        assert!(matches!(a.common.options(166_101).unwrap().sampling, GenerationSampling::Greedy));
    }
    assert!(parse("start", "generate", &["--temperature-milli", "900"]).is_err());
    assert!(args("generate", &["--seed", "BAD"]).validate().is_err());
}
#[test]
fn explicit_retention_keys_paths_and_memory_are_mandatory() {
    for axis in 0..8 {
        let mut a = args("generate", &[]);
        match axis { 0 => a.store_results = false, 1 => a.job_dir = "-".into(), 2 => a.key_file = "".into(),
            3 => a.limits_file = "-".into(), 4 => a.max_stream_mib = 0, 5 => a.max_input_lines = 0,
            6 => a.journal_memory_mib = u64::MAX, _ => a.serialization_memory_mib = 0 }
        assert!(a.validate().is_err());
    }
}
#[test]
fn whole_original_envelopes_and_complete_result_storage_have_separate_caps() {
    let a = args("generate", &[]); let limits = a.validate().unwrap();
    a.check_lifetime(lifetime(), limits).unwrap();
    let mut j = lifetime(); j.max_input_bytes_per_item = a.common.max_input_bytes + 1;
    assert!(a.check_lifetime(j, limits).is_err());
    let mut j = lifetime(); j.max_result_bytes = limits.result_bytes - 1;
    assert!(a.check_lifetime(j, limits).is_err());
    let mut j = lifetime(); j.max_snapshot_bytes = a.max_stream_mib * MIB + 1;
    assert!(a.check_lifetime(j, limits).is_err());
}
#[test]
fn every_native_work_axis_is_required_before_input_or_model_io() {
    let a = args("generate", &[]); let limits = a.validate().unwrap();
    for axis in 0..5 {
        let mut j = lifetime();
        match axis { 0 => j.max_work.model.forward_positions = 0, 1 => j.max_work.model.projected_logits = 0,
            2 => j.max_work.model.attention_pairs = 0, 3 => j.max_work.model.projections.dot_products = 0,
            _ => j.max_work.model.projections.multiply_accumulates = 0 }
        assert!(a.check_lifetime(j, limits).is_err());
    }
}
#[test]
fn transport_changes_do_not_grant_more_generation_or_lifetime_authority() {
    let a = args("chat", &[]); let b = args("chat", &["--max-input-lines", "200000", "--max-stream-mib", "128"]);
    a.validate().unwrap(); b.validate().unwrap();
    assert!(a.common.options(166_101).unwrap() == b.common.options(166_101).unwrap());
    assert_eq!(a.validate().unwrap().result_bytes, b.validate().unwrap().result_bytes);
}
#[test]
fn unsupported_schedules_refuse_before_reading_keys_or_population() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for operation in ["start", "resume"] {
        let (mode, a) = parse(operation, "generate", &["--prefill-rows", "4"]).unwrap().into_parts();
        assert_eq!(execute_owned(mode, a, NoIo, &mut NoIo).unwrap_err().code, "text_job_serial_schedule");
    }
    for flag in ["--cohort-rows", "--active-rows", "--schema", "--defaults", "--key"] {
        assert!(parse("start", "generate", &[flag, "4"]).is_err());
    }
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_feature_graph_never_opens_key_config_input_or_output() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for task in ["generate", "chat"] {
        assert_eq!(execute_owned(RunMode::Start, args(task, &[]), NoIo, &mut NoIo).unwrap_err().code,
            "owned_text_job_profile_unavailable");
    }
}
