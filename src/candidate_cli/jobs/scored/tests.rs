//! Parsing/defaults/feature boundaries only, not native inference evidence.
use super::*;

fn parse(operation: &str, task: &str, extra: &[&str]) -> Result<ScoreJobCommand, clap::Error> {
    let mut argv = vec!["candidate", "score-job", operation, "--task", task,
        "--model", "model.fnlpq", "--memory-mib", "8192", "--store-results",
        "--job-dir", "protected-job", "--job-id", "0123456789abcdef0123456789abcdef",
        "--key-file", "protected.key", "--limits", "limits.json"];
    argv.extend_from_slice(extra);
    let m = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::ScoreJob(command) = CandidateCommand::from_matches(&m)?
        else { panic!("scored job route") };
    Ok(command)
}
fn command(operation: &str, task: &str, extra: &[&str]) -> ScoreJobCommand {
    parse(operation, task, extra).unwrap()
}
pub(in crate::candidate_cli) fn args(task: &str, extra: &[&str]) -> ScoreJobArgs {
    command("start", task, extra).into_parts().1
}
#[test]
fn both_scored_tasks_have_distinct_start_and_authenticated_resume_routes() {
    for task in ["classify", "sentiment"] {
        let (mode, args) = command("start", task, &[]).into_parts();
        assert!(matches!(mode, RunMode::Start)); args.validate().unwrap();
        let (mode, args) = command("resume", task, &[]).into_parts();
        assert!(matches!(mode, RunMode::Resume { discard_uncommitted: false })); args.validate().unwrap();
        let (mode, _) = command("resume", task, &["--discard-uncommitted-tail"]).into_parts();
        assert!(matches!(mode, RunMode::Resume { discard_uncommitted: true }));
    }
}
#[test]
fn scoring_jobs_cannot_accept_generation_schema_mask_or_start_repair_switches() {
    // Every negative fixture starts with ALL required options, so a missing
    // model, secret or directory cannot accidentally satisfy the assertion.
    assert!(parse("start", "classify", &[]).is_ok());
    for (flag, value) in [("--seed", "00"), ("--schema", "schema.json"), ("--max-new-tokens", "10"),
        ("--max-mask-node-visits", "10"), ("--instruction", "private")] {
        assert!(parse("start", "classify", &[flag, value]).is_err(), "{flag}");
    }
    assert!(parse("start", "classify", &["--discard-uncommitted-tail"]).is_err());
    for task in ["generate", "extract", "ner", "judge"] {
        assert!(parse("start", task, &[]).is_err());
    }
}
#[test]
fn privacy_consent_protected_paths_and_memory_budgets_are_mandatory() {
    let mut a = args("classify", &[]); a.store_results = false;
    assert!(a.validate().is_err());
    for axis in 0..8 {
        let mut a = args("classify", &[]);
        match axis { 0 => a.job_dir = "-".into(), 1 => a.key_file = "".into(),
            2 => a.limits_file = "-".into(), 3 => a.defaults = Some("-".into()),
            4 => a.max_stream_mib = 0, 5 => a.max_input_lines = 0,
            6 => a.journal_memory_mib = u64::MAX, _ => a.serialization_memory_mib = 0 }
        assert!(a.validate().is_err(), "{axis}");
    }
}
#[test]
fn more_transport_records_do_not_buy_more_native_work() {
    let a = args("sentiment", &[]);
    let b = args("sentiment", &["--max-input-lines", "200000", "--max-stream-mib", "128"]);
    a.validate().unwrap(); b.validate().unwrap();
    assert_eq!(a.host.work_ceiling(), b.host.work_ceiling());
}
#[test]
fn defaults_are_the_same_typed_private_settings_as_the_live_scoring_pipe() {
    let a = args("classify", &[]); let (_, l) = a.validate().unwrap(); let b = a.host.budget(l);
    let text = r#"{"labels":[{"id":"法律","description":"private category"},{"id":"other"}],"mode":"multi_label"}"#;
    let scored_batch::Defaults::Classify(Some(values)) = a.parse_defaults(Some(text), b).unwrap()
        else { panic!("classification settings") };
    assert_eq!(values.labels[0].id, "法律"); assert_eq!(values.labels[0].description, "private category");
    assert_eq!(values.mode, crate::tasks::classify::ClassificationMode::MultiLabel);
    assert_eq!(values.budget, b);
    for invalid in ["{}", r#"{"labels":[],"document":"private"}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"budget":{}}"#,
        r#"{"labels":[{"id":"a"},{"id":"b"}],"\u006cabels":[]}"#] {
        assert!(a.parse_defaults(Some(invalid), b).is_err());
    }
    let s = args("sentiment", &[]);
    let scored_batch::Defaults::Sentiment { args, policy } = s.parse_defaults(None, b).unwrap()
        else { panic!("sentiment defaults") };
    assert_eq!(args.axes, crate::tasks::sentiment::SentimentAxis::ALL.to_vec());
    assert_eq!(policy.minimum_peak_weight_ppm, 0);
    assert_eq!(policy.maximum_normalized_entropy_ppm, 1_000_000);
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_before_key_defaults_input_or_output_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flush") }
    }
    for task in ["classify", "sentiment"] {
        let a = args(task, &["--defaults", "never-open.json"]);
        let error = execute_owned(RunMode::Start, a, NoIo, &mut NoIo).unwrap_err();
        assert_eq!(error.code, "owned_score_job_profile_unavailable");
    }
}
