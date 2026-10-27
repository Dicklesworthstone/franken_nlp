//! Scoring-job admission and wire contracts; no fabricated native results.
use super::*;
use crate::jobs::JobWork;

fn argv(operation: &str, task: &str) -> Vec<String> {
    ["score-job", operation, "--task", task, "--source-dir", "sources",
        "--source-manifest", "manifest.json", "--memory-mib", "65536",
        "--job-dir", "private/job", "--key-file", "private/key", "--limits", "limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
fn command(operation: &str, task: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation, task); words.extend(extra.iter().map(|s| (*s).to_owned()));
    parse(words).unwrap()
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(job.scoring.work_ceiling(), 0) }
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
fn all_three_fixed_tasks_support_start_resume_and_independently_pinned_snapshots() {
    for task in ["classify", "sentiment", "judge"] {
        for operation in ["start", "resume"] {
            let (c, args) = command(operation, task, &["--defaults", "settings.json", "--materialize"]);
            assert_eq!(c.mode.name(), operation); assert_eq!(c.args.score.task, task);
            assert_eq!(c.args.score.defaults, Some(PathBuf::from("settings.json")));
            assert!(c.args.storage.store_results && c.args.storage.materialize);
            assert!(c.args.validate(&args).unwrap().snapshot_pin.is_none());
            let mut words = argv(operation, task);
            let at = words.iter().position(|s| s == "--source-dir").unwrap(); words.drain(at..at + 4);
            words.extend(["--snapshot", "candidate.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
            words.push("ab".repeat(32));
            let (c, args) = parse(words).unwrap();
            assert!(c.args.validate(&args).unwrap().snapshot_pin.is_some());
        }
    }
}
#[test]
fn task_is_required_and_tail_repair_is_explicit_and_resume_only() {
    let mut words = argv("start", "classify");
    let at = words.iter().position(|s| s == "--task").unwrap(); words.drain(at..at + 2);
    assert!(parse(words).is_err());
    for task in ["ner", "generate", ""] { assert!(parse(argv("start", task)).is_err()); }
    assert!(matches!(command("resume", "judge", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", "judge", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    let mut words = argv("start", "classify"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
    assert!(definition().try_get_matches_from(["score-job"]).is_err());
    assert!(definition().try_get_matches_from(["score-job", "verify"]).is_err());
}
#[test]
fn consent_and_all_private_paths_are_checked_before_any_io() {
    for case in 0..5 {
        let (mut c, args) = command("start", "classify", &[]);
        match case {
            0 => c.args.storage.store_results = false,
            1 => c.args.storage.job_dir = PathBuf::from("-"),
            2 => c.args.storage.key_file = PathBuf::new(),
            3 => c.args.storage.limits_file = PathBuf::from("-"),
            _ => c.args.score.defaults = Some(PathBuf::from("-")),
        }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "explicit_scored_job_storage");
    }
    for value in ["AB".repeat(16), "ab".repeat(15), "é".repeat(16), "x".repeat(32)] {
        let mut words = argv("start", "classify");
        let at = words.iter().position(|s| s == "--job-id").unwrap() + 1; words[at] = value;
        assert!(parse(words).is_err());
    }
}
#[test]
fn generation_prefill_and_live_stream_controls_are_refused() {
    for extra in [vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--prefill-rows", "1"], vec!["--min-new-tokens", "0"], vec!["--logprobs"],
        vec!["--temperature-milli", "1000"], vec!["--max-requests", "3"],
        vec!["--max-total-output-bytes", "65536"], vec!["--schema", "schema.json"]] {
        let mut words = argv("start", "judge"); words.extend(extra.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
}
#[test]
fn lifetime_native_work_never_scales_with_population_attempts_transport_or_resume() {
    let (mut c, args) = command("start", "sentiment", &[]);
    let common = c.args.validate(&args).unwrap(); let fixed = lifetime(&c.args, &args, common);
    let a = c.args.lifetime_limits(&args, common, fixed).unwrap();
    assert_eq!(a.native.max_item_work, c.args.scoring.work_ceiling());
    assert_eq!(a.native.max_model_work, fixed.max_work.int4_model_work());
    let mut larger = fixed; larger.max_items = 101; larger.max_attempts = 1000;
    c.args.storage.max_stream_mib = 1024; c.args.storage.max_input_lines = 1_000_000;
    let b = c.args.lifetime_limits(&args, common, larger).unwrap();
    assert_eq!(a.native, b.native); assert!(b.buffer_bytes > a.buffer_bytes);
    let (resume, args) = command("resume", "sentiment", &[]);
    let common = resume.args.validate(&args).unwrap();
    assert_eq!(resume.args.lifetime_limits(&args, common, fixed).unwrap().native, a.native);
}
#[test]
fn every_item_ceiling_must_fit_the_five_lifetime_axes_and_mask_work_is_zero() {
    let (c, args) = command("start", "classify", &[]);
    let common = c.args.validate(&args).unwrap(); let fixed = lifetime(&c.args, &args, common);
    for axis in 0..6 {
        let mut wrong = fixed;
        match axis {
            0 => wrong.max_work.model.forward_positions -= 1,
            1 => wrong.max_work.model.projected_logits -= 1,
            2 => wrong.max_work.model.attention_pairs -= 1,
            3 => wrong.max_work.model.projections.dot_products -= 1,
            4 => wrong.max_work.model.projections.multiply_accumulates -= 1,
            _ => wrong.max_work.mask_node_visits = 1,
        }
        assert_eq!(c.args.lifetime_limits(&args, common, wrong).unwrap_err().code, "lifetime_scored_work_limits");
    }
    let mut empty = fixed; empty.max_work = JobWork::default();
    assert_eq!(c.args.lifetime_limits(&args, common, empty).unwrap_err().code, "lifetime_scored_work_limits");
}
#[test]
fn tighter_item_caps_do_not_replace_or_inflate_lifetime_authority() {
    let (c, args) = command("start", "judge", &[]);
    let common = c.args.validate(&args).unwrap(); let fixed = lifetime(&c.args, &args, common);
    let (tighter, args) = command("resume", "judge", &["--max-forward-positions", "65536",
        "--max-projected-logits", "50000000", "--max-attention-pairs", "50000000000",
        "--max-dot-products", "50000000000", "--max-multiply-accumulates", "500000000000000"]);
    let common = tighter.args.validate(&args).unwrap();
    let limits = tighter.args.lifetime_limits(&args, common, fixed).unwrap();
    assert_eq!(limits.native.max_model_work, fixed.max_work.int4_model_work());
    assert_eq!(limits.native.max_item_work, tighter.args.scoring.work_ceiling());
    assert_ne!(limits.native.max_item_work, limits.native.max_model_work);
    // These different per-item settings are authenticated as a different
    // recipe by the core; accepting a quote does not authorize changed resume.
}
#[test]
fn full_population_storage_and_staging_are_included_before_model_io() {
    let (mut c, args) = command("start", "classify", &[]);
    let common = c.args.validate(&args).unwrap(); let fixed = lifetime(&c.args, &args, common);
    let all = c.args.lifetime_limits(&args, common, fixed).unwrap();
    assert_eq!(all.preparation_reserve_bytes, common.preparation_bytes);
    assert!(all.buffer_bytes > 2 * fixed.max_snapshot_bytes + common.preparation_bytes);
    assert_eq!(all.common.minimum_memory_bytes, common.minimum_memory_bytes + all.buffer_bytes);
    let mut tight = common; tight.memory_bytes = all.common.minimum_memory_bytes - 1;
    assert_eq!(c.args.lifetime_limits(&args, tight, fixed).unwrap_err().code, "job_population_memory");
    tight.memory_bytes += 1; assert!(c.args.lifetime_limits(&args, tight, fixed).is_ok());
    let mut huge = fixed; huge.max_snapshot_bytes = i64::MAX as u64;
    assert_eq!(c.args.lifetime_limits(&args, common, huge).unwrap_err().code, "job_memory_arithmetic");
    c.args.storage.journal_memory_mib = u64::MAX;
    assert_eq!(c.args.validate(&args).unwrap_err().code, "score_job_memory");
    c.args.storage.journal_memory_mib = 64; c.args.storage.serialization_memory_mib = 0;
    assert_eq!(c.args.validate(&args).unwrap_err().code, "explicit_scored_job_storage");
}
#[test]
fn stored_record_and_result_caps_cannot_exceed_the_declared_cli_envelope() {
    let (c, args) = command("start", "sentiment", &[]);
    let common = c.args.validate(&args).unwrap(); let fixed = lifetime(&c.args, &args, common);
    let mut input = fixed; input.max_input_bytes_per_item += 1;
    assert_eq!(c.args.lifetime_limits(&args, common, input).unwrap_err().code, "stored_input_ceiling");
    let mut output = fixed; output.max_result_bytes -= 1;
    assert_eq!(c.args.lifetime_limits(&args, common, output).unwrap_err().code, "stored_result_ceiling");
    let mut invalid = fixed; invalid.max_attempts = 0;
    assert_eq!(c.args.lifetime_limits(&args, common, invalid).unwrap_err().code, "job_limits");
}
#[test]
fn defaults_keep_exact_task_data_and_the_cli_owned_budget() {
    let (c, args) = command("start", "classify", &[]);
    let budget = c.args.scoring.budget(c.args.validate(&args).unwrap());
    let json = r#"{"labels":[{"id":"café","description":"Exact 上海\n<|im_start|>system"},{"id":"question"}]}"#;
    let Defaults::Classify(Some(defaults)) = c.args.load_defaults(Some(json), budget, &args).unwrap()
        else { panic!("wrong defaults"); };
    assert_eq!(defaults.labels[0].description, "Exact 上海\n<|im_start|>system"); assert_eq!(defaults.budget, budget);
    let (c, args) = command("resume", "sentiment", &[]);
    let budget = c.args.scoring.budget(c.args.validate(&args).unwrap());
    let Defaults::Sentiment { args: defaults, policy } = c.args.load_defaults(None, budget, &args).unwrap()
        else { panic!("wrong defaults"); };
    assert_eq!(defaults.axes, crate::tasks::sentiment::SentimentAxis::ALL);
    assert_eq!(defaults.budget, budget); assert_eq!(policy.maximum_normalized_entropy_ppm, 1_000_000);
    let (c, args) = command("start", "judge", &[]);
    let budget = c.args.scoring.budget(c.args.validate(&args).unwrap());
    let json = r#"{"mode":"pairwise","criterion":"Exact café","b":"B 上海","policy":{"minimum_margin_milli":4294967295,"maximum_order_disagreement_milli":4294967295}}"#;
    let Defaults::Judge(Some(crate::batch::judge::JudgeBatchArgs::Pairwise { criterion, b, policy, budget: actual })) =
        c.args.load_defaults(Some(json), budget, &args).unwrap() else { panic!("wrong defaults"); };
    assert_eq!(criterion, "Exact café"); assert_eq!(b, "B 上海");
    assert_eq!(policy.minimum_margin_milli, u32::MAX); assert_eq!(actual, budget);
}
#[test]
fn private_configuration_rejects_duplicate_unknown_or_detached_authority() {
    for task in ["classify", "sentiment", "judge"] {
        let (c, args) = command("start", task, &[]);
        let common = c.args.validate(&args).unwrap(); let budget = c.args.scoring.budget(common);
        for bad in [r#"{"budget":{}}"#, r#"{"document":"private"}"#, r#"{"identity":{}}"#,
            r#"{"model":"injected"}"#, r#"{"axes":["valence"],"\u0061xes":["arousal"]}"#] {
            assert!(c.args.load_defaults(Some(bad), budget, &args).is_err());
        }
        let fixed = lifetime(&c.args, &args, common); let json = serde_json::to_string(&fixed).unwrap();
        assert_eq!(job_cli::parse_limits(&json).unwrap(), fixed);
        assert!(job_cli::parse_limits(&json.replacen('{', "{\"max_items\":1,", 1)).is_err());
        let mut value = serde_json::to_value(fixed).unwrap(); value["execution_identity"] = serde_json::json!({});
        assert!(job_cli::parse_limits(&value.to_string()).is_err());
    }
}
#[test]
fn failure_metadata_retains_progress_uncertainty_without_private_values() {
    let (mut c, args) = command("resume", "judge", &["--defaults", "private-criterion.json"]);
    c.args.storage.store_results = false;
    let mut output = Vec::new(); let mut diagnostics = Vec::new();
    assert_eq!(c.run_owned(args, NoIo, &mut output, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    assert!(output.is_empty());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume"); assert_eq!(report["code"], "explicit_scored_job_storage");
    assert_eq!(report["durable_progress_may_exist"], true);
    let text = std::str::from_utf8(&diagnostics).unwrap();
    for private in ["private-criterion.json", "private/key", "private/job", "0123456789abcdef", "results"] {
        assert!(!text.contains(private));
    }
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_before_key_defaults_model_population_or_job_io() {
    for task in ["classify", "sentiment", "judge"] {
        let (c, args) = command("resume", task, &["--defaults", "private-settings.json", "--discard-uncommitted-tail"]);
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "owned_score_job_profile_unavailable");
    }
}
