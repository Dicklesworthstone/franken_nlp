//! Command/configuration and no-IO regressions; no simulated inference results.
use super::*;
use crate::{jobs::JobWork, native_engine::{portable_int8::ProjectionWork, strict_int8::Int8Work},
    batch::extract::ExtractionBatchArgs};

pub(in crate::candidate_cli) fn command(operation: &str, task: &str, extra: &[&str]) -> JobCommand {
    let mut argv = vec!["candidate", "job", operation, "--task", task, "--model", "local.fnlpq",
        "--memory-mib", "8192", "--job-dir", "private/job", "--key-file", "private/key",
        "--limits", "limits.json", "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Job(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("job dispatched outside owned job path") };
    command
}
pub(in crate::candidate_cli) fn args(task: &str, extra: &[&str]) -> JobArgs { command("start", task, extra).into_parts().1 }
pub(in crate::candidate_cli) fn lifetime() -> JobLimits {
    JobLimits { max_items: 100, max_id_bytes: 128, max_input_bytes_per_item: 1 << 20,
        max_snapshot_bytes: 64 << 20, max_result_bytes: 1 << 20, max_spool_bytes: 128 << 20,
        max_materialized_bytes: 128 << 20, max_journal_bytes: 64 << 20, max_attempts: 200,
        max_work: JobWork { model: Int8Work { forward_positions: 500_000, projected_logits: 4_000_000_000,
            attention_pairs: 1_000_000_000_000, projections: ProjectionWork { dot_products: 1_000_000_000_000,
                multiply_accumulates: 100_000_000_000_000_000 } }, mask_node_visits: 200_000_000_000 } }
}
#[test]
fn all_five_native_job_tasks_have_distinct_start_and_resume_routes() {
    for task in ["ner", "keyphrases", "summarize", "answer", "extract"] {
        for mode in ["start", "resume"] {
            let c = command(mode, task, &[]);
            assert_eq!(c.mode().name(), mode);
            let (_, args) = c.into_parts(); args.validate().unwrap();
        }
    }
    assert!(definition().try_get_matches_from(["job"]).is_err());
    assert!(definition().try_get_matches_from(["job", "start"]).is_err());
    for task in ["classify", "generate", "sentiment"] {
        assert!(definition().try_get_matches_from(["job", "start", "--task", task]).is_err());
    }
}
#[test]
fn retention_consent_and_original_identity_are_mandatory_before_any_io() {
    let mut a = args("ner", &[]); a.store_results = false;
    let c = JobCommand { operation: Operation::Start(a) };
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read without storage consent") } }
    let mut stdout = Vec::new();
    let error = c.execute_owned(NoRead, &mut stdout).err().unwrap();
    assert_eq!(error.code, "explicit_storage_or_job_options"); assert!(stdout.is_empty());
    for value in ["ab".repeat(15), "ab".repeat(17), "AB".repeat(16), "é".repeat(16), "x".repeat(32)] {
        assert!(parse_job_id(&value).is_err());
    }
    let id = "0123456789abcdef0123456789abcdef";
    assert_eq!(job_id_hex(parse_job_id(id).unwrap()), id);
}
#[test]
fn repair_is_an_explicit_resume_only_flag_and_not_an_implicit_default() {
    let c = command("resume", "ner", &[]);
    assert!(matches!(c.mode(), RunMode::Resume { discard_uncommitted: false }));
    let c = command("resume", "ner", &["--discard-uncommitted-tail"]);
    assert!(matches!(c.mode(), RunMode::Resume { discard_uncommitted: true }));
    assert!(definition().try_get_matches_from(["job", "start", "--discard-uncommitted-tail"]).is_err());
}
fn input_options(extra: &[&str]) -> Result<InputRetentionArgs, clap::Error> {
    let definition = InputRetentionArgs::augment_args(clap::Command::new("input-selection")
        .arg(clap::Arg::new("input").default_value("-")));
    let mut argv = vec!["input-selection"];
    argv.extend_from_slice(extra);
    InputRetentionArgs::from_arg_matches(&definition.try_get_matches_from(argv)?)
}
#[test]
fn input_retention_requires_separate_consent_and_an_explicit_positive_disk_cap() {
    let default = args("ner", &[]);
    assert!(default.store_results);
    assert!(!default.input_retention.requested());
    assert!(default.input_retention.limits().unwrap().is_none());
    for invalid in [
        vec!["--spool-input"], vec!["--use-retained-input"],
        vec!["--spool-input", "--max-retained-input-mib", "0"],
        vec!["--use-retained-input", "--max-retained-input-mib", "0"],
        vec!["--spool-input", "--use-retained-input", "--max-retained-input-mib", "1"],
    ] { assert!(input_options(&invalid).is_err(), "{invalid:?}"); }
    for invalid in [
        vec!["--max-retained-input-mib", "1"],
        vec!["--spool-input", "--max-retained-input-mib", "8796093022208"],
        vec!["--spool-input", "--max-retained-input-mib", "18446744073709551615"],
    ] { assert!(input_options(&invalid).unwrap().limits().is_err(), "{invalid:?}"); }
    let retained = input_options(&["--spool-input", "--max-retained-input-mib", "7"]).unwrap();
    assert_eq!(retained.limits().unwrap(), Some(InputRetentionLimits { max_spool_bytes: 7 * MIB }));
}
#[test]
fn input_memory_mode_requires_valid_consent_and_removes_only_complete_population_copies() {
    use crate::jobs::JobInputMemory;
    let external = InputRetentionArgs::default();
    let spool = input_options(&["--spool-input", "--max-retained-input-mib", "7"]).unwrap();
    let retained = input_options(&["--use-retained-input", "--max-retained-input-mib", "7"]).unwrap();
    assert_eq!(external.input_memory().unwrap(), JobInputMemory::CompletePopulation);
    assert_eq!(spool.input_memory().unwrap(), JobInputMemory::RetainedRecord);
    assert_eq!(retained.input_memory().unwrap(), JobInputMemory::RetainedRecord);
    let job = lifetime();
    let full = external.buffer_bytes(job).unwrap();
    assert_eq!(spool.buffer_bytes(job).unwrap(), full - 2 * job.max_snapshot_bytes);
    assert_eq!(retained.buffer_bytes(job).unwrap(), full - 2 * job.max_snapshot_bytes);
    let mut larger = job; larger.max_snapshot_bytes *= 2;
    assert_eq!(external.buffer_bytes(larger).unwrap(), full + 2 * job.max_snapshot_bytes);
    for mode in [spool, retained] {
        assert_eq!(mode.buffer_bytes(larger).unwrap(), mode.buffer_bytes(job).unwrap());
        let mut more_items = job; more_items.max_items += 1;
        assert_eq!(mode.buffer_bytes(more_items).unwrap(), mode.buffer_bytes(job).unwrap() + 1024);
        let mut larger_record = job; larger_record.max_input_bytes_per_item += 1;
        assert!(mode.buffer_bytes(larger_record).unwrap() > mode.buffer_bytes(job).unwrap());
    }
    for invalid in [
        InputRetentionArgs { spool_input: true, ..InputRetentionArgs::default() },
        InputRetentionArgs { use_retained_input: true, ..InputRetentionArgs::default() },
        InputRetentionArgs { spool_input: true, ..retained },
        InputRetentionArgs { max_retained_input_mib: Some(7), ..InputRetentionArgs::default() },
        InputRetentionArgs { max_retained_input_mib: Some(0), ..spool },
        InputRetentionArgs { max_retained_input_mib: Some(u64::MAX), ..retained },
    ] {
        assert_eq!(invalid.input_memory().unwrap_err().code, "job_input_retention_options");
        assert_eq!(invalid.buffer_bytes(job).unwrap_err().code, "job_input_retention_options");
    }
}
#[test]
fn retained_replay_rejects_explicit_population_arguments_including_dash() {
    for population in ["original.ndjson", "-"] {
        assert!(input_options(&["--use-retained-input", "--max-retained-input-mib", "1", population]).is_err());
    }
    let retained = input_options(&["--use-retained-input", "--max-retained-input-mib", "1"]).unwrap();
    assert!(retained.validate(RunMode::Resume { discard_uncommitted: false }, std::path::Path::new("-")).is_ok());
    assert!(retained.validate(RunMode::Resume { discard_uncommitted: false }, std::path::Path::new("original.ndjson")).is_err());
}
#[test]
fn input_retention_mode_mismatches_fail_before_configuration_or_input_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected population read") } }
    for (mode, flag) in [("start", "--use-retained-input"), ("resume", "--spool-input")] {
        let c = command(mode, "ner", &[flag, "--max-retained-input-mib", "1"]);
        let mut stdout = Vec::new();
        let error = c.execute_owned(NoRead, &mut stdout).unwrap_err();
        assert_eq!(error.code, "job_input_retention_options");
        assert!(stdout.is_empty());
    }
}
#[test]
fn retained_population_reader_never_touches_stdin_or_falls_back_to_an_external_file() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("retained resume read inherited input") } }
    let retained = input_options(&["--use-retained-input", "--max-retained-input-mib", "1"]).unwrap();
    let mode = RunMode::Resume { discard_uncommitted: false };
    let mut reader = population_reader(mode, retained, std::path::Path::new("-"), NoRead, 8192).unwrap();
    let mut bytes = Vec::new();
    assert_eq!(reader.read_to_end(&mut bytes).unwrap(), 0);
    assert!(bytes.is_empty());
    let error = population_reader(mode, retained, std::path::Path::new("/nonexistent/private-population"), NoRead, 8192)
        .err().unwrap();
    assert_eq!(error.code, "job_input_retention_options");
    let exact = b"{\"id\":\"x\",\"text\":\"private\"}\r\n";
    let spool = input_options(&["--spool-input", "--max-retained-input-mib", "1"]).unwrap();
    for mode in [InputRetentionArgs::default(), spool] {
        // Retained creation still consumes the selected one-shot stream exactly;
        // only authenticated retained replay substitutes the empty reader.
        let mut external = population_reader(RunMode::Start, mode, std::path::Path::new("-"),
            io::Cursor::new(exact), 8192).unwrap();
        bytes.clear(); external.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, exact);
    }
}
#[test]
fn schema_options_never_change_the_fixed_source_task() {
    assert!(args("ner", &["--schema", "s.json"]).validate().is_err());
    assert!(args("extract", &["--schema", "-"]).validate().is_err());
    assert!(args("extract", &["--schema", "s.json", "--source-membership"]).validate().is_ok());
    for bad in [["job", "start", "--source-membership"], ["job", "start", "--seed"], ["job", "start", "--key"]] {
        assert!(definition().try_get_matches_from(bad).is_err());
    }
}
#[test]
fn lifetime_work_is_never_multiplied_by_items_attempts_or_transport_lines() {
    let a = args("ner", &[]); let limit = lifetime();
    let n = native_limits(&a, limit).unwrap();
    assert_eq!(n.max_model_work, limit.max_work.model);
    assert_eq!(n.masks.max_visits_per_run, limit.max_work.mask_node_visits);
    assert_eq!(n.masks.max_visits_per_item, a.host.max_mask_node_visits);
    let mut changed = limit; changed.max_items *= 2; changed.max_attempts *= 2;
    assert_eq!(native_limits(&a, changed).unwrap().max_model_work, n.max_model_work);
    for axis in 0..6 {
        let mut l = limit;
        match axis { 0 => l.max_work.model.forward_positions = 0, 1 => l.max_work.model.projected_logits = 0,
            2 => l.max_work.model.attention_pairs = 0, 3 => l.max_work.model.projections.dot_products = 0,
            4 => l.max_work.model.projections.multiply_accumulates = 0, _ => l.max_work.mask_node_visits = 0 }
        assert!(native_limits(&a, l).is_err());
    }
    let mut smaller = limit; smaller.max_result_bytes -= 1;
    assert!(native_limits(&a, smaller).is_err());
}
#[test]
fn original_limit_file_rejects_duplicate_unknown_and_partial_fields() {
    let l = lifetime(); let json = serde_json::to_string(&l).unwrap();
    assert_eq!(parse_limits(&json).unwrap(), l);
    let duplicated = json.replacen('{', "{\"max_items\":1,", 1);
    assert!(parse_limits(&duplicated).is_err());
    let mut value = serde_json::to_value(l).unwrap(); value["model"] = serde_json::json!("injected");
    assert!(parse_limits(&value.to_string()).is_err());
    assert!(parse_limits("{}").is_err());
    assert!(parse_limits(&" ".repeat(LIMIT_BYTES + 1)).is_err());
}
#[test]
fn defaults_cannot_replace_the_task_or_any_ceiling_and_qa_has_no_invented_evidence() {
    let mut a = args("ner", &[]); let (_, limits) = a.validate().unwrap(); let ceiling = a.host.task_budget(limits);
    let Defaults::Source(Some(defaults)) = load_defaults(&a, ceiling, None, None).unwrap() else { panic!("missing NER defaults") };
    let before = serde_json::to_value(defaults).unwrap(); a.defaults = Some("defaults.json".into());
    for field in ["max_input_tokens", "max_output_tokens", "max_output_bytes", "max_grammar_states", "max_kv_bytes"] {
        let mut changed = before.clone(); changed["budget"][field] = serde_json::json!(u64::MAX);
        assert!(load_defaults(&a, ceiling, Some(&changed.to_string()), None).is_err(), "{field}");
    }
    let mut changed = before; changed["task"] = serde_json::json!("answer");
    assert!(load_defaults(&a, ceiling, Some(&changed.to_string()), None).is_err());
    let qa = args("answer", &[]);
    assert!(matches!(load_defaults(&qa, ceiling, None, None).unwrap(), Defaults::Source(None)));
}
#[test]
fn exact_schema_decimals_and_source_bound_defaults_survive_without_fake_documents() {
    let a = args("extract", &["--schema", "schema.json"]); let (_, limits) = a.validate().unwrap();
    let ceiling = a.host.task_budget(limits);
    let schema = " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678} \n";
    let Defaults::Extract(Some(d)) = load_defaults(&a, ceiling, None, Some(schema.to_owned())).unwrap() else { panic!("schema absent") };
    assert_eq!(d.schema, schema);
    let a = args("extract", &["--schema", "schema.json", "--source-membership"]);
    let source = r#"{"type":"string","x-fnlp-source":"verbatim","maxLength":64}"#;
    assert!(load_defaults(&a, ceiling, None, Some(source.to_owned())).is_ok());
    let mut a = args("extract", &[]); a.defaults = Some("d.json".into());
    let defaults = ExtractionBatchArgs { schema: schema.to_owned(), grounding: crate::batch::extract::ExtractionBatchGrounding::Structural,
        budget: ceiling };
    assert!(load_defaults(&a, ceiling, Some(&serde_json::to_string(&defaults).unwrap()), None).is_ok());
    let mut invalid = defaults; invalid.schema = r#"{"type":"string","type":"integer"}"#.to_owned();
    assert!(load_defaults(&a, ceiling, Some(&serde_json::to_string(&invalid).unwrap()), None).is_err());
}
#[test]
fn configuration_and_path_caps_fail_closed() {
    for (flag, value) in [("--max-stream-mib", "0"), ("--max-input-lines", "0"),
        ("--journal-memory-mib", "18446744073709551615"), ("--serialization-memory-mib", "0")] {
        assert!(args("ner", &[flag, value]).validate().is_err());
    }
    let mut a = args("ner", &[]); a.key_file = "-".into(); assert!(a.validate().is_err());
    let mut a = args("ner", &[]); a.defaults = Some("d.json".into());
    let (_, l) = a.validate().unwrap(); let ceiling = a.host.task_budget(l);
    for invalid in [r#"{"task":"ner","\u0074ask":"ner"}"#, "{}"] {
        assert!(load_defaults(&a, ceiling, Some(invalid), None).is_err());
    }
    assert!(load_defaults(&a, ceiling, Some(&" ".repeat(DEFAULT_BYTES + 1)), None).is_err());
}
#[test]
fn failure_reports_disclose_no_private_paths_or_false_rollback_promise() {
    let mut a = args("ner", &[]); a.store_results = false;
    a.key_file = "PRIVATE_SECRET_PATH".into(); a.job_dir = "PRIVATE_JOB_PATH".into();
    let mut output = Vec::new(); let mut errors = Vec::new();
    let status = JobCommand { operation: Operation::Start(a) }.run_owned(io::empty(), &mut output, &mut errors);
    assert_ne!(status, ExitCode::SUCCESS); assert!(output.is_empty());
    let text = String::from_utf8(errors).unwrap();
    assert!(!text.contains("PRIVATE_")); assert!(!text.contains("no result published"));
    let json = canonjson::parse_str(&text).unwrap();
    assert_eq!(json["durable_progress_may_exist"], true);
    assert_eq!(json["operation"], "start");
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unsupported_build_never_reads_private_input_or_writes_progress() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unsupported job read") } }
    for task in ["ner", "keyphrases", "summarize", "answer", "extract"] {
        let mut out = Vec::new();
        let error = command("resume", task, &[]).execute_owned(NoRead, &mut out).err().unwrap();
        assert_eq!(error.code, "owned_job_profile_unavailable"); assert!(out.is_empty());
    }
}
