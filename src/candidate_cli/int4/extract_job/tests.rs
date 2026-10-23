//! Pure parser, lifetime-authority and pre-IO admission regressions.
//! These checks do not fabricate native model results or durable receipts.
use super::*;
use crate::{
    batch::extract::ExtractionBatchGrounding,
    jobs::JobWork,
    native_engine::{constrained_int4, strict_int8::Int8Work},
};

fn argv(operation: &str) -> Vec<String> {
    ["extract-job", operation, "--source-dir", "sources", "--source-manifest", "manifest.json",
        "--memory-mib", "65536", "--schema", "schema.json", "--job-dir", "private/job",
        "--key-file", "private/key", "--limits", "limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect()
}
fn parse(argv: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(argv)?)
}
fn command(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation);
    words.extend(extra.iter().map(|s| (*s).to_owned()));
    parse(words).unwrap()
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    let work = constrained_int4::planned_work(args.group_width, common.max_prompt_tokens,
        args.max_new_tokens).unwrap();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(work, job.extraction.mask_visits() * 3) }
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
fn start_and_resume_keep_explicit_model_schema_and_storage_identity() {
    for operation in ["start", "resume"] {
        let (c, args) = command(operation, &["--source-membership", "--materialize"]);
        assert_eq!(c.mode.name(), operation);
        assert_eq!(args.input, PathBuf::from("-"));
        assert_eq!(c.args.extraction.schema_path(), std::path::Path::new("schema.json"));
        assert!(c.args.extraction.source_membership());
        assert!(c.args.storage.store_results && c.args.storage.materialize);
        assert_eq!(job_cli::job_id_hex(c.args.storage.job_id), "0123456789abcdef0123456789abcdef");
        assert!(c.args.validate(&args).is_ok());
    }
    let mut words = argv("resume");
    let source = words.iter().position(|s| s == "--source-dir").unwrap();
    words.drain(source..source + 4);
    words.extend(["--snapshot", "model.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
    words.push("a".repeat(64));
    let (c, args) = parse(words).unwrap();
    assert!(c.args.validate(&args).unwrap().snapshot_pin.is_some());
}

#[test]
fn repair_is_resume_only_and_requires_an_explicit_flag() {
    assert!(matches!(command("resume", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    let mut words = argv("start"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
    assert!(definition().try_get_matches_from(["extract-job"]).is_err());
    assert!(definition().try_get_matches_from(["extract-job", "verify"]).is_err());
}

#[test]
fn missing_storage_consent_and_nonlocal_storage_paths_refuse_before_io() {
    for case in 0..4 {
        let (mut c, args) = command("start", &[]);
        match case {
            0 => c.args.storage.store_results = false,
            1 => c.args.storage.job_dir = PathBuf::from("-"),
            2 => c.args.storage.key_file = PathBuf::new(),
            _ => c.args.storage.limits_file = PathBuf::from("-"),
        }
        let error = c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err();
        assert_eq!(error.code, "explicit_storage_or_job_options");
    }
    for value in ["AB".repeat(16), "ab".repeat(15), "ab".repeat(17), "é".repeat(16), "x".repeat(32)] {
        let mut words = argv("start");
        let at = words.iter().position(|s| s == "--job-id").unwrap() + 1;
        words[at] = value;
        assert!(parse(words).is_err());
    }
}

#[test]
fn shared_schema_is_mandatory_and_no_defaults_or_generation_policy_is_accepted() {
    let mut missing = argv("start");
    let at = missing.iter().position(|s| s == "--schema").unwrap();
    missing.drain(at..at + 2);
    assert!(parse(missing).is_err());
    for extra in [
        vec!["--defaults", "defaults.json"], vec!["--task", "extract"],
        vec!["--prefill-rows", "1"], vec!["--max-output-bytes", "65536"],
        vec!["--min-new-tokens", "0"], vec!["--logprobs"],
    ] {
        let mut words = argv("start"); words.extend(extra.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
    let (mut c, args) = command("start", &[]);
    c.args.storage.max_stream_mib = 0;
    assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
        "explicit_storage_or_job_options");
}

#[test]
fn exact_group_work_and_masks_never_scale_with_population_attempts_or_transport() {
    let (mut c, args) = command("start", &[]);
    let common = c.args.validate(&args).unwrap();
    let limits = lifetime(&c.args, &args, common);
    let a = c.args.lifetime_limits(&args, common, limits).unwrap();
    assert_eq!(a.native.max_model_work, limits.max_work.int4_model_work());
    assert_eq!(a.native.masks.max_visits_per_run, limits.max_work.mask_node_visits);
    assert_eq!(a.native.masks.max_visits_per_item, c.args.extraction.mask_visits());
    let ordinary = Int8Work::for_sequence(0, a.native.max_model_work.forward_positions as usize,
        a.native.max_model_work.projected_logits as usize).unwrap();
    assert!(a.native.max_model_work.projections.dot_products > ordinary.projections.dot_products);
    let mut changed = limits; changed.max_items = 101; changed.max_attempts = 1000;
    c.args.storage.max_input_lines = 1_000_000;
    c.args.storage.max_stream_mib = 1024;
    let b = c.args.lifetime_limits(&args, common, changed).unwrap();
    assert_eq!(b.native.max_model_work, a.native.max_model_work);
    assert_eq!(b.native.masks.max_visits_per_run, a.native.masks.max_visits_per_run);
    assert!(b.buffer_bytes > a.buffer_bytes);
}

#[test]
fn zero_native_axes_and_insufficient_lifetime_masks_refuse() {
    let (c, args) = command("start", &[]);
    let common = c.args.validate(&args).unwrap();
    let limits = lifetime(&c.args, &args, common);
    for axis in 0..6 {
        let mut bad = limits;
        match axis {
            0 => bad.max_work.model.forward_positions = 0,
            1 => bad.max_work.model.projected_logits = 0,
            2 => bad.max_work.model.attention_pairs = 0,
            3 => bad.max_work.model.projections.dot_products = 0,
            4 => bad.max_work.model.projections.multiply_accumulates = 0,
            _ => bad.max_work.mask_node_visits = c.args.extraction.mask_visits() - 1,
        }
        assert_eq!(c.args.lifetime_limits(&args, common, bad).unwrap_err().code,
            "lifetime_native_or_mask_limits");
    }
}

#[test]
fn stored_input_and_result_caps_cannot_outrun_cli_or_task_admission() {
    let (c, args) = command("start", &[]);
    let common = c.args.validate(&args).unwrap();
    let limits = lifetime(&c.args, &args, common);
    let mut input = limits; input.max_input_bytes_per_item += 1;
    assert_eq!(c.args.lifetime_limits(&args, common, input).unwrap_err().code, "stored_input_ceiling");
    let mut output = limits; output.max_result_bytes -= 1;
    assert_eq!(c.args.lifetime_limits(&args, common, output).unwrap_err().code, "stored_result_ceiling");
    let mut invalid = limits; invalid.max_attempts = 0;
    assert_eq!(c.args.lifetime_limits(&args, common, invalid).unwrap_err().code, "job_limits");
}

#[test]
fn full_population_and_two_preparation_lifetimes_fit_the_process_floor() {
    let (c, args) = command("start", &[]);
    let common = c.args.validate(&args).unwrap();
    let limits = lifetime(&c.args, &args, common);
    let full = c.args.lifetime_limits(&args, common, limits).unwrap();
    assert_eq!(full.preparation_reserve_bytes, common.preparation_bytes);
    assert!(full.buffer_bytes > common.preparation_bytes + 2 * limits.max_snapshot_bytes);
    assert_eq!(full.common.minimum_memory_bytes,
        common.minimum_memory_bytes + full.buffer_bytes);
    let mut insufficient = common;
    insufficient.memory_bytes = full.common.minimum_memory_bytes - 1;
    assert_eq!(c.args.lifetime_limits(&args, insufficient, limits).unwrap_err().code, "job_population_memory");
    insufficient.memory_bytes += 1;
    assert!(c.args.lifetime_limits(&args, insufficient, limits).is_ok());
}

#[test]
fn full_population_arithmetic_and_explicit_staging_cannot_overflow() {
    let (mut c, args) = command("start", &[]);
    let common = c.args.validate(&args).unwrap();
    let mut limits = lifetime(&c.args, &args, common);
    limits.max_snapshot_bytes = i64::MAX as u64;
    assert_eq!(c.args.lifetime_limits(&args, common, limits).unwrap_err().code, "job_memory_arithmetic");
    c.args.storage.journal_memory_mib = u64::MAX;
    assert_eq!(c.args.validate(&args).unwrap_err().code, "job_memory_options");
    c.args.storage.journal_memory_mib = 64;
    c.args.storage.serialization_memory_mib = 0;
    assert_eq!(c.args.validate(&args).unwrap_err().code, "explicit_storage_or_job_options");
}

#[test]
fn immutable_limit_json_rejects_duplicate_unknown_or_incomplete_authority() {
    let (c, args) = command("start", &[]);
    let common = c.args.validate(&args).unwrap();
    let limits = lifetime(&c.args, &args, common);
    let text = serde_json::to_string(&limits).unwrap();
    assert_eq!(job_cli::parse_limits(&text).unwrap(), limits);
    assert!(job_cli::parse_limits(&text.replacen('{', "{\"max_items\":1,", 1)).is_err());
    let mut unknown = serde_json::to_value(limits).unwrap();
    unknown["group_width"] = serde_json::json!(128);
    assert!(job_cli::parse_limits(&unknown.to_string()).is_err());
    assert!(job_cli::parse_limits("{}").is_err());
}

#[test]
fn shared_schema_preserves_exact_decimals_and_binds_source_later() {
    let (c, args) = command("start", &[]);
    let common = c.args.validate(&args).unwrap();
    let budget = c.args.extraction.budget(&args, common);
    let schema = r#"{"type":"object","properties":{"n":{"type":"number","const":12345678901234567890123456789012345678}},"required":["n"],"additionalProperties":false}"#;
    assert_eq!(c.args.extraction.shared_defaults(schema.to_owned(), budget).unwrap().schema, schema);
    let (source, args) = command("start", &["--source-membership"]);
    let common = source.args.validate(&args).unwrap();
    let budget = source.args.extraction.budget(&args, common);
    let schema = r#"{"type":"object","properties":{"value":{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}},"required":["value"],"additionalProperties":false}"#;
    let defaults = source.args.extraction.shared_defaults(schema.to_owned(), budget).unwrap();
    assert_eq!(defaults.schema, schema);
    assert_eq!(defaults.grounding, ExtractionBatchGrounding::SourceMembership);
    assert!(source.args.extraction.shared_defaults(
        r#"{"type":"object","type":"object"}"#.to_owned(), budget).is_err());
}


#[test]
fn failure_reports_are_metadata_only_and_do_not_imply_rollback() {
    let (mut c, args) = command("resume", &[]);
    c.args.storage.store_results = false;
    c.args.storage.job_dir = PathBuf::from("private-content-marker");
    let mut stdout = Vec::new(); let mut diagnostics = Vec::new();
    let code = c.run_owned(args, NoRead, &mut stdout, &mut diagnostics);
    assert_eq!(code, ErrorCode::Usage.as_process_exit());
    assert!(stdout.is_empty());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume");
    assert_eq!(report["code"], "explicit_storage_or_job_options");
    assert_eq!(report["durable_progress_may_exist"], true);
    let text = std::str::from_utf8(&diagnostics).unwrap();
    for private in ["private-content-marker", "schema.json", "private/key", "0123456789abcdef"] {
        assert!(!text.contains(private));
    }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_before_reading_any_private_file_or_input() {
    let (c, args) = command("resume", &["--discard-uncommitted-tail"]);
    assert_eq!(c.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code,
        "owned_job_profile_unavailable");
}
