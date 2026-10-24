//! Durable command authority, complete-population limits and independent key scopes.
use super::*;
use crate::tasks::redact::batch::RedactionBatchArgs;

fn argv(operation: &str) -> Vec<String> {
    ["redact-job", operation, "--source-dir", "explicit-source", "--source-manifest", "explicit-manifest",
        "--memory-mib", "65536", "--store-results", "--job-dir", "private-job", "--key-file", "private-job-key",
        "--job-id", "0123456789abcdef0123456789abcdef", "--limits", "private-limits"]
        .into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
pub(super) fn command(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation); words.extend(extra.iter().map(|s| (*s).to_owned()));
    parse(words).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, document: DocumentLimits) -> JobLimits {
    let mut item = document.mapping.max_model_work;
    item.projected_logits -= item.projected_logits % NANBEIGE_VOCAB_SIZE as u64;
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: args.max_input_bytes * 2 + 1024,
        max_snapshot_bytes: 16 * MIB, max_result_bytes: document.common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(item, document.mapping.max_mask_visits) }
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refusal consumed private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refusal published private output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refusal flushed output"); }
}

#[test]
fn complete_dispatch_requires_owned_input_and_supports_both_model_routes_and_modes() {
    for operation in ["start", "resume"] {
        let (job, args) = command(operation, &["--materialize"]);
        assert_eq!(job.mode.name(), operation);
        assert!(job.args.storage.store_results && job.args.storage.materialize);
        assert!(job.args.validate(&args).unwrap().common.snapshot_pin.is_none());
        let mut words = argv(operation);
        let at = words.iter().position(|s| s == "--source-dir").unwrap(); words.drain(at..at + 4);
        words.extend(["--snapshot", "pinned.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
        words.push("ab".repeat(32));
        let (snapshot, args) = parse(words).unwrap();
        assert!(snapshot.args.validate(&args).unwrap().common.snapshot_pin.is_some());
        let mut root = vec!["candidate".to_owned(), "source-int4".to_owned()];
        root.extend(argv(operation));
        let matches = crate::candidate_cli::definition().try_get_matches_from(root).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong candidate dispatcher"); };
        assert!(matches!(command.task, Int4Task::RedactJob));
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn repair_is_resume_only_and_retention_paths_and_immutable_limits_are_explicit() {
    assert!(matches!(command("resume", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    for required in ["--job-dir", "--key-file", "--job-id", "--limits"] {
        let mut words = argv("start");
        let at = words.iter().position(|s| s == required).unwrap(); words.drain(at..at + 2);
        assert!(parse(words).is_err());
    }
    let mut words = argv("start"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
    assert!(definition().try_get_matches_from(["redact-job"]).is_err());
    for operation in ["status", "verify", "materialize", "partition", "merge"] {
        assert!(definition().try_get_matches_from(["redact-job", operation]).is_err());
    }
    // Independently requested snapshot setup keeps the existing INT4 job contract.
    assert!(command("resume", &["--save-snapshot", "new-model.int4"]).0.args.storage.store_results);
}

#[test]
fn storage_and_private_key_refusals_happen_before_any_owned_io() {
    for case in 0..8 {
        let (mut job, args) = command("start", &[]);
        match case {
            0 => job.args.storage.store_results = false,
            1 => job.args.storage.job_dir = PathBuf::from("-"),
            2 => job.args.storage.key_file = PathBuf::new(),
            3 => job.args.storage.limits_file = PathBuf::from("-"),
            4 => job.args.storage.max_stream_mib = 0,
            5 => job.args.storage.max_input_lines = 0,
            6 => job.args.storage.journal_memory_mib = 0,
            _ => job.args.storage.serialization_memory_mib = 0,
        }
        assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_storage_or_job_options");
    }
    for flags in [vec!["--key-stdin"], vec!["--action", "pseudonymize"],
        vec!["--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"]] {
        let (job, args) = command("start", &flags);
        assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options");
    }
    let (job, args) = command("start", &["population.ndjson", "--action", "pseudonymize",
        "--key-stdin", "--key-id", "rotation", "--namespace", "private-scope"]);
    assert_eq!(job.args.storage.key_file, PathBuf::from("private-job-key"));
    assert!(job.args.validate(&args).is_ok());
}

#[test]
fn fixed_redaction_options_and_original_text_cap_remain_independent_of_encoded_envelopes() {
    let (job, args) = command("start", &["--max-input-bytes", "32", "--rules", "email,date",
        "--action", "placeholder", "--include-map", "--no-verify", "--ner-options", "typed-options.json"]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    assert!(stored.max_input_bytes_per_item > args.max_input_bytes);
    let limits = job.args.lifetime_limits(&args, document, stored).unwrap();
    let request = job.args.document.request(&args, limits.document);
    assert_eq!(request.actions.default_action, RedactionAction::Placeholder);
    assert!(request.actions.include_map); assert!(!request.verify);
    assert_eq!(limits.document.mapping.chunks.max_input_bytes, limits.common.result_bytes);
    assert_eq!(request.rule_budget.max_input_bytes, limits.common.result_bytes);
    assert_eq!(job.args.document.redaction.source.options_path().unwrap(), std::path::Path::new("typed-options.json"));
    assert!(serde_json::from_str::<RedactionBatchArgs>("{}").is_ok());
    for text in [r#"{"verify":false}"#, r#"{"action":"mask"}"#, r#"{"namespace":"changed"}"#,
        r#"{"key":"secret"}"#, r#"{"types":["person"]}"#, r#"{"max_chunks":1}"#] {
        assert!(serde_json::from_str::<RedactionBatchArgs>(text).is_err());
    }
}

#[test]
fn all_lifetime_work_is_copied_exactly_and_items_attempts_or_transport_never_replenish_it() {
    let (mut job, args) = command("resume", &[]);
    let document = job.args.validate(&args).unwrap();
    let mut stored = lifetime(&args, document);
    stored.max_work.model.projected_logits += 17;
    let original = job.args.lifetime_limits(&args, document, stored).unwrap();
    assert_eq!(original.model_work, stored.max_work.int4_model_work());
    assert_eq!(original.masks, stored.max_work.mask_node_visits);
    assert_eq!(original.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 17);
    stored.max_items = 999; stored.max_attempts = 9999;
    job.args.storage.max_stream_mib = 1; job.args.storage.max_input_lines = 1;
    let changed = job.args.lifetime_limits(&args, document, stored).unwrap();
    assert_eq!(changed.model_work, original.model_work); assert_eq!(changed.masks, original.masks);
    assert_eq!(changed.transport.max_stream_bytes, MIB); assert_eq!(changed.transport.max_lines, 1);
    assert!(changed.buffer_bytes > original.buffer_bytes);
}

#[test]
fn every_native_axis_mask_and_retained_result_bound_must_fit_one_complete_attempt() {
    let (job, args) = command("start", &[]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    for axis in 0..6 {
        let mut narrow = stored;
        match axis {
            0 => narrow.max_work.model.forward_positions -= 1,
            1 => narrow.max_work.model.projected_logits -= 1,
            2 => narrow.max_work.model.attention_pairs -= 1,
            3 => narrow.max_work.model.projections.dot_products -= 1,
            4 => narrow.max_work.model.projections.multiply_accumulates -= 1,
            _ => narrow.max_work.mask_node_visits -= 1,
        }
        assert!(matches!(job.args.lifetime_limits(&args, document, narrow),
            Err(Failure { code: "lifetime_native_or_mask_limits", .. })));
    }
    let mut small = stored; small.max_result_bytes = document.common.result_bytes - 1;
    assert!(matches!(job.args.lifetime_limits(&args, document, small),
        Err(Failure { code: "stored_result_ceiling", .. })));
    let mut overflow = stored; overflow.max_snapshot_bytes = i64::MAX as u64;
    assert!(matches!(job.args.lifetime_limits(&args, document, overflow),
        Err(Failure { code: "job_memory_arithmetic", .. })));
}

#[test]
fn population_memory_replaces_capture_but_retains_full_map_native_secret_and_two_preparations() {
    let (job, mut args) = command("start", &[]);
    let document = job.args.validate(&args).unwrap();
    let stored = lifetime(&args, document);
    let limits = job.args.lifetime_limits(&args, document, stored).unwrap();
    let expected = document.common.minimum_memory_bytes - args.max_input_bytes as u64 - 1
        - document.common.preparation_bytes + limits.buffer_bytes + RECIPE_RESERVE_BYTES;
    assert_eq!(limits.common.minimum_memory_bytes, expected);
    assert_eq!(limits.document.workspace_bytes, document.workspace_bytes);
    assert_eq!(limits.common.native, document.common.native);
    assert_eq!(limits.preparation_reserve_bytes, document.common.preparation_bytes);
    let mut larger = stored; larger.max_result_bytes *= 2;
    let expanded = job.args.lifetime_limits(&args, document, larger).unwrap();
    assert_eq!(expanded.common.minimum_memory_bytes - expected, stored.max_result_bytes as u64 * 4);
    assert_eq!(expanded.common.result_bytes, limits.common.result_bytes);
    args.memory_mib = expected.saturating_sub(1) / MIB;
    let smaller = job.args.validate(&args).unwrap();
    assert!(matches!(job.args.lifetime_limits(&args, smaller, stored),
        Err(Failure { code: "job_population_memory", .. })));
    args.memory_mib = expected.div_ceil(MIB);
    let admitted = job.args.validate(&args).unwrap();
    assert!(job.args.lifetime_limits(&args, admitted, stored).is_ok());
}

#[test]
fn live_stream_unrequested_storage_and_generation_options_cannot_silently_change_a_job() {
    for extra in [vec!["--max-requests", "3"], vec!["--ndjson"], vec!["--chunked"],
        vec!["--max-corpus-mask-node-visits", "9"], vec!["--rules-only"],
        vec!["--defaults", "private"], vec!["--spool-input"], vec!["--key", "secret"],
        vec!["--prefill-rows", "1"], vec!["--seed", "ab"], vec!["--logprobs"]] {
        let mut words = argv("start"); words.extend(extra.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
    for bad in ["AB".repeat(16), "ab".repeat(15), "é".repeat(16), "x".repeat(32)] {
        let mut words = argv("start"); let at = words.iter().position(|s| s == "--job-id").unwrap() + 1;
        words[at] = bad; assert!(parse(words).is_err());
    }
    let (job, mut args) = command("start", &[]); args.seed = Some("ab".repeat(32));
    assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options");
}

#[test]
fn exact_kernel_selection_is_hardware_checked_without_changing_job_or_storage_authority() {
    let (baseline, args) = command("start", &[]);
    let baseline = baseline.args.validate(&args).unwrap();
    for kind in Int8KernelKind::ALL {
        let (job, args) = command("start", &["--kernel", kind.id()]);
        assert_eq!(args.kernel, kind);
        match PackedInt4Kernel::detect(kind) {
            Ok(_) => {
                let selected = job.args.validate(&args).unwrap();
                assert_eq!(selected.common.kernel.kind(), kind);
                assert_eq!(selected.common.minimum_memory_bytes, baseline.common.minimum_memory_bytes);
                assert_eq!(selected.common.native, baseline.common.native);
                assert_eq!(selected.mapping.max_model_work, baseline.mapping.max_model_work);
            }
            Err(_) => assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "invalid_job_options"),
        }
    }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_owned_profile_refuses_without_reading_keys_or_the_population() {
    for operation in ["start", "resume"] {
        let (job, args) = command(operation, &[]);
        assert_eq!(job.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code, "owned_job_profile_unavailable");
    }
}
