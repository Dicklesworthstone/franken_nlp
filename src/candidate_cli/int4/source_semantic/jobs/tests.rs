//! Parser, immutable admission and private-IO refusal regressions.
//! These source fixtures do not execute a model or acknowledge durable output.
use super::*;
use crate::{jobs::JobWork, tokenizer::untrusted::UntrustedEncoding};

const PIN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
fn name(kind: Kind) -> &'static str {
    match kind { Kind::Summarize => "summarize-semantic-job", Kind::Answer => "answer-semantic-job",
        _ => panic!("summary or answer fixture required") }
}
fn argv(kind: Kind, operation: &str, snapshot: bool, extra: &[&str]) -> Vec<String> {
    let mut words = vec!["candidate", "source-int4", name(kind), operation,
        "--memory-mib", "65536", "--semantic-options", "private-review-policy.json",
        "--job-dir", "private/job", "--key-file", "private/key", "--limits", "private-limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
    if snapshot { words.extend(["--snapshot", "private.snapshot", "--expected-model-sha256", PIN]); }
    else { words.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest.json"]); }
    words.extend_from_slice(extra);
    words.into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<Int4Command, clap::Error> {
    let matches = crate::candidate_cli::definition().try_get_matches_from(words)?;
    match CandidateCommand::from_matches(&matches)? {
        CandidateCommand::SourceInt4(command) => Ok(command),
        _ => panic!("wrong candidate dispatcher"),
    }
}
pub(super) fn command(kind: Kind, operation: &str, snapshot: bool, extra: &[&str]) -> (JobCommand, Int4Args) {
    let command = parse(argv(kind, operation, snapshot, extra)).unwrap();
    assert!(matches!(command.task, Int4Task::SourceSemanticJob));
    assert!(command.batch.is_none());
    let Some(Int4JobCommand::SourceSemantic(job)) = command.job else { panic!("wrong retained semantic source route"); };
    assert_eq!(job.args.kind, kind);
    (job, command.args)
}
pub(super) fn options(kind: Kind) -> Int4SourceSemanticOptions { super::super::tests::fixture(kind).3 }
pub(super) fn lifetime(command: &JobArgs, args: &Int4Args, options: &Int4SourceSemanticOptions) -> JobLimits {
    let twice = options.max_model_work.checked_add(options.max_model_work).unwrap();
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: options.limits.max_output_bytes as usize,
        max_spool_bytes: 32 * MIB, max_materialized_bytes: 32 * MIB, max_journal_bytes: 64 * MIB,
        max_attempts: 5, max_work: JobWork::from_int4(twice, command.semantic.source.mask_visits() * 3) }
}
pub(super) struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused job read private population"); }
}
pub(super) struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused job flushed output"); }
}

#[test]
fn both_tasks_keep_exact_model_options_storage_and_compact_team_contracts() {
    for kind in [Kind::Summarize, Kind::Answer] { for operation in ["start", "resume"] {
        for snapshot in [false, true] { for compact in [false, true] {
            let mut flags = vec!["--compute-threads", "4", "--materialize", "--options", "private-source-options.json",
                if operation == "start" { "--spool-input" } else { "--use-retained-input" },
                "--max-retained-input-mib", "8"];
            if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
            let (command, args) = command(kind, operation, snapshot, &flags);
            command.args.storage.input_retention.validate(command.mode, &args.input).unwrap();
            let initial = command.args.validate(&args).unwrap();
            let options = options(kind); let job = lifetime(&command.args, &args, &options);
            let limits = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
            let task = load_task(kind, None, limits.task, limits.planning, &options).unwrap();
            assert_eq!(task.built_in(), source::task(kind)); assert_eq!(task.budget(), limits.task);
            assert_eq!(command.mode.name(), operation);
            assert_eq!(args.input, PathBuf::from("-"));
            assert_eq!(initial.snapshot_pin.is_some(), snapshot);
            assert_eq!(limits.common.compute.threads, 4); assert!(limits.common.prefill.is_none());
            assert_eq!(limits.common.native, initial.native);
            assert_eq!(limits.task.max_output_bytes, 1 << 20);
            assert_eq!(limits.common.result_bytes, options.limits.max_output_bytes as usize);
            assert_eq!(command.args.semantic.source.options_path(), Some(std::path::Path::new("private-source-options.json")));
            assert_eq!(command.args.semantic.semantic_options, PathBuf::from("private-review-policy.json"));
            assert_eq!(job_cli::job_id_hex(command.args.storage.job_id), "0123456789abcdef0123456789abcdef");
            assert!(command.args.storage.store_results && command.args.storage.materialize);
            assert!(args.input_encoding.is_default(), "semantic route owns the whole-attempt encoding ledger");
            assert_eq!(matches!(command.args.semantic.input_encoding.policy().unwrap(), UntrustedEncoding::SafeBpeV1(_)), compact);
        } }
        let (serial, args) = command(kind, "start", false, &["--save-snapshot", "private-new.snapshot"]);
        let common = serial.args.validate(&args).unwrap();
        assert_eq!(common.compute.threads, 1); assert_eq!(common.compute.workspace_bytes, 0);
        assert_eq!(args.save_snapshot, Some(PathBuf::from("private-new.snapshot")));
    } }
}

#[test]
fn owned_dispatch_refuses_borrowed_io_and_invalid_storage_without_private_data() {
    for kind in [Kind::Summarize, Kind::Answer] {
        assert_eq!(parse(argv(kind, "start", false, &[])).unwrap().execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Arguments));
        let mut command = parse(argv(kind, "resume", true, &[])).unwrap();
        let Some(Int4JobCommand::SourceSemantic(job)) = &mut command.job else { panic!("wrong dispatcher"); };
        job.args.storage.store_results = false;
        let mut diagnostics = Vec::new();
        assert_eq!(command.run_owned(NoRead, NoWrite, &mut diagnostics), ErrorCode::Usage.as_process_exit());
        let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(report["operation"], "resume"); assert_eq!(report["task"], task_name(kind));
        assert_eq!(report["code"], "semantic_source_job_storage_options");
        assert_eq!(report["durable_progress_may_exist"], true); assert!(report.get("cancellation").is_none());
        assert!(diagnostics.len() < job_cli::REPORT_BYTES);
        let text = String::from_utf8(diagnostics).unwrap();
        for private in ["private-review", "private.snapshot", "private/job", "private/key", PIN] {
            assert!(!text.contains(private));
        }
    }
}

#[test]
fn native_and_mask_lifetime_axes_never_scale_with_population_or_attempt_limits() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (mut command, args) = command(kind, "start", false, &[]); let options = options(kind);
        let initial = command.args.validate(&args).unwrap(); let job = lifetime(&command.args, &args, &options);
        let first = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
        assert_eq!(first.native.max_model_work, job.max_work.int4_model_work());
        assert_eq!(first.native.masks.max_visits_per_run, job.max_work.mask_node_visits);
        assert_eq!(first.native.masks.max_visits_per_item, command.args.semantic.source.mask_visits());
        let mut larger = job; larger.max_items = 100_000; larger.max_attempts = 1_000_000;
        command.args.storage.max_stream_mib *= 2; command.args.storage.max_input_lines *= 2;
        let changed = command.args.lifetime_limits(&args, initial, &options, larger).unwrap();
        assert_eq!(changed.native.max_model_work, first.native.max_model_work);
        assert_eq!(changed.native.masks.max_visits_per_run, first.native.masks.max_visits_per_run);
        assert_eq!(changed.native.masks.max_visits_per_item, first.native.masks.max_visits_per_item);
        assert_eq!(changed.task, first.task); assert_eq!(changed.common.native, first.common.native);
        assert_eq!(changed.transport.max_stream_bytes, first.transport.max_stream_bytes * 2);
        assert_eq!(changed.transport.max_lines, first.transport.max_lines * 2);
        assert_eq!(changed.io_reserve_bytes, first.io_reserve_bytes);
        for axis in 0..6 {
            let mut short = job;
            match axis {
                0 => short.max_work.model.forward_positions = options.max_model_work.forward_positions - 1,
                1 => short.max_work.model.projected_logits = options.max_model_work.projected_logits - 1,
                2 => short.max_work.model.attention_pairs = options.max_model_work.attention_pairs - 1,
                3 => short.max_work.model.projections.dot_products = options.max_model_work.projections.dot_products - 1,
                4 => short.max_work.model.projections.multiply_accumulates = options.max_model_work.projections.multiply_accumulates - 1,
                _ => short.max_work.mask_node_visits = command.args.semantic.source.mask_visits() - 1,
            }
            assert_eq!(command.args.lifetime_limits(&args, initial, &options, short).unwrap_err().code,
                "semantic_source_job_lifetime_work", "{kind:?} axis={axis}");
        }
    }
}

#[test]
fn complete_qa_input_and_combined_result_bounds_follow_the_frozen_record() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (command, args) = command(kind, "resume", true, &[]); let options = options(kind);
        let initial = command.args.validate(&args).unwrap(); let job = lifetime(&command.args, &args, &options);
        let limits = command.args.lifetime_limits(&args, initial, &options, job).unwrap();
        assert_eq!(limits.planning.max_input_bytes, job.max_input_bytes_per_item);
        for invalid in 0..4 {
            let mut short = job;
            match invalid {
                0 => short.max_result_bytes = limits.task.max_output_bytes as usize,
                1 => short.max_result_bytes -= 1,
                2 => short.max_input_bytes_per_item += 1,
                _ => short.max_id_bytes = 257,
            }
            assert_eq!(command.args.lifetime_limits(&args, initial, &options, short).unwrap_err().code,
                "semantic_source_job_input_or_combined_result_limit");
        }
        let mut smaller = job; smaller.max_input_bytes_per_item = 2048;
        let constrained = command.args.lifetime_limits(&args, initial, &options, smaller).unwrap();
        let facts = command.args.input_ceiling(constrained, &options);
        assert_eq!(constrained.planning.max_input_bytes, 2048); assert_eq!(facts.source_bytes, 2048);
        assert_eq!(facts.question_bytes, if kind == Kind::Answer { 2048 } else { 0 });
        assert_eq!(facts.passage_count, if kind == Kind::Answer { constrained.planning.max_passages } else { 0 });
        assert_eq!(constrained.task, limits.task);
        assert_eq!(constrained.native.max_model_work, limits.native.max_model_work);
    }
}

#[test]
fn source_defaults_use_the_same_closed_compiler_as_the_resident_semantic_route() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (command, args) = command(kind, "start", false, &[]); let semantic = options(kind);
        let limits = command.args.lifetime_limits(&args, command.args.validate(&args).unwrap(),
            &semantic, lifetime(&command.args, &args, &semantic)).unwrap();
        let default = load_task(kind, None, limits.task, limits.planning, &semantic).unwrap();
        let value = match kind { Kind::Summarize => serde_json::to_value(SummaryOptions::default()).unwrap(),
            Kind::Answer => serde_json::to_value(AnswerOptions::default()).unwrap(), _ => unreachable!() };
        assert_eq!(load_task(kind, Some(&value.to_string()), limits.task, limits.planning, &semantic).unwrap(), default);
        for field in ["task", "budget", "semantic_options", "identity", "prompt", "passages"] {
            let mut invalid = value.clone(); invalid[field] = serde_json::json!({});
            assert!(load_task(kind, Some(&invalid.to_string()), limits.task, limits.planning, &semantic).is_err());
        }
        assert!(load_task(kind, Some("{}"), limits.task, limits.planning, &semantic).is_err());
        let mut short = limits.planning; short.compiler.max_states = 1;
        assert!(load_task(kind, None, limits.task, short, &semantic).is_err());
    }
}

#[test]
fn retention_and_tail_repair_remain_explicit_and_operation_specific() {
    for kind in [Kind::Summarize, Kind::Answer] {
        assert!(matches!(command(kind, "resume", false, &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
        assert!(matches!(command(kind, "resume", false, &["--discard-uncommitted-tail"]).0.mode,
            RunMode::Resume { discard_uncommitted: true }));
        assert!(parse(argv(kind, "start", false, &["--discard-uncommitted-tail"])).is_err());
        for (operation, flag) in [("start", "--use-retained-input"), ("resume", "--spool-input")] {
            let (command, args) = command(kind, operation, false, &[flag, "--max-retained-input-mib", "8"]);
            assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "job_input_retention_options");
        }
        for flags in [&["--use-retained-input"][..],
            &["--use-retained-input", "--max-retained-input-mib", "8", "population.ndjson"][..],
            &["--use-retained-input", "--max-retained-input-mib", "8", "-"][..]] {
            assert!(parse(argv(kind, "resume", true, flags)).is_err());
        }
        assert!(parse(argv(kind, "verify", false, &[])).is_err());
    }
}

#[test]
fn unsupported_generation_projection_and_live_batch_controls_are_not_inherited() {
    for kind in [Kind::Summarize, Kind::Answer] {
        for flags in [&["--seed", PIN][..], &["--logprobs"][..], &["--prefill-rows", "1"][..],
            &["--max-selected-rows", "32"][..], &["--cohort-documents", "2"][..],
            &["--defaults", "private-defaults"][..], &["--think"][..],
            &["--max-output-bytes", "65536"][..], &["--min-new-tokens", "0"][..],
            &["--max-total-forward-positions", "99999"][..],
            &["--max-total-tokenizer-input-bytes", "99999"][..]] {
            assert!(parse(argv(kind, "start", false, flags)).is_err(), "{kind:?} {flags:?}");
        }
        for required in ["--semantic-options", "--job-dir", "--key-file", "--limits", "--job-id"] {
            let mut words = argv(kind, "start", false, &[]);
            let at = words.iter().position(|word| word == required).unwrap(); words.drain(at..at + 2);
            assert!(parse(words).is_err(), "{kind:?} {required}");
        }
    }
}

#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_owned_profile_refuses_before_private_configuration_or_population() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (command, args) = command(kind, "start", false, &[]);
        assert_eq!(command.execute_owned(args, NoRead, &mut NoWrite).unwrap_err().code, "owned_job_profile_unavailable");
    }
}
