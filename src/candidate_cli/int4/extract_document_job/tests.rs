//! UNRUN argument, lifetime and privacy fixtures; no native output is fabricated.
use super::*;

fn argv(operation: &str) -> Vec<String> {
    ["extract-document-job", operation, "--source-dir", "sources", "--source-manifest", "manifest.json",
        "--memory-mib", "65536", "--max-chunks", "8", "--schema", "private-schema.json",
        "--job-dir", "private/job", "--key-file", "private/key", "--limits", "limits.json",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"]
        .into_iter().map(str::to_owned).collect()
}
fn parse(words: Vec<String>) -> Result<(JobCommand, Int4Args), clap::Error> {
    JobCommand::from_matches(&definition().try_get_matches_from(words)?)
}
pub(super) fn command(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = argv(operation); words.extend(extra.iter().map(|word| (*word).to_owned()));
    parse(words).unwrap()
}
pub(super) fn lifetime(args: &Int4Args, map: MapLimits) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: map.max_result_bytes,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork::from_int4(map.mapping.mapping.max_model_work, map.mapping.mapping.max_mask_visits) }
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused extraction document job read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused extraction document job published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused extraction document job flushed output"); }
}

#[test]
fn start_and_resume_use_complete_document_limits_and_an_explicit_fixed_schema() {
    for operation in ["start", "resume"] {
        let (c, args) = command(operation, &[]); let map = c.args.validate(&args).unwrap();
        assert_eq!(c.mode.name(), operation); assert!(c.args.storage.store_results);
        assert_eq!(c.args.map.extraction.schema_path(), std::path::Path::new("private-schema.json"));
        assert_eq!(map.mapping.mapping.chunks.max_chunks, 8);
        assert_eq!(map.mapping.mapping.chunks.max_input_bytes, args.max_input_bytes);
        assert_eq!(map.mapping.mapping.chunks.context_tokens, args.context_tokens);
        assert_eq!(map.max_result_bytes, 16_777_216);
        assert_eq!(map.common.result_bytes, 1_048_576);
        let admitted = c.args.lifetime_limits(&args, map, lifetime(&args, map)).unwrap();
        assert_eq!(admitted.map.mapping.mapping.max_model_work, map.mapping.mapping.max_model_work);
        assert_eq!(admitted.map.mapping.mapping.max_mask_visits, map.mapping.mapping.max_mask_visits);
        assert!(admitted.map.common.minimum_memory_bytes > map.common.minimum_memory_bytes);
    }
}
#[test]
fn pinned_snapshot_and_source_conversion_preserve_schema_and_document_work() {
    let (source, source_args) = command("resume", &["--source-membership"]);
    let original = source.args.validate(&source_args).unwrap();
    let mut words = argv("resume");
    let at = words.iter().position(|s| s == "--source-dir").unwrap(); words.drain(at..at + 4);
    words.extend(["--snapshot", "candidate.int4", "--expected-model-sha256"].into_iter().map(str::to_owned));
    words.push("ab".repeat(32)); words.push("--source-membership".to_owned());
    let (snapshot, args) = parse(words).unwrap(); let loaded = snapshot.args.validate(&args).unwrap();
    assert!(loaded.common.snapshot_pin.is_some()); assert!(snapshot.args.map.extraction.source_membership());
    assert_eq!(loaded.mapping.mapping.max_model_work, original.mapping.mapping.max_model_work);
    assert_eq!(loaded.mapping.mapping.max_mask_visits, original.mapping.mapping.max_mask_visits);
    assert_eq!(loaded.workspace_bytes, original.workspace_bytes);
    let mut unpinned = argv("resume");
    let at = unpinned.iter().position(|s| s == "--source-dir").unwrap(); unpinned.drain(at..at + 4);
    unpinned.extend(["--snapshot", "candidate.int4"].into_iter().map(str::to_owned));
    assert!(parse(unpinned).is_err());
}
#[test]
fn explicit_operation_schema_and_tail_recovery_are_required_or_scoped() {
    assert!(definition().try_get_matches_from(["extract-document-job"]).is_err());
    let mut no_schema = argv("start");
    let at = no_schema.iter().position(|s| s == "--schema").unwrap(); no_schema.drain(at..at + 2);
    assert!(parse(no_schema).is_err());
    assert!(matches!(command("resume", &[]).0.mode, RunMode::Resume { discard_uncommitted: false }));
    assert!(matches!(command("resume", &["--discard-uncommitted-tail"]).0.mode,
        RunMode::Resume { discard_uncommitted: true }));
    let mut words = argv("start"); words.push("--discard-uncommitted-tail".to_owned());
    assert!(parse(words).is_err());
}
#[test]
fn generation_and_other_document_task_controls_cannot_enter_extraction_jobs() {
    for flags in [vec!["--max-output-bytes", "65536"], vec!["--prefill-rows", "1"],
        vec!["--min-new-tokens", "0"], vec!["--logit-bias", "42=0"],
        vec!["--question", "private-question.txt"], vec!["--synthesize-summary"], vec!["--map-only"]] {
        let mut words = argv("start"); words.extend(flags.into_iter().map(str::to_owned));
        assert!(parse(words).is_err());
    }
}
#[test]
fn explicit_storage_local_configuration_and_nonzero_lifetime_caps_precede_io() {
    for axis in 0..11 {
        let (mut c, args) = command("start", &[]);
        match axis {
            0 => c.args.storage.store_results = false, 1 => c.args.storage.job_dir = PathBuf::from("-"),
            2 => c.args.storage.key_file = PathBuf::new(), 3 => c.args.storage.limits_file = PathBuf::from("-"),
            4 => c.args.storage.journal_memory_mib = 0, 5 => c.args.storage.serialization_memory_mib = 0,
            6 => c.args.storage.max_stream_mib = 0, 7 => c.args.storage.max_input_lines = 0,
            8 => c.args.lifetime.max_lifetime_extraction_fields = 0,
            9 => c.args.lifetime.max_lifetime_extraction_evidence_spans = 0,
            _ => c.args.lifetime.max_lifetime_extraction_scan_steps = 0,
        }
        assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
            "explicit_extraction_document_job_storage");
    }
    for value in ["AB".repeat(16), "ab".repeat(15), "é".repeat(16), "x".repeat(32)] {
        let mut words = argv("start");
        let at = words.iter().position(|s| s == "--job-id").unwrap() + 1; words[at] = value;
        assert!(parse(words).is_err());
    }
}
#[test]
fn per_document_chunk_work_output_and_verification_refuse_before_private_input() {
    for flags in [vec!["--max-chunk-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-map-result-bytes", "1048575"], vec!["--reduction-reserve-mib", "0"],
        vec!["--max-total-mask-node-visits", "1"], vec!["--max-forward-positions", "0"],
        vec!["--max-extraction-fields", "0"], vec!["--max-extraction-evidence-spans", "0"],
        vec!["--max-extraction-scan-steps", "0"], vec!["--preparation-mib", "1"]] {
        let (c, args) = command("resume", &flags);
        assert!(c.execute_owned(args, NoIo, &mut NoIo).is_err());
    }
}
#[test]
fn all_five_native_axes_and_masks_must_fit_exact_lifetime_work() {
    let (mut c, args) = command("start", &[]); let map = c.args.validate(&args).unwrap();
    let fixed = lifetime(&args, map);
    for axis in 0..6 {
        let mut job = fixed;
        match axis { 0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1, 2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            _ => job.max_work.mask_node_visits -= 1 }
        assert_eq!(c.args.lifetime_limits(&args, map, job).unwrap_err().code, "lifetime_document_work_limits");
    }
    let first = c.args.lifetime_limits(&args, map, fixed).unwrap();
    let mut larger = fixed; larger.max_items = 101; larger.max_attempts = 1000;
    c.args.storage.max_stream_mib = 1024; c.args.storage.max_input_lines = 1_000_000;
    let second = c.args.lifetime_limits(&args, map, larger).unwrap();
    assert_eq!(first.map.mapping.mapping.max_model_work, second.map.mapping.mapping.max_model_work);
    assert_eq!(first.map.mapping.mapping.max_mask_visits, second.map.mapping.mapping.max_mask_visits);
    assert!(second.buffer_bytes > first.buffer_bytes);
}
#[test]
fn every_verification_axis_covers_all_attempts_on_start_and_resume() {
    for operation in ["start", "resume"] {
        let (mut c, args) = command(operation, &[]); let map = c.args.validate(&args).unwrap();
        let fixed = lifetime(&args, map); let per = map.mapping.verification;
        let exact = LifetimeArgs {
            max_lifetime_extraction_fields: per.max_fields * fixed.max_attempts as usize,
            max_lifetime_extraction_evidence_spans: per.max_matches * fixed.max_attempts as usize,
            max_lifetime_extraction_scan_steps: per.max_scan_steps * fixed.max_attempts,
        };
        c.args.lifetime = exact;
        assert!(c.args.lifetime_limits(&args, map, fixed).is_ok());
        for axis in 0..3 {
            c.args.lifetime = exact;
            match axis { 0 => c.args.lifetime.max_lifetime_extraction_fields -= 1,
                1 => c.args.lifetime.max_lifetime_extraction_evidence_spans -= 1,
                _ => c.args.lifetime.max_lifetime_extraction_scan_steps -= 1 }
            assert_eq!(c.args.lifetime_limits(&args, map, fixed).unwrap_err().code,
                "lifetime_extraction_verification");
        }
    }
    let (c, args) = command("start", &[]); let map = c.args.validate(&args).unwrap();
    let overflow = GroundingBudget { max_fields: 1, max_matches: 1, max_scan_steps: u64::MAX };
    assert!(c.args.lifetime.check(overflow, 2).is_err());
    assert!(c.args.lifetime.check(map.mapping.verification, 0).is_err());
}
#[test]
fn schema_bytes_decimal_precision_and_source_membership_are_not_rewritten() {
    let (c, args) = command("start", &[]); let map = c.args.validate(&args).unwrap();
    let budget = c.args.map.extraction.budget(&args, map.common);
    let raw = " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678}\n";
    assert_eq!(c.args.map.request(raw.to_owned(), budget).unwrap().schema, raw);
    for raw in ["{", r#"{"type":"boolean","type":"string"}"#, r#"{"$ref":"remote"}"#] {
        assert!(c.args.map.request(raw.to_owned(), budget).is_err());
    }
    let (c, _) = command("resume", &["--source-membership"]);
    let raw = r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#;
    let request = c.args.map.request(raw.to_owned(), budget).unwrap();
    assert_eq!(request.schema, raw);
    assert_eq!(request.grounding, crate::batch::extract::ExtractionBatchGrounding::SourceMembership);
    assert_eq!(request.budget.max_kv_bytes, budget.max_kv_bytes);
}
#[test]
fn complete_population_and_output_memory_are_added_to_the_real_document_quote() {
    let (c, args) = command("start", &[]); let map = c.args.validate(&args).unwrap();
    let fixed = lifetime(&args, map); let all = c.args.lifetime_limits(&args, map, fixed).unwrap();
    assert_eq!(all.map.common.minimum_memory_bytes, map.common.minimum_memory_bytes + all.buffer_bytes);
    let mut tight = map; tight.common.memory_bytes = all.map.common.minimum_memory_bytes - 1;
    assert_eq!(c.args.lifetime_limits(&args, tight, fixed).unwrap_err().code, "job_population_memory");
    tight.common.memory_bytes += 1; assert!(c.args.lifetime_limits(&args, tight, fixed).is_ok());
    let mut input = fixed; input.max_input_bytes_per_item += 1;
    assert_eq!(c.args.lifetime_limits(&args, map, input).unwrap_err().code, "stored_input_ceiling");
    let mut output = fixed; output.max_result_bytes -= 1;
    assert_eq!(c.args.lifetime_limits(&args, map, output).unwrap_err().code, "stored_document_result_ceiling");
    let mut huge = fixed; huge.max_snapshot_bytes = i64::MAX as u64;
    assert_eq!(c.args.lifetime_limits(&args, map, huge).unwrap_err().code, "job_memory_arithmetic");
}
#[test]
fn root_dispatch_requires_owned_input_and_emits_private_safe_durable_failures() {
    let mut words = vec!["candidate".to_owned(), "source-int4".to_owned()]; words.extend(argv("resume"));
    let matches = crate::candidate_cli::definition().try_get_matches_from(words.clone()).unwrap();
    let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong extraction document dispatcher"); };
    assert_eq!(c.task, Int4Task::ExtractDocumentJob);
    assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    words.retain(|word| word != "--store-results");
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(c) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong extraction document dispatcher"); };
    let mut diagnostics = Vec::new();
    assert_eq!(c.run_owned(NoIo, NoIo, &mut diagnostics), ErrorCode::Usage.as_process_exit());
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["operation"], "resume"); assert_eq!(report["task"], "extract");
    assert_eq!(report["durable_progress_may_exist"], true);
    let text = std::str::from_utf8(&diagnostics).unwrap();
    for private in ["private-schema.json", "private/key", "private/job", "0123456789abcdef"] {
        assert!(!text.contains(private));
    }
}
#[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[test]
fn unavailable_profile_refuses_before_schema_key_model_or_population_access() {
    let (c, args) = command("resume", &["--source-membership"]);
    assert_eq!(c.execute_owned(args, NoIo, &mut NoIo).unwrap_err().code,
        "owned_extraction_document_job_profile_unavailable");
}
