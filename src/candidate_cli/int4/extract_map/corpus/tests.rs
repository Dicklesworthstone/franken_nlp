//! Complete-source command admission; fixtures do not perform native inference.
use super::*;
use crate::batch::extract::ExtractionBatchGrounding;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "extract-corpus", "--memory-mib", "131072"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("source INT4 required"); };
    Ok(command)
}
fn command(flags: &[&str]) -> Int4Command {
    let mut all = vec!["--source-dir", "private-source", "--source-manifest", "private-manifest"];
    if !flags.contains(&"--schema") { all.extend_from_slice(&["--schema", "private-schema.json"]); }
    all.extend_from_slice(flags);
    parsed(&all).unwrap()
}
pub(super) fn arguments(flags: &[&str]) -> (CorpusArgs, Int4Args) {
    let command = command(flags);
    let Int4Task::ExtractCorpus(options) = command.task else { panic!("extraction corpus required"); };
    (options, command.args)
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused corpus read private population"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused corpus published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused corpus flushed output"); }
}

#[test]
fn selected_projection_composes_with_exact_array_reduction_without_renewing_stream_limits() {
    for flags in [&["--root-array-union"][..],
        &["--root-array-union", "--cohort-chunks", "3", "--source-membership"][..]] {
        let (plain, args) = arguments(flags);
        let baseline = plain.validate(&args).unwrap();
        assert!(plain.document.extraction.selected_rows().unwrap().is_none());
        for cap in ["1", "7", "166144"] {
            let mut selected_flags = flags.to_vec(); selected_flags.extend(["--max-selected-rows", cap]);
            let (selected, _) = arguments(&selected_flags);
            let admitted = selected.validate(&args).unwrap();
            assert_eq!(selected.document.extraction.selected_rows().unwrap().unwrap().max_rows_per_step.to_string(), cap);
            assert_eq!(admitted.memory, baseline.memory);
            assert_eq!(admitted.transport, baseline.transport);
            assert_eq!(admitted.max_run_mask_visits, baseline.max_run_mask_visits);
            assert_eq!(admitted.max_run_verification, baseline.max_run_verification);
            assert_eq!(admitted.document.mapping.mapping.max_model_work, baseline.document.mapping.mapping.max_model_work);
            assert_eq!(admitted.document.mapping.root_array_union, baseline.document.mapping.root_array_union);
            let budget = selected.document.extraction.budget(&args, admitted.common);
            let raw = if flags.contains(&"--source-membership") {
                " {\"type\":\"array\",\"maxItems\":8,\"items\":{\"type\":\"string\",\"maxLength\":64,\"x-fnlp-source\":\"verbatim\"}}\n"
            } else {
                " {\"type\":\"array\",\"maxItems\":8,\"items\":{\"type\":\"number\",\"const\":12345678901234567890123456789012345678}}\n"
            };
            assert_eq!(selected.load_defaults(raw.to_owned(), budget).unwrap().schema, raw);
        }
    }
}

#[test]
fn document_corpus_prices_the_same_exact_array_policy_once_without_renewing_native_work() {
    for cohorts in [None, Some(3)] {
        let (mut plain, args) = arguments(&[]);
        let (mut union, _) = arguments(&["--root-array-union", "--max-union-records", "32",
            "--max-union-origins", "128", "--max-union-input-bytes", "1048576",
            "--max-union-json-bytes", "65536", "--max-union-nodes", "4096",
            "--max-union-result-bytes", "262144"]);
        plain.document.cohort_chunks = cohorts; union.document.cohort_chunks = cohorts;
        let before = plain.validate(&args).unwrap();
        let after = union.validate(&args).unwrap();
        let policy = after.document.mapping.root_array_union.unwrap();
        assert_eq!(after.memory.document_workspace_bytes - before.memory.document_workspace_bytes,
            policy.requirements().unwrap().workspace_bytes);
        assert_eq!(after.memory.native, before.memory.native);
        assert_eq!(after.memory.input_bytes, before.memory.input_bytes);
        assert_eq!(after.memory.output_reservation_bytes, before.memory.output_reservation_bytes);
        assert_eq!(after.transport.max_work, before.transport.max_work);
        assert_eq!(after.max_run_mask_visits, before.max_run_mask_visits);
        assert_eq!(after.max_run_verification, before.max_run_verification);
        assert_eq!(after.document.mapping.mapping.chunks, before.document.mapping.mapping.chunks);
        let budget = union.document.extraction.budget(&args, after.common);
        let raw = r#"{"type":"array","maxItems":32,"items":{"type":"number"}}"#;
        assert_eq!(union.load_defaults(raw.to_owned(), budget).unwrap().schema, raw);
        assert!(union.load_defaults(r#"{"type":"number"}"#.to_owned(), budget).is_err());
    }
}

#[test]
fn whole_document_route_requires_owned_dispatch_and_an_explicit_schema() {
    assert_eq!(command(&[]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    assert!(parsed(&["--source-dir", "source", "--source-manifest", "manifest"]).is_err());
    let (options, args) = arguments(&[]);
    let limits = options.validate(&args).unwrap();
    assert_eq!(format!("{options:?}"), "ExtractionCorpusArgs");
    assert_eq!(limits.transport.max_document_bytes, args.max_input_bytes);
    assert_eq!(limits.document.mapping.mapping.chunks.max_input_bytes, args.max_input_bytes);
    assert_eq!(limits.memory.max_output_bytes as usize, limits.document.max_result_bytes);
    assert_eq!(limits.transport.max_output_line_bytes, limits.document.max_result_bytes + FRAME_ALLOWANCE as usize);
}

#[test]
fn fixed_schema_preserves_exact_numbers_and_checks_unbound_source_declarations() {
    let (options, args) = arguments(&[]);
    let limits = options.validate(&args).unwrap();
    let budget = options.document.extraction.budget(&args, limits.common);
    let schema = " {\n\"type\":\"number\", \"const\":12345678901234567890123456789012345678\n} ";
    let request = options.load_defaults(schema.to_owned(), budget).unwrap();
    assert_eq!(request.schema, schema);
    assert_eq!(request.grounding, ExtractionBatchGrounding::Structural);
    assert_eq!(request.budget, budget);

    let (source, args) = arguments(&["--source-membership"]);
    let limits = source.validate(&args).unwrap();
    let budget = source.document.extraction.budget(&args, limits.common);
    // No invented document can establish feasibility of this source enum.
    let verbatim = r#"{"type":"string","x-fnlp-source":"verbatim","enum":["café 上海"]}"#;
    let request = source.load_defaults(verbatim.to_owned(), budget).unwrap();
    assert_eq!(request.schema, verbatim);
    assert_eq!(request.grounding, ExtractionBatchGrounding::SourceMembership);
    assert!(options.load_defaults(verbatim.to_owned(), budget).is_err());
    assert!(source.load_defaults(r#"{"type":"string"}"#.to_owned(), budget).is_err());
    for malformed in [r#"{"type":"string","type":"string"}"#,
        r#"{"type":"string","pattern":".*"}"#, r#"{"$ref":"private"}"#,
        r#"{"type":"object","properties":{},"additionalProperties":true}"#, "null"] {
        assert!(options.load_defaults(malformed.to_owned(), budget).is_err(), "{malformed}");
        assert!(source.load_defaults(malformed.to_owned(), budget).is_err(), "{malformed}");
    }
}

#[test]
fn source_and_snapshot_keep_exact_task_work_and_complete_document_memory() {
    let (options, args) = arguments(&["--source-membership", "--cohort-chunks", "3"]);
    let source = options.validate(&args).unwrap();
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--schema", "private-schema.json", "--source-membership", "--cohort-chunks", "3"]).unwrap();
    let Int4Task::ExtractCorpus(restored) = snapshot.task else { panic!("snapshot corpus required"); };
    let loaded = restored.validate(&snapshot.args).unwrap();
    assert!(source.common.snapshot_pin.is_none()); assert!(loaded.common.snapshot_pin.is_some());
    assert_eq!(source.memory, loaded.memory);
    assert_eq!(source.transport, loaded.transport);
    assert_eq!(source.document.mapping.mapping.chunks, loaded.document.mapping.mapping.chunks);
    assert_eq!(source.document.mapping.mapping.max_model_work, loaded.document.mapping.mapping.max_model_work);
    assert_eq!(source.max_run_mask_visits, loaded.max_run_mask_visits);
    assert_eq!(source.max_run_verification, loaded.max_run_verification);
    assert_eq!(options.document.extraction.budget(&args, source.common),
        restored.document.extraction.budget(&snapshot.args, loaded.common));
    for flags in [vec!["--schema", "schema"], vec!["--snapshot", "snapshot", "--schema", "schema"],
        vec!["--snapshot", "snapshot", "--expected-model-sha256", &pin, "--schema", "schema", "--source-dir", "source"],
        vec!["--snapshot", "snapshot", "--expected-model-sha256", &pin, "--schema", "schema", "--save-snapshot", "new"]] {
        assert!(parsed(&flags).is_err());
    }
}

#[test]
fn record_capacity_and_cohorts_do_not_multiply_any_of_nine_run_counters() {
    let (mut options, args) = arguments(&["--max-requests", "10"]);
    let serial = options.validate(&args).unwrap();
    options.stream.max_requests = 1000;
    options.document.cohort_chunks = Some(3);
    let grouped = options.validate(&args).unwrap();
    assert_eq!(serial.document.mapping.mapping.max_model_work, grouped.document.mapping.mapping.max_model_work);
    assert_eq!(serial.transport.max_work, grouped.transport.max_work);
    assert_eq!(serial.max_run_mask_visits, grouped.max_run_mask_visits);
    assert_eq!(serial.max_run_verification, grouped.max_run_verification);
    assert_eq!(serial.document.mapping.mapping.max_mask_visits, grouped.document.mapping.mapping.max_mask_visits);
    assert_eq!(serial.document.mapping.verification, grouped.document.mapping.verification);
    assert_eq!(options.document.extraction.budget(&args, serial.common),
        options.document.extraction.budget(&args, grouped.common));
    assert_eq!(serial.memory.max_output_bytes, grouped.memory.max_output_bytes);
    assert_eq!(grouped.memory.native.kv_bytes, serial.memory.native.kv_bytes * 3);
    for (count, limits) in [(10, serial), (1000, grouped)] {
        assert_eq!(limits.transport.max_output_bytes + (count + 4) * FRAME_ALLOWANCE, limits.output_bytes);
    }
    // The operator can intentionally fund only one full verification attempt
    // while allowing transport to receive more records; execution stops on exhaustion.
    options.stream.max_run_mask_node_visits = serial.document.mapping.mapping.max_mask_visits;
    options.stream.max_run_extraction_fields = serial.document.mapping.verification.max_fields;
    options.stream.max_run_extraction_evidence_spans = serial.document.mapping.verification.max_matches;
    options.stream.max_run_extraction_scan_steps = serial.document.mapping.verification.max_scan_steps;
    options.validate(&args).unwrap();
}

#[test]
fn selected_corpus_policy_preserves_fixed_schema_native_memory_and_whole_run_caps() {
    for cohorts in [None, Some("3")] {
        let mut flags = vec!["--source-membership"];
        if let Some(width) = cohorts { flags.extend(["--cohort-chunks", width]); }
        let (dense, args) = arguments(&flags); let before = dense.validate(&args).unwrap();
        flags.extend(["--max-selected-rows", "4096"]);
        let (selected, args) = arguments(&flags); let after = selected.validate(&args).unwrap();
        assert_eq!(selected.document.extraction.selected_rows().unwrap().unwrap().max_rows_per_step, 4096);
        assert_eq!(before.memory, after.memory);
        assert_eq!(before.transport, after.transport);
        assert_eq!(before.max_run_mask_visits, after.max_run_mask_visits);
        assert_eq!(before.max_run_verification, after.max_run_verification);
        assert_eq!(before.document.mapping.mapping.max_model_work, after.document.mapping.mapping.max_model_work);
        let budget = selected.document.extraction.budget(&args, after.common);
        let schema = " {\"type\":\"string\",\"x-fnlp-source\":\"verbatim\",\"enum\":[\"café 上海\"]}\n";
        let request = selected.load_defaults(schema.to_owned(), budget).unwrap();
        assert_eq!(request.schema, schema);
        assert_eq!(request.grounding, ExtractionBatchGrounding::SourceMembership);
        assert_eq!(request.budget, budget);
    }
}

#[test]
fn selected_corpus_accepts_pinned_snapshots_and_refuses_invalid_caps_before_private_io() {
    let pin = "ab".repeat(32);
    for cap in ["1", "166144"] {
        let command = parsed(&["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
            "--schema", "private-schema.json", "--max-selected-rows", cap]).unwrap();
        let Int4Task::ExtractCorpus(options) = command.task else { panic!("extraction corpus required"); };
        assert!(options.validate(&command.args).unwrap().common.snapshot_pin.is_some());
    }
    for cap in ["0", "166145"] {
        let (options, args) = arguments(&["--max-selected-rows", cap]);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn all_complete_stream_owners_fit_the_declared_peak_before_io() {
    for chunks in [None, Some(1), Some(3), Some(64)] {
        let (mut options, mut args) = arguments(&["--max-result-bytes", "65536"]);
        options.document.cohort_chunks = chunks;
        let limits = options.validate(&args).unwrap();
        let m = limits.memory;
        let execution = m.native.kv_bytes + m.workspace_bytes + m.input_bytes
            + m.document_workspace_bytes + m.output_reservation_bytes;
        assert_eq!(limits.common.minimum_memory_bytes,
            grouped_memory_floor(&args, limits.common, execution).unwrap() + limits.io_bytes);
        assert_eq!(limits.cli_preparation_bytes, limits.common.preparation_bytes + limits.io_bytes);
        assert_eq!(m.native.kv_bytes, limits.common.native.kv_bytes * chunks.unwrap_or(1) as u64);
        assert!(limits.cli_preparation_bytes + limits.common.load.weight_payload_bytes
            + ALLOCATOR_BYTES + execution + CLEANUP_BYTES <= limits.common.minimum_memory_bytes);
        args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
        options.validate(&args).unwrap();
        args.memory_mib -= 1;
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn invalid_transport_map_native_mask_and_verification_limits_refuse_before_all_io() {
    for flags in [vec!["--schema", "-"], vec!["--schema", "https://private.example/schema"],
        vec!["--max-requests", "0"], vec!["--max-requests", "100001"],
        vec!["--max-total-input-bytes", "0"], vec!["--max-total-input-bytes", "17179869185"],
        vec!["--max-total-output-bytes", "1"], vec!["--max-total-output-bytes", "17179869185"],
        vec!["--max-line-bytes", "1048575"], vec!["--max-line-bytes", "4194305"],
        vec!["--max-chunks", "0"], vec!["--max-chunks", "257"], vec!["--max-chunk-bytes", "3"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-map-result-bytes", "0"], vec!["--max-forward-positions", "0"],
        vec!["--max-projected-logits", "0"], vec!["--max-attention-pairs", "0"],
        vec!["--max-dot-products", "0"], vec!["--max-multiply-accumulates", "0"],
        vec!["--max-run-mask-node-visits", "127999999999"],
        vec!["--max-run-extraction-fields", "4095"], vec!["--max-run-extraction-evidence-spans", "16383"],
        vec!["--max-run-extraction-scan-steps", "67108863"],
        vec!["--max-run-mask-node-visits", "1000000000000000001"],
        vec!["--max-run-extraction-fields", "1000000001"],
        vec!["--max-run-extraction-evidence-spans", "1000000001"],
        vec!["--max-run-extraction-scan-steps", "1000000000000000001"],
        vec!["--preparation-mib", "1"], vec!["--timeout-seconds", "0"], vec!["--max-checkpoints", "1"]] {
        let (options, args) = arguments(&flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
}

#[test]
fn single_context_controls_and_retained_job_options_cannot_leak_into_the_live_route() {
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-documents", "1"],
        vec!["--defaults", "other.json"],
        vec!["--min-new-tokens", "0"], vec!["--logprobs"], vec!["--seed", "ab"],
        vec!["--store-results"], vec!["--job-dir", "private-job"], vec!["--materialize"]] {
        let mut all = vec!["--source-dir", "source", "--source-manifest", "manifest", "--schema", "schema"];
        all.extend_from_slice(&flags);
        assert!(parsed(&all).is_err(), "{flags:?}");
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unsupported_profile_refuses_before_schema_or_population_io() {
    let (options, args) = arguments(&[]);
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
}
