//! Parser/configuration/resource mechanics only; no native-model success claims.
use super::*;
use crate::{batch::extract::ExtractionBatchGrounding, candidate_cli::definition as root};

fn command(task: &str, flags: &[&str]) -> BatchCommand {
    let mut argv = vec!["candidate", "batch", "--task", task, "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = root().try_get_matches_from(argv).unwrap();
    match CandidateCommand::from_matches(&matches).unwrap() {
        CandidateCommand::Batch(command) => command,
        _ => panic!("wrong corpus route"),
    }
}
fn chunked(flags: &[&str]) -> BatchCommand {
    let mut extra = vec!["--chunked", "--schema", "schema.json", "--preparation-mib", "1024"];
    extra.extend_from_slice(flags); command("extract", &extra)
}
fn configuration(command: &BatchCommand, schema: &str) -> ExtractionDocumentBatchConfig {
    let (_, limits, envelope) = command.validate().unwrap();
    let request = command.load_extract_defaults(Some(schema.to_owned()), None, command.host.task_budget(limits)).unwrap().unwrap();
    command.document_configuration(request, envelope).unwrap()
}
#[test]
fn the_single_context_route_and_its_original_work_and_framing_are_unchanged() {
    let command = command("extract", &["--max-requests", "3"]);
    let (_, _, envelope) = command.validate().unwrap();
    assert!(!command.document.chunked); assert!(command.document_settings().unwrap().is_none());
    let one = constrained_int8::planned_work(command.host.context_tokens - command.host.max_new_tokens,
        command.host.max_new_tokens).unwrap();
    assert_eq!(envelope.native.max_model_work, scale_work(one, 3).unwrap());
    assert_eq!(envelope.native.masks.max_visits_per_item, command.host.max_mask_node_visits);
    assert_eq!(envelope.native.masks.max_visits_per_run, 3 * command.host.max_mask_node_visits);
    assert_eq!(envelope.transport.max_output_line_bytes, command.host.max_result_bytes + 4096);
    assert!(command.document_reduction_bytes().is_err());
}
#[test]
fn chunked_mode_requires_a_fixed_extract_schema_and_refuses_ignored_switches() {
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        assert!(command(task, &["--chunked", "--schema", "s"]).validate().is_err());
    }
    for flags in [vec!["--chunked"], vec!["--chunked", "--schema", "s", "--defaults", "d"],
        vec!["--max-chunks", "2"], vec!["--max-chunk-bytes", "16"], vec!["--max-document-result-bytes", "16777216"],
        vec!["--reduction-reserve-mib", "64"], vec!["--max-extraction-fields", "1"],
        vec!["--max-extraction-evidence-spans", "1"], vec!["--max-extraction-scan-steps", "1"]] {
        let mut argv = vec!["candidate", "batch", "--task", "extract", "--model", "m", "--memory-mib", "8192"];
        argv.extend(flags); assert!(root().try_get_matches_from(argv).is_err());
    }
    let mut c = chunked(&[]); c.schema = None;
    assert!(c.validate().is_err());
    let mut c = chunked(&[]); c.defaults = Some(PathBuf::from("d"));
    assert!(c.validate().is_err());
    let mut c = command("extract", &[]); c.document.max_chunks = Some(2);
    assert!(c.validate().is_err());
}
#[test]
fn every_work_axis_scales_by_independent_chunks_and_requests_not_a_giant_context() {
    for chunks in ["1", "3", "64"] {
        let c = chunked(&["--max-chunks", chunks, "--max-requests", "3"]);
        let (_, _, envelope) = c.validate().unwrap();
        let n: u64 = chunks.parse().unwrap();
        let one = constrained_int8::planned_work(c.host.context_tokens - c.host.max_new_tokens, c.host.max_new_tokens).unwrap();
        let config = configuration(&c, r#"{"type":"boolean"}"#);
        assert_eq!(config.limits.mapping.max_model_work, scale_work(one, n).unwrap());
        assert_eq!(config.max_model_work, scale_work(one, n * 3).unwrap());
        assert_eq!(envelope.native.max_model_work, config.max_model_work);
        assert_eq!(envelope.transport.max_work.forward_positions, config.max_model_work.forward_positions);
        assert_eq!(envelope.transport.max_work.projected_logits, config.max_model_work.projected_logits);
        assert_eq!(config.limits.mapping.mask_visits_per_chunk, c.host.max_mask_node_visits);
        assert_eq!(config.limits.mapping.max_mask_visits, c.host.max_mask_node_visits * n);
        assert_eq!(config.max_mask_visits, c.host.max_mask_node_visits * n * 3);
        assert_eq!(envelope.native.masks.max_visits_per_item, config.limits.mapping.max_mask_visits);
    }
}
#[test]
fn complete_document_results_keep_one_event_and_one_whole_source_limit() {
    let c = chunked(&["--max-chunks", "7", "--max-requests", "3", "--max-chunk-bytes", "16"]);
    let (_, _, envelope) = c.validate().unwrap();
    let config = configuration(&c, r#"{"type":"boolean"}"#);
    assert_eq!(config.limits.mapping.chunks.max_input_bytes, c.host.max_input_bytes);
    assert_eq!(config.limits.mapping.chunks.max_chunk_bytes, 16);
    assert_eq!(envelope.transport.max_document_bytes, c.host.max_input_bytes);
    assert_eq!(envelope.transport.max_requests, 3);
    assert_eq!(envelope.transport.max_output_line_bytes, 16 * 1024 * 1024 + 4096);
    assert_eq!(config.limits.mapping.reduction.max_result_bytes, 16 * 1024 * 1024);
    assert_eq!(envelope.transport.max_output_bytes + (3 + 4) * FRAME_ALLOWANCE, envelope.output_bytes);
    assert_eq!(envelope.io_bytes, envelope.transport.max_output_line_bytes as u64 + FRAME_ALLOWANCE + IO_BUFFER_BYTES as u64 * 2);
}
#[test]
fn verification_is_whole_document_and_corpus_bounded_not_renewed_for_each_chunk() {
    for chunks in ["1", "64"] {
        let c = chunked(&["--max-chunks", chunks, "--max-requests", "3", "--max-extraction-fields", "17",
            "--max-extraction-evidence-spans", "19", "--max-extraction-scan-steps", "23"]);
        let config = configuration(&c, r#"{"type":"boolean"}"#);
        assert_eq!(config.limits.verification, GroundingBudget { max_fields: 17, max_matches: 19, max_scan_steps: 23 });
        assert_eq!(config.max_verification, GroundingBudget { max_fields: 51, max_matches: 57, max_scan_steps: 69 });
    }
}
#[test]
fn exact_schema_bytes_and_explicit_grounding_survive_the_document_configuration() {
    let c = chunked(&[]);
    let schema = " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678}\n";
    let config = configuration(&c, schema);
    assert_eq!(config.request.schema.as_bytes(), schema.as_bytes());
    assert_eq!(config.request.grounding, ExtractionBatchGrounding::Structural);
    let c = chunked(&["--source-membership"]);
    let schema = r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#;
    let config = configuration(&c, schema);
    assert_eq!(config.request.schema, schema);
    assert_eq!(config.request.grounding, ExtractionBatchGrounding::SourceMembership);
    // No fabricated empty source was used to reject a satisfiable future item.
    let (_, limits, _) = c.validate().unwrap();
    assert!(c.load_extract_defaults(Some(r#"{"type":"string","type":"number"}"#.to_owned()), None,
        c.host.task_budget(limits)).is_err());
}
#[test]
fn all_document_geometry_and_verification_switches_have_finite_domains() {
    for (flag, value) in [("--max-chunks", "0"), ("--max-chunks", "257"), ("--max-chunk-bytes", "3"),
        ("--max-document-result-bytes", "0"), ("--max-document-result-bytes", "67108865"),
        ("--reduction-reserve-mib", "0"), ("--reduction-reserve-mib", "18446744073709551615"),
        ("--max-extraction-fields", "0"), ("--max-extraction-fields", "1000001"),
        ("--max-extraction-evidence-spans", "0"), ("--max-extraction-evidence-spans", "1000001"),
        ("--max-extraction-scan-steps", "0"), ("--max-extraction-scan-steps", "1000000000001")] {
        assert!(chunked(&[flag, value]).validate().is_err(), "{flag}={value}");
    }
    chunked(&["--max-extraction-fields", "1000000", "--max-extraction-evidence-spans", "1000000",
        "--max-extraction-scan-steps", "1000000000000"]).validate().unwrap();
}
#[test]
fn preparation_and_every_scaled_native_axis_refuse_underfunding_or_overflow_before_io() {
    let c = chunked(&["--max-chunks", "2"]); let (_, limits, original) = c.validate().unwrap();
    let mut small = limits; small.preparation_bytes = 1;
    assert!(c.document_envelope(small, original).is_err());
    let mut small = limits; small.memory_bytes = 1;
    assert!(c.document_envelope(small, original).is_err());
    for axis in 0..7 {
        let mut bad = original;
        match axis { 0 => bad.native.max_model_work.forward_positions = u64::MAX,
            1 => bad.native.max_model_work.projected_logits = u64::MAX,
            2 => bad.native.max_model_work.attention_pairs = u64::MAX,
            3 => bad.native.max_model_work.projections.dot_products = u64::MAX,
            4 => bad.native.max_model_work.projections.multiply_accumulates = u64::MAX,
            5 => bad.native.masks.max_visits_per_item = u64::MAX,
            _ => bad.native.masks.max_visits_per_run = u64::MAX }
        assert!(c.document_envelope(limits, bad).is_err(), "axis {axis}");
    }
}
#[test]
fn help_and_completion_keep_independent_values_and_nonzero_document_failure_exit() {
    let help = super::super::definition().render_long_help().to_string();
    assert!(help.contains("--task extract --chunked --schema FILE"));
    assert!(help.contains("Required fields must be satisfiable per chunk"));
    assert!(help.contains("No partial document success"));
    assert_eq!(completed(BatchSummary { failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn a_disabled_runtime_rejects_chunked_extraction_before_schema_model_or_corpus_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    assert_eq!(chunked(&[]).execute_owned(NoRead, NoWrite), Err(CandidateError::Unavailable));
}
