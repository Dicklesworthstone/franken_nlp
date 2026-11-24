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
    assert!(help.contains("By default, required fields must be satisfiable per chunk"));
    assert!(help.contains("--reconcile-object"));
    assert!(help.contains("conflicts or missing required fields refuse the whole document"));
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

#[test]
fn chunked_extraction_composes_all_schedules_without_expanding_semantic_authority() {
    let baseline = chunked(&["--max-requests", "3"]);
    let (_, original_limits, original) = baseline.validate().unwrap();
    for width in ["1", "4"] {
        for rows in [None, Some("1"), Some("7")] {
            let mut flags = vec!["--max-requests", "3", "--compute-threads", width];
            if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
            let mut selected = chunked(&flags); selected.host.memory_mib = 16384;
            let (args, limits, envelope) = selected.validate().unwrap();
            assert_eq!(limits.compute, args.compute.admit_prefill(selected.prefill.limits().unwrap()).unwrap());
            assert_eq!(envelope.native.max_model_work, original.native.max_model_work);
            assert_eq!(envelope.native.masks.max_visits_per_run, original.native.masks.max_visits_per_run);
            assert_eq!(selected.host.task_budget(limits), baseline.host.task_budget(original_limits));
            assert_eq!(envelope.transport.max_output_line_bytes, original.transport.max_output_line_bytes);
        }
    }
}

#[test]
fn chunked_schedule_funds_one_complete_frontier_without_accumulating_the_corpus() {
    let mut selected = chunked(&["--compute-threads", "4", "--prefill-rows", "7", "--max-requests", "3"]);
    selected.host.memory_mib = 16384;
    let (_, mut limits, envelope) = selected.validate().unwrap();
    let prefill = selected.prefill.limits().unwrap();
    let peak = selected.schedule_peak_bytes(limits, envelope, prefill).unwrap();
    limits.memory_bytes = peak;
    selected.check_scheduling_memory(limits, envelope, prefill).unwrap();
    limits.memory_bytes -= 1;
    assert_eq!(selected.check_scheduling_memory(limits, envelope, prefill), Err(CandidateError::Arguments));
    let (retained, tokens) = selected.schedule_document_reserve().unwrap();
    assert!(retained > 4 * MAX_DOCUMENT_RESULT as u64);
    assert_eq!(tokens, 64 * selected.host.max_new_tokens as u64);
    selected.max_requests = 6;
    let (_, larger_limits, larger) = selected.validate().unwrap();
    assert_eq!(larger.native.max_model_work, scale_work(envelope.native.max_model_work, 2).unwrap());
    assert_eq!(selected.schedule_peak_bytes(larger_limits, larger, prefill).unwrap(), peak,
        "another record may spend work, but cannot retain another complete result");
    selected.document.max_chunks = Some(65);
    let (larger_frontier, larger_tokens) = selected.schedule_document_reserve().unwrap();
    assert_eq!(larger_tokens - tokens, selected.host.max_new_tokens as u64);
    assert_eq!(larger_frontier - retained,
        4 * selected.host.max_result_bytes as u64 + 8 * selected.host.max_new_tokens as u64);
}

#[test]
fn compact_chunked_encoding_has_one_stream_allowance_and_preserves_all_native_budgets() {
    use crate::tokenizer::untrusted::{SafeBpeWorkLimits, UntrustedEncoding};
    let baseline = chunked(&["--max-requests", "3"]);
    let (_, original_limits, original) = baseline.validate().unwrap();
    for width in ["1", "4"] {
        for rows in [None, Some("7")] {
            let mut flags = vec!["--max-requests", "3", "--compute-threads", width,
                "--input-encoding", "safe-bpe-v1", "--max-total-tokenizer-input-bytes", "3000000",
                "--max-total-tokenizer-output-tokens", "4000000",
                "--max-total-tokenizer-pair-checks", "5000000", "--max-total-tokenizer-pair-bytes", "6000000"];
            if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
            let mut selected = chunked(&flags); selected.host.memory_mib = 16384;
            let (_, mut limits, envelope) = selected.validate().unwrap();
            let UntrustedEncoding::SafeBpeV1(call) = selected.encoding_policy(limits).unwrap()
                else { panic!("compact document policy lost") };
            assert_eq!(call.max_input_bytes, selected.host.max_input_bytes + crate::candidate_cli::extract::SCHEMA_BYTES);
            assert_eq!(call.max_output_tokens, call.max_input_bytes);
            assert!(call.max_output_tokens > limits.max_prompt_tokens);
            assert_eq!(selected.document_encoding_work_limits().unwrap(), Some(SafeBpeWorkLimits {
                max_input_bytes: 3_000_000, max_output_tokens: 4_000_000,
                max_pair_checks: 5_000_000, max_pair_bytes: 6_000_000 }));
            assert_eq!(selected.encoding_workspace_bytes(limits).unwrap(), call.requirements().unwrap().workspace_bytes + 1024);
            assert_eq!(envelope.native.max_model_work, original.native.max_model_work);
            assert_eq!(envelope.native.masks.max_visits_per_run, original.native.masks.max_visits_per_run);
            assert_eq!(selected.host.task_budget(limits), baseline.host.task_budget(original_limits));
            let prefill = selected.prefill.limits().unwrap();
            limits.memory_bytes = selected.schedule_peak_bytes(limits, envelope, prefill).unwrap();
            selected.check_scheduling_memory(limits, envelope, prefill).unwrap();
            limits.memory_bytes -= 1;
            assert_eq!(selected.check_scheduling_memory(limits, envelope, prefill), Err(CandidateError::Arguments));
        }
    }
    assert_eq!(baseline.document_encoding_work_limits().unwrap(), None);
    assert_eq!(baseline.encoding_workspace_bytes(original_limits).unwrap(), 0);
}

#[test]
fn document_tokenizer_totals_reject_wrong_routes_and_invalid_caps_before_private_io() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        assert_eq!(chunked(&[flag, "1"]).execute_owned(NoRead, NoWrite), Err(CandidateError::Arguments));
        for task in ["extract", "ner", "answer"] {
            assert_eq!(command(task, &["--input-encoding", "safe-bpe-v1", flag, "1"])
                .execute_owned(NoRead, NoWrite), Err(CandidateError::Arguments));
        }
        for value in ["0", "18446744073709551615"] {
            assert_eq!(chunked(&["--input-encoding", "safe-bpe-v1", flag, value])
                .execute_owned(NoRead, NoWrite), Err(CandidateError::Arguments));
        }
    }
}


fn reconciled(flags: &[&str]) -> BatchCommand {
    let mut options = vec!["--reconcile-object"];
    options.extend_from_slice(flags);
    let mut c = chunked(&options); c.host.memory_mib = 32768; c
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn object_reconciliation_requires_complete_document_extraction_before_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    for task in ["extract", "ner", "keyphrases", "summarize", "answer"] {
        assert_eq!(command(task, &["--reconcile-object"]).execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    }
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        assert_eq!(command(task, &["--chunked", "--schema", "s", "--reconcile-object"])
            .execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let mut missing = reconciled(&[]); missing.schema = None;
    assert_eq!(missing.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    let mut defaults = reconciled(&[]); defaults.defaults = Some(PathBuf::from("private-defaults"));
    assert_eq!(defaults.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    for flag in ["--max-object-properties", "--max-object-origins", "--max-object-input-bytes",
        "--max-object-json-bytes", "--max-object-nodes", "--max-object-result-bytes"] {
        assert!(root().try_get_matches_from(["candidate", "batch", "--task", "extract",
            "--model", "m", "--chunked", "--schema", "s", flag, "32"]).is_err());
        assert_eq!(reconciled(&[flag, "0"]).execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    }
    let mut orphan = reconciled(&["--max-object-properties", "8"]);
    orphan.document.reconciliation.reconcile_object = false;
    assert!(orphan.document.reconciliation.requested());
    assert_eq!(orphan.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn object_bounds_share_the_existing_single_document_contract_and_leave_defaults_unchanged() {
    let ordinary = chunked(&[]);
    assert_eq!(ordinary.document_object_limits().unwrap(), None);
    assert_eq!(ordinary.document_object_workspace_bytes().unwrap(), 0);
    assert_eq!(ReconciliationArgs::default().limits(4096, 1024).unwrap(), None);
    let selected = reconciled(&[]);
    assert_eq!(selected.document_object_limits().unwrap(), Some(RootObjectLimits {
        max_properties: 4096, max_origins: 16384, max_input_bytes: MAX_DOCUMENT_RESULT,
        max_json_bytes: 16 * 1024 * 1024, max_nodes: 262144, max_result_bytes: 16 * 1024 * 1024 }));
    for (flag, value) in [("--max-object-properties", "1000001"), ("--max-object-origins", "1"),
        ("--max-object-input-bytes", "67108865"), ("--max-object-json-bytes", "16777217"),
        ("--max-object-nodes", "1000001"), ("--max-object-result-bytes", "16777217")] {
        assert!(reconciled(&[flag, value]).validate().is_err(), "{flag}={value}");
    }
    let selected = reconciled(&["--max-object-properties", "8", "--max-object-origins", "16",
        "--max-object-input-bytes", "4096", "--max-object-json-bytes", "1024",
        "--max-object-nodes", "512", "--max-object-result-bytes", "4096"]);
    selected.validate().unwrap();
    let object = selected.document_object_limits().unwrap().unwrap();
    assert_eq!(object.max_properties, 8); assert_eq!(object.max_origins, 16);
    assert_eq!(object.max_input_bytes, 4096); assert_eq!(object.max_json_bytes, 1024);
    assert_eq!(object.max_nodes, 512); assert_eq!(object.max_result_bytes, 4096);
    assert_eq!(selected.schedule_document_reserve().unwrap().0 - ordinary.schedule_document_reserve().unwrap().0,
        selected.document_object_workspace_bytes().unwrap());
}

#[test]
fn final_required_object_may_exceed_each_native_chunk_output_cap() {
    let c = reconciled(&["--max-result-bytes", "2048"]);
    let (_, limits, _) = c.validate().unwrap();
    let budget = c.host.task_budget(limits);
    let compiler = crate::candidate_cli::extract::compiler(&c.host);
    let object = c.document_object_limits().unwrap().unwrap();
    let value = "x".repeat(896);
    let schema = format!(r#"{{"type":"object","additionalProperties":false,"properties":{{"a":{{"type":"string","const":"{value}"}},"b":{{"type":"string","const":"{value}"}},"c":{{"type":"string","const":"{value}"}}}},"required":["a","b","c"]}}"#);
    assert!(crate::grammar::runtime::JsonValidationSchema::compile(&schema, compiler).is_err());
    assert!(c.load_extract_defaults(Some(schema.clone()), None, budget).is_err());
    let request = reconciled_request(schema.clone(), false, budget, compiler, object, &mut Continue).unwrap();
    assert_eq!(request.schema, schema); assert_eq!(request.budget, budget);
    assert_eq!(request.budget.max_output_bytes, 2048);
    assert_eq!(request.grounding, ExtractionBatchGrounding::Structural);
}

#[test]
fn reconciled_request_retains_original_exact_numbers_and_rejects_wrong_shape_or_grounding() {
    let c = reconciled(&[]); let (_, limits, _) = c.validate().unwrap();
    let budget = c.host.task_budget(limits);
    let compiler = crate::candidate_cli::extract::compiler(&c.host);
    let object = c.document_object_limits().unwrap().unwrap();
    let schema = " {\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"number\",\"const\":9007199254740993}},\"required\":[\"n\"],\"additionalProperties\":false}\n";
    let request = reconciled_request(schema.to_owned(), false, budget, compiler, object, &mut Continue).unwrap();
    assert_eq!(request.schema.as_bytes(), schema.as_bytes());
    let annotated = r#"{"type":"object","properties":{"name":{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}},"required":["name"],"additionalProperties":false}"#;
    assert!(reconciled_request(annotated.to_owned(), false, budget, compiler, object, &mut Continue).is_err());
    assert!(reconciled_request(schema.to_owned(), true, budget, compiler, object, &mut Continue).is_err());
    let grounded = reconciled_request(annotated.to_owned(), true, budget, compiler, object, &mut Continue).unwrap();
    assert_eq!(grounded.schema, annotated);
    assert_eq!(grounded.grounding, ExtractionBatchGrounding::SourceMembership);
    for invalid in [r#"{"type":"boolean"}"#, r#"{"type":"array","items":{"type":"boolean"},"maxItems":2}"#,
        r#"{"type":"object","type":"object","properties":{},"additionalProperties":false}"#] {
        assert!(reconciled_request(invalid.to_owned(), false, budget, compiler, object, &mut Continue).is_err());
    }
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    assert!(matches!(reconciled_request(schema.to_owned(), false, budget, compiler, object, &mut Stop),
        Err(CandidateError::Timeout)));
}

#[test]
fn object_corpus_admits_the_whole_peak_for_serial_prefill_team_and_compact_routes() {
    let baseline = chunked(&["--max-requests", "3"]);
    let (_, original_limits, original) = baseline.validate().unwrap();
    for encoding in ["byte-fallback", "safe-bpe-v1"] {
        for schedule in [&[][..], &["--prefill-rows", "7"][..],
            &["--compute-threads", "4"][..], &["--compute-threads", "4", "--prefill-rows", "7"][..]] {
            let mut flags = vec!["--max-requests", "3", "--input-encoding", encoding];
            flags.extend_from_slice(schedule);
            let selected = reconciled(&flags);
            let (_, mut limits, envelope) = selected.validate().unwrap();
            assert_eq!(envelope.native.max_model_work, original.native.max_model_work);
            assert_eq!(envelope.native.masks.max_visits_per_run, original.native.masks.max_visits_per_run);
            assert_eq!(selected.host.task_budget(limits), baseline.host.task_budget(original_limits));
            assert_eq!(envelope.transport.max_output_line_bytes, original.transport.max_output_line_bytes);
            let prefill = selected.prefill.limits().unwrap();
            limits.memory_bytes = selected.schedule_peak_bytes(limits, envelope, prefill).unwrap();
            selected.check_scheduling_memory(limits, envelope, prefill).unwrap();
            limits.memory_bytes -= 1;
            assert_eq!(selected.check_scheduling_memory(limits, envelope, prefill), Err(CandidateError::Arguments));
        }
    }
}
