//! Source-only parser, admission and private-boundary regressions.
use super::*;

const PIN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
pub(super) fn command(extra: &[&str], snapshot: bool) -> (Int4Args, BatchArgs) {
    let mut words = vec!["candidate", "source-int4", "extract-semantic-batch",
        "--memory-mib", "65536", "--schema", "private-schema", "--semantic-options", "private-policy"];
    if snapshot { words.extend(["--snapshot", "private-snapshot", "--expected-model-sha256", PIN]); }
    else { words.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest"]); }
    words.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    let Int4Task::SemanticBatch(batch) = command.task else { panic!("wrong semantic batch route"); };
    (command.args, batch)
}
pub(super) fn options() -> Int4SemanticOptions { super::super::tests::fixture().3 }
pub(super) struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused command read private input"); }
}
pub(super) struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused command published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused command flushed output"); }
}

#[test]
fn source_snapshot_save_and_compact_team_routes_keep_fixed_semantic_policy() {
    for snapshot in [false, true] {
        for flags in [&[][..], &["--compute-threads", "4"][..],
            &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..]] {
            let mut selected = flags.to_vec();
            selected.push("population.ndjson");
            let (args, batch) = command(&selected, snapshot);
            let initial = batch.validate(&args).unwrap();
            let limits = batch.policy_limits(&args, initial, &options()).unwrap();
            assert_eq!(args.input, PathBuf::from("population.ndjson"));
            assert_eq!(initial.snapshot_pin.is_some(), snapshot);
            assert_eq!(limits.common.compute.threads, if flags.is_empty() { 1 } else { 4 });
            assert!(args.input_encoding.is_default(), "semantic tokenizer policy has its own aggregate ledger");
            assert_eq!(limits.task.max_output_bytes, 1 << 20);
            assert_eq!(limits.common.result_bytes, options().limits.max_output_bytes as usize);
            assert!(limits.common.prefill.is_none());
        }
    }
    let (args, batch) = command(&["--save-snapshot", "private-new-snapshot"], false);
    batch.policy_limits(&args, batch.validate(&args).unwrap(), &options()).unwrap();
    assert_eq!(args.save_snapshot, Some(PathBuf::from("private-new-snapshot")));
}

#[test]
fn borrowed_execution_cannot_enter_the_owned_population_route() {
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "extract-semantic-batch", "--source-dir", "private-source",
        "--source-manifest", "private-manifest", "--memory-mib", "65536",
        "--schema", "private-schema", "--semantic-options", "private-policy",
    ]).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
}

#[test]
fn request_count_never_renews_or_multiplies_complete_document_work() {
    let (args, mut batch) = command(&["--max-requests", "2"], false);
    let options = options();
    let initial = batch.validate(&args).unwrap();
    let first = batch.policy_limits(&args, initial, &options).unwrap();
    batch.stream.max_requests = 200;
    let larger = batch.policy_limits(&args, initial, &options).unwrap();
    assert_eq!(first.native.max_model_work, options.max_model_work);
    assert_eq!(first.native.max_model_work, larger.native.max_model_work);
    assert_eq!(first.native.masks.max_visits_per_run, batch.semantic.extraction.mask_visits());
    assert_eq!(first.native.masks.max_visits_per_run, larger.native.masks.max_visits_per_run);
    assert_eq!(first.transport.max_work, larger.transport.max_work);
    assert_eq!(first.task, larger.task);
    assert_eq!(first.io_bytes, larger.io_bytes);
    assert_eq!(first.transport.max_output_bytes - larger.transport.max_output_bytes, 198 * FRAME_ALLOWANCE);
}

#[test]
fn every_total_native_and_mask_axis_must_cover_one_complete_attempt() {
    let (args, base) = command(&[], false);
    let options = options();
    let initial = base.validate(&args).unwrap();
    for axis in 0..6 {
        let mut batch = base.clone();
        match axis {
            0 => batch.stream.max_total_forward_positions = Some(options.max_model_work.forward_positions - 1),
            1 => batch.stream.max_total_projected_logits = Some(options.max_model_work.projected_logits - 1),
            2 => batch.stream.max_total_attention_pairs = Some(options.max_model_work.attention_pairs - 1),
            3 => batch.stream.max_total_dot_products = Some(options.max_model_work.projections.dot_products - 1),
            4 => batch.stream.max_total_multiply_accumulates = Some(options.max_model_work.projections.multiply_accumulates - 1),
            _ => batch.stream.max_total_mask_node_visits = Some(batch.semantic.extraction.mask_visits() - 1),
        }
        assert!(batch.policy_limits(&args, initial, &options).is_err());
    }
    let mut doubled = base.clone();
    let twice = options.max_model_work.checked_add(options.max_model_work).unwrap();
    doubled.stream.max_total_forward_positions = Some(twice.forward_positions);
    doubled.stream.max_total_projected_logits = Some(twice.projected_logits);
    doubled.stream.max_total_attention_pairs = Some(twice.attention_pairs);
    doubled.stream.max_total_dot_products = Some(twice.projections.dot_products);
    doubled.stream.max_total_multiply_accumulates = Some(twice.projections.multiply_accumulates);
    doubled.stream.max_total_mask_node_visits = Some(2 * base.semantic.extraction.mask_visits());
    let limits = doubled.policy_limits(&args, initial, &options).unwrap();
    assert_eq!(limits.native.max_model_work, twice);
    assert_eq!(limits.native.masks.max_visits_per_item, base.semantic.extraction.mask_visits());
    assert_eq!(limits.native.masks.max_visits_per_run, 2 * limits.native.masks.max_visits_per_item);
}

#[test]
fn combined_result_transport_uses_the_review_cap_instead_of_extraction_bytes() {
    let (args, mut batch) = command(&[], false);
    let options = options();
    let initial = batch.validate(&args).unwrap();
    let limits = batch.policy_limits(&args, initial, &options).unwrap();
    assert_eq!(limits.task.max_output_bytes, 1 << 20);
    assert_eq!(limits.transport.max_output_line_bytes, options.limits.max_output_bytes as usize + 4096);
    assert!(limits.transport.max_output_line_bytes as u64 > limits.task.max_output_bytes);
    assert_eq!(limits.output_bytes, limits.transport.max_output_bytes
        + (batch.stream.max_requests + 4) * FRAME_ALLOWANCE);
    batch.stream.max_total_output_bytes = FRAME_ALLOWANCE;
    assert!(batch.policy_limits(&args, initial, &options).is_err());
}

#[test]
fn fixed_declaration_keeps_exact_schema_and_checks_claim_rules_without_a_dummy_source() {
    let schema = r#"{"type":"object","properties":{"amount":{"type":"integer","const":9007199254740993}},"required":["amount"],"additionalProperties":false}"#;
    let (args, batch) = command(&[], false);
    let options = options();
    let limits = batch.policy_limits(&args, batch.validate(&args).unwrap(), &options).unwrap();
    let defaults = batch.load_defaults(schema.to_owned(), limits.task, &options).unwrap();
    assert_eq!(defaults.schema, schema);
    assert_eq!(defaults.budget, limits.task);
    let mut wrong = options.clone();
    wrong.specification.claims[0].path = vec![
        crate::tasks::extract::semantic::ClaimPathStep::Property("absent".to_owned())];
    assert!(batch.load_defaults(schema.to_owned(), limits.task, &wrong).is_err());

    let (args, source_bound) = command(&["--source-membership"], false);
    let limits = source_bound.policy_limits(&args, source_bound.validate(&args).unwrap(), &options).unwrap();
    assert!(source_bound.load_defaults(schema.to_owned(), limits.task, &options).is_err());
    let schema = r#"{"type":"object","properties":{"amount":{"type":"integer"},"excerpt":{"type":"string","enum":["上海"],"maxLength":2,"x-fnlp-source":"verbatim"}},"required":["amount","excerpt"],"additionalProperties":false}"#;
    let defaults = source_bound.load_defaults(schema.to_owned(), limits.task, &options).unwrap();
    assert_eq!(defaults.schema, schema);
    assert_eq!(defaults.grounding, crate::batch::extract::ExtractionBatchGrounding::SourceMembership);
    assert!(batch.load_defaults(schema.to_owned(), limits.task, &options).is_err());
}

#[test]
fn invalid_stream_paths_and_limits_refuse_before_options_population_or_output_io() {
    for flags in [
        &["--max-requests", "0"][..], &["--max-requests", "100001"][..],
        &["--max-line-bytes", "1"][..], &["--max-line-bytes", "4194305"][..],
        &["--max-total-input-bytes", "0"][..], &["--max-total-output-bytes", "0"][..],
        &["--max-total-input-bytes", "18446744073709551615"][..],
        &["--max-total-output-bytes", "18446744073709551615"][..],
        &["--max-total-forward-positions", "0"][..], &["--max-total-projected-logits", "0"][..],
        &["--max-total-attention-pairs", "0"][..], &["--max-total-dot-products", "0"][..],
        &["--max-total-multiply-accumulates", "0"][..], &["--max-total-mask-node-visits", "0"][..],
        &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
        &["--max-mask-node-visits", "1"][..], &["--preparation-mib", "1"][..],
    ] {
        let (args, batch) = command(flags, false);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage, "{flags:?}");
    }
    for path in ["", "-", "https://example.invalid/private-policy"] {
        let (args, mut batch) = command(&[], false);
        batch.semantic.semantic_options = PathBuf::from(path);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn generation_projection_and_cohort_controls_are_never_silently_accepted() {
    for flags in [
        &["--seed", PIN][..], &["--logprobs"][..], &["--prefill-rows", "1"][..],
        &["--max-selected-rows", "32"][..], &["--cohort-documents", "2"][..],
        &["--defaults", "private-defaults"][..], &["--think"][..],
    ] {
        let mut words = vec!["candidate", "source-int4", "extract-semantic-batch",
            "--source-dir", "private-source", "--source-manifest", "private-manifest",
            "--schema", "private-schema", "--semantic-options", "private-policy", "--memory-mib", "65536"];
        words.extend_from_slice(flags);
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(words) {
            assert!(CandidateCommand::from_matches(&matches).is_err(), "{flags:?}");
        }
    }
}

#[test]
fn stderr_error_report_preserves_every_original_cancellation_without_private_fields() {
    use crate::native_engine::decode::DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let mut stderr = Vec::new();
        report_failure(Failure { exit: ErrorCode::Cancelled, code: "fixture_cancelled",
            cancellation: Some(cause) }, &mut stderr);
        let value: serde_json::Value = serde_json::from_slice(&stderr).unwrap();
        assert_eq!(value["protocol"], "fnlp-candidate-int4-semantic-extraction-batch-v1");
        assert_eq!(value["event"], "error");
        assert_eq!(value["code"], "fixture_cancelled");
        assert_eq!(value["cancellation"], serde_json::to_value(cause).unwrap());
        assert_eq!(value["exit_code"], serde_json::to_value(ErrorCode::Cancelled).unwrap());
        assert_eq!(value["earlier_complete_records_may_exist"], true);
        assert!(stderr.len() < 4096);
        for private in ["document", "schema", "options", "source", "key", "private-policy"] {
            assert!(value.get(private).is_none());
        }
    }
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::ErrorKind::BrokenPipe.into()) }
        fn flush(&mut self) -> io::Result<()> { panic!("failed diagnostic write must stop"); }
    }
    assert_eq!(report_failure(Failure::usage("refused"), &mut Broken), ErrorCode::Generic.as_process_exit());
}

#[test]
fn any_failed_record_is_nonzero_even_after_protocol_completion() {
    completed(BatchSummary::default()).unwrap();
    let failure = completed(BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() }).unwrap_err();
    assert_eq!(failure.code, "semantic_batch_items_failed");
    assert_ne!(failure.exit, ErrorCode::Ok);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_runtime_refuses_without_reading_private_configuration_or_population() {
    let (args, batch) = command(&[], false);
    assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().code, "semantic_batch_profile_unavailable");
}
