//! Parser and complete population-budget regression definitions.
use super::*;

pub(super) fn fixture() -> (CorpusArgs, Int4Args, CorpusLimits, Int4SemanticOptions) {
    let (document, args, initial, options) = super::super::tests::fixture();
    let stream = StreamArgs { max_requests: 1000, max_total_input_bytes: 64 * MIB,
        max_total_output_bytes: 64 * MIB, max_line_bytes: 4 * MIB as usize,
        max_run_forward_positions: None, max_run_projected_logits: None,
        max_run_attention_pairs: None, max_run_dot_products: None,
        max_run_multiply_accumulates: None, max_run_mask_node_visits: None,
        max_run_extraction_fields: None, max_run_extraction_evidence_spans: None,
        max_run_extraction_scan_steps: None };
    let command = CorpusArgs { document, stream };
    let limits = command.policy_limits(&args, initial, &options).unwrap();
    (command, args, limits, options)
}
fn argv(extra: &[&str]) -> Vec<String> {
    let mut args = super::super::tests::argv(extra);
    args[2] = "extract-semantic-document-corpus".to_owned();
    args
}
fn command(extra: &[&str]) -> Int4Command {
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(extra)).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate route"); };
    command
}

#[test]
fn explicit_corpus_route_keeps_complete_document_and_compute_options() {
    for threads in ["1", "2", "4"] {
        let parsed = command(&["--compute-threads", threads]);
        let Int4Task::SemanticObjectCorpus(command) = parsed.task else { panic!("wrong document corpus route"); };
        let limits = command.validate(&parsed.args).unwrap();
        assert_eq!(limits.map.common.compute.threads, threads.parse::<usize>().unwrap());
        assert_eq!(limits.mapping().mapping.mapping.chunks.max_chunks, 4);
        assert!(parsed.args.input_encoding.is_default());
    }
}

#[test]
fn request_count_never_renews_or_multiplies_a_document_allowance() {
    let (mut command, args, limits, options) = fixture();
    assert_eq!(limits.work.max_model_work, options.max_model_work);
    assert_eq!(limits.work.max_mask_node_visits, limits.document.mapping().mapping.mapping.max_mask_visits);
    assert_eq!(limits.work.max_verification, limits.document.mapping().mapping.verification);
    command.stream.max_requests = 2;
    let fewer = command.policy_limits(&args, command.validate(&args).unwrap(), &options).unwrap();
    assert_eq!(fewer.work.max_model_work, limits.work.max_model_work);
    assert_eq!(fewer.work.max_verification, limits.work.max_verification);
    assert_eq!(fewer.work.max_mask_node_visits, limits.work.max_mask_node_visits);
}

#[test]
fn all_five_native_run_bounds_and_three_source_bounds_refuse_underfunding() {
    for axis in 0..9 {
        let (mut c, args, limits, options) = fixture();
        let item = options.max_model_work;
        match axis {
            0 => c.stream.max_run_forward_positions = Some(item.forward_positions - 1),
            1 => c.stream.max_run_projected_logits = Some(item.projected_logits - 1),
            2 => c.stream.max_run_attention_pairs = Some(item.attention_pairs - 1),
            3 => c.stream.max_run_dot_products = Some(item.projections.dot_products - 1),
            4 => c.stream.max_run_multiply_accumulates = Some(item.projections.multiply_accumulates - 1),
            5 => c.stream.max_run_mask_node_visits = Some(limits.work.max_mask_node_visits - 1),
            6 => c.stream.max_run_extraction_fields = Some(limits.work.max_verification.max_fields - 1),
            7 => c.stream.max_run_extraction_evidence_spans = Some(limits.work.max_verification.max_matches - 1),
            _ => c.stream.max_run_extraction_scan_steps = Some(limits.work.max_verification.max_scan_steps - 1),
        }
        assert!(c.policy_limits(&args, limits.document, &options).is_err(), "axis {axis}");
    }
}

#[test]
fn combined_output_and_entire_source_review_determine_transport_not_one_chunk() {
    let (_, args, limits, options) = fixture();
    assert!(limits.transport.max_output_line_bytes as u64 >= options.limits.max_output_bytes + FRAME_ALLOWANCE);
    assert_eq!(limits.transport.max_document_bytes, args.max_input_bytes);
    assert_eq!(limits.review.chunks.max_input_bytes, args.max_input_bytes + options.limits.max_claim_bytes);
    assert_eq!(limits.transport.max_work.forward_positions, limits.work.max_model_work.forward_positions);
    assert_eq!(limits.transport.max_work.projected_logits, limits.work.max_model_work.projected_logits);
    assert!(limits.io_bytes > limits.transport.max_output_line_bytes as u64);
}

#[test]
fn unsupported_schedules_and_borrowed_dispatch_never_read_private_input() {
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid route read private population"); }
    }
    for extra in [vec!["--max-selected-rows", "16"], vec!["--cohort-chunks", "2"],
        vec!["--max-run-mask-node-visits", "0"], vec!["--max-line-bytes", "1"]] {
        let parsed = command(&extra);
        let Int4Task::SemanticObjectCorpus(command) = parsed.task else { panic!("wrong route"); };
        assert!(command.execute_owned(parsed.args, NoRead, Vec::<u8>::new()).is_err());
    }
    let parsed = command(&[]);
    assert_eq!(parsed.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[test]
fn errors_remain_typed_and_do_not_echo_private_paths_or_policy() {
    let mut diagnostics = Vec::new();
    let failure = Failure { exit: ErrorCode::BudgetOrTimeout, code: "semantic_object_corpus_budget",
        cancellation: Some(crate::native_engine::decode::DecodeCancellationKind::CostBudget) };
    let exit = report_failure(failure, &mut diagnostics);
    assert_eq!(exit, ErrorCode::BudgetOrTimeout.as_process_exit());
    let value: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-semantic-object-corpus-v1");
    assert_eq!(value["earlier_complete_records_may_exist"], true);
    assert!(value.get("schema").is_none());
    assert!(value.get("semantic_options").is_none());
}
