//! Parser and bounded policy regressions; native/model execution is separate.
use super::*;
use crate::batch::source::semantic::Int4SemanticSourceBatchArgs;

const PIN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
fn name(kind: Kind) -> &'static str {
    match kind { Kind::Summarize => "summarize-semantic-batch", Kind::Answer => "answer-semantic-batch",
        _ => panic!("summary or answer fixture required") }
}
pub(super) fn command(kind: Kind, extra: &[&str], snapshot: bool) -> (Int4Args, BatchArgs) {
    let mut words = vec!["candidate", "source-int4", name(kind), "--memory-mib", "65536",
        "--semantic-options", "private-review-policy"];
    if snapshot { words.extend(["--snapshot", "private-snapshot", "--expected-model-sha256", PIN]); }
    else { words.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest"]); }
    words.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(words).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate dispatcher"); };
    let Int4Task::SourceSemanticBatch(batch) = command.task else { panic!("wrong semantic population route"); };
    assert_eq!(batch.kind, kind);
    (command.args, batch)
}
pub(super) fn options(kind: Kind) -> Int4SourceSemanticOptions { super::super::tests::fixture(kind).3 }
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
fn both_fixed_tasks_support_source_snapshot_save_and_compact_team_routes() {
    for kind in [Kind::Summarize, Kind::Answer] {
        for snapshot in [false, true] {
            for flags in [&[][..], &["--compute-threads", "4"][..],
                &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..]] {
                let mut selected = flags.to_vec(); selected.push("population.ndjson");
                let (args, batch) = command(kind, &selected, snapshot);
                let initial = batch.validate(&args).unwrap();
                let semantic = options(kind);
                let limits = batch.policy_limits(&args, initial, &semantic).unwrap();
                let task = batch.load_task(None, limits.task, batch.semantic.source.planning(&args), &semantic).unwrap();
                assert_eq!(task.built_in(), source::task(kind));
                assert_eq!(task.budget(), limits.task);
                assert_eq!(args.input, PathBuf::from("population.ndjson"));
                assert_eq!(initial.snapshot_pin.is_some(), snapshot);
                assert_eq!(limits.common.compute.threads, if flags.is_empty() { 1 } else { 4 });
                assert!(args.input_encoding.is_default(), "semantic input uses its own aggregate ledger");
                assert_eq!(limits.common.result_bytes, semantic.limits.max_output_bytes as usize);
                assert_eq!(limits.task.max_output_bytes, 1 << 20);
                assert!(limits.common.prefill.is_none());
            }
        }
        let (args, batch) = command(kind, &["--save-snapshot", "private-new-snapshot"], false);
        batch.policy_limits(&args, batch.validate(&args).unwrap(), &options(kind)).unwrap();
        assert_eq!(args.save_snapshot, Some(PathBuf::from("private-new-snapshot")));
    }
}

#[test]
fn borrowed_dispatch_refuses_both_owned_population_routes_before_io() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name(kind), "--source-dir", "private-source",
            "--source-manifest", "private-manifest", "--memory-mib", "65536",
            "--semantic-options", "private-review-policy",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong candidate dispatcher"); };
        assert_eq!(command.execute(&mut NoRead, &mut NoWrite), Err(CandidateError::Arguments));
    }
}

#[test]
fn population_size_never_renews_or_multiplies_generation_review_or_mask_work() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (args, mut batch) = command(kind, &["--max-requests", "2"], false);
        let options = options(kind); let initial = batch.validate(&args).unwrap();
        let first = batch.policy_limits(&args, initial, &options).unwrap();
        batch.stream.max_requests = 200;
        let larger = batch.policy_limits(&args, initial, &options).unwrap();
        assert_eq!(first.native.max_model_work, options.max_model_work);
        assert_eq!(first.native.max_model_work, larger.native.max_model_work);
        assert_eq!(first.native.masks.max_visits_per_run, batch.semantic.source.mask_visits());
        assert_eq!(first.native.masks.max_visits_per_run, larger.native.masks.max_visits_per_run);
        assert_eq!(first.transport.max_work, larger.transport.max_work);
        assert_eq!(first.task, larger.task); assert_eq!(first.io_bytes, larger.io_bytes);
        assert_eq!(first.transport.max_output_bytes - larger.transport.max_output_bytes, 198 * FRAME_ALLOWANCE);
    }
}

#[test]
fn every_total_axis_covers_a_complete_attempt_and_can_independently_grow() {
    let (args, base) = command(Kind::Answer, &[], false);
    let options = options(Kind::Answer); let initial = base.validate(&args).unwrap();
    for axis in 0..6 {
        let mut batch = base.clone();
        match axis {
            0 => batch.stream.max_total_forward_positions = Some(options.max_model_work.forward_positions - 1),
            1 => batch.stream.max_total_projected_logits = Some(options.max_model_work.projected_logits - 1),
            2 => batch.stream.max_total_attention_pairs = Some(options.max_model_work.attention_pairs - 1),
            3 => batch.stream.max_total_dot_products = Some(options.max_model_work.projections.dot_products - 1),
            4 => batch.stream.max_total_multiply_accumulates = Some(options.max_model_work.projections.multiply_accumulates - 1),
            _ => batch.stream.max_total_mask_node_visits = Some(batch.semantic.source.mask_visits() - 1),
        }
        assert!(batch.policy_limits(&args, initial, &options).is_err(), "axis {axis}");
    }
    let mut doubled = base.clone(); let twice = options.max_model_work.checked_add(options.max_model_work).unwrap();
    doubled.stream.max_total_forward_positions = Some(twice.forward_positions);
    doubled.stream.max_total_projected_logits = Some(twice.projected_logits);
    doubled.stream.max_total_attention_pairs = Some(twice.attention_pairs);
    doubled.stream.max_total_dot_products = Some(twice.projections.dot_products);
    doubled.stream.max_total_multiply_accumulates = Some(twice.projections.multiply_accumulates);
    doubled.stream.max_total_mask_node_visits = Some(2 * base.semantic.source.mask_visits());
    let limits = doubled.policy_limits(&args, initial, &options).unwrap();
    assert_eq!(limits.native.max_model_work, twice);
    assert_eq!(limits.native.masks.max_visits_per_item, base.semantic.source.mask_visits());
    assert_eq!(limits.native.masks.max_visits_per_run, 2 * limits.native.masks.max_visits_per_item);
}

#[test]
fn fixed_source_options_refuse_authority_overrides_and_impossible_review_shapes() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (args, batch) = command(kind, &[], false); let semantic = options(kind);
        let limits = batch.policy_limits(&args, batch.validate(&args).unwrap(), &semantic).unwrap();
        let planning = batch.semantic.source.planning(&args);
        let default = batch.load_task(None, limits.task, planning, &semantic).unwrap();
        let good = match kind {
            Kind::Summarize => serde_json::to_value(SummaryOptions::default()).unwrap(),
            Kind::Answer => serde_json::to_value(AnswerOptions::default()).unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(batch.load_task(Some(&good.to_string()), limits.task, planning, &semantic).unwrap(), default);
        for field in ["task", "budget", "semantic_options", "identity", "prompt", "passages"] {
            let mut value = good.clone(); value[field] = serde_json::json!({});
            assert!(batch.load_task(Some(&value.to_string()), limits.task, planning, &semantic).is_err(), "{field}");
        }
        assert!(batch.load_task(Some("{}"), limits.task, planning, &semantic).is_err());
        let mut short = planning; short.compiler.max_states = 1;
        assert!(batch.load_task(None, limits.task, short, &semantic).is_err());
        let bad = match kind {
            Kind::Summarize => serde_json::to_string(&SummaryOptions { max_bullets: semantic.limits.max_statements + 1,
                ..SummaryOptions::default() }).unwrap(),
            Kind::Answer => serde_json::to_string(&AnswerOptions { max_answer_scalars: 0,
                ..AnswerOptions::default() }).unwrap(),
            _ => unreachable!(),
        };
        assert!(batch.load_task(Some(&bad), limits.task, planning, &semantic).is_err());
    }
}

#[test]
fn qa_wire_arguments_contain_passages_only_and_reject_nested_replacements() {
    let valid = serde_json::json!({"passages":[{"id":"原文","text":"The amount is 7. 上海"}]});
    let data: Int4SemanticSourceBatchArgs = serde_json::from_value(valid.clone()).unwrap();
    assert_eq!(data.passages[0].id, "原文");
    assert_eq!(data.passages[0].text, "The amount is 7. 上海");
    for field in ["task", "options", "budget", "semantic_options", "identity", "question", "token_ids"] {
        let mut value = valid.clone(); value[field] = serde_json::json!({});
        assert!(serde_json::from_value::<Int4SemanticSourceBatchArgs>(value).is_err(), "{field}");
    }
    let mut value = valid; value["passages"][0]["options"] = serde_json::json!({});
    assert!(serde_json::from_value::<Int4SemanticSourceBatchArgs>(value).is_err());
    assert!(serde_json::from_str::<Int4SemanticSourceBatchArgs>("{}").is_err());
}

#[test]
fn qa_input_geometry_and_combined_output_are_separate_from_question_and_generation_caps() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (args, mut batch) = command(kind, &[], false); let options = options(kind);
        let initial = batch.validate(&args).unwrap();
        let limits = batch.policy_limits(&args, initial, &options).unwrap();
        let facts = input_ceiling(kind, &args, &options).unwrap();
        assert_eq!(facts.source_bytes, args.max_input_bytes.min(options.limits.max_source_bytes));
        assert_eq!(facts.question_bytes, if kind == Kind::Answer { args.max_input_bytes.min(options.limits.max_claim_bytes) } else { 0 });
        assert_eq!(facts.passage_count, if kind == Kind::Answer { crate::candidate_cli::source::MAX_PASSAGES } else { 0 });
        assert_eq!(batch.semantic.source.planning(&args).max_input_bytes, limits.transport.max_document_bytes);
        assert_eq!(limits.transport.max_output_line_bytes, options.limits.max_output_bytes as usize + 4096);
        assert!(limits.transport.max_output_line_bytes as u64 > limits.task.max_output_bytes);
        assert_eq!(limits.output_bytes, limits.transport.max_output_bytes + (batch.stream.max_requests + 4) * FRAME_ALLOWANCE);
        batch.stream.max_total_output_bytes = FRAME_ALLOWANCE;
        assert!(batch.policy_limits(&args, initial, &options).is_err());
    }
}

#[test]
fn invalid_paths_limits_and_unsupported_kinds_refuse_before_private_io() {
    for kind in [Kind::Summarize, Kind::Answer] {
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
            &["--options", "-"][..], &["--options", "https://example.invalid/private-options"][..],
        ] {
            let (args, batch) = command(kind, flags, false);
            assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage, "{kind:?} {flags:?}");
        }
        for path in ["", "-", "https://example.invalid/private-review-policy"] {
            let (args, mut batch) = command(kind, &[], false);
            batch.semantic.semantic_options = PathBuf::from(path);
            assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
        }
    }
    for wrong in [Kind::Ner, Kind::Keyphrases] {
        let (args, mut batch) = command(Kind::Summarize, &[], false); batch.kind = wrong;
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
fn generation_selected_rows_prefill_and_cohorts_are_never_silently_accepted() {
    for kind in [Kind::Summarize, Kind::Answer] {
        for flags in [
            &["--seed", PIN][..], &["--logprobs"][..], &["--prefill-rows", "1"][..],
            &["--max-selected-rows", "32"][..], &["--cohort-documents", "2"][..],
            &["--defaults", "private-defaults"][..], &["--think"][..],
        ] {
            let mut words = vec!["candidate", "source-int4", name(kind), "--source-dir", "private-source",
                "--source-manifest", "private-manifest", "--semantic-options", "private-review-policy", "--memory-mib", "65536"];
            words.extend_from_slice(flags);
            if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(words) {
                assert!(CandidateCommand::from_matches(&matches).is_err(), "{kind:?} {flags:?}");
            }
        }
    }
}

#[test]
fn stderr_retains_all_cancellation_causes_and_omits_private_configuration() {
    use crate::native_engine::decode::DecodeCancellationKind::*;
    for kind in [Kind::Summarize, Kind::Answer] {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
            let mut stderr = Vec::new();
            report_failure(kind, Failure { exit: ErrorCode::Cancelled, code: "fixture_cancelled",
                cancellation: Some(cause) }, &mut stderr);
            let value: serde_json::Value = serde_json::from_slice(&stderr).unwrap();
            assert_eq!(value["protocol"], PROTOCOL); assert_eq!(value["task"], task_name(kind));
            assert_eq!(value["event"], "error"); assert_eq!(value["code"], "fixture_cancelled");
            assert_eq!(value["cancellation"], serde_json::to_value(cause).unwrap());
            assert_eq!(value["exit_code"], serde_json::to_value(ErrorCode::Cancelled).unwrap());
            assert_eq!(value["earlier_complete_records_may_exist"], true); assert!(stderr.len() < 4096);
            for key in ["document", "question", "passages", "schema", "options", "source", "key", "semantic_options"] {
                assert!(value.get(key).is_none());
            }
        }
        let (_, batch) = command(kind, &["--options", "PRIVATE_SOURCE_OPTIONS"], false);
        let debug = format!("{batch:?}");
        assert!(!debug.contains("PRIVATE_SOURCE_OPTIONS") && !debug.contains("private-review-policy"));
    }
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::ErrorKind::BrokenPipe.into()) }
        fn flush(&mut self) -> io::Result<()> { panic!("failed diagnostic write must stop"); }
    }
    assert_eq!(report_failure(Kind::Answer, Failure::usage("refused"), &mut Broken), ErrorCode::Generic.as_process_exit());
}

#[test]
fn any_failed_record_prevents_success_even_after_protocol_completion() {
    completed(BatchSummary::default()).unwrap();
    let failure = completed(BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() }).unwrap_err();
    assert_eq!(failure.code, "semantic_source_batch_items_failed"); assert_ne!(failure.exit, ErrorCode::Ok);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_runtime_refuses_both_routes_without_private_io() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (args, batch) = command(kind, &[], false);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite).unwrap_err().code, "semantic_source_batch_profile_unavailable");
    }
}
