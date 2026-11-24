//! Model-free resource/command regressions; no synthetic native successes.
use super::*;
fn command(task: &str, extra: &[&str]) -> BatchCommand {
    let mut argv = vec!["candidate", "batch", "--task", task, "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Batch(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("batch dispatched as a single document") };
    command
}
fn ceiling(command: &BatchCommand) -> TaskBudget {
    let (_, limits, _) = command.validate().unwrap(); command.host.task_budget(limits)
}
#[test]
fn all_source_tasks_parse_without_displacing_single_document_commands() {
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        assert_eq!(command(task, &[]).kind().unwrap().name(), task);
        command(task, &[]).validate().unwrap();
        assert!(super::super::definition().try_get_matches_from(["candidate", task,
            "--model", "m", "--memory-mib", "8192"]).is_ok());
    }
    for args in [vec!["candidate", "batch"], vec!["candidate", "batch", "--task", "ner"],
        vec!["candidate", "batch", "--model", "m", "--memory-mib", "8192"]] {
        assert!(super::super::definition().try_get_matches_from(args).is_err());
    }
    assert!(super::super::definition().try_get_matches_from(["candidate", "batch", "--task", "generate",
        "--model", "m", "--memory-mib", "8192"]).is_err());
}
#[test]
fn all_whole_stream_limits_are_finite_and_checked_before_io() {
    for (flag, value) in [("--max-requests", "0"), ("--max-requests", "100001"),
        ("--max-input-mib", "0"), ("--max-input-mib", "1048577"),
        ("--max-output-mib", "0"), ("--max-output-mib", "1048577"),
        ("--max-output-mib", "1"), // cannot reserve framing for 1000 requests
        ("--max-line-bytes", "1"), ("--max-line-bytes", "4194305"),
        ("--defaults", "-")] {
        assert!(command("ner", &[flag, value]).validate().is_err(), "{flag}");
    }
}
#[test]
fn stream_work_is_the_sum_of_independent_contexts_and_never_a_long_context() {
    let cmd = command("ner", &["--max-requests", "3"]);
    let (_, _, e) = cmd.validate().unwrap();
    let one = constrained_int8::planned_work(cmd.host.context_tokens - cmd.host.max_new_tokens,
        cmd.host.max_new_tokens).unwrap();
    let sum = one.checked_add(one).unwrap().checked_add(one).unwrap();
    assert_eq!(e.native.max_model_work, sum);
    assert_eq!(e.transport.max_work.forward_positions, sum.forward_positions);
    assert_eq!(e.transport.max_work.projected_logits, sum.projected_logits);
    assert_eq!(e.native.masks.max_visits_per_run, cmd.host.max_mask_node_visits * 3);
}

#[test]
fn compact_schema_and_source_corpora_preserve_whole_run_native_and_mask_ceilings() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    for task in ["ner", "keyphrases", "summarize", "answer", "extract"] {
        let byte = command(task, &["--max-requests", "3"]);
        let compact = command(task, &["--max-requests", "3", "--input-encoding", "safe-bpe-v1"]);
        let (_, _, original) = byte.validate().unwrap();
        let (args, limits, envelope) = compact.validate().unwrap();
        assert!(matches!(args.input_encoding_policy(limits).unwrap(), UntrustedEncoding::SafeBpeV1(_)));
        assert_eq!(envelope.native.max_model_work, original.native.max_model_work);
        assert_eq!(envelope.native.masks.max_visits_per_run, original.native.masks.max_visits_per_run);
        assert_eq!(envelope.transport.max_requests, 3);
        assert_eq!(envelope.transport.max_input_bytes, original.transport.max_input_bytes);
        assert_eq!(envelope.output_bytes, original.output_bytes);
    }
}

#[test]
fn invalid_compact_corpus_policies_refuse_before_owned_stream_access() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("corpus input read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("corpus output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("corpus output flushed") }
    }
    for (task, flags) in [
        ("ner", vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"]),
        ("answer", vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "0"]),
        ("extract", vec!["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"]),
        ("extract", vec!["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "851968"]),
        ("extract", vec!["--chunked", "--schema", "must-not-open.json", "--input-encoding", "safe-bpe-v1"]),
    ] {
        assert_eq!(command(task, &flags).execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn compact_extraction_admits_schema_and_source_together_before_private_io() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let cmd = command("extract", &["--input-encoding", "safe-bpe-v1", "--schema", "must-not-open"]);
    let (_, limits, _) = cmd.validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = cmd.encoding_policy(limits).unwrap() else { panic!("wrong encoding"); };
    assert_eq!(policy.max_input_bytes, cmd.host.max_input_bytes + extract::SCHEMA_BYTES);
    assert_eq!(policy.max_output_tokens, limits.max_prompt_tokens);
    let mut large = command("extract", &["--input-encoding", "safe-bpe-v1",
        "--max-input-bytes", "1048576", "--preparation-mib", "1024"]);
    let (_, limits, _) = large.validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = large.encoding_policy(limits).unwrap() else { panic!("wrong encoding"); };
    assert_eq!(policy.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    large.host.max_input_bytes = 851968;
    large.host.preparation_mib = 512;
    // Document-only admission fits. The complete schema/source quote must
    // still reject before either a shared schema or the NDJSON stream opens.
    assert!(large.host.common_with_encoding(large.input.clone(), &large.encoding).is_ok());
    assert!(large.validate().is_err());
}
#[test]
fn every_native_work_axis_uses_checked_scaling() {
    for axis in 0..5 {
        let mut work = Int8Work::default();
        match axis { 0 => work.forward_positions = u64::MAX, 1 => work.projected_logits = u64::MAX,
            2 => work.attention_pairs = u64::MAX, 3 => work.projections.dot_products = u64::MAX,
            _ => work.projections.multiply_accumulates = u64::MAX }
        assert!(scale_work(work, 2).is_err());
    }
    assert!(scale_work(Int8Work::default(), 0).is_err());
}
#[test]
fn outer_provenance_is_reserved_separately_from_native_payload_bytes() {
    let cmd = command("ner", &[]); let (_, _, e) = cmd.validate().unwrap();
    assert_eq!(e.transport.max_output_bytes + (cmd.max_requests + 4) * FRAME_ALLOWANCE, e.output_bytes);
    assert_eq!(e.output_bytes, cmd.max_output_mib * MIB);
    assert_eq!(e.transport.max_input_bytes, cmd.max_input_mib * MIB);
    assert!(e.io_bytes >= e.transport.max_output_line_bytes as u64 + IO_BUFFER_BYTES as u64);
}
#[test]
fn plain_source_defaults_have_the_exact_cli_budget_and_qa_has_no_invented_evidence() {
    for task in ["ner", "keyphrases", "summarize"] {
        let cmd = command(task, &[]); let cap = ceiling(&cmd);
        let defaults = cmd.load_defaults(None, cap).unwrap().unwrap();
        let value = serde_json::to_value(defaults).unwrap();
        assert_eq!(value["task"], task);
        assert_eq!(value["budget"], serde_json::to_value(cap).unwrap());
    }
    let cmd = command("answer", &[]);
    assert!(cmd.load_defaults(None, ceiling(&cmd)).unwrap().is_none());
}
#[test]
fn default_task_and_every_budget_axis_cannot_override_the_command() {
    let cmd = command("ner", &[]); let cap = ceiling(&cmd);
    let original = serde_json::to_value(cmd.load_defaults(None, cap).unwrap().unwrap()).unwrap();
    for field in ["max_input_tokens", "max_output_tokens", "max_output_bytes", "max_grammar_states", "max_kv_bytes"] {
        let mut value = original.clone(); value["budget"][field] = serde_json::json!(u64::MAX);
        assert!(cmd.load_defaults(Some(&value.to_string()), cap).is_err(), "{field}");
    }
    let mut smaller_kv = original.clone(); smaller_kv["budget"]["max_kv_bytes"] = serde_json::json!(1);
    assert!(cmd.load_defaults(Some(&smaller_kv.to_string()), cap).is_err());
    let other = SourceBatchArgs::Keyphrases { options: KeyphraseOptions::default(), budget: cap };
    assert!(cmd.load_defaults(Some(&serde_json::to_string(&other).unwrap()), cap).is_err());
}
#[test]
fn lower_complete_defaults_are_accepted_without_modifying_the_host_ceiling() {
    let cmd = command("ner", &[]); let cap = ceiling(&cmd); let mut lower = cap;
    lower.max_output_tokens = 32;
    let args = SourceBatchArgs::Ner { options: NerOptions { max_entities: 2, ..NerOptions::default() }, budget: lower };
    let defaults = cmd.load_defaults(Some(&serde_json::to_string(&args).unwrap()), cap).unwrap().unwrap();
    let SourceBatchArgs::Ner { options, budget } = defaults else { panic!("wrong task") };
    assert_eq!(options.max_entities, 2); assert_eq!(budget.max_output_tokens, 32);
    assert_eq!(ceiling(&cmd).max_output_tokens, cap.max_output_tokens);
}
#[test]
fn duplicate_keys_unknown_fields_and_partial_defaults_are_not_coerced() {
    let cmd = command("ner", &[]); let cap = ceiling(&cmd);
    for raw in [r#"{}"#, r#"{"task":"ner","task":"answer"}"#,
        r#"{"task":"ner","\u0074ask":"ner"}"#] {
        assert!(cmd.load_defaults(Some(raw), cap).is_err());
    }
    let mut value = serde_json::to_value(cmd.load_defaults(None, cap).unwrap().unwrap()).unwrap();
    value["instruction"] = serde_json::json!("private override");
    assert!(cmd.load_defaults(Some(&value.to_string()), cap).is_err());
    let huge = " ".repeat(MAX_SOURCE_ARGUMENT_BYTES + 1);
    assert!(cmd.load_defaults(Some(&huge), cap).is_err());
}
#[test]
fn qa_default_passages_are_bounded_and_keep_unique_original_ids() {
    let cmd = command("answer", &[]); let cap = ceiling(&cmd);
    let args = |ids: &[&str]| SourceBatchArgs::Answer {
        passages: ids.iter().map(|id| crate::tasks::answer::AnswerPassage { id: id.to_string(), text: "Alice".to_owned() }).collect(),
        options: crate::tasks::answer::AnswerOptions::default(), budget: cap,
    };
    assert!(cmd.load_defaults(Some(&serde_json::to_string(&args(&["a", "b"])).unwrap()), cap).is_ok());
    for ids in [vec![], vec![""], vec!["same", "same"], vec!["a\n"], vec!["a"; 33]] {
        assert!(cmd.load_defaults(Some(&serde_json::to_string(&args(&ids)).unwrap()), cap).is_err());
    }
}
#[test]
fn protocol_completion_is_not_all_documents_success() {
    let summary = BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() };
    assert_eq!(completed(summary), Err(CandidateError::Batch));
    assert!(completed(BatchSummary { succeeded: 2, ..BatchSummary::default() }).is_ok());
    assert!(!CandidateError::Batch.message().contains("no result published"));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn feature_disabled_batch_never_reads_or_writes_owned_streams() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let cmd = command(task, &["--defaults", "never-open-defaults.json"]);
        assert_eq!(cmd.execute_owned(NoRead, NoWrite), Err(CandidateError::Unavailable));
    }
}

#[test]
fn structured_corpora_compose_prefill_and_workers_without_renewing_record_or_run_budgets() {
    for task in ["ner", "keyphrases", "summarize", "answer", "extract"] {
        for compact in [false, true] {
            let mut base_flags = vec!["--max-requests", "3"];
            if compact { base_flags.extend(["--input-encoding", "safe-bpe-v1"]); }
            let baseline = command(task, &base_flags);
            let (_, original_limits, original) = baseline.validate().unwrap();
            for width in ["1", "4"] {
                for rows in [None, Some("1"), Some("7")] {
                    let mut flags = base_flags.clone();
                    flags.extend(["--compute-threads", width]);
                    if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
                    let mut selected = command(task, &flags);
                    selected.host.memory_mib = 16384;
                    let (args, limits, envelope) = selected.validate().unwrap();
                    let prefill = selected.prefill.limits().unwrap();
                    assert_eq!(limits.compute, args.compute.admit_prefill(prefill).unwrap());
                    assert_eq!(limits.compute.threads, width.parse::<usize>().unwrap());
                    assert_eq!(envelope.native.max_model_work, original.native.max_model_work);
                    assert_eq!(envelope.native.masks.max_visits_per_item, original.native.masks.max_visits_per_item);
                    assert_eq!(envelope.native.masks.max_visits_per_run, original.native.masks.max_visits_per_run);
                    assert_eq!(envelope.transport.max_work, original.transport.max_work);
                    assert_eq!(envelope.output_bytes, original.output_bytes);
                    assert_eq!(selected.host.task_budget(limits), baseline.host.task_budget(original_limits));
                    assert_eq!(selected.encoding_policy(limits).unwrap(), baseline.encoding_policy(original_limits).unwrap());
                }
            }
        }
    }
}

#[test]
fn structured_scheduling_prices_every_simultaneous_owner_and_exact_memory_boundary() {
    for task in ["ner", "keyphrases", "summarize", "answer", "extract"] {
        let mut selected = command(task, &["--compute-threads", "4", "--prefill-rows", "7",
            "--input-encoding", "safe-bpe-v1"]);
        selected.host.memory_mib = 16384;
        let (_, mut limits, envelope) = selected.validate().unwrap();
        let prefill = selected.prefill.limits().unwrap();
        let peak = selected.schedule_peak_bytes(limits, envelope, prefill).unwrap();
        limits.memory_bytes = peak;
        selected.check_scheduling_memory(limits, envelope, prefill).unwrap();
        limits.memory_bytes -= 1;
        assert_eq!(selected.check_scheduling_memory(limits, envelope, prefill), Err(CandidateError::Arguments));
        let mut without_workers = limits;
        without_workers.compute.workspace_bytes = 0;
        assert_eq!(peak - selected.schedule_peak_bytes(without_workers, envelope, prefill).unwrap(),
            limits.compute.workspace_bytes);
        assert_eq!(peak - selected.schedule_peak_bytes(limits, envelope, None).unwrap(),
            prefill.unwrap().validate().unwrap());
        let mut retained = envelope;
        retained.io_bytes += 19;
        assert_eq!(selected.schedule_peak_bytes(limits, retained, prefill).unwrap(), peak + 19);
        let mut caller = limits;
        caller.preparation_bytes += 23;
        assert_eq!(selected.schedule_peak_bytes(caller, envelope, prefill).unwrap(), peak + 46);
        retained.io_bytes = u64::MAX;
        assert!(selected.schedule_peak_bytes(limits, retained, prefill).is_err());
    }
}

#[test]
fn invalid_or_underfunded_structured_schedules_refuse_before_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private corpus input read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("partial corpus output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("corpus output flushed") }
    }
    for task in ["ner", "keyphrases", "summarize", "answer", "extract"] {
        for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
            vec!["--prefill-rows", "0"], vec!["--prefill-rows", "65"],
            vec!["--compute-threads", "4", "--prefill-rows", "7", "--max-weight-mib", "8192"]] {
            assert_eq!(command(task, &flags).execute_owned(NoIo, NoIo), Err(CandidateError::Arguments));
        }
    }
}
