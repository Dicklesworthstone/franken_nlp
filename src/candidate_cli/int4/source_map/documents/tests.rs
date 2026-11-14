//! UNRUN parser, scope and simultaneous-resource regression definitions.
use super::*;

pub(super) fn parsed(name: &str, flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4"];
    if name == "answer" { argv.extend_from_slice(&["source-corpus", "answer"]); }
    else { argv.push(name); }
    argv.extend_from_slice(&["--memory-mib", "131072"]);
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("source INT4 required"); };
    Ok(command)
}
pub(super) fn arguments(name: &str, flags: &[&str]) -> (CorpusArgs, Int4Args) {
    let mut all = vec!["--source-dir", "private-source", "--source-manifest", "private-manifest"];
    if name == "answer" && !flags.contains(&"--question") {
        all.extend_from_slice(&["--question", "private-question.txt"]);
    }
    all.extend_from_slice(flags);
    let command = parsed(name, &all).unwrap();
    let Int4Task::SourceCorpus(options) = command.task else { panic!("complete source corpus required"); };
    (options, command.args)
}
pub(super) struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused source corpus read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused source corpus published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused source corpus flushed output"); }
}

#[test]
fn persistent_teams_cover_every_document_mode_without_widening_semantic_budgets() {
    for (name, flags) in [
        ("ner-corpus", &[][..]), ("keyphrases-corpus", &["--reduce-keyphrases"][..]),
        ("summarize-corpus", &[][..]), ("summarize-corpus", &["--synthesize-summary"][..]),
        ("summarize-corpus", &["--synthesize-summary", "--hierarchical-summary"][..]),
        ("answer", &[][..]), ("answer", &["--synthesize-answer"][..]),
        ("answer", &["--synthesize-answer", "--hierarchical-answer"][..]),
    ] {
        let (serial, serial_args) = arguments(name, flags);
        let baseline = serial.validate(&serial_args).unwrap();
        for width in ["2", "4", "64"] {
            let mut selected = flags.to_vec(); selected.extend_from_slice(&["--compute-threads", width]);
            let (command, args) = arguments(name, &selected);
            let quoted = command.validate(&args).unwrap();
            assert_eq!(quoted.common.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(quoted.run, baseline.run);
            assert_eq!(quoted.transport, baseline.transport);
            assert_eq!(quoted.document.mapping.max_model_work, baseline.document.mapping.max_model_work);
            assert_eq!(quoted.document.mapping.verification, baseline.document.mapping.verification);
            assert_eq!(quoted.document.workspace_bytes, baseline.document.workspace_bytes);
            assert_eq!(quoted.execution_bytes - baseline.execution_bytes, quoted.common.compute.workspace_bytes);
            assert_eq!(quoted.common.compute.workspace_bytes,
                crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(
                    width.parse().unwrap(), args.group_width).unwrap());
            assert_eq!(command.mode_name(quoted.document), serial.mode_name(baseline.document));
        }
    }
}

#[test]
fn corpus_team_aliases_and_compact_input_share_the_actual_parser_and_admission() {
    for (task, short) in [("ner", "ner-corpus"), ("keyphrases", "keyphrases-corpus"),
        ("summarize", "summarize-corpus")] {
        for encoding in ["byte-fallback", "safe-bpe-v1"] {
            let (short_options, short_args) = arguments(short,
                &["--compute-threads", "3", "--input-encoding", encoding]);
            let argv = vec!["candidate", "source-int4", "source-corpus", task,
                "--memory-mib", "131072", "--source-dir", "private-source", "--source-manifest", "private-manifest",
                "--compute-threads", "3", "--input-encoding", encoding];
            let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
            let CandidateCommand::SourceInt4(long) = CandidateCommand::from_matches(&matches).unwrap()
                else { panic!("source INT4 required"); };
            let Int4Task::SourceCorpus(long_options) = long.task else { panic!("source corpus required"); };
            assert_eq!(long_options, short_options);
            assert_eq!(long.args.compute, short_args.compute);
            assert_eq!(long.args.compute.threads(), 3);
            assert_eq!(long_options.validate(&long.args).unwrap().common.compute,
                short_options.validate(&short_args).unwrap().common.compute);
        }
    }
}

#[test]
fn corpus_teams_refuse_unfunded_widths_and_schedule_mixtures_before_owned_io() {
    for name in ["ner-corpus", "keyphrases-corpus", "summarize-corpus", "answer"] {
        for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
            vec!["--compute-threads", "3", "--cohort-chunks", "1"],
            vec!["--compute-threads", "3", "--cohort-chunks", "2"]] {
            let (command, args) = arguments(name, &flags);
            assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
        }
        let (command, mut args) = arguments(name, &["--compute-threads", "3"]);
        let quote = command.validate(&args).unwrap();
        args.memory_mib = quote.common.minimum_memory_bytes.div_ceil(MIB) - 1;
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
    for (name, synthesis, hierarchy) in [("summarize-corpus", "--synthesize-summary", "--hierarchical-summary"),
        ("answer", "--synthesize-answer", "--hierarchical-answer")] {
        let (command, args) = arguments(name,
            &[synthesis, hierarchy, "--compute-threads", "3", "--cohort-hierarchy-groups", "1"]);
        assert_eq!(command.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn four_owned_routes_preserve_every_existing_semantic_document_mode() {
    for (name, flags, mode) in [
        ("ner-corpus", &[][..], "source_map"),
        ("keyphrases-corpus", &[][..], "source_map"),
        ("keyphrases-corpus", &["--reduce-keyphrases"][..], "keyphrase_ranking"),
        ("summarize-corpus", &[][..], "summary_selection"),
        ("summarize-corpus", &["--synthesize-summary"][..], "summary_synthesis"),
        ("summarize-corpus", &["--synthesize-summary", "--hierarchical-summary"][..], "summary_hierarchy"),
        ("answer", &[][..], "question_discovery"),
        ("answer", &["--synthesize-answer"][..], "question_synthesis"),
        ("answer", &["--synthesize-answer", "--hierarchical-answer"][..], "question_hierarchy"),
    ] {
        let (options, args) = arguments(name, flags);
        let limits = options.validate(&args).unwrap();
        assert_eq!(options.mode_name(limits.document), mode);
        assert_eq!(format!("{options:?}"), "SourceDocumentCorpusArgs");
        assert_eq!(limits.transport.max_document_bytes, args.max_input_bytes);
        assert_eq!(limits.transport.max_output_line_bytes,
            limits.document.max_result_bytes + FRAME_ALLOWANCE as usize);
        assert_eq!(limits.document.mapping.chunks.max_input_bytes, args.max_input_bytes);
        let command = Int4Command { task: Int4Task::SourceCorpus(options), args, batch: None, job: None };
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn umbrella_routes_match_short_forms_and_keep_per_record_questions_separate() {
    for (task, short) in [("ner", "ner-corpus"), ("keyphrases", "keyphrases-corpus"), ("summarize", "summarize-corpus")] {
        let (short_args, _) = arguments(short, &[]);
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", "source-corpus", task, "--memory-mib", "131072",
            "--source-dir", "private-source", "--source-manifest", "private-manifest",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("source corpus required"); };
        let Int4Task::SourceCorpus(long_args) = command.task else { panic!("fixed document pipeline required"); };
        assert_eq!(long_args, short_args);
    }
    let question = parsed("answer-corpus", &["--source-dir", "source", "--source-manifest", "manifest"]).unwrap();
    assert!(matches!(question.task, Int4Task::AnswerCorpus(_)));
    assert!(parsed("answer", &["--source-dir", "source", "--source-manifest", "manifest"]).is_err());
}

#[test]
fn record_and_cohort_counts_never_multiply_native_mask_or_verifier_allowances() {
    for name in ["ner-corpus", "keyphrases-corpus", "summarize-corpus", "answer"] {
        let (mut options, args) = arguments(name, &["--max-requests", "10"]);
        let serial = options.validate(&args).unwrap();
        options.stream.max_requests = 1000;
        options.document.cohort_chunks = Some(3);
        let grouped = options.validate(&args).unwrap();
        assert_eq!(serial.run, grouped.run);
        assert_eq!(serial.transport.max_work, grouped.transport.max_work);
        assert_eq!(serial.document.mapping.max_mask_visits, grouped.document.mapping.max_mask_visits);
        assert_eq!(serial.document.mapping.verification, grouped.document.mapping.verification);
        assert_eq!(options.document.source.budget(&args, serial.common),
            options.document.source.budget(&args, grouped.common));
        assert_eq!(grouped.document.cohort.unwrap().native.kv_bytes, serial.common.native.kv_bytes * 3);
        assert_eq!(grouped.transport.max_output_bytes + 1004 * FRAME_ALLOWANCE, grouped.output_bytes);
        // Transport may admit more records than the independent verifier can
        // fund. A depleted execution budget must refuse, never renew at flush.
        let per_document = options.document.document_verification(grouped.document).unwrap();
        options.stream.max_run_mask_node_visits = grouped.document.mapping.max_mask_visits;
        options.stream.max_run_source_fields = per_document.max_fields;
        options.stream.max_run_source_evidence_spans = per_document.max_matches;
        options.stream.max_run_source_scan_steps = per_document.max_scan_steps;
        options.validate(&args).unwrap();
    }
}

#[test]
fn snapshots_and_source_conversion_keep_identical_document_and_stream_authority() {
    let (source, args) = arguments("answer", &["--synthesize-answer", "--hierarchical-answer",
        "--cohort-chunks", "3", "--cohort-hierarchy-groups", "2"]);
    let original = source.validate(&args).unwrap();
    let pin = "ab".repeat(32);
    let command = parsed("answer", &["--snapshot", "private.snapshot", "--expected-model-sha256", &pin,
        "--question", "private-question.txt", "--synthesize-answer", "--hierarchical-answer",
        "--cohort-chunks", "3", "--cohort-hierarchy-groups", "2"]).unwrap();
    let Int4Task::SourceCorpus(restored) = command.task else { panic!("wrong snapshot route"); };
    let loaded = restored.validate(&command.args).unwrap();
    assert!(original.common.snapshot_pin.is_none()); assert!(loaded.common.snapshot_pin.is_some());
    assert_eq!(original.transport, loaded.transport); assert_eq!(original.run, loaded.run);
    assert_eq!(original.execution_bytes, loaded.execution_bytes);
    assert_eq!(original.document.cohort, loaded.document.cohort);
    assert_eq!(original.document.hierarchy_cohorts, loaded.document.hierarchy_cohorts);
}

#[test]
fn corpus_and_hierarchy_owners_are_admitted_together_before_any_io() {
    for flags in [&[][..], &["--cohort-chunks", "3"][..],
        &["--synthesize-summary", "--hierarchical-summary", "--cohort-hierarchy-groups", "2"][..],
        &["--synthesize-summary", "--hierarchical-summary", "--cohort-chunks", "3", "--cohort-hierarchy-groups", "2"][..]] {
        let (options, mut args) = arguments("summarize-corpus", flags);
        let limits = options.validate(&args).unwrap();
        assert_eq!(limits.cli_preparation_bytes, limits.common.preparation_bytes + limits.io_bytes);
        assert_eq!(limits.common.minimum_memory_bytes,
            grouped_memory_floor(&args, limits.common, limits.execution_bytes).unwrap() + limits.io_bytes);
        args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
        options.validate(&args).unwrap();
        args.memory_mib -= 1;
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn insufficient_run_budgets_and_invalid_transport_refuse_before_private_io() {
    for flags in [vec!["--max-requests", "0"], vec!["--max-requests", "100001"],
        vec!["--max-total-input-bytes", "0"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-line-bytes", "1048575"], vec!["--max-line-bytes", "4194305"],
        vec!["--max-run-mask-node-visits", "127999999999"], vec!["--max-run-source-fields", "4095"],
        vec!["--max-run-source-evidence-spans", "16383"], vec!["--max-run-source-scan-steps", "67108863"],
        vec!["--max-run-summary-scan-steps", "536870911"], vec!["--max-run-keyphrase-scan-work", "0"],
        vec!["--max-run-source-fields", "1000000001"], vec!["--max-forward-positions", "0"],
        vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--max-chunks", "2", "--cohort-chunks", "3"], vec!["--preparation-mib", "1"]] {
        let (options, args) = arguments("summarize-corpus", &flags);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments), "{flags:?}");
    }
    let (options, args) = arguments("keyphrases-corpus", &["--reduce-keyphrases", "--max-run-keyphrase-scan-work", "536870911"]);
    assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn single_context_and_durable_controls_cannot_leak_into_live_complete_sources() {
    for flags in [vec!["--prefill-rows", "1"], vec!["--cohort-documents", "1"],
        vec!["--max-selected-rows", "128"], vec!["--defaults", "private.json"],
        vec!["--store-results"], vec!["--job-dir", "private-job"], vec!["--materialize"],
        vec!["--map-only"], vec!["--logprobs"], vec!["--seed", "ab"]] {
        let mut all = vec!["--source-dir", "source", "--source-manifest", "manifest"];
        all.extend_from_slice(&flags);
        assert!(parsed("ner-corpus", &all).is_err(), "{flags:?}");
    }
    assert!(parsed("answer", &["--source-dir", "source", "--source-manifest", "manifest"]).is_err());
    assert!(parsed("summarize-corpus", &["--source-dir", "source", "--source-manifest", "manifest",
        "--hierarchical-summary"]).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn missing_runtime_refuses_before_options_question_model_or_population_io() {
    for name in ["ner-corpus", "keyphrases-corpus", "summarize-corpus", "answer"] {
        let (options, args) = arguments(name, &["--options", "private-options.json"]);
        assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Unavailable));
    }
}

#[test]
fn compact_source_populations_preserve_whole_run_limits_and_per_document_workspaces() {
    for (name, mode) in [
        ("ner-corpus", &[][..]), ("keyphrases-corpus", &[][..]),
        ("keyphrases-corpus", &["--reduce-keyphrases"][..]),
        ("summarize-corpus", &[][..]), ("answer", &[][..]),
        ("summarize-corpus", &["--synthesize-summary"][..]),
        ("summarize-corpus", &["--synthesize-summary", "--hierarchical-summary"][..]),
        ("answer", &["--synthesize-answer"][..]),
        ("answer", &["--synthesize-answer", "--hierarchical-answer"][..]),
    ] {
        for schedule in [&[][..], &["--cohort-chunks", "2"][..], &["--compute-threads", "3"][..],
            &["--cohort-chunks", "2", "--cohort-hierarchy-groups", "2"][..]] {
            if schedule.contains(&"--cohort-hierarchy-groups")
                && !mode.contains(&"--hierarchical-summary") && !mode.contains(&"--hierarchical-answer") { continue; }
            let mut ordinary = mode.to_vec(); ordinary.extend_from_slice(schedule);
            let (baseline, args) = arguments(name, &ordinary);
            let byte = baseline.validate(&args).unwrap();
            let mut flags = ordinary;
            flags.extend_from_slice(&["--input-encoding", "safe-bpe-v1",
                "--max-total-tokenizer-input-bytes", "123456",
                "--max-total-tokenizer-output-tokens", "234567"]);
            let (compact, mut args) = arguments(name, &flags);
            let admitted = compact.validate(&args).unwrap();
            let encoding = admitted.document.encoding.unwrap();
            assert_eq!(encoding.complete.max_input_bytes, 123456);
            assert_eq!(encoding.complete.max_output_tokens, 234567);
            if name == "answer" || mode.contains(&"--synthesize-summary") {
                assert_eq!(encoding.call.max_input_bytes,
                    args.max_input_bytes.min(crate::tokenizer::untrusted::SafeBpeLimits::MAX_INPUT_BYTES));
            }
            assert!(encoding.workspace_bytes().unwrap() < admitted.common.preparation_bytes);
            assert_eq!(admitted.transport, byte.transport);
            assert_eq!(admitted.run.max_model_work, byte.run.max_model_work);
            assert_eq!(admitted.run.max_mask_visits, byte.run.max_mask_visits);
            assert_eq!(admitted.run.max_verification, byte.run.max_verification);
            assert_eq!(admitted.document.mapping.chunks, byte.document.mapping.chunks);
            assert_eq!(admitted.document.mapping.reduction, byte.document.mapping.reduction);
            assert_eq!(admitted.document.workspace_bytes, byte.document.workspace_bytes);
            assert_eq!(admitted.document.cohort, byte.document.cohort);
            assert_eq!(admitted.document.hierarchy_cohorts, byte.document.hierarchy_cohorts);
            assert_eq!(admitted.common.compute, byte.common.compute);
            assert_eq!(admitted.common.minimum_memory_bytes, byte.common.minimum_memory_bytes);
            args.preparation_mib = 1;
            assert_eq!(compact.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
        }
    }
}

#[test]
fn compact_source_populations_refuse_invalid_work_before_private_io() {
    for (name, mode) in [
        ("ner-corpus", &[][..]), ("keyphrases-corpus", &[][..]),
        ("summarize-corpus", &[][..]), ("answer", &[][..]),
        ("summarize-corpus", &["--synthesize-summary"][..]),
        ("summarize-corpus", &["--synthesize-summary", "--hierarchical-summary"][..]),
        ("answer", &["--synthesize-answer"][..]),
        ("answer", &["--synthesize-answer", "--hierarchical-answer"][..]),
    ] {
        for extra in [
            &["--max-total-tokenizer-input-bytes", "1"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-total-tokenizer-pair-checks", "0"][..],
        ] {
            let mut flags = mode.to_vec(); flags.extend_from_slice(extra);
            let (options, args) = arguments(name, &flags);
            assert_eq!(options.execute_owned(args, NoIo, NoIo), Err(CandidateError::Arguments));
        }
    }
}
