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
