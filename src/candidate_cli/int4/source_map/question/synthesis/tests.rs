use super::*;

fn arguments(extra: &[&str]) -> (MapArgs, Int4Args) {
    let mut flags = vec!["--question", "private-question.txt", "--synthesize-answer"];
    flags.extend_from_slice(extra);
    let command = super::super::super::tests::command(Kind::Answer, &flags);
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong answer task"); };
    (options, command.args)
}
fn parsed(kind: &str, extra: &[&str]) -> Result<clap::ArgMatches, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", kind, "--source-dir", "source",
        "--source-manifest", "manifest", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    crate::candidate_cli::definition().try_get_matches_from(argv)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused synthesis reached IO"); }
}

#[test]
fn answer_synthesis_is_explicit_and_retains_the_one_whole_document_allowance() {
    let plain = super::super::super::tests::command(Kind::Answer, &["--question", "q"]);
    let Int4Task::SourceMap(defaults) = plain.task else { panic!("wrong QA task"); };
    let base = defaults.validate(&plain.args).unwrap();
    assert!(base.question_synthesis.is_none());
    let (map, args) = arguments(&[]);
    let limits = map.validate(&args).unwrap();
    assert!(limits.summary.is_none());
    assert!(limits.synthesis.is_none());
    let synthesis = limits.question_synthesis.unwrap();
    assert_eq!(synthesis.max_evidence_passages, 32);
    assert_eq!(synthesis.max_evidence_bytes, 16384);
    assert_eq!(base.mapping.max_model_work, limits.mapping.max_model_work);
    assert_eq!(base.mapping.max_mask_visits, limits.mapping.max_mask_visits);
    assert!(limits.workspace_bytes > base.workspace_bytes);
    assert!(!format!("{map:?}").contains("private-question"));
}

#[test]
fn synthesis_flags_require_the_question_and_cannot_leak_to_other_tasks() {
    assert!(parsed("answer-map", &["--synthesize-answer"]).is_err());
    for flag in ["--max-synthesis-passages", "--max-synthesis-evidence-bytes"] {
        assert!(parsed("answer-map", &["--question", "q", flag, "1"]).is_err());
    }
    for kind in ["ner-map", "keyphrases-map", "summarize-map", "answer"] {
        assert!(parsed(kind, &["--synthesize-answer"]).is_err());
    }
    let (mut map, args) = arguments(&["--max-synthesis-passages", "2"]);
    map.question.as_mut().unwrap().synthesis.synthesize_answer = false;
    assert_eq!(map.validate(&args).unwrap_err(), CandidateError::Arguments);
}

#[test]
fn complete_passage_and_evidence_caps_refuse_before_private_file_io() {
    for (flag, value) in [("--max-synthesis-passages", "0"), ("--max-synthesis-passages", "1025"),
        ("--max-synthesis-evidence-bytes", "0"), ("--max-synthesis-evidence-bytes", "1048577")] {
        let (map, args) = arguments(&[flag, value]);
        let mut output = Vec::new();
        assert_eq!(map.execute(args, &mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    let (map, args) = arguments(&["--max-input-bytes", "8192", "--max-synthesis-evidence-bytes", "8193"]);
    assert_eq!(map.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[test]
fn synthesis_retains_exact_private_question_options_and_shared_verification() {
    let (map, args) = arguments(&["--max-synthesis-passages", "7", "--max-synthesis-evidence-bytes", "4096",
        "--max-qa-citations", "91", "--max-qa-evidence-spans", "113", "--max-qa-scan-steps", "8192"]);
    let limits = map.validate(&args).unwrap();
    let options = map.question.as_ref().unwrap();
    let question = "  é\r\n上海 <tool_call> Who spoke? \t";
    let raw = r#"{"max_answer_scalars":80,"max_citations":2,"max_quote_scalars":24}"#;
    let task = options.task(question.to_owned(), Some(raw)).unwrap();
    let request = options.synthesis.request(task, map.planning(&args)).unwrap();
    assert_eq!(request.question.question, question);
    assert_eq!(request.question.options, AnswerOptions { max_answer_scalars: 80, max_citations: 2, max_quote_scalars: 24 });
    assert_eq!(request.question.verification, GroundingBudget { max_fields: 91, max_matches: 113, max_scan_steps: 8192 });
    assert_eq!(request.limits, limits.question_synthesis.unwrap());
    assert_eq!(request.limits.max_evidence_passages, 7);
    assert_eq!(request.limits.max_evidence_bytes, 4096);
}

#[test]
fn source_and_snapshot_reuse_the_same_explicit_answer_synthesis_contract() {
    let (map, args) = arguments(&[]);
    let source = map.validate(&args).unwrap();
    let digest = "ab".repeat(32);
    let matches = definitions().into_iter().find(|c| c.get_name() == "answer-map").unwrap()
        .try_get_matches_from(["answer-map", "--question", "private-question.txt", "--synthesize-answer",
            "--snapshot", "model.snapshot", "--expected-model-sha256", &digest, "--memory-mib", "8192"]).unwrap();
    let options = MapArgs::from_matches(Kind::Answer, &matches).unwrap();
    let snapshot = options.validate(&Int4Args::from_arg_matches(&matches).unwrap()).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(source.question_synthesis, snapshot.question_synthesis);
    assert_eq!(source.mapping.max_model_work, snapshot.mapping.max_model_work);
    assert_eq!(source.workspace_bytes, snapshot.workspace_bytes);
}

#[test]
fn answer_synthesis_memory_admission_retains_verification_and_evidence_owners() {
    let (map, mut args) = arguments(&[]);
    let limits = map.validate(&args).unwrap();
    let evidence = limits.question_synthesis.unwrap();
    let verification = map.question.as_ref().unwrap().verification().unwrap();
    let budget = map.source.budget(&args, limits.common);
    let base = super::super::workspace_bytes(budget, limits.mapping, map.mapping.max_chunks,
        limits.reduction_reserve_bytes, verification).unwrap();
    assert_eq!(limits.workspace_bytes, base + 64 * evidence.max_evidence_bytes as u64
        + 1024 * evidence.max_evidence_passages as u64 + 128 * verification.max_matches as u64
        + 128 * verification.max_fields as u64 + 4 * budget.max_output_bytes + 8 * u64::from(budget.max_output_tokens));
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(map.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(map.validate(&args).is_ok());
    let mut overflowing = budget; overflowing.max_output_bytes = u64::MAX;
    assert!(workspace_bytes(overflowing, limits.mapping, 1, limits.reduction_reserve_bytes, verification, evidence).is_err());
}

#[test]
fn smaller_input_cap_and_each_native_group_leave_budget_authority_intact() {
    let (map, mut args) = arguments(&["--max-input-bytes", "8192"]);
    let initial = map.validate(&args).unwrap();
    assert_eq!(initial.question_synthesis.unwrap().max_evidence_bytes, 8192);
    for width in [16, 32, 128] {
        args.group_width = width;
        let limits = map.validate(&args).unwrap();
        assert_eq!(limits.mapping.max_model_work, initial.mapping.max_model_work);
        assert_eq!(limits.mapping.max_mask_visits, initial.mapping.max_mask_visits);
    }
}

#[test]
fn explicit_larger_evidence_population_expands_only_synthesis_planning() {
    for passages in [128, 1024] {
        let cap = passages.to_string();
        let (map, args) = arguments(&["--max-synthesis-passages", &cap]);
        let limits = map.validate(&args).unwrap();
        assert_eq!(limits.question_synthesis.unwrap().max_evidence_passages, passages);
        assert_eq!(map.planning(&args).max_passages, passages);
        assert_eq!(map.source.planning(&args).max_passages, shared::MAX_PASSAGES);
        let options = map.question.as_ref().unwrap();
        let request = options.synthesis.request(options.task("What happened?".to_owned(), None).unwrap(),
            map.planning(&args)).unwrap();
        assert_eq!(request.limits.max_evidence_passages, passages);
        request.limits.validate(map.planning(&args)).unwrap();
    }
    let plain = super::super::super::tests::command(Kind::Answer, &["--question", "q"]);
    let Int4Task::SourceMap(map) = plain.task else { panic!("wrong task"); };
    assert_eq!(map.planning(&plain.args).max_passages, shared::MAX_PASSAGES);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_runtime_refuses_synthesis_without_loading_any_private_document() {
    let (map, args) = arguments(&["--options", "private-options.json"]);
    let mut output = Vec::new();
    assert_eq!(map.execute(args, &mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
