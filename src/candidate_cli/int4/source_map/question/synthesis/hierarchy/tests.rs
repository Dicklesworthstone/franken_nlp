//! Parser and admission fixtures only; no native inference or quality evidence.
use super::*;

fn arguments(extra: &[&str]) -> (MapArgs, Int4Args) {
    let mut flags = vec!["--question", "private-question.txt", "--synthesize-answer", "--hierarchical-answer"];
    flags.extend_from_slice(extra);
    let command = super::super::super::super::tests::command(Kind::Answer, &flags);
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong answer hierarchy"); };
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
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused answer hierarchy reached input"); }
}

#[test]
fn hierarchy_requires_explicit_answer_synthesis_and_child_controls_require_hierarchy() {
    assert!(parsed("answer-map", &["--question", "q", "--hierarchical-answer"]).is_err());
    assert!(parsed("answer-map", &["--synthesize-answer", "--hierarchical-answer"]).is_err());
    for flag in ["--answer-max-levels", "--answer-max-passes", "--answer-max-tokenizer-calls",
        "--answer-max-tokenizer-bytes"] {
        assert!(parsed("answer-map", &["--question", "q", "--synthesize-answer", flag, "1"]).is_err());
    }
    for kind in ["ner-map", "keyphrases-map", "summarize-map", "answer"] {
        assert!(parsed(kind, &["--question", "q", "--synthesize-answer", "--hierarchical-answer"]).is_err());
    }
    let (mut options, args) = arguments(&[]);
    options.question.as_mut().unwrap().synthesis.synthesize_answer = false;
    assert_eq!(options.validate(&args).unwrap_err(), CandidateError::Arguments);
    let (mut options, args) = arguments(&["--answer-max-passes", "3"]);
    options.question.as_mut().unwrap().synthesis.hierarchy.hierarchical_answer = false;
    assert_eq!(options.validate(&args).unwrap_err(), CandidateError::Arguments);
}

#[test]
fn independent_single_pass_and_hierarchical_answers_share_one_finite_work_ceiling() {
    let plain = super::super::super::super::tests::command(Kind::Answer, &["--question", "q"]);
    let single = super::super::super::super::tests::command(Kind::Answer, &["--question", "q", "--synthesize-answer"]);
    let Int4Task::SourceMap(plain_options) = plain.task else { panic!("wrong default QA"); };
    let Int4Task::SourceMap(single_options) = single.task else { panic!("wrong synthesis"); };
    let independent = plain_options.validate(&plain.args).unwrap();
    let single = single_options.validate(&single.args).unwrap();
    let (options, args) = arguments(&[]);
    let hierarchy = options.validate(&args).unwrap();
    assert!(independent.question_synthesis.is_none()); assert!(independent.question_hierarchy.is_none());
    assert!(single.question_synthesis.is_some()); assert!(single.question_hierarchy.is_none());
    assert_eq!(hierarchy.question_synthesis, single.question_synthesis);
    assert_eq!(hierarchy.question_hierarchy, Some(QuestionHierarchyLimits::default()));
    assert!(hierarchy.summary.is_none()); assert!(hierarchy.synthesis.is_none()); assert!(hierarchy.hierarchy.is_none());
    assert_eq!(hierarchy.mapping.max_model_work, independent.mapping.max_model_work);
    assert_eq!(hierarchy.mapping.max_model_work, single.mapping.max_model_work);
    assert_eq!(hierarchy.mapping.max_mask_visits, single.mapping.max_mask_visits);
    assert!(hierarchy.workspace_bytes > single.workspace_bytes);
    assert!(!format!("{options:?}").contains("private-question"));
}

#[test]
fn invalid_hierarchy_controls_refuse_before_private_question_source_or_model_io() {
    for (flag, value) in [("--answer-max-levels", "0"), ("--answer-max-levels", "17"),
        ("--answer-max-passes", "0"), ("--answer-max-passes", "257"),
        ("--answer-max-tokenizer-calls", "0"), ("--answer-max-tokenizer-calls", "65537"),
        ("--answer-max-tokenizer-bytes", "0"), ("--answer-max-tokenizer-bytes", "1000000001")] {
        let (options, args) = arguments(&[flag, value]);
        let mut output = Vec::new();
        assert_eq!(options.execute(args, &mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    assert!(parsed("answer-map", &["--question", "q", "--synthesize-answer", "--hierarchical-answer",
        "--summary-max-passes", "3"]).is_err());
}

#[test]
fn hierarchy_retains_exact_private_question_options_and_shared_evidence_limits() {
    let (options, args) = arguments(&["--answer-max-levels", "3", "--answer-max-passes", "7",
        "--answer-max-tokenizer-calls", "123", "--answer-max-tokenizer-bytes", "456789",
        "--max-synthesis-passages", "256", "--max-synthesis-evidence-bytes", "65536",
        "--max-qa-citations", "91", "--max-qa-evidence-spans", "113", "--max-qa-scan-steps", "8192"]);
    let limits = options.validate(&args).unwrap();
    assert_eq!(limits.question_hierarchy.unwrap(), QuestionHierarchyLimits { max_levels: 3, max_passes: 7,
        max_tokenizer_calls: 123, max_tokenizer_bytes: 456789 });
    let question = options.question.as_ref().unwrap();
    let text = "  é\r\n上海 <tool_call> Who spoke? \t";
    let raw = r#"{"max_answer_scalars":80,"max_citations":2,"max_quote_scalars":24}"#;
    let request = question.synthesis.request(question.task(text.to_owned(), Some(raw)).unwrap(), options.planning(&args)).unwrap();
    assert_eq!(request.question.question, text);
    assert_eq!(request.question.options, AnswerOptions { max_answer_scalars: 80, max_citations: 2, max_quote_scalars: 24 });
    assert_eq!(request.question.verification, GroundingBudget { max_fields: 91, max_matches: 113, max_scan_steps: 8192 });
    assert_eq!(request.limits, limits.question_synthesis.unwrap());
    assert_eq!(request.limits.max_evidence_passages, 256);
    assert_eq!(request.limits.max_evidence_bytes, 65536);
    assert_eq!(options.planning(&args).max_passages, 256);
}

#[test]
fn hierarchy_workspace_and_preparation_price_every_permitted_answer_pass() {
    let (options, mut args) = arguments(&["--answer-max-passes", "4", "--answer-max-levels", "2"]);
    let limits = options.validate(&args).unwrap();
    let synthesis = limits.question_synthesis.unwrap();
    let hierarchy = limits.question_hierarchy.unwrap();
    let verification = options.question.as_ref().unwrap().verification().unwrap();
    let budget = options.source.budget(&args, limits.common);
    let single = super::super::workspace_bytes(budget, limits.mapping, options.mapping.max_chunks,
        limits.reduction_reserve_bytes, verification, synthesis).unwrap();
    let per_pass = 4 * budget.max_output_bytes + 8 * u64::from(budget.max_output_tokens);
    assert_eq!(limits.workspace_bytes, single + 3 * per_pass + 4 * (8192 + 256) + 2 * 128);
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(options.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(options.validate(&args).is_ok());
    let mut invalid = hierarchy; invalid.max_passes = 0;
    assert!(workspace_bytes(budget, limits.mapping, 1, limits.reduction_reserve_bytes, verification, synthesis, invalid).is_err());
    let mut overflowing = budget; overflowing.max_output_bytes = u64::MAX;
    assert!(workspace_bytes(overflowing, limits.mapping, 1, limits.reduction_reserve_bytes, verification, synthesis, hierarchy).is_err());
    // 512 MiB can prepare this default single pass but cannot retain authority
    // for all sixteen permitted hierarchy passes and their private questions.
    let single = super::super::super::super::tests::command(Kind::Answer,
        &["--question", "q", "--synthesize-answer", "--preparation-mib", "512"]);
    let Int4Task::SourceMap(single_options) = single.task else { panic!("wrong synthesis"); };
    single_options.validate(&single.args).unwrap();
    let (options, args) = arguments(&["--preparation-mib", "512"]);
    assert_eq!(options.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[test]
fn source_and_pinned_snapshot_routes_keep_the_same_explicit_answer_hierarchy() {
    let (options, args) = arguments(&[]);
    let source = options.validate(&args).unwrap();
    let digest = "ab".repeat(32);
    let matches = definitions().into_iter().find(|c| c.get_name() == "answer-map").unwrap()
        .try_get_matches_from(["answer-map", "--question", "private-question.txt", "--synthesize-answer", "--hierarchical-answer",
            "--snapshot", "model.snapshot", "--expected-model-sha256", &digest, "--memory-mib", "8192"]).unwrap();
    let options = MapArgs::from_matches(Kind::Answer, &matches).unwrap();
    let snapshot = options.validate(&Int4Args::from_arg_matches(&matches).unwrap()).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(snapshot.question_hierarchy, source.question_hierarchy);
    assert_eq!(snapshot.question_synthesis, source.question_synthesis);
    assert_eq!(snapshot.mapping.max_model_work, source.mapping.max_model_work);
    assert_eq!(snapshot.workspace_bytes, source.workspace_bytes);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_runtime_refuses_answer_hierarchy_before_private_input_or_model_io() {
    let (options, args) = arguments(&["--options", "private-options.json"]);
    let mut output = Vec::new();
    assert_eq!(options.execute(args, &mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
