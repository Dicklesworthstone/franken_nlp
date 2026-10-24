use super::*;

fn arguments(extra: &[&str]) -> (MapArgs, Int4Args) {
    let mut flags = vec!["--question", "private-question.txt"];
    flags.extend_from_slice(extra);
    let command = super::super::tests::command(Kind::Answer, &flags);
    let Int4Task::SourceMap(map) = command.task else { panic!("wrong complete-document QA task"); };
    (map, command.args)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused QA reached input"); }
}

#[test]
fn complete_document_qa_requires_its_private_question_on_both_load_routes() {
    let (map, args) = arguments(&[]);
    let source = map.validate(&args).unwrap();
    assert!(map.question.is_some());
    assert!(map.summary.is_none());
    assert_eq!(args.max_input_bytes, 1_048_576);
    assert_eq!(map.question.as_ref().unwrap().path(), std::path::Path::new("private-question.txt"));
    assert!(!format!("{map:?}").contains("private-question"));
    let digest = "ab".repeat(32);
    let matches = definitions().into_iter().find(|c| c.get_name() == "answer-map").unwrap()
        .try_get_matches_from(["answer-map", "--question", "private-question.txt",
            "--snapshot", "model.snapshot", "--expected-model-sha256", &digest,
            "--memory-mib", "8192"]).unwrap();
    let snapshot = MapArgs::from_matches(Kind::Answer, &matches).unwrap()
        .validate(&Int4Args::from_arg_matches(&matches).unwrap()).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(snapshot.workspace_bytes, source.workspace_bytes);
    assert_eq!(snapshot.mapping.max_model_work, source.mapping.max_model_work);
    for flags in [vec![], vec!["--question", "q", "--question", "other"]] {
        let mut argv = vec!["answer-map", "--source-dir", "source", "--source-manifest", "manifest",
            "--memory-mib", "8192"];
        argv.extend(flags);
        assert!(definitions().into_iter().find(|c| c.get_name() == "answer-map").unwrap()
            .try_get_matches_from(argv).is_err());
    }
}

#[test]
fn private_question_and_complete_typed_options_retain_exact_unicode() {
    let (map, _) = arguments(&[]);
    let args = map.question.as_ref().unwrap();
    let question = "é\r\n上海 <tool_call> What happened?\n ";
    let raw = r#"{"max_answer_scalars":80,"max_citations":2,"max_quote_scalars":24}"#;
    let task = args.task(question.to_owned(), Some(raw)).unwrap();
    assert_eq!(task.question, question);
    assert_eq!(task.options, AnswerOptions { max_answer_scalars: 80, max_citations: 2, max_quote_scalars: 24 });
    assert_eq!(task.verification.max_fields, 4096);
    assert_eq!(task.verification.max_matches, 16384);
    for invalid in ["{", "{}", r#"{"max_answer_scalars":80,"max_citations":2}"#,
        r#"{"max_answer_scalars":80,"max_citations":2,"max_citations":3,"max_quote_scalars":24}"#,
        r#"{"max_answer_scalars":80,"max_citations":2,"max_quote_scalars":24,"question":"override"}"#] {
        assert!(args.task(question.to_owned(), Some(invalid)).is_err());
    }
    for invalid in ["".to_owned(), " \t\r\n".to_owned(), "x".repeat(QUESTION_BYTES + 1)] {
        assert!(args.task(invalid, None).is_err());
    }
    assert!(args.task(question.to_owned(), Some(&" ".repeat(shared::OPTIONS_BYTES + 1))).is_err());
}

#[test]
fn independent_complete_document_verification_bounds_refuse_before_io() {
    for flags in [vec!["--max-qa-citations", "0"], vec!["--max-qa-citations", "1000001"],
        vec!["--max-qa-evidence-spans", "0"], vec!["--max-qa-evidence-spans", "1000001"],
        vec!["--max-qa-scan-steps", "0"], vec!["--max-qa-scan-steps", "1000000000001"]] {
        let (map, args) = arguments(&flags);
        let mut output = Vec::new();
        assert_eq!(map.execute(args, &mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    for path in ["", "-"] {
        let (mut map, args) = arguments(&[]);
        map.question.as_mut().unwrap().question = PathBuf::from(path);
        assert_eq!(map.execute(args, &mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
    let (map, args) = arguments(&["--max-qa-citations", "7", "--max-source-fields", "11"]);
    assert_eq!(map.question.as_ref().unwrap().verification().unwrap().max_fields, 7);
    assert_eq!(map.validate(&args).unwrap().mapping.verification.max_fields, 11);
}

#[test]
fn qa_admission_prices_private_question_and_both_verifier_owners() {
    let (map, mut args) = arguments(&[]);
    let limits = map.validate(&args).unwrap();
    let budget = map.source.budget(&args, limits.common);
    let verification = map.question.as_ref().unwrap().verification().unwrap();
    let base = super::super::workspace_bytes(budget, limits.mapping,
        limits.mapping.chunks.max_chunks, limits.reduction_reserve_bytes).unwrap();
    assert!(limits.workspace_bytes > base);
    assert_eq!(limits.workspace_bytes, workspace_bytes(budget, limits.mapping,
        limits.mapping.chunks.max_chunks, limits.reduction_reserve_bytes, verification).unwrap());
    let single = map.source.validate(&args).unwrap();
    assert!(limits.common.minimum_memory_bytes > single.minimum_memory_bytes
        + single.preparation_bytes + limits.workspace_bytes + QUESTION_BYTES as u64);
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(map.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(map.validate(&args).is_ok());
    let mut overflow = budget; overflow.max_output_bytes = u64::MAX;
    assert!(workspace_bytes(overflow, limits.mapping, 2, 0, verification).is_err());
}

#[test]
fn question_controls_do_not_leak_into_other_source_tasks_or_free_generation() {
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize] {
        let name = format!("{}-map", kind.name());
        for flags in [vec!["--question", "q"], vec!["--max-qa-citations", "2"]] {
            let mut argv = vec![name.as_str(), "--source-dir", "source", "--source-manifest", "manifest",
                "--memory-mib", "8192"];
            argv.extend(flags);
            assert!(definitions().into_iter().find(|c| c.get_name() == name).unwrap()
                .try_get_matches_from(argv).is_err());
        }
    }
    for flags in [vec!["--prefill-rows", "1"], vec!["--min-new-tokens", "0"],
        vec!["--logit-bias", "42=0"], vec!["--seed", "ab"]] {
        let mut argv = vec!["candidate", "source-int4", "answer-map", "--question", "q",
            "--source-dir", "source", "--source-manifest", "manifest", "--memory-mib", "8192"];
        argv.extend(flags);
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
    }
    let (mut map, args) = arguments(&[]);
    map.kind = Kind::Ner;
    assert!(map.validate(&args).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_qa_runtime_never_opens_question_source_or_options() {
    let (map, args) = arguments(&["--options", "private-options.json"]);
    let mut output = Vec::new();
    assert_eq!(map.execute(args, &mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
