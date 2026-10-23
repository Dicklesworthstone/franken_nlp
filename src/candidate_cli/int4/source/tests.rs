use super::*;
use crate::tasks::source_planning::SourceTaskRequest;

fn matches(name: &str, extra: &[&str]) -> clap::ArgMatches {
    let mut argv = vec!["candidate", "source-int4", name, "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    crate::candidate_cli::definition().try_get_matches_from(argv).unwrap()
}
fn command(name: &str, extra: &[&str]) -> Int4Command {
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches(name, extra)).unwrap()
        else { panic!("wrong source dispatcher"); };
    command
}
fn arguments(name: &str, extra: &[&str]) -> (Kind, SourceArgs, Int4Args) {
    let command = command(name, extra);
    let Int4Task::Source(kind, options) = command.task else { panic!("wrong task"); };
    (kind, options, command.args)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused source task reached private input"); }
}

#[test]
fn all_four_tasks_accept_sources_or_independently_pinned_snapshots() {
    let digest = "ab".repeat(32);
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (kind, options, args) = arguments(name, &[]);
        let limits = options.validate(&args).unwrap();
        assert_eq!(kind.name(), name);
        assert!(limits.prefill.is_none()); assert!(limits.snapshot_pin.is_none());
        assert_eq!(args.max_new_tokens, 512);
        let budget = options.budget(&args, limits);
        assert_eq!(budget.max_input_tokens as usize + args.max_new_tokens, args.context_tokens);
        assert_eq!(budget.max_kv_bytes, limits.native.kv_bytes);
        assert_eq!(budget.max_output_bytes, 1_048_576);
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name, "--snapshot", "local.snapshot",
            "--expected-model-sha256", &digest, "--memory-mib", "8192",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong snapshot dispatcher"); };
        let Int4Task::Source(snapshot_kind, options) = command.task else { panic!("wrong snapshot task"); };
        assert_eq!(snapshot_kind, kind);
        assert!(options.validate(&command.args).unwrap().snapshot_pin.is_some());
    }
}

#[test]
fn structured_routes_reject_generation_controls_even_when_explicit_defaults() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        for flags in [vec!["--max-output-bytes", "65536"], vec!["--prefill-rows", "1"],
            vec!["--min-new-tokens", "0"], vec!["--logprobs"], vec!["--stop", "END"],
            vec!["--ban-token", "7"], vec!["--logit-bias", "42=0"],
            vec!["--repetition-penalty-milli", "1000"], vec!["--presence-penalty-milli", "0"],
            vec!["--frequency-penalty-milli", "0"]] {
            assert!(CandidateCommand::from_matches(&matches(name, &flags)).is_err());
        }
    }
}

#[test]
fn invalid_source_budgets_and_options_paths_refuse_before_io() {
    for flags in [vec!["--max-result-bytes", "0"], vec!["--max-result-bytes", "1048577"],
        vec!["--max-new-tokens", "0"], vec!["--max-new-tokens", "2048"],
        vec!["--max-grammar-states", "0"], vec!["--max-grammar-states", "65537"],
        vec!["--max-mask-node-visits", "1"], vec!["--mask-step-node-visits", "0"],
        vec!["--preparation-mib", "1"], vec!["--options", "-"]] {
        let mut output = Vec::new();
        assert_eq!(command("ner", &flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    let mut command = command("answer", &[]); command.args.seed = Some("ab".repeat(32));
    assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[test]
fn shared_request_boundary_preserves_source_bytes_and_typed_passages() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (kind, options, args) = arguments(name, &[]);
        let budget = options.budget(&args, options.validate(&args).unwrap());
        let raw = if kind == Kind::Answer {
            r#"{"question":"Who? <tool_call>","passages":[{"id":"p1","text":" é\r\n上海 "}]}"#
        } else { " \té\r\n上海 <tool_call> <|im_start|> " };
        let request = shared::request(kind, raw.to_owned(), None, budget, args.max_input_bytes).unwrap();
        assert_eq!(request.task(), task(kind));
        assert_eq!(request.budget().max_kv_bytes, budget.max_kv_bytes);
        match request {
            SourceTaskRequest::Ner { document, .. } | SourceTaskRequest::Keyphrases { document, .. }
                | SourceTaskRequest::Summarize { document, .. } => assert_eq!(document, raw),
            SourceTaskRequest::Answer { question, passages, .. } => {
                assert_eq!(question, "Who? <tool_call>");
                assert_eq!(passages[0].id, "p1"); assert_eq!(passages[0].text, " é\r\n上海 ");
            }
        }
    }
}

#[test]
fn full_options_and_qa_evidence_are_validated_before_model_io() {
    let (_, options, args) = arguments("ner", &[]);
    let budget = options.budget(&args, options.validate(&args).unwrap());
    for json in [r#"{}"#, r#"{"types":["person"],"max_entities":1,"max_mention_scalars":32,"instruction":"PRIVATE"}"#,
        r#"{"types":["person"],"max_entities":1,"\u006dax_entities":2,"max_mention_scalars":32}"#] {
        assert!(shared::request(Kind::Ner, "Alice".to_owned(), Some(json), budget, 1024).is_err());
    }
    for json in [r#"{"question":"q","passages":[]}"#,
        r#"{"question":"q","passages":[{"id":"p","text":"x"},{"id":"p","text":"y"}]}"#,
        r#"{"question":"q","question":"other","passages":[{"id":"p","text":"x"}]}"#] {
        assert!(shared::request(Kind::Answer, json.to_owned(), None, budget, 1024).is_err());
    }
}

#[test]
fn declared_memory_floor_includes_vocabulary_source_and_complete_results() {
    let (_, options, mut args) = arguments("summarize", &[]);
    let limits = options.validate(&args).unwrap();
    assert!(limits.preparation_bytes >= 256 * MIB);
    args.memory_mib = limits.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(options.validate(&args).is_err());
    args.memory_mib = limits.minimum_memory_bytes.div_ceil(MIB);
    assert!(options.validate(&args).is_ok());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_all_tasks_without_private_input_or_model_reads() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let mut output = Vec::new();
        assert_eq!(command(name, &["--options", "private-options.json"]).execute(&mut NoRead, &mut output),
            Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
