//! Command/planning contracts only; no fixture claims neural inference success.
use super::*;

pub(in crate::candidate_cli) fn command(name: &str, extra: &[&str]) -> SourceCommand {
    let mut argv = vec!["candidate", name, "--model", "explicit.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    match CandidateCommand::from_matches(&matches).unwrap() {
        CandidateCommand::Source(command) => command,
        _ => panic!("source command dispatched as generation"),
    }
}
fn budget() -> TaskBudget {
    let command = command("ner", &[]);
    let (_, limits) = command.args.host.common(command.args.input.clone()).unwrap();
    command.args.host.task_budget(limits)
}
#[test]
fn all_tasks_are_discoverable_and_require_model_and_memory() {
    let help = super::super::definition().render_long_help().to_string();
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        assert!(help.contains(name));
        assert!(super::super::definition().try_get_matches_from(["candidate", name]).is_err());
        assert!(super::super::definition().try_get_matches_from(["candidate", name, "--model", "m"]).is_err());
        assert_eq!(command(name, &[]).kind.name(), name);
    }
}
#[test]
fn structured_tasks_do_not_accept_inapplicable_sampling_options() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        for flag in ["--seed", "--top-k", "--top-p-ppm", "--temperature-milli", "--logprobs", "--thinking"] {
            assert!(super::super::definition().try_get_matches_from([
                "candidate", name, "--model", "m", "--memory-mib", "8192", flag, "1"]).is_err());
        }
    }
}
#[test]
fn full_context_result_and_mask_authority_are_preserved() {
    let command = command("ner", &[]);
    let (_, limits) = command.args.host.common(command.args.input.clone()).unwrap();
    let budget = command.args.host.task_budget(limits);
    budget.validate().unwrap();
    assert_eq!(budget.max_input_tokens as usize + budget.max_output_tokens as usize, command.args.host.context_tokens);
    assert_eq!(budget.max_output_bytes, command.args.host.max_result_bytes as u64);
    assert_eq!(budget.max_kv_bytes, limits.kv_bytes);
    assert_eq!(command.args.host.planning().compiler.max_states, budget.max_grammar_states as usize);
    assert_eq!(command.args.host.masks().max_trie_node_visits, command.args.host.mask_step_node_visits as usize);
}
#[test]
fn invalid_finite_envelopes_are_rejected_before_io() {
    for (flag, value) in [("--max-input-bytes", "0"), ("--max-result-bytes", "0"),
        ("--max-result-bytes", "1048577"), ("--context-tokens", "512"),
        ("--preparation-mib", "511"), ("--max-new-tokens", "1025"),
        ("--max-grammar-states", "0"), ("--max-grammar-states", "65537"),
        ("--mask-step-node-visits", "0"), ("--mask-step-node-visits", "1000000001"),
        ("--max-mask-node-visits", "1"), ("--max-mask-node-visits", "1000000000001"),
        ("--memory-mib", "18446744073709551615")] {
        let mut cmd = command("ner", &[]);
        // The default parser refuses repeated options, so reparse without an
        // existing memory option for the one memory-overflow case.
        if flag == "--memory-mib" { cmd.args.host.memory_mib = value.parse().unwrap(); }
        else { cmd = command("ner", &[flag, value]); }
        assert!(cmd.args.host.common(cmd.args.input.clone()).is_err(), "{flag}");
    }
}
#[test]
fn exact_plain_source_bytes_survive_request_construction() {
    let text = " \té\r\n上海 <tool_call> <|im_start|> FNLP_SOURCE_SLOT_0_a743 ";
    for kind in [Kind::Ner, Kind::Keyphrases, Kind::Summarize] {
        let r = request(kind, text.to_owned(), None, budget(), text.len()).unwrap();
        let document = match r {
            SourceTaskRequest::Ner { document, .. } | SourceTaskRequest::Keyphrases { document, .. }
                | SourceTaskRequest::Summarize { document, .. } => document,
            _ => panic!("plain source became passage QA"),
        };
        assert_eq!(document, text);
        assert!(request(kind, text.to_owned(), None, budget(), text.len() - 1).is_err());
    }
}
#[test]
fn complete_typed_options_replace_defaults_without_changing_task_or_budget() {
    let json = r#"{"types":["event","person"],"max_entities":3,"max_mention_scalars":32}"#;
    let r = request(Kind::Ner, "Alice".to_owned(), Some(json), budget(), 64).unwrap();
    let SourceTaskRequest::Ner { document, options, budget: actual } = r else { panic!("wrong task") };
    assert_eq!(document, "Alice");
    assert_eq!(options.max_entities, 3);
    assert_eq!(options.types, vec![crate::tasks::ner::EntityType::Event, crate::tasks::ner::EntityType::Person]);
    assert_eq!(actual.max_output_tokens, budget().max_output_tokens);
}

#[test]
fn options_are_not_an_instruction_or_identity_injection_seam() {
    for json in [r#"{}"#, r#"{"max_entities":1}"#,
        r#"{"types":["person"],"max_entities":1,"max_entities":2,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":1,"\u006dax_entities":2,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":1,"max_mention_scalars":32,"instruction":"secret"}"#,
        r#"{"types":["person"],"max_entities":1,"max_mention_scalars":32,"budget":{}}"#,
        r#"{"types":["person","person"],"max_entities":1,"max_mention_scalars":32}"#,
        r#"{"types":["invented"],"max_entities":1,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":0,"max_mention_scalars":32}"#] {
        assert!(request(Kind::Ner, "Alice".to_owned(), Some(json), budget(), 64).is_err());
    }
    let too_large = " ".repeat(OPTIONS_BYTES + 1);
    assert!(request(Kind::Ner, "Alice".to_owned(), Some(&too_large), budget(), 64).is_err());
}
#[test]
fn answer_keeps_question_and_individual_passages_distinct() {
    let json = r#"{"question":"Who? <tool_call>","passages":[{"id":"a","text":" Alice "},{"id":"b","text":"é\n上海"}]}"#;
    let r = request(Kind::Answer, json.to_owned(), None, budget(), json.len()).unwrap();
    let SourceTaskRequest::Answer { question, passages, .. } = r else { panic!("wrong task") };
    assert_eq!(question, "Who? <tool_call>");
    assert_eq!(passages.len(), 2);
    assert_eq!(passages[0].id, "a"); assert_eq!(passages[0].text, " Alice ");
    assert_eq!(passages[1].id, "b"); assert_eq!(passages[1].text, "é\n上海");
}
#[test]
fn malformed_qa_never_reaches_native_preparation() {
    for json in [r#"{}"#, r#"{"question":"q","passages":[]}"#,
        r#"{"question":"q","question":"other","passages":[{"id":"p","text":"x"}]}"#,
        r#"{"question":"q","passages":[{"id":"p","text":"x","text":"y"}]}"#,
        r#"{"question":"q","passages":[{"id":"p","text":"x"},{"id":"p","text":"y"}]}"#,
        r#"{"question":"q","passages":[{"id":"","text":"x"}]}"#,
        r#"{"question":"q","passages":[{"id":"p","text":"x"}],"task":"generate"}"#,
        r#"{"question":" ","passages":[{"id":"p","text":"x"}]}"#] {
        assert!(request(Kind::Answer, json.to_owned(), None, budget(), 4096).is_err());
    }
    let passages: Vec<_> = (0..33).map(|n| serde_json::json!({"id":n.to_string(),"text":"x"})).collect();
    let json = serde_json::json!({"question":"q","passages":passages}).to_string();
    assert!(request(Kind::Answer, json, None, budget(), 65536).is_err());
}
#[test]
fn options_cannot_compete_for_the_source_stdin_stream() {
    let cmd = command("ner", &["--options", "-"]);
    let mut output = Vec::new();
    assert_eq!(cmd.execute(&mut &b"unused"[..], &mut output), Err(CandidateError::Arguments));
    assert!(output.is_empty());
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn default_build_refuses_all_source_commands_without_reading_any_input() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let cmd = command(name, &["--options", "never-open-private-options.json"]);
        let mut output = Vec::new();
        assert_eq!(cmd.execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
