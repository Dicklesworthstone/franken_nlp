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

#[test]
fn compact_source_flags_are_explicit_and_workspace_is_admitted_before_io() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let cmd = command(task, &["--input-encoding", "safe-bpe-v1",
            "--max-tokenizer-pair-checks", "1234", "--max-tokenizer-pair-bytes", "10000"]);
        let (args, limits) = cmd.args.host.common_with_encoding(cmd.args.input.clone(), &cmd.encoding).unwrap();
        let UntrustedEncoding::SafeBpeV1(policy) = args.input_encoding_policy(limits).unwrap()
            else { panic!("compact source policy was discarded"); };
        assert_eq!(policy.max_pair_checks, 1234); assert_eq!(policy.max_pair_bytes, 10000);
        assert_eq!(policy.max_input_bytes, cmd.args.host.max_input_bytes);
        assert!(limits.preparation_bytes >= 256 * MIB + policy.requirements().unwrap().workspace_bytes);
        let byte = command(task, &[]);
        let (args, limits) = byte.args.host.common_with_encoding(byte.args.input.clone(), &byte.encoding).unwrap();
        assert_eq!(args.input_encoding_policy(limits).unwrap(), UntrustedEncoding::ByteFallback);
    }
    // The extra 1-MiB-source encoder cannot hide behind the existing 512-MiB
    // source preparation floor. A larger explicit reserve is accepted.
    let cmd = command("ner", &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576"]);
    assert!(cmd.args.host.common_with_encoding(cmd.args.input.clone(), &cmd.encoding).is_err());
    let cmd = command("ner", &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576",
        "--preparation-mib", "1024"]);
    cmd.args.host.common_with_encoding(cmd.args.input.clone(), &cmd.encoding).unwrap();
}

#[test]
fn all_four_one_request_routes_admit_explicit_widths_without_changing_source_policy() {
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let serial = command(task, &[]);
        let (_, baseline) = serial.validate().unwrap();
        assert_eq!(baseline.compute.threads, 1); assert_eq!(baseline.compute.workspace_bytes, 0);
        for width in ["1", "2", "7", "64"] {
            let cmd = command(task, &["--compute-threads", width, "--selected-rows", "32",
                "--input-encoding", "safe-bpe-v1"]);
            let (common, limits) = cmd.validate().unwrap();
            assert_eq!(limits.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(common.compute, cmd.args.compute);
            assert_eq!(limits.kv_bytes, baseline.kv_bytes);
            assert_eq!(limits.preparation_bytes, baseline.preparation_bytes);
            assert_eq!(cmd.args.host.task_budget(limits), serial.args.host.task_budget(baseline));
            assert_eq!(cmd.args.head.limits().unwrap().unwrap().max_rows_per_step, 32);
            assert!(matches!(common.input_encoding_policy(limits).unwrap(),
                crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(_)));
        }
    }
}

#[test]
fn source_team_width_prefill_and_aggregate_memory_refuse_before_all_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("source team read private input") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("source team wrote output") }
        fn flush(&mut self) -> io::Result<()> { panic!("source team flushed output") }
    }
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        for extra in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
            vec!["--compute-threads", "2", "--prefill-rows", "0"],
            vec!["--compute-threads", "7", "--prefill-rows", "65"],
            vec!["--compute-threads", "2", "--max-weight-mib", "8192"]] {
            let mut flags = extra; flags.extend(["--options", "must-not-open-team-options.json"]);
            let cmd = command(task, &flags);
            assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
        let mut cmd = command(task, &["--compute-threads", "3", "--options", "must-not-open-team-options.json"]);
        let (common, limits) = cmd.validate().unwrap();
        let required = compute::source_peak_bytes(&common, limits, cmd.args.host.max_result_bytes, None).unwrap();
        // Even a nearly sufficient aggregate ceiling must refuse before
        // document, options, model, vocabulary or output allocations begin.
        cmd.args.host.memory_mib = (required - 1) / MIB;
        assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        command(task, &["--compute-threads", "1", "--prefill-rows", "4"]).validate().unwrap();
    }
}

#[test]
fn route_local_source_teams_do_not_enable_maps_corpora_jobs_or_extraction() {
    for route in ["map", "batch"] {
        let base = ["candidate", route, "--task", "ner", "--model", "m", "--memory-mib", "8192"];
        assert!(super::super::definition().try_get_matches_from(base).is_ok());
        let mut argv = base.to_vec(); argv.extend(["--compute-threads", "2"]);
        assert_eq!(super::super::definition().try_get_matches_from(argv).unwrap_err().kind(),
            clap::error::ErrorKind::UnknownArgument);
    }
    for mode in ["start", "resume"] {
        let mut argv = vec!["candidate", "job", mode, "--task", "ner", "--model", "m",
            "--memory-mib", "8192", "--job-dir", "private/job", "--key-file", "private/key",
            "--limits", "limits.json", "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
        assert!(super::super::definition().try_get_matches_from(&argv).is_ok());
        argv.extend(["--compute-threads", "2"]);
        assert_eq!(super::super::definition().try_get_matches_from(argv).unwrap_err().kind(),
            clap::error::ErrorKind::UnknownArgument);
    }
    assert_eq!(super::super::definition().try_get_matches_from(["candidate", "extract", "--model", "m",
        "--memory-mib", "8192", "--schema", "schema.json", "--compute-threads", "2"])
        .unwrap_err().kind(), clap::error::ErrorKind::UnknownArgument);
    // All other consumers receive the shared host's default, even when a
    // one-shot SourceCommand separately selected a team.
    let cmd = command("ner", &["--compute-threads", "7"]);
    let (common, limits) = cmd.args.host.common(cmd.args.input.clone()).unwrap();
    assert_eq!(common.compute.threads(), 1); assert_eq!(limits.compute.threads, 1);
}

#[test]
fn invalid_compact_source_limits_never_read_options_document_or_model() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("source output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("source output flushed") }
    }
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
            let cmd = command(task, &["--options", "must-not-open.json", "--input-encoding", "safe-bpe-v1", flag, "0"]);
            assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
            let matches = super::super::definition().try_get_matches_from(["candidate", task,
                "--model", "must-not-open.fnlpq", "--memory-mib", "8192", flag, "1"]).unwrap();
            assert!(CandidateCommand::from_matches(&matches).is_err());
        }
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn default_build_refuses_all_source_commands_without_reading_any_input() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input read") } }
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        for width in ["1", "3"] {
            let cmd = command(name, &["--options", "never-open-private-options.json",
                "--compute-threads", width, "--prefill-rows", "4"]);
            let mut output = Vec::new();
            assert_eq!(cmd.execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
            assert!(output.is_empty());
        }
    }
}

#[test]
fn source_prompt_morsels_and_workers_compose_without_inheriting_generation_policy() {
    use crate::native_engine::strict_int8::projection::int8_team_workspace_bytes_with_prefill;
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let serial = command(task, &["--input-encoding", "safe-bpe-v1", "--selected-rows", "32"]);
        let (_, baseline) = serial.validate().unwrap();
        for width in [1_usize, 2, 7, 64] {
            for rows in [1_usize, 4, 64] {
                let width_text = width.to_string(); let rows_text = rows.to_string();
                let mut cmd = command(task, &["--compute-threads", &width_text, "--prefill-rows", &rows_text,
                    "--input-encoding", "safe-bpe-v1", "--selected-rows", "32"]);
                cmd.args.host.memory_mib = 16384;
                let (common, limits) = cmd.validate().unwrap();
                let prefill = cmd.args.prefill.limits().unwrap().unwrap();
                assert_eq!(limits.compute.threads, width);
                assert_eq!(limits.compute.workspace_bytes, if width == 1 { 0 }
                    else { int8_team_workspace_bytes_with_prefill(width, rows).unwrap() });
                assert_eq!(prefill.max_batch_rows, rows);
                assert!(common.policy.prefill().unwrap().is_none());
                assert_eq!(common.compute, cmd.args.compute);
                assert_eq!(cmd.args.host.task_budget(limits), serial.args.host.task_budget(baseline));
                assert_eq!(cmd.args.head.limits().unwrap().unwrap().max_rows_per_step, 32);
                assert_eq!(limits.preparation_bytes, baseline.preparation_bytes);
                assert!(matches!(common.input_encoding_policy(limits).unwrap(),
                    crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(_)));
            }
        }
    }
}

#[test]
fn composed_source_memory_refuses_before_document_options_or_model_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("source input read"); } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("source output written"); }
        fn flush(&mut self) -> io::Result<()> { panic!("source output flushed"); }
    }
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        for width in ["1", "4", "64"] {
            let mut cmd = command(task, &["--compute-threads", width, "--prefill-rows", "64",
                "--options", "must-not-open-source-options.json"]);
            cmd.args.host.memory_mib = 16384;
            let (common, limits) = cmd.validate().unwrap();
            let prefill = cmd.args.prefill.limits().unwrap();
            let required = compute::source_peak_bytes(&common, limits,
                cmd.args.host.max_result_bytes, prefill).unwrap();
            cmd.args.host.memory_mib = (required - 1) / MIB;
            assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
    }
}
