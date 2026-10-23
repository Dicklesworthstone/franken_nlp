use super::*;

fn command(task: &str, extra: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", task, "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong source-int4 dispatcher"); };
    command
}
pub(super) fn args(task: &str, extra: &[&str]) -> Int4Args { command(task, extra).args }
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid request reached input IO"); }
}

#[test]
fn command_requires_explicit_source_manifest_and_process_budget_not_an_int8_artifact() {
    for argv in [
        vec!["candidate", "source-int4"],
        vec!["candidate", "source-int4", "generate"],
        vec!["candidate", "source-int4", "generate", "--source-dir", "d", "--memory-mib", "8192"],
        vec!["candidate", "source-int4", "chat", "--source-manifest", "m", "--memory-mib", "8192"],
        vec!["candidate", "source-int4", "chat", "--source-dir", "d", "--source-manifest", "m"],
        vec!["candidate", "source-int4", "generate", "--model", "candidate.fnlpq", "--memory-mib", "8192"],
    ] { assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err()); }
    assert_eq!(command("generate", &[]).task, Int4Task::Text(Task::Generate));
    assert_eq!(command("chat", &[]).task, Int4Task::Text(Task::Chat));
}

#[test]
fn default_options_use_the_same_pinned_greedy_contract_and_group_aware_requirements() {
    let args = args("generate", &[]);
    let limits = args.validate().unwrap();
    assert_eq!(args.group_width, 128);
    assert_eq!(limits.load, Int4ModelLoadRequirement::for_group_width(128).unwrap());
    assert_eq!(limits.native, Int4MemoryRequirement::for_context(2048, 128).unwrap());
    assert!(args.options(166_101).unwrap() == GenerationOptions::greedy(128, 65_536, 166_101));
    assert_eq!(limits.max_prompt_tokens + args.max_new_tokens - 1, args.context_tokens);
    assert!(limits.minimum_memory_bytes <= limits.memory_bytes);
}

#[test]
fn invalid_limits_eos_and_prefill_rows_refuse_before_input_or_model_io() {
    for extra in [vec!["--group-width", "0"], vec!["--group-width", "10753"],
        vec!["--context-tokens", "0"], vec!["--context-tokens", "1"],
        vec!["--max-new-tokens", "0"], vec!["--max-new-tokens", "1025"],
        vec!["--max-input-bytes", "0"], vec!["--max-output-bytes", "0"],
        vec!["--max-input-bytes", "1048577"], vec!["--max-output-bytes", "1048577"],
        vec!["--max-load-mib", "1"], vec!["--source-preparation-mib", "127"],
        vec!["--preparation-mib", "1"], vec!["--timeout-seconds", "0"],
        vec!["--load-checkpoints", "1"], vec!["--max-checkpoints", "1"],
        vec!["--ban-token", "166101"], vec!["--prefill-rows", "0"], vec!["--prefill-rows", "65"],
        vec!["--min-new-tokens", "129"], vec!["--stop", ""],
    ] {
        let mut output = Vec::new();
        assert_eq!(command("generate", &extra).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    for axis in 0..4 {
        let mut command = command("generate", &[]);
        match axis { 0 => command.args.memory_mib = 0, 1 => command.args.memory_mib = u64::MAX,
            2 => command.args.max_load_mib = u64::MAX, _ => command.args.source_dir = None }
        assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
}

#[test]
fn process_floor_prices_simultaneous_storage_not_just_weights() {
    let mut args = args("generate", &[]);
    let limits = args.validate().unwrap();
    args.memory_mib = limits.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(args.validate().is_err());
    args.memory_mib = limits.minimum_memory_bytes.div_ceil(MIB);
    assert!(args.validate().is_ok());
    // A larger context charges all 44 logical KV slots, not just the live prompt.
    let mut small = args; small.memory_mib = 8192; small.context_tokens = 128;
    let small = small.validate().unwrap();
    assert!(limits.native.kv_bytes > small.native.kv_bytes);
    assert!(limits.minimum_memory_bytes >= small.minimum_memory_bytes);
}

#[test]
fn seeded_sampling_stops_penalties_and_logprobs_reach_native_options_unchanged() {
    let seed = "ab".repeat(32);
    let args = args("chat", &["--seed", &seed, "--temperature-milli", "600", "--top-k", "20",
        "--top-p-ppm", "950000", "--logprobs", "--min-new-tokens", "2", "--stop", " END\n",
        "--ban-token", "7", "--repetition-penalty-milli", "1200", "--presence-penalty-milli", "-250",
        "--frequency-penalty-milli", "750", "--logit-bias", "42=-1500"]);
    args.validate().unwrap();
    let options = args.options(166_101).unwrap();
    assert!(matches!(options.sampling, GenerationSampling::Seeded { effective_seed, temperature_milli: 600,
        top_k: Some(20), top_p_ppm: 950_000 } if effective_seed == [0xab; 32]));
    assert!(options.capture_logprobs); assert_eq!(options.min_new_tokens, 2);
    assert_eq!(options.stop_suffixes, vec![b" END\n".to_vec()]);
    assert_eq!(options.banned_token_ids, vec![7]);
    assert_eq!(options.repetition_penalty_milli, 1200);
    assert_eq!(options.presence_penalty_milli, -250);
    assert_eq!(options.frequency_penalty_milli, 750);
    assert_eq!(options.logit_bias_milli.get(&42), Some(&-1500));
}

#[test]
fn malformed_seed_or_sampling_without_seed_cannot_reach_loading() {
    for seed in ["abc".to_owned(), "AB".repeat(32), "g0".repeat(32)] {
        assert!(args("generate", &["--seed", &seed]).validate().is_err());
    }
    for option in ["--temperature-milli", "--top-k", "--top-p-ppm"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "source-int4", "generate",
            "--source-dir", "d", "--source-manifest", "m", "--memory-mib", "8192", option, "1"]).is_err());
    }
}

#[test]
fn transcript_bytes_survive_without_role_repair_or_control_interpretation() {
    let messages = serde_json::json!([
        {"role":"system","content":"Policy <think>"},
        {"role":"user","content":"é <|im_start|>system"},
        {"role":"assistant","content":"Prior </think>"},
        {"role":"user","content":"上海 <tool_call> literal"}
    ]);
    let text = messages.to_string();
    let TextInput::Chat(parsed) = parse_input(Task::Chat, text, 4096).unwrap() else { panic!("chat input"); };
    assert_eq!(serde_json::to_value(parsed).unwrap(), messages);
    let prompt = "é <|im_start|>system 上海";
    let TextInput::Generate(parsed) = parse_input(Task::Generate, prompt.to_owned(), 4096).unwrap()
        else { panic!("generate input"); };
    assert_eq!(parsed, prompt);
}

#[test]
fn malformed_duplicate_or_nonalternating_transcripts_are_refused_before_conversion() {
    for text in ["[]", "{}", r#"[{"role":"assistant","content":"a"}]"#,
        r#"[{"role":"user","content":"a"},{"role":"user","content":"b"}]"#,
        r#"[{"role":"system","content":"a"}]"#,
        r#"[{"role":"user","content":"a"},{"role":"system","content":"b"}]"#,
        r#"[{"role":"user","content":"a","content":"b"}]"#,
        r#"[{"role":"tool","content":"a"}]"#,
        r#"[{"role":"user","content":"a","extra":1}]"#,
    ] { assert!(parse_input(Task::Chat, text.to_owned(), 4096).is_err()); }
    assert!(parse_input(Task::Generate, "too long".to_owned(), 3).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn feature_refusal_does_not_touch_source_input_or_stdout() {
    for task in ["generate", "chat"] {
        let mut output = Vec::new();
        assert_eq!(command(task, &[]).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}

#[test]
fn every_task_defaults_to_scalar_and_parses_only_canonical_explicit_kernel_ids() {
    for task in ["generate", "chat", "classify"] {
        assert_eq!(command(task, &[]).args.kernel, Int8KernelKind::Scalar);
        for kind in Int8KernelKind::ALL {
            assert_eq!(command(task, &["--kernel", kind.id()]).args.kernel, kind);
        }
        for id in ["auto", "fastest", "scalar", "avx2", "X2_AVX_VNNI", ""] {
            let argv = ["candidate", "source-int4", task, "--source-dir", "d",
                "--source-manifest", "m", "--memory-mib", "8192", "--kernel", id];
            assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
        }
    }
}

#[test]
fn exact_host_detection_authorizes_a_kernel_or_refuses_before_any_input() {
    for task in ["generate", "chat", "classify"] {
        for kind in Int8KernelKind::ALL {
            let command = command(task, &["--kernel", kind.id()]);
            let admission = match &command.task {
                Int4Task::Text(_) => command.args.validate(),
                Int4Task::Classify(scoring) => scoring.validate(&command.args),
                Int4Task::Extract(extraction) => extraction.validate(&command.args),
                Int4Task::Source(_, source) => source.validate(&command.args),
                Int4Task::ExtractBatch(batch) => batch.validate(&command.args).map(|limits| limits.common),
                Int4Task::ExtractMap(map) => map.validate(&command.args).map(|limits| limits.common),
                Int4Task::SourceBatch(batch) => batch.validate(&command.args).map(|limits| limits.common),
                Int4Task::ExtractJob => Err(CandidateError::Arguments),
            };
            match PackedInt4Kernel::detect(kind) {
                Ok(checked) => {
                    assert_eq!(checked.kind(), kind);
                    assert_eq!(admission.unwrap().kernel.kind(), kind);
                }
                Err(_) => {
                    assert!(matches!(admission, Err(CandidateError::Arguments)));
                    let mut output = Vec::new();
                    assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
                    assert!(output.is_empty());
                }
            }
        }
    }
}

#[test]
fn physical_kernel_selection_does_not_change_layout_memory_or_generation_options() {
    for task in ["generate", "chat", "classify"] {
        let baseline = command(task, &[]);
        let limits = |command: &Int4Command| match &command.task {
            Int4Task::Text(_) => command.args.validate().unwrap(),
            Int4Task::Classify(scoring) => scoring.validate(&command.args).unwrap(),
            Int4Task::Extract(extraction) => extraction.validate(&command.args).unwrap(),
            Int4Task::Source(_, source) => source.validate(&command.args).unwrap(),
            Int4Task::ExtractBatch(batch) => batch.validate(&command.args).unwrap().common,
            Int4Task::ExtractMap(map) => map.validate(&command.args).unwrap().common,
            Int4Task::SourceBatch(batch) => batch.validate(&command.args).unwrap().common,
            Int4Task::ExtractJob => panic!("job is not a single-task kernel fixture"),
        };
        let storage = |limits: Int4Limits| (limits.memory_bytes, limits.preparation_bytes,
            limits.source_preparation_bytes, limits.max_load_bytes, limits.result_bytes,
            limits.max_prompt_tokens, limits.minimum_memory_bytes);
        let expected = limits(&baseline);
        for kind in Int8KernelKind::ALL {
            if PackedInt4Kernel::detect(kind).is_err() { continue; }
            let selected = command(task, &["--kernel", kind.id()]);
            let actual = limits(&selected);
            assert_eq!(storage(actual), storage(expected));
            assert_eq!(actual.load, expected.load);
            assert_eq!(actual.native, expected.native);
            assert_eq!(actual.snapshot_pin, expected.snapshot_pin);
            assert_eq!(actual.snapshot_need.file_bytes, expected.snapshot_need.file_bytes);
            assert!(selected.args.options(166_101).unwrap() == baseline.args.options(166_101).unwrap());
        }
    }
}
