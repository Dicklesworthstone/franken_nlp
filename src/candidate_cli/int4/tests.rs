use super::*;

mod scored_encoding;
mod structured_compute;

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
    assert_eq!(limits.compute, compute::Admission { threads: 1, workspace_bytes: 0 });
}

#[test]
fn fixed_compute_teams_are_opt_in_and_fully_priced_before_input() {
    for operation in ["generate", "chat"] {
        let mut serial = args(operation, &["--compute-threads", "1"]);
        let mut parallel = args(operation, &["--compute-threads", "4"]);
        for a in [&mut serial, &mut parallel] {
            a.source_dir = None; a.source_manifest = None;
            a.snapshot = Some(PathBuf::from("private-snapshot"));
            a.expected_model_sha256 = Some("39".repeat(32));
        }
        let serial_limits = serial.validate_text_compute().unwrap();
        let parallel_limits = parallel.validate_text_compute().unwrap();
        let extra = crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(4, 128).unwrap();
        assert_eq!(parallel_limits.compute, compute::Admission { threads: 4, workspace_bytes: extra });
        assert_eq!(serial_limits.native, parallel_limits.native);
        assert_eq!(serial_limits.load, parallel_limits.load);
        assert_eq!(parallel_limits.minimum_memory_bytes, serial_limits.minimum_memory_bytes + extra);
        assert!(serial.options(166_101).unwrap() == parallel.options(166_101).unwrap());
        // Shared batch validation must not accept a width it cannot use.
        assert!(parallel.validate().is_err());
        parallel.memory_mib = (parallel_limits.minimum_memory_bytes - 1) / MIB;
        assert!(parallel.validate_text_compute().is_err());
        parallel.memory_mib = parallel_limits.minimum_memory_bytes.div_ceil(MIB);
        parallel.validate_text_compute().unwrap();
    }
}

#[test]
fn unsupported_compute_widths_and_prompt_schedules_refuse_before_io() {
    for operation in ["generate", "chat"] {
        for flags in [
            &["--compute-threads", "0"][..], &["--compute-threads", "65"][..],
            &["--compute-threads", "2", "--prefill-rows", "1"][..],
            &["--compute-threads", "4", "--prefill-rows", "4"][..],
        ] {
            let mut output = Vec::new();
            assert_eq!(command(operation, flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
        args(operation, &["--compute-threads", "1", "--prefill-rows", "4"]).validate().unwrap();
    }
    // The option is absent from workflows without a persistent numerical team.
    for route in [vec!["text-batch"], vec!["text-corpus"],
        vec!["resolve"], vec!["resolve-corpus"], vec!["redact"]] {
        let mut argv = vec!["candidate", "source-int4"];
        argv.extend(route);
        argv.extend(["--source-dir", "private-source", "--source-manifest", "private-manifest",
            "--memory-mib", "8192", "--compute-threads", "2"]);
        assert_eq!(crate::candidate_cli::definition().try_get_matches_from(argv).unwrap_err().kind(),
            clap::error::ErrorKind::UnknownArgument);
    }
}

#[test]
fn explicit_safe_bpe_text_admits_workspace_and_unwired_tasks_stay_byte_only() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    for task in ["generate", "chat"] {
        let options = args(task, &["--input-encoding", "safe-bpe-v1"]);
        let limits = options.validate().unwrap();
        assert!(matches!(options.input_encoding.policy(options.max_input_bytes, limits.max_prompt_tokens).unwrap(),
            UntrustedEncoding::SafeBpeV1(_)));
        // One MiB of source requires >256 MiB of additional encoder payload;
        // the default 256-MiB preparation grant must fail before opening files.
        let oversized = command(task, &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576"]);
        assert_eq!(oversized.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
        let invalid = command(task, &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"]);
        assert_eq!(invalid.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
    // A task that still uses byte-counted chunk planning must never accept a
    // tokenization switch which its planner would silently ignore.
    for task in ["resolve", "redact"] {
        let argv = ["candidate", "source-int4", task, "--source-dir", "source",
            "--source-manifest", "manifest.json", "--memory-mib", "8192",
            "--input-encoding", "safe-bpe-v1"];
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv).is_err());
    }
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
    for task in ["generate", "chat", "classify", "sentiment", "judge", "resolve", "redact", "redact-document", "redact-corpus"] {
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
    for task in ["generate", "chat", "classify", "sentiment", "judge", "resolve", "redact", "redact-document", "redact-corpus"] {
        for kind in Int8KernelKind::ALL {
            let command = command(task, &["--kernel", kind.id()]);
            let admission = match &command.task {
                Int4Task::Text(_) => command.args.validate(),
                Int4Task::Classify(scoring) => scoring.validate(&command.args),
                Int4Task::ClassifyMap(map) => map.validate(&command.args).map(|limits| limits.common),
                Int4Task::Sentiment(sentiment) => sentiment.validate(&command.args),
                Int4Task::SentimentMap(map) => map.validate(&command.args).map(|limits| limits.common),
                Int4Task::Judge(judge) => judge.validate(&command.args),
                Int4Task::FaithfulnessMap(map) => map.validate(&command.args).map(|limits| limits.common),
                Int4Task::Resolve(resolve) => resolve.validate(&command.args),
                Int4Task::Redact(redact) => redact.validate(&command.args).map(|limits| limits.common),
                Int4Task::RedactDocument(redact) => redact.validate(&command.args).map(|limits| limits.common),
                Int4Task::RedactCorpus(redact) => redact.validate(&command.args).map(|limits| limits.common),
                Int4Task::Extract(extraction) => extraction.validate(&command.args),
                Int4Task::Semantic(semantic) => semantic.validate(&command.args),
                Int4Task::Source(_, source) => source.validate(&command.args),
                Int4Task::ExtractBatch(batch) => batch.validate(&command.args).map(|limits| limits.common),
                Int4Task::ExtractMap(map) => map.validate(&command.args).map(|limits| limits.common),
                Int4Task::SourceMap(map) => map.validate(&command.args).map(|limits| limits.common),
                Int4Task::SourceBatch(batch) => batch.validate(&command.args).map(|limits| limits.common),
                Int4Task::ScoredBatch(batch) => batch.validate(&command.args).map(|limits| limits.common),
                Int4Task::Stream(..) | Int4Task::ResolveCorpus(_) | Int4Task::ClassifyCorpus(_) | Int4Task::SentimentCorpus(_) | Int4Task::FaithfulnessCorpus(_)
                    | Int4Task::SourceCorpus(_) | Int4Task::AnswerCorpus(_) | Int4Task::ExtractCorpus(_) | Int4Task::TextJob | Int4Task::ExtractJob | Int4Task::ExtractDocumentJob | Int4Task::ScoredJob | Int4Task::ClassificationDocumentJob | Int4Task::SentimentDocumentJob | Int4Task::FaithfulnessDocumentJob | Int4Task::QuestionDocumentJob | Int4Task::SourceJob | Int4Task::DocumentJob | Int4Task::RedactJob => Err(CandidateError::Arguments),
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
    for task in ["generate", "chat", "classify", "sentiment", "judge", "resolve", "redact", "redact-document", "redact-corpus"] {
        let baseline = command(task, &[]);
        let limits = |command: &Int4Command| match &command.task {
            Int4Task::Text(_) => command.args.validate().unwrap(),
            Int4Task::Classify(scoring) => scoring.validate(&command.args).unwrap(),
            Int4Task::ClassifyMap(map) => map.validate(&command.args).unwrap().common,
            Int4Task::Sentiment(sentiment) => sentiment.validate(&command.args).unwrap(),
            Int4Task::SentimentMap(map) => map.validate(&command.args).unwrap().common,
            Int4Task::Judge(judge) => judge.validate(&command.args).unwrap(),
            Int4Task::FaithfulnessMap(map) => map.validate(&command.args).unwrap().common,
            Int4Task::Resolve(resolve) => resolve.validate(&command.args).unwrap(),
            Int4Task::Redact(redact) => redact.validate(&command.args).unwrap().common,
            Int4Task::RedactDocument(redact) => redact.validate(&command.args).unwrap().common,
            Int4Task::RedactCorpus(redact) => redact.validate(&command.args).unwrap().common,
            Int4Task::Extract(extraction) => extraction.validate(&command.args).unwrap(),
            Int4Task::Semantic(semantic) => semantic.validate(&command.args).unwrap(),
            Int4Task::Source(_, source) => source.validate(&command.args).unwrap(),
            Int4Task::ExtractBatch(batch) => batch.validate(&command.args).unwrap().common,
            Int4Task::ExtractMap(map) => map.validate(&command.args).unwrap().common,
            Int4Task::SourceMap(map) => map.validate(&command.args).unwrap().common,
            Int4Task::SourceBatch(batch) => batch.validate(&command.args).unwrap().common,
            Int4Task::ScoredBatch(batch) => batch.validate(&command.args).unwrap().common,
            Int4Task::Stream(..) | Int4Task::ResolveCorpus(_) | Int4Task::ClassifyCorpus(_) | Int4Task::SentimentCorpus(_) | Int4Task::FaithfulnessCorpus(_)
                | Int4Task::SourceCorpus(_) | Int4Task::AnswerCorpus(_) | Int4Task::ExtractCorpus(_) | Int4Task::TextJob | Int4Task::ExtractJob | Int4Task::ExtractDocumentJob | Int4Task::ScoredJob | Int4Task::ClassificationDocumentJob | Int4Task::SentimentDocumentJob | Int4Task::FaithfulnessDocumentJob | Int4Task::QuestionDocumentJob | Int4Task::SourceJob | Int4Task::DocumentJob | Int4Task::RedactJob => panic!("owned workflow is not a single-task kernel fixture"),
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
