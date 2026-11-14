//! Parser and finite policy admission only; no native execution evidence.
use super::*;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{ir::TaskBudget, judge::FaithfulnessPolicy,
        source_planning::grouped_int4::semantic::SourceSemanticLimits},
    tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};

fn argv(task: &str, extra: &[&str]) -> Vec<String> {
    ["candidate", "source-int4", task, "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192",
        "--semantic-options", "semantic.json"].into_iter().chain(extra.iter().copied()).map(str::to_owned).collect()
}
pub(super) fn command(task: &str, extra: &[&str]) -> Int4Command {
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(task, extra)).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate route"); };
    command
}
pub(super) fn fixture(kind: Kind) -> (SemanticSourceArgs, Int4Args, Int4Limits, Int4SourceSemanticOptions) {
    let command = command(match kind { Kind::Summarize => "summarize-semantic", Kind::Answer => "answer-semantic",
        _ => panic!("summary or answer fixture required") }, &[]);
    let Int4Task::SourceSemantic(actual, semantic) = command.task else { panic!("wrong semantic task"); };
    assert_eq!(actual, kind);
    let args = command.args; let common = semantic.validate(&args).unwrap();
    let options = Int4SourceSemanticOptions { schema_version: 1, experimental_opt_in: true,
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 8192, max_evidence_windows: 8, max_evidence_spans: 8 },
        judge_budget: TaskBudget { max_input_tokens: 1985, max_output_tokens: 64,
            max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: common.native.kv_bytes },
        limits: SourceSemanticLimits { max_statements: 8, max_source_bytes: args.max_input_bytes,
            max_claim_bytes: 4096, max_total_claim_bytes: 16_384, max_total_prompt_tokens: 8192,
            max_total_projected_logits: 100_000_000, max_output_bytes: 4 << 20 },
        max_model_work: Int4Work::for_sequence(args.group_width, 0, 32_768, 2048 * NANBEIGE_VOCAB_SIZE).unwrap() };
    (semantic, args, common, options)
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused semantic task read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused semantic task wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused semantic task flushed output"); }
}

#[test]
fn each_explicit_task_supports_both_model_inputs_without_enabling_review_on_ordinary_tasks() {
    for kind in [Kind::Summarize, Kind::Answer] {
        let (semantic, args, common, options) = fixture(kind);
        let admitted = semantic.admit_options(&args, common, &options).unwrap();
        assert_eq!(admitted.result_bytes, 4 << 20);
        assert_eq!(semantic.source.budget(&args, admitted), semantic.source.budget(&args, common));
        assert_eq!(semantic.source.budget(&args, admitted).max_output_bytes, 1 << 20);
        assert!(admitted.prefill.is_none()); assert_eq!(admitted.compute.threads, 1);
        let pin = "ab".repeat(32);
        let name = format!("{}-semantic", kind.name());
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name.as_str(), "--snapshot", "local.snapshot",
            "--expected-model-sha256", &pin, "--memory-mib", "8192", "--semantic-options", "semantic.json",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong snapshot route"); };
        let Int4Task::SourceSemantic(actual, semantic) = command.task else { panic!("wrong semantic route"); };
        assert_eq!(actual, kind);
        assert!(semantic.validate(&command.args).unwrap().snapshot_pin.is_some());
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv(kind.name(), &[])).is_err());
    }
}

#[test]
fn incompatible_generation_and_projection_options_refuse_before_io() {
    let seed = "ab".repeat(32);
    for name in ["summarize-semantic", "answer-semantic"] {
        for flags in [vec!["--seed", seed.as_str()], vec!["--logprobs"], vec!["--stop", "END"],
            vec!["--prefill-rows", "1"], vec!["--presence-penalty-milli", "0"],
            vec!["--max-output-bytes", "65536"], vec!["--max-selected-rows", "166144"],
            vec!["--cohort-documents", "2"],
            vec!["--think"], vec!["--tools", "tools.json"],
            vec!["--retry", "1"]] {
            if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv(name, &flags)) {
                if let Ok(command) = CandidateCommand::from_matches(&matches) {
                    assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments), "{name} {flags:?}");
                }
            }
        }
    }
}

#[test]
fn compact_source_reviews_admit_both_model_inputs_with_separate_whole_operation_tokenizer_caps() {
    for (name, kind) in [("summarize-semantic", Kind::Summarize), ("answer-semantic", Kind::Answer)] {
        let (byte, byte_args, byte_common, options) = fixture(kind);
        assert_eq!(byte.input_encoding.policy().unwrap(), UntrustedEncoding::ByteFallback);
        assert_eq!(byte.input_encoding.workspace_bytes().unwrap(), 0);
        assert_eq!(byte_args.preparation_mib, 512);
        for snapshot in [false, true] {
            let mut command = command(name, &["--input-encoding", "safe-bpe-v1"]);
            assert_eq!(command.args.preparation_mib, 1024);
            if snapshot {
                command.args.source_dir = None; command.args.source_manifest = None;
                command.args.snapshot = Some(PathBuf::from("private.snapshot"));
                command.args.expected_model_sha256 = Some("ab".repeat(32));
            }
            let Int4Task::SourceSemantic(actual, semantic) = command.task else { panic!("wrong semantic route"); };
            assert_eq!(actual, kind);
            assert!(command.args.input_encoding.is_default(), "semantic policy is separately owned");
            let UntrustedEncoding::SafeBpeV1(limits) = semantic.input_encoding.policy().unwrap()
                else { panic!("compact policy required"); };
            assert_eq!(limits, SafeBpeLimits { max_input_bytes: SafeBpeLimits::MAX_INPUT_BYTES,
                max_output_tokens: SafeBpeLimits::MAX_OUTPUT_TOKENS,
                max_pair_checks: SafeBpeLimits::MAX_PAIR_CHECKS, max_pair_bytes: SafeBpeLimits::MAX_PAIR_BYTES });
            assert!(limits.max_output_tokens > command.args.context_tokens);
            assert_eq!(semantic.input_encoding.workspace_bytes().unwrap(), limits.requirements().unwrap().workspace_bytes);
            let common = semantic.validate(&command.args).unwrap();
            let admitted = semantic.admit_options(&command.args, common, &options).unwrap();
            assert_eq!(admitted.snapshot_pin.is_some(), snapshot);
            assert_eq!(admitted.native, byte_common.native);
            assert_eq!(semantic.source.budget(&command.args, admitted), byte.source.budget(&byte_args, byte_common));
        }
        let command = command(name, &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "4096",
            "--max-tokenizer-output-tokens", "2048", "--max-tokenizer-pair-checks", "7777",
            "--max-tokenizer-pair-bytes", "65536"]);
        let Int4Task::SourceSemantic(_, semantic) = command.task else { panic!("wrong semantic route"); };
        assert_eq!(semantic.input_encoding.policy().unwrap(), UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 2048, max_pair_checks: 7777, max_pair_bytes: 65536 }));
        assert!(semantic.validate(&command.args).is_ok());
    }
}

#[test]
fn compact_source_preparation_defaults_preserve_explicit_overrides_and_refuse_an_underquote_before_io() {
    for name in ["summarize-semantic", "answer-semantic"] {
        assert_eq!(command(name, &["--input-encoding", "byte-fallback"]).args.preparation_mib, 512);
        for preparation in ["1", "512"] {
            let command = command(name, &["--input-encoding", "safe-bpe-v1", "--preparation-mib", preparation]);
            assert_eq!(command.args.preparation_mib, preparation.parse::<u64>().unwrap());
            assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
        let command = command(name, &["--input-encoding", "safe-bpe-v1", "--preparation-mib", "1536"]);
        assert_eq!(command.args.preparation_mib, 1536);
        let Int4Task::SourceSemantic(_, semantic) = command.task else { panic!("wrong semantic route"); };
        assert!(semantic.validate(&command.args).is_ok());
    }
}

#[test]
fn source_review_tokenizer_options_refuse_missing_opt_in_and_invalid_bounds_before_io() {
    for name in ["summarize-semantic", "answer-semantic"] {
        for (flag, above_maximum) in [("--max-tokenizer-input-bytes", "1048577"),
            ("--max-tokenizer-output-tokens", "1048577"), ("--max-tokenizer-pair-checks", "67108865"),
            ("--max-tokenizer-pair-bytes", "536870913")] {
            let matches = crate::candidate_cli::definition().try_get_matches_from(argv(name, &[flag, "1"])).unwrap();
            assert!(CandidateCommand::from_matches(&matches).is_err(), "{name} {flag}");
            for value in ["0", above_maximum] {
                let matches = crate::candidate_cli::definition().try_get_matches_from(argv(name, &[
                    "--input-encoding", "safe-bpe-v1", flag, value])).unwrap();
                if let Ok(command) = CandidateCommand::from_matches(&matches) {
                    assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments),
                        "{name} {flag} {value}");
                }
            }
        }
    }
}

#[test]
fn bad_paths_resource_limits_and_wrong_task_kinds_refuse_before_input_or_output() {
    for kind in [Kind::Summarize, Kind::Answer] {
        for path in ["", "-", "https://example.invalid/private-options.json"] {
            let (mut semantic, args, _, _) = fixture(kind);
            semantic.semantic_options = PathBuf::from(path);
            assert_eq!(semantic.execute(kind, args, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
    }
    for name in ["summarize-semantic", "answer-semantic"] {
        assert_eq!(command(name, &["--options", "https://example.invalid/private-task-options.json"])
            .execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        for flags in [vec!["--max-result-bytes", "0"], vec!["--context-tokens", "0"],
            vec!["--max-mask-node-visits", "1"], vec!["--preparation-mib", "1"]] {
            assert_eq!(command(name, &flags).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
    }
    for wrong in [Kind::Ner, Kind::Keyphrases] {
        let (semantic, args, _, _) = fixture(Kind::Summarize);
        assert_eq!(semantic.execute(wrong, args, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn strict_semantic_options_reject_unknown_duplicate_missing_and_nested_authority() {
    let (_, _, _, options) = fixture(Kind::Summarize);
    let original = serde_json::to_value(&options).unwrap();
    for field in ["identity", "model", "source_task", "result", "judge_receipts"] {
        let mut value = original.clone(); value[field] = serde_json::json!({});
        assert!(parse_options(&value.to_string()).is_err());
    }
    let mut nested = original.clone(); nested["limits"]["model"] = serde_json::json!("replacement");
    assert!(parse_options(&nested.to_string()).is_err());
    let mut missing = original.clone(); missing.as_object_mut().unwrap().remove("max_model_work");
    assert!(parse_options(&missing.to_string()).is_err());
    let text = serde_json::to_string(&options).unwrap();
    let duplicate = format!("{{\"schema_version\":1,{}", &text[1..]);
    assert!(parse_options(&duplicate).is_err());
    assert!(parse_options(&" ".repeat(OPTIONS_BYTES + 1)).is_err());
    assert_eq!(serde_json::to_value(parse_options(&text).unwrap()).unwrap(), original);
}

#[test]
fn opt_in_every_native_axis_and_aggregate_review_limits_remain_mandatory() {
    for axis in 0..11 {
        let (_, _, _, mut options) = fixture(Kind::Summarize);
        match axis {
            0 => options.experimental_opt_in = false,
            1 => options.schema_version = 2,
            2 => options.judge_policy.max_evidence_windows = 32,
            3 => options.max_model_work.forward_positions = 0,
            4 => options.max_model_work.projected_logits = 0,
            5 => options.max_model_work.attention_pairs = 0,
            6 => options.max_model_work.projections.dot_products = 0,
            7 => options.max_model_work.projections.multiply_accumulates = 0,
            8 => options.limits.max_total_projected_logits = 0,
            9 => options.limits.max_statements = 0,
            _ => options.limits.max_total_claim_bytes = 0,
        }
        assert!(parse_options(&serde_json::to_string(&options).unwrap()).is_err(), "axis {axis}");
    }
}

#[test]
fn judge_context_whole_kv_and_original_and_combined_result_budgets_cannot_substitute() {
    for kind in [Kind::Summarize, Kind::Answer] {
        for axis in 0..8 {
            let (semantic, args, common, mut options) = fixture(kind);
            match axis {
                0 => options.judge_budget.max_kv_bytes -= 1,
                1 => options.judge_budget.max_kv_bytes += 1,
                2 => options.judge_budget.max_input_tokens += 1,
                3 => options.judge_budget.max_output_tokens = 1025,
                4 => options.judge_budget.max_grammar_states = 65_537,
                5 => options.judge_budget.max_output_bytes = options.limits.max_output_bytes + 1,
                6 => options.limits.max_output_bytes = common.result_bytes as u64 - 1,
                _ => options.judge_budget.max_input_tokens = u32::MAX,
            }
            assert!(semantic.admit_options(&args, common, &options).is_err(), "{kind:?} axis {axis}");
        }
    }
}

#[test]
fn statement_population_does_not_renew_prompt_logit_or_candidate_limits() {
    let (_, _, _, mut options) = fixture(Kind::Summarize);
    let before = judge_limits(&options); options.limits.max_statements *= 2;
    let after = judge_limits(&options);
    assert_eq!(before.max_total_prompt_tokens, after.max_total_prompt_tokens);
    assert_eq!(before.max_total_projected_logits, after.max_total_projected_logits);
    assert_eq!(before.max_total_candidate_tokens, after.max_total_candidate_tokens);
    assert_eq!(after.max_output_bytes, options.judge_budget.max_output_bytes);
    assert_eq!(after.per_head.max_candidates, 3);
    options.max_model_work.projected_logits = 10;
    assert_eq!(judge_limits(&options).max_total_projected_logits, 10);
    assert_eq!(judge_limits(&options).per_head.max_projected_logits, 10);
}

#[test]
fn input_ceiling_retains_all_qa_dimensions_and_debug_redacts_paths() {
    let (mut semantic, args, _, options) = fixture(Kind::Answer);
    let summary = input_ceiling(Kind::Summarize, &args, &options).unwrap();
    let answer = input_ceiling(Kind::Answer, &args, &options).unwrap();
    assert_eq!(summary.source_bytes, answer.source_bytes);
    assert_eq!(summary.question_bytes, 0); assert_eq!(summary.passage_count, 0);
    assert_eq!(answer.question_bytes, args.max_input_bytes.min(options.limits.max_claim_bytes));
    assert_eq!(answer.passage_count, crate::candidate_cli::source::MAX_PASSAGES);
    assert!(input_ceiling(Kind::Ner, &args, &options).is_err());
    semantic.semantic_options = PathBuf::from("PRIVATE_REVIEW_POLICY.json");
    assert_eq!(format!("{semantic:?}"), "SemanticSourceArgs");
}

#[test]
fn both_source_reviews_admit_one_explicit_team_without_multiplying_task_budgets() {
    for (name, kind) in [("summarize-semantic", Kind::Summarize), ("answer-semantic", Kind::Answer)] {
        let (_, _, _, options) = fixture(kind);
        for (encoding, width) in ["byte-fallback", "safe-bpe-v1"].into_iter()
            .flat_map(|encoding| ["1", "2", "4"].into_iter().map(move |width| (encoding, width))) {
            let command = command(name, &["--compute-threads", width, "--input-encoding", encoding]);
            let Int4Task::SourceSemantic(actual, semantic) = command.task else { panic!("wrong semantic route"); };
            assert_eq!(actual, kind);
            let args = command.args;
            let common = semantic.validate(&args).unwrap();
            let admitted = semantic.admit_options(&args, common, &options).unwrap();
            assert_eq!(admitted.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(admitted.compute, common.compute);
            assert_eq!(semantic.source.budget(&args, admitted), semantic.source.budget(&args, common));
            assert_eq!(admitted.compute.workspace_bytes == 0, width == "1");
            assert!(admitted.prefill.is_none() && admitted.cohort.is_none());
            assert!(args.input_encoding.is_default());
            assert_eq!(matches!(semantic.input_encoding.policy().unwrap(), UntrustedEncoding::SafeBpeV1(_)),
                encoding == "safe-bpe-v1");
        }
        for width in ["0", "65"] {
            assert_eq!(command(name, &["--compute-threads", width]).execute(&mut NoIo, &mut NoIo),
                Err(CandidateError::Arguments));
        }
        for flags in [vec!["--compute-threads", "2", "--prefill-rows", "1"],
            vec!["--compute-threads", "2", "--cohort-documents", "2"],
            vec!["--compute-threads", "2", "--max-selected-rows", "8"]] {
            if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv(name, &flags)) {
                assert!(CandidateCommand::from_matches(&matches).is_err());
            }
        }
    }
}
