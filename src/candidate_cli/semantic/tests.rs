//! Model-free command/options admission regressions. These fixtures do not
//! supply native model results or establish semantic quality.
use super::*;
use crate::{
    native_engine::strict_int8::Int8Work,
    tasks::{extract::semantic::{ClaimRule, ClaimValueKind, SemanticLimits, SemanticVerificationSpec},
        ir::TaskBudget, judge::FaithfulnessPolicy},
};

fn command(extra: &[&str]) -> SemanticCommand {
    let mut argv = vec!["candidate", "extract-semantic", "--model", "must-not-open.fnlpq",
        "--memory-mib", "8192", "--schema", "schema.json", "--semantic-options", "semantic.json",
        "--max-input-bytes", "512", "--max-new-tokens", "128", "--max-result-bytes", "16384"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    match CandidateCommand::from_matches(&matches).unwrap() {
        CandidateCommand::SemanticExtract(command) => command,
        _ => panic!("wrong semantic command route"),
    }
}
fn options(limits: Limits) -> Int8SemanticOptions {
    let mut work = Int8Work::default();
    work.forward_positions = 4096; work.projected_logits = 64_000_000;
    work.attention_pairs = 16_777_216; work.projections.dot_products = 100_000_000_000;
    work.projections.multiply_accumulates = 1_000_000_000_000_000;
    Int8SemanticOptions {
        specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "semantic-cli-test-v1".into(),
            claims: vec![ClaimRule { id: "amount-v1".into(), path: vec![], value_kind: ClaimValueKind::Number,
                prefix: "The amount is ".into(), suffix: ".".into() }],
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 512, max_evidence_windows: 1, max_evidence_spans: 1 },
            judge_budget: TaskBudget { max_input_tokens: 2016, max_output_tokens: 32,
                max_output_bytes: 65_536, max_grammar_states: 128, max_kv_bytes: limits.kv_bytes },
        },
        limits: SemanticLimits { max_fields: 2, max_claim_bytes: 256, max_total_claim_bytes: 1024,
            max_template_bytes: 2048, max_walk_nodes: 128, max_input_json_bytes: 16_384,
            max_total_prompt_tokens: 8192, max_total_projected_logits: 16_777_216,
            max_output_bytes: 1_048_576 },
        max_model_work: work,
    }
}

#[test]
fn semantic_route_requires_explicit_schema_options_model_and_memory() {
    let required = [("--model", "m"), ("--memory-mib", "8192"),
        ("--schema", "schema.json"), ("--semantic-options", "semantic.json")];
    for omitted in 0..required.len() {
        let mut argv = vec!["candidate", "extract-semantic"];
        for (index, (flag, value)) in required.iter().enumerate() {
            if index != omitted { argv.extend([*flag, *value]); }
        }
        assert!(super::super::definition().try_get_matches_from(argv).is_err(), "required {omitted}");
    }
    let command = command(&["document.txt", "--source-membership"]);
    command.validate().unwrap();
    assert_eq!(command.input, std::path::Path::new("document.txt"));
    assert!(command.source_membership);
    assert_eq!(command.semantic_options, std::path::Path::new("semantic.json"));
}

#[test]
fn semantic_route_keeps_serial_byte_defaults_and_full_projection_scope() {
    let command = command(&[]);
    let (args, limits) = command.validate().unwrap();
    assert_eq!(args.compute.threads(), 1);
    assert_eq!(limits.compute, compute::Args::default().admit_prefill(None).unwrap());
    assert_eq!(args.input_encoding_policy(limits).unwrap(), UntrustedEncoding::ByteFallback);
    assert!(args.policy.prefill().unwrap().is_none());
    for option in ["--selected-rows",
        "--seed", "--temperature-milli", "--top-k", "--top-p-ppm", "--options"] {
        assert!(super::super::definition().try_get_matches_from([
            "candidate", "extract-semantic", "--model", "m", "--memory-mib", "8192",
            "--schema", "s", "--semantic-options", "o", option, "1",
        ]).is_err(), "{option}");
    }
}

#[test]
fn invalid_private_paths_and_preparation_refuse_before_any_io() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source read"); }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("semantic output written"); }
        fn flush(&mut self) -> io::Result<()> { panic!("semantic output flushed"); }
    }
    for axis in 0..6 {
        let mut command = command(&[]);
        match axis {
            0 => command.schema = PathBuf::from("-"),
            1 => command.semantic_options = PathBuf::from("-"),
            2 => command.schema = PathBuf::new(),
            3 => command.semantic_options = PathBuf::new(),
            4 => command.host.preparation_mib = 511,
            _ => command.host.memory_mib = 0,
        }
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments), "axis {axis}");
    }
}

#[test]
fn options_parser_rejects_duplicates_unknown_fields_missing_opt_in_and_zero_native_axes() {
    let command = command(&[]); let (_, limits) = command.validate().unwrap();
    let value = options(limits);
    let json = serde_json::to_string(&value).unwrap();
    let parsed = parse_options(&json).unwrap();
    assert_eq!(canonjson::canonical_bytes(&parsed).unwrap(), canonjson::canonical_bytes(&value).unwrap());
    for key in [r#""max_model_work""#, r#""max_\u006dodel_work""#] {
        let duplicate = format!("{{{key}:{{}},{}", &json[1..]);
        assert!(matches!(parse_options(&duplicate), Err(CandidateError::Input)));
    }
    let mut unknown = serde_json::to_value(&value).unwrap();
    unknown.as_object_mut().unwrap().insert("automatic_acceptance".into(), serde_json::json!(true));
    assert!(matches!(parse_options(&unknown.to_string()), Err(CandidateError::Input)));
    let mut missing = value.clone(); missing.specification.experimental_opt_in = false;
    assert!(matches!(parse_options(&serde_json::to_string(&missing).unwrap()), Err(CandidateError::Arguments)));
    for axis in 0..5 {
        let mut zero = value.clone();
        match axis { 0 => zero.max_model_work.forward_positions = 0, 1 => zero.max_model_work.projected_logits = 0,
            2 => zero.max_model_work.attention_pairs = 0, 3 => zero.max_model_work.projections.dot_products = 0,
            _ => zero.max_model_work.projections.multiply_accumulates = 0 }
        assert!(matches!(parse_options(&serde_json::to_string(&zero).unwrap()), Err(CandidateError::Arguments)), "axis {axis}");
    }
    assert!(matches!(parse_options(&" ".repeat(OPTIONS_BYTES + 1)), Err(CandidateError::Input)));
}

#[test]
fn judge_context_and_complete_result_limits_are_admitted_together() {
    let command = command(&[]); let (args, limits) = command.validate().unwrap();
    let value = options(limits);
    assert_eq!(limits.kv_bytes, 369_098_752);
    assert_eq!(command.admit_options(&args, limits, &value).unwrap(), 1_048_576 + 4096);
    for axis in 0..9 {
        let mut bad = value.clone();
        match axis {
            0 => bad.specification.judge_budget.max_kv_bytes -= 1,
            1 => bad.specification.judge_budget.max_input_tokens = args.context_tokens as u32 + 1,
            2 => bad.specification.judge_budget.max_output_tokens = 1025,
            3 => bad.specification.judge_budget.max_input_tokens = args.context_tokens as u32,
            4 => bad.specification.judge_budget.max_grammar_states = 65_537,
            5 => bad.specification.judge_budget.max_output_bytes = bad.limits.max_output_bytes + 1,
            6 => bad.limits.max_output_bytes = command.host.max_result_bytes as u64 - 1,
            7 => bad.limits.max_input_json_bytes = command.host.max_result_bytes - 1,
            _ => bad.specification.judge_budget.max_output_tokens = 0,
        }
        assert_eq!(command.admit_options(&args, limits, &bad), Err(CandidateError::Arguments), "axis {axis}");
    }
}

#[test]
fn judge_limits_preserve_complete_three_way_heads_and_shared_whole_request_caps() {
    let command = command(&[]); let (_, limits) = command.validate().unwrap();
    let mut options = options(limits);
    options.max_model_work.projected_logits = 7_000_000;
    let planned = judge_limits(&options);
    assert_eq!(planned.per_head.max_candidates, 3);
    assert_eq!(planned.per_head.max_total_tokens, 127);
    assert_eq!(planned.per_head.max_nodes, 128);
    assert_eq!(planned.per_head.max_depth, 32);
    assert_eq!(planned.max_total_candidate_tokens, 128);
    assert_eq!(planned.max_total_prompt_tokens, options.limits.max_total_prompt_tokens);
    assert_eq!(planned.max_total_projected_logits, 7_000_000);
    assert_eq!(planned.per_head.max_projected_logits, 7_000_000);
    assert_eq!(planned.max_output_bytes, options.specification.judge_budget.max_output_bytes);
}

#[test]
fn compact_semantic_cli_admits_one_complete_request_allowance_before_io() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let compact = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "8192",
        "--max-tokenizer-pair-checks", "1000000", "--max-tokenizer-pair-bytes", "8000000"]);
    let (_, limits) = compact.validate().unwrap();
    assert_eq!(compact.host.preparation_mib, 1024);
    assert_eq!(compact.encoding_policy().unwrap(), UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65_536, max_output_tokens: 8192,
        max_pair_checks: 1_000_000, max_pair_bytes: 8_000_000,
    }));
    assert!(compact.encoding.workspace_bytes().unwrap() > 0);
    assert_eq!(limits.compute.threads, 1);

    for flag in ["--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
        "--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let matches = definition().try_get_matches_from(["extract-semantic",
            "--model", "m", "--memory-mib", "8192", "--schema", "s", "--semantic-options", "o", flag, "1"]).unwrap();
        assert!(SemanticCommand::from_matches(&matches).is_err(), "{flag} accepted for byte input");
        let invalid = command(&["--input-encoding", "safe-bpe-v1", flag, "0"]);
        assert!(invalid.validate().is_err(), "{flag} accepted zero");
    }
}

#[test]
fn compact_semantic_workspace_does_not_replace_original_schema_or_options_memory() {
    let mut compact = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "8192"]);
    compact.validate().unwrap();
    let encoder = compact.encoding.workspace_bytes().unwrap();
    let floor = 256 * MIB + compact.host.max_input_bytes as u64 * 32
        + extract::SCHEMA_BYTES as u64 * 32 + compact.host.max_result_bytes as u64 * 16
        + OPTIONS_BYTES as u64 * 16 + encoder;
    compact.host.preparation_mib = floor.div_ceil(MIB).max(512);
    compact.validate().unwrap();
    compact.host.preparation_mib = 511;
    assert!(compact.validate().is_err());
}

#[test]
fn semantic_cli_admits_one_shared_prefill_and_compute_schedule() {
    for width in [1, 2, 4, 64] {
        for rows in [None, Some(1), Some(4), Some(64)] {
            let width_text = width.to_string();
            let row_text = rows.map(|n| n.to_string());
            let mut extra = vec!["--compute-threads", width_text.as_str()];
            if let Some(text) = &row_text { extra.extend(["--prefill-rows", text.as_str()]); }
            let command = command(&extra);
            let (args, limits) = command.validate().unwrap();
            let prefill = command.prefill().unwrap();
            assert_eq!(prefill.map(|limits| limits.max_batch_rows), rows);
            assert_eq!(args.compute.threads(), width);
            assert_eq!(limits.compute, args.compute.admit_prefill(prefill).unwrap());
            assert_eq!(limits.compute.workspace_bytes == 0, width == 1);
            assert!(args.policy.prefill().unwrap().is_none());
            assert_eq!(command.encoding_policy().unwrap(), UntrustedEncoding::ByteFallback);
        }
    }
    let command = command(&["--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "8"]);
    let (args, limits) = command.validate().unwrap();
    assert_eq!(limits.compute, args.compute.admit_prefill(command.prefill().unwrap()).unwrap());
    assert!(matches!(command.encoding_policy().unwrap(), UntrustedEncoding::SafeBpeV1(_)));
}

#[test]
fn semantic_cli_refuses_unbounded_worker_or_prompt_geometry_before_io() {
    for flag in ["--compute-threads", "--prefill-rows"] {
        let largest = usize::MAX.to_string();
        for value in ["0", "65", largest.as_str()] {
            let command = command(&[flag, value]);
            assert!(command.validate().is_err(), "{flag} accepted {value}");
        }
    }
}
