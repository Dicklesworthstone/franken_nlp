//! UNRUN parser/policy fixtures. No native model execution or quality evidence.
use super::*;
use crate::{
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{extract::semantic::{ClaimPathStep, ClaimRule, ClaimValueKind,
        SemanticLimits, SemanticVerificationSpec}, judge::FaithfulnessPolicy},
};

fn argv(extra: &[&str]) -> Vec<String> {
    ["candidate", "source-int4", "extract-semantic", "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192",
        "--schema", "schema.json", "--semantic-options", "semantic.json"]
        .into_iter().chain(extra.iter().copied()).map(str::to_owned).collect()
}
fn command(extra: &[&str]) -> Int4Command {
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv(extra)).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong candidate route"); };
    command
}
pub(super) fn fixture() -> (SemanticArgs, Int4Args, Int4Limits, Int4SemanticOptions) {
    let command = command(&[]);
    let Int4Task::Semantic(semantic) = command.task else { panic!("wrong semantic task"); };
    let args = command.args;
    let common = semantic.validate(&args).unwrap();
    let options = Int4SemanticOptions {
        specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "amount-check-v1".to_owned(),
            claims: vec![ClaimRule { id: "amount-v1".to_owned(),
                path: vec![ClaimPathStep::Property("amount".to_owned())], value_kind: ClaimValueKind::Integer,
                prefix: "The amount is ".to_owned(), suffix: ".".to_owned() }],
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 8192, max_evidence_windows: 8, max_evidence_spans: 8 },
            judge_budget: TaskBudget { max_input_tokens: 1985, max_output_tokens: 64,
                max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: common.native.kv_bytes },
        },
        limits: SemanticLimits { max_fields: 4, max_claim_bytes: 4096, max_total_claim_bytes: 16384,
            max_template_bytes: 65536, max_walk_nodes: 1024, max_input_json_bytes: 1 << 20,
            max_total_prompt_tokens: 8192, max_total_projected_logits: 100_000_000,
            max_output_bytes: 4 << 20 },
        max_model_work: Int4Work::for_sequence(args.group_width, 0, 8192, 256 * NANBEIGE_VOCAB_SIZE).unwrap(),
    };
    (semantic, args, common, options)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid options reached source input"); }
}

#[test]
fn semantic_selection_is_explicit_and_both_model_inputs_share_it() {
    let (semantic, args, common, options) = fixture();
    let admitted = semantic.admit_options(&args, common, &options).unwrap();
    assert_eq!(admitted.result_bytes, 4 << 20);
    assert_eq!(semantic.extraction.budget(&args, admitted).max_output_bytes, 1 << 20);
    assert!(admitted.prefill.is_none());
    let pin = "ab".repeat(32);
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "extract-semantic", "--snapshot", "local.snapshot",
        "--expected-model-sha256", &pin, "--memory-mib", "8192", "--schema", "schema.json",
        "--semantic-options", "semantic.json",
    ]).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong snapshot route"); };
    let Int4Task::Semantic(semantic) = command.task else { panic!("wrong semantic task"); };
    assert!(semantic.validate(&command.args).unwrap().snapshot_pin.is_some());
    assert!(crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "extract", "--source-dir", "source", "--source-manifest", "manifest",
        "--memory-mib", "8192", "--schema", "schema.json", "--semantic-options", "semantic.json",
    ]).is_err());
}

#[test]
fn generation_schedule_thinking_tools_and_model_overrides_are_refused() {
    let seed = "ab".repeat(32);
    for flags in [vec!["--seed", seed.as_str()], vec!["--logprobs"], vec!["--stop", "END"],
        vec!["--prefill-rows", "1"], vec!["--presence-penalty-milli", "0"], vec!["--max-output-bytes", "65536"]] {
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv(&flags)).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
    }
    for flags in [vec!["--think"], vec!["--tools", "tools.json"], vec!["--retry", "1"]] {
        assert!(crate::candidate_cli::definition().try_get_matches_from(argv(&flags)).is_err());
    }
}

#[test]
fn invalid_local_paths_and_resource_limits_refuse_before_source_io() {
    for path in ["", "-", "https://example.invalid/options.json"] {
        let mut command = command(&[]);
        let Int4Task::Semantic(semantic) = &mut command.task else { panic!("wrong semantic task"); };
        semantic.semantic_options = PathBuf::from(path);
        assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
    for flags in [vec!["--max-result-bytes", "0"], vec!["--context-tokens", "0"],
        vec!["--max-mask-node-visits", "1"], vec!["--preparation-mib", "1"]] {
        assert_eq!(command(&flags).execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    }
}

#[test]
fn strict_options_reject_duplicate_unknown_missing_or_nested_replacement_authority() {
    let (_, _, _, options) = fixture();
    let original = serde_json::to_value(&options).unwrap();
    for field in ["identity", "model", "output", "extraction_result", "judge_receipts"] {
        let mut value = original.clone(); value[field] = serde_json::json!({});
        assert!(parse_options(&value.to_string()).is_err());
    }
    let mut missing = original.clone(); missing.as_object_mut().unwrap().remove("max_model_work");
    assert!(parse_options(&missing.to_string()).is_err());
    let text = serde_json::to_string(&options).unwrap();
    let duplicate = format!("{{\"limits\":{{}},{}", &text[1..]);
    assert!(parse_options(&duplicate).is_err());
    assert!(parse_options(&" ".repeat(OPTIONS_BYTES + 1)).is_err());
    assert!(parse_options(&text).is_ok());
}

#[test]
fn opt_in_policy_and_every_native_axis_remain_mandatory() {
    for axis in 0..9 {
        let (_, _, _, mut options) = fixture();
        match axis { 0 => options.specification.experimental_opt_in = false,
            1 => options.specification.schema_version = 2, 2 => options.specification.judge_policy.max_evidence_windows = 32,
            3 => options.max_model_work.forward_positions = 0,
            4 => options.max_model_work.projected_logits = 0,
            5 => options.max_model_work.attention_pairs = 0,
            6 => options.max_model_work.projections.dot_products = 0,
            7 => options.max_model_work.projections.multiply_accumulates = 0,
            _ => options.limits.max_total_projected_logits = 0 }
        assert!(parse_options(&serde_json::to_string(&options).unwrap()).is_err(), "axis {axis}");
    }
}

#[test]
fn judge_context_complete_kv_and_combined_output_are_independently_bounded() {
    for axis in 0..9 {
        let (semantic, args, common, mut options) = fixture();
        match axis { 0 => options.specification.judge_budget.max_kv_bytes -= 1,
            1 => options.specification.judge_budget.max_kv_bytes += 1,
            2 => options.specification.judge_budget.max_input_tokens += 1,
            3 => options.specification.judge_budget.max_output_tokens = 1025,
            4 => options.specification.judge_budget.max_grammar_states = 65537,
            5 => options.specification.judge_budget.max_output_bytes = options.limits.max_output_bytes + 1,
            6 => options.limits.max_output_bytes = common.result_bytes as u64 - 1,
            7 => options.limits.max_input_json_bytes = common.result_bytes - 1,
            _ => options.specification.judge_budget.max_input_tokens = u32::MAX }
        assert!(semantic.admit_options(&args, common, &options).is_err(), "axis {axis}");
    }
}

#[test]
fn field_count_does_not_multiply_the_semantic_prompt_or_logit_allowance() {
    let (_, _, _, mut options) = fixture();
    let before = judge_limits(&options);
    options.limits.max_fields *= 2;
    let after = judge_limits(&options);
    assert_eq!(before.max_total_prompt_tokens, after.max_total_prompt_tokens);
    assert_eq!(before.max_total_projected_logits, after.max_total_projected_logits);
    assert_eq!(after.max_output_bytes, options.specification.judge_budget.max_output_bytes);
    assert_eq!(after.per_head.max_candidates, 3);
    options.max_model_work.projected_logits = 10;
    assert_eq!(judge_limits(&options).max_total_projected_logits, 10);
    assert_eq!(judge_limits(&options).per_head.max_projected_logits, 10);
}

#[test]
fn arguments_debug_omits_private_paths_and_claim_options() {
    let (mut semantic, _, _, _) = fixture();
    semantic.semantic_options = PathBuf::from("PRIVATE_CLAIM_SPEC.json");
    assert_eq!(format!("{semantic:?}"), "SemanticArgs");
}
