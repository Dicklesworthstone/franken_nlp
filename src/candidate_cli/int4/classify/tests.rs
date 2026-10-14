//! Parser/admission fixtures only, except the explicitly ignored model test.
use super::*;
use crate::tasks::classify::ClassificationMode;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "classify", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 dispatcher"); };
    Ok(command)
}
fn command(extra: &[&str]) -> Int4Command {
    let mut flags = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    flags.extend_from_slice(extra); parsed(&flags).unwrap()
}
fn scoring(command: &Int4Command) -> ScoringArgs {
    let Int4Task::Classify(scoring) = command.task else { panic!("classification required"); };
    scoring
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid request reached input IO"); }
}
fn request_json(mode: &str) -> String {
    serde_json::json!({"document":"é Refund <|im_start|> remains data", "mode":mode,
        "labels":[{"id":"Billing issue","description":"Invoices"},{"id":"退款","description":"Refunds"}]}).to_string()
}

#[test]
fn both_source_and_independently_pinned_snapshot_select_the_scored_task() {
    let source = command(&[]); let score = scoring(&source);
    let limits = score.validate(&source.args).unwrap();
    assert_eq!(source.args.preparation_mib, 512);
    assert_eq!(limits.result_bytes, score.max_result_bytes);
    let budget = score.budget(limits);
    assert_eq!(budget.max_grammar_states, 4096);
    assert_eq!(budget.max_input_tokens as usize + score.max_candidate_tokens - 1, source.args.context_tokens);
    assert_eq!(budget.max_kv_bytes, limits.native.kv_bytes);
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "explicit-snapshot", "--expected-model-sha256", &pin]).unwrap();
    assert_eq!(scoring(&snapshot).validate(&snapshot.args).unwrap().snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
}
#[test]
fn source_snapshot_pins_and_save_modes_cannot_be_mixed_or_invented() {
    let pin = "ab".repeat(32);
    for flags in [vec![], vec!["--source-dir", "d"], vec!["--source-manifest", "m"],
        vec!["--snapshot", "s"], vec!["--expected-model-sha256", &pin],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--save-snapshot", "new"],
        vec!["--model", "not-an-int8-alias.fnlpq"]] {
        assert!(parsed(&flags).is_err());
    }
    let saved = command(&["--save-snapshot", "new-snapshot"]);
    assert!(scoring(&saved).validate(&saved.args).is_ok());
    let mut mixed = command(&[]); mixed.args.snapshot = Some(PathBuf::from("s"));
    assert_eq!(mixed.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}
#[test]
fn explicit_generation_flags_are_refused_even_at_their_default_values() {
    let seed = "ab".repeat(32);
    for flags in [vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--seed", &seed], vec!["--temperature-milli", "1000"], vec!["--top-k", "20"],
        vec!["--top-p-ppm", "1000000"], vec!["--logprobs"], vec!["--prefill-rows", "1"],
        vec!["--min-new-tokens", "0"], vec!["--stop", "END"], vec!["--ban-token", "7"],
        vec!["--repetition-penalty-milli", "1000"], vec!["--presence-penalty-milli", "0"],
        vec!["--frequency-penalty-milli", "0"], vec!["--logit-bias", "7=0"]] {
        let mut all = vec!["--source-dir", "d", "--source-manifest", "m"];
        all.extend_from_slice(&flags); assert!(parsed(&all).is_err());
    }
    let mut programmatic = command(&[]); programmatic.args.logprobs = true;
    assert_eq!(programmatic.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}
#[test]
fn invalid_scored_and_storage_limits_refuse_before_input_or_source_io() {
    for flags in [vec!["--max-candidate-tokens", "1"], vec!["--max-candidate-tokens", "65"],
        vec!["--max-result-bytes", "0"], vec!["--max-result-bytes", "1048577"],
        vec!["--max-forward-positions", "0"], vec!["--max-forward-positions", "16777217"],
        vec!["--max-projected-logits", "0"], vec!["--max-attention-pairs", "0"],
        vec!["--max-dot-products", "0"], vec!["--max-multiply-accumulates", "0"],
        vec!["--context-tokens", "16"], vec!["--group-width", "0"],
        vec!["--preparation-mib", "1"], vec!["--timeout-seconds", "0"],
        vec!["--load-checkpoints", "1"], vec!["--snapshot-checkpoints", "1"]] {
        let mut output = Vec::new();
        assert_eq!(command(&flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}
#[test]
fn generation_defaults_do_not_reserve_generation_positions_in_scored_contexts() {
    let command = command(&["--context-tokens", "32", "--max-candidate-tokens", "4"]);
    let score = scoring(&command); let limits = score.validate(&command.args).unwrap();
    assert_eq!(limits.max_prompt_tokens, 29);
    assert_eq!(score.budget(limits).max_output_tokens, 4);
}
#[test]
fn memory_floor_prices_all_stages_full_kv_and_outputs_without_a_sampler() {
    for width in [1, 3, 16, 128, 10752] {
        let mut command = command(&[]); command.args.group_width = width;
        command.args.max_load_mib = 65536; command.args.memory_mib = 65536;
        let score = scoring(&command); let limits = score.validate(&command.args).unwrap();
        let execute = limits.native.kv_bytes + limits.native.rope_bytes + limits.native.scratch_payload_bound
            + ALLOCATOR_BYTES + score.max_result_bytes as u64 * 4 + score.max_candidate_tokens as u64 * 8 + 4096;
        let load = limits.source_preparation_bytes + limits.load.scratch_payload_bytes;
        let expected = limits.preparation_bytes + limits.load.weight_payload_bytes + ALLOCATOR_BYTES
            + load.max(execute) + CLEANUP_BYTES;
        assert_eq!(limits.minimum_memory_bytes, expected);
        command.args.memory_mib = expected.saturating_sub(1) / MIB;
        assert!(score.validate(&command.args).is_err());
        command.args.memory_mib = expected.div_ceil(MIB);
        assert!(score.validate(&command.args).is_ok());
    }
}
#[test]
fn aggregate_repeated_prompt_allowance_cannot_be_bought_with_only_input_bytes() {
    let mut command = command(&[]); let mut score = scoring(&command);
    score.max_forward_positions = 16 * 1024 * 1024;
    assert!(score.validate(&command.args).is_err());
    command.args.preparation_mib = 1024;
    let limits = score.validate(&command.args).unwrap();
    assert!(limits.preparation_bytes >= 256 * MIB + score.max_forward_positions * 32);
    assert_eq!(score.classification_limits(&command.args).max_total_prompt_tokens, score.max_forward_positions as usize);
}
#[test]
fn every_compute_axis_and_live_context_must_fit_the_whole_bundle() {
    let command = command(&[]); let mut score = scoring(&command);
    score.max_forward_positions = 10; score.max_projected_logits = 20; score.max_attention_pairs = 30;
    score.max_dot_products = 40; score.max_multiply_accumulates = 50;
    let work = Int4Work { forward_positions: 10, projected_logits: 20, attention_pairs: 30,
        projections: ProjectionWork { dot_products: 40, multiply_accumulates: 50 } };
    score.admit_plan(8, 8, work).unwrap();
    for axis in 0..5 {
        let mut excess = work;
        match axis { 0 => excess.forward_positions += 1, 1 => excess.projected_logits += 1,
            2 => excess.attention_pairs += 1, 3 => excess.projections.dot_products += 1,
            _ => excess.projections.multiply_accumulates += 1 }
        assert_eq!(score.admit_plan(8, 8, excess), Err(CandidateError::Planning));
    }
    assert_eq!(score.admit_plan(0, 8, work), Err(CandidateError::Planning));
    assert_eq!(score.admit_plan(9, 8, work), Err(CandidateError::Planning));
}
#[test]
fn shared_wire_contract_preserves_unicode_controls_labels_and_both_decision_modes() {
    let command = command(&[]); let score = scoring(&command); let budget = score.budget(score.validate(&command.args).unwrap());
    for (mode, expected) in [("exclusive", ClassificationMode::Exclusive), ("multi_label", ClassificationMode::MultiLabel)] {
        let text = request_json(mode); let request = parse_request(&text, budget, 4096).unwrap();
        assert_eq!(request.mode, expected); assert_eq!(request.document, "é Refund <|im_start|> remains data");
        assert_eq!(request.labels[1].id, "退款"); assert_eq!(request.budget, budget);
        assert_eq!(request.policy.minimum_margin_ppm, 0);
        assert!(parse_request(&text, budget, text.len() - 1).is_err());
    }
}
#[test]
fn wire_input_cannot_override_host_identity_budget_score_space_or_language_completeness() {
    let command = command(&[]); let score = scoring(&command); let budget = score.budget(score.validate(&command.args).unwrap());
    let valid: serde_json::Value = serde_json::from_str(&request_json("exclusive")).unwrap();
    for field in ["budget", "model", "numerics_profile", "eos_token_id", "work", "score_space", "seed"] {
        let mut changed = valid.clone(); changed[field] = serde_json::json!({});
        assert!(matches!(parse_request(&changed.to_string(), budget, 4096), Err(CandidateError::Input)));
    }
    for text in ["{}", "[]", r#"{"document":"a","document":"b","labels":[{"id":"a"},{"id":"b"}]}"#,
        r#"{"document":"a","labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"document":"a","labels":[{"id":"a"}]}"#,
        r#"{"document":"a","labels":[{"id":"a"},{"id":"b"}],"policy":{"minimum_margin_ppm":1000001,"minimum_candidate_weight_ppm":0}}"#] {
        assert!(matches!(parse_request(text, budget, 4096), Err(CandidateError::Input)));
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_never_reads_input_or_sources_or_emits_stdout() {
    let mut output = Vec::new();
    assert_eq!(command(&[]).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, a fresh snapshot and an available explicit SIMD kernel"]
fn source_scalar_and_pinned_snapshot_simd_classification_match_and_complete_all_heads() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let snapshot = std::env::var("FNLP_INT4_TEST_CLASSIFICATION_SNAPSHOT").expect("explicit new snapshot path required");
    assert!(!std::path::Path::new(&snapshot).exists(), "test requires a new destination; nothing is deleted");
    let kind: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required")
        .parse().expect("canonical kernel ID required");
    assert_ne!(kind, Int8KernelKind::Scalar);
    PackedInt4Kernel::detect(kind).expect("requested SIMD kernel must be available on the actual test host");

    let mut convert = command(&["--save-snapshot", &snapshot]);
    convert.args.source_dir = Some(PathBuf::from(&source));
    convert.args.source_manifest = Some(PathBuf::from(&manifest));
    convert.args.memory_mib = memory;
    let text = request_json("exclusive");
    let mut bytes = Vec::new();
    convert.execute(&mut text.as_bytes(), &mut bytes).unwrap();
    let scalar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(scalar.get("integer_kernel").is_none());
    assert_eq!(scalar["snapshot"]["operation"], "saved");
    assert_eq!(scalar["output"]["head_count"], 1);
    // The trust pin comes from successful verified source conversion, NOT the
    // snapshot header or an unauthenticated checksum from that same file.
    let pin = scalar["model_content_sha256"].as_str().unwrap().to_owned();
    for mode in ["exclusive", "multi_label"] {
        let mut reload = parsed(&["--snapshot", &snapshot, "--expected-model-sha256", &pin,
            "--kernel", kind.id()]).unwrap();
        reload.args.memory_mib = memory;
        assert!(reload.args.source_dir.is_none()); assert!(reload.args.source_manifest.is_none());
        let text = request_json(mode); let mut bytes = Vec::new();
        reload.execute(&mut text.as_bytes(), &mut bytes).unwrap();
        let selected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(selected["integer_kernel"], kind.id());
        assert_eq!(selected["snapshot"]["operation"], "loaded");
        assert_eq!(selected["model_content_sha256"], pin);
        assert_eq!(selected["output"]["numerics_profile"], "strict-quantized-v2");
        assert_eq!(selected["output"]["result"]["mode"], mode);
        if mode == "exclusive" { assert_eq!(selected["output"], scalar["output"]); }
        else {
            assert_eq!(selected["output"]["head_count"], 2);
            assert_eq!(selected["output"]["result"]["result"]["labels"].as_array().unwrap().len(), 2);
        }
    }
    // The explicitly requested snapshot is preserved, including on a later
    // failed assertion. No model-quality or performance claim is inferred.
}
