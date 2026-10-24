//! Input/admission contracts; real native execution belongs to an explicit gate.
use super::*;
use crate::candidate_cli::judge::tests::fixture;
use serde_json::json;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "judge", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 judge dispatcher"); };
    Ok(command)
}
fn command(extra: &[&str]) -> Int4Command {
    let mut flags = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    flags.extend_from_slice(extra); parsed(&flags).unwrap()
}
fn options(command: &Int4Command) -> JudgeArgs {
    let Int4Task::Judge(options) = command.task else { panic!("judge required"); };
    options
}
fn budget() -> TaskBudget {
    let command = command(&[]); let selected = options(&command);
    selected.scoring.budget(selected.validate(&command.args).unwrap())
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused judge reached private input"); }
}

#[test]
fn all_judge_modes_share_explicit_source_and_independently_pinned_snapshot_routes() {
    let source = command(&[]); let selected = options(&source);
    let limits = selected.validate(&source.args).unwrap();
    assert_eq!(selected.scoring.budget(limits).max_kv_bytes, limits.native.kv_bytes);
    assert_eq!(source.args.preparation_mib, 512);
    assert!(limits.prefill.is_none());
    let pin = "ab".repeat(32);
    let reload = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &pin]).unwrap();
    let snapshot = options(&reload).validate(&reload.args).unwrap();
    assert_eq!(snapshot.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(snapshot.native.kv_bytes, limits.native.kv_bytes);
    for flags in [vec![], vec!["--snapshot", "s"], vec!["--model", "int8.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--save-snapshot", "new"]] {
        assert!(parsed(&flags).is_err());
    }
}

#[test]
fn typed_wire_keeps_each_modes_exact_text_and_host_owned_budget() {
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let request = parse_request(&fixture(mode).to_string(), budget(), 65536).unwrap();
        let actual = match &request {
            JudgeRequest::Pairwise { budget, .. } | JudgeRequest::Rubric { budget, .. }
                | JudgeRequest::Faithfulness { budget, .. } => *budget,
        };
        assert_eq!(actual, budget());
        assert_eq!(request_shape(&request).unwrap().0, mode);
    }
    let exact = "  é <tool_call> 上海\r\n😀  ";
    let mut pair = fixture("pairwise"); pair["a"] = json!(exact);
    pair["policy"]["maximum_order_disagreement_milli"] = json!(u32::MAX);
    let JudgeRequest::Pairwise { a, policy, .. } = parse_request(&pair.to_string(), budget(), 65536).unwrap()
        else { panic!("pairwise request lost"); };
    assert_eq!(a, exact);
    assert_eq!(policy.maximum_order_disagreement_milli, u32::MAX);
    let mut rubric = fixture("rubric"); rubric["document"] = json!(exact);
    let JudgeRequest::Rubric { document, rubric: definition, .. } =
        parse_request(&rubric.to_string(), budget(), 65536).unwrap() else { panic!("rubric request lost"); };
    assert_eq!(document, exact);
    assert_eq!(definition.criteria.len(), 2);
    assert_eq!(definition.declared_origin_digest.to_hex(),
        rubric["rubric"]["declared_origin_digest"].as_str().unwrap());
    let mut faithfulness = fixture("faithfulness"); faithfulness["source"] = json!(exact);
    let JudgeRequest::Faithfulness { source, .. } =
        parse_request(&faithfulness.to_string(), budget(), 65536).unwrap() else { panic!("faithfulness request lost"); };
    assert_eq!(source, exact);
}

#[test]
fn complete_population_includes_both_orders_all_criteria_and_every_evidence_window() {
    for (mode, heads) in [("pairwise", 2), ("rubric", 2), ("faithfulness", 1)] {
        let request = parse_request(&fixture(mode).to_string(), budget(), 65536).unwrap();
        assert_eq!(request_shape(&request).unwrap(), (mode, heads));
    }
    let mut value = fixture("faithfulness"); value["source"] = json!("é上海x");
    value["policy"]["evidence_window_bytes"] = json!(4);
    let request = parse_request(&value.to_string(), budget(), 65536).unwrap();
    // Exact UTF-8 windows are é, 上 and 海x, plus the separate full source.
    assert_eq!(request_shape(&request).unwrap(), ("faithfulness", 4));
    value["policy"]["max_evidence_windows"] = json!(2);
    value["policy"]["max_evidence_spans"] = json!(2);
    assert!(matches!(parse_request(&value.to_string(), budget(), 65536), Err(CandidateError::Input)));
}

#[test]
fn all_modes_require_explicit_policy_and_reject_execution_overrides_or_duplicate_keys() {
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let mut missing = fixture(mode); missing.as_object_mut().unwrap().remove("policy");
        assert!(matches!(parse_request(&missing.to_string(), budget(), 65536), Err(CandidateError::Input)));
        for field in ["model", "budget", "work", "numerics_profile", "eos_token_id", "score_space",
            "prompt", "execution_identity", "seed", "tools", "thinking"] {
            let mut value = fixture(mode); value[field] = json!({});
            assert!(matches!(parse_request(&value.to_string(), budget(), 65536), Err(CandidateError::Input)));
        }
        let text = fixture(mode).to_string();
        assert!(matches!(parse_request(&text, budget(), text.len() - 1), Err(CandidateError::Input)));
        let duplicate = text.replacen("\"policy\":", "\"policy\":{},\"policy\":", 1);
        assert!(matches!(parse_request(&duplicate, budget(), 65536), Err(CandidateError::Input)));
    }
    for text in ["[]", "null", "{}", r#"{"mode":"unknown"}"#,
        r#"{"mode":"pairwise","\u006dode":"rubric"}"#] {
        assert!(matches!(parse_request(text, budget(), 65536), Err(CandidateError::Input)));
    }
}

#[test]
fn free_generation_or_invalid_resources_refuse_before_document_or_model_io() {
    for flags in [vec!["--max-new-tokens", "128"], vec!["--max-output-bytes", "65536"],
        vec!["--seed", "ab"], vec!["--temperature-milli", "1000"], vec!["--top-k", "1"],
        vec!["--logprobs"], vec!["--prefill-rows", "1"], vec!["--min-new-tokens", "0"],
        vec!["--stop", "END"], vec!["--ban-token", "7"], vec!["--logit-bias", "7=0"]] {
        let mut all = vec!["--source-dir", "d", "--source-manifest", "m"];
        all.extend(flags); assert!(parsed(&all).is_err());
    }
    for flags in [vec!["--max-candidate-tokens", "1"], vec!["--max-candidate-tokens", "65"],
        vec!["--max-result-bytes", "0"], vec!["--max-forward-positions", "0"],
        vec!["--max-projected-logits", "0"], vec!["--max-attention-pairs", "0"],
        vec!["--max-dot-products", "0"], vec!["--max-multiply-accumulates", "0"],
        vec!["--context-tokens", "16"], vec!["--preparation-mib", "1"], vec!["--group-width", "0"]] {
        let mut bytes = Vec::new();
        assert_eq!(command(&flags).execute(&mut NoRead, &mut bytes), Err(CandidateError::Arguments));
        assert!(bytes.is_empty());
    }
    let mut changed = command(&[]); changed.args.logprobs = true;
    assert_eq!(changed.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[test]
fn memory_admission_prices_both_preparation_owners_and_full_live_kv() {
    for width in [3, 16, 128, 10752] {
        let mut command = command(&[]);
        command.args.group_width = width;
        command.args.max_load_mib = 65536; command.args.memory_mib = 65536;
        let options = options(&command);
        let single = options.scoring.validate(&command.args).unwrap();
        let limits = options.validate(&command.args).unwrap();
        assert_eq!(limits.minimum_memory_bytes, single.minimum_memory_bytes + single.preparation_bytes);
        assert_eq!(limits.native.kv_bytes,
            Int4MemoryRequirement::for_context(command.args.context_tokens, width).unwrap().kv_bytes);
        command.args.memory_mib = limits.minimum_memory_bytes.saturating_sub(1) / MIB;
        assert!(options.validate(&command.args).is_err());
        command.args.memory_mib = limits.minimum_memory_bytes.div_ceil(MIB);
        assert!(options.validate(&command.args).is_ok());
    }
}

#[test]
fn all_five_work_axes_and_language_limits_cover_the_complete_judgment() {
    let command = command(&[]); let options = options(&command);
    let task = options.scoring.budget(options.validate(&command.args).unwrap());
    let cap = options.scoring.work_ceiling();
    let limits = options.scoring.judge_limits(task);
    assert_eq!(limits.max_total_prompt_tokens as u64, cap.forward_positions);
    assert_eq!(limits.max_total_projected_logits, cap.projected_logits);
    assert_eq!(limits.max_total_candidate_tokens, task.max_grammar_states as usize);
    assert_eq!(limits.max_output_bytes, task.max_output_bytes);
    assert_eq!(limits.per_head.max_candidates, 33);
    assert_eq!(limits.per_head.max_depth, task.max_output_tokens as usize);
    assert_eq!(limits.per_head.max_nodes, task.max_grammar_states as usize);
    options.scoring.admit_plan(8, 8, cap).unwrap();
    for axis in 0..5 {
        let mut excess = cap;
        match axis {
            0 => excess.forward_positions += 1,
            1 => excess.projected_logits += 1,
            2 => excess.attention_pairs += 1,
            3 => excess.projections.dot_products += 1,
            _ => excess.projections.multiply_accumulates += 1,
        }
        assert_eq!(options.scoring.admit_plan(8, 8, excess), Err(CandidateError::Planning));
    }
    assert_eq!(options.scoring.admit_plan(9, 8, cap), Err(CandidateError::Planning));
    assert_eq!(options.scoring.admit_plan(0, 8, cap), Err(CandidateError::Planning));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_never_opens_private_input_or_sources_or_publishes() {
    let mut bytes = Vec::new();
    assert_eq!(command(&[]).execute(&mut NoRead, &mut bytes), Err(CandidateError::Unavailable));
    assert!(bytes.is_empty());
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, a fresh snapshot and an available explicit SIMD kernel"]
fn all_three_modes_match_between_source_scalar_and_pinned_snapshot_simd() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let snapshot = std::env::var("FNLP_INT4_TEST_JUDGE_SNAPSHOT").expect("explicit new snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture requires a fresh path and deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required")
        .parse().expect("canonical kernel ID required");
    assert_ne!(kernel, Int8KernelKind::Scalar);
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be available on this actual host");
    let mut trusted_pin = None;
    for (index, mode) in ["pairwise", "rubric", "faithfulness"].into_iter().enumerate() {
        let mut wire = fixture(mode);
        if mode == "faithfulness" { wire["policy"]["evidence_window_bytes"] = json!(8); }
        let request = wire.to_string();
        let mut convert = command(&[]);
        convert.args.source_dir = Some(PathBuf::from(&source));
        convert.args.source_manifest = Some(PathBuf::from(&manifest));
        convert.args.memory_mib = memory;
        if index == 0 { convert.args.save_snapshot = Some(PathBuf::from(&snapshot)); }
        let mut bytes = Vec::new(); convert.execute(&mut request.as_bytes(), &mut bytes).unwrap();
        let scalar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let expected = request_shape(&parse_request(&request, budget(), 65536).unwrap()).unwrap().1;
        assert_eq!(scalar["output"]["head_count"], expected);
        assert_eq!(scalar["output"]["result"]["mode"], mode);
        assert_eq!(scalar["output"]["numerics_profile"], "strict-quantized-v2");
        let pin = scalar["model_content_sha256"].as_str().unwrap().to_owned();
        if index == 0 {
            assert_eq!(scalar["snapshot"]["operation"], "saved");
            trusted_pin = Some(pin.clone());
        }
        assert_eq!(trusted_pin.as_deref(), Some(pin.as_str()));
        // The independent pin comes from source conversion, not the snapshot header.
        let mut reload = parsed(&["--snapshot", &snapshot, "--expected-model-sha256", &pin,
            "--kernel", kernel.id()]).unwrap();
        reload.args.memory_mib = memory;
        let mut bytes = Vec::new(); reload.execute(&mut request.as_bytes(), &mut bytes).unwrap();
        let selected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(selected["snapshot"]["operation"], "loaded");
        assert_eq!(selected["integer_kernel"], kernel.id());
        assert_eq!(selected["output"], scalar["output"]);
    }
    // The caller-owned snapshot is preserved on success and on failure.
}
