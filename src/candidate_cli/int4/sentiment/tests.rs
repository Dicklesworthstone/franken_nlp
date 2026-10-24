//! Wire/admission fixtures; native inference is confined to one ignored gate.
use super::*;
use crate::tasks::sentiment::SentimentAxis;

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "sentiment", "--memory-mib", "8192"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 sentiment dispatcher"); };
    Ok(command)
}
fn command(extra: &[&str]) -> Int4Command {
    let mut flags = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    flags.extend_from_slice(extra); parsed(&flags).unwrap()
}
fn options(command: &Int4Command) -> SentimentArgs {
    let Int4Task::Sentiment(options) = command.task else { panic!("sentiment required"); };
    options
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused sentiment reached input"); }
}

#[test]
fn independent_sentiment_uses_both_source_and_independently_pinned_snapshot_routes() {
    let source = command(&[]); let selected = options(&source);
    let limits = selected.validate(&source.args).unwrap();
    let budget = selected.scoring.budget(limits);
    assert_eq!(budget.max_kv_bytes, limits.native.kv_bytes);
    assert_eq!(budget.max_input_tokens + budget.max_output_tokens - 1, source.args.context_tokens as u32);
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
fn one_typed_wire_contract_retains_exact_document_axes_and_uncalibrated_policy() {
    let command = command(&[]); let options = options(&command);
    let budget = options.scoring.budget(options.validate(&command.args).unwrap());
    let text = r#"{"document":"é\r\n上海 <tool_call> mixed feelings"}"#;
    let (request, policy) = parse_request(text, budget, 4096).unwrap();
    assert_eq!(request.document, "é\r\n上海 <tool_call> mixed feelings");
    assert_eq!(request.axes, SentimentAxis::ALL);
    assert_eq!(request.budget, budget);
    assert_eq!(policy.minimum_peak_weight_ppm, 0);
    assert_eq!(policy.maximum_normalized_entropy_ppm, 1_000_000);
    let (selected, policy) = parse_request(r#"{"document":"exact", "axes":["approach","valence"],
        "policy":{"minimum_peak_weight_ppm":600000,"maximum_normalized_entropy_ppm":700000}}"#,
        budget, 4096).unwrap();
    assert_eq!(selected.axes, [SentimentAxis::Approach, SentimentAxis::Valence]);
    assert_eq!(policy.minimum_peak_weight_ppm, 600000);
    assert_eq!(policy.maximum_normalized_entropy_ppm, 700000);
    assert!(parse_request(text, budget, text.len() - 1).is_err());
}

#[test]
fn wire_input_cannot_replace_native_semantics_or_duplicate_an_axis() {
    let command = command(&[]); let options = options(&command);
    let budget = options.scoring.budget(options.validate(&command.args).unwrap());
    for field in ["model", "budget", "work", "numerics_profile", "eos_token_id", "score_space",
        "mode", "prompt", "labels", "anchors", "seed", "tools", "thinking"] {
        let mut value = serde_json::json!({"document":"exact"});
        value[field] = serde_json::json!({});
        assert!(matches!(parse_request(&value.to_string(), budget, 4096), Err(CandidateError::Input)));
    }
    for invalid in ["{}", "[]", r#"{"document":""}"#, r#"{"document":"a","document":"b"}"#,
        r#"{"document":"a","axes":[]}"#, r#"{"document":"a","axes":["joy"]}"#,
        r#"{"document":"a","axes":["valence","valence"]}"#,
        r#"{"document":"a","policy":{"minimum_peak_weight_ppm":1000001,"maximum_normalized_entropy_ppm":0}}"#,
        r#"{"document":"a","policy":{"minimum_peak_weight_ppm":0}}"#] {
        assert!(matches!(parse_request(invalid, budget, 4096), Err(CandidateError::Input)));
    }
}

#[test]
fn generation_switches_and_invalid_ceilings_refuse_before_any_input_or_model_io() {
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
fn memory_admission_keeps_two_preparations_full_context_kv_and_complete_output() {
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
fn all_five_native_work_axes_and_scoring_limits_cover_the_entire_bundle() {
    let command = command(&[]); let options = options(&command);
    let cap = options.scoring.work_ceiling();
    let limits = options.scoring.sentiment_limits(&command.args);
    assert_eq!(limits.max_total_prompt_tokens as u64, cap.forward_positions);
    assert_eq!(limits.max_total_projected_logits, cap.projected_logits);
    assert_eq!(limits.max_total_candidate_tokens, 4096);
    assert_eq!(limits.max_total_nodes, 4100);
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
fn unavailable_runtime_never_opens_input_or_sources_or_publishes() {
    let mut bytes = Vec::new();
    assert_eq!(command(&[]).execute(&mut NoRead, &mut bytes), Err(CandidateError::Unavailable));
    assert!(bytes.is_empty());
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, a fresh snapshot and an available explicit SIMD kernel"]
fn source_scalar_and_pinned_snapshot_simd_complete_the_same_independent_axes() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let snapshot = std::env::var("FNLP_INT4_TEST_SENTIMENT_SNAPSHOT").expect("explicit new snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture requires a fresh path and deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required")
        .parse().expect("canonical kernel ID required");
    assert_ne!(kernel, Int8KernelKind::Scalar);
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be available on this actual host");
    let request = r#"{"document":"The result was useful, although the delay was frustrating."}"#;
    let mut convert = command(&["--save-snapshot", &snapshot]);
    convert.args.source_dir = Some(PathBuf::from(source));
    convert.args.source_manifest = Some(PathBuf::from(manifest));
    convert.args.memory_mib = memory;
    let mut bytes = Vec::new(); convert.execute(&mut request.as_bytes(), &mut bytes).unwrap();
    let scalar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(scalar["snapshot"]["operation"], "saved");
    assert_eq!(scalar["output"]["head_count"], 4);
    assert_eq!(scalar["output"]["numerics_profile"], "strict-quantized-v2");
    for dimension in scalar["output"]["result"]["dimensions"].as_array().unwrap() {
        assert_eq!(dimension["anchors"].as_array().unwrap().len(), 5);
        assert_eq!(dimension["scores"]["candidates"].as_array().unwrap().len(), 5);
        assert_eq!(dimension["scores"]["full_vocab_denominators_computed"], true);
        assert_eq!(dimension["scores"]["eos_rule"], "append_and_score_exactly_one_eos");
    }
    // Obtain the independent pin from successful source conversion, never
    // from the snapshot header whose authenticity it is intended to establish.
    let pin = scalar["model_content_sha256"].as_str().unwrap();
    let mut reload = parsed(&["--snapshot", &snapshot, "--expected-model-sha256", pin,
        "--kernel", kernel.id()]).unwrap();
    reload.args.memory_mib = memory;
    let mut bytes = Vec::new(); reload.execute(&mut request.as_bytes(), &mut bytes).unwrap();
    let selected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(selected["snapshot"]["operation"], "loaded");
    assert_eq!(selected["integer_kernel"], kernel.id());
    assert_eq!(selected["output"], scalar["output"]);
    // The caller-requested snapshot is preserved on success and failure.
}
