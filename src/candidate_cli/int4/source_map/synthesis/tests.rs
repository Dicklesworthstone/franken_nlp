use super::*;

fn arguments(extra: &[&str]) -> (MapArgs, Int4Args) {
    let mut flags = vec!["--synthesize-summary"];
    flags.extend_from_slice(extra);
    let command = super::super::tests::command(Kind::Summarize, &flags);
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong summary task"); };
    (options, command.args)
}
fn parsed(kind: &str, extra: &[&str]) -> Result<clap::ArgMatches, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", kind, "--source-dir", "source",
        "--source-manifest", "manifest", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    crate::candidate_cli::definition().try_get_matches_from(argv)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused synthesis reached IO"); }
}

#[test]
fn explicit_synthesis_preserves_the_default_exact_union_and_whole_work_caps() {
    let plain = super::super::tests::command(Kind::Summarize, &[]);
    let Int4Task::SourceMap(defaults) = plain.task else { panic!("wrong map"); };
    let default_limits = defaults.validate(&plain.args).unwrap();
    assert!(default_limits.summary.is_some());
    assert!(default_limits.synthesis.is_none());
    let (synthesis, args) = arguments(&[]);
    let limits = synthesis.validate(&args).unwrap();
    assert!(limits.summary.is_none());
    let requested = limits.synthesis.unwrap();
    assert_eq!(requested.max_evidence_segments, 256);
    assert_eq!(requested.max_evidence_bytes, 65_536);
    assert_eq!(requested.verification, GroundingBudget::default());
    assert_eq!(limits.mapping.max_model_work, default_limits.mapping.max_model_work);
    assert_eq!(limits.mapping.max_mask_visits, default_limits.mapping.max_mask_visits);
    assert!(limits.workspace_bytes > default_limits.workspace_bytes);
    assert_eq!(limits.max_result_bytes, default_limits.max_result_bytes);
}

#[test]
fn synthesis_controls_are_exclusive_to_explicit_summary_synthesis() {
    for flag in ["--synthesis-bullets", "--max-synthesis-evidence-segments",
        "--max-synthesis-evidence-bytes", "--max-synthesis-fields",
        "--max-synthesis-matches", "--max-synthesis-scan-steps"] {
        assert!(parsed("summarize-map", &[flag, "1"]).is_err());
    }
    for kind in ["ner-map", "keyphrases-map", "answer-map", "summarize"] {
        assert!(parsed(kind, &["--synthesize-summary"]).is_err());
    }
    for flag in ["--max-summary-bullets", "--max-summary-unique-bullets",
        "--max-summary-citations", "--max-summary-evidence-spans",
        "--max-summary-value-bytes", "--max-summary-scan-steps"] {
        assert!(parsed("summarize-map", &["--synthesize-summary", flag, "1"]).is_err());
    }
    assert!(parsed("summarize-map", &["--hierarchical-summary"]).is_err());
    let (mut map, args) = arguments(&["--synthesis-bullets", "2"]);
    map.summary.as_mut().unwrap().synthesis.synthesize_summary = false;
    assert_eq!(map.validate(&args).unwrap_err(), CandidateError::Arguments);
}

#[test]
fn all_evidence_and_verification_limits_reject_before_input_or_model_io() {
    for (flag, value) in [("--synthesis-bullets", "0"), ("--synthesis-bullets", "1025"),
        ("--max-synthesis-evidence-segments", "0"), ("--max-synthesis-evidence-segments", "1025"),
        ("--max-synthesis-evidence-bytes", "0"), ("--max-synthesis-evidence-bytes", "1048577"),
        ("--max-synthesis-fields", "0"), ("--max-synthesis-fields", "1000001"),
        ("--max-synthesis-matches", "0"), ("--max-synthesis-matches", "1000001"),
        ("--max-synthesis-scan-steps", "0"), ("--max-synthesis-scan-steps", "1000000000001")] {
        let (map, args) = arguments(&[flag, value]);
        let mut output = Vec::new();
        assert_eq!(map.execute(args, &mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn final_bullet_cap_changes_only_the_final_neural_options() {
    let (map, args) = arguments(&["--synthesis-bullets", "3", "--max-synthesis-evidence-segments", "17",
        "--max-synthesis-evidence-bytes", "4096", "--max-synthesis-fields", "23",
        "--max-synthesis-matches", "41", "--max-synthesis-scan-steps", "8192"]);
    let limits = map.validate(&args).unwrap();
    let text = "é\r\n上海 <tool_call> private source \t";
    let raw = r#"{"max_bullets":2,"max_bullet_scalars":80,"max_citations_per_bullet":1,"max_quote_scalars":24}"#;
    let (retained, task) = map.request(text.to_owned(), Some(raw), map.source.budget(&args, limits.common),
        args.max_input_bytes).unwrap();
    assert_eq!(retained, text);
    let SourceMapTask::Summarize(map_options) = task else { panic!("wrong summary options"); };
    let request = map.summary.unwrap().synthesis.request(map_options, map.source.planning(&args)).unwrap();
    assert_eq!(request.map_options.max_bullets, 2);
    assert_eq!(request.synthesis_options.max_bullets, 3);
    assert_eq!(request.synthesis_options.max_bullet_scalars, 80);
    assert_eq!(request.synthesis_options.max_quote_scalars, 24);
    assert_eq!(request.limits, limits.synthesis.unwrap());
    assert_eq!(request.limits.max_evidence_segments, 17);
    assert_eq!(request.limits.max_evidence_bytes, 4096);
    assert_eq!(request.limits.verification, GroundingBudget { max_fields: 23, max_matches: 41, max_scan_steps: 8192 });
}

#[test]
fn source_and_independently_pinned_snapshot_use_the_same_synthesis_limits() {
    let (map, args) = arguments(&[]);
    let source = map.validate(&args).unwrap();
    let digest = "ab".repeat(32);
    let matches = definitions().into_iter().find(|c| c.get_name() == "summarize-map").unwrap()
        .try_get_matches_from(["summarize-map", "--synthesize-summary", "--snapshot", "model.snapshot",
            "--expected-model-sha256", &digest, "--memory-mib", "8192"]).unwrap();
    let options = MapArgs::from_matches(Kind::Summarize, &matches).unwrap();
    let snapshot = options.validate(&Int4Args::from_arg_matches(&matches).unwrap()).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(source.synthesis, snapshot.synthesis);
    assert_eq!(source.mapping.max_model_work, snapshot.mapping.max_model_work);
    assert_eq!(source.workspace_bytes, snapshot.workspace_bytes);
}

#[test]
fn synthesis_memory_admission_prices_evidence_copies_and_one_more_native_result() {
    let (map, mut args) = arguments(&[]);
    let limits = map.validate(&args).unwrap();
    let request = limits.synthesis.unwrap();
    let budget = map.source.budget(&args, limits.common);
    let base = super::super::workspace_bytes(budget, limits.mapping, map.mapping.max_chunks,
        limits.reduction_reserve_bytes).unwrap();
    let extra = 4 * budget.max_output_bytes + 8 * u64::from(budget.max_output_tokens)
        + 64 * request.max_evidence_bytes as u64 + 128 * request.max_evidence_segments as u64
        + 128 * request.verification.max_matches as u64 + 128 * request.verification.max_fields as u64;
    assert_eq!(limits.workspace_bytes, base + extra);
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(map.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(map.validate(&args).is_ok());
    let mut overflowing = budget; overflowing.max_output_bytes = u64::MAX;
    assert!(workspace_bytes(overflowing, limits.mapping, 1, limits.reduction_reserve_bytes, request).is_err());
    assert!(workspace_bytes(budget, limits.mapping, 0, limits.reduction_reserve_bytes, request).is_err());
}

#[test]
fn evidence_defaults_respect_the_original_source_cap_and_fixed_group_work_ceiling() {
    let (map, mut args) = arguments(&["--max-input-bytes", "16384"]);
    let initial = map.validate(&args).unwrap();
    assert_eq!(initial.synthesis.unwrap().max_evidence_bytes, 16384);
    for width in [16, 32, 128] {
        args.group_width = width;
        let current = map.validate(&args).unwrap();
        assert_eq!(current.mapping.max_model_work, initial.mapping.max_model_work);
        assert_eq!(current.mapping.max_mask_visits, initial.mapping.max_mask_visits);
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_runtime_refuses_synthesis_without_reading_any_private_file() {
    let (map, args) = arguments(&["--options", "private-options.json"]);
    let mut output = Vec::new();
    assert_eq!(map.execute(args, &mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
