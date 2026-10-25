//! Parser and admission fixtures only; no native inference or quality claim.
use super::*;

fn arguments(extra: &[&str]) -> (MapArgs, Int4Args) {
    let mut flags = vec!["--synthesize-summary", "--hierarchical-summary"];
    flags.extend_from_slice(extra);
    let command = super::super::super::tests::command(Kind::Summarize, &flags);
    let Int4Task::SourceMap(options) = command.task else { panic!("wrong hierarchy task"); };
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
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused hierarchy reached IO"); }
}

#[test]
fn hierarchy_requires_explicit_synthesis_and_every_child_requires_hierarchy() {
    assert!(parsed("summarize-map", &["--hierarchical-summary"]).is_err());
    for flag in ["--summary-max-levels", "--summary-max-passes", "--summary-max-tokenizer-calls",
        "--summary-max-tokenizer-bytes"] {
        assert!(parsed("summarize-map", &["--synthesize-summary", flag, "1"]).is_err());
    }
    for kind in ["ner-map", "keyphrases-map", "answer-map", "summarize"] {
        assert!(parsed(kind, &["--synthesize-summary", "--hierarchical-summary"]).is_err());
    }
    let (mut options, args) = arguments(&[]);
    options.summary.as_mut().unwrap().synthesis.synthesize_summary = false;
    assert_eq!(options.validate(&args).unwrap_err(), CandidateError::Arguments);
}

#[test]
fn plain_union_single_pass_and_hierarchy_keep_distinct_policies_with_one_work_ceiling() {
    let plain = super::super::super::tests::command(Kind::Summarize, &[]);
    let single = super::super::super::tests::command(Kind::Summarize, &["--synthesize-summary"]);
    let Int4Task::SourceMap(plain_options) = plain.task else { panic!("wrong union"); };
    let Int4Task::SourceMap(single_options) = single.task else { panic!("wrong synthesis"); };
    let union = plain_options.validate(&plain.args).unwrap();
    let synthesis = single_options.validate(&single.args).unwrap();
    let (options, args) = arguments(&[]);
    let hierarchy = options.validate(&args).unwrap();
    assert!(union.summary.is_some()); assert!(union.synthesis.is_none()); assert!(union.hierarchy.is_none());
    assert!(synthesis.summary.is_none()); assert!(synthesis.synthesis.is_some()); assert!(synthesis.hierarchy.is_none());
    assert!(hierarchy.summary.is_none()); assert_eq!(hierarchy.synthesis, synthesis.synthesis);
    assert_eq!(hierarchy.hierarchy, Some(SummaryHierarchyLimits::default()));
    assert_eq!(hierarchy.mapping.max_model_work, union.mapping.max_model_work);
    assert_eq!(hierarchy.mapping.max_model_work, synthesis.mapping.max_model_work);
    assert_eq!(hierarchy.mapping.max_mask_visits, synthesis.mapping.max_mask_visits);
    assert!(hierarchy.workspace_bytes > synthesis.workspace_bytes);
}

#[test]
fn invalid_level_pass_and_tokenizer_limits_refuse_before_any_input() {
    for (flag, value) in [("--summary-max-levels", "0"), ("--summary-max-levels", "17"),
        ("--summary-max-passes", "0"), ("--summary-max-passes", "257"),
        ("--summary-max-tokenizer-calls", "0"), ("--summary-max-tokenizer-calls", "65537"),
        ("--summary-max-tokenizer-bytes", "0"), ("--summary-max-tokenizer-bytes", "1000000001")] {
        let (options, args) = arguments(&[flag, value]);
        let mut output = Vec::new();
        assert_eq!(options.execute(args, &mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    assert!(parsed("summarize-map", &["--synthesize-summary", "--hierarchical-summary", "--max-summary-bullets", "4"]).is_err());
}

#[test]
fn explicit_limits_and_neural_options_change_no_map_summary_authority() {
    let (options, args) = arguments(&["--summary-max-levels", "3", "--summary-max-passes", "7",
        "--summary-max-tokenizer-calls", "123", "--summary-max-tokenizer-bytes", "456789",
        "--synthesis-bullets", "3"]);
    let limits = options.validate(&args).unwrap();
    assert_eq!(limits.hierarchy.unwrap(), SummaryHierarchyLimits { max_levels: 3, max_passes: 7,
        max_tokenizer_calls: 123, max_tokenizer_bytes: 456789 });
    let map_options = SummaryOptions { max_bullets: 2, max_bullet_scalars: 80,
        max_citations_per_bullet: 1, max_quote_scalars: 24 };
    let request = options.summary.unwrap().synthesis.request(map_options, options.planning(&args)).unwrap();
    assert_eq!(request.map_options, map_options);
    assert_eq!(request.synthesis_options.max_bullets, 3);
    assert_eq!(request.synthesis_options.max_quote_scalars, 24);
}

#[test]
fn hierarchy_workspace_prices_all_pass_results_and_private_level_witnesses() {
    let (options, mut args) = arguments(&["--summary-max-passes", "4", "--summary-max-levels", "2"]);
    let limits = options.validate(&args).unwrap();
    let synthesis = limits.synthesis.unwrap();
    let hierarchy = limits.hierarchy.unwrap();
    let budget = options.source.budget(&args, limits.common);
    let single = super::super::workspace_bytes(budget, limits.mapping, options.mapping.max_chunks,
        limits.reduction_reserve_bytes, synthesis).unwrap();
    let per_pass = 4 * budget.max_output_bytes + 8 * u64::from(budget.max_output_tokens);
    assert_eq!(limits.workspace_bytes, single + 3 * per_pass + 4 * (8192 + 256) + 2 * 128);
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(options.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(options.validate(&args).is_ok());
    let mut invalid = hierarchy; invalid.max_passes = 0;
    assert!(workspace_bytes(budget, limits.mapping, 1, limits.reduction_reserve_bytes, synthesis, invalid).is_err());
    let mut overflowing = budget; overflowing.max_output_bytes = u64::MAX;
    assert!(workspace_bytes(overflowing, limits.mapping, 1, limits.reduction_reserve_bytes, synthesis, hierarchy).is_err());
}

#[test]
fn source_and_snapshot_preserve_the_same_explicit_hierarchy_admission() {
    let (options, args) = arguments(&[]);
    let source = options.validate(&args).unwrap();
    let digest = "ab".repeat(32);
    let matches = definitions().into_iter().find(|c| c.get_name() == "summarize-map").unwrap()
        .try_get_matches_from(["summarize-map", "--synthesize-summary", "--hierarchical-summary",
            "--snapshot", "model.snapshot", "--expected-model-sha256", &digest, "--memory-mib", "8192"]).unwrap();
    let options = MapArgs::from_matches(Kind::Summarize, &matches).unwrap();
    let snapshot = options.validate(&Int4Args::from_arg_matches(&matches).unwrap()).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(snapshot.hierarchy, source.hierarchy);
    assert_eq!(snapshot.synthesis, source.synthesis);
    assert_eq!(snapshot.mapping.max_model_work, source.mapping.max_model_work);
    assert_eq!(snapshot.workspace_bytes, source.workspace_bytes);
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_runtime_refuses_hierarchy_before_source_options_or_model_io() {
    let (options, args) = arguments(&["--options", "private-options.json"]);
    let mut output = Vec::new();
    assert_eq!(options.execute(args, &mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
