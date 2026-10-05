//! CLI contracts only; no inference or quality evidence.
use super::*;
use crate::candidate_cli::map::tests::command;
#[test]
fn hierarchy_requires_explicit_synthesis_and_its_children_require_hierarchy() {
    let root = || crate::candidate_cli::definition();
    for extra in [vec!["--hierarchical-summary"], vec!["--synthesize-summary", "--summary-max-passes", "2"]] {
        let mut args = vec!["candidate", "map", "--task", "summarize", "--model", "m", "--memory-mib", "8192"];
        args.extend(extra);
        assert!(root().try_get_matches_from(args).is_err());
    }
    for task in ["ner", "keyphrases", "answer", "extract"] {
        assert!(command(task, &["--synthesize-summary", "--hierarchical-summary"]).validate().is_err());
    }
}
#[test]
fn default_single_pass_and_explicit_hierarchy_keep_distinct_admission_policies() {
    let single = command("summarize", &["--synthesize-summary"]);
    assert!(single.summary.synthesis.hierarchy.limits(true).unwrap().is_none());
    let tree = command("summarize", &["--synthesize-summary", "--hierarchical-summary"]);
    tree.validate().unwrap();
    let h = tree.summary.synthesis.hierarchy.limits(true).unwrap().unwrap();
    assert_eq!(h, SummaryHierarchyLimits::default()); assert_eq!(h.max_passes, 16);
    assert!(tree.summary.limits(tree.max_map_result_bytes).unwrap().is_none());
    assert_eq!(single.work_ceiling(), tree.work_ceiling());
    assert_eq!(single.summary.synthesis.request(SummaryOptions::default(), single.host.planning()).unwrap().limits,
        tree.summary.synthesis.request(SummaryOptions::default(), tree.host.planning()).unwrap().limits);
}
#[test]
fn invalid_depth_calls_and_grouping_work_are_rejected_before_input() {
    for (flag, value) in [("--summary-max-levels", "0"), ("--summary-max-levels", "17"),
        ("--summary-max-passes", "0"), ("--summary-max-passes", "257"),
        ("--summary-max-tokenizer-calls", "0"), ("--summary-max-tokenizer-calls", "65537"),
        ("--summary-max-tokenizer-bytes", "0"), ("--summary-max-tokenizer-bytes", "1000000001")] {
        assert!(command("summarize", &["--synthesize-summary", "--hierarchical-summary", flag, value]).validate().is_err());
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_reads_no_source_options_model_or_output() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected output") }
        fn flush(&mut self) -> io::Result<()> { panic!("unexpected flush") }
    }
    let c = command("summarize", &["--synthesize-summary", "--hierarchical-summary", "--options", "never-open.json"]);
    assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}
