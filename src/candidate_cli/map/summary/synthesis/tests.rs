//! CLI shape and resource tests only; no neural execution receipts.
use super::*;
use crate::candidate_cli::map::tests::command;
#[test]
fn opt_in_synthesis_retains_map_options_and_has_a_separate_bound_final_schema() {
    let c = command("summarize", &["--synthesize-summary", "--synthesis-bullets", "2"]);
    c.validate().unwrap();
    let map = SummaryOptions { max_bullets: 5, ..SummaryOptions::default() };
    let request = c.summary.synthesis.request(map, c.host.planning()).unwrap();
    assert_eq!(request.map_options, map); assert_eq!(request.synthesis_options.max_bullets, 2);
    assert_eq!(request.synthesis_options.max_quote_scalars, map.max_quote_scalars);
    assert!(c.summary.limits(c.max_map_result_bytes).unwrap().is_none());
    assert!(!c.summary.reduce_summary);
}
#[test]
fn unrelated_tasks_and_conflicting_reducers_cannot_acquire_synthesis() {
    for task in ["ner", "keyphrases", "answer", "extract"] {
        assert!(command(task, &["--synthesize-summary"]).validate().is_err());
    }
    let root = || crate::candidate_cli::definition();
    assert!(root().try_get_matches_from(["candidate", "map", "--task", "summarize",
        "--model", "m", "--memory-mib", "8192", "--synthesize-summary", "--reduce-summary"]).is_err());
    assert!(root().try_get_matches_from(["candidate", "map", "--task", "summarize",
        "--model", "m", "--memory-mib", "8192", "--synthesis-bullets", "2"]).is_err());
}
#[test]
fn zero_unbounded_and_overlarge_synthesis_limits_fail_closed() {
    for (flag, value) in [("--synthesis-bullets", "0"), ("--synthesis-bullets", "1025"),
        ("--max-synthesis-evidence-segments", "0"), ("--max-synthesis-evidence-segments", "1025"),
        ("--max-synthesis-evidence-bytes", "0"), ("--max-synthesis-evidence-bytes", "1048577"),
        ("--max-synthesis-fields", "0"), ("--max-synthesis-fields", "1000001"),
        ("--max-synthesis-matches", "0"), ("--max-synthesis-matches", "1000001"),
        ("--max-synthesis-scan-steps", "0"), ("--max-synthesis-scan-steps", "1000000000001")] {
        assert!(command("summarize", &["--synthesize-summary", flag, value]).validate().is_err(), "{flag}={value}");
    }
}
#[test]
fn actual_preparation_limits_are_checked_before_io_and_do_not_expand_with_more_chunks() {
    let c = command("summarize", &["--synthesize-summary", "--max-synthesis-evidence-bytes", "1024"]);
    let mut planning = c.host.planning(); planning.max_input_bytes = 1023;
    assert!(c.summary.synthesis.check_planning(planning).is_err());
    let a = command("summarize", &["--synthesize-summary"]);
    let b = command("summarize", &["--synthesize-summary", "--max-chunks", "128", "--preparation-mib", "1024"]);
    let a_limits = a.summary.synthesis.request(SummaryOptions::default(), a.host.planning()).unwrap().limits;
    let b_limits = b.summary.synthesis.request(SummaryOptions::default(), b.host.planning()).unwrap().limits;
    assert_eq!(a_limits, b_limits); assert_eq!(a.work_ceiling(), b.work_ceiling());
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_reads_no_private_source_options_or_model_and_writes_nothing() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected output") }
        fn flush(&mut self) -> io::Result<()> { panic!("unexpected flush") }
    }
    let c = command("summarize", &["--synthesize-summary", "--options", "never-open.json"]);
    assert_eq!(c.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}
