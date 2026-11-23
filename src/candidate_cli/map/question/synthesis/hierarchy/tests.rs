//! Public option/admission contracts; no native model execution.
use super::*;
use crate::candidate_cli::map::tests::command;

#[test]
fn hierarchy_and_its_limits_require_explicit_answer_synthesis() {
    for extra in [vec!["--hierarchical-answer"], vec!["--synthesize-answer", "--answer-max-passes", "2"]] {
        let mut flags = vec!["candidate", "map", "--task", "answer", "--question", "q", "--model", "m", "--memory-mib", "8192"];
        flags.extend(extra);
        assert!(crate::candidate_cli::definition().try_get_matches_from(flags).is_err());
    }
    for task in ["ner", "keyphrases", "summarize", "extract"] {
        assert!(command(task, &["--question", "q", "--synthesize-answer", "--hierarchical-answer"]).validate().is_err());
    }
}
#[test]
fn all_hierarchy_controls_are_bounded_before_any_input() {
    for (flag, value) in [("--answer-max-levels", "0"), ("--answer-max-levels", "17"),
        ("--answer-max-passes", "0"), ("--answer-max-passes", "257"),
        ("--answer-max-tokenizer-calls", "0"), ("--answer-max-tokenizer-calls", "65537"),
        ("--answer-max-tokenizer-bytes", "0"), ("--answer-max-tokenizer-bytes", "1000000001")] {
        assert!(command("answer", &["--question", "q", "--synthesize-answer", "--hierarchical-answer", flag, value]).validate().is_err());
    }
}
#[test]
fn hierarchy_preserves_global_allowances_and_only_explicit_synthesis_expands_passages() {
    let plain = command("answer", &["--question", "q"]);
    assert!(plain.question.synthesis.hierarchy.limits(false).unwrap().is_none());
    let tree = command("answer", &["--question", "q", "--synthesize-answer", "--hierarchical-answer",
        "--max-synthesis-passages", "128", "--input-encoding", "safe-bpe-v1"]);
    tree.validate().unwrap();
    assert_eq!(tree.question.synthesis.hierarchy.limits(true).unwrap(), Some(QuestionHierarchyLimits::default()));
    assert_eq!(plain.work_ceiling(), tree.work_ceiling());
    assert_eq!(plain.max_total_mask_node_visits, tree.max_total_mask_node_visits);
    assert_eq!(plain.question.synthesis.planning(plain.host.planning()).max_passages, source::MAX_PASSAGES);
    assert_eq!(tree.question.synthesis.planning(tree.host.planning()).max_passages, 128);
    let request = tree.question.synthesis.request(tree.question.task("Who?".to_owned(), None).unwrap(), &tree.host).unwrap();
    assert_eq!(request.limits.max_evidence_passages, 128);
}
#[test]
fn hierarchy_rejects_an_impossible_simultaneous_memory_quote_before_io() {
    let mut tree = command("answer", &["--question", "q", "--synthesize-answer", "--hierarchical-answer"]);
    tree.validate().unwrap();
    // Each preparation reserve fits individually, but both retained owners
    // plus native KV/output and every hierarchy transcript cannot fit.
    tree.host.memory_mib = 1024;
    tree.host.max_weight_mib = 512;
    assert!(tree.validate().is_err());
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_hierarchy_never_reads_source_question_options_or_model() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    let command = command("answer", &["--question", "never-open", "--options", "never-open",
        "--synthesize-answer", "--hierarchical-answer", "--input-encoding", "safe-bpe-v1"]);
    assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}
