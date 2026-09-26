//! Command, typed-option and finite-resource checks, not inference fixtures.
use super::*;

pub(in crate::candidate_cli) fn command(task: &str, extra: &[&str]) -> MapCommand {
    let mut args = vec!["candidate", "map", "--task", task, "--model", "local.fnlpq", "--memory-mib", "8192"];
    args.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(args).unwrap();
    let CandidateCommand::Map(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("map route was replaced") };
    command
}
#[test]
fn all_three_tasks_require_explicit_task_model_and_memory() {
    for task in ["ner", "keyphrases", "summarize"] {
        command(task, &[]).validate().unwrap();
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", task]).is_err());
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", task,
            "--model", "m"]).is_err());
    }
    assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--model", "m",
        "--memory-mib", "8192"]).is_err());
}
#[test]
fn qa_generation_schema_and_sampling_cannot_enter_a_source_map() {
    for task in ["answer", "generate", "extract", "classify"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", task,
            "--model", "m", "--memory-mib", "8192"]).is_err());
    }
    for flag in ["--seed", "--schema", "--top-k", "--instruction"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "map", "--task", "ner",
            "--model", "m", "--memory-mib", "8192", flag, "private"]).is_err());
    }
}
#[test]
fn finite_limits_refuse_zero_overflow_and_inconsistent_envelopes() {
    for (flag, value) in [("--max-input-bytes", "3"), ("--max-chunks", "0"), ("--max-chunks", "257"),
        ("--max-chunk-bytes", "3"), ("--max-chunk-bytes", "1048577"),
        ("--max-tokenizer-calls", "0"), ("--max-tokenizer-calls", "1000001"),
        ("--max-map-result-bytes", "0"), ("--max-map-result-bytes", "67108865"),
        ("--max-live-value-bytes", "1048576"), ("--max-total-value-bytes", "1048576"),
        ("--reduction-reserve-mib", "0"), ("--reduction-reserve-mib", "18446744073709551615"),
        ("--max-total-mask-node-visits", "0"), ("--max-forward-positions", "0"),
        ("--max-projected-logits", "0"), ("--max-attention-pairs", "0"),
        ("--max-dot-products", "0"), ("--max-multiply-accumulates", "0"), ("--options", "-")] {
        assert!(command("ner", &[flag, value]).validate().is_err(), "{flag}");
    }
}
#[test]
fn retained_prompt_and_grammar_capacity_requires_aggregate_preparation() {
    let small = command("ner", &[]); small.validate().unwrap();
    assert!(command("ner", &["--max-chunks", "256"]).validate().is_err());
    command("ner", &["--max-chunks", "256", "--preparation-mib", "1024"]).validate().unwrap();
}
#[test]
fn raising_chunk_count_never_multiplies_compute_or_mask_authority() {
    let a = command("ner", &[]);
    let b = command("ner", &["--max-chunks", "256", "--preparation-mib", "1024"]);
    assert_eq!(a.work_ceiling(), b.work_ceiling());
    assert_eq!(a.max_total_mask_node_visits, b.max_total_mask_node_visits);
}
#[test]
fn every_whole_document_work_axis_has_an_exact_boundary() {
    let cmd = command("ner", &[]); let cap = cmd.work_ceiling();
    cmd.admit_work(cap, cmd.max_total_mask_node_visits).unwrap();
    for axis in 0..5 {
        let mut bad = cap;
        match axis { 0 => bad.forward_positions += 1, 1 => bad.projected_logits += 1,
            2 => bad.attention_pairs += 1, 3 => bad.projections.dot_products += 1,
            _ => bad.projections.multiply_accumulates += 1 }
        assert_eq!(cmd.admit_work(bad, 1), Err(CandidateError::Planning));
    }
    assert_eq!(cmd.admit_work(cap, cmd.max_total_mask_node_visits + 1), Err(CandidateError::Planning));
}
#[test]
fn options_are_exact_typed_data_not_a_document_or_execution_recipe() {
    let cmd = command("ner", &[]);
    let raw = r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32}"#;
    let SourceMapTask::Ner(options) = cmd.map_task(Some(raw)).unwrap() else { panic!("wrong task") };
    assert_eq!(options.max_entities, 2);
    for bad in ["{}", r#"{"types":[],"max_entities":2,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":2,"\u006dax_entities":3,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32,"document":"secret"}"#] {
        assert!(cmd.map_task(Some(bad)).is_err());
    }
    assert!(cmd.map_task(Some(&" ".repeat(source::OPTIONS_BYTES + 1))).is_err());
}
#[test]
fn help_does_not_mislabel_chunk_summaries_as_global_synthesis() {
    let help = definition().render_long_help().to_string();
    assert!(help.contains("not a global synthesized summary"));
    assert!(help.contains("No partial success"));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_never_opens_input_options_model_or_output() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read") } }
    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    for task in ["ner", "keyphrases", "summarize"] {
        assert_eq!(command(task, &["--options", "never-open.json"]).execute(&mut NoRead, &mut NoWrite),
            Err(CandidateError::Unavailable));
    }
}
