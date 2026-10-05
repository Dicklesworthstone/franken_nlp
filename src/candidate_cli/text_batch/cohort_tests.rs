use super::*;
fn command(extra: &[&str]) -> TextBatchCommand {
    let mut args = vec!["text-batch", "--model", "local.fnlpq", "--memory-mib", "8192"];
    args.extend_from_slice(extra);
    TextBatchCommand::from_arg_matches(&definition().try_get_matches_from(args).unwrap()).unwrap()
}
#[test]
fn cohort_execution_is_explicit_and_width_is_bounded_before_io() {
    assert!(command(&[]).cohort_rows.is_none());
    for width in ["1", "4", "64"] {
        let selected = command(&["--cohort-rows", width, "--preparation-mib", "512"]);
        assert_eq!(selected.cohort_rows, Some(width.parse().unwrap()));
        assert!(selected.validate().is_ok());
    }
    for width in ["0", "65", "1000000"] {
        assert!(command(&["--cohort-rows", width]).validate().is_err());
    }
}
#[test]
fn prompt_morsels_and_document_cohorts_compose_with_independent_widths() {
    assert!(command(&["--prefill-rows", "4"]).validate().is_ok());
    assert!(command(&[]).common.policy.prefill().unwrap().is_none());
    for (documents, tokens) in [("1", "64"), ("4", "16"), ("4", "1"), ("64", "3")] {
        let selected = command(&["--cohort-rows", documents, "--prefill-rows", tokens, "--preparation-mib", "512"]);
        assert!(selected.validate().is_ok());
        assert_eq!(selected.cohort_rows, Some(documents.parse().unwrap()));
        assert_eq!(selected.common.policy.prefill().unwrap().unwrap().max_batch_rows, tokens.parse::<usize>().unwrap());
    }
}
#[test]
fn grouping_keeps_shared_generation_policy_and_seed_unchanged() {
    let seed = "ab".repeat(32);
    let common = ["--seed", seed.as_str(), "--stop", "END", "--min-new-tokens", "2", "--logprobs"];
    let serial = command(&common);
    let mut args = common.to_vec(); args.extend_from_slice(&["--cohort-rows", "4", "--prefill-rows", "16"]);
    let grouped = command(&args);
    assert!(serial.validate().is_ok()); assert!(grouped.validate().is_ok());
    assert!(serial.common.options(166_101).unwrap() == grouped.common.options(166_101).unwrap());
}
#[test]
fn preparation_reserve_prices_all_retained_cohort_plans() {
    assert!(command(&[]).validate().is_ok());
    assert!(command(&["--cohort-rows", "64"]).validate().is_err());
    assert!(command(&["--cohort-rows", "64", "--preparation-mib", "512"]).validate().is_ok());
}
#[test]
fn other_commands_cannot_accept_and_ignore_a_document_cohort_switch() {
    for task in ["generate", "chat", "ner", "classify"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", task,
            "--model", "local.fnlpq", "--memory-mib", "8192", "--cohort-rows", "4"]).is_err());
    }
    assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "stream", "generate",
        "--model", "local.fnlpq", "--memory-mib", "8192", "--cohort-rows", "4"]).is_err());
}
#[test]
fn invalid_packed_width_is_not_masked_by_a_valid_document_cohort() {
    for tokens in ["0", "65", "1000000"] {
        assert!(command(&["--cohort-rows", "4", "--prefill-rows", tokens]).validate().is_err());
    }
}
#[test]
fn invalid_combined_strategy_refuses_before_reading_or_writing() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid strategy read input") }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("invalid strategy wrote output") }
        fn flush(&mut self) -> io::Result<()> { panic!("invalid strategy flushed output") }
    }
    for tokens in ["0", "65"] {
        assert!(command(&["--cohort-rows", "4", "--prefill-rows", tokens]).execute(&mut NoIo, &mut NoIo).is_err());
    }
}
