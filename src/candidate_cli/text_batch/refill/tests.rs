use super::*;
fn command(extra: &[&str]) -> TextBatchCommand {
    let mut args = vec!["text-batch", "--model", "local.fnlpq", "--memory-mib", "8192"];
    args.extend_from_slice(extra);
    TextBatchCommand::from_arg_matches(&definition().try_get_matches_from(args).unwrap()).unwrap()
}
#[test]
fn refill_is_explicit_and_requires_bounded_queued_and_active_widths() {
    assert!(command(&[]).refill_strategy(1).unwrap().is_none());
    assert!(command(&["--cohort-rows", "4"]).refill_strategy(4).unwrap().is_none());
    for (queued, active) in [("1", "1"), ("4", "1"), ("4", "4"), ("64", "2")] {
        let selected = command(&["--cohort-rows", queued, "--active-rows", active, "--preparation-mib", "512"]);
        assert!(selected.validate().is_ok());
        assert_eq!(selected.refill_strategy(queued.parse().unwrap()).unwrap().unwrap().0, active.parse::<usize>().unwrap());
    }
    for args in [vec!["--active-rows", "1"], vec!["--cohort-rows", "4", "--active-rows", "0"],
        vec!["--cohort-rows", "4", "--active-rows", "5"], vec!["--cohort-rows", "64", "--active-rows", "65"]] {
        assert!(command(&args).validate().is_err());
    }
}
#[test]
fn default_token_width_and_partial_tail_use_actual_live_membership() {
    let selected = command(&["--cohort-rows", "8", "--active-rows", "4"]);
    let (active, tokens) = selected.refill_strategy(8).unwrap().unwrap();
    assert_eq!(active, 4); assert_eq!(tokens.max_batch_rows, 4);
    let (active, tokens) = selected.refill_strategy(2).unwrap().unwrap();
    assert_eq!(active, 2); assert_eq!(tokens.max_batch_rows, 2);
    assert!(selected.refill_strategy(0).is_err()); assert!(selected.refill_strategy(9).is_err());
    let explicit = command(&["--cohort-rows", "8", "--active-rows", "4", "--prefill-rows", "16"]);
    let (active, tokens) = explicit.refill_strategy(2).unwrap().unwrap();
    assert_eq!(active, 2); assert_eq!(tokens.max_batch_rows, 16);
    assert_eq!(tokens.max_extra_scratch_bytes, Int8PrefillLimits::required_extra_scratch_bytes(16).unwrap());
}
#[test]
fn token_width_can_be_smaller_than_live_slots_without_overriding_policy() {
    let seed = "ab".repeat(32);
    let base = ["--seed", seed.as_str(), "--stop", "END", "--min-new-tokens", "2", "--logprobs"];
    let ordinary = command(&base);
    let mut args = base.to_vec(); args.extend_from_slice(&["--cohort-rows", "8", "--active-rows", "4", "--prefill-rows", "1"]);
    let selected = command(&args); assert!(selected.validate().is_ok());
    assert_eq!(selected.refill_strategy(8).unwrap().unwrap().1.max_batch_rows, 1);
    assert!(ordinary.common.options(166_101).unwrap() == selected.common.options(166_101).unwrap());
}
#[test]
fn preparation_prices_the_entire_queue_not_only_one_live_slot() {
    assert!(command(&["--cohort-rows", "64", "--active-rows", "1"]).validate().is_err());
    assert!(command(&["--cohort-rows", "64", "--active-rows", "1", "--preparation-mib", "512"]).validate().is_ok());
}
#[test]
fn other_commands_cannot_accept_and_ignore_active_rows() {
    for task in ["generate", "chat", "classify", "ner"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", task,
            "--model", "local.fnlpq", "--memory-mib", "8192", "--active-rows", "1"]).is_err());
    }
    assert!(crate::candidate_cli::definition().try_get_matches_from(["candidate", "stream", "generate",
        "--model", "local.fnlpq", "--memory-mib", "8192", "--active-rows", "1"]).is_err());
}
#[test]
fn malformed_refill_selection_is_refused_before_input_model_or_output_io() {
    struct Never;
    impl Read for Never { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("input touched") } }
    impl Write for Never {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output touched") }
        fn flush(&mut self) -> io::Result<()> { panic!("output flushed") }
    }
    assert!(command(&["--active-rows", "1"]).execute(&mut Never, &mut Never).is_err());
    assert!(command(&["--cohort-rows", "4", "--active-rows", "5"]).execute(&mut Never, &mut Never).is_err());
}
