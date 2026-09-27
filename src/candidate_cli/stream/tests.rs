//! Complete CLI fixtures. No model, runtime, input or network is required.
use super::*;

fn parse(operation: &str, extra: &[&str]) -> Result<StreamCommand, clap::Error> {
    let mut argv = vec!["candidate", "stream", operation, "--model", "explicit.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::Stream(command) = CandidateCommand::from_matches(&matches)? else { panic!("stream route") };
    Ok(command)
}
pub(in crate::candidate_cli) fn args(operation: &str, extra: &[&str]) -> StreamArgs {
    parse(operation, extra).unwrap().into_parts().1
}
#[test]
fn both_stream_routes_keep_the_original_candidate_generation_options() {
    for (operation, expected) in [("generate", Task::Generate), ("chat", Task::Chat)] {
        let (task, args) = parse(operation, &[]).unwrap().into_parts();
        assert_eq!(task, expected); let (limits, bounds) = args.validate().unwrap();
        assert_eq!(bounds.tokens, args.common.max_new_tokens);
        assert_eq!(bounds.content_bytes, args.common.max_output_bytes);
        assert_eq!(bounds.terminal_bytes, limits.result_bytes + FRAME_BYTES);
        assert!(args.common.seed.is_none()); assert!(!bounds.capture_logprobs);
    }
    let seed = "07".repeat(32);
    let a = args("chat", &["--seed", &seed, "--temperature-milli", "700", "--top-k", "32", "--logprobs"]);
    let (_, bounds) = a.validate().unwrap(); assert!(bounds.capture_logprobs);
    let options = a.common.options(166101).unwrap(); assert!(options.capture_logprobs);
    assert!(matches!(options.sampling, GenerationSampling::Seeded { temperature_milli: 700, top_k: Some(32), .. }));
}
#[test]
fn unrelated_tasks_privileged_modes_and_implicit_sampling_are_not_stream_options() {
    assert!(parse("generate", &[]).is_ok());
    for operation in ["classify", "sentiment", "extract", "ner", "job"] { assert!(parse(operation, &[]).is_err()); }
    for flag in ["--think", "--tools", "--store-results", "--discard-uncommitted-tail"] {
        assert!(parse("generate", &[flag]).is_err(), "{flag}");
    }
    for (flag, value) in [("--schema", "s.json"), ("--top-k", "8"), ("--temperature-milli", "700")] {
        assert!(parse("generate", &[flag, value]).is_err(), "{flag}");
    }
    assert!(args("generate", &["--seed", "bad"]).validate().is_err());
}
#[test]
fn transport_limit_reserves_every_token_frame_and_the_full_terminal_result() {
    let mut a = args("generate", &[]); let (limits, bounds) = a.validate().unwrap();
    let floor = a.common.max_new_tokens as u64 * FRAME_BYTES as u64
        + a.common.max_output_bytes as u64 * 4 + FRAME_BYTES as u64 + bounds.terminal_bytes as u64;
    a.max_stream_bytes = floor; a.validate().unwrap();
    a.max_stream_bytes -= 1;
    assert_eq!(a.validate().err().unwrap().code, "stream_transport_too_small");
    assert_eq!(limits.result_bytes + FRAME_BYTES, bounds.terminal_bytes);
    a.max_stream_bytes = 0; assert!(a.validate().is_err());
    a.max_stream_bytes = u64::MAX; assert!(a.validate().is_err());
}
#[test]
fn transport_capacity_never_changes_native_sampling_or_task_identity_inputs() {
    let a = args("generate", &[]);
    let b = args("generate", &["--max-stream-bytes", "33554432"]);
    let (la, _) = a.validate().unwrap(); let (lb, _) = b.validate().unwrap();
    assert_eq!(la.max_prompt_tokens, lb.max_prompt_tokens); assert_eq!(la.kv_bytes, lb.kv_bytes);
    assert_eq!(canonjson::canonical_bytes(&a.common.options(166101).unwrap()).unwrap(),
        canonjson::canonical_bytes(&b.common.options(166101).unwrap()).unwrap());
}
#[test]
fn impossible_stream_memory_floor_is_refused_before_input_or_model_io() {
    let mut a = args("generate", &[]); a.common.memory_mib = 6144;
    // Common model/preparation fields individually fit, but the complete
    // two-stage retained preparation plus sink and weights does not.
    a.common.validate().unwrap();
    assert_eq!(a.validate().err().unwrap().code, "stream_memory_floor");
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_profile_never_opens_input_model_or_output() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected input") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected output") }
        fn flush(&mut self) -> io::Result<()> { panic!("unexpected flush") }
    }
    for (operation, task) in [("generate", Task::Generate), ("chat", Task::Chat)] {
        let error = execute(task, args(operation, &[]), &mut NoIo, NoIo).unwrap_err();
        assert_eq!(error.code, "stream_profile_unavailable");
    }
}
