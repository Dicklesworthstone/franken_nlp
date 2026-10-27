//! Parser, full-memory and transport refusals without any model or private IO.
use super::*;

fn parse(operation: &str, extra: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "stream", operation,
        "--source-dir", "private-source", "--source-manifest", "private-manifest", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)? else {
        panic!("expected INT4 stream command");
    };
    Ok(command)
}

pub(in crate::candidate_cli::int4) fn args(operation: &str, extra: &[&str]) -> (StreamArgs, Int4Args) {
    let command = parse(operation, extra).unwrap();
    let Int4Task::Stream(_, options) = command.task else { panic!("expected INT4 stream task"); };
    (options, command.args)
}

struct NoIo;
impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input accessed"); } }
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("stdout accessed"); }
    fn flush(&mut self) -> io::Result<()> { panic!("stdout flushed"); }
}

#[test]
fn text_operations_keep_the_full_buffered_generation_policy() {
    let seed = "19".repeat(32);
    let flags = ["--seed", seed.as_str(), "--temperature-milli", "700", "--top-k", "20",
        "--top-p-ppm", "950000", "--logprobs", "--stop", "END", "--min-new-tokens", "3",
        "--repetition-penalty-milli", "1100", "--presence-penalty-milli", "-100",
        "--frequency-penalty-milli", "200", "--ban-token", "42", "--logit-bias", "41=20"];
    for operation in ["generate", "chat"] {
        let command = parse(operation, &flags).unwrap();
        let Int4Task::Stream(task, options) = command.task else { panic!("stream route"); };
        assert_eq!(task, if operation == "chat" { Task::Chat } else { Task::Generate });
        let (limits, bounds) = options.validate(&command.args).unwrap();
        let buffered = super::super::tests::args(operation, &flags);
        assert_eq!(canonjson::canonical_bytes(&command.args.options(166101).unwrap()).unwrap(),
            canonjson::canonical_bytes(&buffered.options(166101).unwrap()).unwrap());
        assert_eq!(limits.native, buffered.validate().unwrap().native);
        assert_eq!(bounds.terminal_bytes, limits.result_bytes + FRAME_BYTES);
        assert!(bounds.capture_logprobs);
    }
}

#[test]
fn source_snapshot_and_backend_contracts_do_not_change_for_streaming() {
    let pin = "23".repeat(32);
    for operation in ["generate", "chat"] {
        let matches = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", "stream", operation, "--snapshot", "private.snapshot",
            "--expected-model-sha256", &pin, "--memory-mib", "8192", "--prefill-rows", "4",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("INT4 route"); };
        let Int4Task::Stream(_, options) = command.task else { panic!("stream route"); };
        let (limits, _) = options.validate(&command.args).unwrap();
        assert_eq!(limits.snapshot_pin.unwrap().model_content_sha256, [0x23; 32]);
        assert_eq!(limits.prefill.unwrap().max_batch_rows, 4);
        assert!(command.args.source_dir.is_none());
    }
    for name in ["extract", "resolve", "sentiment", "text-job", "ner"] { assert!(parse(name, &[]).is_err()); }
    for flag in ["--think", "--tools", "--store-results"] { assert!(parse("generate", &[flag]).is_err()); }
    assert!(parse("generate", &["--top-k", "20"]).is_err());
    assert!(parse("generate", &["--snapshot", "another.snapshot", "--expected-model-sha256", &pin]).is_err());
    assert!(parse("generate", &["--model", "int8.fnlpq"]).is_err());
    let (options, a) = args("generate", &["--seed", "invalid"]);
    assert!(options.validate(&a).is_err());
}

#[test]
fn every_token_and_the_completed_result_have_nonrenewable_transport_space() {
    let (mut options, a) = args("generate", &[]);
    let (_, bounds) = options.validate(&a).unwrap();
    let floor = a.max_new_tokens as u64 * FRAME_BYTES as u64 + a.max_output_bytes as u64 * 4
        + FRAME_BYTES as u64 + bounds.terminal_bytes as u64;
    options.max_stream_bytes = floor;
    options.validate(&a).unwrap();
    options.max_stream_bytes -= 1;
    assert_eq!(options.validate(&a).err().unwrap().code, "stream_transport_too_small");
    for bytes in [0, u64::MAX] {
        options.max_stream_bytes = bytes;
        assert_eq!(options.validate(&a).err().unwrap().code, "stream_transport_options");
    }
}

#[test]
fn stream_memory_includes_native_and_packed_prefill_plus_both_retained_owners() {
    for flags in [&[][..], &["--prefill-rows", "4"][..]] {
        let (options, a) = args("generate", flags);
        let common = a.validate().unwrap();
        let (stream, bounds) = options.validate(&a).unwrap();
        assert_eq!(stream.minimum_memory_bytes, common.minimum_memory_bytes
            + common.preparation_bytes + bounds.sink_memory_bytes);
        assert_eq!(stream.native, common.native);
        let mut tight = a;
        tight.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
        tight.validate().unwrap();
        let error = options.validate(&tight).err().unwrap();
        assert_eq!(error.exit, ErrorCode::AdmissionOrResourceLimit);
        assert_eq!(error.code, "stream_memory_floor");
    }
}

#[test]
fn borrowed_dispatch_and_invalid_transport_never_read_or_publish() {
    for operation in ["generate", "chat"] {
        assert_eq!(parse(operation, &[]).unwrap().execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        let mut diagnostics = Vec::new();
        let status = parse(operation, &["--max-stream-bytes", "0"]).unwrap()
            .run_owned(NoIo, NoIo, &mut diagnostics);
        assert_eq!(status, ErrorCode::Usage.as_process_exit());
        let value: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(value["event"], "run_error");
        assert_eq!(value["scope"], "real-source-current-candidate-int4");
        assert_eq!(value["status"], "incomplete");
        assert_eq!(value["code"], "stream_transport_options");
        assert!(!String::from_utf8(diagnostics).unwrap().contains("private"));
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_without_private_input_model_or_output_access() {
    for (operation, task) in [("generate", Task::Generate), ("chat", Task::Chat)] {
        let (options, a) = args(operation, &[]);
        assert_eq!(options.execute(task, a, &mut NoIo, NoIo).unwrap_err().code, "stream_profile_unavailable");
    }
}
