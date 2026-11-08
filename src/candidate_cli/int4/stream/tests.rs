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
fn stream_teams_preserve_text_policy_and_admit_worker_scratch_before_io() {
    for operation in ["generate", "chat"] {
        let (default, serial) = args(operation, &[]);
        let (explicit, one) = args(operation, &["--compute-threads", "1"]);
        let (serial_limits, serial_bounds) = default.validate(&serial).unwrap();
        let (one_limits, one_bounds) = explicit.validate(&one).unwrap();
        assert_eq!(one_limits.compute.threads, 1);
        assert_eq!(one_limits.compute.workspace_bytes, 0);
        assert_eq!(serial_limits.minimum_memory_bytes, one_limits.minimum_memory_bytes);
        assert_eq!(serial_bounds.stream_bytes, one_bounds.stream_bytes);
        for width in ["2", "4", "64"] {
            let flags = ["--compute-threads", width, "--input-encoding", "safe-bpe-v1"];
            let (options, a) = args(operation, &flags);
            let (stream, bounds) = options.validate(&a).unwrap();
            let buffered = super::super::tests::args(operation, &flags);
            let common = buffered.validate_text_compute().unwrap();
            assert_eq!(stream.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(stream.compute, common.compute);
            assert!(stream.compute.workspace_bytes > 0);
            assert_eq!(stream.compute.workspace_bytes,
                crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(
                    stream.compute.threads, a.group_width).unwrap());
            assert_eq!(stream.minimum_memory_bytes, common.minimum_memory_bytes
                + common.preparation_bytes + bounds.sink_memory_bytes);
            assert_eq!(stream.native, common.native);
            assert_eq!(stream.max_prompt_tokens, common.max_prompt_tokens);
            assert_eq!(stream.result_bytes, common.result_bytes);
            assert_eq!(stream.preparation_bytes, common.preparation_bytes);
            assert!(a.options(166101).unwrap() == buffered.options(166101).unwrap());
            assert_eq!(a.input_encoding.policy(a.max_input_bytes, stream.max_prompt_tokens).unwrap(),
                buffered.input_encoding.policy(buffered.max_input_bytes, common.max_prompt_tokens).unwrap());
        }
    }
}

#[test]
fn invalid_stream_team_or_combined_schedule_refuses_before_private_io() {
    for operation in ["generate", "chat"] {
        for extra in [
            &["--compute-threads", "0"][..],
            &["--compute-threads", "65"][..],
            &["--compute-threads", "2", "--prefill-rows", "1"][..],
            &["--compute-threads", "4", "--prefill-rows", "64"][..],
        ] {
            let mut diagnostics = Vec::new();
            let status = parse(operation, extra).unwrap().run_owned(NoIo, NoIo, &mut diagnostics);
            assert_eq!(status, ErrorCode::Usage.as_process_exit());
            let error: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
            assert_eq!(error["code"], "stream_options");
            assert_eq!(error["status"], "incomplete");
            assert!(!String::from_utf8(diagnostics).unwrap().contains("private"));
        }
        assert!(parse(operation, &["--compute-threads", "2", "--cohort-heads", "2"]).is_err());
        let (options, a) = args(operation, &["--compute-threads", "1", "--prefill-rows", "4"]);
        let (limits, _) = options.validate(&a).unwrap();
        assert_eq!(limits.compute.threads, 1);
        assert_eq!(limits.prefill.unwrap().max_batch_rows, 4);
    }
}

#[test]
fn safe_bpe_keeps_buffered_identity_limits_and_prices_the_complete_workspace() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    for operation in ["generate", "chat"] {
        let flags = ["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "12345",
            "--max-tokenizer-pair-bytes", "456789", "--prefill-rows", "4"];
        let (options, mut a) = args(operation, &flags);
        let (limits, _) = options.validate(&a).unwrap();
        let policy = a.input_encoding.policy(a.max_input_bytes, limits.max_prompt_tokens).unwrap();
        let buffered = super::super::tests::args(operation, &flags);
        assert_eq!(policy, buffered.input_encoding.policy(buffered.max_input_bytes,
            buffered.validate().unwrap().max_prompt_tokens).unwrap());
        let UntrustedEncoding::SafeBpeV1(encoding) = policy else { panic!("explicit BPE required"); };
        assert_eq!(encoding.max_input_bytes, a.max_input_bytes);
        assert_eq!(encoding.max_output_tokens, limits.max_prompt_tokens);
        assert_eq!(encoding.max_pair_checks, 12345);
        assert_eq!(encoding.max_pair_bytes, 456789);
        let byte_floor = 128 * MIB + a.max_input_bytes as u64 * 32
            + (limits.result_bytes as u64 + 4096) * 2;
        a.preparation_mib = byte_floor.div_ceil(MIB);
        let (_, mut byte_default) = args(operation, &["--prefill-rows", "4"]);
        byte_default.preparation_mib = a.preparation_mib;
        options.validate(&byte_default).unwrap();
        assert_eq!(options.validate(&a).err().unwrap().exit, ErrorCode::Usage);
        a.preparation_mib = (byte_floor + encoding.requirements().unwrap().workspace_bytes).div_ceil(MIB);
        options.validate(&a).unwrap();
    }
    for operation in ["generate", "chat"] {
        assert!(parse(operation, &["--max-tokenizer-pair-checks", "1"]).is_err());
        assert!(parse(operation, &["--max-tokenizer-pair-bytes", "1"]).is_err());
        assert!(parse(operation, &["--input-encoding", "canonical-bpe"]).is_err());
    }
}

#[test]
fn invalid_bpe_work_or_workspace_refuses_before_input_model_and_output() {
    for operation in ["generate", "chat"] {
        for extra in [
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "0"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "0"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-checks", "67108865"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-tokenizer-pair-bytes", "536870913"][..],
            &["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576"][..],
        ] {
            let mut diagnostics = Vec::new();
            let status = parse(operation, extra).unwrap().run_owned(NoIo, NoIo, &mut diagnostics);
            assert_eq!(status, ErrorCode::Usage.as_process_exit());
            let error: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
            assert_eq!(error["code"], "stream_options");
            assert!(!String::from_utf8(diagnostics).unwrap().contains("private"));
        }
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
    for flags in [&[][..], &["--prefill-rows", "4"][..], &["--compute-threads", "4"][..]] {
        let (options, a) = args("generate", flags);
        let common = a.validate_text_compute().unwrap();
        let (stream, bounds) = options.validate(&a).unwrap();
        assert_eq!(stream.minimum_memory_bytes, common.minimum_memory_bytes
            + common.preparation_bytes + bounds.sink_memory_bytes);
        assert_eq!(stream.native, common.native);
        let mut tight = a;
        tight.memory_mib = common.minimum_memory_bytes.div_ceil(MIB);
        tight.validate_text_compute().unwrap();
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
        for flags in [&[][..], &["--compute-threads", "4"][..]] {
            let (options, a) = args(operation, flags);
            assert_eq!(options.execute(task, a, &mut NoIo, NoIo).unwrap_err().code, "stream_profile_unavailable");
        }
    }
}
