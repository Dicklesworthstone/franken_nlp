use super::*;
use crate::candidate_cli::tests::args;

#[test]
fn buffered_and_streamed_text_expose_the_fixed_team_without_enabling_corpora() {
    for task in ["generate", "chat"] {
        for width in ["1", "2", "7", "64"] {
            let matches = super::super::definition().try_get_matches_from(["candidate", task,
                "--model", "m", "--memory-mib", "8192", "--compute-threads", width]).unwrap();
            let CandidateCommand::Text { args, .. } = CandidateCommand::from_matches(&matches).unwrap()
                else { panic!("wrong route") };
            let limits = args.validate().unwrap();
            assert_eq!(limits.compute.threads, width.parse::<usize>().unwrap());
            assert_eq!(limits.compute.workspace_bytes, if width == "1" { 0 }
                else { int8_team_workspace_bytes(limits.compute.threads).unwrap() });
        }
        let stream = crate::candidate_cli::stream::tests::args(task, &["--compute-threads", "2"]);
        assert_eq!(stream.validate().unwrap().0.compute.threads, 2);
    }
    assert_eq!(super::super::definition().try_get_matches_from(["candidate", "text-batch",
        "--model", "m", "--memory-mib", "8192", "--compute-threads", "2"])
        .unwrap_err().kind(), clap::error::ErrorKind::UnknownArgument);
}

#[test]
fn team_width_prefill_and_complete_memory_fail_before_any_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted input"); } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unadmitted output"); }
        fn flush(&mut self) -> io::Result<()> { panic!("unadmitted flush"); }
    }
    for extra in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--compute-threads", "2", "--prefill-rows", "0"],
        vec!["--compute-threads", "7", "--prefill-rows", "65"],
        vec!["--compute-threads", "2", "--max-weight-mib", "8192"]] {
        for task in [Task::Generate, Task::Chat] {
            let command = CandidateCommand::Text { task, args: args(&extra) };
            assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
    }
    // Existing explicit serial prefill remains supported.
    args(&["--compute-threads", "1", "--prefill-rows", "4"]).validate().unwrap();
}

#[test]
fn compact_input_retains_exact_policy_and_preparation_charge_with_a_team() {
    let serial = args(&["--input-encoding", "safe-bpe-v1"]);
    let threaded = args(&["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"]);
    let serial_limits = serial.validate().unwrap(); let threaded_limits = threaded.validate().unwrap();
    assert_eq!(serial.input_encoding_policy(serial_limits).unwrap(), threaded.input_encoding_policy(threaded_limits).unwrap());
    assert_eq!(serial_limits.preparation_bytes, threaded_limits.preparation_bytes);
    assert!(threaded_limits.compute.workspace_bytes > 0);
}

#[test]
fn source_peak_retains_both_preparation_owners_with_an_exact_memory_boundary() {
    use crate::candidate_cli::source::tests::command;
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let command = command(task, &["--compute-threads", "4"]);
        let (args, mut limits) = command.validate().unwrap();
        let native = Int8MemoryRequirement::for_context(args.context_tokens).unwrap();
        let one_owner = request_peak_bytes(limits, native, command.args.host.max_result_bytes as u64,
            args.max_new_tokens as u64, 0, 0).unwrap();
        let required = source_peak_bytes(&args, limits, command.args.host.max_result_bytes, None).unwrap();
        assert_eq!(required - one_owner, limits.preparation_bytes,
            "transferred plan, vocabulary and source retain their own whole reservation");
        for memory_bytes in [one_owner, required - 1] {
            limits.memory_bytes = memory_bytes;
            assert_eq!(check_source_memory(&args, limits, command.args.host.max_result_bytes, None), Err(CandidateError::Arguments));
        }
        limits.memory_bytes = required;
        check_source_memory(&args, limits, command.args.host.max_result_bytes, None).unwrap();
        let mut larger = limits; larger.preparation_bytes += 1;
        assert_eq!(source_peak_bytes(&args, larger, command.args.host.max_result_bytes, None).unwrap(), required + 2);
        larger.preparation_bytes = u64::MAX;
        assert_eq!(source_peak_bytes(&args, larger, command.args.host.max_result_bytes, None), Err(CandidateError::Arguments));
    }
}

#[cfg(feature = "asupersync-runtime")]
#[test]
fn omitted_and_explicit_serial_width_leave_wire_bytes_identical() {
    // Framing fixture only; do not fabricate a successful native result.
    let response = serde_json::json!({"schema_version": 1, "output": {"framing_fixture": true}});
    let bytes = serde_json::to_vec(&response).unwrap();
    for extra in [vec![], vec!["--compute-threads", "1"]] {
        assert_eq!(args(&extra).validate().unwrap().compute.threads, 1);
        let wrapped = Response { response: &response, compute: None };
        assert_eq!(serde_json::to_vec(&wrapped).unwrap(), bytes);
    }
    let compute = crate::hosted::Int8ComputeReceipt { schedule: "int8-output-row-team-v1",
        compute_threads: 3, completed_projections: 2, completed_rows: 13, completed_worker_rows: 8 };
    let wrapped = serde_json::to_value(Response { response: &response, compute: Some(&compute) }).unwrap();
    assert_eq!(wrapped["output"], response["output"]);
    assert_eq!(wrapped["compute"], serde_json::to_value(compute).unwrap());
}


#[test]
fn buffered_prompt_batches_share_the_team_without_changing_generation_semantics() {
    for task in ["generate", "chat"] {
        for width in [2_usize, 7, 64] { for rows in [1_usize, 4, 64] {
            let width_text = width.to_string(); let rows_text = rows.to_string();
            let matches = super::super::definition().try_get_matches_from(["candidate", task,
                "--model", "m", "--memory-mib", "16384", "--compute-threads", &width_text,
                "--prefill-rows", &rows_text]).unwrap();
            let CandidateCommand::Text { args: combined, .. } = CandidateCommand::from_matches(&matches).unwrap()
                else { panic!("wrong route") };
            let limits = combined.validate().unwrap();
            assert_eq!(limits.compute.threads, width);
            assert_eq!(limits.compute.workspace_bytes, int8_team_workspace_bytes_with_prefill(width, rows).unwrap());
            assert!(combined.options(166_101).unwrap() == args(&[]).options(166_101).unwrap());
            if rows == 1 {
                assert_eq!(limits.compute.workspace_bytes, int8_team_workspace_bytes(width).unwrap());
            }
        } }
    }
}

#[test]
fn combined_memory_peak_includes_coordinator_prefill_and_fixed_child_capacity() {
    let args = args(&["--compute-threads", "7", "--prefill-rows", "64"]);
    let mut limits = args.validate().unwrap();
    let native = Int8MemoryRequirement::for_context(args.context_tokens).unwrap();
    let base = request_peak_bytes(limits, native, limits.result_bytes as u64,
        args.max_new_tokens as u64, SAMPLER_BYTES, 0).unwrap();
    let extra = args.policy.prefill().unwrap().unwrap().validate().unwrap();
    assert!(extra > 0);
    assert!(limits.compute.workspace_bytes > int8_team_workspace_bytes(7).unwrap());
    limits.memory_bytes = base + extra;
    check_memory(&args, limits, native).unwrap();
    limits.memory_bytes -= 1;
    assert_eq!(check_memory(&args, limits, native), Err(CandidateError::Arguments));
}


#[test]
fn source_prompt_workspace_and_child_capacity_remain_separate_exact_charges() {
    use crate::candidate_cli::source::tests::command;
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        for width in ["1", "4", "64"] {
            let mut command = command(task, &["--compute-threads", width, "--prefill-rows", "64"]);
            command.args.host.memory_mib = 16384;
            let (args, mut limits) = command.validate().unwrap();
            let prefill = command.args.prefill.limits().unwrap().unwrap();
            let base = source_peak_bytes(&args, limits, command.args.host.max_result_bytes, None).unwrap();
            let required = source_peak_bytes(&args, limits, command.args.host.max_result_bytes, Some(prefill)).unwrap();
            assert_eq!(required - base, prefill.validate().unwrap());
            limits.memory_bytes = required;
            check_source_memory(&args, limits, command.args.host.max_result_bytes, Some(prefill)).unwrap();
            limits.memory_bytes -= 1;
            assert_eq!(check_source_memory(&args, limits, command.args.host.max_result_bytes, Some(prefill)),
                Err(CandidateError::Arguments));
            let native = Int8MemoryRequirement::for_context(args.context_tokens).unwrap();
            let one_owner = request_peak_bytes(limits, native, command.args.host.max_result_bytes as u64,
                args.max_new_tokens as u64, 0, 0).unwrap();
            assert_eq!(required - one_owner, limits.preparation_bytes + prefill.validate().unwrap());
            if width != "1" {
                assert!(limits.compute.workspace_bytes > int8_team_workspace_bytes(limits.compute.threads).unwrap());
            }
            let mut larger_cap = prefill; larger_cap.max_extra_scratch_bytes = u64::MAX;
            // The host reserves the derived payload, not an arbitrary larger cap.
            assert_eq!(source_peak_bytes(&args, limits, command.args.host.max_result_bytes,
                Some(larger_cap)).unwrap(), required);
        }
    }
}

#[test]
fn even_serial_compute_admission_validates_explicit_prompt_geometry() {
    for threads in [1_usize, 2, 64] {
        for (rows, bytes) in [(0, 0), (65, u64::MAX), (4, 0)] {
            assert_eq!(Args { threads }.admit_prefill(Some(Int8PrefillLimits {
                max_batch_rows: rows, max_extra_scratch_bytes: bytes })), Err(CandidateError::Arguments));
        }
    }
}
