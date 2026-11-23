use super::*;
use crate::candidate_cli::tests::args;

#[test]
fn only_buffered_generate_and_chat_expose_the_fixed_compute_team() {
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
        let matches = super::super::definition().try_get_matches_from(["candidate", "stream", task,
            "--model", "m", "--memory-mib", "8192", "--compute-threads", "2"]);
        assert_eq!(matches.unwrap_err().kind(), clap::error::ErrorKind::UnknownArgument);
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
        vec!["--compute-threads", "2", "--prefill-rows", "1"],
        vec!["--compute-threads", "7", "--prefill-rows", "64"],
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

