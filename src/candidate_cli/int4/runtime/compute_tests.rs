use super::*;

#[test]
fn changed_worker_admission_refuses_before_runtime_and_private_io() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted team read input"); }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unadmitted team published output"); }
        fn flush(&mut self) -> io::Result<()> { panic!("unadmitted team flushed output"); }
    }
    let ordinary = super::super::tests::args("generate", &[]);
    let limits = ordinary.validate().unwrap();
    let parallel = super::super::tests::args("generate", &["--compute-threads", "2"]);
    assert_eq!(execute_task(Int4Task::Text(Task::Generate), parallel, limits, &mut NoIo, &mut NoIo),
        Err(CandidateError::Arguments));
    let parallel = super::super::tests::args("generate", &["--compute-threads", "4"]);
    let mut limits = parallel.validate_text_compute().unwrap();
    limits.compute.workspace_bytes -= 1;
    assert_eq!(execute_task(Int4Task::Text(Task::Generate), parallel, limits, &mut NoIo, &mut NoIo),
        Err(CandidateError::Arguments));
}

#[test]
fn physical_team_metadata_is_separate_and_serial_wire_bytes_stay_identical() {
    let result = serde_json::json!({"output":{"content":"x"},"candidate_format":"fixture"});
    let serial = compute::Response { response: &result, compute: None };
    assert_eq!(serde_json::to_vec(&serial).unwrap(), serde_json::to_vec(&result).unwrap());
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1",
        compute_threads: 4, completed_projections: 31, completed_rows: 1000, completed_worker_rows: 750 };
    let parallel = serde_json::to_value(compute::Response { response: &result, compute: Some(&receipt) }).unwrap();
    assert_eq!(parallel["output"], result["output"]);
    assert_eq!(parallel["compute"]["compute_threads"], 4);
    assert_eq!(parallel["compute"]["completed_worker_rows"], 750);
    for absent in ["speedup", "parity", "qualified", "prompt_digest"] { assert!(parallel.get(absent).is_none()); }
}

#[test]
fn native_and_task_deadlines_keep_their_timeout_category_through_cli_dispatch() {
    use crate::{native_engine::{constrained_int4::Int4JsonError,
        generation::grouped_int4::Int4GenerationError, strict_int4::StrictInt4Error},
        tasks::{chat::grouped_int4::Int4ChatError,
            classify::grouped_int4::Int4ClassificationError, extract::grouped_int4::Int4ExtractError}};
    fn extraction(cause: DecodeCancellationKind) -> Int4ExtractError {
        Int4JsonError::from(StrictInt4Error::Cancelled(cause)).into()
    }
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        for error in [
            Int4HostError::Native(StrictInt4Error::Cancelled(cause)),
            Int4HostError::Chat(Int4ChatError::from(Int4GenerationError::from(StrictInt4Error::Cancelled(cause)))),
            Int4HostError::Classification(Int4ClassificationError::Native(StrictInt4Error::Cancelled(cause))),
            Int4HostError::Extraction(extraction(cause)),
            Int4HostError::SourceTask(extraction(cause).into()),
        ] {
            assert_eq!(execution_error(HostedError::Int4(error)), CandidateError::Timeout);
        }
    }
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Input))),
        CandidateError::Execution);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
}

#[test]
#[ignore = "run alone in a fresh process with controller-admitted INT4 snapshot, independent pin and real multi-thread inference"]
fn snapshot_fixed_teams_match_full_serial_results_and_drain_charges() {
    let snapshot = std::env::var("FNLP_INT4_STREAM_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_STREAM_MODEL_SHA256").expect("independent content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().unwrap();
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory.checked_mul(MIB).unwrap(),
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let make = |operation: &str, threads: &str, seeded: bool, bpe: bool| {
        let mut flags = vec!["--compute-threads", threads, "--min-new-tokens", "1"];
        if bpe { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
        let mut args = super::super::tests::args(operation, &flags);
        args.source_dir = None; args.source_manifest = None;
        args.snapshot = Some(PathBuf::from(&snapshot)); args.expected_model_sha256 = Some(pin.clone());
        args.memory_mib = memory; args.context_tokens = 256; args.max_new_tokens = 4;
        args.max_output_bytes = 256; args.logprobs = true;
        if seeded {
            args.seed = Some("39".repeat(32)); args.temperature_milli = Some(700);
            args.top_k = Some(20); args.top_p_ppm = Some(950_000);
        }
        args
    };
    let base = make("generate", "1", false, false);
    let limits = base.validate().unwrap();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes).unwrap().commit().unwrap();
    let model = load_model(&engine, &base, limits, Duration::from_secs(3600), CancellationToken::default()).unwrap();
    let resident = engine.resources().memory_snapshot();
    for (operation, task, seeded, bpe) in [
        ("generate", Task::Generate, false, false), ("generate", Task::Generate, true, true),
        ("chat", Task::Chat, true, false), ("chat", Task::Chat, false, true),
    ] {
        let text = if task == Task::Chat {
            r#"[{"role":"user","content":"Give a short greeting."}]"#
        } else { "Give a short greeting." };
        let mut serial = None;
        for width in ["1", "2", "4"] {
            let args = make(operation, width, seeded, bpe);
            let limits = args.validate_text_compute().unwrap();
            let request = Input::Text(parse_input(task, text.to_owned(), args.max_input_bytes).unwrap());
            let prepared = prepare(request, &model, &args, limits, &mut DeadlineControl {
                started: Instant::now(), deadline: Duration::from_secs(3600),
            }).unwrap();
            let result = prepared.execute(&engine, &model, NativeLimits {
                context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: RunLimits { max_elapsed: Duration::from_secs(3600),
                    max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES },
            }, limits, CancellationToken::default()).unwrap();
            let value = serde_json::to_value(&result).unwrap();
            if width == "1" { assert!(result.compute().is_none()); serial = Some(value); }
            else {
                assert_eq!(&value, serial.as_ref().unwrap());
                let receipt = result.compute().unwrap();
                assert_eq!(receipt.compute_threads, width.parse::<usize>().unwrap());
                assert!(receipt.completed_projections > 0);
                assert!(receipt.completed_worker_rows > 0);
                assert!(receipt.completed_worker_rows < receipt.completed_rows);
            }
            drop(result);
            assert_eq!(engine.resources().memory_snapshot(), resident);
            let closures = engine.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0);
            assert_eq!(closures.wrapper_cancelled_closures, 0);
        }
    }
    drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}
