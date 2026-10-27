//! Typed stop/error policy and pre-IO invariants; no fabricated native success.
use super::*;
use crate::{candidate_cli::int4::stream::tests::args, hosted::RunStop};

#[test]
fn all_eleven_causes_keep_budget_cancellation_and_failure_distinct() {
    use DecodeCancellationKind::*;
    for cause in [Timeout, Deadline, PollQuota, CostBudget, User, ParentCancelled, Shutdown,
        ResourceUnavailable, FailFast, LinkedExit, RaceLost] {
        let expected = match cause {
            Timeout | Deadline | PollQuota | CostBudget => ErrorCode::BudgetOrTimeout,
            User | ParentCancelled | Shutdown => ErrorCode::Cancelled,
            ResourceUnavailable => ErrorCode::AdmissionOrResourceLimit,
            FailFast | LinkedExit | RaceLost => ErrorCode::Generic,
        };
        for error in [
            HostedError::Stopped { stop: RunStop { kind: cause, runtime_reason: None,
                runtime_checkpoint_error: None }, task_error: None },
            HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Chat(Int4ChatError::Native(
                Int4GenerationError::Generation(GenerationError::Cancelled(cause))))),
        ] {
            let failure = host_failure(error);
            assert_eq!(failure.exit, expected);
            assert_eq!(failure.cancellation, Some(cause));
        }
    }
}

#[test]
fn finalization_writer_and_supervision_failures_never_become_completed_output() {
    let failure = host_failure(HostedError::Int4(Int4HostError::Chat(
        Int4ChatError::Chat(ChatError::NoResult("private invalid output")))));
    assert_eq!(failure.exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(failure.code, "stream_final_task_no_result");
    let failure = host_failure(HostedError::Int4(Int4HostError::Chat(
        Int4ChatError::Native(Int4GenerationError::Generation(GenerationError::Stream)))));
    assert_eq!(failure.exit, ErrorCode::Generic);
    assert_eq!(failure.code, "stream_output_or_protocol");
    for kind in [DecodeCancellationKind::FailFast, DecodeCancellationKind::LinkedExit] {
        let failure = host_failure(HostedError::Stopped {
            stop: RunStop { kind, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Panicked)),
        });
        assert_eq!(failure.exit, ErrorCode::Generic);
        assert_eq!(failure.cancellation, Some(kind));
    }
    assert_eq!(host_failure(HostedError::Panicked).exit, ErrorCode::Generic);
    assert_eq!(host_failure(HostedError::Reentrant).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(host_failure(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Integrity))).exit,
        ErrorCode::ArtifactIntegrityOrFormatOrVersion);
}

#[test]
fn changed_admitted_options_refuse_before_runtime_or_any_private_input() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read after refused options"); } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write after refused options"); }
        fn flush(&mut self) -> io::Result<()> { panic!("flush after refused options"); }
    }
    for axis in 0..4 {
        let (options, mut a) = args("generate", &[]);
        let (limits, mut bounds) = options.validate(&a).unwrap();
        match axis {
            0 => a.kernel = Int8KernelKind::Avx2WidenedI16,
            1 => bounds.tokens += 1,
            2 => bounds.content_bytes += 1,
            _ => bounds.capture_logprobs = !bounds.capture_logprobs,
        }
        let failure = execute(Task::Generate, a, limits, bounds, &mut NoIo, NoIo).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage);
        assert_eq!(failure.code, "stream_admitted_options_mismatch");
    }
}

#[test]
#[ignore = "requires controller-admitted existing INT4 snapshot, independent content pin and real native execution"]
fn snapshot_stream_completion_and_bytes_equal_buffered_generate_and_chat() {
    use std::sync::Mutex;
    use crate::native_engine::decode::{DecodeEventSink, DecodeTokenEvent};

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum Fault { Write, Flush, BeforeRun, Reserve, Permit }
    #[derive(Default)]
    struct Wire { bytes: Vec<u8>, writes: usize, flushes: usize }
    #[derive(Clone)]
    struct Output { wire: Arc<Mutex<Wire>>, fault: Option<Fault> }
    impl Write for Output {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let mut wire = self.wire.lock().unwrap();
            wire.writes += 1;
            if self.fault == Some(Fault::Write) && wire.writes > 1 {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "private writer failure"));
            }
            let count = if self.fault == Some(Fault::Write) { bytes.len().min(7) } else { bytes.len() };
            wire.bytes.extend_from_slice(&bytes[..count]);
            Ok(count)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.wire.lock().unwrap().flushes += 1;
            if self.fault == Some(Fault::Flush) { Err(io::Error::other("private flush failure")) }
            else { Ok(()) }
        }
    }
    struct Controlled<S> {
        inner: S, cancellation: CancellationToken, fault: Fault,
        callbacks: Arc<Mutex<[usize; 2]>>,
    }
    impl<S: DecodeEventSink> DecodeEventSink for Controlled<S> {
        type Permit = S::Permit;
        type Error = S::Error;
        fn reserve(&mut self, event: &DecodeTokenEvent) -> Result<Self::Permit, Self::Error> {
            self.callbacks.lock().unwrap()[0] += 1;
            let permit = self.inner.reserve(event)?;
            if self.fault == Fault::Reserve { self.cancellation.cancel(DecodeCancellationKind::User); }
            Ok(permit)
        }
        fn permit(&mut self, permit: Self::Permit, event: DecodeTokenEvent) -> Result<(), Self::Error> {
            self.callbacks.lock().unwrap()[1] += 1;
            self.inner.permit(permit, event)?;
            if self.fault == Fault::Permit { self.cancellation.cancel(DecodeCancellationKind::User); }
            Ok(())
        }
    }
    fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
        bytes.split(|&b| b == b'\n').filter(|b| !b.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap()).collect()
    }

    let snapshot = std::env::var("FNLP_INT4_STREAM_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_STREAM_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().unwrap();
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: memory.checked_mul(MIB).unwrap(),
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let assert_drained = |expected: &crate::MemorySnapshot| {
        assert_eq!(engine.resources().memory_snapshot(), *expected);
        let closures = engine.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0);
        assert_eq!(closures.wrapper_cancelled_closures, 0);
    };
    let seed = "39".repeat(32);
    let make = |operation, flags: &[&str], seeded| {
        let mut flags = flags.to_vec();
        flags.extend_from_slice(&["--min-new-tokens", "1"]);
        let (options, mut a) = args(operation, &flags);
        a.source_dir = None; a.source_manifest = None;
        a.snapshot = Some(PathBuf::from(&snapshot)); a.expected_model_sha256 = Some(pin.clone());
        a.memory_mib = memory; a.context_tokens = 256; a.max_new_tokens = 4;
        a.max_output_bytes = 256; a.logprobs = true;
        if seeded {
            a.seed = Some(seed.clone()); a.temperature_milli = Some(700);
            a.top_k = Some(20); a.top_p_ppm = Some(950_000);
        }
        (options, a)
    };
    // Each task exercises greedy and addressed sampling, with both admitted
    // prompt schedules represented. Equality covers every completion field.
    for (operation, task, flags, seeded) in [
        ("generate", Task::Generate, &[][..], false),
        ("generate", Task::Generate, &["--prefill-rows", "4"][..], true),
        ("chat", Task::Chat, &[][..], true),
        ("chat", Task::Chat, &["--prefill-rows", "4"][..], false),
    ] {
        let text = if task == Task::Chat {
            r#"[{"role":"user","content":"Reply with a short English greeting."}]"#
        } else { "Reply with a short English greeting." };
        let (_, buffered_args) = make(operation, flags, seeded);
        let buffered_limits = buffered_args.validate().unwrap();
        let mut buffered = Vec::new();
        super::super::execute(task, buffered_args, buffered_limits, &mut text.as_bytes(), &mut buffered).unwrap();
        assert_drained(&idle);
        let buffered: serde_json::Value = serde_json::from_slice(&buffered).unwrap();
        let (options, stream_args) = make(operation, flags, seeded);
        let (limits, bounds) = options.validate(&stream_args).unwrap();
        let collected = Arc::new(Mutex::new(Wire::default()));
        execute(task, stream_args, limits, bounds, &mut text.as_bytes(),
            Output { wire: Arc::clone(&collected), fault: None }).unwrap();
        assert_drained(&idle);
        let wire = collected.lock().unwrap();
        let frames = frames(&wire.bytes);
        assert_eq!(frames.first().unwrap()["event"], "run_start");
        assert_eq!(frames.last().unwrap()["event"], "run_complete");
        assert_eq!(frames.last().unwrap()["data"], buffered["output"]);
        let expected_seed = if seeded { serde_json::Value::String(seed.clone()) } else { serde_json::Value::Null };
        assert_eq!(frames.last().unwrap()["data"]["result"]["effective_seed"], expected_seed);
        let bytes: Vec<u8> = frames.iter().filter(|f| f["event"] == "token")
            .flat_map(|f| f["data"]["decoded_bytes"].as_array().unwrap().iter()
                .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())).collect();
        assert_eq!(bytes, buffered["output"]["result"]["content"].as_str().unwrap().as_bytes());
        assert!(wire.bytes.len() as u64 <= bounds.stream_bytes);
    }

    // One resident model serves the failure cases. Observe the real process
    // ledger after each physical return, not merely the async wrapper result.
    let (options, load_args) = make("generate", &[], false);
    let (load_limits, _) = options.validate(&load_args).unwrap();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, load_limits.preparation_bytes)
        .unwrap().commit().unwrap();
    let model = load_model(&engine, &load_args, load_limits, Duration::from_secs(load_args.timeout_seconds),
        CancellationToken::default()).unwrap();
    let resident = engine.resources().memory_snapshot();
    for flags in [&[][..], &["--prefill-rows", "4"][..]] {
        for fault in [Fault::Write, Fault::Flush, Fault::BeforeRun, Fault::Reserve, Fault::Permit] {
            let (options, a) = make("generate", flags, true);
            let (limits, bounds) = options.validate(&a).unwrap();
            let Prepared::Text(prepared) = prepare(Input::Text(TextInput::Generate(
                "Reply with a short English greeting.".to_owned())), &model, &a, limits,
                &mut DeadlineControl { started: Instant::now(), deadline: Duration::from_secs(a.timeout_seconds) })
                .unwrap() else { panic!("text preparation required") };
            let wire = Arc::new(Mutex::new(Wire::default()));
            let callbacks = Arc::new(Mutex::new([0; 2]));
            let cancellation = CancellationToken::default();
            if fault == Fault::BeforeRun { cancellation.cancel(DecodeCancellationKind::User); }
            let sink = Controlled {
                inner: TokenSink::new_int4(Output { wire: Arc::clone(&wire), fault: Some(fault) },
                    Task::Generate, bounds, model.model_identity(), prepared.native_plan().options()).unwrap(),
                cancellation: cancellation.clone(), fault, callbacks: Arc::clone(&callbacks),
            };
            let stream_limits = ChatStreamLimits {
                native: NativeLimits { context_tokens: a.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                    run: RunLimits { max_elapsed: Duration::from_secs(a.timeout_seconds),
                        max_checkpoints: a.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
                max_sampler_bytes: SAMPLER_BYTES, preparation_reserve_bytes: limits.preparation_bytes,
                sink_reserve_bytes: bounds.sink_memory_bytes,
            };
            let result = match limits.prefill {
                Some(prefill) => engine.execute_int4_chat_stream_layer_major(&model, prepared, 1,
                    stream_limits, prefill, limits.kernel, sink, cancellation),
                None => engine.execute_int4_chat_stream(&model, prepared, 1,
                    stream_limits, limits.kernel, sink, cancellation),
            };
            let failure = host_failure(result.err().expect("failed delivery/cancellation must have no completion"));
            assert_drained(&resident);
            assert!(!failure.to_string().contains("private"));
            let wire = wire.lock().unwrap();
            assert!(!wire.bytes.windows(b"run_complete".len()).any(|w| w == b"run_complete"));
            match fault {
                Fault::Write | Fault::Flush => {
                    assert_eq!(failure.exit, ErrorCode::Generic);
                    assert_eq!(failure.code, "stream_output_or_protocol");
                    assert_eq!(*callbacks.lock().unwrap(), [1, 1]);
                    // One short prefix followed by one failure, or one start
                    // and token followed by one failed flush: never retry.
                    assert_eq!(wire.writes, 2);
                    assert_eq!(wire.flushes, usize::from(fault == Fault::Flush));
                    if fault == Fault::Write { assert_eq!(wire.bytes.len(), 7); }
                    else { assert_eq!(frames(&wire.bytes).len(), 2); }
                }
                Fault::BeforeRun | Fault::Reserve | Fault::Permit => {
                    assert_eq!(failure.exit, ErrorCode::Cancelled);
                    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::User));
                    let expected = match fault { Fault::BeforeRun => [0, 0], Fault::Reserve => [1, 0], _ => [1, 1] };
                    assert_eq!(*callbacks.lock().unwrap(), expected);
                    if fault == Fault::Permit {
                        let frames = frames(&wire.bytes);
                        assert_eq!(frames.len(), 2);
                        assert_eq!(frames[0]["event"], "run_start");
                        assert_eq!(frames[1]["event"], "token");
                        assert!(frames.iter().all(|f| f["provisional"] == true));
                    } else { assert!(wire.bytes.is_empty()); }
                }
            }
        }
    }
    drop(model);
    drop(preparation);
    drop(lease);
    assert_drained(&idle);
}
