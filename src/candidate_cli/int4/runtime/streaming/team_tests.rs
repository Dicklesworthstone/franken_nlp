//! Controller-only model fixture. Definitions are not parity or speed evidence.
use super::*;
use crate::{candidate_cli::int4::stream::tests::args,
    native_engine::decode::{DecodeEventSink, DecodeTokenEvent}};
use std::sync::Mutex;

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
        wire.bytes.extend_from_slice(&bytes[..count]); Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.wire.lock().unwrap().flushes += 1;
        if self.fault == Some(Fault::Flush) { Err(io::Error::other("private flush failure")) }
        else { Ok(()) }
    }
}
struct Controlled<S> {
    inner: S, cancellation: CancellationToken, fault: Fault, callbacks: Arc<Mutex<[usize; 2]>>,
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
fn plan(text: &str, task: Task, model: &ResidentInt4, args: &Int4Args, limits: Int4Limits)
    -> crate::tasks::chat::grouped_int4::PreparedInt4Chat {
    let input = Input::Text(parse_input(task, text.to_owned(), args.max_input_bytes).unwrap());
    let Prepared::Text(prepared) = prepare(input, model, args, limits, &mut DeadlineControl {
        started: Instant::now(), deadline: Duration::from_secs(3600),
    }).unwrap() else { panic!("stream fixture must prepare text"); };
    prepared
}
fn native(args: &Int4Args) -> NativeLimits {
    NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES } }
}

#[test]
#[ignore = "run alone with controller-admitted INT4 snapshot, independent pin and real serial/two/four-worker streamed inference"]
fn snapshot_teams_preserve_streamed_tokens_complete_results_and_physical_cleanup() {
    let snapshot = std::env::var("FNLP_INT4_STREAM_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_STREAM_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().unwrap();
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory.checked_mul(MIB).unwrap(),
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let assert_drained = |expected: &crate::MemorySnapshot| {
        assert_eq!(engine.resources().memory_snapshot(), *expected);
        let closures = engine.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0);
        assert_eq!(closures.wrapper_cancelled_closures, 0);
    };
    let make = |operation, seeded| {
        let (options, mut a) = args(operation, &["--compute-threads", "4",
            "--input-encoding", "safe-bpe-v1", "--min-new-tokens", "1"]);
        a.source_dir = None; a.source_manifest = None;
        a.snapshot = Some(PathBuf::from(&snapshot)); a.expected_model_sha256 = Some(pin.clone());
        a.memory_mib = memory; a.context_tokens = 256; a.max_new_tokens = 4;
        a.max_output_bytes = 256; a.logprobs = true;
        if seeded {
            a.seed = Some("39".repeat(32)); a.temperature_milli = Some(700);
            a.top_k = Some(20); a.top_p_ppm = Some(950_000);
        }
        (options, a)
    };
    // This reservation precedes model/planner/sink allocations and retains
    // private comparison values until the fixture discards them.
    let (options, load_args) = make("generate", false);
    let (load_limits, _) = options.validate(&load_args).unwrap();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, load_limits.preparation_bytes)
        .unwrap().commit().unwrap();
    let model = load_model(&engine, &load_args, load_limits, Duration::from_secs(3600),
        CancellationToken::default()).unwrap();
    let resident = engine.resources().memory_snapshot();
    for (operation, task) in [("generate", Task::Generate), ("chat", Task::Chat)] {
        let text = if task == Task::Chat {
            r#"[{"role":"user","content":"Reply with a short English greeting for Zoë."}]"#
        } else { "Reply with a short English greeting for Zoë." };
        for seeded in [false, true] {
            let (options, a) = make(operation, seeded);
            let (limits, bounds) = options.validate(&a).unwrap();
            let prepared = plan(text, task, &model, &a, limits);
            let identity = prepared.execution_identity().clone();
            let buffered = engine.execute_int4_chat(&model, prepared, 1, native(&a), SAMPLER_BYTES,
                limits.kernel, CancellationToken::default()).unwrap();
            let expected = serde_json::to_value(buffered.result()).unwrap();
            drop(buffered); assert_drained(&resident);
            for width in [2, 4] {
                let prepared = plan(text, task, &model, &a, limits);
                assert_eq!(prepared.execution_identity(), &identity);
                let wire = Arc::new(Mutex::new(Wire::default()));
                let sink = TokenSink::new_int4(Output { wire: Arc::clone(&wire), fault: None }, task,
                    bounds, model.model_identity(), prepared.native_plan().options()).unwrap()
                    .with_input_encoding(a.input_encoding.policy(a.max_input_bytes, limits.max_prompt_tokens).unwrap()).unwrap();
                let stream_limits = ChatStreamLimits { native: native(&a), max_sampler_bytes: SAMPLER_BYTES,
                    preparation_reserve_bytes: limits.preparation_bytes, sink_reserve_bytes: bounds.sink_memory_bytes };
                let mut completed = engine.execute_int4_chat_stream_with_threads(&model, prepared, 1,
                    stream_limits, limits.kernel, width, sink, CancellationToken::default()).unwrap();
                assert_eq!(serde_json::to_value(completed.result()).unwrap(), expected);
                let receipt = *completed.compute().unwrap();
                assert_eq!(receipt.compute_threads, width);
                assert!(receipt.completed_projections > 0 && receipt.completed_worker_rows > 0);
                assert!(receipt.completed_worker_rows < receipt.completed_rows);
                let closures = engine.resources().outstanding_closure_snapshot();
                assert_eq!(closures.active_closures, 0);
                assert_eq!(closures.wrapper_cancelled_closures, 0);
                completed.with_sink(|result, sink| sink.finish_int4_with_compute(result, &receipt)).unwrap();
                drop(completed); assert_drained(&resident);
                let wire = wire.lock().unwrap();
                let frames = frames(&wire.bytes);
                let terminal = frames.last().unwrap();
                assert_eq!(terminal["event"], "run_complete");
                assert_eq!(terminal["data"], expected);
                assert_eq!(terminal["compute"], serde_json::to_value(receipt).unwrap());
                let bytes: Vec<u8> = frames.iter().filter(|f| f["event"] == "token")
                    .flat_map(|f| f["data"]["decoded_bytes"].as_array().unwrap().iter()
                        .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())).collect();
                assert_eq!(bytes, expected["result"]["content"].as_str().unwrap().as_bytes());
                assert!(wire.bytes.len() as u64 <= bounds.stream_bytes);
            }
        }
    }
    for width in [2, 4] {
        for fault in [Fault::Write, Fault::Flush, Fault::BeforeRun, Fault::Reserve, Fault::Permit] {
            let (options, a) = make("generate", true);
            let (limits, bounds) = options.validate(&a).unwrap();
            let prepared = plan("Reply with a short English greeting.", Task::Generate, &model, &a, limits);
            let wire = Arc::new(Mutex::new(Wire::default()));
            let callbacks = Arc::new(Mutex::new([0; 2]));
            let cancellation = CancellationToken::default();
            if fault == Fault::BeforeRun { cancellation.cancel(DecodeCancellationKind::User); }
            let sink = Controlled {
                inner: TokenSink::new_int4(Output { wire: Arc::clone(&wire), fault: Some(fault) },
                    Task::Generate, bounds, model.model_identity(), prepared.native_plan().options()).unwrap(),
                cancellation: cancellation.clone(), fault, callbacks: Arc::clone(&callbacks),
            };
            let stream_limits = ChatStreamLimits { native: native(&a), max_sampler_bytes: SAMPLER_BYTES,
                preparation_reserve_bytes: limits.preparation_bytes, sink_reserve_bytes: bounds.sink_memory_bytes };
            let result = engine.execute_int4_chat_stream_with_threads(&model, prepared, 1, stream_limits,
                limits.kernel, width, sink, cancellation);
            let failure = host_failure(result.err().expect("failed delivery must not complete"));
            assert_drained(&resident);
            let wire = wire.lock().unwrap();
            assert!(!wire.bytes.windows(b"run_complete".len()).any(|w| w == b"run_complete"));
            match fault {
                Fault::Write | Fault::Flush => {
                    assert_eq!(failure.code, "stream_output_or_protocol");
                    assert_eq!(*callbacks.lock().unwrap(), [1, 1]);
                    assert_eq!(wire.writes, 2);
                    assert_eq!(wire.flushes, usize::from(fault == Fault::Flush));
                }
                Fault::BeforeRun | Fault::Reserve | Fault::Permit => {
                    assert_eq!(failure.exit, ErrorCode::Cancelled);
                    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::User));
                    let expected = match fault { Fault::BeforeRun => [0, 0], Fault::Reserve => [1, 0], _ => [1, 1] };
                    assert_eq!(*callbacks.lock().unwrap(), expected);
                    if fault != Fault::Permit { assert!(wire.bytes.is_empty()); }
                    else { assert!(frames(&wire.bytes).iter().all(|f| f["provisional"] == true)); }
                }
            }
        }
    }
    drop(model); drop(preparation); drop(lease); assert_drained(&idle);
}
