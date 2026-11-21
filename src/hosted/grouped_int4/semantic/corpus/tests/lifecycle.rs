//! Controller-only physical lifetime checks with no successful model forward.
use super::*;
use std::{io::{self, Cursor, Read}, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};

#[derive(Clone, Copy)]
enum Delivery { Complete, PartialTerminal, CancelAtEof, CancelAtTerminal,
    PanicAtInput, PanicAtRecord, PanicAtTerminal }
#[derive(Default)]
struct Observed {
    bytes: Vec<u8>, terminals: usize, reader_drops: usize, writer_drops: usize,
    reader_unwound: bool, writer_unwound: bool, charged_at_drop: bool,
}
struct Reader {
    input: Cursor<Vec<u8>>, observed: Arc<Mutex<Observed>>, panic_on_read: bool,
}
impl Read for Reader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        assert!(!self.panic_on_read, "injected semantic source panic");
        self.input.read(bytes)
    }
}
impl BufRead for Reader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        assert!(!self.panic_on_read, "injected semantic source panic");
        self.input.fill_buf()
    }
    fn consume(&mut self, bytes: usize) { self.input.consume(bytes); }
}
impl Drop for Reader {
    fn drop(&mut self) {
        let mut observed = self.observed.lock().unwrap();
        observed.reader_drops += 1;
        observed.reader_unwound = std::thread::panicking();
    }
}
struct Writer {
    host: Arc<NlpEngine>, observed: Arc<Mutex<Observed>>, stop: CancellationToken,
    delivery: Delivery, terminal: bool, eof: bool, job_bytes: u64,
}
impl Writer {
    fn charged(&self) -> bool {
        let memory = self.host.resources().memory_snapshot();
        memory.by_class.get(&MemoryClass::JobBuffers).is_some_and(|s| s.committed_bytes >= self.job_bytes)
            && [MemoryClass::KvPages, MemoryClass::ActivationScratch].iter()
                .all(|class| memory.by_class.get(class).is_some_and(|s| s.committed_bytes > 0))
    }
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.terminal = event["event"] == "run_complete";
        self.eof = event["event"] == "flush" && event["eof"] == true;
        assert!(self.charged(), "complete input and output claims must survive all delivery callbacks");
        assert_eq!(self.host.resources().outstanding_closure_snapshot().active_closures, 1);
        if !self.terminal && matches!(self.delivery, Delivery::PanicAtRecord) {
            panic!("injected semantic record writer panic");
        }
        if self.terminal {
            self.observed.lock().unwrap().terminals += 1;
            if matches!(self.delivery, Delivery::PanicAtTerminal) { panic!("injected semantic footer panic"); }
            if matches!(self.delivery, Delivery::PartialTerminal) {
                self.observed.lock().unwrap().bytes.extend_from_slice(&bytes[..7]);
                return Err(io::Error::other("injected semantic partial footer"));
            }
        }
        self.observed.lock().unwrap().bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.eof && matches!(self.delivery, Delivery::CancelAtEof) { self.stop.cancel(DecodeCancellationKind::Deadline); }
        if self.terminal && matches!(self.delivery, Delivery::CancelAtTerminal) { self.stop.cancel(DecodeCancellationKind::Shutdown); }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        let charged = self.charged();
        let mut observed = self.observed.lock().unwrap();
        observed.writer_drops += 1;
        observed.writer_unwound = std::thread::panicking();
        observed.charged_at_drop = charged;
    }
}
struct NoRead { host: Arc<NlpEngine>, dropped: Arc<AtomicUsize>, active_at_drop: Arc<AtomicUsize> }
impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted semantic source read"); } }
impl BufRead for NoRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("unadmitted semantic source fill"); }
    fn consume(&mut self, _: usize) { panic!("unadmitted semantic source consume"); }
}
impl Drop for NoRead {
    fn drop(&mut self) {
        self.active_at_drop.store(self.host.resources().outstanding_closure_snapshot().active_closures, Ordering::SeqCst);
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0);
    assert_eq!(closures.wrapper_cancelled_closures, 0);
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights exercise owned IO, cancellation and physical drain; no inference evidence"]
fn semantic_corpus_retains_actual_inputs_complete_output_and_native_claims_through_terminal_paths() {
    assert!(crate::installed_process_resources().is_none());
    let host = Arc::new(NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16 * 1024 * MIB,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap());
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, MIB).unwrap(),
        || Ok(projection_fixture(WIDTH).0)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let baseline = host.resources().memory_snapshot();
    for threads in [1, 4] {
        for delivery in [Delivery::Complete, Delivery::PartialTerminal, Delivery::CancelAtEof,
            Delivery::CancelAtTerminal, Delivery::PanicAtInput, Delivery::PanicAtRecord, Delivery::PanicAtTerminal] {
            let config = fixture();
            let required = config.requirements(WIDTH).unwrap();
            let stop = CancellationToken::default();
            let observed = Arc::new(Mutex::new(Observed::default()));
            let input = Reader { input: Cursor::new(b"invalid-json\n{\"flush\":true}\n".to_vec()),
                observed: Arc::clone(&observed), panic_on_read: matches!(delivery, Delivery::PanicAtInput) };
            let output = Writer { host: Arc::clone(&host), observed: Arc::clone(&observed), stop: stop.clone(),
                delivery, terminal: false, eof: false,
                job_bytes: required.input_bytes + required.output_reservation_bytes };
            let result = if threads == 1 {
                host.batch_int4_semantic_extract(&model, config, PackedInt4Kernel::scalar(), input, output, stop)
            } else {
                host.batch_int4_semantic_extract_with_threads(&model, config,
                    PackedInt4Kernel::scalar(), threads, input, output, stop)
            };
            match delivery {
                Delivery::Complete => {
                    let summary = result.unwrap();
                    assert_eq!(summary.succeeded, 0); assert_eq!(summary.failed, 1); assert_eq!(summary.flushes, 2);
                }
                Delivery::PartialTerminal => assert!(matches!(result, Err(HostedError::Batch(error))
                    if error.fault.code == BatchCode::OutputIo)),
                Delivery::CancelAtEof => assert!(matches!(result, Err(HostedError::Stopped { stop, .. })
                    if stop.kind == DecodeCancellationKind::Deadline)),
                Delivery::CancelAtTerminal => assert!(matches!(result, Err(HostedError::Stopped { stop, .. })
                    if stop.kind == DecodeCancellationKind::Shutdown)),
                Delivery::PanicAtInput | Delivery::PanicAtRecord | Delivery::PanicAtTerminal =>
                    assert!(matches!(result, Err(HostedError::Panicked | HostedError::Scope { .. }))),
            }
            let observed = observed.lock().unwrap();
            assert_eq!(observed.reader_drops, 1); assert_eq!(observed.writer_drops, 1);
            assert!(observed.charged_at_drop);
            if matches!(delivery, Delivery::PanicAtInput | Delivery::PanicAtRecord) {
                assert!(!observed.reader_unwound && !observed.writer_unwound,
                    "pre-join callback panics must return owned input for post-join disposal");
            }
            let terminal = !matches!(delivery, Delivery::CancelAtEof | Delivery::PanicAtInput | Delivery::PanicAtRecord);
            assert_eq!(observed.terminals, if terminal { 1 } else { 0 });
            if !terminal { assert!(!std::str::from_utf8(&observed.bytes).unwrap().contains("run_complete")); }
            drained(&host, &baseline);
        }
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let stop = CancellationToken::default(); stop.cancel(cause);
            let dropped = Arc::new(AtomicUsize::new(0)); let active = Arc::new(AtomicUsize::new(usize::MAX));
            let input = NoRead { host: Arc::clone(&host), dropped: Arc::clone(&dropped), active_at_drop: Arc::clone(&active) };
            let result = if threads == 1 {
                host.batch_int4_semantic_extract(&model, fixture(), PackedInt4Kernel::scalar(), input, Vec::new(), stop)
            } else {
                host.batch_int4_semantic_extract_with_threads(&model, fixture(),
                    PackedInt4Kernel::scalar(), threads, input, Vec::new(), stop)
            };
            assert!(matches!(result, Err(HostedError::Stopped { stop, .. }) if stop.kind == cause));
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
            assert_eq!(active.load(Ordering::SeqCst), 0);
            drained(&host, &baseline);
        }
        let mut config = fixture(); config.extraction.schema = "{invalid-schema}".to_owned();
        let dropped = Arc::new(AtomicUsize::new(0)); let active = Arc::new(AtomicUsize::new(usize::MAX));
        let input = NoRead { host: Arc::clone(&host), dropped: Arc::clone(&dropped), active_at_drop: Arc::clone(&active) };
        let result = if threads == 1 {
            host.batch_int4_semantic_extract(&model, config, PackedInt4Kernel::scalar(), input, Vec::new(), CancellationToken::default())
        } else {
            host.batch_int4_semantic_extract_with_threads(&model, config, PackedInt4Kernel::scalar(),
                threads, input, Vec::new(), CancellationToken::default())
        };
        assert!(matches!(result, Err(HostedError::BatchSetup(_))),
            "invalid fixed schema must fail before the private reader is invoked");
        assert_eq!(dropped.load(Ordering::SeqCst), 1); drained(&host, &baseline);
    }
    for checkpoints in [2, 3] {
        let mut config = fixture(); config.limits.native.run.max_checkpoints = checkpoints;
        let dropped = Arc::new(AtomicUsize::new(0)); let active = Arc::new(AtomicUsize::new(usize::MAX));
        let input = NoRead { host: Arc::clone(&host), dropped: Arc::clone(&dropped), active_at_drop: Arc::clone(&active) };
        let result = host.batch_int4_semantic_extract_with_threads(&model, config,
            PackedInt4Kernel::scalar(), 4, input, Vec::new(), CancellationToken::default());
        assert!(matches!(result, Err(HostedError::Stopped { stop, .. })
            if stop.kind == DecodeCancellationKind::PollQuota));
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 0, "partial team formation must finish before queued source disposal");
        drained(&host, &baseline);
    }
    drop(model); drained(&host, &idle);
}
