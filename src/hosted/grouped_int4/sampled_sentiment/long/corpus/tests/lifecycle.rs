//! Fresh-process ownership/transport tests with no successful model inference.
use super::*;
use std::{io::{self, Cursor, Read}, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};

#[derive(Clone, Copy)]
enum Delivery { Complete, PartialTerminal, CancelAtEof, CancelAtTerminal, PanicAtTerminal,
    PanicAtInput, PanicAtRecord }
#[derive(Default)]
struct Observed { bytes: Vec<u8>, terminals: usize, drops: usize, charged_at_drop: bool,
    reader_drops: usize, reader_dropped_unwinding: bool, writer_dropped_unwinding: bool }
struct Reader { inner: Cursor<Vec<u8>>, observed: Arc<Mutex<Observed>>, panic_on_read: bool }
impl Read for Reader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        assert!(!self.panic_on_read, "injected source reader panic");
        self.inner.read(bytes)
    }
}
impl BufRead for Reader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        assert!(!self.panic_on_read, "injected source reader panic");
        self.inner.fill_buf()
    }
    fn consume(&mut self, bytes: usize) { self.inner.consume(bytes); }
}
impl Drop for Reader {
    fn drop(&mut self) {
        let mut observed = self.observed.lock().unwrap();
        observed.reader_drops += 1;
        observed.reader_dropped_unwinding = std::thread::panicking();
    }
}
struct Writer { host: Arc<NlpEngine>, observed: Arc<Mutex<Observed>>, stop: CancellationToken,
    delivery: Delivery, terminal: bool, eof: bool }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.terminal = event["event"] == "run_complete";
        self.eof = event["event"] == "flush" && event["eof"] == true;
        let snapshot = self.host.resources().memory_snapshot();
        for class in [MemoryClass::JobBuffers, MemoryClass::KvPages, MemoryClass::ActivationScratch] {
            assert!(snapshot.by_class[&class].committed_bytes > 0,
                "input/native charges survive every write including the post-join footer");
        }
        assert_eq!(self.host.resources().outstanding_closure_snapshot().active_closures, 1);
        if !self.terminal && matches!(self.delivery, Delivery::PanicAtRecord) {
            panic!("injected pre-join writer panic");
        }
        if self.terminal {
            self.observed.lock().unwrap().terminals += 1;
            if matches!(self.delivery, Delivery::PanicAtTerminal) { panic!("injected terminal writer panic"); }
            if matches!(self.delivery, Delivery::PartialTerminal) {
                self.observed.lock().unwrap().bytes.extend_from_slice(&bytes[..7]);
                return Err(io::Error::other("injected terminal partial write"));
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
        let snapshot = self.host.resources().memory_snapshot();
        let mut observed = self.observed.lock().unwrap();
        observed.charged_at_drop = [MemoryClass::JobBuffers, MemoryClass::KvPages, MemoryClass::ActivationScratch]
            .iter().all(|class| snapshot.by_class.get(class).is_some_and(|state| state.committed_bytes > 0));
        observed.drops += 1;
        observed.writer_dropped_unwinding = std::thread::panicking();
    }
}
struct NoRead { host: Arc<NlpEngine>, drops: Arc<AtomicUsize>, active_at_drop: Arc<AtomicUsize> }
impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted source read"); } }
impl BufRead for NoRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("unadmitted source fill"); }
    fn consume(&mut self, _: usize) { panic!("unadmitted source consume"); }
}
impl Drop for NoRead {
    fn drop(&mut self) {
        self.active_at_drop.store(self.host.resources().outstanding_closure_snapshot().active_closures,
            Ordering::SeqCst);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}
fn config(schedule: usize) -> SampledSentimentDocumentBatchHostConfig {
    let mut config = fixture(WIDTH);
    config.document.reuse_prefill = schedule >= 3;
    if schedule % 3 == 1 { config.document.prefill = Some(prefill(4)); }
    config
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights exercise transport, original cancellation and physical drain"]
fn complete_document_stream_keeps_charges_until_delivery_and_drains_every_terminal_path() {
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
    for schedule in 0..6 {
        for delivery in [Delivery::Complete, Delivery::PartialTerminal, Delivery::CancelAtEof,
            Delivery::CancelAtTerminal, Delivery::PanicAtTerminal, Delivery::PanicAtInput,
            Delivery::PanicAtRecord] {
            let stop = CancellationToken::default(); let observed = Arc::new(Mutex::new(Observed::default()));
            let writer = Writer { host: Arc::clone(&host), observed: Arc::clone(&observed),
                stop: stop.clone(), delivery, terminal: false, eof: false };
            let input = Reader { inner: Cursor::new(b"invalid-json\n{\"flush\":true}\n".to_vec()),
                observed: Arc::clone(&observed), panic_on_read: matches!(delivery, Delivery::PanicAtInput) };
            let result = if schedule % 3 == 2 {
                host.batch_int4_sampled_sentiment_documents_with_threads(&model, config(schedule),
                    PackedInt4Kernel::scalar(), 4, input, writer, stop)
            } else { host.batch_int4_sampled_sentiment_documents(&model, config(schedule),
                PackedInt4Kernel::scalar(), input, writer, stop) };
            match delivery {
                Delivery::Complete => { let summary = result.unwrap(); assert_eq!(summary.failed, 1);
                    assert_eq!(summary.succeeded, 0); assert_eq!(summary.flushes, 2); },
                Delivery::PartialTerminal => assert!(matches!(result, Err(HostedError::Batch(error))
                    if error.fault.code == BatchCode::OutputIo)),
                Delivery::CancelAtEof => assert!(matches!(result, Err(HostedError::Stopped { stop, .. })
                    if stop.kind == DecodeCancellationKind::Deadline)),
                Delivery::CancelAtTerminal => assert!(matches!(result, Err(HostedError::Stopped { stop, .. })
                    if stop.kind == DecodeCancellationKind::Shutdown)),
                Delivery::PanicAtTerminal | Delivery::PanicAtInput | Delivery::PanicAtRecord => assert!(matches!(result,
                    Err(HostedError::Panicked | HostedError::Scope { .. }))),
            }
            let observed = observed.lock().unwrap();
            assert_eq!(observed.drops, 1); assert!(observed.charged_at_drop);
            assert_eq!(observed.reader_drops, 1);
            if matches!(delivery, Delivery::PanicAtInput | Delivery::PanicAtRecord) {
                assert!(!observed.reader_dropped_unwinding && !observed.writer_dropped_unwinding,
                    "pre-join callback panic must return the actual input owners for deferred disposal");
            }
            let terminals = if matches!(delivery,
                Delivery::CancelAtEof | Delivery::PanicAtInput | Delivery::PanicAtRecord) { 0 } else { 1 };
            assert_eq!(observed.terminals, terminals);
            if terminals == 0 { assert!(!std::str::from_utf8(&observed.bytes).unwrap().contains("run_complete")); }
            drained(&host, &baseline);
        }
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let stop = CancellationToken::default(); stop.cancel(cause);
            let dropped = Arc::new(AtomicUsize::new(0)); let active = Arc::new(AtomicUsize::new(usize::MAX));
            let input = NoRead { host: Arc::clone(&host), drops: Arc::clone(&dropped), active_at_drop: Arc::clone(&active) };
            let result = if schedule % 3 == 2 {
                host.batch_int4_sampled_sentiment_documents_with_threads(&model, config(schedule),
                    PackedInt4Kernel::scalar(), 4, input, Vec::new(), stop)
            } else { host.batch_int4_sampled_sentiment_documents(&model, config(schedule),
                PackedInt4Kernel::scalar(), input, Vec::new(), stop) };
            assert!(matches!(result, Err(HostedError::Stopped { stop, .. }) if stop.kind == cause));
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
            assert_eq!(active.load(Ordering::SeqCst), 0,
                "queued cancellation must retain input until the physical completion handoff");
            drained(&host, &baseline);
        }
    }
    for checkpoints in [2, 3] {
        let mut config = config(0); config.document.native.run.max_checkpoints = checkpoints;
        let dropped = Arc::new(AtomicUsize::new(0)); let active = Arc::new(AtomicUsize::new(usize::MAX));
        let input = NoRead { host: Arc::clone(&host), drops: Arc::clone(&dropped), active_at_drop: Arc::clone(&active) };
        let result = host.batch_int4_sampled_sentiment_documents_with_threads(&model, config,
            PackedInt4Kernel::scalar(), 4, input, Vec::new(), CancellationToken::default());
        assert!(matches!(result, Err(HostedError::Stopped { stop, .. })
            if stop.kind == DecodeCancellationKind::PollQuota));
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 0,
            "partial team formation must join before releasing queued input");
        drained(&host, &baseline);
    }
    for threads in [0, 1, 5, 65, usize::MAX] {
        let dropped = Arc::new(AtomicUsize::new(0)); let active = Arc::new(AtomicUsize::new(usize::MAX));
        let input = NoRead { host: Arc::clone(&host), drops: Arc::clone(&dropped), active_at_drop: Arc::clone(&active) };
        let result = host.batch_int4_sampled_sentiment_documents_with_threads(&model, config(0),
            PackedInt4Kernel::scalar(), threads, input, Vec::new(), CancellationToken::default());
        assert!(matches!(result, Err(HostedError::Limits(_))));
        assert_eq!(dropped.load(Ordering::SeqCst), 1); drained(&host, &baseline);
    }
    drop(model); drained(&host, &idle);
}
