//! Actual dispatch/transport ownership fixtures; no synthetic judgment parity.
use super::*;
use std::{io::{self, Cursor, Read}, sync::Mutex};
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    hosted::RunStop, native_engine::{decode::DecodeCancellationKind,
        strict_int4::weights::projection_fixture}, tasks::judge::FaithfulnessPolicy};

const INPUT: &[u8] = b"invalid-json\n{\"flush\":true}\n";
const PREPARATION: u64 = 128 << 20;
#[derive(Clone, Copy)]
enum Delivery { Complete, PartialTerminal, CancelAtEof, CancelAtTerminalFlush, PanicAtTerminal }
#[derive(Default)]
struct Observation { bytes: Vec<u8>, writes: usize, terminals: usize, drops: usize, input_at_drop: u64 }
struct Writer { host: Arc<NlpEngine>, observed: Arc<Mutex<Observation>>, run: RunLimits,
    threads: usize, delivery: Delivery, stop: CancellationToken, terminal: bool, eof: bool }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.terminal = event["event"] == "run_complete";
        self.eof = event["event"] == "flush" && event["eof"] == true;
        assert_eq!(self.host.resources().outstanding_closure_snapshot().active_closures, 1);
        assert!(matches!(dispatch::preflight(&self.host, self.run), Err(HostedError::Reentrant)));
        if self.terminal {
            let snapshot = self.host.resources().memory_snapshot();
            let kv = snapshot.by_class.get(&MemoryClass::KvPages).map_or(0, |value| value.committed_bytes);
            if self.threads > 1 {
                assert_eq!(kv, 0, "native state must drop before post-join completion");
                assert_eq!(snapshot.by_class[&MemoryClass::ActivationScratch].committed_bytes,
                    int4_task_team_workspace_bytes(self.threads, 32).unwrap());
            } else { assert!(kv > 0, "preserve the scalar terminal schedule"); }
            self.observed.lock().unwrap().terminals += 1;
        }
        self.observed.lock().unwrap().writes += 1;
        if self.terminal && matches!(self.delivery, Delivery::PanicAtTerminal) {
            panic!("injected faithfulness terminal writer panic");
        }
        let mut observed = self.observed.lock().unwrap();
        if self.terminal && matches!(self.delivery, Delivery::PartialTerminal) {
            observed.bytes.extend_from_slice(&bytes[..bytes.len().min(7)]);
            return Err(io::Error::other("injected faithfulness terminal partial write"));
        }
        observed.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.eof && matches!(self.delivery, Delivery::CancelAtEof) {
            self.stop.cancel(DecodeCancellationKind::Deadline);
        }
        if self.terminal && matches!(self.delivery, Delivery::CancelAtTerminalFlush) {
            self.stop.cancel(DecodeCancellationKind::Shutdown);
        }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        let snapshot = self.host.resources().memory_snapshot();
        let mut observed = self.observed.lock().unwrap();
        observed.input_at_drop = snapshot.by_class.get(&MemoryClass::JobBuffers)
            .map_or(0, |value| value.committed_bytes + value.reserved_bytes);
        observed.drops += 1;
    }
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted faithfulness input read"); }
}
impl BufRead for NoRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("unadmitted faithfulness input buffered"); }
    fn consume(&mut self, _: usize) { panic!("unadmitted faithfulness input consumed"); }
}
fn policy(max_heads: usize) -> FaithfulnessPolicy {
    let windows = u32::try_from(if max_heads == 1 { 1 } else { max_heads - 1 }).unwrap();
    FaithfulnessPolicy { minimum_candidate_weight_ppm: 730_000, minimum_margin_milli: 250,
        evidence_window_bytes: 16, max_evidence_windows: windows, max_evidence_spans: windows }
}
fn planner(model: &ResidentInt4, max_heads: usize, cohorts: Option<usize>)
    -> (Int4FaithfulnessDocumentBatchPlanner, CorpusLimits) {
    let (budget, planning, mapping, mut limits) = fixture();
    limits.native.run.max_elapsed = Duration::from_secs(60);
    limits.native.run.max_checkpoints = 100_000;
    let compiler = Int4JudgePlanner::for_model(model.model_identity(), budget, planning).unwrap();
    let planner = Int4FaithfulnessDocumentBatchPlanner::new(compiler, policy(max_heads),
        mapping, mapping.max_model_work).unwrap().with_max_heads(max_heads).unwrap();
    let planner = if let Some(chunks) = cohorts { planner.with_chunk_cohorts(chunks).unwrap() } else { planner };
    (planner, limits)
}
#[allow(clippy::too_many_arguments)]
fn execute<R: BufRead + Send + 'static>(host: &Arc<NlpEngine>, model: &ResidentInt4,
    threads: Option<usize>, max_heads: usize, cohorts: Option<usize>, delivery: Delivery,
    input: R, stop: CancellationToken) -> (Result<BatchSummary, HostedError>, Arc<Mutex<Observation>>) {
    let (planner, limits) = planner(model, max_heads, cohorts);
    let observed = Arc::new(Mutex::new(Observation::default()));
    let writer = Writer { host: Arc::clone(host), observed: Arc::clone(&observed), run: limits.native.run,
        threads: threads.unwrap_or(1), delivery, stop: stop.clone(), terminal: false, eof: false };
    let result = if let Some(threads) = threads {
        host.batch_int4_faithfulness_documents_with_threads(model, planner, limits,
            PackedInt4Kernel::scalar(), threads, input, writer, stop)
    } else {
        host.batch_int4_faithfulness_documents(model, planner, limits,
            PackedInt4Kernel::scalar(), input, writer, stop)
    };
    (result, observed)
}
fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights prove ownership, never native faithfulness parity"]
fn faithfulness_stream_team_joins_before_terminal_delivery_and_drains_failures() {
    assert!(crate::installed_process_resources().is_none());
    let host = Arc::new(NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
        helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 1024 << 20,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap());
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
        || Ok(projection_fixture(32).0)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let baseline = host.resources().memory_snapshot();
    let (task, planning, mapping, limits) = fixture();
    let charge = int4_faithfulness_document_corpus_requirements_with_max_heads(
        32, task, planning, mapping, limits, 1, None).unwrap().input_bytes;
    for max_heads in [1, 4, 32] {
        let mut expected = None;
        for threads in [None, Some(2), Some(3)] {
            let (result, observed) = execute(&host, &model, threads, max_heads, None,
                Delivery::Complete, Cursor::new(INPUT.to_vec()), CancellationToken::default());
            let summary = result.unwrap();
            assert_eq!(summary.failed, 1); assert_eq!(summary.succeeded, 0); assert_eq!(summary.flushes, 2);
            assert_eq!(summary.reserved_work, crate::batch::BatchWork::default());
            let observed = observed.lock().unwrap();
            assert_eq!(observed.drops, 1); assert_eq!(observed.terminals, 1);
            assert!(observed.input_at_drop >= PREPARATION + charge);
            if let Some(expected) = &expected { assert_eq!(&observed.bytes, expected); }
            else { expected = Some(observed.bytes.clone()); }
            drained(&host, &baseline);
        }
    }
    for delivery in [Delivery::PartialTerminal, Delivery::CancelAtEof,
        Delivery::CancelAtTerminalFlush, Delivery::PanicAtTerminal] {
        let (result, observed) = execute(&host, &model, Some(3), 4, None, delivery,
            Cursor::new(INPUT.to_vec()), CancellationToken::default());
        match delivery {
            Delivery::PartialTerminal => assert!(matches!(result, Err(HostedError::Batch(error))
                if error.fault.code == BatchCode::OutputIo)),
            Delivery::CancelAtEof => assert!(matches!(result, Err(HostedError::Stopped {
                stop: RunStop { kind: DecodeCancellationKind::Deadline, .. }, .. }))),
            Delivery::CancelAtTerminalFlush => assert!(matches!(result, Err(HostedError::Stopped {
                stop: RunStop { kind: DecodeCancellationKind::Shutdown, .. }, .. }))),
            Delivery::PanicAtTerminal => assert!(matches!(result, Err(HostedError::Panicked))),
            Delivery::Complete => unreachable!(),
        }
        let observed = observed.lock().unwrap();
        assert_eq!(observed.drops, 1); assert!(observed.input_at_drop >= PREPARATION + charge);
        let text = std::str::from_utf8(&observed.bytes).unwrap();
        if matches!(delivery, Delivery::CancelAtEof) {
            assert_eq!(observed.terminals, 0); assert!(!text.contains("run_complete"));
            assert!(text.contains("run_error"));
        } else { assert_eq!(observed.terminals, 1); assert!(!text.contains("run_error")); }
        drained(&host, &baseline);
    }
    for (threads, cohorts) in [(0, None), (1, None), (4, None), (65, None), (usize::MAX, None),
        (2, Some(1)), (3, Some(2))] {
        let (result, observed) = execute(&host, &model, Some(threads), 32, cohorts,
            Delivery::Complete, NoRead, CancellationToken::default());
        assert!(matches!(result, Err(HostedError::Limits(_))));
        let observed = observed.lock().unwrap();
        assert_eq!(observed.writes, 0); assert_eq!(observed.drops, 1);
        drained(&host, &baseline);
    }
    {
        let required = int4_faithfulness_document_corpus_requirements_with_max_heads(
            32, task, planning, mapping, limits, 32, None).unwrap();
        let room = sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, int4_task_team_workspace_bytes(3, 32).unwrap() - 1]).unwrap();
        let bytes = host.resources().config().memory_ceiling_bytes
            .checked_sub(baseline.committed_bytes + baseline.reserved_bytes).unwrap().checked_sub(room).unwrap();
        let lease = host.resources().acquire_lease();
        let held = Pending::reserve(&lease, MemoryClass::JobBuffers, bytes).unwrap().commit().unwrap();
        let (result, observed) = execute(&host, &model, Some(3), 32, None,
            Delivery::Complete, NoRead, CancellationToken::default());
        assert!(matches!(result, Err(HostedError::Reservation(_))));
        let observed = observed.lock().unwrap();
        assert_eq!(observed.writes, 0); assert_eq!(observed.drops, 1);
        drop(held); drop(lease);
        drained(&host, &baseline);
    }
    drop(model); drop(preparation);
    drained(&host, &idle);
}
