//! Real runtime/ledger ownership through the public structured stream APIs.
//! Tiny projection fixtures are NOT complete models: input here contains only
//! transport refusals/flushes, so no neural forward or model parity is claimed.
use super::*;
use std::{io::{self, Cursor, Read}, sync::Mutex};
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture},
    tasks::{BuiltInTask, ir::TaskBudget, source_planning::{SourcePlanningLimits, SourceTaskPlanner}}};

const INPUT: &[u8] = b"invalid-json\n{\"flush\":true}\n";

#[derive(Clone, Copy)]
enum Delivery { Complete, PartialTerminal, CancelAtEof, CancelAtTerminalFlush, PanicAtTerminal }
#[derive(Default)]
struct Observation {
    bytes: Vec<u8>, writes: usize, terminal_checks: usize, dropped: usize,
    job_bytes_at_drop: u64,
}
struct Writer {
    host: Arc<NlpEngine>, observation: Arc<Mutex<Observation>>, run: RunLimits,
    threads: usize, delivery: Delivery, cancellation: CancellationToken,
    last_terminal: bool, last_eof: bool,
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let event: serde_json::Value = serde_json::from_slice(bytes).expect("complete canonical transport record");
        self.last_terminal = event["event"] == "run_complete";
        self.last_eof = event["event"] == "flush" && event["eof"] == true;
        assert_eq!(self.host.resources().outstanding_closure_snapshot().active_closures, 1);
        assert!(matches!(dispatch::preflight(&self.host, self.run), Err(HostedError::Reentrant)),
            "writer callbacks must remain inside the original native boundary");
        if self.last_terminal {
            let snapshot = self.host.resources().memory_snapshot();
            let kv = snapshot.by_class.get(&MemoryClass::KvPages).map_or(0, |charge| charge.committed_bytes);
            if self.threads > 1 {
                assert_eq!(kv, 0, "native engine must drop before post-join terminal delivery");
                assert_eq!(snapshot.by_class[&MemoryClass::ActivationScratch].committed_bytes,
                    int4_task_team_workspace_bytes(self.threads, 4).unwrap(),
                    "dispatcher must retain its worker charge through final delivery");
            } else {
                assert!(kv > 0, "legacy serial completion schedule changed");
            }
            self.observation.lock().unwrap().terminal_checks += 1;
        }
        self.observation.lock().unwrap().writes += 1;
        if self.last_terminal && matches!(self.delivery, Delivery::PanicAtTerminal) {
            panic!("injected terminal writer panic");
        }
        let mut observed = self.observation.lock().unwrap();
        if self.last_terminal && matches!(self.delivery, Delivery::PartialTerminal) {
            observed.bytes.extend_from_slice(&bytes[..bytes.len().min(7)]);
            return Err(io::Error::other("injected terminal partial write"));
        }
        observed.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.last_eof && matches!(self.delivery, Delivery::CancelAtEof) {
            self.cancellation.cancel(DecodeCancellationKind::Deadline);
        }
        if self.last_terminal && matches!(self.delivery, Delivery::CancelAtTerminalFlush) {
            self.cancellation.cancel(DecodeCancellationKind::Shutdown);
        }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        let snapshot = self.host.resources().memory_snapshot();
        let mut observed = self.observation.lock().unwrap();
        observed.job_bytes_at_drop = snapshot.by_class.get(&MemoryClass::JobBuffers)
            .map_or(0, |charge| charge.committed_bytes + charge.reserved_bytes);
        observed.dropped += 1;
    }
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted stream read input"); }
}
impl BufRead for NoRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("unadmitted stream filled input"); }
    fn consume(&mut self, _: usize) { panic!("unadmitted stream consumed input"); }
}

fn writer(host: &Arc<NlpEngine>, threads: usize, delivery: Delivery, cancellation: &CancellationToken)
    -> (Writer, Arc<Mutex<Observation>>) {
    let observation = Arc::new(Mutex::new(Observation::default()));
    (Writer { host: Arc::clone(host), observation: Arc::clone(&observation), run: limits().native.run,
        threads, delivery, cancellation: cancellation.clone(), last_terminal: false, last_eof: false }, observation)
}

#[allow(clippy::too_many_arguments)]
fn execute<R: BufRead + Send + 'static>(host: &NlpEngine, model: &ResidentInt4,
    vocabulary: &Arc<ExtractionVocabulary>, source: &Arc<SourceTaskPlanner>, task: Option<BuiltInTask>,
    threads: usize, selected_rows: Option<Int4JsonSparseLimits>, input: R, writer: Writer,
    cancellation: CancellationToken) -> Result<BatchSummary, HostedError> {
    let limits = limits();
    let ceiling = TaskBudget { max_input_tokens: 32, max_output_tokens: 4, max_output_bytes: 8192,
        max_grammar_states: 256, max_kv_bytes: 32 * KV_BYTES_PER_TOKEN as u64 };
    let mut native = native_work(4);
    if let Some(selected) = selected_rows {
        native.max_model_work = constrained_int4::sparse::planned_work(4, 8, 4, selected).unwrap();
    }
    let planner = Int4ExtractionBatchPlanner::for_model(model.model_identity(), ceiling,
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap();
    if let Some(task) = task {
        let mut identity = planner.base_execution_identity().clone();
        identity.task_spec = task.spec().identity();
        identity.tokenizer_digest = source.tokenizer_digest();
        identity.template_digest = *source.template_digest();
        drop(planner);
        let config = Int4SourceCorpusConfig { identity, task_ceiling: ceiling,
            planning: SourcePlanningLimits::default(), defaults: None, native_work: native, selected_rows };
        if threads == 1 {
            host.batch_int4_source(model, Arc::clone(source), Arc::clone(vocabulary), config, limits,
                PackedInt4Kernel::scalar(), input, writer, cancellation)
        } else {
            host.batch_int4_source_with_threads(model, Arc::clone(source), Arc::clone(vocabulary), config,
                limits, PackedInt4Kernel::scalar(), threads, input, writer, cancellation)
        }
    } else {
        let planner = match selected_rows { Some(limits) => planner.with_selected_rows(limits).unwrap(), None => planner };
        if threads == 1 {
            host.batch_int4_extract(model, planner, Arc::clone(vocabulary), native, limits,
                PackedInt4Kernel::scalar(), input, writer, cancellation)
        } else {
            host.batch_int4_extract_with_threads(model, planner, Arc::clone(vocabulary), native, limits,
                PackedInt4Kernel::scalar(), threads, input, writer, cancellation)
        }
    }
}

fn drained(host: &NlpEngine, baseline: &crate::orchestrator::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}

#[test]
#[ignore = "run alone under controller-admitted real runtime; tiny protocol fixture performs no neural inference"]
fn structured_stream_team_delivery_refusal_cancellation_and_panic_keep_owned_io_charged() {
    assert!(crate::installed_process_resources().is_none(), "fixture requires a fresh process");
    let host = Arc::new(NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
        helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 512 << 20,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap());
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 128 << 20).unwrap().commit().unwrap();
    let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos = registry.template_controls().entries().iter().find(|entry|
        entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(registry.template_controls()).unwrap());
    let source = Arc::new(SourceTaskPlanner::pinned(registry.template_controls(), eos).unwrap());
    let (weights, _) = projection_fixture(4);
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(), || Ok(weights)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let baseline = host.resources().memory_snapshot();
    let input_charge = requirements(4, native_work(4), limits()).unwrap().input_bytes;
    for task in [None, Some(BuiltInTask::Ner), Some(BuiltInTask::Keyphrases),
        Some(BuiltInTask::Summarize), Some(BuiltInTask::Answer)] {
        for selected in [None, Some(Int4JsonSparseLimits { max_rows_per_step: 7 })] {
            let mut expected = None;
            for threads in [1, 2, 3] {
                let cancellation = CancellationToken::default();
                let (writer, observed) = writer(&host, threads, Delivery::Complete, &cancellation);
                let summary = execute(&host, &model, &vocabulary, &source, task, threads,
                    selected, Cursor::new(INPUT.to_vec()), writer, cancellation).unwrap();
                assert_eq!(summary.failed, 1); assert_eq!(summary.succeeded, 0); assert_eq!(summary.flushes, 2);
                assert_eq!(summary.reserved_work, crate::batch::BatchWork::default());
                let observed = observed.lock().unwrap();
                assert_eq!(observed.dropped, 1); assert_eq!(observed.terminal_checks, 1);
                assert!(observed.job_bytes_at_drop >= input_charge + (128 << 20));
                if let Some(expected) = &expected { assert_eq!(&observed.bytes, expected); }
                else { expected = Some(observed.bytes.clone()); }
                drained(&host, &baseline);
            }
        }
        for delivery in [Delivery::PartialTerminal, Delivery::CancelAtEof,
            Delivery::CancelAtTerminalFlush, Delivery::PanicAtTerminal] {
            let cancellation = CancellationToken::default();
            let (writer, observed) = writer(&host, 3, delivery, &cancellation);
            let error = execute(&host, &model, &vocabulary, &source, task, 3,
                Some(Int4JsonSparseLimits { max_rows_per_step: 7 }), Cursor::new(INPUT.to_vec()),
                writer, cancellation).unwrap_err();
            match delivery {
                Delivery::PartialTerminal => assert!(matches!(error, HostedError::Batch(error)
                    if error.fault.code == BatchCode::OutputIo)),
                Delivery::CancelAtEof => assert!(matches!(error, HostedError::Stopped {
                    stop: RunStop { kind: DecodeCancellationKind::Deadline, .. }, .. })),
                Delivery::CancelAtTerminalFlush => assert!(matches!(error, HostedError::Stopped {
                    stop: RunStop { kind: DecodeCancellationKind::Shutdown, .. }, .. })),
                Delivery::PanicAtTerminal => assert!(matches!(error, HostedError::Panicked)),
                Delivery::Complete => unreachable!(),
            }
            let observed = observed.lock().unwrap();
            assert_eq!(observed.dropped, 1);
            assert!(observed.job_bytes_at_drop >= input_charge + (128 << 20));
            let text = std::str::from_utf8(&observed.bytes).unwrap();
            if matches!(delivery, Delivery::CancelAtEof) {
                assert_eq!(observed.terminal_checks, 0); assert!(!text.contains("run_complete"));
                assert!(text.contains("run_error"));
            } else {
                assert_eq!(observed.terminal_checks, 1); assert!(!text.contains("run_error"));
                if matches!(delivery, Delivery::PanicAtTerminal) { assert!(!text.contains("run_complete")); }
            }
            drained(&host, &baseline);
        }
        for threads in [0, 4, 65] {
            let cancellation = CancellationToken::default();
            let (writer, observed) = writer(&host, threads, Delivery::Complete, &cancellation);
            let error = execute(&host, &model, &vocabulary, &source, task, threads, None,
                NoRead, writer, cancellation).unwrap_err();
            assert!(matches!(error, HostedError::Limits(_)));
            let observed = observed.lock().unwrap();
            assert_eq!(observed.writes, 0); assert_eq!(observed.dropped, 1);
            drained(&host, &baseline);
        }
    }
    drop(model); drop(source); drop(vocabulary); drop(preparation);
    drained(&host, &idle);
}
