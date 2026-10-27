//! Admission/accounting and transport tests construct no model or native
//! execution authority. The fixture processor exercises only the shared wire
//! protocol and the same nonrenewable ledger used by the resident adapter.
use super::*;
use std::{io::Cursor, sync::atomic::{AtomicUsize, Ordering}};
use crate::{batch::source::SourceMaskBudget, corpus::native_resolve::NativeResolveLimits,
    tasks::{ir::TaskBudget, source_planning::SourcePlanningLimits}};

fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn set_axis(work: &mut Int4Work, axis: usize, value: u64) {
    match axis {
        0 => work.forward_positions = value, 1 => work.projected_logits = value,
        2 => work.attention_pairs = value, 3 => work.projections.dot_products = value,
        4 => work.projections.multiply_accumulates = value, _ => unreachable!(),
    }
}
fn task() -> Int4EntityConfig {
    let budget = TaskBudget { max_input_tokens: 64, max_output_tokens: 16,
        max_output_bytes: 4096, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(128, 32).unwrap().kv_bytes };
    let mut source_planning = SourcePlanningLimits::default(); source_planning.max_context_tokens = 128;
    let mut planning = NativeResolveLimits::default(); planning.max_context_tokens = 128; planning.per_head = budget;
    Int4EntityConfig { ner: Default::default(), ner_budget: budget, source_planning,
        masks: SourceMaskBudget { per_mask: Default::default(), max_visits_per_item: 10, max_visits_per_run: 100 },
        resolution: Default::default(), graph: Default::default(),
        scoring: Int4ResolveLimits { planning, max_model_work: work(100) },
        verification: GroundingBudget { max_fields: 100, max_matches: 100, max_scan_steps: 100 },
        max_model_work: work(100), max_result_bytes: 4096, chunks: None, max_snapshot_chunks: 256 }
}
fn native() -> NativeLimits {
    NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
        run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100,
            cleanup_reserve_bytes: 4096 } }
}
fn transport() -> EntityStreamLimits {
    EntityStreamLimits { max_line_bytes: 1024, max_document_bytes: 512, max_corpus_bytes: 4096,
        max_documents: 16, max_input_bytes: 65_536, max_records: 100, max_snapshots: 10,
        max_output_line_bytes: 8192, max_output_bytes: 65_536 }
}
fn quote(task: &Int4EntityConfig, transport: EntityStreamLimits) -> Result<Int4EntityCorpusHostRequirements, HostedError> {
    Int4EntityCorpusHostRequirements::for_config(task, transport, native(), 4096, 8192, 4096, 32)
}
fn charge() -> Charge {
    Charge { reserved: work(30), actual: work(2), scoring: work(1), masks_reserved: 40, masks_actual: 1,
        verification: EntityVerificationWork { fields: 2, matches: 3, scan_steps: 4 } }
}
fn zero_charge() -> Charge {
    Charge { reserved: work(0), actual: work(0), scoring: work(0), masks_reserved: 0, masks_actual: 0,
        verification: EntityVerificationWork::default() }
}

#[test]
fn quote_includes_stream_storage_full_native_capacity_and_independent_terminal() {
    let task = task(); let t = transport(); let q = quote(&task, t).unwrap();
    let actual = Int4MemoryRequirement::for_context(128, 32).unwrap();
    assert_eq!(q.input_payload_bytes, 4 * t.max_corpus_bytes as u64 + 512 * t.max_documents as u64
        + 256 * task.max_snapshot_chunks as u64);
    assert_eq!(q.retained_bytes, q.input_payload_bytes + 32 * t.max_line_bytes as u64 + 4096 + 4096);
    assert_eq!(q.temporary_bytes, 8192 + 4 * task.ner_budget.max_output_bytes
        + 8 * u64::from(task.ner_budget.max_output_tokens));
    assert_eq!(q.output_bytes, 4 * t.max_output_line_bytes as u64 + 4096 + 4 * FOOTER_BYTES + 4096);
    assert_eq!(q.kv_bytes, actual.kv_bytes);
    assert_eq!(q.scratch_bytes, actual.rope_bytes + actual.scratch_payload_bound + 4096);
    assert_eq!(q.total_bytes().unwrap(), q.retained_bytes + q.temporary_bytes + q.output_bytes
        + q.kv_bytes + q.scratch_bytes + q.cleanup_bytes);
    let mut reduced_line = t; reduced_line.max_output_line_bytes = 128;
    assert_eq!(quote(&task, reduced_line).unwrap().output_bytes,
        4 * task.max_result_bytes as u64 + 4096 + 4 * FOOTER_BYTES + 4096);
}

#[test]
fn invalid_transport_policy_and_full_kv_are_rejected_by_pure_admission() {
    for axis in 0..16 {
        let mut task = task(); let mut t = transport();
        match axis {
            0 => t.max_line_bytes = 0,
            1 => t.max_line_bytes = 64 * 1024 * 1024 + 1,
            2 => t.max_input_bytes = 0,
            3 => t.max_records = 0,
            4 => t.max_snapshots = 0,
            5 => t.max_output_line_bytes = 1,
            6 => t.max_output_bytes = FOOTER_BYTES,
            7 => t.max_documents = task.graph.max_documents + 1,
            8 => t.max_corpus_bytes = task.graph.max_input_bytes + 1,
            9 => t.max_document_bytes = task.source_planning.max_input_bytes + 1,
            10 => task.ner_budget.max_kv_bytes -= 1,
            11 => task.scoring.planning.per_head.max_kv_bytes -= 1,
            12 => task.resolution.minimum_margin_milli = 0,
            13 => task.graph.max_cluster_checks = 0,
            14 => task.source_planning.max_passages = 0,
            _ => task.max_snapshot_chunks = 0,
        }
        assert!(quote(&task, t).is_err(), "axis {axis}");
    }
    for (prep, graph, io) in [(0, 8192, 4096), (4096, 0, 4096), (4096, 8192, 0),
        (u64::MAX, 8192, 4096), (4096, u64::MAX, 4096), (4096, 8192, u64::MAX)] {
        assert!(Int4EntityCorpusHostRequirements::for_config(&task(), transport(), native(), prep, graph, io, 32).is_err());
    }
    let mut q = quote(&task(), transport()).unwrap(); q.cleanup_bytes = u64::MAX;
    assert!(q.total_bytes().is_err());
    assert!(product(u64::MAX, 2).is_err());
}

#[test]
fn reserved_native_and_mask_capacity_is_never_refunded_after_early_eos_or_flush() {
    let mut ledger = Ledger::new(&task());
    ledger.begin().unwrap(); ledger.complete(charge()).unwrap();
    let remaining = ledger.remaining;
    assert_eq!(remaining.work, work(70));
    assert_eq!(remaining.scoring, work(99));
    assert_eq!(remaining.masks, 60);
    assert_eq!(remaining.verification, GroundingBudget { max_fields: 98, max_matches: 97, max_scan_steps: 96 });
    ledger.acknowledge().unwrap();
    assert_eq!(ledger.remaining, remaining);
    assert_eq!(ledger.summary.reserved_model_work, work(30));
    assert_eq!(ledger.summary.model_work, work(2));
    ledger.begin().unwrap(); ledger.complete(charge()).unwrap(); ledger.acknowledge().unwrap();
    assert_eq!(ledger.remaining.work, work(40));
    assert_eq!(ledger.remaining.masks, 20);
    assert_eq!(ledger.summary.reserved_model_work, work(60));
    assert_eq!(ledger.summary.model_work, work(4));
    let before = ledger.remaining;
    ledger.begin().unwrap();
    assert!(ledger.complete(charge()).is_err());
    assert_eq!(ledger.remaining, before);
    assert_eq!(ledger.state, State::Failed);
    assert!(ledger.begin().is_err());
}

#[test]
fn every_cumulative_axis_is_atomic_and_an_exceeded_axis_permanently_closes_the_run() {
    for axis in 0..14 {
        let mut ledger = Ledger::new(&task()); let before = ledger.remaining;
        let mut c = zero_charge();
        match axis {
            0..=4 => set_axis(&mut c.reserved, axis, 101),
            5..=9 => set_axis(&mut c.scoring, axis - 5, 101),
            10 => c.masks_reserved = 101,
            11 => c.verification.fields = 101,
            12 => c.verification.matches = 101,
            _ => c.verification.scan_steps = 101,
        }
        ledger.begin().unwrap();
        assert!(ledger.complete(c).is_err(), "axis {axis}");
        assert_eq!(ledger.remaining, before, "axis {axis}");
        assert_eq!(ledger.summary, Int4EntityCorpusSummary::default(), "axis {axis}");
        assert_eq!(ledger.state, State::Failed);
        assert!(ledger.acknowledge().is_err());
        assert!(ledger.begin().is_err());
    }
}

#[test]
fn actual_work_cannot_exceed_a_reserved_axis_and_delivery_cannot_be_replayed() {
    for axis in 0..6 {
        let mut ledger = Ledger::new(&task()); let mut c = charge();
        if axis < 5 { set_axis(&mut c.actual, axis, 31); } else { c.masks_actual = 41; }
        ledger.begin().unwrap(); assert!(ledger.complete(c).is_err());
        assert_eq!(ledger.state, State::Failed);
    }
    for phase in 0..3 {
        let mut ledger = Ledger::new(&task());
        if phase > 0 { ledger.begin().unwrap(); }
        if phase > 1 { ledger.complete(charge()).unwrap(); }
        ledger.abandon();
        assert!(ledger.begin().is_err()); assert!(ledger.acknowledge().is_err());
    }
    let mut ledger = Ledger::new(&task());
    ledger.begin().unwrap(); ledger.complete(charge()).unwrap();
    assert!(ledger.begin().is_err());
    assert!(ledger.acknowledge().is_err());
}

#[test]
fn empty_flush_does_not_refill_exhausted_allowances_or_skip_nonempty_ner() {
    let mut ledger = Ledger::new(&task());
    let mut all = zero_charge(); all.reserved = work(100); all.scoring = work(100); all.masks_reserved = 100;
    all.verification = EntityVerificationWork { fields: 100, matches: 100, scan_steps: 100 };
    ledger.begin().unwrap(); ledger.complete(all).unwrap(); ledger.acknowledge().unwrap();
    let remaining = ledger.begin().unwrap();
    assert!(narrowed_task(&task(), remaining, false).is_err());
    let empty = narrowed_task(&task(), remaining, true).unwrap(); empty.validate().unwrap();
    assert_eq!(empty.max_model_work, work(0)); assert_eq!(empty.masks.max_visits_per_run, 0);
    assert_eq!(empty.verification, GroundingBudget { max_fields: 1, max_matches: 1, max_scan_steps: 1 });
    ledger.complete(zero_charge()).unwrap(); ledger.acknowledge().unwrap();
    assert_eq!(ledger.remaining, remaining);
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct OutputGuard(Arc<AtomicUsize>);
impl Drop for OutputGuard { fn drop(&mut self) { self.0.fetch_sub(1, Ordering::SeqCst); } }
#[derive(Serialize)]
struct FixtureSnapshot { fixture_only: bool, document_count: usize, cluster_scope: &'static str }
struct FixtureProcessor {
    ledger: Ledger, calls: usize, acknowledged: usize, abandoned: usize,
    ids: Vec<Vec<String>>, live: Arc<AtomicUsize>,
}
impl FixtureProcessor {
    fn new(live: Arc<AtomicUsize>) -> Self {
        Self { ledger: Ledger::new(&task()), calls: 0, acknowledged: 0, abandoned: 0, ids: Vec::new(), live }
    }
}
impl EntitySnapshotProcessor for FixtureProcessor {
    type Output = GuardedOutput<FixtureSnapshot, OutputGuard>;
    type Error = HostedError;
    fn execute_snapshot<C: DecodeStepControl>(&mut self, documents: Vec<EntityDocument>, _: &mut C)
        -> Result<Self::Output, Self::Error> {
        self.calls += 1;
        self.ledger.begin().map_err(entities::execution_error)?;
        let count = documents.len();
        self.ids.push(documents.into_iter().map(|d| d.id).collect());
        self.ledger.complete(if count == 0 { zero_charge() } else { charge() }).map_err(entities::execution_error)?;
        self.live.fetch_add(1, Ordering::SeqCst);
        Ok(GuardedOutput::new(FixtureSnapshot { fixture_only: true, document_count: count, cluster_scope: "epoch_local" },
            OutputGuard(Arc::clone(&self.live))))
    }
    fn acknowledge_snapshot(&mut self) -> Result<(), Self::Error> {
        assert_eq!(self.live.load(Ordering::SeqCst), 1);
        self.ledger.acknowledge().map_err(entities::execution_error)?;
        self.acknowledged += 1;
        Ok(())
    }
    fn abandon_snapshot(&mut self) { self.abandoned += 1; self.ledger.abandon(); }
}
struct Writer {
    bytes: Vec<u8>, live: Arc<AtomicUsize>, writes: usize, flushes: usize,
    fail_after: Option<usize>, fail_flush: bool,
}
impl Writer {
    fn new(live: Arc<AtomicUsize>) -> Self {
        Self { bytes: Vec::new(), live, writes: 0, flushes: 0, fail_after: None, fail_flush: false }
    }
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        assert_eq!(self.live.load(Ordering::SeqCst), 1);
        self.writes += 1;
        let count = self.fail_after.map_or(bytes.len(), |cap| cap.saturating_sub(self.bytes.len()).min(bytes.len()));
        if count == 0 { return Err(std::io::Error::other("private writer fixture")); }
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        assert_eq!(self.live.load(Ordering::SeqCst), 1);
        self.flushes += 1;
        if self.fail_flush { Err(std::io::Error::other("private flush fixture")) } else { Ok(()) }
    }
}
fn events(bytes: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(bytes).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect()
}

#[test]
fn shared_protocol_preserves_empty_flush_eof_and_epoch_local_ids_with_guards() {
    let bytes = b"\n{\"id\":\"a\",\"text\":\"Alpha\"}\r\n{\"flush\":true}\n{\"flush\":true}\n{\"id\":\"a\",\"text\":\"Again\"}";
    let live = Arc::new(AtomicUsize::new(0));
    let mut processor = FixtureProcessor::new(Arc::clone(&live)); let mut writer = Writer::new(Arc::clone(&live));
    let summary = entity_stream::run_ndjson(&mut Cursor::new(bytes), &mut writer, &mut processor, transport(), &mut Continue).unwrap();
    assert_eq!(summary.records, 4); assert_eq!(summary.snapshots, 3); assert_eq!(summary.documents, 2);
    assert_eq!(summary.input_bytes, bytes.len() as u64); assert_eq!(summary.output_bytes, writer.bytes.len() as u64);
    assert_eq!(processor.calls, 3); assert_eq!(processor.acknowledged, 3); assert_eq!(processor.abandoned, 0);
    assert_eq!(processor.ids, vec![vec!["a".to_owned()], Vec::<String>::new(), vec!["a".to_owned()]]);
    assert_eq!(processor.ledger.summary.reserved_model_work, work(60));
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let frames = events(&writer.bytes);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame["protocol"], ENTITY_STREAM_PROTOCOL); assert_eq!(frame["event"], "corpus");
        assert_eq!(frame["epoch"], index + 1); assert_eq!(frame["result"]["cluster_scope"], "epoch_local");
    }
    assert_eq!(frames[0]["through_request_seq"], 2); assert_eq!(frames[1]["through_request_seq"], 3);
    assert_eq!(frames[2]["through_request_seq"], 4);
    assert_eq!(frames[0]["eof"], false); assert_eq!(frames[1]["eof"], false); assert_eq!(frames[2]["eof"], true);
    assert_eq!(frames[1]["result"]["document_count"], 0);
}

#[test]
fn eof_after_flush_and_empty_input_do_not_invent_an_extra_snapshot() {
    for (bytes, epochs, records) in [(b"".as_slice(), 0, 0), (b"{\"flush\":true}\n".as_slice(), 1, 1)] {
        let live = Arc::new(AtomicUsize::new(0)); let mut processor = FixtureProcessor::new(Arc::clone(&live));
        let mut writer = Writer::new(Arc::clone(&live));
        let summary = entity_stream::run_ndjson(&mut Cursor::new(bytes), &mut writer, &mut processor, transport(), &mut Continue).unwrap();
        assert_eq!(summary.snapshots, epochs); assert_eq!(summary.records, records); assert_eq!(summary.documents, 0);
        assert_eq!(processor.calls as u64, epochs);
        assert_eq!(processor.ledger.summary.reserved_model_work, work(0));
    }
}

#[test]
fn malformed_or_duplicate_late_record_cannot_publish_a_pending_epoch() {
    for invalid in ["{\"id\":\"a\",\"id\":\"b\",\"text\":\"x\"}", "{\"id\":\"a\",\"text\":\"duplicate\"}",
        "{\"flush\":true,\"extra\":1}", "{\"id\":\"b\",\"text\":\"x\",\"mentions\":[]}"] {
        let bytes = format!("{{\"id\":\"a\",\"text\":\"first\"}}\n{invalid}\n");
        let live = Arc::new(AtomicUsize::new(0)); let mut processor = FixtureProcessor::new(Arc::clone(&live));
        let mut writer = Writer::new(live);
        assert!(entity_stream::run_ndjson(&mut Cursor::new(bytes), &mut writer, &mut processor, transport(), &mut Continue).is_err());
        assert_eq!(processor.calls, 0); assert!(writer.bytes.is_empty());
    }
    let bytes = b"{\"id\":\"a\",\"text\":\"first\"}\n{\"flush\":true}\n{\"id\":\"b\",\"text\":\"pending\"}\nnot-json\n";
    let live = Arc::new(AtomicUsize::new(0)); let mut processor = FixtureProcessor::new(Arc::clone(&live));
    let mut writer = Writer::new(live);
    assert!(entity_stream::run_ndjson(&mut Cursor::new(bytes), &mut writer, &mut processor, transport(), &mut Continue).is_err());
    assert_eq!(processor.calls, 1); assert_eq!(processor.acknowledged, 1);
    assert_eq!(events(&writer.bytes).len(), 1);
    assert!(!String::from_utf8(writer.bytes).unwrap().contains("run_complete"));
}

#[test]
fn partial_write_flush_and_complete_output_cap_failure_abandon_without_retry_or_next_epoch() {
    let bytes = b"{\"id\":\"a\",\"text\":\"first\"}\n{\"flush\":true}\n{\"id\":\"b\",\"text\":\"second\"}\n";
    for failure in 0..3 {
        let live = Arc::new(AtomicUsize::new(0)); let mut processor = FixtureProcessor::new(Arc::clone(&live));
        let mut writer = Writer::new(Arc::clone(&live)); let mut limits = transport();
        match failure { 0 => writer.fail_after = Some(7), 1 => writer.fail_flush = true,
            _ => limits.max_output_bytes = 1 }
        let mut reader = Cursor::new(bytes);
        let result = entity_stream::run_ndjson(&mut reader, &mut writer, &mut processor, limits, &mut Continue);
        if failure < 2 { assert!(matches!(result, Err(EntityStreamError::OutputIo))); }
        else { assert!(matches!(result, Err(EntityStreamError::OutputLimit))); assert!(writer.bytes.is_empty()); }
        assert_eq!(processor.calls, 1); assert_eq!(processor.acknowledged, 0); assert_eq!(processor.abandoned, 1);
        assert_eq!(processor.ledger.state, State::Failed); assert!(processor.ledger.begin().is_err());
        assert_eq!(live.load(Ordering::SeqCst), 0); assert!(reader.position() < bytes.len() as u64);
        if failure == 0 { assert_eq!(writer.bytes.len(), 7); assert_eq!(writer.writes, 2); assert_eq!(writer.flushes, 0); }
        if failure == 1 { assert_eq!(writer.writes, 1); assert_eq!(writer.flushes, 1); }
    }
}

fn provenance() -> Provenance {
    Provenance { scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve-corpus",
        model_id: "fixture-only", source_revision: "fixture-only", source_root_sha256: "00".repeat(32),
        model_content_sha256: "01".repeat(32), candidate_format: "fixture-only".to_owned(),
        quant_recipe: "fixture-only".to_owned(), group_width: 32, integer_kernel: "fixture-only", cluster_scope: "epoch_local" }
}
#[test]
fn footer_has_final_record_sequence_and_never_claims_a_global_snapshot() {
    let summary = Int4EntityCorpusSummary { transport: EntityStreamSummary {
        records: 4, snapshots: 2, documents: 2, input_bytes: 100, output_bytes: 999 }, ..Default::default() };
    let bytes = footer_bytes(&provenance(), &summary, summary.transport.output_bytes + FOOTER_BYTES).unwrap();
    let frames = events(&bytes); assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    assert_eq!(frame["protocol"], ENTITY_STREAM_PROTOCOL); assert_eq!(frame["event"], "run_complete");
    assert_eq!(frame["epoch"], 2); assert_eq!(frame["through_request_seq"], 4); assert_eq!(frame["eof"], true);
    assert_eq!(frame["provenance"]["cluster_scope"], "epoch_local");
    assert_eq!(frame["provenance"]["evidence"], "non_authoritative");
    assert_eq!(frame["summary"]["transport"]["output_bytes"], 999);
    assert!(frame.get("result").is_none());
    let exact = summary.transport.output_bytes + bytes.len() as u64;
    assert!(footer_bytes(&provenance(), &summary, exact).is_ok());
    assert!(footer_bytes(&provenance(), &summary, exact - 1).is_err());
    let empty = footer_bytes(&provenance(), &Int4EntityCorpusSummary::default(), FOOTER_BYTES).unwrap();
    assert_eq!(events(&empty)[0]["epoch"], 0);
    let mut excessive = provenance(); excessive.source_root_sha256 = "x".repeat(FOOTER_BYTES as usize);
    assert!(footer_bytes(&excessive, &summary, u64::MAX).is_err());
}

#[test]
fn all_cancellation_causes_survive_native_transport_and_post_drain_terminal_checks() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let entity = || entities::execution_error(Int4EntityError::Graph(crate::corpus::resolve::ResolveError::Cancelled(cause)));
        for error in [Int4EntityCorpusError::Host(entity()), Int4EntityCorpusError::Transport(EntityStreamError::Handler(entity())),
            Int4EntityCorpusError::Transport(EntityStreamError::Resolution(crate::corpus::resolve::ResolveError::Cancelled(cause))),
            Int4EntityCorpusError::Host(native_error(StrictInt4Error::Cancelled(cause)))] {
            assert_eq!(error.cancellation(), Some(cause));
        }
        let token = CancellationToken::default(); token.cancel(cause);
        assert_eq!(terminal_checkpoint(Instant::now() + Duration::from_secs(1), &token).unwrap_err().cancellation(), Some(cause));
    }
    assert_eq!(terminal_checkpoint(Instant::now(), &CancellationToken::default()).unwrap_err().cancellation(), Some(Deadline));
    assert!(terminal_checkpoint(Instant::now() + Duration::from_secs(1), &CancellationToken::default()).is_ok());
    let scrubbed = Int4EntityCorpusError::Host(HostedError::Limits("private fixture diagnostic"));
    assert!(!format!("{scrubbed:?}").contains("private fixture"));
}

#[test]
fn owned_completion_and_errors_cross_the_physical_handoff() {
    fn send<T: Send + 'static>() {}
    send::<HostedInt4EntityCorpus<Vec<u8>>>();
    send::<Int4EntityCorpusError>();
    send::<Input<Cursor<Vec<u8>>, Vec<u8>>>();
}
