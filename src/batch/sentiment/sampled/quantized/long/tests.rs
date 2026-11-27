//! Model-free transport, lifetime, address and nonrenewable debit fixtures.
//! Synthetic drivers exercise the private adapter boundary, never model quality.
use super::*;
use std::{cell::Cell, io::{self, Cursor, Write}, rc::Rc, sync::OnceLock};
use serde::Serialize;
use crate::{batch::{BatchLimits, run_ndjson}, native_engine::decode::DecodeCancellationKind,
    grammar::mask::MaskWorkLimits, tasks::mapreduce::ChunkLimits,
    tokenizer::untrusted::SafeBpeLimits};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".into(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".into(),
        recipe_id: "nanbeige42-int8-v1".into(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32) }
}
fn options() -> SampledSentimentOptions {
    SampledSentimentOptions { effective_seed: [31; 32], samples: 2, temperature_milli: 1000,
        top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 128, max_justification_chars: 64 }
}
fn real_task() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 1024, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn vocabulary() -> Arc<SampledSentimentVocabulary> {
    static VOCABULARY: OnceLock<Arc<SampledSentimentVocabulary>> = OnceLock::new();
    Arc::clone(VOCABULARY.get_or_init(|| Arc::new(SampledSentimentVocabulary::pinned().unwrap())))
}
fn mapping() -> Int8SampledSentimentMapLimits {
    Int8SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 1024,
        max_chunk_bytes: 8, max_chunk_tokens: 4096, context_tokens: 4096,
        reserved_tokens: 127, max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX,
                multiply_accumulates: u64::MAX } }, max_retained_result_bytes: 4 << 20,
        max_result_bytes: 8 << 20 }
}
fn compiler(encoding: UntrustedEncoding) -> Int8SampledSentimentDocumentPlanner {
    let planner = Int8SampledSentimentPlanner::for_model_with_encoding(&facts(), options(), real_task(),
        SampledSentimentLimits::default(), encoding).unwrap();
    let mapping = mapping();
    Int8SampledSentimentDocumentPlanner::new(planner, vocabulary(), real_task(), mapping,
        Int8SampledSentimentBatchLimits { max_item_work: mapping.max_model_work,
            max_model_work: mapping.max_model_work, ..limits(1) }).unwrap()
}
fn document(id: &str, text: &str) -> BatchDocument<SampledSentimentBatchArgs> {
    BatchDocument { id: id.into(), text: text.into(), task_args: None }
}

#[test]
fn actual_plans_bind_item_and_complete_source_without_order_or_flush_coordinates() {
    let planner = compiler(UntrustedEncoding::ByteFallback);
    let plan = |id| planner.prepare(document(id, "abcdefghabcdefgh"), &mut Continue, &mut SafeBpeWork::default()).unwrap();
    let a = plan("a"); let b = plan("b"); let replay = plan("a");
    assert_eq!(a.chunk_count(), 2);
    assert_eq!(a.execution_identity(), replay.execution_identity());
    assert_eq!(a.planned_work(), replay.planned_work());
    assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
    let changed = planner.prepare(document("a", "abcdefghabcdefgi"), &mut Continue,
        &mut SafeBpeWork::default()).unwrap();
    assert_ne!(a.execution_identity(), changed.execution_identity());
    let mut reversed = document("a", "abcdefghabcdefgh");
    reversed.task_args = Some(SampledSentimentBatchArgs {
        axes: SentimentAxis::ALL.into_iter().rev().collect(), budget: None });
    let reversed = planner.prepare(reversed, &mut Continue, &mut SafeBpeWork::default()).unwrap();
    assert_eq!(a.execution_identity(), reversed.execution_identity());
    assert_eq!(a.planned_work(), reversed.planned_work());
    let (owner, vocabulary) = planner.retained_owners();
    assert!(Arc::ptr_eq(&owner, &planner.planner));
    assert!(Arc::ptr_eq(&vocabulary, &planner.vocabulary));
    assert!(planner.preparation_requirement_bytes() >= a.preparation_requirement_bytes());
}

#[test]
fn rejected_preparation_retains_the_same_compact_encoding_work_and_policy() {
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096,
        max_output_tokens: 4096, max_pair_checks: 1_000_000, max_pair_bytes: 64 << 20 });
    let planner = compiler(policy.clone());
    let mut spent = SafeBpeWork::default();
    let first = planner.prepare(document("a", "testing testing testing"), &mut Continue, &mut spent).unwrap();
    let before = spent;
    let second = planner.prepare(document("a", "testing testing testing"), &mut Continue, &mut spent).unwrap();
    assert_eq!(first.execution_identity(), second.execution_identity());
    assert_eq!(spent, before.checked_add(before).unwrap());
    while spent.input_bytes() < 4096 { spent = spent.checked_add(spent).unwrap(); }
    let exhausted = spent;
    assert!(planner.prepare(document("next", "testing"), &mut Continue, &mut spent).is_err());
    assert_eq!(spent, exhausted);
    assert_eq!(planner.document_encoding(), &policy);
}

#[test]
fn preparation_and_job_quotes_price_one_shared_complete_document_and_private_recipe() {
    let planner = compiler(UntrustedEncoding::ByteFallback);
    let bytes = int8_sampled_sentiment_document_preparation_bytes(real_task(), planner.planning_limits(),
        mapping(), &options(), &UntrustedEncoding::ByteFallback).unwrap();
    assert_eq!(planner.preparation_requirement_bytes(), bytes);
    let job = int8_sampled_sentiment_document_job_preparation_bytes(real_task(), planner.planning_limits(),
        mapping(), &options(), &UntrustedEncoding::ByteFallback).unwrap();
    assert_eq!(job, bytes + super::super::super::SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64);
    let mut bad = options(); bad.samples = 0;
    assert!(int8_sampled_sentiment_document_preparation_bytes(real_task(), planner.planning_limits(),
        mapping(), &bad, &UntrustedEncoding::ByteFallback).is_err());
}

fn work(n: u64) -> Int8Work {
    Int8Work { forward_positions: 40 * n, projected_logits: 50 * n, attention_pairs: 60 * n,
        projections: ProjectionWork { dot_products: 70 * n, multiply_accumulates: 80 * n } }
}
fn limits(n: u64) -> Int8SampledSentimentBatchLimits {
    Int8SampledSentimentBatchLimits { max_item_work: work(1), max_model_work: work(n),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 100,
            checkpoint_interval_nodes: 8 }, max_visits_per_item: 10, max_visits_per_run: 10 * n },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 32, max_output_tokens: 16, max_output_bytes: 8192,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(32).unwrap().kv_bytes }
}
fn identity() -> ExecutionIdentity {
    static IDENTITY: OnceLock<ExecutionIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| compiler(UntrustedEncoding::ByteFallback).base_execution_identity().clone()).clone()
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
#[derive(Clone, Copy)]
enum Mode { Success, Refuse, Incomplete, Dirty, Panic }
struct Plan { identity: ExecutionIdentity, mode: Mode }
#[derive(Serialize)]
struct Output { chunks: usize, complete: bool, model_work: Int8Work }
impl DocumentPlan for Plan {
    type Output = Output;
    fn facts(&self) -> Int8SampledSentimentDocumentPlanFacts<'_> {
        Int8SampledSentimentDocumentPlanFacts { identity: &self.identity, model_work: work(1),
            sample_count: 2, chunk_count: 2, required_context: 16, original_task: task(),
            max_result_bytes: 16_384, max_output_tokens: 32, preparation_bytes: 1024 }
    }
    fn verify_identity(&self, identity: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
        if identity == &self.identity { Ok(()) } else { Err(BatchItemFailure::fatal(BatchCode::Admission)) }
    }
    fn verify_completed<C: DecodeStepControl>(&self, output: &Output, masks: u64, control: &mut C)
        -> Result<(), BatchItemFailure> {
        checkpoint(control)?;
        if !output.complete || output.chunks != 2 || !fits(output.model_work, work(1)) || masks != 10 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(())
    }
}
#[derive(Default)]
struct Probe { live: Cell<bool>, running: Cell<bool>, calls: Cell<usize>, drops: Cell<usize> }
struct Guard(Rc<Probe>);
impl Drop for Guard {
    fn drop(&mut self) {
        assert!(!self.0.running.get()); assert!(self.0.live.replace(false));
        self.0.drops.set(self.0.drops.get() + 1);
    }
}
struct Physical(Rc<Probe>);
impl Drop for Physical { fn drop(&mut self) { self.0.running.set(false); } }
struct Admission { probe: Rc<Probe>, refuse: bool, wrong: bool }
impl Int8SampledSentimentDocumentBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int8SampledSentimentDocumentAdmission<'_>)
        -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        assert_eq!(request.facts.chunk_count, 2);
        assert_eq!(request.kv_reservation_bytes, task().max_kv_bytes);
        assert_eq!(request.mask_node_visits, 10);
        assert!(request.facts.max_result_bytes > request.facts.original_task.max_output_bytes);
        if self.refuse { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert!(!self.probe.live.replace(true));
        let mut identity = request.facts.identity.clone();
        if self.wrong { identity.prompt_digest = Sha256Digest::of_bytes(b"wrong"); }
        Ok((identity, Guard(Rc::clone(&self.probe))))
    }
}
struct Driver { probe: Rc<Probe>, clean: bool, actual: Int8Work }
impl DocumentDriver<Plan> for Driver {
    fn clean(&self) -> bool { self.clean }
    fn capacity(&self) -> usize { 32 }
    fn mask_charge(&self, _: &Plan, budget: Int8JsonBudget, _: u64) -> Result<u64, BatchItemFailure> {
        if budget.json.max_total_mask_node_visits < 10 { Err(BatchItemFailure::reject(BatchCode::WorkLimit)) }
        else { Ok(10) }
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &Plan, budget: Int8JsonBudget,
        _: u64, _: &mut C) -> Result<Output, BatchItemFailure> {
        assert_eq!(budget.native.max_forward_positions, work(1).forward_positions);
        assert_eq!(budget.native.max_projection_work, work(1).projections);
        assert_eq!(budget.json.max_total_mask_node_visits, 10);
        assert!(self.probe.live.get()); assert!(!self.probe.running.replace(true));
        let _physical = Physical(Rc::clone(&self.probe));
        self.probe.calls.set(self.probe.calls.get() + 1);
        match plan.mode {
            Mode::Refuse => return Err(BatchItemFailure::reject(BatchCode::Execution)),
            Mode::Dirty => self.clean = false,
            Mode::Panic => panic!("fixture native failure"),
            _ => (),
        }
        Ok(Output { chunks: if matches!(plan.mode, Mode::Incomplete) { 1 } else { 2 },
            complete: true, model_work: self.actual })
    }
}
fn host(probe: &Rc<Probe>, total: u64) -> (Driver, Admission, RunState) {
    (Driver { probe: Rc::clone(probe), clean: true, actual: work(1) },
        Admission { probe: Rc::clone(probe), refuse: false, wrong: false }, RunState::new(limits(total)))
}
fn prepared(mode: Mode, seal: &Arc<()>, driver: &Driver) -> PreparedRecord<Plan> {
    prepare_document(Plan { identity: identity(), mode }, seal, driver, limits(1), 1024, &mut Continue).unwrap()
}

#[test]
fn admission_and_failed_chunks_spend_all_six_counters_without_partial_success() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut run) = host(&probe, 3);
    admission.refuse = true;
    assert!(execute_document(prepared(Mode::Success, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(1, 1), &mut Continue).is_err());
    assert_eq!(run.remaining, work(2)); assert_eq!(run.remaining_masks, 20); assert_eq!(probe.calls.get(), 0);
    admission.refuse = false;
    assert!(execute_document(prepared(Mode::Refuse, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(2, 2), &mut Continue).is_err());
    assert_eq!(run.remaining, work(1)); assert_eq!(run.remaining_masks, 10); assert_eq!(probe.calls.get(), 1);
    driver.actual = work(0);
    let output = execute_document(prepared(Mode::Success, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(3, 99), &mut Continue).unwrap();
    assert!(probe.live.get()); assert_eq!(run.remaining, work(0)); assert_eq!(run.remaining_masks, 0);
    drop(output); assert_eq!(probe.drops.get(), 2);
}
#[test]
fn each_short_counter_refuses_atomically_before_admission_or_native_work() {
    for axis in 0..6 {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 1);
        match axis { 0 => run.remaining.forward_positions -= 1,
            1 => run.remaining.projected_logits -= 1, 2 => run.remaining.attention_pairs -= 1,
            3 => run.remaining.projections.dot_products -= 1,
            4 => run.remaining.projections.multiply_accumulates -= 1,
            _ => run.remaining_masks -= 1 }
        let before = (run.remaining, run.remaining_masks);
        assert!(execute_document(prepared(Mode::Success, &seal, &driver), &seal, &mut driver,
            &mut admission, &mut run, context(1, 1), &mut Continue).is_err());
        assert_eq!((run.remaining, run.remaining_masks), before); assert_eq!(probe.calls.get(), 0);
        assert!(!probe.live.get());
    }
}
#[test]
fn foreign_factories_wrong_identity_incomplete_and_dirty_results_close_the_adapter() {
    for fault in 0..4 {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 2);
        admission.wrong = fault == 1;
        let expected = if fault == 0 { Arc::new(()) } else { Arc::clone(&seal) };
        let mode = if fault == 2 { Mode::Incomplete } else if fault == 3 { Mode::Dirty } else { Mode::Success };
        let error = execute_document(prepared(mode, &seal, &driver), &expected, &mut driver,
            &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert!(run.enter().is_err()); assert!(!probe.live.get()); assert!(!probe.running.get());
    }
}
#[test]
fn unwind_keeps_the_debit_and_latches_failure_after_physical_cleanup() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut run) = host(&probe, 2);
    let prepared = prepared(Mode::Panic, &seal, &driver);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = execute_document(prepared, &seal, &mut driver, &mut admission, &mut run,
            context(1, 1), &mut Continue);
    }));
    assert!(result.is_err()); assert_eq!(run.remaining, work(1)); assert_eq!(run.remaining_masks, 10);
    assert!(run.enter().is_err()); assert!(!probe.running.get()); assert!(!probe.live.get());
}
struct Stop { calls: usize, at: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.at).then_some(self.cause)
    }
}
#[test]
fn every_terminal_cause_is_preserved_before_document_publication() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 2);
        let prepared = prepared(Mode::Success, &seal, &driver);
        let error = execute_document(prepared, &seal, &mut driver, &mut admission, &mut run,
            context(1, 1), &mut Stop { calls: 0, at: 4, cause }).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
        assert_eq!(probe.calls.get(), 1); assert_eq!(run.remaining, work(1));
        assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
    }
}

struct Harness { driver: Driver, admission: Admission, run: RunState, seal: Arc<()> }
impl BatchProcessor for Harness {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedRecord<Plan>;
    type Output = GuardedOutput<Output, Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            item_budget(document.task_args.unwrap_or_default().budget, task())?;
            let plan = Plan { identity: identity(),
                mode: if document.text == "refuse" { Mode::Refuse } else { Mode::Success } };
            let record = prepare_document(plan, &self.seal, &self.driver, self.run.limits, 1024, control)?;
            self.run.check_available(record.plan.facts().model_work, record.mask_node_visits)?;
            Ok(record)
        })();
        settle(result, self.driver.clean, &mut self.run)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { transport_work(prepared.plan.facts().model_work) }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_document(prepared, &self.seal, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}
struct Sink { probe: Rc<Probe>, bytes: Vec<u8>, delivering: bool, fail: bool }
impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.windows(b"\"result\":".len()).any(|part| part == b"\"result\":") {
            assert!(self.probe.live.get()); assert!(!self.probe.running.get()); self.delivering = true;
            if self.fail { return Err(io::Error::other("fixture sink failure")); }
        }
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.delivering { assert!(self.probe.live.get()); self.delivering = false; } Ok(())
    }
}
#[test]
fn real_ndjson_flush_and_failed_records_do_not_renew_native_or_mask_allowances() {
    let probe = Rc::new(Probe::default()); let (driver, admission, run) = host(&probe, 3);
    let mut harness = Harness { driver, admission, run, seal: Arc::new(()) };
    let mut source = Cursor::new(b"{\"id\":\"same\",\"text\":\"ok\"}\n{\"id\":\"same\",\"text\":\"duplicate\"}\n{\"flush\":true}\n{\"id\":\"same\",\"text\":\"refuse\"}\n{\"id\":\"third\",\"text\":\"ok\"}\n{\"id\":\"fourth\",\"text\":\"over\"}\n");
    let mut sink = Sink { probe: Rc::clone(&probe), bytes: Vec::new(), delivering: false, fail: false };
    let transport = BatchLimits { max_work: transport_work(work(8)), ..BatchLimits::default() };
    let summary = run_ndjson(&mut source, &mut sink, &mut harness, transport, &mut Continue).unwrap();
    assert_eq!((summary.succeeded, summary.failed, summary.flushes), (2, 3, 1));
    assert_eq!(summary.reserved_work, transport_work(work(3)));
    assert_eq!(harness.run.remaining, work(0)); assert_eq!(harness.run.remaining_masks, 0);
    assert_eq!(probe.calls.get(), 3); assert_eq!(probe.drops.get(), 3);
    let rows: Vec<serde_json::Value> = std::str::from_utf8(&sink.bytes).unwrap().lines()
        .map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(rows.last().unwrap()["event"], "run_complete");
    assert_eq!(rows.iter().filter(|row| row["event"] == "doc").count(), 2);
    assert!(rows.iter().any(|row| row["error"]["code"] == "duplicate_id"));
    assert!(rows.iter().any(|row| row["error"]["code"] == "work_limit"));
}
#[test]
fn failed_delivery_holds_guard_through_write_and_prevents_next_document() {
    let probe = Rc::new(Probe::default()); let (driver, admission, run) = host(&probe, 2);
    let mut harness = Harness { driver, admission, run, seal: Arc::new(()) };
    let mut source = Cursor::new(b"{\"id\":\"a\",\"text\":\"ok\"}\n{\"id\":\"b\",\"text\":\"ok\"}\n");
    let mut sink = Sink { probe: Rc::clone(&probe), bytes: Vec::new(), delivering: false, fail: true };
    let transport = BatchLimits { max_work: transport_work(work(8)), ..BatchLimits::default() };
    let error = run_ndjson(&mut source, &mut sink, &mut harness, transport, &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::OutputIo); assert_eq!(probe.calls.get(), 1);
    assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
    assert_eq!(harness.run.remaining, work(1));
    assert!(!std::str::from_utf8(&sink.bytes).unwrap().contains("run_complete"));
}
