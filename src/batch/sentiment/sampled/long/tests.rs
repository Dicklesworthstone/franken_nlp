//! Source-level regression definitions for complete-document transport and
//! physical ownership. Synthetic drivers exercise the real adapter protocol;
//! they are not model execution, sampled-quality or DSR evidence.
use super::*;
use std::{cell::Cell, io::{self, Cursor, Write}, rc::Rc};
use serde::Serialize;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchLimits, run_ndjson},
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::mapreduce::ChunkLimits,
};

fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * n, projected_logits: 50 * n, attention_pairs: 60 * n,
        projections: ProjectionWork { dot_products: 70 * n, multiply_accumulates: 80 * n } }
}
fn limits(n: u64) -> SampledSentimentBatchLimits {
    SampledSentimentBatchLimits { max_item_work: work(1), max_model_work: work(n),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 100,
            checkpoint_interval_nodes: 8 }, max_visits_per_item: 10, max_visits_per_run: 10 * n },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 32, max_output_tokens: 16, max_output_bytes: 8192,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(32, 32).unwrap().kv_bytes }
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"sampled-document-protocol-fixture-not-model-authority");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: digest,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: digest, tokenizer_digest: digest, template_digest: digest,
        task_spec: "sentiment-v1".to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "fixture".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop { calls: usize, at: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.at).then_some(self.cause)
    }
}
#[derive(Clone, Copy)]
enum Mode { Success, RejectedChunk, Dirty, WrongBinding, Panic }
struct FixturePlan { identity: ExecutionIdentity, mode: Mode, context: usize }
#[derive(Serialize)]
struct FixtureOutput { chunks: usize, complete: bool, reuse_prefill: bool, work: Int4Work }
impl DocumentPlan for FixturePlan {
    type Output = FixtureOutput;
    fn facts(&self) -> SampledSentimentDocumentPlanFacts<'_> {
        SampledSentimentDocumentPlanFacts { identity: &self.identity, model_work: work(1), group_width: 32,
            sample_count: 2, chunk_count: 2, required_context: self.context, original_task: task(),
            max_result_bytes: 16_384, max_output_tokens: 32, preparation_bytes: 1024 }
    }
    fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
        if admitted == &self.identity { Ok(()) } else { Err(BatchItemFailure::fatal(BatchCode::Admission)) }
    }
    fn verify_completed<C: DecodeStepControl>(&self, output: &FixtureOutput, masks: u64,
        reuse_prefill: bool, control: &mut C) -> Result<(), BatchItemFailure> {
        checkpoint(control)?;
        if !output.complete || output.chunks != 2 || output.reuse_prefill != reuse_prefill
            || !fits(output.work, work(1)) || masks != 10 {
            return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
        }
        Ok(())
    }
}
#[derive(Default)]
struct Probe { live: Cell<bool>, running: Cell<bool>, calls: Cell<usize>, admissions: Cell<usize>, drops: Cell<usize> }
struct Guard(Rc<Probe>);
impl Drop for Guard {
    fn drop(&mut self) {
        assert!(!self.0.running.get()); assert!(self.0.live.replace(false));
        self.0.drops.set(self.0.drops.get() + 1);
    }
}
struct Physical(Rc<Probe>);
impl Drop for Physical { fn drop(&mut self) { self.0.running.set(false); } }
struct Admission { probe: Rc<Probe>, refuse: bool, wrong_identity: bool }
impl SampledSentimentDocumentBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: SampledSentimentDocumentAdmission<'_>)
        -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.probe.admissions.set(self.probe.admissions.get() + 1);
        assert_eq!(request.facts.chunk_count, 2); assert_eq!(request.facts.sample_count, 2);
        assert_eq!(request.kv_reservation_bytes, task().max_kv_bytes);
        assert_eq!(request.mask_node_visits, 10);
        assert!(request.facts.max_result_bytes > request.facts.original_task.max_output_bytes);
        assert!(request.facts.max_output_tokens > u64::from(request.facts.original_task.max_output_tokens));
        if self.refuse { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert!(!self.probe.live.replace(true));
        let mut identity = request.facts.identity.clone();
        if self.wrong_identity { identity.prompt_digest = Sha256Digest::of_bytes(b"other-document"); }
        Ok((identity, Guard(Rc::clone(&self.probe))))
    }
}
struct Driver { probe: Rc<Probe>, clean: bool, prefill: Option<Int4PrefillLimits>, reuse: bool, actual: Int4Work }
impl DocumentDriver<FixturePlan> for Driver {
    fn prefill(&self) -> Option<Int4PrefillLimits> { self.prefill }
    fn reuse_prefill(&self) -> bool { self.reuse }
    fn clean(&self) -> bool { self.clean }
    fn capacity(&self) -> usize { 32 }
    fn group_width(&self) -> usize { 32 }
    fn mask_charge(&self, _: &FixturePlan, budget: Int4JsonBudget, _: u64) -> Result<u64, BatchItemFailure> {
        if budget.json.max_total_mask_node_visits < 10 { Err(BatchItemFailure::reject(BatchCode::WorkLimit)) } else { Ok(10) }
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &FixturePlan, budget: Int4JsonBudget,
        _: u64, _: &mut C) -> Result<FixtureOutput, BatchItemFailure> {
        assert_eq!(budget.native.max_forward_positions, work(1).forward_positions);
        assert_eq!(budget.json.max_total_mask_node_visits, 10);
        assert!(self.probe.live.get()); assert!(!self.probe.running.replace(true));
        let _physical = Physical(Rc::clone(&self.probe));
        self.probe.calls.set(self.probe.calls.get() + 1);
        match plan.mode {
            Mode::RejectedChunk => return Err(map_failure(SampledSentimentMapError::Sampled(SampledSentimentError::NoResult))),
            Mode::Dirty => self.clean = false,
            Mode::Panic => panic!("synthetic document execution panic"), _ => (),
        }
        Ok(FixtureOutput { chunks: if matches!(plan.mode, Mode::WrongBinding) { 1 } else { 2 },
            complete: true, reuse_prefill: self.reuse, work: self.actual })
    }
}
fn host(probe: &Rc<Probe>, total: u64) -> (Driver, Admission, RunState) {
    (Driver { probe: Rc::clone(probe), clean: true, prefill: None, reuse: false, actual: work(1) },
        Admission { probe: Rc::clone(probe), refuse: false, wrong_identity: false }, RunState::new(limits(total)))
}
fn prepared(mode: Mode, seal: &Arc<()>, driver: &Driver) -> PreparedRecord<FixturePlan> {
    prepare_document(FixturePlan { identity: identity(), mode, context: 8 }, seal,
        driver, limits(2), 1024, &mut Continue).unwrap()
}

#[test]
fn complete_output_and_chunk_limits_have_independent_admission_authority() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut state) = host(&probe, 1);
    let output = execute_document(prepared(Mode::Success, &seal, &driver), &seal,
        &mut driver, &mut admission, &mut state, context(1, 1), &mut Continue).unwrap();
    assert!(probe.live.get()); assert!(!probe.running.get());
    assert_eq!(output.result().chunks, 2); assert_eq!(state.remaining, work(0));
    assert_eq!(state.remaining_masks, 0); drop(output); assert_eq!(probe.drops.get(), 1);
    let too_long = FixturePlan { identity: identity(), mode: Mode::Success, context: 33 };
    assert_eq!(prepare_document(too_long, &seal, &driver, limits(1), 1024, &mut Continue)
        .err().unwrap().fault.code, BatchCode::Admission);
}

#[test]
fn failed_admission_or_one_failed_chunk_keeps_the_entire_cold_document_debit() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut state) = host(&probe, 3);
    admission.refuse = true;
    let error = execute_document(prepared(Mode::Success, &seal, &driver), &seal,
        &mut driver, &mut admission, &mut state, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(probe.calls.get(), 0); assert_eq!(state.remaining, work(2));
    admission.refuse = false;
    let error = execute_document(prepared(Mode::RejectedChunk, &seal, &driver), &seal,
        &mut driver, &mut admission, &mut state, context(2, 7), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(probe.calls.get(), 1); assert_eq!(probe.drops.get(), 1);
    assert_eq!(state.remaining, work(1)); assert_eq!(state.remaining_masks, 10);
    let output = execute_document(prepared(Mode::Success, &seal, &driver), &seal,
        &mut driver, &mut admission, &mut state, context(3, 8), &mut Continue).unwrap(); drop(output);
    assert_eq!(state.remaining, work(0)); assert_eq!(state.remaining_masks, 0);
}

#[test]
fn foreign_factory_identity_and_incomplete_or_dirty_output_stop_the_adapter() {
    for mode in [Mode::WrongBinding, Mode::Dirty] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut state) = host(&probe, 2);
        let error = execute_document(prepared(mode, &seal, &driver), &seal,
            &mut driver, &mut admission, &mut state, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(state.state, State::Failed); assert!(state.enter().is_err());
        assert!(!probe.live.get()); assert!(!probe.running.get()); assert_eq!(probe.drops.get(), 1);
    }
    for foreign_factory in [false, true] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut state) = host(&probe, 2);
        admission.wrong_identity = !foreign_factory;
        let expected = if foreign_factory { Arc::new(()) } else { Arc::clone(&seal) };
        let error = execute_document(prepared(Mode::Success, &seal, &driver), &expected,
            &mut driver, &mut admission, &mut state, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission); assert_eq!(probe.calls.get(), 0);
        assert!(!probe.live.get()); assert!(!probe.running.get());
    }
}

#[test]
fn every_late_cancellation_preserves_cause_withholds_document_and_releases_after_drain() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget,
        DecodeCancellationKind::FailFast, DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut state) = host(&probe, 2);
        let record = prepared(Mode::Success, &seal, &driver);
        let mut stop = Stop { calls: 0, at: 3, cause };
        let error = execute_document(record, &seal, &mut driver, &mut admission, &mut state,
            context(1, 1), &mut stop).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
        assert_eq!(probe.calls.get(), 1); assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
        assert_eq!(state.remaining, work(1)); assert_eq!(state.remaining_masks, 10);
    }
}

#[test]
fn unwind_latches_failure_and_cannot_reopen_the_spent_document_allowance() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut state) = host(&probe, 2);
    let record = prepared(Mode::Panic, &seal, &driver);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = execute_document(record, &seal, &mut driver, &mut admission, &mut state,
            context(1, 1), &mut Continue);
    }));
    assert!(result.is_err()); assert!(!probe.running.get()); assert!(!probe.live.get());
    assert_eq!(probe.drops.get(), 1); assert_eq!(state.remaining, work(1)); assert!(state.enter().is_err());
}

struct Harness { driver: Driver, admission: Admission, state: RunState, seal: Arc<()> }
impl BatchProcessor for Harness {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedRecord<FixturePlan>;
    type Output = GuardedOutput<FixtureOutput, Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>,
        control: &mut C) -> Result<Self::Prepared, BatchItemFailure> {
        self.state.enter()?;
        let result = (|| {
            let args = document.task_args.unwrap_or_default(); item_budget(args.budget, task())?;
            let plan = FixturePlan { identity: identity(), context: 8,
                mode: if document.text == "reject" { Mode::RejectedChunk } else { Mode::Success } };
            let record = prepare_document(plan, &self.seal, &self.driver, self.state.limits, 1024, control)?;
            self.state.check_available(record.plan.facts().model_work, record.mask_node_visits)?;
            Ok(record)
        })();
        settle(result, self.driver.clean, &mut self.state)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { transport_work(prepared.plan.facts().model_work) }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_document(prepared, &self.seal, &mut self.driver, &mut self.admission, &mut self.state, context, control)
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
fn real_transport_retains_guards_through_flush_and_never_refunds_cold_work() {
    for reuse in [false, true] {
        let probe = Rc::new(Probe::default()); let (mut driver, admission, state) = host(&probe, 3);
        driver.reuse = reuse;
        if reuse { driver.actual = work(0); }
        let mut harness = Harness { driver, admission, state, seal: Arc::new(()) };
        let mut source = Cursor::new(b"{\"id\":\"same\",\"text\":\"ok\"}\n{\"id\":\"same\",\"text\":\"duplicate\"}\n{\"flush\":true}\n{\"id\":\"same\",\"text\":\"reject\"}\n{\"id\":\"third\",\"text\":\"ok\"}\n{\"id\":\"fourth\",\"text\":\"unfunded\"}\n");
        let mut sink = Sink { probe: Rc::clone(&probe), bytes: Vec::new(), delivering: false, fail: false };
        let transport = BatchLimits { max_work: transport_work(work(8)), ..BatchLimits::default() };
        let summary = run_ndjson(&mut source, &mut sink, &mut harness, transport, &mut Continue).unwrap();
        assert_eq!((summary.succeeded, summary.failed), (2, 3)); assert_eq!(summary.reserved_work, transport_work(work(3)));
        assert_eq!(harness.state.remaining, work(0)); assert_eq!(harness.state.remaining_masks, 0);
        assert_eq!(probe.calls.get(), 3); assert_eq!(probe.drops.get(), 3); assert!(!probe.live.get());
        let rows: Vec<serde_json::Value> = std::str::from_utf8(&sink.bytes).unwrap().lines()
            .map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(rows.last().unwrap()["event"], "run_complete");
        assert_eq!(rows.iter().filter(|row| row["event"] == "doc").count(), 2);
        assert!(rows.iter().any(|row| row["error"]["code"] == "duplicate_id"));
        assert!(rows.iter().any(|row| row["error"]["code"] == "work_limit"));
    }
}

#[test]
fn failed_result_delivery_never_executes_the_next_document_or_publishes_completion() {
    let probe = Rc::new(Probe::default()); let (driver, admission, state) = host(&probe, 2);
    let mut harness = Harness { driver, admission, state, seal: Arc::new(()) };
    let mut source = Cursor::new(b"{\"id\":\"a\",\"text\":\"ok\"}\n{\"id\":\"b\",\"text\":\"ok\"}\n");
    let mut sink = Sink { probe: Rc::clone(&probe), bytes: Vec::new(), delivering: false, fail: true };
    let transport = BatchLimits { max_work: transport_work(work(8)), ..BatchLimits::default() };
    let error = run_ndjson(&mut source, &mut sink, &mut harness, transport, &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::OutputIo); assert_eq!(probe.calls.get(), 1);
    assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
    assert_eq!(harness.state.remaining, work(1)); assert_eq!(harness.state.remaining_masks, 10);
    assert!(!std::str::from_utf8(&sink.bytes).unwrap().contains("run_complete"));
}

#[test]
fn preparation_quote_is_the_same_complete_map_quote_with_one_shared_encoder() {
    let task = TaskBudget { max_input_tokens: 4096, max_output_tokens: 1024,
        max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
    let options = SampledSentimentOptions { effective_seed: [53; 32], samples: 2,
        temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
        max_sample_tokens: 128, max_justification_chars: 64 };
    let mapping = SampledSentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 1024,
        max_chunk_bytes: 8, max_chunk_tokens: 4096, context_tokens: 4096, reserved_tokens: 127,
        max_chunks: 64, max_tokenizer_calls: 1024 }, max_model_work: Int4Work {
            forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        max_retained_result_bytes: 4 << 20, max_result_bytes: 8 << 20 };
    let encoding = UntrustedEncoding::ByteFallback;
    let planning = SampledSentimentLimits::default();
    let expected = SampledSentimentMapMemoryLimits { context_tokens: 4096,
        preparation_reserve_bytes: MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: 1024, max_axes: 4, max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
        .requirements(32, task, planning, mapping, &options, &encoding).unwrap();
    assert_eq!(sampled_sentiment_document_preparation_bytes(32, task, planning, mapping, &options, &encoding).unwrap(),
        expected.input_bytes);
    let mut invalid = options.clone(); invalid.samples = 0;
    assert!(sampled_sentiment_document_preparation_bytes(32, task, planning, mapping, &invalid, &encoding).is_err());
}
