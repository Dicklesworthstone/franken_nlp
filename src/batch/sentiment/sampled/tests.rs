//! Private synthetic lifecycle fixtures. These exercise the real transport,
//! accounting and output guards without pretending to execute model weights.

use super::*;
use std::{cell::Cell, io::{self, Cursor, Write}, rc::Rc};
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchLimits, run_ndjson},
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::sentiment::sampled::{SampledSentimentDimension, SampledSentimentSample,
        SAMPLED_SENTIMENT_AGGREGATION_VERSION},
};

fn work(units: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * units, projected_logits: 50 * units, attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn limits(total: u64) -> SampledSentimentBatchLimits {
    SampledSentimentBatchLimits { max_item_work: work(1), max_model_work: work(total),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 100,
            checkpoint_interval_nodes: 8 }, max_visits_per_item: 5, max_visits_per_run: 5 * total },
        max_sampler_bytes: json_sampling_workspace_bytes().unwrap() }
}
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"sampled-batch-private-fixture-not-a-model");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: digest,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, 32_usize)).unwrap()),
        tokenizer_digest: digest, template_digest: digest, task_spec: "sentiment-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "fixture".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 32, max_output_tokens: 16, max_output_bytes: 8192,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(32, 32).unwrap().kv_bytes }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct StopAt { calls: usize, stop: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for StopAt {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop).then_some(self.cause)
    }
}
#[derive(Clone, Copy)]
enum Mode { Success, NoResult, Dirty, Cancel, WrongWork, WrongSchedule, Panic }
struct FixturePlan { identity: ExecutionIdentity, mode: Mode, required_context: usize }
fn plan(mode: Mode) -> FixturePlan { FixturePlan { identity: identity(), mode, required_context: 8 } }
impl Plan for FixturePlan {
    fn facts(&self) -> SampledSentimentPlanFacts<'_> {
        SampledSentimentPlanFacts { identity: &self.identity, model_work: work(1), group_width: 32,
            sample_count: 1, required_context: self.required_context, task_budget: task(),
            max_result_bytes: 8192, preparation_bytes: 1024 }
    }
    fn verify(&self, admitted: &ExecutionIdentity) -> Result<(), BatchItemFailure> {
        if &self.identity != admitted { Err(BatchItemFailure::fatal(BatchCode::Admission)) } else { Ok(()) }
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
struct Admission { probe: Rc<Probe>, refuse: bool, wrong_identity: bool, cancel_as_rejection: bool }
impl SampledSentimentBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: SampledSentimentAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.probe.admissions.set(self.probe.admissions.get() + 1);
        assert_eq!(request.facts.sample_count, 1);
        assert_eq!(request.kv_reservation_bytes, task().max_kv_bytes);
        assert_eq!(request.mask_node_visits, 5);
        if self.refuse { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        if self.cancel_as_rejection {
            return Err(BatchItemFailure::reject(BatchFault::cancelled(DecodeCancellationKind::Shutdown)));
        }
        assert!(!self.probe.live.replace(true));
        let mut identity = request.facts.identity.clone();
        if self.wrong_identity { identity.prompt_digest = Sha256Digest::of_bytes(b"other-private-document"); }
        Ok((identity, Guard(Rc::clone(&self.probe))))
    }
}
struct Engine {
    probe: Rc<Probe>, clean: bool, capacity: usize, masks: u64, prefill: Option<Int4PrefillLimits>,
    reuse_prefill: bool, actual_work: Int4Work,
}
impl Driver<FixturePlan> for Engine {
    fn prefill(&self) -> Option<Int4PrefillLimits> { self.prefill }
    fn reuse_prefill(&self) -> bool { self.reuse_prefill }
    fn clean(&self) -> bool { self.clean }
    fn capacity(&self) -> usize { self.capacity }
    fn group_width(&self) -> usize { 32 }
    fn mask_charge(&self, _: &FixturePlan, budget: Int4JsonBudget, _: u64) -> Result<u64, BatchItemFailure> {
        if self.masks > budget.json.max_total_mask_node_visits {
            Err(BatchItemFailure::reject(BatchCode::WorkLimit))
        } else { Ok(self.masks) }
    }
    fn execute<C: DecodeStepControl>(&mut self, plan: &FixturePlan, identity: &ExecutionIdentity,
        budget: Int4JsonBudget, _: u64, _: &mut C) -> Result<SampledSentimentRun, BatchItemFailure> {
        plan.verify(identity)?;
        assert_eq!(budget.native.max_forward_positions, work(1).forward_positions);
        assert_eq!(budget.json.max_total_mask_node_visits, 5);
        assert!(self.probe.live.get()); assert!(!self.probe.running.replace(true));
        let _physical = Physical(Rc::clone(&self.probe)); self.probe.calls.set(self.probe.calls.get() + 1);
        match plan.mode {
            Mode::NoResult => return Err(task_failure(SampledSentimentError::NoResult)),
            Mode::Cancel => return Err(task_failure(SampledSentimentError::Cancelled(DecodeCancellationKind::Deadline))),
            Mode::Panic => panic!("private sampled batch panic fixture"),
            Mode::Dirty => self.clean = false, _ => (),
        }
        let sample = SampledSentimentSample { sample_index: 0, score: 0,
            justification: "Synthetic lifecycle fixture.".to_owned(), generated_tokens: 2,
            model_work: self.actual_work, mask_node_visit_charge: 3 };
        let reuse_prefill = self.reuse_prefill ^ matches!(plan.mode, Mode::WrongSchedule);
        let execution = if reuse_prefill { SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION }
            else { SAMPLED_SENTIMENT_EXECUTION };
        let mut output = SampledSentimentRun { schema_version: 1, execution: execution.to_owned(),
            task_spec: "sentiment-v1".to_owned(), prompt_template_sha256: identity.template_digest,
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: 32,
            aggregation: SAMPLED_SENTIMENT_AGGREGATION_VERSION.to_owned(),
            interpretation: "generated_score_variation_not_calibrated_confidence".to_owned(),
            justification_status: "unverified_model_explanations_from_untrusted_document".to_owned(),
            sampler_version: JSON_SAMPLER_VERSION.to_owned(), samples_per_axis: 1,
            dimensions: vec![SampledSentimentDimension { axis: SentimentAxis::Valence,
                low_anchor: "negative".to_owned(), high_anchor: "positive".to_owned(),
                mean: 0.0, median: 0.0, minimum: 0, maximum: 0, population_stddev: 0.0, samples: vec![sample] }],
            model_work: self.actual_work, generated_tokens: 2, mask_node_visit_charge: 3 };
        if matches!(plan.mode, Mode::WrongWork) { output.model_work.projections.multiply_accumulates += 1; }
        Ok(output)
    }
}
fn host(probe: &Rc<Probe>, total: u64) -> (Engine, Admission, RunState) {
    (Engine { probe: Rc::clone(probe), clean: true, capacity: 32, masks: 5, prefill: None,
        reuse_prefill: false, actual_work: work(1) },
        Admission { probe: Rc::clone(probe), refuse: false, wrong_identity: false, cancel_as_rejection: false },
        RunState::new(limits(total)))
}
fn prepared(mode: Mode, seal: &Arc<()>, driver: &Engine) -> PreparedRecord<FixturePlan> {
    prepare_checked(plan(mode), seal, driver, limits(2), 1024, &mut Continue).unwrap()
}
fn decrement(work: &mut Int4Work, axis: usize) {
    match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1,
        2 => work.attention_pairs -= 1, 3 => work.projections.dot_products -= 1,
        _ => work.projections.multiply_accumulates -= 1 }
}

#[test]
fn record_arguments_allow_axes_and_tighter_budgets_with_complete_defaults() {
    let defaults: SampledSentimentBatchArgs = serde_json::from_str("{}").unwrap();
    assert_eq!(defaults.axes, SentimentAxis::ALL); assert!(defaults.budget.is_none());
    let selected: SampledSentimentBatchArgs = serde_json::from_str(r#"{"axes":["arousal"]}"#).unwrap();
    assert_eq!(selected.axes, [SentimentAxis::Arousal]); assert!(selected.budget.is_none());
    let requested = serde_json::json!({"axes": ["arousal"], "budget": task()});
    let selected: SampledSentimentBatchArgs = serde_json::from_value(requested).unwrap();
    assert_eq!(selected.budget, Some(task()));
    for bad in [r#"{"seed":"secret"}"#, r#"{"samples":1}"#, r#"{"budget":{}}"#,
        r#"{"axes":null}"#, r#"{"axes":["unknown"]}"#, r#"{"document":"replace"}"#,
        r#"{"id":"replace-semantic-address"}"#, r#"{"reuse_prefill":true}"#] {
        assert!(serde_json::from_str::<SampledSentimentBatchArgs>(bad).is_err());
    }
}

#[test]
fn all_native_mask_and_sampler_limits_are_checked_before_construction() {
    limits(2).validate().unwrap();
    for axis in 0..5 {
        let mut bad = limits(1); decrement(&mut bad.max_model_work, axis);
        assert!(bad.validate().is_err());
    }
    for axis in 0..5 {
        let mut bad = limits(1);
        match axis { 0 => bad.masks.per_mask.max_trie_node_visits = 0,
            1 => bad.masks.per_mask.checkpoint_interval_nodes = 0,
            2 => bad.masks.max_visits_per_item = 0, 3 => bad.masks.max_visits_per_run = 4,
            _ => bad.max_sampler_bytes -= 1 }
        assert!(bad.validate().is_err());
    }
}

#[test]
fn native_and_mask_debits_are_atomic_and_do_not_depend_on_flush_epochs() {
    for axis in 0..6 {
        let mut state = RunState::new(limits(1));
        if axis < 5 { decrement(&mut state.remaining, axis); } else { state.remaining_masks = 4; }
        let before = (state.remaining, state.remaining_masks);
        state.enter().unwrap();
        assert!(state.charge(work(1), 5, context(1, 1)).unwrap_err().stop);
        assert_eq!((state.remaining, state.remaining_masks), before);
        assert_eq!(state.last_sequence, 0);
    }
    let mut state = RunState::new(limits(2));
    for (sequence, epoch) in [(1, 1), (3, 2)] {
        state.enter().unwrap(); state.charge(work(1), 5, context(sequence, epoch)).unwrap();
        settle(Ok(()), true, &mut state).unwrap();
    }
    assert_eq!(state.remaining, work(0)); assert_eq!(state.remaining_masks, 0);
    assert!(state.check_available(work(1), 5).is_err());
}

#[test]
fn failed_admission_and_semantic_no_result_keep_all_reserved_work() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut run) = host(&probe, 3);
    admission.refuse = true;
    let error = execute_owned(prepared(Mode::Success, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(run.remaining, work(2)); assert_eq!(run.remaining_masks, 10);
    assert_eq!(probe.calls.get(), 0);
    admission.refuse = false;
    let error = execute_owned(prepared(Mode::NoResult, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(3, 2), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(run.state, State::Ready);
    assert_eq!(run.remaining, work(1)); assert_eq!(run.remaining_masks, 5);
    assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
    let output = execute_owned(prepared(Mode::Success, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(4, 2), &mut Continue).unwrap();
    assert!(probe.live.get()); assert!(!probe.running.get()); drop(output);
    assert_eq!(run.remaining, work(0)); assert_eq!(run.remaining_masks, 0);
}

#[test]
fn cross_factory_wrong_identity_dirty_completion_and_cancellation_close_the_adapter() {
    for mode in [Mode::Dirty, Mode::Cancel, Mode::WrongWork] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 2);
        let error = execute_owned(prepared(mode, &seal, &driver), &seal, &mut driver,
            &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(run.state, State::Failed);
        assert_eq!(run.remaining, work(1)); assert_eq!(run.remaining_masks, 5);
        assert!(!probe.live.get()); assert!(!probe.running.get()); assert!(run.enter().is_err());
    }
    for wrong_factory in [true, false] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 2);
        admission.wrong_identity = !wrong_factory;
        let record = prepared(Mode::Success, &seal, &driver);
        let expected = if wrong_factory { Arc::new(()) } else { Arc::clone(&seal) };
        let error = execute_owned(record, &expected, &mut driver, &mut admission,
            &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
        assert_eq!(probe.calls.get(), 0); assert!(!probe.live.get());
    }
}

#[test]
fn admission_cannot_make_a_cancelled_fault_recoverable() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut run) = host(&probe, 2);
    admission.cancel_as_rejection = true;
    let error = execute_owned(prepared(Mode::Success, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Shutdown));
    assert_eq!(run.state, State::Failed); assert_eq!(run.remaining_masks, 5);
    assert_eq!(probe.calls.get(), 0);
}

#[test]
fn panicking_native_work_drains_before_guards_drop_and_keeps_failure_latched() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut run) = host(&probe, 2);
    let record = prepared(Mode::Panic, &seal, &driver);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = execute_owned(record, &seal, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue);
    }));
    assert!(result.is_err()); assert!(!probe.running.get()); assert!(!probe.live.get());
    assert_eq!(probe.drops.get(), 1); assert_eq!(run.remaining, work(1)); assert_eq!(run.remaining_masks, 5);
    assert!(run.enter().is_err());
}

#[test]
fn every_typed_cancellation_after_execution_withholds_output_and_retains_debits() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget,
        DecodeCancellationKind::FailFast, DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 2);
        let record = prepared(Mode::Success, &seal, &driver);
        let mut control = StopAt { calls: 0, stop: 3, cause };
        let error = execute_owned(record, &seal, &mut driver, &mut admission, &mut run,
            context(1, 1), &mut control).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
        assert_eq!(probe.calls.get(), 1); assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
        assert_eq!(run.remaining, work(1)); assert_eq!(run.remaining_masks, 5);
    }
}

struct Harness { driver: Engine, admission: Admission, run: RunState, seal: Arc<()> }
impl Harness {
    fn new(probe: &Rc<Probe>, total: u64) -> Self {
        let (driver, admission, run) = host(probe, total);
        Self { driver, admission, run, seal: Arc::new(()) }
    }
}
impl BatchProcessor for Harness {
    type Args = SampledSentimentBatchArgs;
    type Prepared = PreparedRecord<FixturePlan>;
    type Output = GuardedOutput<SampledSentimentRun, Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { panic!("control bypass") }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.run.enter()?;
        let result = (|| {
            item_budget(document.task_args.unwrap_or_default().budget, task())?;
            let mode = if document.text == "reject" { Mode::NoResult } else { Mode::Success };
            let record = prepare_checked(plan(mode), &self.seal, &self.driver, self.run.limits, 1024, control)?;
            self.run.check_available(record.plan.facts().model_work, record.mask_node_visits)?;
            Ok(record)
        })();
        settle(result, self.driver.clean(), &mut self.run)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { transport_work(prepared.plan.facts().model_work) }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("context bypass")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_owned(prepared, &self.seal, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}
struct Sink { bytes: Vec<u8>, probe: Rc<Probe>, delivering_result: bool, fail_result: bool }
impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.windows(b"\"result\":".len()).any(|w| w == b"\"result\":") {
            assert!(self.probe.live.get()); assert!(!self.probe.running.get()); self.delivering_result = true;
            if self.fail_result { return Err(io::Error::other("private transport failure")); }
        }
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.delivering_result { assert!(self.probe.live.get()); self.delivering_result = false; }
        Ok(())
    }
}
fn transport() -> BatchLimits {
    BatchLimits { max_work: transport_work(work(8)), ..BatchLimits::default() }
}
fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(bytes).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect()
}

#[test]
fn real_transport_flushes_ids_without_refunding_native_or_mask_work() {
    let probe = Rc::new(Probe::default()); let mut processor = Harness::new(&probe, 3);
    let input = b"{\"id\":\"same\",\"text\":\"ok\"}\n{\"id\":\"same\",\"text\":\"duplicate\"}\n{\"flush\":true}\n{\"id\":\"same\",\"text\":\"reject\"}\n{\"id\":\"third\",\"text\":\"ok\"}\n{\"id\":\"fourth\",\"text\":\"unfunded\"}\n";
    let mut reader = Cursor::new(input); let mut sink = Sink { bytes: Vec::new(), probe: Rc::clone(&probe),
        delivering_result: false, fail_result: false };
    let summary = run_ndjson(&mut reader, &mut sink, &mut processor, transport(), &mut Continue).unwrap();
    assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 3); assert_eq!(summary.flushes, 2);
    assert_eq!(summary.reserved_work, transport_work(work(3)));
    assert_eq!(probe.calls.get(), 3); assert_eq!(probe.drops.get(), 3); assert!(!probe.live.get());
    assert_eq!(processor.run.remaining, work(0)); assert_eq!(processor.run.remaining_masks, 0);
    let events = frames(&sink.bytes);
    assert_eq!(events.last().unwrap()["event"], "run_complete");
    assert_eq!(events.iter().filter(|event| event["event"] == "doc").count(), 2);
    assert!(events.iter().any(|event| event["error"]["code"] == "duplicate_id"));
    assert!(events.iter().any(|event| event["error"]["code"] == "work_limit"));
}

#[test]
fn failed_delivery_never_retries_or_emits_terminal_success() {
    let probe = Rc::new(Probe::default()); let mut processor = Harness::new(&probe, 2);
    let mut reader = Cursor::new(b"{\"id\":\"one\",\"text\":\"ok\"}\n{\"id\":\"two\",\"text\":\"ok\"}\n");
    let mut sink = Sink { bytes: Vec::new(), probe: Rc::clone(&probe), delivering_result: false, fail_result: true };
    let error = run_ndjson(&mut reader, &mut sink, &mut processor, transport(), &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::OutputIo); assert_eq!(probe.calls.get(), 1);
    assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
    assert_eq!(processor.run.remaining, work(1)); assert_eq!(processor.run.remaining_masks, 5);
    assert!(!frames(&sink.bytes).iter().any(|event| event["event"] == "run_complete"));
}

#[test]
fn deferred_footer_uses_original_control_after_worker_join() {
    let probe = Rc::new(Probe::default()); let mut processor = Harness::new(&probe, 1);
    let mut reader = Cursor::new(b"{\"id\":\"one\",\"text\":\"ok\"}\n");
    let mut bytes = Vec::new();
    let completion = crate::batch::deferred::run_ndjson_deferred(&mut reader, &mut bytes,
        &mut processor, transport(), &mut Continue).unwrap();
    assert_eq!(probe.calls.get(), 1); assert!(!probe.live.get());
    assert!(!frames(&bytes).iter().any(|event| event["event"] == "run_complete"));
    let mut stop = StopAt { calls: 0, stop: 1, cause: DecodeCancellationKind::Shutdown };
    assert_eq!(completion.publish(&mut bytes, &mut stop).unwrap_err().fault.cancellation,
        Some(DecodeCancellationKind::Shutdown));
    assert!(!frames(&bytes).iter().any(|event| event["event"] == "run_complete"));
}

#[test]
fn record_budgets_can_tighten_every_ceiling_without_changing_the_run_defaults() {
    let ceiling = task();
    assert_eq!(item_budget(None, ceiling).unwrap(), ceiling);
    let tighter = TaskBudget { max_input_tokens: ceiling.max_input_tokens / 2,
        max_output_tokens: ceiling.max_output_tokens / 2, max_output_bytes: ceiling.max_output_bytes / 2,
        max_grammar_states: ceiling.max_grammar_states / 2, max_kv_bytes: ceiling.max_kv_bytes / 2 };
    assert_eq!(item_budget(Some(tighter), ceiling).unwrap(), tighter);
    assert_eq!(item_budget(None, ceiling).unwrap(), ceiling);
    for field in 0..5 {
        for zero in [false, true] {
            let mut bad = ceiling;
            match field {
                0 => bad.max_input_tokens = if zero { 0 } else { ceiling.max_input_tokens + 1 },
                1 => bad.max_output_tokens = if zero { 0 } else { ceiling.max_output_tokens + 1 },
                2 => bad.max_output_bytes = if zero { 0 } else { ceiling.max_output_bytes + 1 },
                3 => bad.max_grammar_states = if zero { 0 } else { ceiling.max_grammar_states + 1 },
                _ => bad.max_kv_bytes = if zero { 0 } else { ceiling.max_kv_bytes + 1 },
            }
            let error = item_budget(Some(bad), ceiling).unwrap_err();
            assert_eq!(error.fault.code, BatchCode::Planning); assert!(!error.stop);
        }
    }
}

#[test]
fn widened_record_budgets_reject_before_admission_without_spending_work_or_breaking_the_stream() {
    let probe = Rc::new(Probe::default()); let mut processor = Harness::new(&probe, 1);
    let mut input = String::new();
    for field in 0..5 {
        let mut budget = task();
        match field { 0 => budget.max_input_tokens += 1, 1 => budget.max_output_tokens += 1,
            2 => budget.max_output_bytes += 1, 3 => budget.max_grammar_states += 1, _ => budget.max_kv_bytes += 1 }
        input.push_str(&serde_json::to_string(&serde_json::json!({
            "id": format!("refused-{field}"), "text": "PRIVATE_BUDGET_DOCUMENT", "task_args": {"budget": budget}
        })).unwrap());
        input.push('\n');
    }
    input.push_str("{\"id\":\"accepted\",\"text\":\"ok\"}\n");
    let mut reader = Cursor::new(input.into_bytes());
    let mut sink = Sink { bytes: Vec::new(), probe: Rc::clone(&probe), delivering_result: false, fail_result: false };
    let summary = run_ndjson(&mut reader, &mut sink, &mut processor, transport(), &mut Continue).unwrap();
    assert_eq!((summary.succeeded, summary.failed), (1, 5));
    assert_eq!(summary.reserved_work, transport_work(work(1)));
    assert_eq!(probe.calls.get(), 1); assert_eq!(probe.admissions.get(), 1); assert_eq!(probe.drops.get(), 1);
    assert_eq!(processor.run.remaining, work(0)); assert_eq!(processor.run.remaining_masks, 0);
    let events = frames(&sink.bytes);
    assert_eq!(events.iter().filter(|event| event["error"]["code"] == "planning").count(), 5);
    assert_eq!(events.last().unwrap()["event"], "run_complete");
    assert!(!std::str::from_utf8(&sink.bytes).unwrap().contains("PRIVATE_BUDGET_DOCUMENT"));
}

#[test]
fn invalid_prefill_schedules_stop_before_admission_and_native_work() {
    let required = Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap();
    for prefill in [
        Int4PrefillLimits { max_batch_rows: 0, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: usize::MAX, max_extra_scratch_bytes: u64::MAX },
        Int4PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: required - 1 },
    ] {
        let probe = Rc::new(Probe::default()); let mut processor = Harness::new(&probe, 1);
        processor.driver.prefill = Some(prefill);
        let mut reader = Cursor::new(b"{\"id\":\"one\",\"text\":\"ok\"}\n");
        let mut sink = Sink { bytes: Vec::new(), probe: Rc::clone(&probe),
            delivering_result: false, fail_result: false };
        let error = run_ndjson(&mut reader, &mut sink, &mut processor, transport(), &mut Continue).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::InvalidLimits);
        assert_eq!(probe.admissions.get(), 0); assert_eq!(probe.calls.get(), 0);
        assert_eq!(probe.drops.get(), 0); assert!(!probe.live.get());
        assert_eq!(processor.run.state, State::Failed);
        assert_eq!(processor.run.remaining, work(1)); assert_eq!(processor.run.remaining_masks, 5);
        assert!(!frames(&sink.bytes).iter().any(|event|
            event["event"] == "doc" || event["event"] == "run_complete"));
    }
}

#[test]
fn admitted_prefill_widths_preserve_stream_identity_and_nonrenewable_work() {
    let prefill = |rows| Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
    let mut expected_results = None;
    // These private drivers exercise the actual stream accounting and
    // schedule preflight; task/native model gates own numerical parity.
    for schedule in [None, Some(prefill(1)), Some(prefill(4)), Some(prefill(64))] {
        let probe = Rc::new(Probe::default()); let mut processor = Harness::new(&probe, 2);
        processor.driver.prefill = schedule;
        let input = b"{\"id\":\"one\",\"text\":\"ok\"}\n{\"flush\":true}\n{\"id\":\"two\",\"text\":\"ok\"}\n{\"id\":\"three\",\"text\":\"unfunded\"}\n";
        let mut reader = Cursor::new(input);
        let mut sink = Sink { bytes: Vec::new(), probe: Rc::clone(&probe),
            delivering_result: false, fail_result: false };
        let summary = run_ndjson(&mut reader, &mut sink, &mut processor, transport(), &mut Continue).unwrap();
        assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 1);
        assert_eq!(summary.reserved_work, transport_work(work(2)));
        assert_eq!(probe.admissions.get(), 2); assert_eq!(probe.calls.get(), 2);
        assert_eq!(probe.drops.get(), 2); assert!(!probe.live.get());
        assert_eq!(processor.run.remaining, work(0)); assert_eq!(processor.run.remaining_masks, 0);
        let events = frames(&sink.bytes);
        assert!(events.iter().any(|event| event["error"]["code"] == "work_limit"));
        let results = events.iter().filter(|event| event["event"] == "doc")
            .map(|event| event["result"].clone()).collect::<Vec<_>>();
        assert_eq!(results.len(), 2);
        if let Some(expected) = &expected_results { assert_eq!(&results, expected); }
        else { expected_results = Some(results); }
    }
}

#[test]
fn reused_schedules_report_actual_work_without_refunding_cold_admission_across_flushes() {
    let prefill = |rows| Int4PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
    // This lifecycle driver supplies a bounded observed-work receipt. The
    // task/native differential fixtures own the causal-replay arithmetic and
    // exact seeded-token proof; no model execution is claimed by this fixture.
    let actual = Int4Work { forward_positions: 30, projected_logits: 50, attention_pairs: 40,
        projections: ProjectionWork { dot_products: 50, multiply_accumulates: 60 } };
    for schedule in [None, Some(prefill(1)), Some(prefill(4)), Some(prefill(64))] {
        let probe = Rc::new(Probe::default()); let mut processor = Harness::new(&probe, 2);
        processor.driver.prefill = schedule;
        processor.driver.reuse_prefill = true;
        processor.driver.actual_work = actual;
        let input = b"{\"id\":\"one\",\"text\":\"ok\"}\n{\"flush\":true}\n{\"id\":\"two\",\"text\":\"ok\"}\n{\"id\":\"three\",\"text\":\"unfunded\"}\n";
        let mut reader = Cursor::new(input);
        let mut sink = Sink { bytes: Vec::new(), probe: Rc::clone(&probe),
            delivering_result: false, fail_result: false };
        let summary = run_ndjson(&mut reader, &mut sink, &mut processor, transport(), &mut Continue).unwrap();
        assert_eq!((summary.succeeded, summary.failed), (2, 1));
        assert_eq!(summary.reserved_work, transport_work(work(2)));
        assert_eq!(processor.run.remaining, work(0)); assert_eq!(processor.run.remaining_masks, 0);
        assert_eq!(probe.calls.get(), 2); assert_eq!(probe.admissions.get(), 2);
        assert_eq!(probe.drops.get(), 2); assert!(!probe.live.get());
        let events = frames(&sink.bytes);
        let results = events.iter().filter(|event| event["event"] == "doc")
            .map(|event| &event["result"]).collect::<Vec<_>>();
        assert_eq!(results.len(), 2);
        for result in results {
            assert_eq!(result["execution"], SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION);
            assert_eq!(result["model_work"], serde_json::to_value(actual).unwrap());
            assert_eq!(result["dimensions"][0]["samples"][0]["model_work"],
                serde_json::to_value(actual).unwrap());
            assert_eq!(result["dimensions"][0]["samples"][0]["score"], 0);
            assert_eq!(result["dimensions"][0]["samples"][0]["justification"],
                "Synthetic lifecycle fixture.");
        }
        assert!(events.iter().any(|event| event["error"]["code"] == "work_limit"));
        assert_eq!(events.last().unwrap()["event"], "run_complete");
    }
}

#[test]
fn completed_schedule_substitution_is_fatal_for_both_cold_and_reused_streams() {
    for reuse_prefill in [false, true] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 2);
        driver.reuse_prefill = reuse_prefill;
        let record = prepared(Mode::WrongSchedule, &seal, &driver);
        let error = execute_owned(record, &seal, &mut driver, &mut admission,
            &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        assert_eq!(run.state, State::Failed); assert_eq!(run.remaining, work(1));
        assert_eq!(run.remaining_masks, 5); assert_eq!(probe.calls.get(), 1);
        assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
        assert!(run.enter().is_err());
    }
}

#[test]
fn reused_streams_never_carry_dirty_native_state_into_another_record() {
    let probe = Rc::new(Probe::default()); let seal = Arc::new(());
    let (mut driver, mut admission, mut run) = host(&probe, 2);
    driver.reuse_prefill = true;
    let error = execute_owned(prepared(Mode::Dirty, &seal, &driver), &seal, &mut driver,
        &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    assert_eq!(run.state, State::Failed); assert_eq!(run.remaining, work(1));
    assert_eq!(run.remaining_masks, 5); assert_eq!(probe.drops.get(), 1);
    assert!(prepare_checked(plan(Mode::Success), &seal, &driver, limits(2), 1024, &mut Continue)
        .err().unwrap().stop);
    assert!(run.enter().is_err()); assert_eq!(probe.calls.get(), 1);
    assert_eq!(probe.admissions.get(), 1); assert!(!probe.live.get());
}

#[test]
fn every_original_cancellation_withholds_reused_output_and_keeps_cold_debits() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget,
        DecodeCancellationKind::FailFast, DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let probe = Rc::new(Probe::default()); let seal = Arc::new(());
        let (mut driver, mut admission, mut run) = host(&probe, 2);
        driver.reuse_prefill = true;
        let record = prepared(Mode::Success, &seal, &driver);
        let mut control = StopAt { calls: 0, stop: 3, cause };
        let error = execute_owned(record, &seal, &mut driver, &mut admission, &mut run,
            context(1, 1), &mut control).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
        assert_eq!(run.state, State::Failed); assert_eq!(probe.calls.get(), 1);
        assert_eq!(probe.drops.get(), 1); assert!(!probe.live.get());
        assert_eq!(run.remaining, work(1)); assert_eq!(run.remaining_masks, 5);
    }
}

#[test]
fn durable_preparation_quote_keeps_recipe_and_encoder_storage_on_every_feature_graph() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let planning = SampledSentimentLimits::default();
    let policies = [UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 8192,
        max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024 })];
    for policy in policies {
        let stream = sampled_sentiment_batch_preparation_bytes(task(), planning, &policy).unwrap();
        let durable = sampled_sentiment_job_preparation_bytes(task(), planning, &policy).unwrap();
        assert_eq!(durable - stream, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64);
        let mut narrower = planning; narrower.max_result_bytes = task().max_output_bytes - 1;
        assert_eq!(sampled_sentiment_job_preparation_bytes(task(), narrower, &policy).unwrap(), durable - 4);
    }
    let mut invalid = task(); invalid.max_input_tokens = 0;
    assert!(sampled_sentiment_job_preparation_bytes(invalid, planning, &UntrustedEncoding::ByteFallback).is_err());
    let mut invalid = planning; invalid.max_result_bytes = 0;
    assert!(sampled_sentiment_job_preparation_bytes(task(), invalid, &UntrustedEncoding::ByteFallback).is_err());
    let invalid = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 1,
        max_output_tokens: 1, max_pair_checks: 0, max_pair_bytes: 1 });
    assert!(sampled_sentiment_job_preparation_bytes(task(), planning, &invalid).is_err());
}
