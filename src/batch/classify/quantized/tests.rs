//! Actual pinned planning, admission wrapping and transport; no model inference.
use super::*;
use std::{cell::Cell, io::{self, Cursor, Write}, rc::Rc};
use serde::Serialize;
use crate::{
    batch::{run_ndjson, BatchLimits},
    tasks::classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
    template::{IM_START, IM_END, THINK_START, THINK_END},
};

fn fixture() -> (ClassificationPlanner, ExecutionIdentity) {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let planner = ClassificationPlanner::pinned(controls.template_controls(), entries[1]["id"].as_u64().unwrap() as u32).unwrap();
    let d = Sha256Digest::of_bytes(b"int8-batch-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "synthetic-only".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: "classify-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    (planner, identity)
}
fn args() -> ClassificationBatchArgs {
    ClassificationBatchArgs { labels: vec![
        ClassificationLabel { id: "Billing issue".to_owned(), description: "Charges and invoices".to_owned() },
        ClassificationLabel { id: "étiquette".to_owned(), description: "A second independent category".to_owned() }],
        mode: ClassificationMode::MultiLabel, policy: ClassificationPolicy::default(),
        budget: TaskBudget { max_input_tokens: 4096, max_output_tokens: 8, max_output_bytes: 1_000_000,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
}
fn document(id: &str) -> BatchDocument<ClassificationBatchArgs> {
    BatchDocument { id: id.to_owned(), text: "é Refund <think> literal text".to_owned(), task_args: None }
}
fn compiler(p: &ClassificationPlanner, identity: ExecutionIdentity) -> Int8ClassificationBatchPlanner<'_> {
    Int8ClassificationBatchPlanner::new(p, identity, args().budget, ClassificationLimits::default(), Some(args())).unwrap()
}
fn allowance(n: u64) -> Int8Work {
    Int8Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn ledger(n: u64) -> WorkLedger { WorkLedger { remaining: allowance(n), state: State::Ready } }
struct Guard(Rc<Cell<usize>>);
impl Guard { fn new(alive: &Rc<Cell<usize>>) -> Self { alive.set(alive.get() + 1); Self(Rc::clone(alive)) } }
impl Drop for Guard { fn drop(&mut self) { self.0.set(self.0.get() - 1); } }
#[derive(Default)]
struct Admission { alive: Rc<Cell<usize>>, seen: Vec<Int8Work>, substitute: bool, failure: Option<BatchItemFailure> }
impl Int8ClassificationAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, proposed: &ExecutionIdentity, work: Int8Work) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.seen.push(work);
        if let Some(error) = self.failure { return Err(error); }
        let mut identity = proposed.clone();
        if self.substitute { identity.packing_set_digest = Sha256Digest::of_bytes(b"foreign packing"); }
        Ok((identity, Guard::new(&self.alive)))
    }
}
// Not an Int8ClassificationRun: a transport fixture cannot mint a native receipt.
#[derive(Serialize)]
struct Payload { scope: &'static str, heads: usize }
struct Provider<'p> { compiler: Int8ClassificationBatchPlanner<'p>, ledger: WorkLedger, admission: Admission }
impl BatchProcessor for Provider<'_> {
    type Args = ClassificationBatchArgs;
    type Prepared = PreparedInt8BatchClassification;
    type Output = GuardedOutput<Payload, Guard>;
    fn prepare(&mut self, input: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        self.ledger.ready()?;
        let prepared = self.compiler.prepare(input)?;
        subtract(self.ledger.remaining, prepared.model_work())?; Ok(prepared)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork { transport_work(prepared.model_work()) }
    fn execute<C: DecodeStepControl>(&mut self, prepared: Self::Prepared, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.compiler.verify(&prepared)?; self.ledger.begin(prepared.model_work())?;
        let alive = Rc::clone(&self.admission.alive);
        let result = with_admission(&mut self.admission, &prepared.plan, control, |_, _| {
            assert_eq!(alive.get(), 1);
            Ok(Payload { scope: "fixture-planning-only", heads: prepared.head_count() })
        });
        self.ledger.finish(result.as_ref().err()); result
    }
}
fn input(id: &str) -> String { serde_json::json!({"id":id,"text":"é Refund <think> literal text"}).to_string() + "\n" }
fn transport_limits() -> BatchLimits {
    BatchLimits { max_work: BatchWork { forward_positions: u64::MAX, projected_logits: u64::MAX }, ..BatchLimits::default() }
}

#[test]
fn subtraction_is_atomic_on_every_native_work_axis() {
    for axis in 0..5 {
        let mut l = ledger(10); let mut work = allowance(4);
        match axis { 0 => work.forward_positions = 11, 1 => work.projected_logits = 11,
            2 => work.attention_pairs = 11, 3 => work.projections.dot_products = 11,
            _ => work.projections.multiply_accumulates = 11 }
        let error = l.begin(work).unwrap_err(); assert_eq!(error.fault.code, BatchCode::WorkLimit); assert!(!error.stop);
        assert_eq!(l.remaining, allowance(10)); assert_eq!(l.state, State::Ready);
    }
}
#[test]
fn successful_and_recoverable_failed_attempts_keep_all_charges() {
    let mut l = ledger(10); l.begin(allowance(4)).unwrap(); l.finish(None);
    l.begin(allowance(4)).unwrap(); l.finish(Some(&BatchItemFailure::reject(BatchCode::Admission)));
    assert_eq!(l.remaining, allowance(2)); assert!(l.ready().is_ok());
    assert!(l.begin(allowance(3)).is_err()); assert_eq!(l.remaining, allowance(2));
}
#[test]
fn fatal_and_unwound_attempts_never_restore_adapter_readiness() {
    let mut l = ledger(10); l.begin(allowance(4)).unwrap(); l.finish(Some(&BatchItemFailure::fatal(BatchCode::Execution)));
    assert!(l.ready().is_err()); assert_eq!(l.remaining, allowance(6));
    let mut l = ledger(10);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        l.begin(allowance(4)).unwrap(); panic!("synthetic callback unwind");
    }));
    assert!(panic.is_err()); assert!(l.ready().is_err()); assert_eq!(l.remaining, allowance(6));
}
#[test]
fn factory_rejects_bf16_and_foreign_prepared_limit_or_model_configurations() {
    let (p, id) = fixture(); let own = compiler(&p, id.clone()); let prepared = own.prepare(document("a")).unwrap();
    own.verify(&prepared).unwrap();
    let mut bf16 = id.clone(); bf16.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(Int8ClassificationBatchPlanner::new(&p, bf16, args().budget, ClassificationLimits::default(), None).is_err());
    let other_limits = ClassificationLimits { max_labels: 127, ..ClassificationLimits::default() };
    let other = Int8ClassificationBatchPlanner::new(&p, id.clone(), args().budget, other_limits, Some(args())).unwrap();
    assert!(other.verify(&prepared).unwrap_err().stop);
    let mut different = id; different.logical_model_digest = Sha256Digest::of_bytes(b"other");
    assert!(compiler(&p, different).verify(&prepared).unwrap_err().stop);
}
#[test]
fn missing_arguments_and_increased_request_ceiling_do_not_get_invented_authority() {
    let (p, id) = fixture();
    let missing = Int8ClassificationBatchPlanner::new(&p, id.clone(), args().budget, ClassificationLimits::default(), None).unwrap();
    let error = missing.prepare(document("a")).err().unwrap(); assert_eq!(error.fault.code, BatchCode::Planning); assert!(!error.stop);
    let own = compiler(&p, id); let mut d = document("a"); let mut raised = args(); raised.budget.max_output_tokens += 1; d.task_args = Some(raised);
    assert_eq!(own.prepare(d).err().unwrap().fault.code, BatchCode::Planning);
    let mut value = serde_json::to_value(args()).unwrap(); value["eos_token_id"] = serde_json::json!(0);
    assert!(serde_json::from_value::<ClassificationBatchArgs>(value).is_err());
}
#[test]
fn host_receives_complete_integer_and_attention_work_and_guard_outlives_execution() {
    let (p, id) = fixture(); let own = compiler(&p, id); let prepared = own.prepare(document("a")).unwrap();
    let mut admission = Admission::default(); let alive = Rc::clone(&admission.alive);
    let output = with_admission(&mut admission, &prepared.plan, &mut Continue, |_, _| {
        assert_eq!(alive.get(), 1); Ok(Payload { scope: "fixture-planning-only", heads: prepared.head_count() })
    }).unwrap();
    assert_eq!(admission.seen, [prepared.model_work()]);
    assert!(admission.seen[0].attention_pairs > 0); assert!(admission.seen[0].projections.multiply_accumulates > admission.seen[0].projected_logits);
    assert_eq!(alive.get(), 1); canonjson::canonical_bytes(&output).unwrap(); assert_eq!(alive.get(), 1);
    drop(output); assert_eq!(alive.get(), 0);
}
#[test]
fn substituted_admitted_identity_prevents_native_callback_and_releases_guard() {
    let (p, id) = fixture(); let own = compiler(&p, id); let prepared = own.prepare(document("a")).unwrap();
    let mut admission = Admission { substitute: true, ..Admission::default() }; let calls = Cell::new(0);
    let result = with_admission(&mut admission, &prepared.plan, &mut Continue, |_, _| { calls.set(1); Ok(()) });
    let error = result.err().unwrap(); assert_eq!(error.fault.code, BatchCode::Admission); assert!(error.stop);
    assert_eq!(calls.get(), 0); assert_eq!(admission.alive.get(), 0);
}
#[test]
fn cancellation_after_admission_releases_guard_without_model_work() {
    struct StopAfterAdmission(usize);
    impl DecodeStepControl for StopAfterAdmission {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0 += 1; (self.0 == 2).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let (p, id) = fixture(); let own = compiler(&p, id); let prepared = own.prepare(document("a")).unwrap();
    let mut admission = Admission::default(); let calls = Cell::new(0);
    let result = with_admission(&mut admission, &prepared.plan, &mut StopAfterAdmission(0), |_, _| { calls.set(1); Ok(()) });
    assert_eq!(result.err().unwrap().fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(admission.seen.len(), 1); assert_eq!(calls.get(), 0); assert_eq!(admission.alive.get(), 0);
}
#[test]
fn native_callback_failure_does_not_leak_the_actual_host_guard() {
    let (p, id) = fixture(); let own = compiler(&p, id); let prepared = own.prepare(document("a")).unwrap();
    let mut admission = Admission::default();
    let result: Result<GuardedOutput<(), Guard>, _> = with_admission(&mut admission, &prepared.plan, &mut Continue, |_, _| {
        Err(BatchItemFailure::fatal(BatchFault::cancelled(DecodeCancellationKind::Shutdown)))
    });
    assert_eq!(result.err().unwrap().fault.cancellation, Some(DecodeCancellationKind::Shutdown)); assert_eq!(admission.alive.get(), 0);
}
#[test]
fn one_model_allowance_is_not_renewed_by_flush_or_another_runner_call() {
    let (p, id) = fixture(); let compiler = compiler(&p, id); let work = compiler.prepare(document("a")).unwrap().model_work();
    let mut provider = Provider { compiler, ledger: WorkLedger { remaining: work, state: State::Ready }, admission: Admission::default() };
    let text = input("a") + "{\"flush\":true}\n" + &input("a"); let mut out = Vec::new();
    let summary = run_ndjson(&mut Cursor::new(text.as_bytes()), &mut out, &mut provider, transport_limits(), &mut Continue).unwrap();
    assert_eq!((summary.succeeded, summary.failed), (1, 1)); assert_eq!(provider.admission.seen.len(), 1);
    assert_eq!(provider.ledger.remaining, Int8Work::default()); assert_eq!(provider.admission.alive.get(), 0);
    let summary = run_ndjson(&mut Cursor::new(input("b").as_bytes()), &mut Vec::new(), &mut provider, transport_limits(), &mut Continue).unwrap();
    assert_eq!((summary.succeeded, summary.failed), (0, 1)); assert_eq!(provider.admission.seen.len(), 1);
}
#[test]
fn transport_work_limit_prevents_admission_even_with_native_budget_available() {
    let (p, id) = fixture(); let compiler = compiler(&p, id);
    let mut provider = Provider { compiler, ledger: ledger(u64::MAX), admission: Admission::default() };
    let limits = BatchLimits { max_work: BatchWork::default(), ..transport_limits() };
    let summary = run_ndjson(&mut Cursor::new(input("a").as_bytes()), &mut Vec::new(), &mut provider, limits, &mut Continue).unwrap();
    assert_eq!((summary.succeeded, summary.failed), (0, 1)); assert!(provider.admission.seen.is_empty());
    assert_eq!(provider.ledger.remaining, allowance(u64::MAX));
}
struct Writer { alive: Rc<Cell<usize>>, bytes: Vec<u8>, fail: u8, started: bool, result_flushes: usize }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.started && self.fail == 2 { return Err(io::Error::other("fixture partial output")); }
        if bytes.windows(b"fixture-planning-only".len()).any(|w| w == b"fixture-planning-only") {
            assert_eq!(self.alive.get(), 1); self.started = true;
            if self.fail == 1 { return Ok(0); }
            if self.fail == 2 { self.bytes.extend_from_slice(&bytes[..3]); return Ok(3); }
        }
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.alive.get() == 1 {
            self.result_flushes += 1;
            if self.fail == 3 { return Err(io::Error::other("fixture failed flush")); }
        }
        Ok(())
    }
}
#[test]
fn real_guard_survives_result_write_and_flush_then_drops_before_next_input() {
    let (p, id) = fixture(); let compiler = compiler(&p, id);
    let mut provider = Provider { compiler, ledger: ledger(u64::MAX), admission: Admission::default() };
    let mut writer = Writer { alive: Rc::clone(&provider.admission.alive), bytes: Vec::new(), fail: 0, started: false, result_flushes: 0 };
    let text = input("a") + &input("b");
    let summary = run_ndjson(&mut Cursor::new(text.as_bytes()), &mut writer, &mut provider, transport_limits(), &mut Continue).unwrap();
    assert_eq!(summary.succeeded, 2); assert_eq!(writer.result_flushes, 2); assert_eq!(provider.admission.alive.get(), 0);
}
#[test]
fn failed_result_delivery_stops_without_retrying_or_reading_another_document() {
    let (p, id) = fixture();
    for fail in 1..=3 {
        let compiler = compiler(&p, id.clone()); let first = input("a"); let text = first.clone() + &input("b");
        let mut provider = Provider { compiler, ledger: ledger(u64::MAX), admission: Admission::default() };
        let mut writer = Writer { alive: Rc::clone(&provider.admission.alive), bytes: Vec::new(), fail, started: false, result_flushes: 0 };
        let mut reader = Cursor::new(text.as_bytes());
        let error = run_ndjson(&mut reader, &mut writer, &mut provider, transport_limits(), &mut Continue).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::OutputIo); assert_eq!(provider.admission.seen.len(), 1);
        assert_eq!(reader.position(), first.len() as u64); assert_eq!(provider.admission.alive.get(), 0);
        assert_eq!(provider.ledger.remaining, subtract(allowance(u64::MAX), provider.admission.seen[0]).unwrap());
        assert!(!String::from_utf8_lossy(&writer.bytes).contains("run_error"));
    }
}
#[test]
fn execution_errors_preserve_cancellation_and_stop_instead_of_abstaining() {
    let nested = Int8ClassificationError::Scoring(Int8ScoringError::Native(StrictInt8Error::Cancelled(DecodeCancellationKind::PollQuota)));
    let error = execution_failure(nested); assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::PollQuota));
    assert!(execution_failure(Int8ClassificationError::Native(StrictInt8Error::Primitive)).stop);
    assert!(execution_failure(Int8ClassificationError::Accounting).stop);
    let bounded = planning_failure(Int8ClassificationError::WorkBudget); assert!(!bounded.stop); assert_eq!(bounded.fault.code, BatchCode::WorkLimit);
}
