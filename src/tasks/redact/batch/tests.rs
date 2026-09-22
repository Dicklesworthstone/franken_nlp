//! Accounting/transport fault injection only; no synthetic fixture is publicly
//! accepted as native inference or evidence of redaction quality.
use super::*;
use std::{cell::RefCell, rc::Rc};
use crate::{
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{constrained_int8, decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::TaskBudget, ner::NerOptions, source_planning::SourcePlanningLimits},
};
fn config() -> Int8RedactionBatchConfig {
    let d = Sha256Digest::of_bytes(b"redaction-corpus-accounting-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "ner-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let one = constrained_int8::planned_work(128, 16).unwrap();
    let per = one.checked_add(one).unwrap();
    Int8RedactionBatchConfig { ner_identity: identity,
        detector: Int8RedactionConfig { ner: NerOptions::default(), per_pass: TaskBudget {
            max_input_tokens: 8192, max_output_tokens: 16, max_output_bytes: 1 << 20,
            max_grammar_states: 4096, max_kv_bytes: 1 << 31 }, planning: SourcePlanningLimits::default(),
            max_model_work: per, mask_limits: MaskWorkLimits::default(), mask_visits_per_pass: 1000,
            max_mask_visits: 2000, max_result_bytes: 4 << 20 },
        request: RedactionRequest::default(), max_model_work: per.checked_add(per).unwrap().checked_add(per).unwrap(),
        max_mask_visits: 6000 }
}
fn context(n: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: n, epoch, input_line: n, byte_offset: 0 }
}
fn document(text: &str) -> BatchDocument<RedactionBatchArgs> {
    BatchDocument { id: "opaque-id".to_owned(), text: text.to_owned(), task_args: None }
}
fn error<T>(result: Result<T, BatchItemFailure>) -> BatchItemFailure {
    match result { Err(e) => e, Ok(_) => panic!("expected refusal") }
}
#[derive(Default)]
struct Control { calls: usize, stop_at: Option<usize> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        self.stop_at.filter(|&n| self.calls >= n).map(|_| DecodeCancellationKind::Deadline)
    }
}
type Trace = Rc<RefCell<Vec<&'static str>>>;
struct Guard(Trace);
impl Drop for Guard { fn drop(&mut self) { self.0.borrow_mut().push("guard"); } }
#[derive(Serialize)]
struct Body { text: String, #[serde(skip)] trace: Trace }
impl Drop for Body { fn drop(&mut self) { self.trace.borrow_mut().push("body"); } }
struct Admission { trace: Trace, drift: bool, refuse: bool }
impl Int8RedactionBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, r: Int8RedactionAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.trace.borrow_mut().push("admit");
        assert_eq!(r.mask_node_visits, 2000); assert_eq!(r.kv_reservation_bytes, 4096);
        if self.refuse { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let mut id = r.identity.clone();
        if self.drift { id.prompt_digest = Sha256Digest::of_bytes(b"wrong-admitted-seed"); }
        Ok((id, Guard(self.trace.clone())))
    }
}
fn admission() -> Admission { Admission { trace: Rc::default(), drift: false, refuse: false } }

#[test]
fn input_records_cannot_weaken_verification_or_change_key_scope() {
    for args in [r#"{"verify":false}"#, r#"{"key":"secret"}"#, r#"{"types":[]}"#,
        r#"{"actions":{}}"#, r#"{"namespace":"other"}"#] {
        let text = format!(r#"{{"id":"i","text":"private","task_args":{args}}}"#);
        assert!(serde_json::from_str::<BatchDocument<RedactionBatchArgs>>(&text).is_err());
    }
    assert!(serde_json::from_str::<BatchDocument<RedactionBatchArgs>>(r#"{"id":"i","text":"x","task_args":{}}"#).is_ok());
}
#[test]
fn preparation_preserves_exact_unicode_and_does_not_charge_native_work() {
    let mut s = RunState::default(); let c = config(); let text = "  é Alice\r\n上海\t";
    let p = prepare(&mut s, &c, document(text), &mut Control::default()).unwrap();
    assert_eq!(p.source, text); assert_eq!(s.reserved, Int8Work::default()); assert_eq!(s.masks, 0);
}
#[test]
fn oversized_source_rejects_locally_but_planning_cancellation_stops() {
    let mut c = config(); c.request.rule_budget.max_input_bytes = 1; let mut s = RunState::default();
    assert!(!error(prepare(&mut s, &c, document("é"), &mut Control::default())).stop);
    prepare(&mut s, &c, document("a"), &mut Control::default()).unwrap();
    let failure = error(prepare(&mut s, &c, document("a"), &mut Control { calls: 0, stop_at: Some(1) }));
    assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert!(s.failed);
}
#[test]
fn verification_mask_allowance_must_fit_before_any_item() {
    let mut c = config(); check_limits(&c).unwrap();
    c.detector.max_mask_visits = 1000; assert!(check_limits(&c).is_err());
    c.request.verify = false; check_limits(&c).unwrap();
    assert_eq!(masks_per_item(&c), Some(1000));
    c.detector.mask_visits_per_pass = u64::MAX; c.request.verify = true;
    assert!(check_limits(&c).is_err());
}
#[test]
fn every_native_axis_and_mask_axis_enforces_the_run_ceiling() {
    for axis in 0..6 {
        let mut c = config(); let p = c.detector.max_model_work;
        match axis { 0 => c.max_model_work.forward_positions = p.forward_positions - 1,
            1 => c.max_model_work.projected_logits = p.projected_logits - 1,
            2 => c.max_model_work.attention_pairs = p.attention_pairs - 1,
            3 => c.max_model_work.projections.dot_products = p.projections.dot_products - 1,
            4 => c.max_model_work.projections.multiply_accumulates = p.projections.multiply_accumulates - 1,
            _ => c.max_mask_visits = 1999 }
        let mut s = RunState::default(); assert!(s.begin(&c, context(1, 1)).is_err());
        assert_eq!(s.reserved, Int8Work::default()); assert_eq!(s.masks, 0); assert!(s.failed);
    }
}
#[test]
fn counter_overflow_is_atomic_and_terminal() {
    let c = config(); let mut s = RunState::default(); s.reserved.forward_positions = u64::MAX;
    let before = s.reserved;
    assert!(s.begin(&c, context(1, 1)).is_err()); assert_eq!(s.reserved, before); assert_eq!(s.masks, 0);
    let mut s = RunState::default(); s.masks = u64::MAX;
    assert!(s.begin(&c, context(1, 1)).is_err()); assert_eq!(s.reserved, Int8Work::default()); assert_eq!(s.masks, u64::MAX);
}
#[test]
fn failures_and_new_epochs_do_not_refund_both_passes() {
    let c = config(); let mut s = RunState::default(); let mut a = admission();
    for seq in 1..=3 {
        let failure = error(execute_reserved::<_, _, (), _>(&mut s, &c, context(seq, seq), 4096,
            &mut a, &mut Control::default(), |_| (Err(Int8RedactionError::WorkBudget), true)));
        assert!(!failure.stop); assert_eq!(s.masks, seq * 2000);
    }
    assert_eq!(s.reserved, c.max_model_work);
    assert!(s.begin(&c, context(4, 4)).is_err()); assert!(s.failed);
}
#[test]
fn sequence_replay_is_not_authorized_by_a_new_epoch() {
    let c = config(); let mut s = RunState::default();
    s.begin(&c, context(4, 1)).unwrap(); s.finish(&Ok::<(), BatchItemFailure>(()));
    assert_eq!(error(s.begin(&c, context(4, 2))).fault.code, BatchCode::InvalidExecution);
}
#[test]
fn admission_identity_drift_never_reaches_native_work() {
    let c = config(); let mut s = RunState::default(); let mut a = admission(); a.drift = true;
    let failure = error(execute_reserved::<_, _, (), _>(&mut s, &c, context(1, 1), 4096,
        &mut a, &mut Control::default(), |_| panic!("must not execute")));
    assert_eq!(failure.fault.code, BatchCode::Admission); assert!(s.failed);
    assert_eq!(s.reserved, c.detector.max_model_work); assert_eq!(*a.trace.borrow(), ["admit", "guard"]);
}
#[test]
fn admission_failure_consumes_reservation_and_stops_the_stream() {
    let c = config(); let mut s = RunState::default(); let mut a = admission(); a.refuse = true;
    let failure = error(execute_reserved::<_, _, (), _>(&mut s, &c, context(1, 1), 4096,
        &mut a, &mut Control::default(), |_| panic!("must not execute")));
    assert!(failure.stop); assert_eq!(s.masks, 2000); assert_eq!(s.reserved, c.detector.max_model_work);
}
#[test]
fn successful_output_keeps_its_guard_until_after_storage_drops() {
    let c = config(); let mut s = RunState::default(); let mut a = admission(); let trace = a.trace.clone();
    let out = execute_reserved(&mut s, &c, context(1, 1), 4096, &mut a, &mut Control::default(), |_| {
        trace.borrow_mut().push("execute"); (Ok(Body { text: "safe".to_owned(), trace: trace.clone() }), true)
    }).unwrap();
    assert_eq!(*trace.borrow(), ["admit", "execute"]);
    assert!(serde_json::to_string(&out).unwrap().contains("safe"));
    drop(out); assert_eq!(*trace.borrow(), ["admit", "execute", "body", "guard"]); assert!(!s.failed);
}
#[test]
fn late_cancellation_suppresses_output_and_drops_storage_before_guard() {
    let c = config(); let mut s = RunState::default(); let mut a = admission(); let trace = a.trace.clone();
    let failure = error(execute_reserved(&mut s, &c, context(1, 1), 4096, &mut a,
        &mut Control { calls: 0, stop_at: Some(3) }, |_| {
            (Ok(Body { text: "must-not-escape".to_owned(), trace: trace.clone() }), true)
        }));
    assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline)); assert!(s.failed);
    assert_eq!(*trace.borrow(), ["admit", "body", "guard"]);
}
#[test]
fn poisoned_native_state_makes_even_a_local_refusal_terminal() {
    let c = config(); let mut s = RunState::default(); let mut a = admission();
    let failure = error(execute_reserved::<_, _, (), _>(&mut s, &c, context(1, 1), 4096,
        &mut a, &mut Control::default(), |_| (Err(RedactError::OutputBudget.into()), false)));
    assert!(failure.stop); assert!(s.failed); assert_eq!(*a.trace.borrow(), ["admit", "guard"]);
}
#[test]
fn unwinding_cannot_reuse_uncharged_native_work() {
    let c = config(); let mut s = RunState::default(); let mut a = admission();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = execute_reserved::<_, _, (), _>(&mut s, &c, context(1, 1), 4096,
            &mut a, &mut Control::default(), |_| panic!("synthetic native unwind"));
    }));
    assert!(panic.is_err()); assert!(s.failed); assert_eq!(s.reserved, c.detector.max_model_work);
    assert_eq!(*a.trace.borrow(), ["admit", "guard"]);
}
#[test]
fn typed_cancellation_survives_poisoned_native_completion() {
    let c = config(); let mut s = RunState::default(); let mut a = admission();
    let failure = error(execute_reserved::<_, _, (), _>(&mut s, &c, context(1, 1), 4096,
        &mut a, &mut Control::default(), |_| (Err(Int8RedactionError::Cancelled(DecodeCancellationKind::Deadline)), false)));
    assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline)); assert!(failure.stop);
}
#[test]
fn fixed_full_digest_context_is_order_independent_and_checks_saved_key() {
    use super::super::{PiiKind, pseudonym::PseudonymKey};
    let key = PseudonymKey::from_bytes(&[17; 32], "key-a").unwrap();
    let other = PseudonymKey::from_bytes(&[18; 32], "key-a").unwrap();
    let saved = key.commitment(); let p = Pseudonyms::full256(&key, "corpus", Some(&saved)).unwrap();
    let alice = p.pseudonym(PiiKind::Person, "Alice").unwrap();
    let _ = p.pseudonym(PiiKind::Person, "Bob").unwrap();
    assert_eq!(alice, p.pseudonym(PiiKind::Person, "Alice").unwrap());
    assert!(Pseudonyms::full256(&other, "corpus", Some(&saved)).is_err());
    assert_ne!(alice, Pseudonyms::full256(&key, "other-corpus", Some(&saved)).unwrap().pseudonym(PiiKind::Person, "Alice").unwrap());
}

// Feed the REAL NDJSON runner through the same private transaction boundary.
// This harness uses rules-only edits solely as transport fixtures, not NER.
struct Harness { config: Int8RedactionBatchConfig, state: RunState, admission: Admission }
impl BatchProcessor for Harness {
    type Args = RedactionBatchArgs;
    type Prepared = PreparedRedaction;
    type Output = GuardedOutput<Body, Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        panic!("runner must lend the existing control")
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, doc: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> { prepare(&mut self.state, &self.config, doc, control) }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        let w = self.config.detector.max_model_work;
        BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("runner must provide its actual request sequence")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, p: Self::Prepared, context: BatchRequestContext, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        let trace = self.admission.trace.clone();
        execute_reserved(&mut self.state, &self.config, context, 4096, &mut self.admission, control, |_| {
            trace.borrow_mut().push("execute");
            let edited = super::super::redact_rules(&p.source, &self.config.request, None).unwrap();
            (Ok(Body { text: edited.text().to_owned(), trace }), true)
        })
    }
}
fn harness() -> Harness { Harness { config: config(), state: RunState::default(), admission: admission() } }
fn transport() -> crate::batch::BatchLimits {
    crate::batch::BatchLimits { max_work: BatchWork { forward_positions: u64::MAX, projected_logits: u64::MAX },
        ..crate::batch::BatchLimits::default() }
}
#[test]
fn real_runner_flush_reuses_scope_not_native_allowance() {
    let mut h = harness(); h.config.max_model_work = h.config.detector.max_model_work;
    let input = b"{\"id\":\"same\",\"text\":\"a@example.org\"}\n{\"flush\":true}\n{\"id\":\"same\",\"text\":\"b@example.org\"}\n";
    let mut output = Vec::new();
    let failure = crate::batch::run_ndjson(&mut std::io::Cursor::new(input), &mut output, &mut h,
        transport(), &mut Control::default()).unwrap_err();
    assert_eq!(failure.fault.code, BatchCode::WorkLimit); assert_eq!(failure.summary.succeeded, 1);
    assert_eq!(h.admission.trace.borrow().iter().filter(|&&s| s == "execute").count(), 1);
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("[redacted:email]")); assert!(!text.contains("a@example.org")); assert!(!text.contains("b@example.org"));
}
struct Writer { trace: Trace, bytes: Vec<u8>, fail_document: bool }
impl std::io::Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.trace.borrow_mut().push("write");
        if self.fail_document && bytes.windows(b"\"event\":\"doc\"".len()).any(|w| w == b"\"event\":\"doc\"") {
            return Err(std::io::Error::other("synthetic sink failure"));
        }
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { self.trace.borrow_mut().push("flush"); Ok(()) }
}
#[test]
fn real_runner_flushes_each_output_before_dropping_its_guard() {
    let mut h = harness(); let trace = h.admission.trace.clone();
    let mut writer = Writer { trace: trace.clone(), bytes: Vec::new(), fail_document: false };
    let input = b"{\"id\":\"one\",\"text\":\"a@example.org\"}\n{\"id\":\"two\",\"text\":\"b@example.org\"}\n";
    let summary = crate::batch::run_ndjson(&mut std::io::Cursor::new(input), &mut writer, &mut h,
        transport(), &mut Control::default()).unwrap();
    assert_eq!(summary.succeeded, 2); assert_eq!(h.state.masks, 4000);
    let events = trace.borrow();
    let guards: Vec<_> = events.iter().enumerate().filter_map(|(i, &s)| (s == "guard").then_some(i)).collect();
    assert_eq!(guards.len(), 2);
    for i in guards { assert_eq!(events[i - 1], "body"); assert_eq!(events[i - 2], "flush"); }
}
#[test]
fn real_runner_sink_failure_drains_output_and_does_not_execute_next_document() {
    let mut h = harness(); let trace = h.admission.trace.clone();
    let mut writer = Writer { trace: trace.clone(), bytes: Vec::new(), fail_document: true };
    let input = b"{\"id\":\"one\",\"text\":\"a@example.org\"}\n{\"id\":\"two\",\"text\":\"b@example.org\"}\n";
    let failure = crate::batch::run_ndjson(&mut std::io::Cursor::new(input), &mut writer, &mut h,
        transport(), &mut Control::default()).unwrap_err();
    assert_eq!(failure.fault.code, BatchCode::OutputIo);
    let events = trace.borrow(); assert_eq!(events.iter().filter(|&&s| s == "execute").count(), 1);
    assert_eq!(&events[events.len() - 2..], ["body", "guard"]);
}
