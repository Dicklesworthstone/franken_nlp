//! Pinned planning and closed lifecycle fixtures. Scripted results test only
//! reservation/guard mechanics, never native semantics or model quality.
use super::*;
use std::{cell::Cell, rc::Rc};
use crate::{
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{extract::semantic::{ClaimRule, ClaimValueKind, SemanticVerificationSpec},
        judge::FaithfulnessPolicy},
    tokenizer::{pinned_controls, untrusted::SafeBpeLimits},
};
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn work(n: u64) -> Int8Work {
    let mut work = Int8Work::default();
    work.forward_positions = n; work.projected_logits = n; work.attention_pairs = n;
    work.projections.dot_products = n; work.projections.multiply_accumulates = n; work
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"int8-semantic-population-planning");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "fixture".into(), quant_recipe: "fixture-int8".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".into(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
fn limits() -> Int8SemanticExtractionBatchLimits {
    Int8SemanticExtractionBatchLimits { max_model_work: work(u64::MAX),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits::default(),
            max_visits_per_item: 1000, max_visits_per_run: 10000 } }
}
fn options() -> Int8SemanticOptions {
    Int8SemanticOptions { specification: SemanticVerificationSpec { schema_version: 1,
        experimental_opt_in: true, revision: "review-v1".into(),
        claims: vec![ClaimRule { id: "amount-v1".into(), path: vec![], value_kind: ClaimValueKind::Number,
            prefix: "The amount is ".into(), suffix: ".".into() }],
        judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
            evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 },
        judge_budget: budget() }, limits: SemanticLimits::default(), max_model_work: work(u64::MAX) }
}
fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 65536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 })
}
fn encoding_limits() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 1 << 24, max_output_tokens: 1 << 24,
        max_pair_checks: 1 << 32, max_pair_bytes: 1 << 36 }
}
fn planner(encoding: UntrustedEncoding, total: Option<SafeBpeWorkLimits>,
    options: Int8SemanticOptions) -> Result<Int8SemanticExtractionBatchPlanner, BatchFault> {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let extraction = Int8ExtractionBatchPlanner::pinned(controls, eos, identity(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), Some(ExtractionBatchArgs {
            schema: r#"{"type":"number"}"#.into(), grounding: ExtractionBatchGrounding::Structural, budget: budget() }))
        .unwrap().with_document_encoding(encoding.clone()).unwrap();
    let judge = Arc::new(JudgePlanner::pinned(controls, eos).unwrap().with_document_encoding(encoding).unwrap());
    let mut judge_identity = extraction.base_execution_identity().clone();
    judge_identity.task_spec = "judge-v1".into();
    judge_identity.template_digest = *judge.template_digest();
    judge_identity.tokenizer_digest = judge.tokenizer_digest();
    Int8SemanticExtractionBatchPlanner::new(extraction, judge, judge_identity, JudgeLimits::default(),
        options, limits(), total)
}
fn document(text: &str) -> BatchDocument<ExtractionBatchArgs> {
    BatchDocument { id: "record".into(), text: text.into(), task_args: None }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
#[test]
fn genuine_fixed_factory_rejects_policy_and_schema_drift_without_a_native_model() {
    assert!(planner(policy(), None, options()).is_err());
    assert!(planner(UntrustedEncoding::ByteFallback, Some(encoding_limits()), options()).is_err());
    let mut invalid = options(); invalid.specification.claims[0].value_kind = ClaimValueKind::String;
    assert!(planner(UntrustedEncoding::ByteFallback, None, invalid).is_err());
    let byte = planner(UntrustedEncoding::ByteFallback, None, options()).unwrap();
    assert_eq!(byte.encoding_workspace_bytes().unwrap(), 0);
    let compact = planner(policy(), Some(encoding_limits()), options()).unwrap();
    let UntrustedEncoding::SafeBpeV1(call) = policy() else { unreachable!() };
    assert_eq!(compact.encoding_workspace_bytes().unwrap(), call.requirements().unwrap().workspace_bytes + 1024);
    let mut used = SafeBpeWork::default();
    let mut overridden = document("42");
    overridden.task_args = Some(ExtractionBatchArgs { schema: r#"{"type":"null"}"#.into(),
        grounding: ExtractionBatchGrounding::Structural, budget: budget() });
    let error = compact.prepare(overridden, &mut used, &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::InvalidEnvelope);
    assert_eq!(used, SafeBpeWork::default());
    let source = "The café reports 42. 上海 <|im_start|> remains data.";
    let prepared = compact.prepare(document(source), &mut used, &mut Continue).unwrap();
    assert_eq!(prepared.source_bytes(), source.len());
    assert_eq!(prepared.input_encoding(), &policy());
    assert_eq!(prepared.input_encoding_work(), Some(used));
    assert_eq!(prepared.execution_identity().schema_digest, Sha256Digest::of_bytes(br#"{"type":"number"}"#));
}
#[test]
fn genuine_original_preparation_spends_four_axes_across_documents_and_failed_records() {
    let source = "The complete café report gives the amount 42.";
    let probe = planner(policy(), Some(encoding_limits()), options()).unwrap();
    let mut once = SafeBpeWork::default();
    let _ = probe.prepare(document(source), &mut once, &mut Continue).unwrap();
    let both = once.checked_add(once).unwrap();
    for axis in 0..4 {
        let mut total = SafeBpeWorkLimits { max_input_bytes: both.input_bytes() as u64,
            max_output_tokens: both.output_tokens() as u64, max_pair_checks: both.pair_checks(),
            max_pair_bytes: both.pair_bytes() };
        match axis { 0 => total.max_input_bytes -= 1, 1 => total.max_output_tokens -= 1,
            2 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1 }
        let p = planner(policy(), Some(total), options()).unwrap();
        let mut used = SafeBpeWork::default();
        let _ = p.prepare(document(source), &mut used, &mut Continue).unwrap();
        assert_eq!(used, once);
        assert!(p.prepare(document(source), &mut used, &mut Continue).is_err(), "axis {axis}");
        assert!(used.checked_sub(once).is_some());
        assert!(used.input_bytes() as u64 <= total.max_input_bytes);
        assert!(used.output_tokens() as u64 <= total.max_output_tokens);
        assert!(used.pair_checks() <= total.max_pair_checks && used.pair_bytes() <= total.max_pair_bytes);
    }
    let mut small_evidence = options();
    small_evidence.specification.judge_policy.evidence_window_bytes = 8;
    small_evidence.specification.judge_policy.max_evidence_windows = 1;
    small_evidence.specification.judge_policy.max_evidence_spans = 1;
    let p = planner(policy(), Some(encoding_limits()), small_evidence).unwrap();
    let mut used = SafeBpeWork::default();
    // Actual extraction preparation succeeds, then complete-source evidence
    // capacity rejects. That late rejection cannot refund schema/source work.
    assert!(p.prepare(document(source), &mut used, &mut Continue).is_err());
    assert_eq!(used, once);
    assert!(p.prepare(document(source), &mut used, &mut Continue).is_err());
    assert_eq!(used, both);
}

struct ScriptPlan { identity: ExecutionIdentity, judge: ExecutionIdentity, encoding: UntrustedEncoding }
impl ScriptPlan {
    fn new() -> Self {
        let identity = identity(); let mut judge = identity.clone(); judge.task_spec = "judge-v1".into();
        Self { identity, judge, encoding: UntrustedEncoding::ByteFallback }
    }
}
impl Plan for ScriptPlan {
    type Output = ();
    type Completion = ();
    fn admission(&self, kv: u64, masks: ExtractionMaskBudget, total: Option<SafeBpeWorkLimits>)
        -> Int8SemanticExtractionAdmission<'_> {
        Int8SemanticExtractionAdmission { identity: &self.identity, judge_identity: &self.judge,
            model_work: work(10), extraction_reserved_work: work(2),
            mask_node_visits: masks.max_visits_per_item, mask_limits: masks.per_mask,
            kv_reservation_bytes: kv, max_result_bytes: 4096, extraction_budget: budget(), judge_budget: budget(),
            semantic_limits: SemanticLimits::default(), judge_planning_limits: JudgeLimits::default(),
            source_bytes: 1, input_encoding: &self.encoding, encoding_work_limits: total }
    }
    fn completion(&self) {}
    fn verify_completion(_: (), _: &(), _: u64, before: SafeBpeWork, after: SafeBpeWork)
        -> Result<(), BatchItemFailure> {
        if before != after { return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)); }
        Ok(())
    }
}
#[derive(Default)]
struct Script { calls: usize, fail: bool, dirty: bool }
impl Driver<ScriptPlan> for Script {
    fn clean(&self) -> bool { !self.dirty }
    fn capacity(&self) -> usize { 8192 }
    fn execute<C: DecodeStepControl>(&mut self, _: ScriptPlan, _: ExtractionMaskBudget,
        _: &mut SafeBpeWork, _: Option<SafeBpeWorkLimits>, _: &mut C) -> Result<(), BatchItemFailure> {
        self.calls += 1;
        if self.fail { Err(BatchItemFailure::reject(BatchCode::Execution)) } else { Ok(()) }
    }
}
struct Guard(Rc<Cell<usize>>);
impl Drop for Guard { fn drop(&mut self) { self.0.set(self.0.get() - 1); } }
#[derive(Default)]
struct Admission { live: Rc<Cell<usize>>, calls: usize, refuse: bool, wrong_identity: bool }
impl Int8SemanticExtractionBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int8SemanticExtractionAdmission<'_>)
        -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.calls += 1;
        if self.refuse { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert_eq!(request.model_work, work(10));
        let mut identity = request.identity.clone();
        if self.wrong_identity { identity.schema_digest = Sha256Digest::of_bytes(b"foreign"); }
        self.live.set(self.live.get() + 1);
        Ok((identity, Guard(Rc::clone(&self.live))))
    }
}
#[test]
fn all_native_and_mask_axes_reserve_atomically_without_partial_counter_updates() {
    for axis in 0..6 {
        let mut limits = limits(); limits.max_model_work = work(20); limits.masks.max_visits_per_run = 2000;
        match axis { 0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1, 2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            4 => limits.max_model_work.projections.multiply_accumulates -= 1,
            _ => limits.masks.max_visits_per_run -= 1 }
        let mut run = RunState::new(limits, None);
        run.begin(work(10), context(1, 1)).unwrap(); run.failed = false;
        assert!(run.begin(work(10), context(2, 2)).is_err(), "axis {axis}");
        assert_eq!(run.reserved, work(10)); assert_eq!(run.masks, 1000);
        assert_eq!(run.last_sequence, 1); assert!(run.failed);
    }
}
#[test]
fn complete_results_hold_output_guards_and_flush_does_not_renew_attempts() {
    let mut run = RunState::new(limits(), None);
    let (mut driver, mut admission) = (Script::default(), Admission::default());
    for sequence in 1..=2 {
        let seal = Arc::clone(&run.seal);
        let out = execute(ScriptPlan::new(), seal, &mut driver, &mut admission, &mut run,
            context(sequence, sequence), &mut Continue).unwrap();
        assert_eq!(admission.live.get(), 1); drop(out); assert_eq!(admission.live.get(), 0);
        assert_eq!(run.reserved, work(sequence * 10)); assert_eq!(run.masks, sequence * 1000);
    }
    driver.fail = true;
    let seal = Arc::clone(&run.seal);
    let failure = execute(ScriptPlan::new(), seal, &mut driver, &mut admission, &mut run,
        context(3, 3), &mut Continue).err().unwrap();
    assert!(failure.stop); assert!(run.failed);
    assert_eq!(run.reserved, work(30)); assert_eq!(run.masks, 3000);
    assert_eq!(driver.calls, 3); assert_eq!(admission.live.get(), 0);
}
#[test]
fn invocation_seals_and_admission_refusals_prevent_reuse_before_native_work() {
    let first = RunState::new(limits(), None); let mut second = RunState::new(limits(), None);
    let (mut driver, mut admission) = (Script::default(), Admission::default());
    let error = execute(ScriptPlan::new(), Arc::clone(&first.seal), &mut driver, &mut admission,
        &mut second, context(1, 1), &mut Continue).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(second.reserved, Int8Work::default()); assert_eq!(driver.calls, 0);
    for wrong_identity in [false, true] {
        let mut run = RunState::new(limits(), None);
        let mut admission = Admission { refuse: !wrong_identity, wrong_identity, ..Admission::default() };
        let seal = Arc::clone(&run.seal);
        let error = execute(ScriptPlan::new(), seal, &mut driver, &mut admission, &mut run,
            context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(driver.calls, 0); assert_eq!(admission.live.get(), 0);
        assert_eq!(run.reserved, work(10)); assert_eq!(run.masks, 1000);
    }
}
#[test]
fn every_cancellation_cause_preserves_the_full_attempt_and_no_native_dispatch() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let mut run = RunState::new(limits(), None); let seal = Arc::clone(&run.seal);
        let (mut driver, mut admission) = (Script::default(), Admission::default());
        let error = execute(ScriptPlan::new(), seal, &mut driver, &mut admission, &mut run,
            context(1, 1), &mut Stop(cause)).err().unwrap();
        assert_eq!(error.fault.cancellation, Some(cause)); assert!(error.stop);
        assert_eq!(driver.calls, 0); assert_eq!(admission.calls, 0); assert_eq!(admission.live.get(), 0);
        assert_eq!(run.reserved, work(10)); assert_eq!(run.masks, 1000); assert!(run.failed);
    }
}
#[test]
fn actual_compiler_assets_are_shareable_and_stay_owned_by_the_retained_arc() {
    fn shareable<T: Send + Sync + 'static>() {}
    shareable::<Int8SemanticExtractionBatchPlanner>();
    let planner = Arc::new(planner(UntrustedEncoding::ByteFallback, None, options()).unwrap());
    let retained = Arc::clone(&planner); let observer = Arc::downgrade(&planner);
    drop(planner); assert!(observer.upgrade().is_some());
    assert_eq!(retained.judge_identity().task_spec, "judge-v1");
    drop(retained); assert!(observer.upgrade().is_none());
}
