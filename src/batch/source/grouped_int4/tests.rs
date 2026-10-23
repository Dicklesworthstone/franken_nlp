//! Real pinned planning, private synthetic execution for lifecycle injection.
//! These are not model-inference, semantic-quality or performance receipts.
use super::*;
use std::{cell::Cell, io::{self, Cursor}, rc::Rc};
use crate::{artifact::converter::PINNED_REVISION, execution_identity::Sha256Digest,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{answer::{AnswerResult, AnswerStatus, AnswerCalibration}, extract::ExtractionGrounding, ir::ScoreSpace,
        keyphrases::{KeyphraseResult, KEYPHRASES_RANKING}, summarize::{CitationGuarantee, SummarySemanticSupport}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls}};

fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap()
}
fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 128,
    max_output_bytes: 65536, max_grammar_states: 8192, max_kv_bytes: 1 << 30 } }
fn identity(p: &SourceTaskPlanner, kind: BuiltInTask) -> ExecutionIdentity {
    identity_for_width(p, kind, 32)
}
fn identity_for_width(p: &SourceTaskPlanner, kind: BuiltInTask, width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"int4-source-corpus-fixture");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
    )).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: kind.spec().identity(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
        decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn args(kind: BuiltInTask) -> SourceBatchArgs {
    match kind {
        BuiltInTask::Ner => SourceBatchArgs::Ner { options: NerOptions { max_entities: 4, max_mention_scalars: 32,
            ..Default::default() }, budget: budget() },
        BuiltInTask::Keyphrases => SourceBatchArgs::Keyphrases { options: KeyphraseOptions { max_phrases: 4, max_phrase_scalars: 32 }, budget: budget() },
        BuiltInTask::Summarize => SourceBatchArgs::Summarize { options: SummaryOptions { max_bullets: 2,
            max_bullet_scalars: 64, max_citations_per_bullet: 2, max_quote_scalars: 32 }, budget: budget() },
        BuiltInTask::Answer => SourceBatchArgs::Answer { passages: vec![AnswerPassage { id: "p1".to_owned(), text: "Alice met Bob.".to_owned() }],
            options: AnswerOptions { max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: budget() },
        _ => unreachable!(),
    }
}
fn compiler(p: &SourceTaskPlanner, kind: BuiltInTask) -> Int4SourceBatchPlanner<'_> {
    Int4SourceBatchPlanner::new(p, identity(p, kind), budget(), SourcePlanningLimits::default(), Some(args(kind)), 32).unwrap()
}
fn document() -> BatchDocument<SourceBatchArgs> {
    BatchDocument { id: "item".to_owned(), text: "PRIVATE_SOURCE Alice <think> 上海".to_owned(), task_args: None }
}
fn limits() -> Int4SourceBatchLimits {
    Int4SourceBatchLimits { max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
        attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        masks: SourceMaskBudget { per_mask: crate::grammar::mask::MaskWorkLimits {
            max_trie_node_visits: 10000, checkpoint_interval_nodes: 64 }, max_visits_per_item: 100000, max_visits_per_run: u64::MAX } }
}
fn context(seq: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: seq, byte_offset: 0 }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }

#[test]
fn all_four_batch_plans_equal_the_direct_source_task_plans() {
    let p = planner();
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let c = compiler(&p, kind); let plan = c.prepare_with_control(document(), &mut Continue).unwrap();
        let id = identity(&p, kind); let cx = PlanContext::new(&id, budget()).unwrap();
        let direct = p.plan_int4_with_control(&args(kind).into_request(document().text), &cx,
            SourcePlanningLimits::default(), 32, &mut Continue).unwrap();
        plan.verify_identity(direct.execution_identity()).unwrap();
        assert_eq!(plan.planned_work(), direct.planned_work());
        assert_eq!(plan.planned_work(), constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), 128).unwrap());
        assert_eq!(plan.planned_work().projected_logits, 128 * NANBEIGE_VOCAB_SIZE as u64);
        assert!(plan.planned_work().projections.multiply_accumulates > plan.planned_work().projected_logits * 3072);
    }
}

#[test]
fn group_partition_changes_dot_accounting_without_rewriting_the_source_prompt() {
    let p = planner();
    let plans: Vec<_> = [16, 32, 127].into_iter().map(|width| {
        let compiler = Int4SourceBatchPlanner::new(&p, identity_for_width(&p, BuiltInTask::Keyphrases, width),
            budget(), SourcePlanningLimits::default(), Some(args(BuiltInTask::Keyphrases)), width).unwrap();
        assert_eq!(compiler.group_width(), width);
        assert_eq!(compiler.base_execution_identity().numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
        compiler.prepare_with_control(document(), &mut Continue).unwrap()
    }).collect();
    for pair in plans.windows(2) {
        assert_eq!(pair[0].execution_identity().prompt_digest, pair[1].execution_identity().prompt_digest);
        assert_ne!(pair[0].execution_identity().decision_policy_digest, pair[1].execution_identity().decision_policy_digest);
        assert_eq!(pair[0].planned_work().forward_positions, pair[1].planned_work().forward_positions);
        assert_eq!(pair[0].planned_work().projected_logits, pair[1].planned_work().projected_logits);
        assert_eq!(pair[0].planned_work().projections.multiply_accumulates, pair[1].planned_work().projections.multiply_accumulates);
        assert!(pair[0].planned_work().projections.dot_products > pair[1].planned_work().projections.dot_products);
        assert!(pair[0].verify_identity(pair[1].execution_identity()).is_err());
    }
    for plan in plans {
        assert_eq!(plan.planned_work(), constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), 128).unwrap());
    }
}
#[test]
fn wrong_profile_backend_or_pinned_assets_are_refused_before_compilation() {
    let p = planner();
    for axis in 0..9 {
        let mut id = identity(&p, BuiltInTask::Keyphrases);
        match axis { 0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.backend_semantic_version.push('x'), 2 => id.tokenizer_digest = Sha256Digest::of_bytes(b"bad"),
            3 => id.template_digest = Sha256Digest::of_bytes(b"bad"), 4 => id.task_spec = "extract-v1".to_owned(),
            5 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            6 => id.artifact_format = "fnlpq-v1".to_owned(), 7 => id.packing_set_digest = Sha256Digest::of_bytes(b"bad"),
            _ => id.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=16") }
        assert!(Int4SourceBatchPlanner::new(&p, id, budget(), SourcePlanningLimits::default(), None, 32).is_err());
    }
    assert!(SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Keyphrases), budget(), SourcePlanningLimits::default(), None).is_err());
    for width in [0, 16, 127, usize::MAX] {
        assert!(Int4SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Keyphrases), budget(),
            SourcePlanningLimits::default(), None, width).is_err());
    }
}
#[test]
fn defaults_are_bounded_before_cloning_and_records_cannot_switch_task() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases);
    let baseline = c.prepare_with_control(document(), &mut Continue).unwrap();
    let mut doc = document(); doc.task_args = Some(args(BuiltInTask::Answer));
    assert!(c.prepare_with_control(doc, &mut Continue).is_err());
    let mut doc = document(); doc.task_args = Some(SourceBatchArgs::Keyphrases {
        options: KeyphraseOptions { max_phrases: 1, max_phrase_scalars: 32 }, budget: budget() });
    let changed = c.prepare_with_control(doc, &mut Continue).unwrap();
    assert!(baseline.verify_identity(changed.execution_identity()).is_err());
    baseline.verify_identity(c.prepare_with_control(document(), &mut Continue).unwrap().execution_identity()).unwrap();
    let empty = Int4SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Keyphrases), budget(), SourcePlanningLimits::default(), None, 32).unwrap();
    assert!(empty.prepare_with_control(document(), &mut Continue).is_err());
    let mut oversized = args(BuiltInTask::Answer);
    if let SourceBatchArgs::Answer { passages, .. } = &mut oversized { passages[0].text = "x".repeat(MAX_SOURCE_ARGUMENT_BYTES); }
    assert!(Int4SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Answer), budget(), SourcePlanningLimits::default(), Some(oversized), 32).is_err());
}
#[test]
fn qa_question_and_passages_both_affect_identity_without_becoming_interchangeable() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Answer);
    let first = c.prepare_with_control(document(), &mut Continue).unwrap();
    let mut question = document(); question.text.push('?');
    assert!(first.verify_identity(c.prepare_with_control(question, &mut Continue).unwrap().execution_identity()).is_err());
    let mut changed = args(BuiltInTask::Answer);
    if let SourceBatchArgs::Answer { passages, .. } = &mut changed { passages[0].text.push('!'); }
    let mut doc = document(); doc.task_args = Some(changed);
    assert!(first.verify_identity(c.prepare_with_control(doc, &mut Continue).unwrap().execution_identity()).is_err());
    let mut missing = args(BuiltInTask::Answer);
    if let SourceBatchArgs::Answer { passages, .. } = &mut missing { passages.clear(); }
    let mut doc = document(); doc.task_args = Some(missing);
    let error = c.prepare_with_control(doc, &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Planning);
}
#[test]
fn planning_observes_the_original_control_instead_of_a_refreshed_budget() {
    struct CancelAt { calls: usize, at: usize }
    impl DecodeStepControl for CancelAt {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls >= self.at).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases);
    for at in [1, 3, 4] {
        let mut control = CancelAt { calls: 0, at };
        let error = c.prepare_with_control(document(), &mut control).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
        assert_eq!(control.calls, at);
    }
}
#[test]
fn all_work_axes_and_masks_are_reserved_atomically_with_overflow_checks() {
    let work = Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } };
    for axis in 0..6 {
        let mut l = limits(); l.max_model_work = work;
        match axis { 0 => l.max_model_work.forward_positions -= 1, 1 => l.max_model_work.projected_logits -= 1,
            2 => l.max_model_work.attention_pairs -= 1, 3 => l.max_model_work.projections.dot_products -= 1,
            4 => l.max_model_work.projections.multiply_accumulates -= 1, _ => l.masks.max_visits_per_run = 0 }
        let mut run = RunState::new(l); let error = run.begin(work, context(1, 1)).unwrap_err();
        assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(run.reserved, Int4Work::default()); assert_eq!(run.mask_visits, 0);
        let mut run = RunState::new(limits());
        match axis { 0 => run.reserved.forward_positions = u64::MAX, 1 => run.reserved.projected_logits = u64::MAX,
            2 => run.reserved.attention_pairs = u64::MAX, 3 => run.reserved.projections.dot_products = u64::MAX,
            4 => run.reserved.projections.multiply_accumulates = u64::MAX, _ => run.mask_visits = u64::MAX }
        let before = (run.reserved, run.mask_visits);
        assert!(run.begin(work, context(1, 1)).is_err()); assert_eq!((run.reserved, run.mask_visits), before);
    }
}
#[test]
fn invalid_limits_and_delivery_coordinates_cannot_reopen_execution() {
    for axis in 0..4 {
        let mut l = limits();
        match axis { 0 => l.masks.max_visits_per_item = 0, 1 => l.masks.per_mask.checkpoint_interval_nodes = 0,
            2 => l.masks.max_visits_per_run = 0, _ => l.max_model_work.attention_pairs = 0 }
        assert!(validate_limits(l).is_err());
    }
    let work = Int4Work::for_sequence(32, 0, 1, 1).unwrap();
    for ctx in [context(0, 1), context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut run = RunState::new(limits()); assert!(run.begin(work, ctx).is_err()); assert!(run.failed);
    }
    let mut run = RunState::new(limits()); run.begin(work, context(4, 1)).unwrap(); run.failed = false;
    assert!(run.begin(work, context(4, 2)).is_err()); assert!(run.failed);
}

#[derive(Default)]
struct State { live: Cell<bool>, running: Cell<bool>, drops: Cell<usize>, calls: Cell<usize>, admissions: Cell<usize> }
struct Guard(Rc<State>);
impl Drop for Guard {
    fn drop(&mut self) { assert!(!self.0.running.get()); assert!(self.0.live.replace(false)); self.0.drops.set(self.0.drops.get() + 1); }
}
struct Admission { state: Rc<State>, reject: bool, wrong: bool }
impl Int4SourceBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, req: Int4SourceAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.state.admissions.set(self.state.admissions.get() + 1);
        assert_eq!(req.kv_reservation_bytes, 4096 * KV_BYTES_PER_TOKEN as u64);
        assert_eq!(req.mask_node_visits, limits().masks.max_visits_per_item);
        assert!(req.model_work.attention_pairs > 0); assert_eq!(req.max_result_bytes, budget().max_output_bytes);
        if self.reject { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert!(!self.state.live.replace(true)); let mut id = req.identity.clone();
        if self.wrong { id.source_revision.push('x'); }
        Ok((id, Guard(Rc::clone(&self.state))))
    }
}
#[derive(Clone, Copy)]
enum Mode { Success, Cancel, Panic, Dirty, Oversize, WrongWork }
struct Engine { state: Rc<State>, mode: Mode, clean: bool, width: usize }
struct Physical(Rc<State>);
impl Drop for Physical { fn drop(&mut self) { self.0.running.set(false); } }
fn completion(plan: &PreparedInt4SourceTask) -> Int4SourceTaskRun {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let answer = plan.execution_identity().task_spec == "answer-v1";
    let json: &[u8] = if answer { br#"{"answer":"","answerable":false,"citations":[]}"# } else { b"[]" };
    let mut tokens = tokenizer.tokenizer().encode_byte_fallback_only(json).unwrap();
    tokens.push(tokenizer.eos_token_id().unwrap());
    let w = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), tokens.len()).unwrap();
    let result = if answer {
        SourceTaskResult::Answer(AnswerResult { schema_version: 1, task_spec_version: "answer-v1".to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), status: AnswerStatus::Abstained, answerable: false,
            answer: None, citations: Vec::new(), calibration: AnswerCalibration::Uncalibrated,
            citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
            score_space: ScoreSpace::NotComputed, untrusted_fields: ["answer".to_owned(), "citations".to_owned()],
            generated_token_ids: tokens, forward_positions: w.forward_positions,
            projected_logits: w.projected_logits, mask_node_visit_charge: 5 })
    } else {
        SourceTaskResult::Keyphrases(KeyphraseResult { schema_version: 1, task_spec_version: "keyphrases-v1".to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), ranking_policy: KEYPHRASES_RANKING.to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership, phrases: vec![],
            generated_token_ids: tokens, forward_positions: w.forward_positions, projected_logits: w.projected_logits, mask_node_visit_charge: 5 })
    };
    Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: plan.group_width(), model_work: w, result }
}
impl Driver for Engine {
    fn capacity(&self) -> usize { 4096 }
    fn group_width(&self) -> usize { self.width }
    fn clean(&self) -> bool { self.clean }
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt4SourceTask, id: &ExecutionIdentity,
        _: Int4JsonBudget, _: &mut C) -> Result<Int4SourceTaskRun, Int4SourceError> {
        plan.verify_identity(id)?; assert!(self.state.live.get()); assert!(!self.state.running.replace(true));
        let _physical = Physical(Rc::clone(&self.state)); self.state.calls.set(self.state.calls.get() + 1);
        match self.mode {
            Mode::Panic => panic!("private source lifecycle fixture"),
            Mode::Cancel => { self.clean = false; return Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
            Mode::Oversize => { self.clean = false; return Err(Int4SourceError::Planning(SourcePlanningError::OutputBudget)); }
            Mode::Dirty => self.clean = false, _ => {}
        }
        let mut result = completion(plan);
        if matches!(self.mode, Mode::WrongWork) { result.model_work.projections.multiply_accumulates += 1; }
        Ok(result)
    }
}
fn host(state: &Rc<State>, mode: Mode) -> (Engine, Admission, RunState) {
    (Engine { state: Rc::clone(state), mode, clean: true, width: 32 },
        Admission { state: Rc::clone(state), reject: false, wrong: false }, RunState::new(limits()))
}
#[test]
fn guard_survives_native_work_and_serialization_and_complete_work_is_retained() {
    let p = planner(); let plan = compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).unwrap();
    let work = plan.planned_work(); let state = Rc::new(State::default());
    let (mut driver, mut admission, mut run) = host(&state, Mode::Success);
    let result = execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).unwrap();
    assert!(state.live.get()); assert!(!state.running.get()); assert!(!run.failed);
    assert_eq!(run.reserved, work); assert_eq!(run.mask_visits, limits().masks.max_visits_per_item);
    let bytes = canonjson::canonical_string(&result).unwrap();
    assert!(!bytes.contains("PRIVATE_SOURCE") && !bytes.contains("prompt_digest") && !bytes.contains("guard"));
    assert!(state.live.get()); drop(result); assert_eq!(state.drops.get(), 1);
}
#[test]
fn identity_substitution_never_reaches_native_execution() {
    let p = planner(); let state = Rc::new(State::default()); let (mut driver, mut admission, mut run) = host(&state, Mode::Success);
    admission.wrong = true;
    let plan = compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).unwrap();
    let error = execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(state.calls.get(), 0); assert_eq!(state.drops.get(), 1);
}
#[test]
fn failed_admission_never_refunds_model_or_mask_budgets_across_flush_epochs() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases); let state = Rc::new(State::default());
    let (mut driver, mut admission, mut run) = host(&state, Mode::Success); admission.reject = true;
    let plan = c.prepare_with_control(document(), &mut Continue).unwrap(); let work = plan.planned_work();
    run.limits.masks.max_visits_per_run = run.limits.masks.max_visits_per_item;
    let error = execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop && !run.failed); assert_eq!(run.reserved, work); admission.reject = false;
    let error = execute_admitted(c.prepare_with_control(document(), &mut Continue).unwrap(), &mut driver,
        &mut admission, &mut run, context(3, 2), &mut Continue).err().unwrap();
    assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::WorkLimit);
    assert_eq!(state.admissions.get(), 1); assert_eq!(state.calls.get(), 0); assert_eq!(run.reserved, work);
}
#[test]
fn native_failure_poison_and_bad_work_never_become_success_or_qa_abstention() {
    let p = planner();
    for (kind, mode) in [BuiltInTask::Keyphrases, BuiltInTask::Answer].into_iter()
        .flat_map(|kind| [Mode::Cancel, Mode::Dirty, Mode::Oversize, Mode::WrongWork].map(|mode| (kind, mode))) {
        let c = compiler(&p, kind);
        let state = Rc::new(State::default()); let (mut driver, mut admission, mut run) = host(&state, mode);
        let error = execute_admitted(c.prepare_with_control(document(), &mut Continue).unwrap(), &mut driver,
            &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert!(!state.live.get()); assert!(!state.running.get());
        if matches!(mode, Mode::Cancel) { assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline)); }
        driver.clean = true; driver.mode = Mode::Success;
        assert!(execute_admitted(c.prepare_with_control(document(), &mut Continue).unwrap(), &mut driver,
            &mut admission, &mut run, context(2, 1), &mut Continue).is_err());
        assert_eq!(state.calls.get(), 1); assert_eq!(state.drops.get(), 1);
    }
}
#[test]
fn panic_releases_physical_work_before_guard_and_permanently_latches_run() {
    let p = planner(); let state = Rc::new(State::default()); let (mut driver, mut admission, mut run) = host(&state, Mode::Panic);
    let plan = compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).unwrap();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue)
    }));
    assert!(caught.is_err() && run.failed); assert!(!state.live.get()); assert!(!state.running.get());
    assert_eq!(state.drops.get(), 1); assert!(run.reserved.forward_positions > 0 && run.mask_visits > 0);
}
#[test]
fn cancellation_after_native_completion_refuses_delivery_and_releases_guard() {
    struct Third(usize);
    impl DecodeStepControl for Third {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0 += 1; (self.0 == 3).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let p = planner(); let state = Rc::new(State::default()); let (mut driver, mut admission, mut run) = host(&state, Mode::Success);
    let plan = compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).unwrap();
    let error = execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Third(0)).err().unwrap();
    assert!(error.stop && run.failed); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(state.calls.get(), 1); assert_eq!(state.drops.get(), 1);
}
#[test]
fn every_output_work_axis_and_task_variant_is_checked() {
    let p = planner(); let plan = compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).unwrap();
    let output = completion(&plan); assert!(valid_result(&plan, &output, 5));
    for axis in 0..12 {
        let mut changed = output.clone();
        match axis { 0 => changed.schema_version += 1, 1 => changed.execution.push('x'),
            2 => changed.model_work.forward_positions += 1, 3 => changed.model_work.projected_logits += 1,
            4 => changed.model_work.attention_pairs += 1, 5 => changed.model_work.projections.dot_products += 1,
            6 => changed.model_work.projections.multiply_accumulates += 1, 7 => {
                if let SourceTaskResult::Keyphrases(result) = &mut changed.result { result.mask_node_visit_charge += 1; }
            }, 8 => changed.group_width = 16, 9 => {
                if let SourceTaskResult::Keyphrases(result) = &mut changed.result { result.schema_version += 1; }
            }, 10 => {
                if let SourceTaskResult::Keyphrases(result) = &mut changed.result { result.numerics_profile = "strict-quantized-v1".to_owned(); }
            }, _ => {
                if let SourceTaskResult::Keyphrases(result) = &mut changed.result { result.task_spec_version = "ner-v1".to_owned(); }
            } }
        assert!(!valid_result(&plan, &changed, 5));
    }
    let other = compiler(&p, BuiltInTask::Ner).prepare_with_control(document(), &mut Continue).unwrap();
    assert!(!valid_result(&other, &output, 5));
}

// The production runner, real compiler and private driver together exercise
// control propagation and delivery ordering; public native construction cannot
// accept this harness's driver or synthetic result.
struct Harness<'p> { compiler: Int4SourceBatchPlanner<'p>, driver: Engine, admission: Admission, run: RunState,
    planning: Rc<Cell<bool>> }
impl BatchProcessor for Harness<'_> {
    type Args = SourceBatchArgs; type Prepared = PreparedInt4SourceTask; type Output = GuardedOutput<Int4SourceTaskRun, Guard>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { panic!("runner lost planning control") }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, doc: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.planning.set(true); self.compiler.prepare_with_control(doc, control)
    }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork {
        let w = plan.planned_work(); BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> { unreachable!() }
    fn execute_with_context<C: DecodeStepControl>(&mut self, plan: Self::Prepared, context: BatchRequestContext, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(plan, &mut self.driver, &mut self.admission, &mut self.run, context, control)
    }
}
struct Writer { state: Rc<State>, failure: usize, broken: bool, document: bool, flushed: usize }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.broken { return Err(io::Error::other("private partial-output fixture")); }
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap(); self.document = event.get("result").is_some();
        if self.document {
            assert!(self.state.live.get()); assert!(!self.state.running.get());
            if self.failure == 1 { self.broken = true; return Ok(bytes.len().min(3)); }
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.document {
            assert!(self.state.live.get()); self.flushed += 1;
            if self.failure == 2 { self.broken = true; return Err(io::Error::other("private flush fixture")); }
        }
        Ok(())
    }
}
#[test]
fn real_runner_retains_each_guard_through_write_flush_and_reuses_engine_across_epochs() {
    let p = planner();
    for failure in 0..3 {
        let state = Rc::new(State::default()); let (driver, admission, run) = host(&state, Mode::Success);
        let mut h = Harness { compiler: compiler(&p, BuiltInTask::Keyphrases), driver, admission, run, planning: Rc::new(Cell::new(false)) };
        let mut writer = Writer { state: Rc::clone(&state), failure, broken: false, document: false, flushed: 0 };
        let bytes = b"{\"id\":\"a\",\"text\":\"one\"}\n{\"flush\":true}\n{\"id\":\"a\",\"text\":\"two\"}\n";
        let mut input = Cursor::new(bytes);
        let result = run_ndjson(&mut input, &mut writer, &mut h, BatchLimits::default(), &mut Continue);
        if failure == 0 {
            let summary = result.unwrap(); assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 0);
            assert_eq!(summary.reserved_work.forward_positions, h.run.reserved.forward_positions);
            assert_eq!(summary.reserved_work.projected_logits, h.run.reserved.projected_logits);
            assert_eq!(h.run.mask_visits, 2 * limits().masks.max_visits_per_item); assert_eq!(writer.flushed, 2);
        } else {
            assert_eq!(result.unwrap_err().fault.code, BatchCode::OutputIo);
            assert_eq!(state.calls.get(), 1); assert!(input.position() < bytes.len() as u64);
        }
        assert_eq!(state.drops.get(), state.calls.get()); assert!(!state.live.get());
    }
}
#[test]
fn real_runner_passes_caller_cancellation_into_planning_before_admission() {
    struct CancelPlanning(Rc<Cell<bool>>);
    impl DecodeStepControl for CancelPlanning {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0.get().then_some(DecodeCancellationKind::Deadline)
        }
    }
    let p = planner(); let state = Rc::new(State::default()); let (driver, admission, run) = host(&state, Mode::Success);
    let planning = Rc::new(Cell::new(false));
    let mut h = Harness { compiler: compiler(&p, BuiltInTask::Keyphrases), driver, admission, run, planning: Rc::clone(&planning) };
    let mut output = Vec::new();
    let error = run_ndjson(&mut Cursor::new(b"{\"id\":\"a\",\"text\":\"one\"}\n"), &mut output,
        &mut h, BatchLimits::default(), &mut CancelPlanning(planning)).unwrap_err();
    assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(error.summary.failed, 1); assert_eq!(error.summary.reserved_work, BatchWork::default());
    assert_eq!(state.admissions.get(), 0); assert_eq!(state.calls.get(), 0);
}

#[test]
fn completed_qa_abstention_is_a_successful_ordered_record_with_its_own_output_guard() {
    let p = planner(); let state = Rc::new(State::default()); let (driver, admission, run) = host(&state, Mode::Success);
    let mut h = Harness { compiler: compiler(&p, BuiltInTask::Answer), driver, admission, run, planning: Rc::new(Cell::new(false)) };
    let mut writer = Writer { state: Rc::clone(&state), failure: 0, broken: false, document: false, flushed: 0 };
    let summary = run_ndjson(&mut Cursor::new(b"{\"id\":\"question\",\"text\":\"Is this established?\"}\n"),
        &mut writer, &mut h, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!(summary.succeeded, 1); assert_eq!(summary.failed, 0);
    assert_eq!(writer.flushed, 1); assert_eq!(state.calls.get(), 1); assert_eq!(state.drops.get(), 1);
    let plan = compiler(&p, BuiltInTask::Answer).prepare_with_control(document(), &mut Continue).unwrap();
    let output = completion(&plan); assert!(valid_result(&plan, &output, 5));
    let SourceTaskResult::Answer(answer) = output.result else { panic!("answer task changed"); };
    assert_eq!(answer.status, AnswerStatus::Abstained); assert!(!answer.answerable); assert_eq!(answer.answer, None);
}
#[test]
fn complete_kv_capacity_is_priced_even_for_a_short_live_prefix() {
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases);
    let mut doc = document(); doc.task_args = Some(SourceBatchArgs::Keyphrases {
        options: KeyphraseOptions::default(), budget: TaskBudget { max_kv_bytes: 4096 * KV_BYTES_PER_TOKEN as u64 - 1, ..budget() } });
    let plan = c.prepare_with_control(doc, &mut Continue).unwrap(); assert!(plan.planned_work().forward_positions < 4096);
    let state = Rc::new(State::default()); let (mut driver, mut admission, mut run) = host(&state, Mode::Success);
    let error = execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(state.admissions.get(), 0); assert!(run.reserved.forward_positions > 0);
}

#[test]
fn a_different_resident_group_refuses_before_admission_or_native_work() {
    let p = planner(); let state = Rc::new(State::default());
    let (mut driver, mut admission, mut run) = host(&state, Mode::Success);
    driver.width = 16;
    let plan = compiler(&p, BuiltInTask::Keyphrases).prepare_with_control(document(), &mut Continue).unwrap();
    let error = execute_admitted(plan, &mut driver, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(state.admissions.get(), 0); assert_eq!(state.calls.get(), 0);
}
