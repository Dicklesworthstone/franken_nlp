//! Real pinned planning, private synthetic execution for lifecycle injection.
//! These are not model-inference, semantic-quality or performance receipts.
use super::*;
use std::{cell::Cell, io::{self, Cursor}, rc::Rc};
use crate::{execution_identity::Sha256Digest,
    native_engine::{portable_int8::ProjectionWork, strict_int8::{STRICT_INT8_EXECUTION, STRICT_INT8_PROFILE}},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace,
        keyphrases::{KeyphraseResult, KEYPHRASES_RANKING}},
    template::{IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries}};

fn planner() -> SourceTaskPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap()
}
fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 128,
    max_output_bytes: 65536, max_grammar_states: 8192, max_kv_bytes: 1 << 30 } }
fn identity(p: &SourceTaskPlanner, kind: BuiltInTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"int8-source-corpus-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "nanbeige42-int8-v1".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: kind.spec().identity(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
        decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
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
fn compiler(p: &SourceTaskPlanner, kind: BuiltInTask) -> Int8SourceBatchPlanner<'_> {
    Int8SourceBatchPlanner::new(p, identity(p, kind), budget(), SourcePlanningLimits::default(), Some(args(kind))).unwrap()
}
fn document() -> BatchDocument<SourceBatchArgs> {
    BatchDocument { id: "item".to_owned(), text: "PRIVATE_SOURCE Alice <think> 上海".to_owned(), task_args: None }
}
fn limits() -> Int8SourceBatchLimits {
    Int8SourceBatchLimits { max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
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
        let direct = p.plan_int8_with_control(&args(kind).into_request(document().text), &cx,
            SourcePlanningLimits::default(), &mut Continue).unwrap();
        plan.verify_identity(direct.execution_identity()).unwrap();
        assert_eq!(plan.planned_work(), direct.planned_work());
        assert_eq!(plan.planned_work(), constrained_int8::planned_work(plan.prompt_tokens(), 128).unwrap());
        assert_eq!(plan.planned_work().projected_logits, 128 * NANBEIGE_VOCAB_SIZE as u64);
        assert!(plan.planned_work().projections.multiply_accumulates > plan.planned_work().projected_logits * 3072);
    }
}
#[test]
fn wrong_profile_backend_or_pinned_assets_are_refused_before_compilation() {
    let p = planner();
    for axis in 0..5 {
        let mut id = identity(&p, BuiltInTask::Keyphrases);
        match axis { 0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.backend_semantic_version.push('x'), 2 => id.tokenizer_digest = Sha256Digest::of_bytes(b"bad"),
            3 => id.template_digest = Sha256Digest::of_bytes(b"bad"), _ => id.task_spec = "extract-v1".to_owned() }
        assert!(Int8SourceBatchPlanner::new(&p, id, budget(), SourcePlanningLimits::default(), None).is_err());
    }
    assert!(SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Keyphrases), budget(), SourcePlanningLimits::default(), None).is_err());
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
    let empty = Int8SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Keyphrases), budget(), SourcePlanningLimits::default(), None).unwrap();
    assert!(empty.prepare_with_control(document(), &mut Continue).is_err());
    let mut oversized = args(BuiltInTask::Answer);
    if let SourceBatchArgs::Answer { passages, .. } = &mut oversized { passages[0].text = "x".repeat(MAX_SOURCE_ARGUMENT_BYTES); }
    assert!(Int8SourceBatchPlanner::new(&p, identity(&p, BuiltInTask::Answer), budget(), SourcePlanningLimits::default(), Some(oversized)).is_err());
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
    let work = Int8Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } };
    for axis in 0..6 {
        let mut l = limits(); l.max_model_work = work;
        match axis { 0 => l.max_model_work.forward_positions -= 1, 1 => l.max_model_work.projected_logits -= 1,
            2 => l.max_model_work.attention_pairs -= 1, 3 => l.max_model_work.projections.dot_products -= 1,
            4 => l.max_model_work.projections.multiply_accumulates -= 1, _ => l.masks.max_visits_per_run = 0 }
        let mut run = RunState::new(l); let error = run.begin(work, context(1, 1)).unwrap_err();
        assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(run.reserved, Int8Work::default()); assert_eq!(run.mask_visits, 0);
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
    let work = Int8Work::for_sequence(0, 1, 1).unwrap();
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
impl Int8SourceBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, req: Int8SourceAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
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
struct Engine { state: Rc<State>, mode: Mode, clean: bool }
struct Physical(Rc<State>);
impl Drop for Physical { fn drop(&mut self) { self.0.running.set(false); } }
fn completion(plan: &PreparedInt8SourceTask) -> Int8SourceTaskRun {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let mut tokens = tokenizer.tokenizer().encode_byte_fallback_only(b"[]").unwrap();
    tokens.push(tokenizer.eos_token_id().unwrap());
    let w = constrained_int8::planned_work(plan.prompt_tokens(), tokens.len()).unwrap();
    Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(), model_work: w,
        result: SourceTaskResult::Keyphrases(KeyphraseResult { schema_version: 1, task_spec_version: "keyphrases-v1".to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), ranking_policy: KEYPHRASES_RANKING.to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership, phrases: vec![],
            generated_token_ids: tokens, forward_positions: w.forward_positions, projected_logits: w.projected_logits, mask_node_visit_charge: 5 }) }
}
impl Driver for Engine {
    fn capacity(&self) -> usize { 4096 }
    fn clean(&self) -> bool { self.clean }
    fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt8SourceTask, id: &ExecutionIdentity,
        _: Int8JsonBudget, _: &mut C) -> Result<Int8SourceTaskRun, Int8SourceError> {
        plan.verify_identity(id)?; assert!(self.state.live.get()); assert!(!self.state.running.replace(true));
        let _physical = Physical(Rc::clone(&self.state)); self.state.calls.set(self.state.calls.get() + 1);
        match self.mode {
            Mode::Panic => panic!("private source lifecycle fixture"),
            Mode::Cancel => { self.clean = false; return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
            Mode::Oversize => { self.clean = false; return Err(Int8SourceError::Planning(SourcePlanningError::OutputBudget)); }
            Mode::Dirty => self.clean = false, _ => {}
        }
        let mut result = completion(plan);
        if matches!(self.mode, Mode::WrongWork) { result.model_work.projections.multiply_accumulates += 1; }
        Ok(result)
    }
}
fn host(state: &Rc<State>, mode: Mode) -> (Engine, Admission, RunState) {
    (Engine { state: Rc::clone(state), mode, clean: true },
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
    let p = planner(); let c = compiler(&p, BuiltInTask::Keyphrases);
    for mode in [Mode::Cancel, Mode::Dirty, Mode::Oversize, Mode::WrongWork] {
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
    for axis in 0..8 {
        let mut changed = output.clone();
        match axis { 0 => changed.schema_version += 1, 1 => changed.execution.push('x'),
            2 => changed.model_work.forward_positions += 1, 3 => changed.model_work.projected_logits += 1,
            4 => changed.model_work.attention_pairs += 1, 5 => changed.model_work.projections.dot_products += 1,
            6 => changed.model_work.projections.multiply_accumulates += 1, _ => {
                if let SourceTaskResult::Keyphrases(result) = &mut changed.result { result.mask_node_visit_charge += 1; }
            } }
        assert!(!valid_result(&plan, &changed, 5));
    }
    let other = compiler(&p, BuiltInTask::Ner).prepare_with_control(document(), &mut Continue).unwrap();
    assert!(!valid_result(&other, &output, 5));
}

// The production runner, real compiler and private driver together exercise
// control propagation and delivery ordering; public native construction cannot
// accept this harness's driver or synthetic result.
struct Harness<'p> { compiler: Int8SourceBatchPlanner<'p>, driver: Engine, admission: Admission, run: RunState,
    planning: Rc<Cell<bool>> }
impl BatchProcessor for Harness<'_> {
    type Args = SourceBatchArgs; type Prepared = PreparedInt8SourceTask; type Output = GuardedOutput<Int8SourceTaskRun, Guard>;
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
