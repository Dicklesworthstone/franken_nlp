//! Lifecycle tests use a private driver; they make no model-quality claim.
use super::*;
use std::{cell::Cell, io::{self, Cursor, Write}, rc::Rc};
use crate::{
    artifact::converter::PINNED_REVISION,
    native_engine::{constrained::JsonDecodeOutput, decode::DecodeCancellationKind,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE}},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace},
};

const WIDTH: usize = 127;
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 64, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn args() -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::Structural, budget: budget() }
}
fn planner(width: usize, defaults: Option<ExtractionBatchArgs>) -> Int4ExtractionBatchPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let d = Sha256Digest::of_bytes(b"synthetic-INT4-extraction-batch");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
    )).unwrap());
    let identity = ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None };
    Int4ExtractionBatchPlanner::pinned(registry.template_controls(), eos, identity, width, budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), defaults).unwrap()
}
fn document() -> BatchDocument<ExtractionBatchArgs> {
    BatchDocument { id: "private-id".to_owned(), text: "PRIVATE_SOURCE".to_owned(), task_args: Some(args()) }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
fn limits() -> Int4ExtractionBatchLimits {
    Int4ExtractionBatchLimits {
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork {
                dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
            max_visits_per_item: 10_000, max_visits_per_run: 100_000 },
    }
}

#[test]
fn all_native_and_mask_axes_reserve_atomically_without_refunds_or_overflow() {
    let work = Int4Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } };
    for axis in 0..6 {
        let mut cap = limits(); cap.max_model_work = work;
        match axis { 0 => cap.max_model_work.forward_positions -= 1,
            1 => cap.max_model_work.projected_logits -= 1,
            2 => cap.max_model_work.attention_pairs -= 1,
            3 => cap.max_model_work.projections.dot_products -= 1,
            4 => cap.max_model_work.projections.multiply_accumulates -= 1,
            _ => cap.masks.max_visits_per_run = cap.masks.max_visits_per_item - 1 }
        let mut run = RunState::new(cap);
        let error = run.begin(work, context(1, 1)).unwrap_err();
        assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(run.reserved, Int4Work::default()); assert_eq!(run.mask_visits, 0);
    }
    for axis in 0..6 {
        let mut run = RunState::new(limits());
        match axis { 0 => run.reserved.forward_positions = u64::MAX,
            1 => run.reserved.projected_logits = u64::MAX,
            2 => run.reserved.attention_pairs = u64::MAX,
            3 => run.reserved.projections.dot_products = u64::MAX,
            4 => run.reserved.projections.multiply_accumulates = u64::MAX,
            _ => run.mask_visits = u64::MAX }
        let before = run.reserved; let masks = run.mask_visits;
        assert!(run.begin(work, context(1, 1)).unwrap_err().stop);
        assert_eq!(run.reserved, before); assert_eq!(run.mask_visits, masks);
    }
    let mut cap = limits(); cap.max_model_work = work;
    let mut run = RunState::new(cap); run.begin(work, context(1, 1)).unwrap(); run.failed = false;
    assert!(run.begin(work, context(3, 2)).unwrap_err().stop);
    assert_eq!(run.reserved, work); assert_eq!(run.mask_visits, cap.masks.max_visits_per_item);
}

#[test]
fn invalid_limits_and_replayed_delivery_coordinates_close_the_adapter() {
    for axis in 0..9 {
        let mut cap = limits();
        match axis { 0 => cap.max_model_work.forward_positions = 0,
            1 => cap.max_model_work.projected_logits = 0,
            2 => cap.max_model_work.attention_pairs = 0,
            3 => cap.max_model_work.projections.dot_products = 0,
            4 => cap.max_model_work.projections.multiply_accumulates = 0,
            5 => cap.masks.per_mask.max_trie_node_visits = 0,
            6 => cap.masks.per_mask.checkpoint_interval_nodes = 0,
            7 => cap.masks.max_visits_per_item = 0, _ => cap.masks.max_visits_per_run = 0 }
        assert!(cap.validate().is_err());
    }
    let work = constrained_int4::planned_work(WIDTH, 8, 4).unwrap();
    for ctx in [context(0, 1), context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut run = RunState::new(limits());
        assert!(run.begin(work, ctx).unwrap_err().stop); assert!(run.failed);
    }
    let mut run = RunState::new(limits()); run.begin(work, context(5, 1)).unwrap(); run.failed = false;
    assert!(run.begin(work, context(5, 2)).unwrap_err().stop);
}

#[derive(Default)]
struct State { live: Cell<bool>, native: Cell<bool>, drops: Cell<usize>, admissions: Cell<usize>, calls: Cell<usize> }
struct Guard(Rc<State>);
impl Drop for Guard {
    fn drop(&mut self) {
        assert!(!self.0.native.get()); assert!(self.0.live.replace(false));
        self.0.drops.set(self.0.drops.get() + 1);
    }
}
struct Admission { state: Rc<State>, reject: bool, wrong_identity: bool }
impl Int4ExtractionBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, request: Int4ExtractionAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.state.admissions.set(self.state.admissions.get() + 1);
        if self.reject { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert_eq!(request.kv_reservation_bytes, Int4MemoryRequirement::for_context(4096, WIDTH).unwrap().kv_bytes);
        assert_eq!(request.mask_node_visits, limits().masks.max_visits_per_item);
        assert_eq!(request.mask_limits, limits().masks.per_mask);
        assert!(request.model_work.attention_pairs > 0 && request.model_work.projections.dot_products > 0);
        assert_eq!(request.max_result_bytes, budget().max_output_bytes);
        assert!(!self.state.live.replace(true));
        let mut identity = request.identity.clone();
        if self.wrong_identity { identity.decision_policy_digest = Sha256Digest::of_bytes(b"changed"); }
        Ok((identity, Guard(Rc::clone(&self.state))))
    }
}
#[derive(Clone, Copy)]
enum Mode { Success, Cancel, Panic, Dirty, Oversized, WrongWork, WrongGroup }
struct Engine { state: Rc<State>, mode: Mode, clean: bool, width: usize }
struct NativeCall(Rc<State>);
impl Drop for NativeCall { fn drop(&mut self) { self.0.native.set(false); } }
impl Driver for Engine {
    fn capacity(&self) -> usize { 4096 }
    fn group_width(&self) -> usize { self.width }
    fn clean(&self) -> bool { self.clean }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt4BatchExtraction,
        identity: &ExecutionIdentity, admitted: Int4JsonBudget, _: &mut C)
        -> Result<Int4ExtractRun, Int4ExtractError> {
        prepared.verify_identity(identity)?;
        assert_eq!(admitted.native.max_projection_work, prepared.planned_work().projections);
        assert_eq!(admitted.json.max_projected_logits, prepared.planned_work().projected_logits);
        assert_eq!(admitted.json.max_total_mask_node_visits, limits().masks.max_visits_per_item);
        assert!(self.state.live.get()); assert!(!self.state.native.replace(true));
        let _physical = NativeCall(Rc::clone(&self.state)); self.state.calls.set(self.state.calls.get() + 1);
        match self.mode {
            Mode::Panic => panic!("private lifecycle fixture"),
            Mode::Cancel => { self.clean = false; return Err(Int4JsonError::Native(
                StrictInt4Error::Cancelled(DecodeCancellationKind::Deadline)).into()); }
            Mode::Oversized => { self.clean = false; return Err(ExtractError::OutputBudgetExceeded.into()); }
            Mode::Dirty => self.clean = false,
            _ => {}
        }
        let mut tokens = EmbeddedTokenizer::pinned().unwrap().tokenizer().encode_byte_fallback_only(b"true").unwrap();
        tokens.push(prepared.plan.options().eos_token_id);
        let work = constrained_int4::planned_work(self.width, prepared.plan.prompt_tokens(), tokens.len()).unwrap();
        let output = JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            token_ids: tokens, json: "true".to_owned(), forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 5 };
        let mut result = Int4ExtractRun { schema_version: 1, execution: INT4_EXTRACT_VERSION.to_owned(),
            group_width: self.width, model_work: work, result: ExtractResult { schema_version: 1,
                task_spec_version: "extract-v1".to_owned(), score_space: ScoreSpace::NotComputed,
                grounding: ExtractionGrounding::NotRequested, output, source_fields: Vec::new() } };
        if matches!(self.mode, Mode::WrongWork) { result.model_work.projections.multiply_accumulates += 1; }
        if matches!(self.mode, Mode::WrongGroup) { result.group_width += 1; }
        Ok(result)
    }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn host(state: &Rc<State>, mode: Mode) -> (Engine, Admission, RunState) {
    (Engine { state: Rc::clone(state), mode, clean: true, width: WIDTH },
        Admission { state: Rc::clone(state), reject: false, wrong_identity: false }, RunState::new(limits()))
}

#[test]
fn output_guard_survives_native_cleanup_and_serialization() {
    let prepared = planner(WIDTH, None).prepare(document()).unwrap(); let work = prepared.planned_work();
    let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    let output = execute_admitted(prepared, &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue).unwrap();
    assert!(state.live.get()); assert!(!state.native.get()); assert!(!run.failed);
    assert_eq!(run.reserved, work); assert_eq!(run.mask_visits, limits().masks.max_visits_per_item);
    let bytes = canonjson::canonical_string(&output).unwrap();
    for private in ["PRIVATE_SOURCE", "prompt_digest", "guard"] { assert!(!bytes.contains(private)); }
    assert!(state.live.get()); drop(output); assert_eq!(state.drops.get(), 1);
}

#[test]
fn group_mismatch_and_substituted_identity_refuse_before_native_work() {
    for wrong_group in [false, true] {
        let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
        if wrong_group { engine.width = 128; } else { admission.wrong_identity = true; }
        let error = execute_admitted(planner(WIDTH, None).prepare(document()).unwrap(), &mut engine,
            &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert_eq!(error.fault.code, BatchCode::Admission);
        assert_eq!(state.calls.get(), 0); assert_eq!(state.drops.get(), usize::from(!wrong_group));
    }
}

#[test]
fn failed_admission_keeps_all_model_and_mask_reservations_across_epochs() {
    let compiler = planner(WIDTH, None); let prepared = compiler.prepare(document()).unwrap();
    let work = prepared.planned_work(); let state = Rc::new(State::default());
    let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    admission.reject = true; run.limits.masks.max_visits_per_run = run.limits.masks.max_visits_per_item;
    let error = execute_admitted(prepared, &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop && !run.failed); assert_eq!(run.reserved, work);
    admission.reject = false;
    let error = execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission, &mut run,
        context(3, 2), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
    assert_eq!(state.admissions.get(), 1); assert_eq!(state.calls.get(), 0); assert_eq!(run.reserved, work);
}

#[test]
fn failed_or_malformed_native_completion_permanently_blocks_reuse() {
    let compiler = planner(WIDTH, None);
    for mode in [Mode::Cancel, Mode::Dirty, Mode::Oversized, Mode::WrongWork, Mode::WrongGroup] {
        let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, mode);
        let error = execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission, &mut run,
            context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert!(!state.live.get()); assert!(!state.native.get());
        if matches!(mode, Mode::Cancel) { assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline)); }
        engine.clean = true; engine.mode = Mode::Success;
        assert!(execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission, &mut run,
            context(2, 1), &mut Continue).is_err());
        assert_eq!(state.calls.get(), 1); assert_eq!(state.drops.get(), 1);
    }
}

#[test]
fn panic_drains_native_before_guard_release_and_keeps_the_latch_closed() {
    let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Panic);
    let prepared = planner(WIDTH, None).prepare(document()).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_admitted(prepared, &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue)
    }));
    assert!(result.is_err()); assert!(run.failed); assert!(!state.native.get()); assert!(!state.live.get());
    assert_eq!(state.drops.get(), 1); assert!(run.mask_visits > 0 && run.reserved.forward_positions > 0);
}

#[test]
fn cancellation_before_admission_before_native_or_after_completion_preserves_cause() {
    struct CancelAt { count: usize, at: usize }
    impl DecodeStepControl for CancelAt {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.count += 1; (self.count == self.at).then_some(DecodeCancellationKind::PollQuota)
        }
    }
    let compiler = planner(WIDTH, None);
    for at in 1..=3 {
        let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
        let error = execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission, &mut run,
            context(1, 1), &mut CancelAt { count: 0, at }).err().unwrap();
        assert!(error.stop && run.failed); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::PollQuota));
        assert_eq!(state.calls.get(), usize::from(at == 3));
        assert_eq!(state.drops.get(), usize::from(at > 1)); assert!(!state.live.get());
        assert!(run.reserved.forward_positions > 0 && run.mask_visits > 0);
    }
}

#[test]
fn complete_resident_kv_is_admitted_and_group_partition_changes_dot_work() {
    let compiler = planner(WIDTH, None); let mut input = document();
    input.task_args.as_mut().unwrap().budget.max_kv_bytes =
        Int4MemoryRequirement::for_context(4096, WIDTH).unwrap().kv_bytes - 1;
    let prepared = compiler.prepare(input).unwrap(); assert!(prepared.planned_work().forward_positions < 4096);
    let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    let error = execute_admitted(prepared, &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(state.admissions.get(), 0); assert_eq!(state.calls.get(), 0);
    let narrow = planner(1, None).prepare(document()).unwrap().planned_work();
    let wide = planner(128, None).prepare(document()).unwrap().planned_work();
    assert_eq!(narrow.forward_positions, wide.forward_positions);
    assert_eq!(narrow.projections.multiply_accumulates, wide.projections.multiply_accumulates);
    assert!(narrow.projections.dot_products > wide.projections.dot_products);
}

#[test]
fn shared_defaults_mode_refuses_missing_defaults_and_any_record_override() {
    assert!(planner(WIDTH, None).require_shared_defaults().is_err());
    let compiler = planner(WIDTH, Some(args())).require_shared_defaults().unwrap();
    let mut input = document();
    input.text = "x".repeat(1_000_001);
    input.task_args.as_mut().unwrap().schema = "not a schema".to_owned();
    let error = compiler.prepare(input).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::InvalidEnvelope);
    let input = BatchDocument { task_args: None, ..document() };
    let prepared = compiler.prepare(input).unwrap();
    assert_eq!(prepared.execution_identity().schema_digest, Sha256Digest::of_bytes(args().schema.as_bytes()));
    // Existing mode retains caller-authorized per-item override behavior.
    assert!(planner(WIDTH, Some(args())).prepare(document()).is_ok());
}

struct Harness { compiler: Int4ExtractionBatchPlanner, engine: Engine, admission: Admission, run: RunState }
impl BatchProcessor for Harness {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt4BatchExtraction;
    type Output = GuardedOutput<Int4ExtractRun, Guard>;
    fn prepare(&mut self, input: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { self.compiler.prepare(input) }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork {
        let w = plan.planned_work(); BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, plan: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(plan, &mut self.engine, &mut self.admission, &mut self.run, context, control)
    }
}
struct Writer { state: Rc<State>, partial: bool, broken: bool, document: bool, flushes: usize, bytes: Vec<u8> }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.broken { return Err(io::Error::other("injected partial write")); }
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.document = event.get("result").is_some();
        if self.document {
            assert!(self.state.live.get()); assert!(!self.state.native.get());
            if self.partial { self.broken = true; return Ok(3.min(bytes.len())); }
        }
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.document { assert!(self.state.live.get()); self.flushes += 1; }
        Ok(())
    }
}
#[test]
fn ordered_ndjson_retains_guards_through_flush_and_stops_after_partial_write() {
    for partial in [false, true] {
        let state = Rc::new(State::default()); let (engine, admission, run) = host(&state, Mode::Success);
        let mut harness = Harness { compiler: planner(WIDTH, Some(args())).require_shared_defaults().unwrap(),
            engine, admission, run };
        let mut writer = Writer { state: Rc::clone(&state), partial, broken: false, document: false, flushes: 0, bytes: Vec::new() };
        let bytes = b"{\"id\":\"same\",\"text\":\"one\"}\n{\"flush\":true}\n{\"id\":\"same\",\"text\":\"two\"}\n";
        let mut input = Cursor::new(bytes);
        let result = crate::batch::run_ndjson(&mut input, &mut writer, &mut harness, BatchLimits::default(), &mut Continue);
        if partial {
            assert_eq!(result.unwrap_err().fault.code, BatchCode::OutputIo);
            assert_eq!(state.calls.get(), 1); assert!(input.position() < bytes.len() as u64); assert_eq!(writer.flushes, 0);
        } else {
            let summary = result.unwrap(); assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 0);
            assert_eq!(writer.flushes, 2); assert_eq!(state.calls.get(), 2);
            assert_eq!(summary.reserved_work.projected_logits, harness.run.reserved.projected_logits);
            assert_eq!(harness.run.mask_visits, 2 * limits().masks.max_visits_per_item);
        }
        assert_eq!(state.drops.get(), state.calls.get()); assert!(!state.live.get());
    }
}

#[test]
fn shared_schema_override_is_rejected_without_native_work_and_next_record_can_run() {
    let state = Rc::new(State::default()); let (engine, admission, run) = host(&state, Mode::Success);
    let mut harness = Harness { compiler: planner(WIDTH, Some(args())).require_shared_defaults().unwrap(),
        engine, admission, run };
    let override_line = serde_json::json!({
        "id": "private-id", "text": "PRIVATE_SOURCE", "task_args": args(),
    }).to_string();
    let input = format!("{override_line}\n{{\"id\":\"next\",\"text\":\"source\"}}\n");
    let mut writer = Vec::new();
    let summary = crate::batch::run_ndjson(&mut Cursor::new(input.as_bytes()), &mut writer,
        &mut harness, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!(summary.failed, 1); assert_eq!(summary.succeeded, 1);
    assert_eq!(state.admissions.get(), 1); assert_eq!(state.calls.get(), 1);
    assert_eq!(harness.run.mask_visits, limits().masks.max_visits_per_item);
}

#[test]
fn terminal_eos_and_exact_completed_accounting_are_required_even_for_valid_json() {
    let compiler = planner(WIDTH, None);
    let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    let output = execute_admitted(compiler.prepare(document()).unwrap(), &mut engine, &mut admission,
        &mut run, context(1, 1), &mut Continue).unwrap();
    let prepared = compiler.prepare(document()).unwrap();
    let result = output.result().clone();
    assert!(valid_result(&prepared, &result, 10_000));
    let mut no_eos = result.clone(); no_eos.result.output.token_ids.pop();
    let work = constrained_int4::planned_work(WIDTH, prepared.plan.prompt_tokens(),
        no_eos.result.output.token_ids.len()).unwrap();
    no_eos.model_work = work; no_eos.result.output.forward_positions = work.forward_positions;
    no_eos.result.output.projected_logits = work.projected_logits;
    assert!(!valid_result(&prepared, &no_eos, 10_000));
    let mut oversized_masks = result.clone(); oversized_masks.result.output.mask_node_visit_charge = 10_001;
    assert!(!valid_result(&prepared, &oversized_masks, 10_000));
    let mut wrong_profile = result; wrong_profile.result.output.numerics_profile = "strict-quantized-v1".to_owned();
    assert!(!valid_result(&prepared, &wrong_profile, 10_000));
    drop(output); assert!(!state.live.get());
}
