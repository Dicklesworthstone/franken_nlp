//! Real pinned prompt/schema planning plus private lifecycle fault injection.
//! Synthetic native outputs below exercise transport ownership, not fidelity.
use super::*;
use std::{cell::Cell, io::{self, Cursor, Write}, rc::Rc};
use crate::{
    native_engine::{constrained::JsonDecodeOutput, portable_int8::ProjectionWork,
        strict_int8::STRICT_INT8_EXECUTION},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace},
    tokenizer::specials::ArchivedControlRegistries,
};

fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 64, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn assets() -> (ArchivedControlRegistries, u32, ExecutionIdentity) {
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
    let d = Sha256Digest::of_bytes(b"int8-extraction-batch-unit-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "nanbeige42-int8-v1".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None };
    (controls, eos, identity)
}
fn args(schema: &str, grounding: ExtractionBatchGrounding) -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: schema.to_owned(), grounding, budget: budget() }
}
fn planner(defaults: Option<ExtractionBatchArgs>) -> Int8ExtractionBatchPlanner {
    let (controls, eos, identity) = assets();
    Int8ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), defaults).unwrap()
}
fn document(schema: &str, text: &str, grounding: ExtractionBatchGrounding) -> BatchDocument<ExtractionBatchArgs> {
    BatchDocument { id: "private-id".to_owned(), text: text.to_owned(), task_args: Some(args(schema, grounding)) }
}
fn boolean() -> BatchDocument<ExtractionBatchArgs> {
    document(r#"{"type":"boolean"}"#, "PRIVATE_SOURCE <think>", ExtractionBatchGrounding::Structural)
}
fn context(seq: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: seq, byte_offset: 0 }
}
fn limits() -> Int8ExtractionBatchLimits {
    Int8ExtractionBatchLimits {
        max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
            max_visits_per_item: 128_000_000, max_visits_per_run: u64::MAX },
    }
}

#[test]
fn native_profiles_are_separate_before_any_prompt_can_be_compiled() {
    let (controls, eos, identity) = assets();
    assert!(ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity.clone(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None).is_err());
    for axis in 0..4 {
        let mut changed = identity.clone();
        match axis { 0 => changed.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => changed.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            2 => changed.backend_semantic_version = "other".to_owned(), _ => changed.kv_dtype = "int8".to_owned() }
        assert!(Int8ExtractionBatchPlanner::pinned(controls.template_controls(), eos, changed, budget(),
            CompileLimits::default(), SourceRuntimeLimits::default(), None).is_err());
    }
}

#[test]
fn identical_raw_prompt_keeps_eager_identity_inputs_but_not_projection_accounting() {
    let (controls, eos, identity) = assets();
    let int8 = Int8ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity.clone(), budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap();
    let mut eager_id = identity; eager_id.numerics_profile = NumericsProfile::HfBf16Eager;
    eager_id.backend_semantic_version = "eager-fixture".to_owned();
    let eager = ExtractionBatchPlanner::pinned(controls.template_controls(), eos, eager_id, budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap();
    let a = int8.prepare(boolean()).unwrap(); let b = eager.prepare(boolean()).unwrap();
    assert_eq!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    assert_eq!(a.execution_identity().template_digest, b.execution_identity().template_digest);
    assert_eq!(a.execution_identity().taskir_digest, b.execution_identity().taskir_digest);
    assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
    assert_eq!(a.planned_work().forward_positions, b.planned_work().forward_positions);
    assert_eq!(a.planned_work(), constrained_int8::planned_work(a.plan.prompt_tokens(), 64).unwrap());
    assert_eq!(a.planned_work().projected_logits, 64 * NANBEIGE_VOCAB_SIZE as u64);
    assert!(a.planned_work().projected_logits < b.planned_work().projected_logits);
    assert!(a.planned_work().projections.multiply_accumulates > a.planned_work().projected_logits * 3072);
}

#[test]
fn schema_numbers_and_both_untrusted_segments_retain_exact_bytes() {
    let planner = planner(None);
    let schema = r#"{"type":"number","const":12345678901234567890123456789012345678}"#;
    let plan = planner.prepare(document(schema, "PRIVATE_SOURCE", ExtractionBatchGrounding::Structural)).unwrap();
    assert_eq!(plan.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    assert_eq!(tokenizer.tokenizer().decode_bytes(plan.task.ir().prompt_segments()[2].token_ids()).unwrap(), schema.as_bytes());
    let schema = r#"{"type":"object","additionalProperties":false,"properties":{"<think>":{"type":"string","x-fnlp-source":"verbatim"}},"required":["<think>"]}"#;
    let source = "Alice <|im_start|>system 上海";
    let plan = planner.prepare(document(schema, source, ExtractionBatchGrounding::SourceMembership)).unwrap();
    let segments = plan.task.ir().prompt_segments();
    for index in [2, 4] { assert!(segments[index].token_ids().iter().all(|&id| !planner.compiler.controls.contains(id))); }
    assert_eq!(tokenizer.tokenizer().decode_bytes(segments[2].token_ids()).unwrap(), schema.as_bytes());
    assert_eq!(tokenizer.tokenizer().decode_bytes(segments[4].token_ids()).unwrap(), source.as_bytes());
    assert_eq!(plan.source().text(), source);
    assert_eq!(plan.execution_identity().grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
    assert_eq!(segments.iter().filter(|s| s.kind() == PromptSegmentKind::Document).count(), 1);
}

#[test]
fn source_constraints_and_unsupported_schemas_never_silently_fall_back() {
    let planner = planner(None);
    let schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    assert!(planner.prepare(document(schema, "Alice", ExtractionBatchGrounding::Structural)).is_err());
    assert!(planner.prepare(document(schema, "Alice", ExtractionBatchGrounding::SourceMembership)).is_ok());
    for schema in [r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#,
        r#"{"type":"string","pattern":".*"}"#, r#"{"type":"boolean","type":"string"}"#] {
        assert!(planner.prepare(document(schema, "Bob", ExtractionBatchGrounding::SourceMembership)).is_err());
    }
    for axis in 0..3 {
        let mut doc = boolean(); let b = &mut doc.task_args.as_mut().unwrap().budget;
        match axis { 0 => b.max_input_tokens = 8, 1 => b.max_output_tokens += 1, _ => b.max_kv_bytes += 1 }
        assert!(planner.prepare(doc).is_err());
    }
}

#[test]
fn bounded_defaults_and_per_item_override_do_not_change_subsequent_plans() {
    let (controls, eos, identity) = assets(); let compiler_limits = CompileLimits::default();
    let mut huge = args("", ExtractionBatchGrounding::Structural); huge.schema = "x".repeat(compiler_limits.max_schema_bytes + 1);
    assert!(Int8ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, budget(),
        compiler_limits, SourceRuntimeLimits::default(), Some(huge)).is_err());
    let planner = planner(Some(args(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural)));
    let mut doc = boolean(); doc.task_args = None;
    let first = planner.prepare(doc).unwrap();
    let changed = planner.prepare(document(r#"{"type":"null"}"#, "PRIVATE_SOURCE <think>", ExtractionBatchGrounding::Structural)).unwrap();
    assert!(first.verify_identity(changed.execution_identity()).is_err());
    let mut doc = boolean(); doc.task_args = None;
    first.verify_identity(planner.prepare(doc).unwrap().execution_identity()).unwrap();
}

#[test]
fn complete_identity_is_sealed_and_task_budget_reaches_native_json_execution() {
    let plan = planner(None).prepare(boolean()).unwrap();
    for axis in 0..7 {
        let mut changed = plan.execution_identity().clone(); let d = Sha256Digest::of_bytes(b"changed");
        match axis { 0 => changed.logical_model_digest = d, 1 => changed.source_revision.push('x'),
            2 => changed.packing_set_digest = d, 3 => changed.template_digest = d,
            4 => changed.schema_digest = d, 5 => changed.calibration_digest = d, _ => changed.decision_policy_digest = d }
        assert!(plan.verify_identity(&changed).is_err());
    }
    let b = plan.budget(limits().masks); let w = plan.planned_work();
    assert_eq!(b.json.max_kv_bytes, plan.task.ir().budget().max_kv_bytes);
    assert_eq!(b.json.max_forward_positions, w.forward_positions);
    assert_eq!(b.json.max_projected_logits, w.projected_logits);
    assert_eq!(b.native.max_attention_pairs, w.attention_pairs);
    assert_eq!(b.native.max_projection_work, w.projections);
}

#[test]
fn model_and_mask_charges_are_atomic_checked_and_nonrenewable() {
    let work = Int8Work { forward_positions: 2, projected_logits: 3, attention_pairs: 4,
        projections: ProjectionWork { dot_products: 5, multiply_accumulates: 6 } };
    for axis in 0..6 {
        let mut lim = limits(); lim.max_model_work = work;
        match axis { 0 => lim.max_model_work.forward_positions -= 1, 1 => lim.max_model_work.projected_logits -= 1,
            2 => lim.max_model_work.attention_pairs -= 1, 3 => lim.max_model_work.projections.dot_products -= 1,
            4 => lim.max_model_work.projections.multiply_accumulates -= 1,
            _ => lim.masks.max_visits_per_run = lim.masks.max_visits_per_item - 1 }
        let mut run = RunState::new(lim); let error = run.begin(work, context(1, 1)).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(run.reserved, Int8Work::default()); assert_eq!(run.mask_visits, 0); assert!(run.failed);
    }
    for axis in 0..5 {
        let mut prior = Int8Work::default();
        match axis { 0 => prior.forward_positions = u64::MAX, 1 => prior.projected_logits = u64::MAX,
            2 => prior.attention_pairs = u64::MAX, 3 => prior.projections.dot_products = u64::MAX,
            _ => prior.projections.multiply_accumulates = u64::MAX }
        assert!(add_work(prior, work).is_none());
    }
    let mut run = RunState::new(limits()); run.mask_visits = u64::MAX;
    assert!(run.begin(work, context(1, 1)).is_err()); assert_eq!(run.reserved, Int8Work::default());
    let mut lim = limits(); lim.max_model_work = work; lim.masks.max_visits_per_run = lim.masks.max_visits_per_item;
    let mut run = RunState::new(lim); run.begin(work, context(1, 1)).unwrap(); run.failed = false;
    assert!(run.begin(work, context(3, 2)).is_err()); assert_eq!(run.reserved, work);
    assert_eq!(run.mask_visits, lim.masks.max_visits_per_item);
}

#[test]
fn invalid_limits_and_replayed_delivery_coordinates_fail_closed() {
    for axis in 0..4 {
        let mut lim = limits();
        match axis { 0 => lim.masks.max_visits_per_item = 0, 1 => lim.masks.max_visits_per_run = 0,
            2 => lim.masks.per_mask.checkpoint_interval_nodes = 0, _ => lim.max_model_work.attention_pairs = 0 }
        assert!(lim.validate().is_err());
    }
    let work = Int8Work::for_sequence(0, 1, 1).unwrap();
    for ctx in [context(0, 1), context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut run = RunState::new(limits()); assert!(run.begin(work, ctx).is_err()); assert!(run.failed);
    }
    let mut run = RunState::new(limits()); run.begin(work, context(5, 1)).unwrap(); run.failed = false;
    assert!(run.begin(work, context(5, 2)).is_err());
}

#[derive(Default)]
struct State { live: Cell<bool>, native: Cell<bool>, drops: Cell<usize>, admissions: Cell<usize>, calls: Cell<usize> }
struct Guard(Rc<State>);
impl Drop for Guard {
    fn drop(&mut self) {
        assert!(!self.0.native.get()); assert!(self.0.live.replace(false)); self.0.drops.set(self.0.drops.get() + 1);
    }
}
struct Admission { state: Rc<State>, reject: bool, wrong_identity: bool }
impl Int8ExtractionBatchAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, req: Int8ExtractionAdmission<'_>) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        self.state.admissions.set(self.state.admissions.get() + 1);
        if self.reject { return Err(BatchItemFailure::reject(BatchCode::Admission)); }
        assert_eq!(req.kv_reservation_bytes, 4096 * KV_BYTES_PER_TOKEN as u64);
        assert_eq!(req.mask_node_visits, limits().masks.max_visits_per_item);
        assert!(req.model_work.attention_pairs > 0 && req.model_work.projections.multiply_accumulates > 0);
        assert_eq!(req.max_result_bytes, budget().max_output_bytes);
        assert!(!self.state.live.replace(true));
        let mut identity = req.identity.clone(); if self.wrong_identity { identity.source_revision.push('x'); }
        Ok((identity, Guard(Rc::clone(&self.state))))
    }
}
#[derive(Clone, Copy)]
enum Mode { Success, Cancel, Panic, Dirty, Oversized, WrongWork }
struct Engine { state: Rc<State>, mode: Mode, clean: bool }
struct NativeCall(Rc<State>);
impl Drop for NativeCall { fn drop(&mut self) { self.0.native.set(false); } }
impl Driver for Engine {
    fn capacity(&self) -> usize { 4096 }
    fn clean(&self) -> bool { self.clean }
    fn execute<C: DecodeStepControl>(&mut self, prepared: &PreparedInt8BatchExtraction,
        identity: &ExecutionIdentity, _: Int8JsonBudget, _: &mut C) -> Result<Int8ExtractRun, Int8ExtractError> {
        prepared.verify_identity(identity)?; assert!(self.state.live.get()); assert!(!self.state.native.replace(true));
        let _physical = NativeCall(Rc::clone(&self.state)); self.state.calls.set(self.state.calls.get() + 1);
        match self.mode {
            Mode::Panic => panic!("private lifecycle fixture"),
            Mode::Cancel => { self.clean = false; return Err(Int8JsonError::Native(StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline)).into()); }
            Mode::Oversized => { self.clean = false; return Err(ExtractError::OutputBudgetExceeded.into()); }
            Mode::Dirty => self.clean = false,
            _ => {}
        }
        let mut tokens = EmbeddedTokenizer::pinned().unwrap().tokenizer().encode_byte_fallback_only(b"true").unwrap();
        tokens.push(prepared.plan.options().eos_token_id);
        let work = constrained_int8::planned_work(prepared.plan.prompt_tokens(), tokens.len()).unwrap();
        let output = JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            token_ids: tokens, json: "true".to_owned(), forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 5 };
        let mut result = Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.to_owned(), model_work: work,
            result: ExtractResult { schema_version: 1, task_spec_version: "extract-v1".to_owned(), score_space: ScoreSpace::NotComputed,
                grounding: ExtractionGrounding::NotRequested, output, source_fields: Vec::new() } };
        if matches!(self.mode, Mode::WrongWork) { result.model_work.projections.multiply_accumulates += 1; }
        Ok(result)
    }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn host(state: &Rc<State>, mode: Mode) -> (Engine, Admission, RunState) {
    (Engine { state: Rc::clone(state), mode, clean: true },
        Admission { state: Rc::clone(state), reject: false, wrong_identity: false }, RunState::new(limits()))
}

#[test]
fn guard_survives_native_work_and_serialization_until_the_caller_drops_output() {
    let planner = planner(None); let prepared = planner.prepare(boolean()).unwrap(); let work = prepared.planned_work();
    let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    let output = execute_admitted(prepared, &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue).unwrap();
    assert!(state.live.get()); assert!(!state.native.get()); assert!(!run.failed);
    assert_eq!(run.reserved, work); assert_eq!(run.mask_visits, limits().masks.max_visits_per_item);
    let bytes = canonjson::canonical_string(&output).unwrap();
    assert!(!bytes.contains("PRIVATE_SOURCE") && !bytes.contains("prompt_digest") && !bytes.contains("guard"));
    assert!(state.live.get()); drop(output); assert_eq!(state.drops.get(), 1);
}

#[test]
fn substituted_admission_identity_is_fatal_before_any_native_work() {
    let planner = planner(None); let state = Rc::new(State::default());
    let (mut engine, mut admission, mut run) = host(&state, Mode::Success); admission.wrong_identity = true;
    let error = execute_admitted(planner.prepare(boolean()).unwrap(), &mut engine, &mut admission, &mut run,
        context(1, 1), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(state.calls.get(), 0); assert_eq!(state.drops.get(), 1); assert!(run.failed);
}

#[test]
fn admission_refusal_cannot_refund_model_or_mask_work_at_the_next_epoch() {
    let planner = planner(None); let prepared = planner.prepare(boolean()).unwrap(); let work = prepared.planned_work();
    let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    admission.reject = true; run.limits.masks.max_visits_per_run = run.limits.masks.max_visits_per_item;
    let error = execute_admitted(prepared, &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop); assert!(!run.failed); assert_eq!(run.reserved, work);
    admission.reject = false;
    let error = execute_admitted(planner.prepare(boolean()).unwrap(), &mut engine, &mut admission, &mut run,
        context(3, 2), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
    assert_eq!(state.admissions.get(), 1); assert_eq!(state.calls.get(), 0); assert_eq!(run.reserved, work);
}

#[test]
fn native_failures_preserve_cancellation_and_block_reuse_even_for_soft_error_codes() {
    let planner = planner(None);
    for mode in [Mode::Cancel, Mode::Dirty, Mode::Oversized, Mode::WrongWork] {
        let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, mode);
        let error = execute_admitted(planner.prepare(boolean()).unwrap(), &mut engine, &mut admission, &mut run,
            context(1, 1), &mut Continue).err().unwrap();
        assert!(error.stop && run.failed); assert!(!state.live.get()); assert!(!state.native.get());
        if matches!(mode, Mode::Cancel) { assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline)); }
        engine.clean = true; engine.mode = Mode::Success;
        assert!(execute_admitted(planner.prepare(boolean()).unwrap(), &mut engine, &mut admission, &mut run,
            context(2, 1), &mut Continue).is_err());
        assert_eq!(state.calls.get(), 1); assert_eq!(state.drops.get(), 1);
    }
}

#[test]
fn panic_drains_the_physical_call_before_guard_drop_and_leaves_the_latch_closed() {
    let planner = planner(None); let state = Rc::new(State::default());
    let (mut engine, mut admission, mut run) = host(&state, Mode::Panic);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_admitted(planner.prepare(boolean()).unwrap(), &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue)
    }));
    assert!(result.is_err()); assert!(run.failed); assert!(!state.native.get()); assert!(!state.live.get());
    assert_eq!(state.drops.get(), 1); assert!(run.mask_visits > 0 && run.reserved.forward_positions > 0);
}

#[test]
fn cancellation_after_native_completion_does_not_publish_a_result() {
    struct CancelThird(usize);
    impl DecodeStepControl for CancelThird {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0 += 1; (self.0 == 3).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let planner = planner(None); let state = Rc::new(State::default());
    let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    let error = execute_admitted(planner.prepare(boolean()).unwrap(), &mut engine, &mut admission, &mut run,
        context(1, 1), &mut CancelThird(0)).err().unwrap();
    assert!(error.stop && run.failed); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(state.calls.get(), 1); assert_eq!(state.drops.get(), 1); assert!(!state.live.get());
}

struct Harness { compiler: Int8ExtractionBatchPlanner, engine: Engine, admission: Admission, run: RunState }
impl BatchProcessor for Harness {
    type Args = ExtractionBatchArgs;
    type Prepared = PreparedInt8BatchExtraction;
    type Output = GuardedOutput<Int8ExtractRun, Guard>;
    fn prepare(&mut self, doc: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> { self.compiler.prepare(doc) }
    fn planned_work(&self, plan: &Self::Prepared) -> BatchWork {
        let w = plan.planned_work(); BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, plan: Self::Prepared, ctx: BatchRequestContext, control: &mut C)
        -> Result<Self::Output, BatchItemFailure> {
        execute_admitted(plan, &mut self.engine, &mut self.admission, &mut self.run, ctx, control)
    }
}
struct Writer { state: Rc<State>, fail_partial: bool, broken: bool, document: bool, flushed: usize, bytes: Vec<u8> }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.broken { return Err(io::Error::other("injected partial write")); }
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.document = event.get("result").is_some();
        if self.document {
            assert!(self.state.live.get()); assert!(!self.state.native.get());
            if self.fail_partial { self.broken = true; return Ok(3.min(bytes.len())); }
        }
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.document { assert!(self.state.live.get()); self.flushed += 1; }
        Ok(())
    }
}
#[test]
fn ordered_ndjson_reuses_engine_across_flush_and_stops_on_partial_output_failure() {
    for fail_partial in [false, true] {
        let state = Rc::new(State::default()); let (engine, admission, run) = host(&state, Mode::Success);
        let mut harness = Harness { compiler: planner(Some(args(r#"{"type":"boolean"}"#, ExtractionBatchGrounding::Structural))),
            engine, admission, run };
        let mut writer = Writer { state: Rc::clone(&state), fail_partial, broken: false, document: false, flushed: 0, bytes: Vec::new() };
        let bytes = b"{\"id\":\"same\",\"text\":\"one\"}\n{\"flush\":true}\n{\"id\":\"same\",\"text\":\"two\"}\n";
        let mut input = Cursor::new(bytes);
        let result = crate::batch::run_ndjson(&mut input, &mut writer, &mut harness, BatchLimits::default(), &mut Continue);
        if fail_partial {
            assert_eq!(result.unwrap_err().fault.code, BatchCode::OutputIo);
            assert_eq!(state.calls.get(), 1); assert!(input.position() < bytes.len() as u64); assert_eq!(writer.flushed, 0);
        } else {
            let summary = result.unwrap(); assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 0);
            assert_eq!(writer.flushed, 2); assert_eq!(state.calls.get(), 2);
            assert_eq!(summary.reserved_work.projected_logits, harness.run.reserved.projected_logits);
            assert_eq!(harness.run.mask_visits, 2 * limits().masks.max_visits_per_item);
        }
        assert_eq!(state.drops.get(), state.calls.get()); assert!(!state.live.get());
    }
}

#[test]
fn admission_prices_complete_resident_kv_not_only_the_request_live_prefix() {
    let planner = planner(None); let mut doc = boolean();
    doc.task_args.as_mut().unwrap().budget.max_kv_bytes = 4096 * KV_BYTES_PER_TOKEN as u64 - 1;
    let prepared = planner.prepare(doc).unwrap();
    assert!(prepared.planned_work().forward_positions < 4096);
    let state = Rc::new(State::default()); let (mut engine, mut admission, mut run) = host(&state, Mode::Success);
    let error = execute_admitted(prepared, &mut engine, &mut admission, &mut run, context(1, 1), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(state.admissions.get(), 0); assert_eq!(state.calls.get(), 0);
    assert!(run.reserved.forward_positions > 0 && run.mask_visits > 0);
}
