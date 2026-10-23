//! Planner/recipe and authenticated runner fixtures. These use real INT4
//! planning and job storage, not model inference, accuracy or crash proof.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::extract::ExtractionBatchGrounding,
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField,
        TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION, strict_int8::Int8Work},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls, specials::ArchivedControlRegistries},
};
use serde::Serializer;
use std::{cell::{Cell, RefCell}, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::atomic::{AtomicU64, Ordering}};

const WIDTH: usize = 127;
const NUMBER: &str = "12345678901234567890123456789012345678";
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 64, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn defaults() -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::Structural, budget: budget() }
}
fn native(width: usize) -> Int4ExtractionBatchLimits {
    Int4ExtractionBatchLimits {
        max_model_work: Int4Work::for_sequence(width, 0, 8192, 256 * NANBEIGE_VOCAB_SIZE).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
            max_visits_per_item: 200_000_000, max_visits_per_run: 800_000_000 },
    }
}
fn assets(width: usize) -> (ArchivedControlRegistries, u32, ExecutionIdentity) {
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let d = Sha256Digest::of_bytes(b"synthetic-INT4-durable-schema-fixture");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
    )).unwrap());
    let identity = ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    (controls, eos, identity)
}
fn configured(width: usize, compiler: CompileLimits, source: SourceRuntimeLimits,
    args: ExtractionBatchArgs, cap: Int4ExtractionBatchLimits) -> Result<Int4ExtractionJobPlanner, BatchFault> {
    let (controls, eos, identity) = assets(width);
    Int4ExtractionJobPlanner::pinned(controls.template_controls(), eos, identity, width, budget(),
        compiler, source, args, cap)
}
fn planner() -> Int4ExtractionJobPlanner {
    configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), defaults(), native(WIDTH)).unwrap()
}
fn document(text: &str) -> BatchDocument<()> {
    BatchDocument { id: "alpha".to_owned(), text: text.to_owned(), task_args: None }
}
fn job_limits(cap: Int4ExtractionBatchLimits) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: budget().max_output_bytes as usize,
        max_spool_bytes: 1 << 20, max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(cap.max_model_work, cap.masks.max_visits_per_run) }
}
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Alice private source"}"#;
    let b = br#"{"id":"beta","text":"Bob private source","task_args":null}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn freeze(planner: &Int4ExtractionJobPlanner) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: planner.execution_identity(),
        recipe: &planner.recipe, limits: job_limits(planner.native_limits()) }, inputs(), &mut Continue).unwrap()
}

#[test]
fn exact_schema_and_source_bytes_are_bound_without_privileged_marker_tokens() {
    let mut args = defaults();
    args.schema = format!(" {{\"type\":\"number\",\"const\":{NUMBER}}} \n");
    let p = configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), args.clone(), native(WIDTH)).unwrap();
    assert_eq!(serde_json::to_value(&p.recipe).unwrap()["defaults"]["schema"], args.schema);
    let prepared = p.compiler.prepare(shared_document(document("PRIVATE_SOURCE")).unwrap()).unwrap();
    assert_eq!(prepared.execution_identity().schema_digest, Sha256Digest::of_bytes(args.schema.as_bytes()));
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    assert_eq!(tokenizer.tokenizer().decode_bytes(prepared.task_plan().ir().prompt_segments()[2].token_ids()).unwrap(),
        args.schema.as_bytes());
    assert_ne!(p.execution_identity().tokenizer_digest, assets(WIDTH).2.tokenizer_digest);
    assert_ne!(p.execution_identity().template_digest, assets(WIDTH).2.template_digest);
    assert_eq!(p.execution_identity().numerics_profile, NumericsProfile::StrictQuantized { version: 2 });

    args.schema = r#"{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}"#.to_owned();
    args.grounding = ExtractionBatchGrounding::SourceMembership;
    let p = configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), args, native(WIDTH)).unwrap();
    let source = "é Alice <tool_call> <|im_start|> 上海";
    let prepared = p.compiler.prepare(shared_document(document(source)).unwrap()).unwrap();
    assert_eq!(prepared.source().text(), source);
    let tokens = prepared.task_plan().ir().prompt_segments()[4].token_ids();
    assert_eq!(tokenizer.tokenizer().decode_bytes(tokens).unwrap(), source.as_bytes());
    let controls = pinned_controls::pinned().unwrap();
    assert!(tokens.iter().all(|id| !controls.template_controls().contains(*id)));
    assert_eq!(prepared.execution_identity().grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
    let other = p.compiler.prepare(shared_document(document("Bob")).unwrap()).unwrap();
    assert!(prepared.verify_identity(other.execution_identity()).is_err());
}

#[test]
fn immutable_recipe_includes_all_compiler_source_native_and_policy_settings() {
    let base = freeze(&planner());
    for axis in 0..23 {
        let mut c = CompileLimits::default(); let mut s = SourceRuntimeLimits::default();
        let mut n = native(WIDTH);
        match axis {
            0 => c.max_schema_bytes += 1, 1 => c.max_string_bytes += 1, 2 => c.max_array_items += 1,
            3 => c.max_output_bytes += 1, 4 => c.max_states += 1, 5 => c.max_transitions += 1,
            6 => c.max_mask_bytes += 1, 7 => s.index.max_matches += 1,
            8 => s.verification.max_fields += 1, 9 => s.verification.max_matches += 1,
            10 => s.verification.max_scan_steps += 1, 11 => n.max_model_work.forward_positions += 1,
            12 => n.max_model_work.projected_logits += 1, 13 => n.max_model_work.attention_pairs += 1,
            14 => n.max_model_work.projections.dot_products += 1,
            15 => n.max_model_work.projections.multiply_accumulates += 1,
            16 => n.masks.per_mask.max_trie_node_visits += 1, 17 => n.masks.per_mask.checkpoint_interval_nodes += 1,
            18 => n.masks.max_visits_per_item += 1, 19 => n.masks.max_visits_per_run += 1,
            20 => s.index.max_source_bytes += 1, 21 => s.index.max_index_bytes += 1, _ => s.index.max_build_steps += 1,
        }
        let p = configured(WIDTH, c, s, defaults(), n).unwrap();
        assert_eq!(base.binding.compare(&freeze(&p).binding), Err(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
    }
    for axis in 0..4 {
        let mut p = planner();
        match axis {
            0 => p.recipe.defaults.schema.push(' '),
            1 => p.recipe.defaults.grounding = ExtractionBatchGrounding::SourceMembership,
            2 => p.recipe.defaults.budget.max_output_tokens -= 1,
            _ => p.recipe.input_policy = "changed-policy",
        }
        assert_eq!(base.binding.compare(&freeze(&p).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    let changed = configured(128, CompileLimits::default(), SourceRuntimeLimits::default(), defaults(), native(128)).unwrap();
    assert_eq!(base.binding.compare(&freeze(&changed).binding), Err(JobError::Mismatch(MismatchField::Execution)));
}

#[test]
fn journal_projection_preserves_group_dots_all_counts_and_existing_wire_shape() {
    for n in [0, 1, u64::MAX] {
        let model = Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
            projections: ProjectionWork { dot_products: n, multiply_accumulates: n } };
        let projected = JobWork::from_int4(model, n);
        assert_eq!(projected.int4_model_work(), model);
        let legacy_shape = JobWork { model: Int8Work { forward_positions: n, projected_logits: n,
            attention_pairs: n, projections: model.projections }, mask_node_visits: n };
        assert_eq!(canonjson::canonical_bytes(&projected).unwrap(), canonjson::canonical_bytes(&legacy_shape).unwrap());
    }
    let a = Int4Work::for_sequence(1, 0, 8, NANBEIGE_VOCAB_SIZE).unwrap();
    let b = Int4Work::for_sequence(128, 0, 8, NANBEIGE_VOCAB_SIZE).unwrap();
    assert!(a.projections.dot_products > b.projections.dot_products);
    assert_eq!(a.projections.multiply_accumulates, b.projections.multiply_accumulates);
    let sum = JobWork::from_int4(a, 11).checked_add(JobWork::from_int4(b, 13)).unwrap();
    assert_eq!(sum.int4_model_work().projections.dot_products, a.projections.dot_products + b.projections.dot_products);
    assert_eq!(sum.mask_node_visits, 24);
    assert!(!JobWork::from_int4(a, 11).fits(JobWork::from_int4(b, 11)));
}
fn axis_mut(work: &mut JobWork, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.model.forward_positions, 1 => &mut work.model.projected_logits,
        2 => &mut work.model.attention_pairs, 3 => &mut work.model.projections.dot_products,
        4 => &mut work.model.projections.multiply_accumulates, _ => &mut work.mask_node_visits }
}

#[test]
fn job_limits_must_match_all_six_lifetime_axes_and_full_output_authority() {
    let p = planner(); let exact = job_limits(p.native_limits());
    p.check_job_limits(exact).unwrap();
    for axis in 0..6 {
        for increase in [false, true] {
            let mut changed = exact; let value = axis_mut(&mut changed.max_work, axis);
            if increase { *value += 1; } else { *value -= 1; }
            assert_eq!(p.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()));
        }
    }
    let mut changed = exact; changed.max_result_bytes -= 1;
    assert_eq!(p.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()));
    for value in [budget().max_kv_bytes - 1, budget().max_kv_bytes + 1] {
        let mut args = defaults(); args.budget.max_kv_bytes = value;
        assert!(configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), args, native(WIDTH)).is_err());
    }
    let mut args = defaults(); args.schema = "x".repeat(1024 * 1024);
    assert!(configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), args, native(WIDTH)).is_err());
}

#[test]
fn shared_policy_rejects_programmatic_overrides_and_wrong_profiles_before_planning() {
    let mut input = document("PRIVATE_SOURCE"); input.task_args = Some(());
    let error = shared_document(input).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidEnvelope);
    let p = planner();
    let explicit = BatchDocument { id: "alpha".to_owned(), text: "PRIVATE_SOURCE".to_owned(), task_args: Some(defaults()) };
    assert_eq!(p.compiler.prepare(explicit).err().unwrap().fault.code, BatchCode::InvalidEnvelope);
    let (controls, eos, mut identity) = assets(WIDTH);
    identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    assert!(Int4ExtractionJobPlanner::pinned(controls.template_controls(), eos, identity, WIDTH, budget(),
        CompileLimits::default(), SourceRuntimeLimits::default(), defaults(), native(WIDTH)).is_err());
}

// The following fixture replaces ONLY native execution. It uses the new
// immutable planner, exact group-aware quote, real JobRunner/OwnedJob and
// guarded serialization. It never constructs fake resident model authority.
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-int4-job-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
#[derive(Default)]
struct Events { prepared: RefCell<Vec<String>>, executed: RefCell<Vec<u64>>, held: Cell<bool>,
    serialized: Cell<usize>, held_polls: Cell<usize>, result_dropped: Cell<bool>, releases: Cell<usize> }
struct Guard(Rc<Events>);
impl Drop for Guard {
    fn drop(&mut self) {
        assert!(self.0.result_dropped.get()); assert!(self.0.held.replace(false));
        self.0.releases.set(self.0.releases.get() + 1);
    }
}
#[derive(Serialize)]
struct Data { fixture: &'static str, ordinal: u64, group_width: usize, exact_json: &'static str }
struct Output { data: Data, events: Rc<Events> }
impl Serialize for Output {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get());
        self.events.serialized.set(self.events.serialized.get() + 1);
        self.data.serialize(serializer)
    }
}
impl Drop for Output { fn drop(&mut self) { assert!(self.events.held.get()); self.events.result_dropped.set(true); } }
struct Processor { planner: Int4ExtractionJobPlanner, events: Rc<Events>, fail: bool }
impl Processor {
    fn new(planner: Int4ExtractionJobPlanner, events: Rc<Events>) -> Self { Self { planner, events, fail: false } }
}
impl BatchProcessor for Processor {
    type Args = ();
    type Prepared = PreparedInt4BatchExtraction;
    type Output = GuardedOutput<Output, Guard>;
    fn prepare(&mut self, input: BatchDocument<()>) -> Result<Self::Prepared, BatchItemFailure> {
        self.events.prepared.borrow_mut().push(input.id.clone());
        self.planner.compiler.prepare(shared_document(input)?)
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, input: BatchDocument<()>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.events.prepared.borrow_mut().push(input.id.clone());
        self.planner.compiler.prepare_with_control(shared_document(input)?, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let w = prepared.planned_work();
        BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("durable execution must pass stable delivery coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        assert_eq!(prepared.extraction_plan().group_width(), WIDTH);
        self.events.executed.borrow_mut().push(context.request_seq);
        if self.fail { return Err(BatchItemFailure::fatal(BatchCode::Execution)); }
        assert!(!self.events.held.replace(true)); self.events.result_dropped.set(false);
        let output = Output { data: Data { fixture: "storage-only", ordinal: context.request_seq,
            group_width: WIDTH, exact_json: NUMBER }, events: Rc::clone(&self.events) };
        Ok(GuardedOutput::new(output, Guard(Rc::clone(&self.events))))
    }
}
impl DurableBatchProcessor for Processor {
    type Recipe = Int4ExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { &self.planner.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        Ok(JobWork::from_int4(prepared.planned_work(), self.planner.native_limits().masks.max_visits_per_item))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.extraction_plan().max_result_bytes() }
}
struct CancelAfterSync(Rc<Events>);
impl DecodeStepControl for CancelAfterSync {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if !self.0.held.get() { return None; }
        let n = self.0.held_polls.get() + 1; self.0.held_polls.set(n);
        (n == 2).then_some(DecodeCancellationKind::PollQuota)
    }
}
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(e) => e, Ok(_) => panic!("expected refusal") }
}

#[test]
fn later_nonnull_overrides_refuse_before_any_files_planning_or_execution() {
    for value in ["{}", "[]", "false", "0", "\"schema\"", "{\"schema\":\"{}\"}"] {
        let root = Root::new(); let base = inputs();
        let line = format!("{{\"id\":\"beta\",\"text\":\"PRIVATE_SOURCE\",\"task_args\":{value}}}");
        let inputs = [JobInput { id: base[0].id, original: base[0].original, normalized: base[0].normalized },
            JobInput { id: "beta", original: line.as_bytes(), normalized: line.as_bytes() }];
        let events = Rc::new(Events::default()); let p = planner(); let limits = job_limits(p.native_limits());
        assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs,
            Processor::new(p, Rc::clone(&events)), &mut Continue)), JobRunError::InvalidEnvelope);
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }
}

#[test]
fn resumed_grouped_recipe_skips_commits_and_materializes_exact_ordered_bytes() {
    let root = Root::new(); let full = Root::new(); let inputs = inputs();
    let events = Rc::new(Events::default()); let p = planner(); let limits = job_limits(p.native_limits());
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(p, Rc::clone(&events)), &mut Continue).unwrap();
    let first = run.step(&mut Continue).unwrap().unwrap();
    assert_eq!(first.committed, 1); assert_eq!(first.attempts, 1); drop(run);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(planner(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
    let progress = resumed.run(&mut Continue).unwrap();
    assert_eq!(progress.committed, 2); assert_eq!(progress.attempts, 2);
    assert_eq!(*events.prepared.borrow(), vec!["alpha", "beta"]);
    assert_eq!(*events.executed.borrow(), vec![1, 2]); assert_eq!(events.releases.get(), 2);
    assert!(!events.held.get()); assert!(events.serialized.get() >= 4);
    let result = resumed.read_committed(0, &mut Continue).unwrap();
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&result).unwrap()["exact_json"], NUMBER);
    resumed.verify(&mut Continue).unwrap(); resumed.materialize_ordered(&mut Continue).unwrap();
    let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(planner(), Rc::new(Events::default())), &mut Continue).unwrap();
    baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), fs::read(full.0.join("materialized.ndjson")).unwrap());
    assert!(resumed.step(&mut Continue).unwrap().is_none());
}

#[test]
fn post_sync_cancellation_keeps_grouped_work_spent_and_requires_explicit_tail_repair() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let p = planner(); let limits = job_limits(p.native_limits());
    let mut control = CancelAfterSync(Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(p, Rc::clone(&events)), &mut control).unwrap();
    assert_eq!(error(run.step(&mut control)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    let spent = run.progress().reserved_work;
    assert!(spent.model.projections.dot_products > 0); assert_eq!(spent.mask_node_visits, native(WIDTH).masks.max_visits_per_item);
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().attempts, 1);
    assert!(run.is_stopped()); assert!(!events.held.get()); assert_eq!(events.releases.get(), 1); drop(run);
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(planner(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue)),
        JobRunError::Storage(JobError::UncommittedTail));
    let mut changed = defaults(); changed.schema = r#"{"type":"null"}"#.to_owned();
    let changed = configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), changed, native(WIDTH)).unwrap();
    let before = events.prepared.borrow().len();
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(changed, Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(events.prepared.borrow().len(), before);
    // A recipe mismatch must refuse BEFORE discarding the original orphan.
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(planner(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue)),
        JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(planner(), Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    assert_eq!(resumed.progress().reserved_work, spent);
    assert_eq!(resumed.step(&mut Continue).unwrap().unwrap().reserved_work, spent.checked_add(spent).unwrap());
    let progress = resumed.run(&mut Continue).unwrap();
    assert_eq!(progress.committed, 2); assert_eq!(progress.attempts, 3);
    assert_eq!(*events.executed.borrow(), vec![1, 1, 2]); assert_eq!(events.releases.get(), 3);
}

#[test]
fn all_grouped_work_axes_stay_spent_when_a_new_native_ledger_is_created() {
    for axis in 0..6 {
        let p = planner();
        let prepared = p.compiler.prepare(shared_document(document("Alice private source")).unwrap()).unwrap();
        let quote = JobWork::from_int4(prepared.planned_work(), p.native_limits().masks.max_visits_per_item);
        let mut cap = native(WIDTH); let mut durable = JobWork::from_int4(cap.max_model_work, cap.masks.max_visits_per_run);
        let mut quote_axis = quote;
        *axis_mut(&mut durable, axis) = *axis_mut(&mut quote_axis, axis);
        cap.max_model_work = durable.int4_model_work(); cap.masks.max_visits_per_run = durable.mask_node_visits;
        let make = || configured(WIDTH, CompileLimits::default(), SourceRuntimeLimits::default(), defaults(), cap).unwrap();
        let root = Root::new(); let inputs = inputs(); let limits = job_limits(cap); let events = Rc::new(Events::default());
        let mut processor = Processor::new(make(), Rc::clone(&events)); processor.fail = true;
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs, processor, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!(run.progress().reserved_work, quote); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
            Processor::new(make(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(*events.executed.borrow(), vec![1]); assert_eq!(resumed.progress().reserved_work, quote);
        assert_eq!(resumed.progress().attempts, 1);
    }
}
