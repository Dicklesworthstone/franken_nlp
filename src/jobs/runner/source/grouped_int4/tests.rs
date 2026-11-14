//! Real pinned planning and real owned-job storage with synthetic execution.
//! These fixtures do not establish model inference, accuracy or crash parity.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::source::grouped_int4::Int4SourceAdmission,
    canonjson,
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField,
        TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{constrained_int4, decode::DecodeCancellationKind,
        generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE},
    tasks::{answer::{AnswerOptions, AnswerPassage}, keyphrases::KeyphraseOptions,
        ner::NerOptions, summarize::SummaryOptions},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};
use serde::Serializer;
use std::{cell::Cell, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::atomic::{AtomicU64, Ordering}};

const WIDTH: usize = 32;
fn source() -> Arc<SourceTaskPlanner> {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    Arc::new(SourceTaskPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap()).unwrap())
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 128, max_output_bytes: 65536,
        max_grammar_states: 8192, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes }
}
fn args(task: BuiltInTask) -> SourceBatchArgs {
    match task {
        BuiltInTask::Ner => SourceBatchArgs::Ner { options: NerOptions { max_entities: 4,
            max_mention_scalars: 32, ..Default::default() }, budget: budget() },
        BuiltInTask::Keyphrases => SourceBatchArgs::Keyphrases {
            options: KeyphraseOptions { max_phrases: 4, max_phrase_scalars: 32 }, budget: budget() },
        BuiltInTask::Summarize => SourceBatchArgs::Summarize { options: SummaryOptions {
            max_bullets: 2, max_bullet_scalars: 64, max_citations_per_bullet: 2, max_quote_scalars: 32 }, budget: budget() },
        BuiltInTask::Answer => SourceBatchArgs::Answer { passages: vec![AnswerPassage {
            id: "p1".to_owned(), text: "Alice private passage".to_owned() }], options: AnswerOptions {
            max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 }, budget: budget() },
        _ => unreachable!(),
    }
}
fn argument_budget(args: &mut SourceBatchArgs) -> &mut TaskBudget {
    match args { SourceBatchArgs::Ner { budget, .. } | SourceBatchArgs::Keyphrases { budget, .. }
        | SourceBatchArgs::Summarize { budget, .. } | SourceBatchArgs::Answer { budget, .. } => budget }
}
fn native(width: usize) -> Int4SourceBatchLimits {
    Int4SourceBatchLimits {
        max_model_work: Int4Work::for_sequence(width, 0, 8192, 1024 * NANBEIGE_VOCAB_SIZE).unwrap(),
        masks: SourceMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
            max_visits_per_item: 200_000_000, max_visits_per_run: 800_000_000 },
    }
}
fn identity(planner: &SourceTaskPlanner, task: BuiltInTask, width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"source-job-fixture-not-native-authority");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: packing, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task.spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn configured(task: BuiltInTask, width: usize, defaults: Option<SourceBatchArgs>,
    cap: Int4SourceBatchLimits) -> Int4SourceJobPlanner {
    let p = source(); let id = identity(&p, task, width);
    Int4SourceJobPlanner::configured(p, id, width, budget(), SourcePlanningLimits::default(), defaults, cap).unwrap()
}
fn planner(task: BuiltInTask) -> Int4SourceJobPlanner { configured(task, WIDTH, Some(args(task)), native(WIDTH)) }
fn document() -> BatchDocument<SourceBatchArgs> {
    BatchDocument { id: "alpha".to_owned(), text: "Alice private source 上海 <think>".to_owned(), task_args: None }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn prepare(p: &Int4SourceJobPlanner, input: BatchDocument<SourceBatchArgs>) -> PreparedInt4SourceTask {
    p.compiler().unwrap().prepare_with_control(p.checked_document(input).unwrap(), &mut Continue).unwrap()
}
fn limits(cap: Int4SourceBatchLimits) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(cap.max_model_work, cap.masks.max_visits_per_run) }
}
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Alice private source"}"#;
    let b = br#"{"id":"beta","text":"Bob private source","task_args":null}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn freeze(p: &Int4SourceJobPlanner) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: p.execution_identity(),
        recipe: &p.recipe, limits: limits(p.native_limits()) }, inputs(), &mut Continue).unwrap()
}

#[test]
fn all_four_tasks_compile_real_profile_bound_plans() {
    for kind in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let p = planner(kind); let plan = prepare(&p, document());
        assert_eq!(plan.execution_identity().task_spec, kind.spec().identity());
        assert_eq!(plan.execution_identity().numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
        assert_eq!(plan.planned_work(), constrained_int4::planned_work(WIDTH, plan.prompt_tokens(), 128).unwrap());
        assert_eq!(plan.max_kv_bytes(), budget().max_kv_bytes);
        assert_ne!(plan.execution_identity().prompt_digest, p.execution_identity().prompt_digest);
        let mut changed = document(); changed.text.push(' ');
        assert!(plan.verify_identity(prepare(&p, changed).execution_identity()).is_err());
    }
}

#[test]
fn actual_group_partition_changes_durable_dot_counts() {
    let plans: Vec<_> = [16, 32, 127].into_iter().map(|width| {
        let p = configured(BuiltInTask::Keyphrases, width, Some(args(BuiltInTask::Keyphrases)), native(width));
        prepare(&p, document())
    }).collect();
    for pair in plans.windows(2) {
        assert_eq!(pair[0].execution_identity().prompt_digest, pair[1].execution_identity().prompt_digest);
        assert!(pair[0].planned_work().projections.dot_products > pair[1].planned_work().projections.dot_products);
        assert_eq!(pair[0].planned_work().projections.multiply_accumulates, pair[1].planned_work().projections.multiply_accumulates);
        let work = JobWork::from_int4(pair[0].planned_work(), 17);
        assert_eq!(work.int4_model_work(), pair[0].planned_work()); assert_eq!(work.mask_node_visits, 17);
    }
}

#[test]
fn complete_planning_native_and_mask_settings_change_recipe_authority() {
    let p = planner(BuiltInTask::Keyphrases); let baseline = freeze(&p);
    for axis in 0..25 {
        let mut changed = planner(BuiltInTask::Keyphrases);
        let r = &mut changed.recipe;
        match axis {
            0 => r.planning.compiler.max_schema_bytes += 1, 1 => r.planning.compiler.max_string_bytes += 1,
            2 => r.planning.compiler.max_array_items += 1, 3 => r.planning.compiler.max_output_bytes += 1,
            4 => r.planning.compiler.max_states += 1, 5 => r.planning.compiler.max_transitions += 1,
            6 => r.planning.compiler.max_mask_bytes += 1, 7 => r.planning.max_input_bytes += 1,
            8 => r.planning.max_context_tokens += 1, 9 => r.planning.max_passages += 1,
            10 => r.planning.source.index.max_source_bytes += 1, 11 => r.planning.source.index.max_index_bytes += 1,
            12 => r.planning.source.index.max_build_steps += 1, 13 => r.planning.source.index.max_matches += 1,
            14 => r.planning.source.verification.max_fields += 1, 15 => r.planning.source.verification.max_matches += 1,
            16 => r.planning.source.verification.max_scan_steps += 1, 17 => r.native.max_model_work.forward_positions += 1,
            18 => r.native.max_model_work.projected_logits += 1, 19 => r.native.max_model_work.attention_pairs += 1,
            20 => r.native.max_model_work.projections.dot_products += 1,
            21 => r.native.max_model_work.projections.multiply_accumulates += 1,
            22 => r.native.max_trie_node_visits += 1, 23 => r.native.checkpoint_interval_nodes += 1,
            _ => r.native.max_visits_per_item += 1,
        }
        assert_eq!(baseline.binding.compare(&freeze(&changed).binding), Err(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
    }
    let mut changed = planner(BuiltInTask::Keyphrases); changed.recipe.native.max_visits_per_run += 1;
    assert_eq!(baseline.binding.compare(&freeze(&changed).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    let encoded = serde_json::to_value(&p.recipe).unwrap();
    assert_eq!(encoded["dependency_scope"], "item-local");
    assert_eq!(encoded["input_policy"], INPUT_POLICY);
    assert_eq!(encoded["execution"], INT4_SOURCE_EXECUTION);
    assert_eq!(encoded["source_runtime"], SOURCE_JSON_RUNTIME_VERSION);
    assert_eq!(encoded["planning"]["source"], serde_json::to_value(p.planning.source).unwrap());
}

#[test]
fn private_qa_passages_options_and_task_budgets_remain_in_the_keyed_recipe() {
    let baseline = freeze(&planner(BuiltInTask::Answer));
    for axis in 0..8 {
        let mut changed = args(BuiltInTask::Answer);
        if let SourceBatchArgs::Answer { passages, options, budget } = &mut changed {
            match axis { 0 => passages[0].id.push('x'), 1 => passages[0].text.push(' '),
                2 => options.max_answer_scalars += 1, 3 => options.max_citations += 1,
                4 => options.max_quote_scalars += 1, 5 => budget.max_output_tokens -= 1,
                6 => budget.max_output_bytes -= 1, _ => budget.max_grammar_states -= 1 }
        }
        let changed = configured(BuiltInTask::Answer, WIDTH, Some(changed), native(WIDTH));
        assert_eq!(baseline.binding.compare(&freeze(&changed).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    let other_group = configured(BuiltInTask::Answer, 16, Some(args(BuiltInTask::Answer)), native(16));
    assert_eq!(baseline.binding.compare(&freeze(&other_group).binding), Err(JobError::Mismatch(MismatchField::Execution)));
}

#[test]
fn factory_rejects_wrong_profile_task_and_full_kv_before_job_storage() {
    for axis in 0..5 {
        let p = source(); let mut id = identity(&p, BuiltInTask::Keyphrases, WIDTH);
        let mut ceiling = budget(); let mut planning = SourcePlanningLimits::default();
        match axis { 0 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            1 => id.task_spec = BuiltInTask::Classify.spec().identity(), 2 => ceiling.max_kv_bytes -= 1,
            3 => ceiling.max_kv_bytes += 1, _ => planning.max_context_tokens -= 1 }
        assert!(Int4SourceJobPlanner::configured(p, id, WIDTH, ceiling, planning,
            Some(args(BuiltInTask::Keyphrases)), native(WIDTH)).is_err());
    }
    for delta in [-1_i64, 1] {
        let p = source(); let id = identity(&p, BuiltInTask::Keyphrases, WIDTH);
        let mut defaults = args(BuiltInTask::Keyphrases);
        argument_budget(&mut defaults).max_kv_bytes = (budget().max_kv_bytes as i64 + delta) as u64;
        assert!(Int4SourceJobPlanner::configured(p, id, WIDTH, budget(), SourcePlanningLimits::default(),
            Some(defaults), native(WIDTH)).is_err());
    }
    let p = source(); let id = identity(&p, BuiltInTask::Answer, WIDTH);
    let mut defaults = args(BuiltInTask::Answer);
    if let SourceBatchArgs::Answer { passages, .. } = &mut defaults { passages[0].text = "x".repeat(1024 * 1024); }
    assert!(Int4SourceJobPlanner::configured(p, id, WIDTH, budget(), SourcePlanningLimits::default(),
        Some(defaults), native(WIDTH)).is_err());
}

#[test]
fn records_may_tighten_budgets_but_cannot_switch_task_or_reprice_resident_kv() {
    let p = planner(BuiltInTask::Keyphrases);
    for axis in 0..6 {
        let mut input = document(); let mut a = args(BuiltInTask::Keyphrases); let b = argument_budget(&mut a);
        match axis { 0 => b.max_input_tokens += 1, 1 => b.max_output_tokens += 1, 2 => b.max_output_bytes += 1,
            3 => b.max_grammar_states += 1, 4 => b.max_kv_bytes -= 1, _ => b.max_kv_bytes += 1 }
        input.task_args = Some(a); assert!(p.checked_document(input).is_err());
    }
    let mut input = document(); input.task_args = Some(args(BuiltInTask::Answer));
    assert!(p.checked_document(input).is_err());
    let mut input = document(); let mut a = args(BuiltInTask::Keyphrases);
    let b = argument_budget(&mut a); b.max_input_tokens -= 1; b.max_output_tokens -= 1;
    b.max_output_bytes -= 1; b.max_grammar_states -= 1; input.task_args = Some(a);
    let plan = prepare(&p, input); assert_eq!(plan.task_budget().max_output_tokens, 127);
}

fn axis_mut(work: &mut JobWork, axis: usize) -> &mut u64 {
    match axis { 0 => &mut work.model.forward_positions, 1 => &mut work.model.projected_logits,
        2 => &mut work.model.attention_pairs, 3 => &mut work.model.projections.dot_products,
        4 => &mut work.model.projections.multiply_accumulates, _ => &mut work.mask_node_visits }
}
#[test]
fn durable_limits_match_every_lifetime_axis_exactly() {
    let p = planner(BuiltInTask::Ner); let exact = limits(p.native_limits()); p.check_job_limits(exact).unwrap();
    for axis in 0..6 {
        for increase in [false, true] {
            let mut changed = exact; let value = axis_mut(&mut changed.max_work, axis);
            if increase { *value += 1; } else { *value -= 1; }
            assert!(p.check_job_limits(changed).is_err());
        }
    }
    let mut changed = exact; changed.max_result_bytes -= 1; assert!(p.check_job_limits(changed).is_err());
}
struct Admission;
impl Int4SourceBatchAdmission for Admission {
    type Guard = ();
    fn admit(&mut self, _: Int4SourceAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn real_native_processor_satisfies_the_durable_boundary_without_fixture_authority() {
    fn durable<P: DurableBatchProcessor>() {}
    durable::<Int4SourceJobProcessor<'static, 'static, 'static, 'static, Admission>>();
}

// Only execution is replaced below. The real source planner, group-aware
// quotes, JobRunner, keyed journal/spool and guarded commit path are exercised.
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-int4-source-job-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
#[derive(Default)]
struct Events { held: Cell<bool>, executions: Cell<usize>, serializations: Cell<usize>,
    releases: Cell<usize>, held_polls: Cell<usize> }
struct Hold(Rc<Events>);
impl Drop for Hold {
    fn drop(&mut self) { assert!(self.0.held.replace(false)); self.0.releases.set(self.0.releases.get() + 1); }
}
struct Datum { ordinal: u64, task: String, events: Rc<Events> }
impl Serialize for Datum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get()); self.events.serializations.set(self.events.serializations.get() + 1);
        ("storage-only-no-model-execution", self.ordinal, &self.task).serialize(serializer)
    }
}
struct Processor { planner: Int4SourceJobPlanner, events: Rc<Events>, fail: bool }
impl Processor {
    fn new(planner: Int4SourceJobPlanner, events: Rc<Events>) -> Self { Self { planner, events, fail: false } }
}
impl BatchProcessor for Processor {
    type Args = SourceBatchArgs;
    type Prepared = PreparedInt4SourceTask;
    type Output = GuardedOutput<Datum, Hold>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        panic!("durable planning must use the caller's control")
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, input: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        self.planner.compiler().map_err(BatchItemFailure::fatal)?
            .prepare_with_control(self.planner.checked_document(input)?, control)
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let w = prepared.planned_work(); BatchWork { forward_positions: w.forward_positions, projected_logits: w.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("durable execution must carry stable delivery coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        self.events.executions.set(self.events.executions.get() + 1);
        if self.fail { return Err(BatchItemFailure::fatal(BatchCode::Execution)); }
        assert!(!self.events.held.replace(true));
        Ok(GuardedOutput::new(Datum { ordinal: context.request_seq,
            task: prepared.execution_identity().task_spec.clone(), events: Rc::clone(&self.events) },
            Hold(Rc::clone(&self.events))))
    }
}
impl DurableBatchProcessor for Processor {
    type Recipe = Int4SourceJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { &self.planner.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        Ok(JobWork::from_int4(prepared.planned_work(), self.planner.native_limits().masks.max_visits_per_item))
    }
    fn max_result_bytes(&self, prepared: &Self::Prepared) -> u64 { prepared.max_result_bytes() }
}
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(e) => e, Ok(_) => panic!("expected refusal") }
}
struct CancelAfterSync(Rc<Events>);
impl DecodeStepControl for CancelAfterSync {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if !self.0.held.get() { return None; }
        let n = self.0.held_polls.get() + 1; self.0.held_polls.set(n);
        (n == 2).then_some(DecodeCancellationKind::PollQuota)
    }
}

#[test]
fn resumed_source_job_skips_commits_and_matches_uninterrupted_materialization() {
    let root = Root::new(); let full = Root::new(); let input = inputs();
    let events = Rc::new(Events::default()); let limits = limits(native(WIDTH));
    let make = || Processor::new(planner(BuiltInTask::Keyphrases), Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &input, make(), &mut Continue).unwrap();
    assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let result = resumed.run(&mut Continue).unwrap();
    assert_eq!(result.committed, 2); assert_eq!(result.attempts, 2);
    assert_eq!(events.executions.get(), 2); assert_eq!(events.releases.get(), 2);
    assert!(events.serializations.get() >= 4); assert!(!events.held.get());
    resumed.verify(&mut Continue).unwrap(); resumed.materialize_ordered(&mut Continue).unwrap();
    let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), limits, &input,
        Processor::new(planner(BuiltInTask::Keyphrases), Rc::new(Events::default())), &mut Continue).unwrap();
    baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), fs::read(full.0.join("materialized.ndjson")).unwrap());
}

#[test]
fn every_failed_attempt_work_axis_remains_spent_after_resume() {
    for axis in 0..6 {
        let p = planner(BuiltInTask::Keyphrases);
        let input = inputs(); let parsed: BatchDocument<SourceBatchArgs> = serde_json::from_slice(input[0].original).unwrap();
        let prepared = prepare(&p, parsed);
        let quote = JobWork::from_int4(prepared.planned_work(), p.native_limits().masks.max_visits_per_item);
        let mut cap = native(WIDTH); let mut frozen = JobWork::from_int4(cap.max_model_work, cap.masks.max_visits_per_run);
        let mut quoted = quote; *axis_mut(&mut frozen, axis) = *axis_mut(&mut quoted, axis);
        cap.max_model_work = frozen.int4_model_work(); cap.masks.max_visits_per_run = frozen.mask_node_visits;
        let events = Rc::new(Events::default()); let root = Root::new(); let limits = limits(cap);
        let make = || configured(BuiltInTask::Keyphrases, WIDTH, Some(args(BuiltInTask::Keyphrases)), cap);
        let mut processor = Processor::new(make(), Rc::clone(&events)); processor.fail = true;
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &input, processor, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!(run.progress().reserved_work, quote); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
            Processor::new(make(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1); assert_eq!(resumed.progress().reserved_work, quote);
        assert_eq!(resumed.progress().attempts, 1);
    }
}

#[test]
fn post_sync_cancellation_preserves_work_and_authenticates_before_tail_repair() {
    let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
    let limits = limits(native(WIDTH)); let mut control = CancelAfterSync(Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &input,
        Processor::new(planner(BuiltInTask::Answer), Rc::clone(&events)), &mut control).unwrap();
    assert_eq!(error(run.step(&mut control)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    let spent = run.progress().reserved_work;
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().attempts, 1);
    assert_eq!(spent.mask_node_visits, native(WIDTH).masks.max_visits_per_item);
    assert!(!events.held.get()); assert_eq!(events.releases.get(), 1); drop(run);
    let make = || Processor::new(planner(BuiltInTask::Answer), Rc::clone(&events));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut changed = args(BuiltInTask::Answer);
    if let SourceBatchArgs::Answer { passages, .. } = &mut changed { passages[0].text.push(' '); }
    let changed = configured(BuiltInTask::Answer, WIDTH, Some(changed), native(WIDTH));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        Processor::new(changed, Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    assert_eq!(resumed.progress().reserved_work, spent);
    assert_eq!(resumed.step(&mut Continue).unwrap().unwrap().reserved_work, spent.checked_add(spent).unwrap());
    assert_eq!(resumed.run(&mut Continue).unwrap().committed, 2);
}


fn compact_limits() -> (crate::tokenizer::untrusted::SafeBpeLimits, SafeBpeWorkLimits) {
    (crate::tokenizer::untrusted::SafeBpeLimits { max_input_bytes: 8192,
        max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 },
     SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 32_768,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 })
}
fn compact_planner(task: BuiltInTask, call: crate::tokenizer::untrusted::SafeBpeLimits,
    total: SafeBpeWorkLimits) -> Int4SourceJobPlanner {
    let source = Arc::try_unwrap(source()).ok().unwrap()
        .with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), total).unwrap();
    let id = identity(&source, task, WIDTH);
    Int4SourceJobPlanner::configured_with_document_encoding(Arc::new(source), id, WIDTH,
        budget(), SourcePlanningLimits::default(), Some(args(task)), native(WIDTH)).unwrap()
}

#[test]
fn compact_source_jobs_keep_fixed_identity_as_one_invocation_ledger_is_spent() {
    let (call, total) = compact_limits();
    for task in [BuiltInTask::Ner, BuiltInTask::Keyphrases, BuiltInTask::Summarize, BuiltInTask::Answer] {
        let p = compact_planner(task, call, total);
        let first = prepare(&p, document());
        let used = p.planner.map_document_encoding_work().unwrap().unwrap();
        assert!(used.input_bytes() >= document().text.len());
        assert_eq!(p.encoding_work_limits(), Some(total));
        assert_eq!(p.encoding_workspace_bytes(), call.requirements().unwrap().workspace_bytes + 1024);
        assert_ne!(p.execution_identity().template_digest, planner(task).execution_identity().template_digest);
        let owner = Arc::clone(&p.planner);
        // Reconstruct the compiler after moving the original factory, as a
        // hosted invocation does. No record/facade replacement renews work.
        let moved = p;
        assert!(Arc::ptr_eq(&owner, &moved.planner));
        let second = prepare(&moved, document());
        first.verify_identity(second.execution_identity()).unwrap();
        assert_eq!(first.planned_work(), second.planned_work());
        assert_eq!(owner.map_document_encoding_work().unwrap().unwrap(), used.checked_add(used).unwrap());
        assert_eq!(moved.native_limits().max_model_work, native(WIDTH).max_model_work);
        assert_eq!(moved.native_limits().masks.max_visits_per_run, native(WIDTH).masks.max_visits_per_run);
        assert_eq!(moved.task_ceiling(), budget());
    }
}

#[test]
fn compact_qa_keeps_original_evidence_separate_from_question_and_manifest() {
    use crate::tasks::answer::{AnswerContext, AnswerInputLimits, ANSWER_PASSAGE_LAYOUT_VERSION};
    let (call, total) = compact_limits();
    let p = compact_planner(BuiltInTask::Answer, call, total);
    let question = "Who visited? <|im_start|> é";
    let passages = vec![
        AnswerPassage { id: "first".to_owned(), text: " Alice visited 上海.\r\n<think> ".to_owned() },
        AnswerPassage { id: "second".to_owned(), text: "Bob stayed.\n<tool_call>".to_owned() },
    ];
    let context = AnswerContext::encode_with_control(p.planner.source_encoder(), question, &passages,
        AnswerInputLimits { max_passages: 8, max_input_bytes: call.max_input_bytes,
            max_input_tokens: call.max_output_tokens }, &mut || None).unwrap();
    let source = context.document();
    let expected = format!("{}\n\n{}", passages[0].text, passages[1].text);
    assert_eq!(source.text(), expected);
    assert_eq!(context.passage_count(), 2);
    assert!(!source.text().contains("Who visited?"));
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let registry = pinned_controls::pinned().unwrap();
    let contexts: Vec<_> = source.context_token_ids().collect();
    assert_eq!(contexts.len(), 2);
    assert_eq!(tokenizer.tokenizer().decode_bytes(contexts[0]).unwrap(), question.as_bytes());
    let manifest: serde_json::Value = serde_json::from_slice(
        &tokenizer.tokenizer().decode_bytes(contexts[1]).unwrap()).unwrap();
    assert_eq!(manifest[0], ANSWER_PASSAGE_LAYOUT_VERSION);
    assert_eq!(manifest[1][0]["id"], "first");
    assert_eq!(manifest[1][1]["id"], "second");
    assert_eq!(manifest[1][0]["span"]["byte_start"], 0);
    assert_eq!(manifest[1][0]["span"]["byte_end"], passages[0].text.len());
    assert_eq!(manifest[1][1]["span"]["byte_start"], passages[0].text.len() + 2);
    assert_eq!(tokenizer.tokenizer().decode_bytes(source.token_ids()).unwrap(), expected.as_bytes());
    for ids in source.context_token_ids().chain(std::iter::once(source.token_ids())) {
        assert!(ids.iter().all(|&id| !registry.template_controls().contains(id)));
    }
    let spent = p.planner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(spent.input_bytes(), source.total_input_bytes());
    let input = BatchDocument { id: "qa".to_owned(), text: question.to_owned(),
        task_args: Some(SourceBatchArgs::Answer { passages, options: AnswerOptions::default(), budget: budget() }) };
    prepare(&p, input);
    assert_eq!(p.planner.map_document_encoding_work().unwrap().unwrap(), spent.checked_add(spent).unwrap());
}

#[test]
fn compact_job_requires_an_explicit_cumulative_owner_and_preserves_byte_recipe_shape() {
    let (call, _) = compact_limits();
    let unbounded = Arc::try_unwrap(source()).ok().unwrap()
        .with_document_encoding(UntrustedEncoding::SafeBpeV1(call)).unwrap();
    let id = identity(&unbounded, BuiltInTask::Keyphrases, WIDTH);
    assert!(Int4SourceJobPlanner::configured_with_document_encoding(Arc::new(unbounded), id,
        WIDTH, budget(), SourcePlanningLimits::default(),
        Some(args(BuiltInTask::Keyphrases)), native(WIDTH)).is_err());
    let (call, total) = compact_limits();
    let compact = compact_planner(BuiltInTask::Keyphrases, call, total);
    assert!(Int4SourceJobPlanner::configured(Arc::clone(&compact.planner),
        compact.execution_identity().clone(), WIDTH, budget(), SourcePlanningLimits::default(),
        Some(args(BuiltInTask::Keyphrases)), native(WIDTH)).is_err());
    let original = planner(BuiltInTask::Keyphrases);
    let source = source(); let id = identity(&source, BuiltInTask::Keyphrases, WIDTH);
    let explicit = Int4SourceJobPlanner::configured_with_document_encoding(source, id, WIDTH,
        budget(), SourcePlanningLimits::default(), Some(args(BuiltInTask::Keyphrases)), native(WIDTH)).unwrap();
    assert_eq!(canonjson::canonical_bytes(&original.recipe).unwrap(),
        canonjson::canonical_bytes(&explicit.recipe).unwrap());
    assert_eq!(freeze(&original).binding.compare(&freeze(&explicit).binding), Ok(()));
    let value = serde_json::to_value(&explicit.recipe).unwrap();
    assert!(value.get("input_encoding").is_none());
    assert!(value.get("encoding_work_limits").is_none());
    assert_eq!(explicit.encoding_workspace_bytes(), 0);
    assert_eq!(explicit.encoding_work_limits(), None);
}

#[test]
fn all_compact_call_and_invocation_limits_are_authenticated_before_resume() {
    let (call, total) = compact_limits();
    let original = compact_planner(BuiltInTask::Keyphrases, call, total);
    let binding = freeze(&original);
    let recipe = canonjson::canonical_bytes(&original.recipe).unwrap();
    for axis in 0..8 {
        let mut changed_call = call; let mut changed_total = total;
        match axis {
            0 => changed_call.max_input_bytes -= 1,
            1 => changed_call.max_output_tokens -= 1,
            2 => changed_call.max_pair_checks -= 1,
            3 => changed_call.max_pair_bytes -= 1,
            4 => changed_total.max_input_bytes -= 1,
            5 => changed_total.max_output_tokens -= 1,
            6 => changed_total.max_pair_checks -= 1,
            _ => changed_total.max_pair_bytes -= 1,
        }
        let changed = compact_planner(BuiltInTask::Keyphrases, changed_call, changed_total);
        assert_ne!(recipe, canonjson::canonical_bytes(&changed.recipe).unwrap(), "axis {axis}");
        assert_eq!(binding.binding.compare(&freeze(&changed).binding),
            Err(JobError::Mismatch(MismatchField::Execution)), "axis {axis}");
    }
}

#[test]
fn compact_preparation_and_failed_tokenization_never_renew_the_invocation_allowance() {
    let (call, mut total) = compact_limits();
    total.max_input_bytes = document().text.len() as u64;
    let p = compact_planner(BuiltInTask::Keyphrases, call, total);
    prepare(&p, document());
    let used = p.planner.map_document_encoding_work().unwrap().unwrap();
    let moved = p;
    assert!(moved.compiler().unwrap().prepare_with_control(document(), &mut Continue).is_err());
    assert_eq!(moved.planner.map_document_encoding_work().unwrap().unwrap(), used);

    let mut too_tight = call; too_tight.max_output_tokens = 1;
    let p = compact_planner(BuiltInTask::Keyphrases, too_tight, total);
    assert!(p.compiler().unwrap().prepare_with_control(document(), &mut Continue).is_err());
    let used = p.planner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(used.input_bytes(), document().text.len());
    assert!(used.output_tokens() > 0);
    assert!(p.compiler().unwrap().prepare_with_control(document(), &mut Continue).is_err());
    assert_eq!(p.planner.map_document_encoding_work().unwrap().unwrap(), used);
}

#[test]
fn cancellation_inside_compact_encoding_keeps_all_consumed_work_and_the_original_cause() {
    use crate::{tasks::extract::ExtractError, native_engine::constrained::JsonDecodeError};
    let text = "Alice private café Shanghai";
    let (call, mut total) = compact_limits(); total.max_input_bytes = text.len() as u64;
    let p = compact_planner(BuiltInTask::Keyphrases, call, total);
    let mut checkpoints = 0;
    let refusal = p.planner.source_encoder().encode_with_control(text, call.max_input_bytes,
        call.max_output_tokens, &mut || {
            checkpoints += 1;
            // Source admission, untrusted entry, first byte, ordinary entry,
            // first symbol, then the first already-charged pair comparison.
            (checkpoints == 6).then_some(DecodeCancellationKind::ResourceUnavailable)
        }).err().unwrap();
    assert!(matches!(refusal, ExtractError::Decode(
        JsonDecodeError::Cancelled(DecodeCancellationKind::ResourceUnavailable))));
    let spent = p.planner.map_document_encoding_work().unwrap().unwrap();
    assert_eq!(spent.input_bytes(), text.len());
    assert!(spent.output_tokens() > 0 && spent.pair_checks() > 0 && spent.pair_bytes() > 0);
    let input = BatchDocument { id: "retry".to_owned(), text: text.to_owned(), task_args: None };
    assert!(p.compiler().unwrap().prepare_with_control(input, &mut Continue).is_err());
    assert_eq!(p.planner.map_document_encoding_work().unwrap().unwrap(), spent);
}

#[test]
fn compact_resume_rejects_policy_drift_before_tail_repair_and_keeps_native_debits() {
    let (call, total) = compact_limits();
    let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
    let limits = limits(native(WIDTH));
    let make = || Processor::new(compact_planner(BuiltInTask::Keyphrases, call, total), Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &input,
        make(), &mut Continue).unwrap();
    assert_eq!(error(run.step(&mut CancelAfterSync(Rc::clone(&events)))),
        JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    let spent = run.progress().reserved_work;
    assert_eq!(run.progress().attempts, 1); drop(run);
    let mut changed = total; changed.max_pair_bytes -= 1;
    let wrong = Processor::new(compact_planner(BuiltInTask::Keyphrases, call, changed), Rc::clone(&events));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        wrong, TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Execution)));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &input,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    assert_eq!(resumed.progress().reserved_work, spent);
    assert_eq!(resumed.progress().attempts, 1);
    assert_eq!(resumed.step(&mut Continue).unwrap().unwrap().reserved_work, spent.checked_add(spent).unwrap());
}
