//! Recipe authentication and real owned-job storage with synthetic execution.
//! No fixture scorer can construct the public native processor or certify NLP.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{classify::ClassificationBatchArgs, judge::JudgeBatchArgs, scored_int4::Int4ScoredAdmission},
    canonjson,
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField,
        TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{decode::DecodeCancellationKind, generation::grouped_int4::INT4_SOURCE_FORMAT,
        lmhead::scoring::ScoringLimits, portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::{Int4MemoryRequirement, Int4Work, STRICT_INT4_EXECUTION}},
    tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode, ClassificationPolicy,
            grouped_int4::Int4ClassificationPlanner},
        sentiment::{SentimentAxis, SentimentLimits, batch::SentimentBatchArgs, grouped_int4::Int4SentimentPlanner},
        judge::{JudgeLimits, PairwisePolicy, RubricDefinition, RubricCriterion, RubricPolicy,
            FaithfulnessPolicy, grouped_int4::Int4JudgePlanner}},
};
use serde::{Serialize, Serializer, de::DeserializeOwned};
use std::{cell::Cell, fs, marker::PhantomData, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::atomic::{AtomicU64, Ordering}};

const WIDTH: usize = 128;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(8192, WIDTH).unwrap().kv_bytes }
}
fn work(units: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * units, projected_logits: 50 * units, attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn native() -> Int4ScoredBatchLimits { Int4ScoredBatchLimits { max_item_work: work(2), max_model_work: work(10) } }
fn planning() -> ClassificationLimits { ClassificationLimits { max_context_tokens: 4096, ..Default::default() } }
fn labels() -> ClassificationBatchArgs {
    ClassificationBatchArgs { labels: vec![
        ClassificationLabel { id: "yes".to_owned(), description: "Private label é 上海".to_owned() },
        ClassificationLabel { id: "no".to_owned(), description: "Other category".to_owned() }],
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: budget() }
}
fn sentiment() -> SentimentBatchArgs { SentimentBatchArgs { axes: SentimentAxis::ALL.to_vec(), budget: budget() } }
fn judgment(mode: usize) -> JudgeBatchArgs {
    match mode {
        0 => JudgeBatchArgs::Pairwise { criterion: "clarity".to_owned(), b: "Private candidate B".to_owned(),
            policy: PairwisePolicy { minimum_margin_milli: u32::MAX, maximum_order_disagreement_milli: u32::MAX }, budget: budget() },
        1 => JudgeBatchArgs::Rubric { rubric: RubricDefinition { schema_version: 1, revision: "revision-1".to_owned(),
            declared_origin_digest: Sha256Digest::of_bytes(b"fixture-public-origin"), scale_maximum: 4,
            criteria: vec![RubricCriterion { id: "clarity".to_owned(), description: "Private rubric".to_owned(), weight: 1 }] },
            policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }, budget: budget() },
        _ => JudgeBatchArgs::Faithfulness { claim: "Private claim".to_owned(), policy: FaithfulnessPolicy {
            minimum_candidate_weight_ppm: 0, minimum_margin_milli: u32::MAX, evidence_window_bytes: 64,
            max_evidence_windows: 8, max_evidence_spans: 4 }, budget: budget() },
    }
}
fn recipe(task: usize, cap: Int4ScoredBatchLimits) -> Int4ScoredJobRecipe {
    match task {
        0 => Int4ScoredJobRecipe::classification(WIDTH, budget(), planning(), Some(labels()), cap).unwrap(),
        1 => Int4ScoredJobRecipe::sentiment(WIDTH, budget(), SentimentLimits::default(), Some(sentiment()), cap).unwrap(),
        _ => Int4ScoredJobRecipe::judge(WIDTH, budget(), JudgeLimits::default(), Some(judgment(task - 2)), cap).unwrap(),
    }
}
fn identity(task: usize) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"scored-job-fixture-no-model-authority");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, WIDTH)).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: digest,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: packing, tokenizer_digest: digest, template_digest: digest,
        task_spec: match task { 0 => "classify-v1", 1 => "sentiment-v1", _ => "judge-v1" }.to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn limits(cap: Int4ScoredBatchLimits) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 8,
        max_work: JobWork::from_int4(cap.max_model_work, 0) }
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Private source A"}"#;
    let b = br#"{"id":"beta","text":"Private source B","task_args":null}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn freeze(task: usize, recipe: &Int4ScoredJobRecipe) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: &identity(task),
        recipe, limits: limits(native()) }, inputs(), &mut Continue).unwrap()
}
fn bytes(recipe: &Int4ScoredJobRecipe) -> Vec<u8> { canonjson::canonical_bytes(recipe).unwrap() }
fn alter_scoring(l: &mut ScoringLimits, axis: usize) {
    match axis { 0 => l.max_candidates += 1, 1 => l.max_total_tokens += 1, 2 => l.max_nodes += 1,
        3 => l.max_depth += 1, 4 => l.max_candidate_id_bytes += 1, _ => l.max_projected_logits += 1 }
}
fn work_axis(w: &mut Int4Work, axis: usize) -> &mut u64 {
    match axis { 0 => &mut w.forward_positions, 1 => &mut w.projected_logits, 2 => &mut w.attention_pairs,
        3 => &mut w.projections.dot_products, _ => &mut w.projections.multiply_accumulates }
}

#[test]
fn every_classification_planning_field_is_part_of_the_private_recipe() {
    let original = bytes(&recipe(0, native()));
    for axis in 0..15 {
        let mut l = planning();
        match axis { 0 => l.max_labels += 1, 1 => l.max_input_bytes += 1, 2 => l.max_label_id_bytes += 1,
            3 => l.max_label_description_bytes += 1, 4 => l.max_total_label_bytes += 1, 5 => l.max_context_tokens += 1,
            6 => l.max_total_prompt_tokens += 1, 7 => l.max_work.forward_positions += 1,
            8 => l.max_work.projected_logits += 1, _ => alter_scoring(&mut l.scoring, axis - 9) }
        let changed = Int4ScoredJobRecipe::classification(WIDTH, budget(), l, Some(labels()), native()).unwrap();
        assert_ne!(original, bytes(&changed), "axis {axis}");
        assert_eq!(freeze(0, &recipe(0, native())).binding.compare(&freeze(0, &changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}
#[test]
fn every_sentiment_and_judge_planning_field_is_frozen() {
    let original = bytes(&recipe(1, native()));
    for axis in 0..11 {
        let mut l = SentimentLimits::default();
        match axis { 0..=5 => alter_scoring(&mut l.per_axis, axis), 6 => l.max_total_prompt_tokens += 1,
            7 => l.max_total_candidate_tokens += 1, 8 => l.max_total_nodes += 1,
            9 => l.max_total_projected_logits += 1, _ => l.max_output_bytes += 1 }
        assert_ne!(original, bytes(&Int4ScoredJobRecipe::sentiment(WIDTH, budget(), l, Some(sentiment()), native()).unwrap()));
    }
    let original = bytes(&recipe(2, native()));
    for axis in 0..10 {
        let mut l = JudgeLimits::default();
        match axis { 0..=5 => alter_scoring(&mut l.per_head, axis), 6 => l.max_total_prompt_tokens += 1,
            7 => l.max_total_candidate_tokens += 1, 8 => l.max_total_projected_logits += 1, _ => l.max_output_bytes += 1 }
        assert_ne!(original, bytes(&Int4ScoredJobRecipe::judge(WIDTH, budget(), l, Some(judgment(0)), native()).unwrap()));
    }
}
#[test]
fn all_task_budget_native_work_and_group_fields_change_resume_authority() {
    let original = freeze(0, &recipe(0, native()));
    for axis in 0..16 {
        let mut ceiling = budget(); let mut cap = native(); let mut width = WIDTH;
        match axis { 0 => ceiling.max_input_tokens += 1, 1 => ceiling.max_output_tokens += 1,
            2 => ceiling.max_output_bytes += 1, 3 => ceiling.max_grammar_states += 1, 4 => ceiling.max_kv_bytes += 1,
            5..=9 => *work_axis(&mut cap.max_item_work, axis - 5) += 1,
            10..=14 => *work_axis(&mut cap.max_model_work, axis - 10) += 1, _ => width = 32 }
        let changed = Int4ScoredJobRecipe::classification(width, ceiling, planning(), Some(labels()), cap).unwrap();
        assert_eq!(original.binding.compare(&freeze(0, &changed).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
}
#[test]
fn typed_default_content_order_policies_and_all_judge_modes_are_authenticated() {
    let original = freeze(0, &recipe(0, native()));
    for axis in 0..5 {
        let mut a = labels();
        match axis { 0 => a.labels[0].description.push('!'), 1 => a.labels.swap(0, 1),
            2 => a.mode = ClassificationMode::MultiLabel, 3 => a.policy.minimum_margin_ppm += 1,
            _ => a.budget.max_input_tokens -= 1 }
        let changed = Int4ScoredJobRecipe::classification(WIDTH, budget(), planning(), Some(a), native()).unwrap();
        assert_eq!(original.binding.compare(&freeze(0, &changed).binding), Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    let mut a = sentiment(); a.axes.reverse();
    let changed = Int4ScoredJobRecipe::sentiment(WIDTH, budget(), SentimentLimits::default(), Some(a), native()).unwrap();
    assert_eq!(freeze(1, &recipe(1, native())).binding.compare(&freeze(1, &changed).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    for mode in 0..3 {
        let mut a = judgment(mode);
        match &mut a { JudgeBatchArgs::Pairwise { b, .. } => b.push('!'),
            JudgeBatchArgs::Rubric { rubric, .. } => rubric.criteria[0].description.push('!'),
            JudgeBatchArgs::Faithfulness { claim, .. } => claim.push('!') }
        let changed = Int4ScoredJobRecipe::judge(WIDTH, budget(), JudgeLimits::default(), Some(a), native()).unwrap();
        assert_eq!(freeze(mode + 2, &recipe(mode + 2, native())).binding.compare(&freeze(mode + 2, &changed).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    // Existing milli-log-odds policy domain is the complete u32 domain.
    assert!(String::from_utf8(bytes(&recipe(2, native()))).unwrap().contains("4294967295"));
}
#[test]
fn invalid_defaults_limits_and_oversized_private_recipes_refuse_before_native_cloning() {
    let mut bad = labels(); bad.labels[1].id = bad.labels[0].id.clone();
    assert!(Int4ScoredJobRecipe::classification(WIDTH, budget(), planning(), Some(bad), native()).is_err());
    let mut bad = labels(); bad.budget.max_kv_bytes += 1;
    assert!(Int4ScoredJobRecipe::classification(WIDTH, budget(), planning(), Some(bad), native()).is_err());
    let mut bad = sentiment(); bad.axes.push(bad.axes[0]);
    assert!(Int4ScoredJobRecipe::sentiment(WIDTH, budget(), SentimentLimits::default(), Some(bad), native()).is_err());
    let mut l = planning(); l.max_context_tokens = 8193;
    assert!(Int4ScoredJobRecipe::classification(WIDTH, budget(), l, None, native()).is_err());
    for axis in 0..5 {
        let mut n = native(); *work_axis(&mut n.max_item_work, axis) = 0;
        assert!(Int4ScoredJobRecipe::judge(WIDTH, budget(), JudgeLimits::default(), None, n).is_err());
    }
    let mut huge = judgment(0);
    if let JudgeBatchArgs::Pairwise { b, .. } = &mut huge { *b = "x".repeat(1024 * 1024); }
    assert!(Int4ScoredJobRecipe::judge(WIDTH, budget(), JudgeLimits::default(), Some(huge), native()).is_err());
    assert!(Int4ScoredJobRecipe::judge(0, budget(), JudgeLimits::default(), None, native()).is_err());
}
#[test]
fn job_limits_equal_all_five_lifetime_axes_and_require_zero_mask_work() {
    let r = recipe(0, native()); let exact = limits(native()); r.check_job_limits(exact).unwrap();
    for axis in 0..5 {
        for increase in [false, true] {
            let mut changed = exact; let mut work = changed.max_work.int4_model_work();
            let value = work_axis(&mut work, axis); if increase { *value += 1; } else { *value -= 1; }
            changed.max_work = JobWork::from_int4(work, 0); assert!(r.check_job_limits(changed).is_err());
        }
    }
    let mut changed = exact; changed.max_work.mask_node_visits = 1; assert!(r.check_job_limits(changed).is_err());
    let mut changed = exact; changed.max_result_bytes -= 1; assert!(r.check_job_limits(changed).is_err());
}
struct Admission;
impl Int4ScoredBatchAdmission for Admission {
    type Guard = ();
    fn admit(&mut self, _: Int4ScoredAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
#[test]
fn every_actual_scoring_task_has_the_same_owned_durable_native_boundary() {
    fn durable<P: Int4ScoredJobTask>() {
        fn bound<T: DurableBatchProcessor>() {} fn send<T: Send + 'static>() {}
        bound::<Int4ScoredJobProcessor<'static, 'static, P, Admission>>();
        send::<Int4ScoredJobPlanner<P>>();
    }
    durable::<Int4ClassificationPlanner>(); durable::<Int4SentimentPlanner>(); durable::<Int4JudgePlanner>();
}

// Below, only the scorer is synthetic. JobRunner, keyed manifests, journal,
// spool frames and materialization execute their real owned-storage paths.
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-int4-scored-job-{}-{}",
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
struct Datum { sequence: u64, task: String, events: Rc<Events> }
impl Serialize for Datum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get()); self.events.serializations.set(self.events.serializations.get() + 1);
        ("storage-fixture-not-native-inference", self.sequence, &self.task).serialize(serializer)
    }
}
struct Processor<A> { identity: ExecutionIdentity, recipe: Int4ScoredJobRecipe,
    events: Rc<Events>, fail: bool, marker: PhantomData<A> }
impl<A> Processor<A> {
    fn new(task: usize, cap: Int4ScoredBatchLimits, events: Rc<Events>) -> Self {
        Self { identity: identity(task), recipe: recipe(task, cap), events, fail: false, marker: PhantomData }
    }
}
impl<A: DeserializeOwned> BatchProcessor for Processor<A> {
    type Args = A; type Prepared = BatchDocument<A>; type Output = GuardedOutput<Datum, Hold>;
    fn prepare(&mut self, _: BatchDocument<A>) -> Result<Self::Prepared, BatchItemFailure> { panic!("missing planning control") }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, doc: BatchDocument<A>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        if let Some(cause) = control.prefill_checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        Ok(doc)
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: work(1).forward_positions, projected_logits: work(1).projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("missing durable delivery coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, _: Self::Prepared, context: BatchRequestContext,
        _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        self.events.executions.set(self.events.executions.get() + 1);
        if self.fail { return Err(BatchItemFailure::fatal(BatchCode::Execution)); }
        assert!(!self.events.held.replace(true));
        Ok(GuardedOutput::new(Datum { sequence: context.request_seq, task: self.identity.task_spec.clone(),
            events: Rc::clone(&self.events) }, Hold(Rc::clone(&self.events))))
    }
}
impl<A: DeserializeOwned> DurableBatchProcessor for Processor<A> {
    type Recipe = Int4ScoredJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.recipe.check_job_limits(limits) }
    fn durable_work(&self, _: &Self::Prepared) -> Result<JobWork, BatchItemFailure> { Ok(JobWork::from_int4(work(1), 0)) }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { budget().max_output_bytes }
}
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(error) => error, Ok(_) => panic!("expected refusal") }
}

#[test]
fn all_three_typed_tasks_resume_only_uncommitted_items_with_identical_ordered_materialization() {
    fn check<A: DeserializeOwned>(task: usize) {
        let root = Root::new(); let full = Root::new(); let input = inputs(); let cap = native();
        let events = Rc::new(Events::default()); let make = || Processor::<A>::new(task, cap, Rc::clone(&events));
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input, make(), &mut Continue).unwrap();
        assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            make(), TailPolicy::Refuse, &mut Continue).unwrap();
        let done = resumed.run(&mut Continue).unwrap();
        assert_eq!((done.committed, done.attempts), (2, 2)); assert_eq!(done.reserved_work, JobWork::from_int4(work(2), 0));
        assert_eq!(events.executions.get(), 2); assert_eq!(events.releases.get(), 2); assert!(!events.held.get());
        assert!(events.serializations.get() >= 4); resumed.verify(&mut Continue).unwrap();
        resumed.materialize_ordered(&mut Continue).unwrap();
        let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), limits(cap), &input,
            Processor::<A>::new(task, cap, Rc::new(Events::default())), &mut Continue).unwrap();
        baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
        assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), fs::read(full.0.join("materialized.ndjson")).unwrap());
    }
    check::<ClassificationBatchArgs>(0); check::<SentimentBatchArgs>(1);
    for mode in 0..3 { check::<JudgeBatchArgs>(mode + 2); }
}
#[test]
fn failed_attempts_consume_every_lifetime_axis_across_a_fresh_processor() {
    for axis in 0..5 {
        let mut cap = native(); cap.max_item_work = work(1);
        *work_axis(&mut cap.max_model_work, axis) = *work_axis(&mut work(1), axis);
        let root = Root::new(); let input = inputs(); let events = Rc::new(Events::default());
        let mut p = Processor::<JudgeBatchArgs>::new(2, cap, Rc::clone(&events)); p.fail = true;
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input, p, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!(run.progress().reserved_work, JobWork::from_int4(work(1), 0)); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
            Processor::<JudgeBatchArgs>::new(2, cap, Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1); assert_eq!(resumed.progress().attempts, 1);
        assert_eq!(resumed.progress().reserved_work, JobWork::from_int4(work(1), 0));
    }
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
fn post_sync_cancellation_releases_guard_and_authenticates_recipe_before_tail_repair() {
    let root = Root::new(); let input = inputs(); let cap = native(); let events = Rc::new(Events::default());
    let make = || Processor::<ClassificationBatchArgs>::new(0, cap, Rc::clone(&events));
    let mut stop = CancelAfterSync(Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(cap), &input, make(), &mut stop).unwrap();
    assert_eq!(error(run.step(&mut stop)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().attempts, 1);
    assert_eq!(run.progress().reserved_work, JobWork::from_int4(work(1), 0));
    assert_eq!(events.releases.get(), 1); assert!(!events.held.get()); drop(run);
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut changed = make(); let mut defaults = labels(); defaults.labels[0].description.push('!');
    changed.recipe = Int4ScoredJobRecipe::classification(WIDTH, budget(), planning(), Some(defaults), cap).unwrap();
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
        changed, TailPolicy::DiscardUncommitted, &mut Continue)), JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(cap), &input,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    let done = resumed.run(&mut Continue).unwrap(); assert_eq!((done.committed, done.attempts), (2, 3));
    assert_eq!(done.reserved_work, JobWork::from_int4(work(3), 0));
}
