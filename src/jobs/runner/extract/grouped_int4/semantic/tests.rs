//! Genuine pinned planning plus authenticated storage fixtures. No native
//! forward, model quality, filesystem crash or execution parity is claimed.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchGrounding, semantic::Int4SemanticExtractionAdmission},
    jobs::{FrozenManifest, InputRetentionLimits, JobContract, JobError, JobId, JobInput,
        JobSecret, MismatchField, TailPolicy, runner::{JobRunError, JobRunner}},
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork,
        strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, weights::projection_fixture}},
    tasks::{ir::TaskBudget, extract::semantic::{ClaimPathStep, ClaimRule, ClaimValueKind,
        SemanticLimits, SemanticVerificationSpec}, judge::FaithfulnessPolicy},
    tokenizer::{pinned_controls, untrusted::SafeBpeLimits},
};
use std::{cell::{Cell, RefCell}, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::{OnceLock, atomic::{AtomicU64, Ordering}}};

fn work(n: u64) -> Int4Work {
    Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }
}
fn model() -> &'static Int4ModelIdentity {
    static MODEL: OnceLock<Int4ModelIdentity> = OnceLock::new();
    MODEL.get_or_init(|| projection_fixture(32).0.identity().clone())
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn compact() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 65_536, max_output_tokens: 8192,
        max_pair_checks: 4_000_000, max_pair_bytes: 16_000_000 }
}
fn config() -> Int4SemanticExtractionJobConfig {
    Int4SemanticExtractionJobConfig {
        extraction: ExtractionBatchArgs { schema: " {\"type\":\"number\"} \n".to_owned(),
            grounding: ExtractionBatchGrounding::Structural, budget: budget() },
        compiler_limits: CompileLimits::default(), source_limits: SourceRuntimeLimits::default(),
        options: Int4SemanticOptions { specification: SemanticVerificationSpec {
            schema_version: 1, experimental_opt_in: true, revision: "durable-review-v1".to_owned(),
            claims: vec![ClaimRule { id: "amount".to_owned(), path: vec![], value_kind: ClaimValueKind::Number,
                prefix: "The original document reports the amount ".to_owned(), suffix: ".".to_owned() }],
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 },
            judge_budget: budget() }, limits: SemanticLimits { max_output_bytes: 1 << 20,
                ..SemanticLimits::default() }, max_model_work: work(1 << 56) },
        judge_planning: JudgeLimits::default(), encoding: UntrustedEncoding::ByteFallback,
        work: Int4SemanticExtractionBatchLimits { max_model_work: work(1 << 58),
            masks: ExtractionMaskBudget { per_mask: MaskWorkLimits::default(),
                max_visits_per_item: 1_000_000_000, max_visits_per_run: 4_000_000_000 } },
    }
}
fn planner() -> Int4SemanticExtractionJobPlanner {
    Int4SemanticExtractionJobPlanner::for_model(model(), config()).unwrap()
}
fn limits(c: &Int4SemanticExtractionJobConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: c.options.limits.max_output_bytes as usize,
        max_spool_bytes: 8 << 20, max_materialized_bytes: 8 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(c.work.max_model_work, c.work.masks.max_visits_per_run) }
}
fn quote(c: &Int4SemanticExtractionJobConfig) -> JobWork {
    JobWork::from_int4(c.options.max_model_work, c.work.masks.max_visits_per_item)
}
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Alice reports amount 7."}"#;
    let b = br#"{"id":"beta","text":"Bob reports amount 9.","task_args":null}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn freeze(p: &Int4SemanticExtractionJobPlanner) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: p.execution_identity(),
        recipe: &p.recipe, limits: limits(p.config()) }, inputs(), &mut Continue).unwrap()
}
fn axis(work: &mut JobWork, n: usize) -> &mut u64 {
    match n { 0 => &mut work.model.forward_positions, 1 => &mut work.model.projected_logits,
        2 => &mut work.model.attention_pairs, 3 => &mut work.model.projections.dot_products,
        4 => &mut work.model.projections.multiply_accumulates, _ => &mut work.mask_node_visits }
}

#[test]
fn exact_schema_claims_both_actual_identities_and_encoding_scope_are_frozen() {
    let p = planner();
    let recipe = serde_json::to_value(&p.recipe).unwrap();
    assert_eq!(recipe["settings"]["extraction"]["schema"], config().extraction.schema);
    assert_eq!(recipe["settings"]["options"], serde_json::to_value(config().options).unwrap());
    assert_eq!(recipe["extraction_identity"], serde_json::to_value(p.execution_identity()).unwrap());
    assert_eq!(recipe["judge_identity"], serde_json::to_value(p.judge_identity()).unwrap());
    assert_eq!(recipe["encoding_scope"], ENCODING_SCOPE);
    assert_eq!(recipe["input_policy"], INPUT_POLICY);
    assert_eq!(recipe["group_width"], 32);
    assert_eq!(p.encoding_workspace_bytes().unwrap(), 0);
    let mut c = config(); c.encoding = UntrustedEncoding::SafeBpeV1(compact());
    let encoded = Int4SemanticExtractionJobPlanner::for_model(model(), c).unwrap();
    assert_eq!(encoded.encoding_workspace_bytes().unwrap(), compact().requirements().unwrap().workspace_bytes);
    assert_ne!(p.execution_identity().template_digest, encoded.execution_identity().template_digest);
    assert_ne!(p.judge_identity().template_digest, encoded.judge_identity().template_digest);
    let (different, _) = projection_fixture(64);
    let changed = Int4SemanticExtractionJobPlanner::for_model(different.identity(), config()).unwrap();
    assert_eq!(freeze(&p).binding.compare(&freeze(&changed).binding), Err(JobError::Mismatch(MismatchField::Execution)));
}

#[test]
fn recipe_authenticates_raw_limits_claim_mapping_and_all_work_policies() {
    // Deliberately mutate only the private recipe to isolate authentication.
    // No mutated declaration is granted a compiler or execution authority.
    let mut p = planner(); let base = freeze(&p); let original = p.recipe.config.clone();
    for n in 0..43 {
        p.recipe.config = original.clone(); let c = &mut p.recipe.config;
        match n {
            0 => c.extraction.schema.push(' '),
            1 => c.extraction.grounding = ExtractionBatchGrounding::SourceMembership,
            2 => c.extraction.budget.max_kv_bytes -= 1,
            3 => c.compiler_limits.max_schema_bytes -= 1,
            4 => c.compiler_limits.max_string_bytes -= 1,
            5 => c.compiler_limits.max_array_items -= 1,
            6 => c.compiler_limits.max_output_bytes -= 1,
            7 => c.compiler_limits.max_states -= 1,
            8 => c.compiler_limits.max_transitions -= 1,
            9 => c.compiler_limits.max_mask_bytes -= 1,
            10 => c.source_limits.index.max_source_bytes -= 1,
            11 => c.source_limits.index.max_index_bytes -= 1,
            12 => c.source_limits.index.max_build_steps -= 1,
            13 => c.source_limits.index.max_matches -= 1,
            14 => c.source_limits.verification.max_fields -= 1,
            15 => c.source_limits.verification.max_matches -= 1,
            16 => c.source_limits.verification.max_scan_steps -= 1,
            17 => c.options.specification.claims[0].prefix.push('!'),
            18 => c.options.specification.claims[0].suffix.push('!'),
            19 => c.options.specification.claims[0].id.push('x'),
            20 => c.options.specification.claims[0].path.push(ClaimPathStep::EachItem),
            21 => c.options.specification.judge_policy.minimum_candidate_weight_ppm += 1,
            22 => c.options.specification.judge_policy.evidence_window_bytes -= 1,
            23 => c.options.specification.judge_budget.max_output_bytes -= 1,
            24 => c.options.limits.max_fields -= 1,
            25 => c.options.limits.max_total_prompt_tokens -= 1,
            26 => c.options.limits.max_output_bytes -= 1,
            27 => c.options.max_model_work.projections.dot_products -= 1,
            28 => c.judge_planning.per_head.max_candidates -= 1,
            29 => c.judge_planning.per_head.max_total_tokens -= 1,
            30 => c.judge_planning.per_head.max_nodes -= 1,
            31 => c.judge_planning.per_head.max_depth -= 1,
            32 => c.judge_planning.per_head.max_candidate_id_bytes -= 1,
            33 => c.judge_planning.per_head.max_projected_logits -= 1,
            34 => c.judge_planning.max_total_prompt_tokens -= 1,
            35 => c.judge_planning.max_total_candidate_tokens -= 1,
            36 => c.judge_planning.max_total_projected_logits -= 1,
            37 => c.judge_planning.max_output_bytes -= 1,
            38 => c.work.masks.per_mask.max_trie_node_visits -= 1,
            39 => c.work.masks.per_mask.checkpoint_interval_nodes -= 1,
            40 => c.work.masks.max_visits_per_item -= 1,
            41 => c.work.masks.max_visits_per_run -= 1,
            _ => c.work.max_model_work.projections.multiply_accumulates -= 1,
        }
        assert_eq!(base.binding.compare(&freeze(&p).binding), Err(JobError::Mismatch(MismatchField::Recipe)), "setting {n}");
    }
}

#[test]
fn invalid_shared_contracts_and_six_axis_or_combined_output_mismatches_refuse() {
    let p = planner(); let exact = limits(p.config()); p.check_job_limits(exact).unwrap();
    for n in 0..6 { for increase in [false, true] {
        let mut changed = exact; let value = axis(&mut changed.max_work, n);
        if increase { *value += 1; } else { *value -= 1; }
        assert_eq!(p.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()));
    } }
    let mut changed = exact; changed.max_result_bytes = budget().max_output_bytes as usize;
    assert_eq!(p.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()));
    for n in 0..6 {
        let mut c = config();
        match n {
            0 => c.options.specification.experimental_opt_in = false,
            1 => c.options.specification.claims[0].path.push(ClaimPathStep::Property("missing".to_owned())),
            2 => c.extraction.schema = r#"{"type":"number","type":"string"}"#.to_owned(),
            3 => c.judge_planning.max_total_prompt_tokens = 0,
            4 => c.work.max_model_work.forward_positions = c.options.max_model_work.forward_positions - 1,
            _ => c.extraction.schema = "x".repeat(1024 * 1024),
        }
        assert!(Int4SemanticExtractionJobPlanner::for_model(model(), c).is_err(), "contract {n}");
    }
}

#[test]
fn genuine_processor_quotes_the_full_attempt_and_rejects_foreign_prepared_seals() {
    struct NeverAdmit;
    impl Int4SemanticExtractionBatchAdmission for NeverAdmit {
        type Guard = ();
        fn admit(&mut self, _: Int4SemanticExtractionAdmission<'_>) -> Result<(ExecutionIdentity, ()), BatchItemFailure> {
            panic!("planning-only fixture reached native admission")
        }
    }
    let (weights, _) = projection_fixture(32);
    let controls = pinned_controls::pinned().unwrap();
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls()).unwrap();
    let memory = Int4MemoryRequirement::for_context(1024, 32).unwrap();
    let mut engine = StrictInt4Engine::new(&weights, 1024, Int4MemoryBudget { max_kv_bytes: memory.kv_bytes,
        max_rope_bytes: memory.rope_bytes, max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
    for encoding in [UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1(compact())] {
        let make = || { let mut c = config(); c.encoding = encoding.clone();
            Int4SemanticExtractionJobPlanner::for_model(weights.identity(), c).unwrap() };
        let mut processor = Int4SemanticExtractionJobProcessor::new(make(), &mut engine, &vocabulary, NeverAdmit).unwrap();
        let prepared = processor.prepare_with_control(BatchDocument { id: "alpha".to_owned(),
            text: "The document reports amount 7. Café 上海 <tool_call>.".to_owned(), task_args: None }, &mut Continue).unwrap();
        assert_eq!(processor.durable_work(&prepared).unwrap(), quote(&config()));
        assert_eq!(processor.max_result_bytes(&prepared), config().options.limits.max_output_bytes);
        assert_eq!(processor.native.reserved_model_work(), Int4Work::default());
        drop(processor);
        let other = Int4SemanticExtractionJobProcessor::new(make(), &mut engine, &vocabulary, NeverAdmit).unwrap();
        assert_eq!(other.durable_work(&prepared).unwrap_err().fault.code, BatchCode::Admission);
        assert_eq!(other.native.reserved_model_work(), Int4Work::default());
        drop(other);
        assert!(!engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0));
    }
}

// Storage-only driver: the real immutable recipe and JobRunner are exercised,
// but neither native extraction nor its second reader is simulated as output.
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-semantic-job-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
#[derive(Default)]
struct Events { prepared: RefCell<Vec<String>>, executed: RefCell<Vec<u64>>, held: Cell<bool>,
    held_polls: Cell<usize>, result_dropped: Cell<bool>, released: Cell<usize> }
struct Guard(Rc<Events>);
impl Drop for Guard {
    fn drop(&mut self) { assert!(self.0.result_dropped.get()); assert!(self.0.held.replace(false));
        self.0.released.set(self.0.released.get() + 1); }
}
#[derive(Serialize)]
struct Data { fixture: &'static str, ordinal: u64, input_id: String }
struct Output { data: Data, events: Rc<Events> }
impl Serialize for Output {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get()); self.data.serialize(s)
    }
}
impl Drop for Output { fn drop(&mut self) { assert!(self.events.held.get()); self.events.result_dropped.set(true); } }
struct Processor { planner: Int4SemanticExtractionJobPlanner, events: Rc<Events>, fail: bool }
impl Processor {
    fn new(planner: Int4SemanticExtractionJobPlanner, events: Rc<Events>) -> Self { Self { planner, events, fail: false } }
}
impl BatchProcessor for Processor {
    type Args = ();
    type Prepared = String;
    type Output = GuardedOutput<Output, Guard>;
    fn prepare(&mut self, input: BatchDocument<()>) -> Result<String, BatchItemFailure> {
        let input = shared_document(input)?;
        self.events.prepared.borrow_mut().push(input.id.clone()); Ok(input.id)
    }
    fn planned_work(&self, _: &String) -> BatchWork {
        let work = self.planner.config().options.max_model_work;
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: String, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("durable driver must supply stable delivery coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, input_id: String, context: BatchRequestContext,
        _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        self.events.executed.borrow_mut().push(context.request_seq);
        if self.fail { return Err(BatchItemFailure::fatal(BatchCode::Execution)); }
        assert!(!self.events.held.replace(true)); self.events.result_dropped.set(false);
        Ok(GuardedOutput::new(Output { data: Data { fixture: "storage-only", ordinal: context.request_seq,
            input_id }, events: Rc::clone(&self.events) }, Guard(Rc::clone(&self.events))))
    }
}
impl DurableBatchProcessor for Processor {
    type Recipe = Int4SemanticExtractionJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { &self.planner.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, _: &String) -> Result<JobWork, BatchItemFailure> { Ok(quote(self.planner.config())) }
    fn max_result_bytes(&self, _: &String) -> u64 { self.planner.max_result_bytes() }
}
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(error) => error, Ok(_) => panic!("expected durable refusal") }
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
fn original_and_retained_resume_skip_committed_items_and_match_ordered_materialization() {
    let inputs = inputs(); let cap = limits(&config());
    let baseline = Root::new(); let baseline_events = Rc::new(Events::default());
    let mut run = JobRunner::create(&baseline.0, key(), JobId([9; 16]), cap, &inputs,
        Processor::new(planner(), baseline_events), &mut Continue).unwrap();
    run.run(&mut Continue).unwrap(); run.materialize_ordered(&mut Continue).unwrap(); drop(run);
    let expected = fs::read(baseline.0.join("materialized.ndjson")).unwrap();
    for retained in [false, true] {
        let root = Root::new(); let events = Rc::new(Events::default());
        let processor = Processor::new(planner(), Rc::clone(&events));
        let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
        let mut run = if retained { JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), cap,
            retention, &inputs, processor, &mut Continue) } else { JobRunner::create(&root.0, key(), JobId([9; 16]),
            cap, &inputs, processor, &mut Continue) }.unwrap();
        assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
        let processor = Processor::new(planner(), Rc::clone(&events));
        let mut resumed = if retained { JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), cap,
            retention, processor, TailPolicy::Refuse, &mut Continue) } else { JobRunner::resume(&root.0, key(),
            JobId([9; 16]), cap, &inputs, processor, TailPolicy::Refuse, &mut Continue) }.unwrap();
        let progress = resumed.run(&mut Continue).unwrap();
        assert_eq!((progress.committed, progress.attempts), (2, 2));
        assert_eq!(progress.reserved_work, quote(&config()).checked_add(quote(&config())).unwrap());
        resumed.materialize_ordered(&mut Continue).unwrap(); drop(resumed);
        assert_eq!(*events.prepared.borrow(), vec!["alpha", "beta"]);
        assert_eq!(*events.executed.borrow(), vec![1, 2]); assert_eq!(events.released.get(), 2);
        assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), expected);
    }
}

#[test]
fn failed_combined_attempt_commits_nothing_and_all_six_work_axes_survive_reopen() {
    for n in 0..6 {
        let mut c = config(); let spent = quote(&c); let mut lifetime = limits(&c).max_work;
        *axis(&mut lifetime, n) = *axis(&mut quote(&c), n);
        c.work.max_model_work = lifetime.int4_model_work(); c.work.masks.max_visits_per_run = lifetime.mask_node_visits;
        let make = || Int4SemanticExtractionJobPlanner::for_model(model(), c.clone()).unwrap();
        let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
        let mut processor = Processor::new(make(), Rc::clone(&events)); processor.fail = true;
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(&c), &inputs, processor, &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!((run.progress().committed, run.progress().attempts), (0, 1));
        assert_eq!(run.progress().reserved_work, spent); assert!(run.is_stopped()); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(&c), &inputs,
            Processor::new(make(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(resumed.progress().reserved_work, spent); assert_eq!(resumed.progress().committed, 0);
        assert_eq!(*events.executed.borrow(), vec![1]); assert_eq!(events.released.get(), 0);
    }
}

#[test]
fn recipe_mismatch_refuses_before_tail_repair_and_guard_covers_spool_acknowledgement() {
    let root = Root::new(); let inputs = inputs(); let cap = limits(&config()); let events = Rc::new(Events::default());
    let mut control = CancelAfterSync(Rc::clone(&events));
    let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), cap, &inputs,
        Processor::new(planner(), Rc::clone(&events)), &mut control).unwrap();
    assert_eq!(error(run.step(&mut control)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().reserved_work, quote(&config()));
    assert_eq!(events.released.get(), 1); assert!(!events.held.get()); drop(run);
    let before = events.prepared.borrow().len(); let mut changed = config();
    changed.options.specification.claims[0].prefix.push('!');
    let changed = Int4SemanticExtractionJobPlanner::for_model(model(), changed).unwrap();
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
        Processor::new(changed, Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(events.prepared.borrow().len(), before);
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
        Processor::new(planner(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue)),
        JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
        Processor::new(planner(), Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    let progress = resumed.run(&mut Continue).unwrap();
    assert_eq!((progress.committed, progress.attempts), (2, 3));
    assert_eq!(*events.executed.borrow(), vec![1, 1, 2]); assert_eq!(events.released.get(), 3);
}

#[test]
fn non_null_record_overrides_refuse_before_storage_or_preparation() {
    for args in ["{}", "[]", "false", "\"replace-policy\""] {
        let root = Root::new(); let events = Rc::new(Events::default());
        let input = format!("{{\"id\":\"alpha\",\"text\":\"private\",\"task_args\":{args}}}");
        let inputs = [JobInput { id: "alpha", original: input.as_bytes(), normalized: input.as_bytes() }];
        assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), limits(&config()), &inputs,
            Processor::new(planner(), Rc::clone(&events)), &mut Continue)), JobRunError::InvalidEnvelope);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0); assert!(events.prepared.borrow().is_empty());
    }
}
