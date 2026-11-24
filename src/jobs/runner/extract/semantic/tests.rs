//! Real pinned factory and authenticated storage contracts. Storage fixtures
//! emit explicitly labeled fixture records; no native forward or semantic
//! review, model quality, crash recovery, or execution parity is claimed.
use super::*;
use crate::{
    batch::extract::ExtractionBatchGrounding,
    canonjson, execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobError, JobId, JobInput, JobSecret, MismatchField},
    native_engine::{decode::DecodeCancellationKind, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::TaskBudget, extract::semantic::{ClaimPathStep, ClaimRule, ClaimValueKind,
        SemanticLimits, SemanticVerificationSpec}, judge::FaithfulnessPolicy},
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
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"int8-semantic-durable-planning");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "fixture".into(), quant_recipe: "fixture-int8".into(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".into(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn config() -> Int8SemanticExtractionJobConfig {
    Int8SemanticExtractionJobConfig {
        extraction: ExtractionBatchArgs { schema: " {\"type\":\"number\"} \n".into(),
            grounding: ExtractionBatchGrounding::Structural, budget: budget() },
        compiler_limits: CompileLimits::default(), source_limits: SourceRuntimeLimits::default(),
        options: Int8SemanticOptions { specification: SemanticVerificationSpec { schema_version: 1,
            experimental_opt_in: true, revision: "durable-review-v1".into(),
            claims: vec![ClaimRule { id: "amount".into(), path: vec![], value_kind: ClaimValueKind::Number,
                prefix: "The original source reports the amount ".into(), suffix: ".".into() }],
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 4096, max_evidence_windows: 8, max_evidence_spans: 8 },
            judge_budget: budget() }, limits: SemanticLimits { max_output_bytes: 1 << 20,
                ..SemanticLimits::default() }, max_model_work: work(1 << 56) },
        judge_planning: JudgeLimits::default(), encoding: UntrustedEncoding::ByteFallback,
        encoding_work_limits: None,
        work: Int8SemanticExtractionBatchLimits { max_model_work: work(1 << 58),
            masks: ExtractionMaskBudget { per_mask: MaskWorkLimits::default(),
                max_visits_per_item: 1_000_000_000, max_visits_per_run: 4_000_000_000 } },
    }
}
fn compact_config() -> Int8SemanticExtractionJobConfig {
    let mut c = config();
    c.encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65536, max_output_tokens: 65536, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 });
    c.encoding_work_limits = Some(SafeBpeWorkLimits { max_input_bytes: 1 << 24, max_output_tokens: 1 << 24,
        max_pair_checks: 1 << 32, max_pair_bytes: 1 << 36 }); c
}
fn factory(config: Int8SemanticExtractionJobConfig) -> Result<Int8SemanticExtractionJobPlanner, BatchFault> {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    Int8SemanticExtractionJobPlanner::pinned(controls, eos, identity(), config)
}
fn planner() -> Int8SemanticExtractionJobPlanner { factory(config()).unwrap() }
fn limits(c: &Int8SemanticExtractionJobConfig) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: c.options.limits.max_output_bytes as usize,
        max_spool_bytes: 8 << 20, max_materialized_bytes: 8 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork { model: c.work.max_model_work,
            mask_node_visits: c.work.masks.max_visits_per_run } }
}
#[cfg(unix)]
fn quote(c: &Int8SemanticExtractionJobConfig) -> JobWork {
    JobWork { model: c.options.max_model_work, mask_node_visits: c.work.masks.max_visits_per_item }
}
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Alice reports amount 7."}"#;
    let b = br#"{"id":"beta","text":"Bob reports amount 9.","task_args":null}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn freeze(p: &Int8SemanticExtractionJobPlanner) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([9; 16]), execution: p.execution_identity(),
        recipe: p.job_recipe(), limits: limits(&config()) }, inputs(), &mut Continue).unwrap()
}
fn axis(work: &mut JobWork, n: usize) -> &mut u64 {
    match n { 0 => &mut work.model.forward_positions, 1 => &mut work.model.projected_logits,
        2 => &mut work.model.attention_pairs, 3 => &mut work.model.projections.dot_products,
        4 => &mut work.model.projections.multiply_accumulates, _ => &mut work.mask_node_visits }
}
#[test]
fn exact_raw_schema_options_both_compiler_identities_and_invocation_policy_are_authenticated() {
    let p = planner(); let recipe = serde_json::to_value(p.job_recipe()).unwrap();
    assert_eq!(recipe["settings"]["extraction"]["schema"], config().extraction.schema);
    assert_eq!(recipe["settings"]["options"], serde_json::to_value(config().options).unwrap());
    assert_eq!(recipe["extraction_identity"], serde_json::to_value(p.execution_identity()).unwrap());
    assert_eq!(recipe["judge_identity"], serde_json::to_value(p.judge_identity()).unwrap());
    assert_eq!(recipe["encoding_scope"], ENCODING_SCOPE); assert_eq!(recipe["input_policy"], INPUT_POLICY);
    assert_eq!(recipe["reservation_policy"], RESERVATION_POLICY);
    assert_eq!(p.encoding_workspace_bytes().unwrap(), 0);
    let p = factory(compact_config()).unwrap();
    let UntrustedEncoding::SafeBpeV1(call) = p.document_encoding() else { unreachable!() };
    assert_eq!(p.encoding_workspace_bytes().unwrap(), call.requirements().unwrap().workspace_bytes + 1024);
    let recipe = serde_json::to_value(p.job_recipe()).unwrap();
    assert_eq!(recipe["settings"]["input_encoding"], serde_json::to_value(p.document_encoding()).unwrap());
    assert_eq!(recipe["settings"]["invocation_encoding_work_limits"],
        serde_json::to_value(p.encoding_work_limits()).unwrap());
    assert_eq!(p.execution_identity(), p.compiler.base_execution_identity());
    assert_eq!(p.judge_identity(), p.compiler.judge_identity());
}
#[test]
fn every_per_document_or_invocation_encoding_axis_changes_the_frozen_contract() {
    let p = factory(compact_config()).unwrap(); let frozen = freeze(&p);
    for n in 0..8 {
        let mut c = compact_config();
        if n < 4 {
            let UntrustedEncoding::SafeBpeV1(call) = &mut c.encoding else { unreachable!() };
            match n { 0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
                2 => call.max_pair_checks -= 1, _ => call.max_pair_bytes -= 1 }
        } else {
            let total = c.encoding_work_limits.as_mut().unwrap();
            match n { 4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
                6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1 }
        }
        let changed = factory(c).unwrap();
        assert!(frozen.binding.compare(&freeze(&changed).binding).is_err(), "encoding axis {n}");
    }
    let resumed = factory(compact_config()).unwrap();
    frozen.binding.compare(&freeze(&resumed).binding).unwrap();
    assert_eq!(canonjson::canonical_bytes(p.job_recipe()).unwrap(),
        canonjson::canonical_bytes(resumed.job_recipe()).unwrap());
}
#[test]
fn recipe_projection_authenticates_compiler_source_claim_judge_and_work_axes() {
    let mut p = planner(); let frozen = freeze(&p);
    for n in 0..43 {
        // Mutate only the private recipe projection to test its commitment.
        // No such mutation is exposed by the actual immutable factory.
        let recipe = Arc::get_mut(&mut p.recipe).unwrap();
        recipe.config = config(); let c = &mut recipe.config;
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
        assert_eq!(frozen.binding.compare(&freeze(&p).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)), "setting {n}");
    }
}
#[test]
fn every_lifetime_work_axis_must_equal_the_journal_and_combined_output_must_fit() {
    let p = planner(); let exact = limits(p.config()); p.check_job_limits(exact).unwrap();
    for n in 0..6 { for increase in [false, true] {
        let mut changed = exact; let value = axis(&mut changed.max_work, n);
        if increase { *value += 1; } else { *value -= 1; }
        assert_eq!(p.check_job_limits(changed), Err(BatchCode::InvalidLimits.into()));
    } }
    let mut output = exact; output.max_result_bytes = budget().max_output_bytes as usize;
    assert_eq!(p.check_job_limits(output), Err(BatchCode::InvalidLimits.into()));
    let mut missing = compact_config(); missing.encoding_work_limits = None;
    assert!(factory(missing).is_err());
    let mut byte = config(); byte.encoding_work_limits = compact_config().encoding_work_limits;
    assert!(factory(byte).is_err());
    let mut invalid = config(); invalid.options.specification.experimental_opt_in = false;
    assert!(factory(invalid).is_err());
}
#[test]
fn actual_recipe_and_compiler_owners_survive_consuming_the_factory() {
    fn shareable<T: Send + Sync + 'static>() {}
    shareable::<Int8SemanticExtractionJobPlanner>();
    shareable::<Int8SemanticExtractionJobRecipe>();
    let p = planner(); let recipe = p.retain_recipe(); let compiler = p.retain_compiler();
    let recipe_weak = Arc::downgrade(&recipe); let compiler_weak = Arc::downgrade(&compiler);
    let expected = canonjson::canonical_bytes(p.job_recipe()).unwrap();
    drop(p);
    assert_eq!(canonjson::canonical_bytes(&*recipe).unwrap(), expected);
    assert_eq!(compiler.judge_identity(), &recipe.judge_identity);
    assert!(recipe_weak.upgrade().is_some() && compiler_weak.upgrade().is_some());
    drop(recipe); drop(compiler);
    assert!(recipe_weak.upgrade().is_none() && compiler_weak.upgrade().is_none());
}

#[cfg(unix)]
mod storage {
    use super::*;
    use crate::jobs::{InputRetentionLimits, TailPolicy, runner::{JobRunError, JobRunner}};
    use std::{cell::{Cell, RefCell}, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
        sync::atomic::{AtomicU64, Ordering}};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Root(PathBuf);
    impl Root {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("fnlp-int8-semantic-job-{}-{}",
                std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
        }
    }
    impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    #[derive(Default)]
    struct Events { prepared: RefCell<Vec<String>>, executed: RefCell<Vec<u64>>, held: Cell<bool>,
        held_polls: Cell<usize>, result_dropped: Cell<bool>, released: Cell<usize> }
    struct Guard(Rc<Events>);
    impl Drop for Guard {
        fn drop(&mut self) {
            assert!(self.0.result_dropped.get()); assert!(self.0.held.replace(false));
            self.0.released.set(self.0.released.get() + 1);
        }
    }
    #[derive(Serialize)]
    struct Data { fixture: &'static str, ordinal: u64, input_id: String }
    struct Output { data: Data, events: Rc<Events> }
    impl Serialize for Output {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            assert!(self.events.held.get()); self.data.serialize(serializer)
        }
    }
    impl Drop for Output {
        fn drop(&mut self) { assert!(self.events.held.get()); self.events.result_dropped.set(true); }
    }
    struct Processor { planner: Int8SemanticExtractionJobPlanner, events: Rc<Events>, fail: bool }
    impl Processor {
        fn new(planner: Int8SemanticExtractionJobPlanner, events: Rc<Events>) -> Self {
            Self { planner, events, fail: false }
        }
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
            Err(BatchItemFailure::fatal(BatchCode::InvalidExecution))
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
        type Recipe = Int8SemanticExtractionJobRecipe;
        fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
        fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
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
    fn original_and_retained_resume_skip_committed_items_and_materialize_the_same_fixture_bytes() {
        let inputs = inputs(); let cap = limits(&config());
        let baseline = Root::new();
        let mut run = JobRunner::create(&baseline.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(planner(), Rc::new(Events::default())), &mut Continue).unwrap();
        run.run(&mut Continue).unwrap(); run.materialize_ordered(&mut Continue).unwrap(); drop(run);
        let expected = fs::read(baseline.0.join("materialized.ndjson")).unwrap();
        for retained in [false, true] {
            let root = Root::new(); let events = Rc::new(Events::default());
            let processor = Processor::new(planner(), Rc::clone(&events));
            let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
            let mut run = if retained {
                JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), cap,
                    retention, &inputs, processor, &mut Continue)
            } else { JobRunner::create(&root.0, key(), JobId([9; 16]), cap, &inputs, processor, &mut Continue) }.unwrap();
            assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
            let processor = Processor::new(planner(), Rc::clone(&events));
            let mut resumed = if retained {
                JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), cap, retention,
                    processor, TailPolicy::Refuse, &mut Continue)
            } else { JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
                processor, TailPolicy::Refuse, &mut Continue) }.unwrap();
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
    fn failed_combined_attempt_commits_nothing_and_all_six_debits_survive_reopen() {
        for n in 0..6 {
            let mut c = config(); let spent = quote(&c); let mut lifetime = limits(&c).max_work;
            *axis(&mut lifetime, n) = *axis(&mut quote(&c), n);
            c.work.max_model_work = lifetime.model; c.work.masks.max_visits_per_run = lifetime.mask_node_visits;
            let make = || factory(c.clone()).unwrap();
            let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
            let mut processor = Processor::new(make(), Rc::clone(&events)); processor.fail = true;
            let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(&c), &inputs,
                processor, &mut Continue).unwrap();
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
    fn recipe_drift_refuses_before_tail_repair_and_output_guard_survives_acknowledgement() {
        let root = Root::new(); let inputs = inputs(); let cap = limits(&config());
        let events = Rc::new(Events::default()); let mut control = CancelAfterSync(Rc::clone(&events));
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(planner(), Rc::clone(&events)), &mut control).unwrap();
        assert_eq!(error(run.step(&mut control)),
            JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
        assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().reserved_work, quote(&config()));
        assert_eq!(events.released.get(), 1); assert!(!events.held.get()); drop(run);
        let before = events.prepared.borrow().len(); let mut changed = config();
        changed.options.specification.claims[0].prefix.push('!');
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(factory(changed).unwrap(), Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
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
    fn non_null_record_overrides_refuse_before_any_storage_or_preparation() {
        for args in ["{}", "[]", "false", "\"replace-policy\""] {
            let root = Root::new(); let events = Rc::new(Events::default());
            let input = format!("{{\"id\":\"alpha\",\"text\":\"private\",\"task_args\":{args}}}");
            let inputs = [JobInput { id: "alpha", original: input.as_bytes(), normalized: input.as_bytes() }];
            assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), limits(&config()), &inputs,
                Processor::new(planner(), Rc::clone(&events)), &mut Continue)), JobRunError::InvalidEnvelope);
            assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0); assert!(events.prepared.borrow().is_empty());
        }
    }
}
