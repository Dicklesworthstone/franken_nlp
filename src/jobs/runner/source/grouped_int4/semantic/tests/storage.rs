//! The real immutable recipe and JobRunner with synthetic execution only.
//! Native source/review preparation is covered in the parent module.
use super::*;

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-semantic-source-job-{}-{}",
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
struct Processor { planner: Int4SemanticSourceJobPlanner, events: Rc<Events>, fail: bool }
impl Processor {
    fn new(planner: Int4SemanticSourceJobPlanner, events: Rc<Events>) -> Self { Self { planner, events, fail: false } }
}
impl BatchProcessor for Processor {
    type Args = Int4SemanticSourceBatchArgs;
    type Prepared = String;
    type Output = GuardedOutput<Output, Guard>;
    fn prepare(&mut self, input: BatchDocument<Self::Args>) -> Result<String, BatchItemFailure> {
        let answer = matches!(self.planner.config().task, Int4SemanticSourceBatchTask::Answer { .. });
        if answer != input.task_args.is_some() { return Err(BatchItemFailure::reject(BatchCode::InvalidEnvelope)); }
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
    type Recipe = Int4SemanticSourceJobRecipe;
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
fn original_and_retained_resume_skip_committed_summary_and_qa_items_in_order() {
    for answer in [false, true] {
        let inputs = inputs(answer); let cap = limits(&config(answer));
        let baseline = Root::new(); let baseline_events = Rc::new(Events::default());
        let mut run = JobRunner::create(&baseline.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(planner(answer), baseline_events), &mut Continue).unwrap();
        run.run(&mut Continue).unwrap(); run.materialize_ordered(&mut Continue).unwrap(); drop(run);
        let expected = fs::read(baseline.0.join("materialized.ndjson")).unwrap();
        for retained in [false, true] {
            let root = Root::new(); let events = Rc::new(Events::default());
            let processor = Processor::new(planner(answer), Rc::clone(&events));
            let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
            let mut run = if retained { JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), cap,
                retention, &inputs, processor, &mut Continue) } else { JobRunner::create(&root.0, key(), JobId([9; 16]),
                cap, &inputs, processor, &mut Continue) }.unwrap();
            assert_eq!(run.step(&mut Continue).unwrap().unwrap().committed, 1); drop(run);
            let processor = Processor::new(planner(answer), Rc::clone(&events));
            let mut resumed = if retained { JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]), cap,
                retention, processor, TailPolicy::Refuse, &mut Continue) } else { JobRunner::resume(&root.0, key(),
                JobId([9; 16]), cap, &inputs, processor, TailPolicy::Refuse, &mut Continue) }.unwrap();
            let progress = resumed.run(&mut Continue).unwrap();
            assert_eq!((progress.committed, progress.attempts), (2, 2));
            assert_eq!(progress.reserved_work, quote(&config(answer)).checked_add(quote(&config(answer))).unwrap());
            resumed.materialize_ordered(&mut Continue).unwrap(); drop(resumed);
            assert_eq!(*events.prepared.borrow(), vec!["alpha", "beta"]);
            assert_eq!(*events.executed.borrow(), vec![1, 2]); assert_eq!(events.released.get(), 2);
            assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), expected);
        }
    }
}

#[test]
fn failed_generation_or_review_never_commits_partial_output_or_renews_any_work_axis() {
    for answer in [false, true] { for n in 0..6 {
        let mut c = config(answer); let spent = quote(&c); let mut lifetime = limits(&c).max_work;
        *axis(&mut lifetime, n) = *axis(&mut quote(&c), n);
        c.work.max_model_work = lifetime.int4_model_work(); c.work.masks.max_visits_per_run = lifetime.mask_node_visits;
        let make = || Int4SemanticSourceJobPlanner::for_model(model(), c.clone()).unwrap();
        let root = Root::new(); let inputs = inputs(answer); let events = Rc::new(Events::default());
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
    } }
}

#[test]
fn changed_review_policy_refuses_before_tail_repair_and_guards_cover_durable_ack() {
    for answer in [false, true] {
        let root = Root::new(); let inputs = inputs(answer); let cap = limits(&config(answer));
        let events = Rc::new(Events::default()); let mut control = CancelAfterSync(Rc::clone(&events));
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(planner(answer), Rc::clone(&events)), &mut control).unwrap();
        assert_eq!(error(run.step(&mut control)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
        assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().reserved_work, quote(&config(answer)));
        assert_eq!(events.released.get(), 1); assert!(!events.held.get()); drop(run);
        let before = events.prepared.borrow().len(); let mut changed = config(answer);
        changed.options.judge_policy.minimum_margin_milli += 1;
        let changed = Int4SemanticSourceJobPlanner::for_model(model(), changed).unwrap();
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(changed, Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
        assert_eq!(events.prepared.borrow().len(), before);
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(planner(answer), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue)),
            JobRunError::Storage(JobError::UncommittedTail));
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(planner(answer), Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
        let progress = resumed.run(&mut Continue).unwrap();
        assert_eq!((progress.committed, progress.attempts), (2, 3));
        assert_eq!(*events.executed.borrow(), vec![1, 1, 2]); assert_eq!(events.released.get(), 3);
    }
}

#[test]
fn exact_question_passage_identity_order_and_original_bytes_authenticate_before_resume() {
    let root = Root::new(); let original = inputs(true); let cap = limits(&config(true));
    let events = Rc::new(Events::default());
    let run = JobRunner::create(&root.0, key(), JobId([9; 16]), cap, &original,
        Processor::new(planner(true), Rc::clone(&events)), &mut Continue).unwrap(); drop(run);
    for n in 0..4 {
        let mut value: serde_json::Value = serde_json::from_slice(original[0].original).unwrap();
        match n {
            0 => value["text"] = serde_json::json!("Who met Alice?"),
            1 => value["task_args"]["passages"][0]["id"] = serde_json::json!("replacement"),
            2 => value["task_args"]["passages"][0]["text"] = serde_json::json!("Bob met Alice."),
            _ => value["task_args"]["passages"].as_array_mut().unwrap().reverse(),
        }
        let changed = serde_json::to_vec(&value).unwrap();
        let inputs = [JobInput { id: "alpha", original: &changed, normalized: &changed },
            JobInput { id: "beta", original: original[1].original, normalized: original[1].normalized }];
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), cap, &inputs,
            Processor::new(planner(true), Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Population)));
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
    }
}

#[test]
fn unknown_record_policy_refuses_before_storage_but_fixed_task_shape_is_item_preparation() {
    for args in [r#"{"passages":[],"options":{}}"#, r#"{"passages":[],"budget":{}}"#,
        r#"{"passages":[],"task":"summarize"}"#, r#"{"passages":[{"id":"p","text":"x","extra":"y"}]}"#] {
        let root = Root::new(); let events = Rc::new(Events::default());
        let input = format!("{{\"id\":\"alpha\",\"text\":\"private\",\"task_args\":{args}}}");
        let inputs = [JobInput { id: "alpha", original: input.as_bytes(), normalized: input.as_bytes() }];
        assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), limits(&config(true)), &inputs,
            Processor::new(planner(true), Rc::clone(&events)), &mut Continue)), JobRunError::InvalidEnvelope);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0); assert!(events.prepared.borrow().is_empty());
    }
    for answer in [false, true] {
        let root = Root::new(); let inputs = inputs(!answer); let events = Rc::new(Events::default());
        let mut run = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(&config(answer)), &inputs,
            Processor::new(planner(answer), Rc::clone(&events)), &mut Continue).unwrap();
        assert_eq!(error(run.step(&mut Continue)), JobRunError::Processor(BatchCode::InvalidEnvelope.into()));
        assert_eq!((run.progress().committed, run.progress().attempts), (0, 0));
        assert_eq!(run.progress().reserved_work, JobWork::default()); assert!(events.executed.borrow().is_empty());
    }
}
