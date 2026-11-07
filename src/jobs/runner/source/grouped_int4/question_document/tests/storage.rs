//! Actual protected storage and journal replay over real pinned preparations.
//! Output is explicitly synthetic transport data; no native answer is invented.
use super::*;
use crate::jobs::{TailPolicy, runner::{JobRunError, JobRunner}};
use serde::Serializer;
use std::{cell::Cell, fs, io::Write, os::unix::fs::DirBuilderExt, path::PathBuf,
    rc::Rc, sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-question-document-job-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
#[derive(Default)]
struct Events { held: Cell<bool>, executions: Cell<usize>, serializations: Cell<usize>,
    values_dropped: Cell<usize>, releases: Cell<usize> }
struct Hold(Rc<Events>);
impl Drop for Hold {
    fn drop(&mut self) {
        assert_eq!(self.0.values_dropped.get(), self.0.releases.get() + 1);
        assert!(self.0.held.replace(false)); self.0.releases.set(self.0.releases.get() + 1);
    }
}
struct Datum { source: String, question: String, events: Rc<Events>, fail_serialization: bool }
impl Serialize for Datum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get());
        self.events.serializations.set(self.events.serializations.get() + 1);
        if self.fail_serialization {
            return Err(<S::Error as serde::ser::Error>::custom("synthetic transport encoding refusal"));
        }
        ("synthetic-storage-pair-not-native-answer", &self.source, &self.question).serialize(serializer)
    }
}
impl Drop for Datum {
    fn drop(&mut self) {
        assert!(self.events.held.get()); self.events.values_dropped.set(self.events.values_dropped.get() + 1);
    }
}
struct Prepared { native: PreparedInt4QuestionDocument, source: String, question: String }
struct Processor {
    planner: Int4QuestionDocumentJobPlanner, events: Rc<Events>, fail: bool, fail_serialization: bool,
}
impl Processor {
    fn new(config: Int4QuestionDocumentBatchConfig, limits: JobLimits, events: Rc<Events>) -> Self {
        Self { planner: Int4QuestionDocumentJobPlanner::new(compiler(config), limits).unwrap(),
            events, fail: false, fail_serialization: false }
    }
}
impl BatchProcessor for Processor {
    type Args = QuestionDocumentArgs;
    type Prepared = Prepared;
    type Output = GuardedOutput<Datum, Hold>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        panic!("missing preparation control")
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        let source = document.text.clone();
        let question = document.task_args.as_ref().ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?.question.clone();
        let native = self.planner.prepare_with_control(document, control)?;
        Ok(Prepared { native, source, question })
    }
    fn planned_work(&self, prepared: &Self::Prepared) -> BatchWork {
        let work = prepared.native.planned_work();
        BatchWork { forward_positions: work.forward_positions, projected_logits: work.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("missing durable coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, prepared: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        if let Some(cause) = control.checkpoint(0) { return Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))); }
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        self.planner.durable_work(&prepared.native)?;
        self.events.executions.set(self.events.executions.get() + 1);
        if self.fail { return Err(BatchItemFailure::fatal(BatchCode::Execution)); }
        assert!(!self.events.held.replace(true));
        Ok(GuardedOutput::new(Datum { source: prepared.source, question: prepared.question,
            events: Rc::clone(&self.events), fail_serialization: self.fail_serialization }, Hold(Rc::clone(&self.events))))
    }
}
impl DurableBatchProcessor for Processor {
    type Recipe = Int4QuestionDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { self.planner.execution_identity() }
    fn job_recipe(&self) -> &Self::Recipe { self.planner.job_recipe() }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> { self.planner.check_job_limits(limits) }
    fn durable_work(&self, prepared: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        self.planner.durable_work(&prepared.native)
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.planner.max_result_bytes() }
}
fn inputs() -> [JobInput<'static>; 2] {
    [
        ("first", r#"{"id":"first","text":"Alice and Bob met in 上海. The final é tail names Carol.","task_args":{"question":"Who is named? PRIVATE_QUESTION"}}"#),
        ("second", r#"{"id":"second","text":"Another exact source names Dave in 東京.","task_args":{"question":"Where is Dave named? OTHER_QUESTION"}}"#),
    ].map(|(id, bytes)| JobInput { id, original: bytes.as_bytes(), normalized: bytes.as_bytes() })
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(error) => error, Ok(_) => panic!("expected refusal") }
}

#[test]
fn completed_resume_preserves_each_private_pair_and_matches_uninterrupted_materialization() {
    let root = Root::new(); let full = Root::new(); let inputs = inputs();
    let config = config(2); let limits = limits(config); let events = Rc::new(Events::default());
    let make = || Processor::new(config, limits, Rc::clone(&events));
    let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs, make(), &mut Continue).unwrap();
    let first = job.step(&mut Continue).unwrap().unwrap(); assert_eq!(first.committed, 1);
    let output: serde_json::Value = serde_json::from_slice(&job.read_committed(0, &mut Continue).unwrap()).unwrap();
    assert_eq!(output[1], SOURCE); assert_eq!(output[2], QUESTION); drop(job);
    assert_eq!((events.executions.get(), events.releases.get()), (1, 1)); assert!(!events.held.get());
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let complete = resumed.run(&mut Continue).unwrap(); assert_eq!((complete.committed, complete.attempts), (2, 2));
    assert_eq!((events.executions.get(), events.releases.get()), (2, 2));
    resumed.verify(&mut Continue).unwrap();
    let materialized = resumed.materialize_ordered(&mut Continue).unwrap(); drop(resumed);
    let mut completed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    assert_eq!(completed.run(&mut Continue).unwrap(), materialized); assert_eq!(events.executions.get(), 2); drop(completed);
    let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(config, limits, Rc::new(Events::default())), &mut Continue).unwrap();
    baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), fs::read(full.0.join("materialized.ndjson")).unwrap());
    assert!(events.serializations.get() >= 4); assert_eq!(events.values_dropped.get(), 2);
}

#[test]
fn failed_attempt_debits_all_five_native_axes_and_masks_across_restart() {
    let original = factory(0, false, None);
    let prepared = original.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
    let debit = original.durable_work(&prepared).unwrap();
    for field in 0..6 {
        let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
        let mut config = config(0); config.mapping.max_model_work = debit.int4_model_work();
        config.mapping.max_mask_visits = debit.mask_node_visits;
        if field == 5 { config.max_mask_visits = debit.mask_node_visits; }
        else { *axis(&mut config.max_model_work, field) = *axis(&mut debit.int4_model_work(), field); }
        let limits = limits(config);
        let mut processor = Processor::new(config, limits, Rc::clone(&events)); processor.fail = true;
        let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs, processor, &mut Continue).unwrap();
        assert_eq!(error(job.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!((job.progress().committed, job.progress().attempts), (0, 1));
        assert_eq!(job.progress().reserved_work, debit); drop(job);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
            Processor::new(config, limits, Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(resumed.progress().reserved_work, debit); assert_eq!(resumed.progress().attempts, 1);
        assert_eq!(events.executions.get(), 1); assert_eq!(fs::metadata(root.0.join("results.spool")).unwrap().len(), 0);
        assert!(!root.0.join("materialized.ndjson").exists());
    }
}

#[test]
fn source_question_and_policy_mismatch_refuse_before_uncommitted_tail_repair() {
    let root = Root::new(); let inputs = inputs(); let config = config(2); let limits = limits(config);
    let events = Rc::new(Events::default()); let make = || Processor::new(config, limits, Rc::clone(&events));
    let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs, make(), &mut Continue).unwrap();
    job.step(&mut Continue).unwrap(); drop(job);
    let spool = root.0.join("results.spool");
    fs::OpenOptions::new().append(true).open(&spool).unwrap().write_all(b"uncommitted-incomplete-tail").unwrap();
    let original = fs::read(&spool).unwrap();
    for (from, to) in [("PRIVATE_QUESTION", "CHANGED_QUESTION"), ("final é tail", "changed é tail")] {
        let changed = std::str::from_utf8(inputs[0].original).unwrap().replace(from, to);
        let population = [JobInput { id: inputs[0].id, original: changed.as_bytes(), normalized: changed.as_bytes() },
            JobInput { id: inputs[1].id, original: inputs[1].original, normalized: inputs[1].normalized }];
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &population,
            make(), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Population)));
        assert_eq!(fs::read(&spool).unwrap(), original);
    }
    let mut changed = config; changed.options.max_citations -= 1;
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(changed, limits, Rc::clone(&events)), TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(fs::read(&spool).unwrap(), original);
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    assert_eq!(resumed.run(&mut Continue).unwrap().committed, 2); assert_eq!(events.executions.get(), 2);
}

#[test]
fn committed_result_corruption_is_never_recovered_as_an_uncommitted_tail() {
    let root = Root::new(); let inputs = inputs(); let config = config(1); let limits = limits(config);
    let events = Rc::new(Events::default()); let make = || Processor::new(config, limits, Rc::clone(&events));
    let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs, make(), &mut Continue).unwrap();
    job.run(&mut Continue).unwrap(); drop(job);
    let spool = root.0.join("results.spool"); let mut bytes = fs::read(&spool).unwrap();
    *bytes.last_mut().unwrap() ^= 1; fs::write(&spool, &bytes).unwrap();
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
        make(), TailPolicy::DiscardUncommitted, &mut Continue)), JobRunError::Storage(JobError::Authentication));
    assert_eq!(fs::read(&spool).unwrap(), bytes); assert_eq!(events.executions.get(), 2);
    assert!(!root.0.join("materialized.ndjson").exists());
}

#[test]
fn strict_question_envelopes_and_exact_encoded_line_cap_refuse_before_storage() {
    for bytes in [
        r#"{"id":"invalid","text":"source","task_args":{"question":"Who?","options":{}}}"#,
        r#"{"id":"invalid","text":"source","task_args":{"question":"Who?","question":"Where?"}}"#,
        r#"{"id":"invalid","text":"source","task_args":{"question":42}}"#,
        r#"{"id":"invalid","text":"source","task_args":{"question":"Who?","hierarchy":{}}}"#,
        r#"{"id":"invalid","text":"source","task_args":[]}"#,
    ] {
        let root = Root::new(); let config = config(0); let limits = limits(config); let events = Rc::new(Events::default());
        let inputs = [JobInput { id: "invalid", original: bytes.as_bytes(), normalized: bytes.as_bytes() }];
        assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs,
            Processor::new(config, limits, Rc::clone(&events)), &mut Continue)), JobRunError::InvalidEnvelope);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0); assert_eq!(events.executions.get(), 0);
    }
    let root = Root::new(); let mut config = config(0); config.mapping.chunks.max_input_bytes = 64;
    config.max_question_bytes = 64; let mut limits = limits(config); limits.max_input_bytes_per_item = 64;
    let bytes = r#"{"id":"document","text":"short","task_args":{"question":"Who is this?"}}"#;
    assert!(bytes.len() > limits.max_input_bytes_per_item);
    let inputs = [JobInput { id: "document", original: bytes.as_bytes(), normalized: bytes.as_bytes() }];
    assert!(JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs,
        Processor::new(config, limits, Rc::new(Events::default())), &mut Continue).is_err());
    assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
}

#[test]
fn serialization_failure_releases_complete_output_without_committing_or_refunding() {
    let root = Root::new(); let inputs = inputs(); let config = config(1); let limits = limits(config);
    let events = Rc::new(Events::default()); let mut processor = Processor::new(config, limits, Rc::clone(&events));
    let prepared = processor.planner.prepare_with_control(document(SOURCE, QUESTION), &mut Continue).unwrap();
    let debit = processor.planner.durable_work(&prepared).unwrap(); processor.fail_serialization = true;
    let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs, processor, &mut Continue).unwrap();
    assert_eq!(error(job.step(&mut Continue)), JobRunError::Storage(JobError::Serialization));
    assert_eq!((job.progress().committed, job.progress().attempts), (0, 1));
    assert_eq!(job.progress().reserved_work, debit);
    assert_eq!(fs::metadata(root.0.join("results.spool")).unwrap().len(), 0);
    assert_eq!((events.values_dropped.get(), events.releases.get()), (1, 1)); assert!(!events.held.get());
}
