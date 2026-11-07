//! Real owned-job persistence using explicitly synthetic transport output.
//! Public native planning/execution stays sealed to the corpus processor.
use super::*;
use crate::jobs::{TailPolicy, runner::{JobRunError, JobRunner}};
use serde::Serializer;
use std::{cell::Cell, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-faithfulness-document-job-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

#[derive(Default)]
struct Events { held: Cell<bool>, executions: Cell<usize>, releases: Cell<usize>, held_polls: Cell<usize> }
struct Hold(Rc<Events>);
impl Drop for Hold {
    fn drop(&mut self) { assert!(self.0.held.replace(false)); self.0.releases.set(self.0.releases.get() + 1); }
}
struct Datum { source: String, claim: String, events: Rc<Events> }
impl Serialize for Datum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get());
        ("synthetic-storage-source-claim-not-native-judgment", &self.source, &self.claim).serialize(serializer)
    }
}
struct Processor {
    identity: ExecutionIdentity, recipe: Int4FaithfulnessDocumentJobRecipe,
    limits: JobLimits, events: Rc<Events>, fail: bool,
}
impl Processor {
    fn new(map: Int4FaithfulnessMapLimits, limits: JobLimits, events: Rc<Events>) -> Self {
        Self { identity: identity(), recipe: recipe(map, policy(), budget(), JudgeLimits::default(), limits),
            limits, events, fail: false }
    }
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), BatchItemFailure> {
    match control.checkpoint(0) { Some(cause) => Err(BatchItemFailure::fatal(BatchFault::cancelled(cause))), None => Ok(()) }
}
impl BatchProcessor for Processor {
    type Args = Int4FaithfulnessDocumentArgs;
    type Prepared = BatchDocument<Int4FaithfulnessDocumentArgs>;
    type Output = GuardedOutput<Datum, Hold>;
    fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<Self::Prepared, BatchItemFailure> {
        panic!("missing preparation control")
    }
    fn prepare_with_control<C: DecodeStepControl>(&mut self, document: BatchDocument<Self::Args>, control: &mut C)
        -> Result<Self::Prepared, BatchItemFailure> {
        checkpoint(control)?;
        if document.task_args.as_ref().is_none_or(|args| args.claim.is_empty()) {
            return Err(BatchItemFailure::reject(BatchCode::Planning));
        }
        Ok(document)
    }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork {
        BatchWork { forward_positions: work(1).forward_positions, projected_logits: work(1).projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: Self::Prepared, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        panic!("missing durable coordinates")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, document: Self::Prepared,
        context: BatchRequestContext, control: &mut C) -> Result<Self::Output, BatchItemFailure> {
        checkpoint(control)?; assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        self.events.executions.set(self.events.executions.get() + 1);
        if self.fail { return Err(BatchItemFailure::fatal(BatchCode::Execution)); }
        assert!(!self.events.held.replace(true));
        Ok(GuardedOutput::new(Datum { source: document.text, claim: document.task_args.unwrap().claim,
            events: Rc::clone(&self.events) }, Hold(Rc::clone(&self.events))))
    }
}
impl DurableBatchProcessor for Processor {
    type Recipe = Int4FaithfulnessDocumentJobRecipe;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Self::Recipe { &self.recipe }
    fn validate_job_limits(&self, limits: JobLimits) -> Result<(), BatchFault> {
        if limits != self.limits { return Err(BatchCode::InvalidLimits.into()); } Ok(())
    }
    fn durable_work(&self, _: &Self::Prepared) -> Result<JobWork, BatchItemFailure> {
        Ok(JobWork::from_int4(work(1), 0))
    }
    fn max_result_bytes(&self, _: &Self::Prepared) -> u64 { self.recipe.max_stored_result_bytes }
}
fn inputs() -> [JobInput<'static>; 2] {
    [
        ("first", r#"{"id":"first","text":"Original é source final tail <think>literal</think>","task_args":{"claim":"exact 上海 claim"}}"#),
        ("second", r#"{"id":"second","text":"Second complete 上海 source","task_args":{"claim":"a different é claim"}}"#),
    ].map(|(id, text)| JobInput { id, original: text.as_bytes(), normalized: text.as_bytes() })
}
fn key() -> JobSecret { JobSecret::from_bytes([7; 32]) }
fn error<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(error) => error, Ok(_) => panic!("expected refusal") }
}

#[test]
fn resumed_original_sources_and_claims_match_uninterrupted_order_without_reexecution() {
    let root = Root::new(); let full = Root::new(); let inputs = inputs();
    let events = Rc::new(Events::default());
    let make = || Processor::new(mapping(), limits(), Rc::clone(&events));
    let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(), &inputs, make(), &mut Continue).unwrap();
    assert_eq!(job.step(&mut Continue).unwrap().unwrap().committed, 1); drop(job);
    assert_eq!(events.executions.get(), 1); assert_eq!(events.releases.get(), 1);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let complete = resumed.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 2));
    assert_eq!(complete.reserved_work, JobWork::from_int4(work(2), 0));
    assert_eq!(events.executions.get(), 2); assert_eq!(events.releases.get(), 2); assert!(!events.held.get());
    resumed.verify(&mut Continue).unwrap(); resumed.materialize_ordered(&mut Continue).unwrap();
    let mut baseline = JobRunner::create(&full.0, key(), JobId([9; 16]), limits(), &inputs,
        Processor::new(mapping(), limits(), Rc::new(Events::default())), &mut Continue).unwrap();
    baseline.run(&mut Continue).unwrap(); baseline.materialize_ordered(&mut Continue).unwrap();
    let output = fs::read(root.0.join("materialized.ndjson")).unwrap();
    assert_eq!(output, fs::read(full.0.join("materialized.ndjson")).unwrap());
    let output = String::from_utf8(output).unwrap();
    for original in ["Original é source final tail", "exact 上海 claim", "a different é claim"] {
        assert!(output.contains(original));
    }
}

#[test]
fn failed_source_claim_attempt_preserves_each_lifetime_axis_after_restart() {
    for field in 0..5 {
        let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
        let mut map = mapping(); map.max_model_work = work(1);
        let mut limits = limits(); let mut maximum = limits.max_work.int4_model_work();
        *axis(&mut maximum, field) = *axis(&mut work(1), field); limits.max_work = JobWork::from_int4(maximum, 0);
        let mut processor = Processor::new(map, limits, Rc::clone(&events)); processor.fail = true;
        let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits, &inputs, processor, &mut Continue).unwrap();
        assert_eq!(error(job.step(&mut Continue)), JobRunError::Processor(BatchCode::Execution.into()));
        assert_eq!(job.progress().committed, 0); assert_eq!(job.progress().reserved_work, JobWork::from_int4(work(1), 0));
        drop(job);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits, &inputs,
            Processor::new(map, limits, Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
        assert_eq!(error(resumed.step(&mut Continue)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(events.executions.get(), 1); assert_eq!(resumed.progress().attempts, 1);
    }
}

struct CancelAfterSync(Rc<Events>);
impl DecodeStepControl for CancelAfterSync {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if !self.0.held.get() { return None; }
        let count = self.0.held_polls.get() + 1; self.0.held_polls.set(count);
        (count == 2).then_some(DecodeCancellationKind::PollQuota)
    }
}
#[test]
fn changed_claim_refuses_before_tail_repair_and_cancellation_never_refunds_work() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let make = || Processor::new(mapping(), limits(), Rc::clone(&events));
    let mut stop = CancelAfterSync(Rc::clone(&events));
    let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(), &inputs, make(), &mut stop).unwrap();
    assert_eq!(error(job.step(&mut stop)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::PollQuota)));
    assert_eq!(job.progress().committed, 0); assert_eq!(job.progress().reserved_work, JobWork::from_int4(work(1), 0));
    assert_eq!(events.releases.get(), 1); assert!(!events.held.get()); drop(job);
    let changed_text = std::str::from_utf8(inputs[0].original).unwrap().replace("exact 上海 claim", "changed 上海 claim");
    let changed = [JobInput { id: inputs[0].id, original: changed_text.as_bytes(), normalized: changed_text.as_bytes() },
        JobInput { id: inputs[1].id, original: inputs[1].original, normalized: inputs[1].normalized }];
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &changed,
        make(), TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Population)));
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    let complete = resumed.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 3));
    assert_eq!(complete.reserved_work, JobWork::from_int4(work(3), 0));
}
