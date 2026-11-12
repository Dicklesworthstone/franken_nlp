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
struct Events {
    held: Cell<bool>, preparations: Cell<usize>, executions: Cell<usize>,
    serializations: Cell<usize>, values_dropped: Cell<usize>, releases: Cell<usize>, held_polls: Cell<usize>,
}
struct Hold(Rc<Events>);
impl Drop for Hold {
    fn drop(&mut self) {
        assert_eq!(self.0.values_dropped.get(), self.0.releases.get() + 1,
            "complete private output must be destroyed before its reservation");
        assert!(self.0.held.replace(false)); self.0.releases.set(self.0.releases.get() + 1);
    }
}
struct Datum { source: String, claim: String, fail_serialization: bool, events: Rc<Events> }
impl Serialize for Datum {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get());
        self.events.serializations.set(self.events.serializations.get() + 1);
        if self.fail_serialization {
            return Err(<S::Error as serde::ser::Error>::custom("synthetic output encoding refusal"));
        }
        ("synthetic-storage-source-claim-not-native-judgment", &self.source, &self.claim).serialize(serializer)
    }
}
impl Drop for Datum {
    fn drop(&mut self) {
        assert!(self.events.held.get());
        self.events.values_dropped.set(self.events.values_dropped.get() + 1);
    }
}
struct Processor {
    identity: ExecutionIdentity, recipe: Int4FaithfulnessDocumentJobRecipe,
    limits: JobLimits, events: Rc<Events>, fail: bool, fail_serialization: bool,
}
impl Processor {
    fn new(map: Int4FaithfulnessMapLimits, limits: JobLimits, events: Rc<Events>) -> Self {
        Self { identity: identity(), recipe: recipe(map, policy(), budget(), JudgeLimits::default(), limits),
            limits, events, fail: false, fail_serialization: false }
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
        self.events.preparations.set(self.events.preparations.get() + 1);
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
            fail_serialization: self.fail_serialization, events: Rc::clone(&self.events) }, Hold(Rc::clone(&self.events))))
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
    assert_eq!(job.step(&mut Continue).unwrap().unwrap().committed, 1);
    let retained: serde_json::Value = serde_json::from_slice(&job.read_committed(0, &mut Continue).unwrap()).unwrap();
    assert_eq!(retained[1], "Original é source final tail <think>literal</think>");
    assert_eq!(retained[2], "exact 上海 claim"); drop(job);
    assert_eq!(events.preparations.get(), 1);
    assert_eq!(events.executions.get(), 1); assert_eq!(events.releases.get(), 1);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let complete = resumed.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 2));
    assert_eq!(complete.reserved_work, JobWork::from_int4(work(2), 0));
    assert_eq!(events.executions.get(), 2); assert_eq!(events.releases.get(), 2); assert!(!events.held.get());
    assert_eq!(events.preparations.get(), 2); assert!(events.serializations.get() >= 4);
    resumed.verify(&mut Continue).unwrap();
    let materialized = resumed.materialize_ordered(&mut Continue).unwrap(); drop(resumed);
    let mut completed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        make(), TailPolicy::Refuse, &mut Continue).unwrap();
    assert_eq!(completed.run(&mut Continue).unwrap(), materialized);
    assert_eq!((events.preparations.get(), events.executions.get()), (2, 2)); drop(completed);
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
        assert_eq!(resumed.progress().reserved_work, JobWork::from_int4(work(1), 0));
        assert_eq!(fs::metadata(root.0.join("results.spool")).unwrap().len(), 0);
        assert_eq!(events.releases.get(), 0); assert!(!events.held.get());
    }
}

struct CancelAfterSync(Rc<Events>);
impl DecodeStepControl for CancelAfterSync {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if !self.0.held.get() { return None; }
        let count = self.0.held_polls.get() + 1; self.0.held_polls.set(count);
        // Attempt::commit polls once before encoding, then after spool sync
        // and before the journal authorizes the complete result frame.
        if count == 2 {
            assert!(self.0.serializations.get() >= 2);
            assert_eq!(self.0.values_dropped.get(), 0); assert_eq!(self.0.releases.get(), 0);
        }
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
    assert_eq!(job.progress().attempts, 1); assert_eq!(events.values_dropped.get(), 1);
    assert_eq!(events.releases.get(), 1); assert!(!events.held.get()); drop(job);
    let spool = root.0.join("results.spool"); let tail = fs::read(&spool).unwrap(); assert!(!tail.is_empty());
    for (from, to) in [("exact 上海 claim", "changed 上海 claim"),
        ("exact 上海 claim", " exact 上海 claim "), ("final tail", "CHANGED final tail")] {
        let changed_text = std::str::from_utf8(inputs[0].original).unwrap().replace(from, to);
        let changed = [JobInput { id: inputs[0].id, original: changed_text.as_bytes(), normalized: changed_text.as_bytes() },
            JobInput { id: inputs[1].id, original: inputs[1].original, normalized: inputs[1].normalized }];
        assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &changed,
            make(), TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Population)));
        assert_eq!(fs::read(&spool).unwrap(), tail);
    }
    let mut altered = policy(); altered.minimum_candidate_weight_ppm += 1;
    let mut changed = make(); changed.recipe = recipe(mapping(), altered, budget(), JudgeLimits::default(), limits());
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        changed, TailPolicy::DiscardUncommitted, &mut Continue)),
        JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)));
    assert_eq!(fs::read(&spool).unwrap(), tail);
    assert_eq!(error(JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        make(), TailPolicy::Refuse, &mut Continue)), JobRunError::Storage(JobError::UncommittedTail));
    assert_eq!(events.executions.get(), 1); assert!(!root.0.join("materialized.ndjson").exists());
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        make(), TailPolicy::DiscardUncommitted, &mut Continue).unwrap();
    let complete = resumed.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 3));
    assert_eq!(complete.reserved_work, JobWork::from_int4(work(3), 0));
}

#[test]
fn invalid_claim_envelopes_refuse_before_job_files_or_processor_execution() {
    for bytes in [
        r#"{"id":"invalid","text":"complete source","task_args":{"claim":"claim","policy":{}}}"#,
        r#"{"id":"invalid","text":"complete source","task_args":{"claim":"claim","budget":{}}}"#,
        r#"{"id":"invalid","text":"complete source","task_args":{"claim":"claim","mode":"faithfulness"}}"#,
        r#"{"id":"invalid","text":"complete source","task_args":{"claim":"first","claim":"second"}}"#,
        r#"{"id":"invalid","text":"complete source","task_args":{"claim":42}}"#,
        r#"{"id":"invalid","text":"complete source","task_args":[]}"#,
    ] {
        let root = Root::new(); let events = Rc::new(Events::default());
        let inputs = [JobInput { id: "invalid", original: bytes.as_bytes(), normalized: bytes.as_bytes() }];
        assert_eq!(error(JobRunner::create(&root.0, key(), JobId([9; 16]), limits(), &inputs,
            Processor::new(mapping(), limits(), Rc::clone(&events)), &mut Continue)), JobRunError::InvalidEnvelope);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
        assert_eq!((events.preparations.get(), events.executions.get()), (0, 0));
    }
}

#[test]
fn failed_result_encoding_releases_ownership_without_a_committed_prefix_or_refund() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let mut processor = Processor::new(mapping(), limits(), Rc::clone(&events)); processor.fail_serialization = true;
    let mut job = JobRunner::create(&root.0, key(), JobId([9; 16]), limits(), &inputs, processor, &mut Continue).unwrap();
    assert_eq!(error(job.step(&mut Continue)), JobRunError::Storage(JobError::Serialization));
    assert_eq!((job.progress().committed, job.progress().attempts), (0, 1));
    assert_eq!(job.progress().reserved_work, JobWork::from_int4(work(1), 0));
    assert_eq!(fs::metadata(root.0.join("results.spool")).unwrap().len(), 0);
    assert_eq!((events.values_dropped.get(), events.releases.get()), (1, 1));
    assert!(!events.held.get()); assert!(!root.0.join("materialized.ndjson").exists()); drop(job);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([9; 16]), limits(), &inputs,
        Processor::new(mapping(), limits(), Rc::clone(&events)), TailPolicy::Refuse, &mut Continue).unwrap();
    let complete = resumed.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 3));
    assert_eq!(complete.reserved_work, JobWork::from_int4(work(3), 0));
    assert_eq!((events.values_dropped.get(), events.releases.get()), (3, 3));
}


#[test]
fn retained_source_claim_replay_authenticates_every_compact_ceiling_before_reexecution() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let retention = crate::jobs::InputRetentionLimits { max_spool_bytes: 65_536 };
    let make = || {
        let mut processor = Processor::new(mapping(), limits(), Rc::clone(&events));
        processor.recipe.input_encoding = Some(compact_policy());
        processor
    };
    let mut job = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([9; 16]), limits(),
        retention, &inputs, make(), &mut Continue).unwrap();
    assert_eq!(job.step(&mut Continue).unwrap().unwrap().committed, 1);
    let first = job.read_committed(0, &mut Continue).unwrap();
    drop(job);
    let spool = fs::read(root.0.join("results.spool")).unwrap();
    for axis in 0..5 {
        let mut changed = make();
        if axis == 4 { changed.recipe.input_encoding = None; }
        else {
            let Some(UntrustedEncoding::SafeBpeV1(policy)) = &mut changed.recipe.input_encoding else {
                panic!("compact recipe required");
            };
            match axis { 0 => policy.max_input_bytes -= 1, 1 => policy.max_output_tokens -= 1,
                2 => policy.max_pair_checks -= 1, _ => policy.max_pair_bytes -= 1 }
        }
        assert_eq!(error(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]),
            limits(), retention, changed, TailPolicy::DiscardUncommitted, &mut Continue)),
            JobRunError::Storage(JobError::Mismatch(MismatchField::Recipe)), "axis {axis}");
        assert_eq!((events.preparations.get(), events.executions.get()), (1, 1));
        assert_eq!(fs::read(root.0.join("results.spool")).unwrap(), spool);
    }
    let mut restored = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([9; 16]),
        limits(), retention, make(), TailPolicy::Refuse, &mut Continue).unwrap();
    let complete = restored.run(&mut Continue).unwrap();
    assert_eq!((complete.committed, complete.attempts), (2, 2));
    assert_eq!(complete.reserved_work, JobWork::from_int4(work(2), 0));
    assert_eq!((events.preparations.get(), events.executions.get()), (2, 2));
    assert_eq!(restored.read_committed(0, &mut Continue).unwrap(), first);
    let second: serde_json::Value = serde_json::from_slice(&restored.read_committed(1, &mut Continue).unwrap()).unwrap();
    assert_eq!(second[1], "Second complete 上海 source");
    assert_eq!(second[2], "a different é claim");
    restored.verify(&mut Continue).unwrap();
}
