//! Storage/runner fixtures, not native-model accuracy or crash qualification.
mod streaming;

use super::*;
use crate::{batch::{BatchCode, BatchWork}, jobs::{MismatchField, tests::{identity, key, limits, work, Control}},
    native_engine::decode::DecodeCancellationKind};
use serde::{Deserialize, Serializer};
use std::{cell::{Cell, RefCell}, fs, os::unix::fs::DirBuilderExt, path::PathBuf, rc::Rc,
    sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-job-runner-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
fn inputs() -> [JobInput<'static>; 2] {
    let a = br#"{"id":"alpha","text":"Alice private source","task_args":{"v":1}}"#;
    let b = br#"{"id":"beta","text":"Bob private source","task_args":{"v":2}}"#;
    [JobInput { id: "alpha", original: a, normalized: a }, JobInput { id: "beta", original: b, normalized: b }]
}
fn retention() -> InputRetentionLimits { InputRetentionLimits { max_spool_bytes: 1 << 20 } }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args { v: u64 }
#[derive(Clone, Serialize)]
struct Config { seed: u64 }
#[derive(Default)]
struct Events {
    prepared: RefCell<Vec<String>>, executed: RefCell<Vec<u64>>, held: Cell<bool>,
    serialized: Cell<usize>, commit_checkpoints: Cell<usize>,
}
struct Output { value: u64, events: Rc<Events>, fail: bool }
impl Serialize for Output {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.events.held.get(), "admission guard lost during serialization");
        self.events.serialized.set(self.events.serialized.get() + 1);
        if self.fail { return Err(serde::ser::Error::custom("private fixture serialization failure")); }
        self.value.serialize(serializer)
    }
}
impl Drop for Output { fn drop(&mut self) { assert!(self.events.held.replace(false)); } }
struct Processor {
    identity: ExecutionIdentity, recipe: Config, events: Rc<Events>,
    fail_sequence: Option<u64>, bad_work: bool, fail_serialization: bool,
}
impl Processor {
    fn new(events: Rc<Events>) -> Self {
        Self { identity: identity(), recipe: Config { seed: 7 }, events,
            fail_sequence: None, bad_work: false, fail_serialization: false }
    }
}
impl BatchProcessor for Processor {
    type Args = Args;
    type Prepared = u64;
    type Output = Output;
    fn prepare(&mut self, document: BatchDocument<Args>) -> Result<u64, BatchItemFailure> {
        self.events.prepared.borrow_mut().push(document.id);
        Ok(document.task_args.ok_or_else(|| BatchItemFailure::reject(BatchCode::Planning))?.v)
    }
    fn planned_work(&self, _: &u64) -> BatchWork {
        BatchWork { forward_positions: work().model.forward_positions + u64::from(self.bad_work),
            projected_logits: work().model.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: u64, _: &mut C) -> Result<Output, BatchItemFailure> {
        panic!("durable runner must pass stable delivery context")
    }
    fn execute_with_context<C: DecodeStepControl>(&mut self, value: u64, context: BatchRequestContext, _: &mut C)
        -> Result<Output, BatchItemFailure> {
        assert_eq!(context.epoch, 1); assert_eq!(context.input_line, context.request_seq);
        self.events.executed.borrow_mut().push(context.request_seq);
        if self.fail_sequence == Some(context.request_seq) {
            return Err(BatchItemFailure::reject(BatchCode::Execution));
        }
        assert!(!self.events.held.replace(true));
        Ok(Output { value, events: self.events.clone(), fail: self.fail_serialization })
    }
}
impl DurableBatchProcessor for Processor {
    type Recipe = Config;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &Config { &self.recipe }
    fn durable_work(&self, _: &u64) -> Result<JobWork, BatchItemFailure> { Ok(work()) }
    fn max_result_bytes(&self, _: &u64) -> u64 { 128 }
}
struct GuardControl { events: Rc<Events>, cancel_held: bool }
impl DecodeStepControl for GuardControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.events.held.get() {
            self.events.commit_checkpoints.set(self.events.commit_checkpoints.get() + 1);
            if self.cancel_held { return Some(DecodeCancellationKind::Deadline); }
        }
        None
    }
}
fn err<T>(result: Result<T, JobRunError>) -> JobRunError {
    match result { Err(error) => error, Ok(_) => panic!("expected refusal") }
}

#[test]
fn retained_resume_owns_exact_inputs_after_original_population_is_dropped() {
    let root = Root::new(); let full = Root::new(); let mut control = Control::default();
    let seen = Rc::new(Events::default());
    {
        // A heap-owned record with CR and noncanonical whitespace catches
        // reconstruction through JSON serialization instead of exact bytes.
        let a = b" {\"id\":\"alpha\", \"text\":\"Alice private source\", \"task_args\":{\"v\":1}}\r".to_vec();
        let b = inputs()[1].original.to_vec();
        let population = [JobInput { id: "alpha", original: &a, normalized: &a },
            JobInput { id: "beta", original: &b, normalized: &b }];
        let mut run = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
            retention(), &population, Processor::new(seen.clone()), &mut control).unwrap();
        assert_eq!(run.step(&mut control).unwrap().unwrap().committed, 1);
        let mut baseline = JobRunner::create(&full.0, key(), JobId([7;16]), limits(), &population,
            Processor::new(Rc::new(Events::default())), &mut control).unwrap();
        baseline.run(&mut control).unwrap(); baseline.materialize_ordered(&mut control).unwrap();
    }
    let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        retention(), Processor::new(seen.clone()), TailPolicy::Refuse, &mut control).unwrap();
    resumed.run(&mut control).unwrap(); resumed.materialize_ordered(&mut control).unwrap();
    assert_eq!(*seen.prepared.borrow(), vec!["alpha", "beta"]);
    assert_eq!(*seen.executed.borrow(), vec![1, 2]);
    assert_eq!(resumed.progress().attempts, 2);
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(),
        fs::read(full.0.join("materialized.ndjson")).unwrap());
}

#[test]
fn retained_input_is_explicit_and_does_not_change_original_resume_authority() {
    let root = Root::new(); let retained_root = Root::new(); let population = inputs();
    let mut control = Control::default();
    drop(JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &population,
        Processor::new(Rc::new(Events::default())), &mut control).unwrap());
    assert!(!root.0.join("inputs.spool").exists());
    assert_eq!(err(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        retention(), Processor::new(Rc::new(Events::default())), TailPolicy::DiscardUncommitted, &mut control)),
        JobRunError::Storage(JobError::InputsNotRetained));
    drop(JobRunner::create_with_retained_inputs(&retained_root.0, key(), JobId([7;16]), limits(), retention(),
        &population, Processor::new(Rc::new(Events::default())), &mut control).unwrap());
    let mut resumed = JobRunner::resume(&retained_root.0, key(), JobId([7;16]), limits(), &population,
        Processor::new(Rc::new(Events::default())), TailPolicy::Refuse, &mut control).unwrap();
    assert_eq!(resumed.run(&mut control).unwrap().committed, 2);
}

#[test]
fn retention_limit_refuses_before_any_job_file_is_created() {
    let root = Root::new(); let population = inputs(); let events = Rc::new(Events::default());
    assert_eq!(err(JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        InputRetentionLimits { max_spool_bytes: 1 }, &population, Processor::new(events.clone()),
        &mut Control::default())), JobRunError::Storage(JobError::Limit));
    assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
}

#[test]
fn retained_resume_refreezes_recipe_identity_and_limits_before_result_repair() {
    use std::io::Write;
    let root = Root::new(); let population = inputs(); let mut control = Control::default();
    drop(JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(), retention(),
        &population, Processor::new(Rc::new(Events::default())), &mut control).unwrap());
    let mut spool = fs::OpenOptions::new().append(true).open(root.0.join("results.spool")).unwrap();
    spool.write_all(b"uncommitted result tail").unwrap(); spool.sync_all().unwrap(); drop(spool);
    let before = fs::read(root.0.join("results.spool")).unwrap();
    let input_before = fs::read(root.0.join("inputs.spool")).unwrap();
    for field in [MismatchField::Recipe, MismatchField::Execution, MismatchField::Limits] {
        let events = Rc::new(Events::default()); let mut processor = Processor::new(events.clone());
        let mut cap = limits();
        match field {
            MismatchField::Recipe => processor.recipe.seed += 1,
            MismatchField::Execution => processor.identity.backend_semantic_version.push_str("-changed"),
            MismatchField::Limits => cap.max_attempts += 1,
            _ => unreachable!(),
        }
        assert_eq!(err(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), cap,
            retention(), processor, TailPolicy::DiscardUncommitted, &mut control)),
            JobRunError::Storage(JobError::Mismatch(field)));
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
        assert_eq!(fs::read(root.0.join("results.spool")).unwrap(), before);
        assert_eq!(fs::read(root.0.join("inputs.spool")).unwrap(), input_before);
    }
}

#[test]
fn retained_resume_validates_typed_arguments_in_committed_and_pending_records_before_repair() {
    use std::io::Write;
    for invalid_ordinal in 0..2 {
        let root = Root::new(); let events = Rc::new(Events::default());
        let processor = Processor::new(events.clone()); let mut control = Control::default();
        let invalid = if invalid_ordinal == 0 {
            br#"{"id":"alpha","text":"valid envelope","task_args":{"v":"wrong type"}}"#.as_slice()
        } else {
            br#"{"id":"beta","text":"valid envelope","task_args":{"v":"wrong type"}}"#.as_slice()
        };
        let mut population = inputs();
        population[invalid_ordinal] = JobInput { id: population[invalid_ordinal].id,
            original: invalid, normalized: invalid };
        // The lower-level owner accepts opaque typed arguments. Produce an
        // authentic fixture with the real runner recipe, so only P::Args can
        // detect this incompatibility; stored population hashes all agree.
        let mut fresh = manifest_builder(&key(), JobId([7;16]), limits(), &processor, &mut control).unwrap();
        for input in &population { fresh.push(&key(), input, &mut control).unwrap(); }
        let mut job = OwnedJob::create_with_retained_inputs(&root.0, key(), fresh.finish(&key()).unwrap(),
            retention(), &population, &mut control).unwrap();
        job.begin(&population[0], work(), &mut control).unwrap().commit(&1_u64, &mut control).unwrap();
        drop(job);
        let mut output = fs::OpenOptions::new().append(true).open(root.0.join("results.spool")).unwrap();
        output.write_all(b"uncommitted result tail").unwrap(); output.sync_all().unwrap(); drop(output);
        let before = fs::read(root.0.join("results.spool")).unwrap();
        assert_eq!(err(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
            retention(), processor, TailPolicy::DiscardUncommitted, &mut control)), JobRunError::InvalidEnvelope);
        assert_eq!(fs::read(root.0.join("results.spool")).unwrap(), before);
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
        let stored = crate::jobs::StoredJob::open(&root.0, key(), JobId([7;16]), limits(), &mut control).unwrap();
        assert_eq!(stored.status().unwrap().committed, 1);
    }
}

#[test]
fn retained_replay_read_failure_or_cancellation_stops_before_new_debit_and_releases_owners() {
    use std::io::{Seek, SeekFrom, Write};
    for corrupt in [true, false] {
        let root = Root::new(); let population = inputs(); let mut control = Control::default();
        drop(JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(), retention(),
            &population, Processor::new(Rc::new(Events::default())), &mut control).unwrap());
        let original = fs::read(root.0.join("inputs.spool")).unwrap();
        let events = Rc::new(Events::default());
        let mut run = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
            retention(), Processor::new(events.clone()), TailPolicy::Refuse, &mut control).unwrap();
        #[cfg(target_os = "linux")]
        let retained_fd = fs::read_dir("/proc/self/fd").unwrap().filter_map(Result::ok)
            .map(|entry| entry.path()).find(|path| fs::read_link(path).ok().as_deref()
                == Some(root.0.join("inputs.spool").as_path())).expect("retained file stays open with its owner");
        if corrupt {
            let mut file = fs::OpenOptions::new().write(true).open(root.0.join("inputs.spool")).unwrap();
            file.seek(SeekFrom::Start(crate::jobs::retained::HEADER_BYTES as u64)).unwrap();
            file.write_all(b"X").unwrap(); file.sync_all().unwrap();
        }
        let mut cancel = Control { calls: 0, stop_at: if corrupt { None } else { Some(3) } };
        let expected = if corrupt { JobError::Authentication }
            else { JobError::Cancelled(DecodeCancellationKind::Deadline) };
        assert_eq!(err(run.step(&mut cancel)), JobRunError::Storage(expected));
        assert!(run.is_stopped()); assert!(run.job.is_poisoned());
        assert_eq!(run.progress().attempts, 0); assert_eq!(run.progress().committed, 0);
        assert_eq!(err(run.step(&mut control)), JobRunError::Stopped);
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
        assert_eq!(crate::jobs::tests::error(crate::jobs::StoredJob::open(&root.0, key(), JobId([7;16]),
            limits(), &mut control)), JobError::Busy);
        drop(run);
        #[cfg(target_os = "linux")]
        assert_ne!(fs::read_link(retained_fd).ok().as_deref(), Some(root.0.join("inputs.spool").as_path()));
        // Restore the fixture's externally damaged bytes before checking that
        // the exclusive owner can reopen; this grants no in-session retry.
        if corrupt { fs::write(root.0.join("inputs.spool"), &original).unwrap(); }
        let stored = crate::jobs::StoredJob::open(&root.0, key(), JobId([7;16]), limits(), &mut control).unwrap();
        assert_eq!(stored.status().unwrap().attempts, 0);
    }
}

#[test]
fn complete_retained_resume_starts_at_eof_without_another_delivery() {
    let root = Root::new(); let population = inputs(); let mut control = Control::default();
    let mut created = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        retention(), &population, Processor::new(Rc::new(Events::default())), &mut control).unwrap();
    assert_eq!(created.run(&mut control).unwrap().committed, 2); drop(created);
    let events = Rc::new(Events::default());
    let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        retention(), Processor::new(events.clone()), TailPolicy::Refuse, &mut control).unwrap();
    match &resumed.inputs { Inputs::Retained(cursor) => assert_eq!(cursor.ordinal(), 2),
        Inputs::Borrowed(_) => panic!("retained replay must own its cursor") }
    assert!(resumed.step(&mut control).unwrap().is_none());
    assert_eq!(resumed.run(&mut control).unwrap().attempts, 2);
    assert_eq!(resumed.read_committed(0, &mut control).unwrap(), b"1");
    assert_eq!(resumed.read_committed(1, &mut control).unwrap(), b"2");
    assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
}

#[test]
fn damaged_retained_input_blocks_repair_even_when_original_inputs_are_supplied() {
    use std::io::{Seek, SeekFrom, Write};
    let root = Root::new(); let population = inputs(); let mut control = Control::default();
    drop(JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(), retention(),
        &population, Processor::new(Rc::new(Events::default())), &mut control).unwrap());
    let mut input = fs::OpenOptions::new().write(true).open(root.0.join("inputs.spool")).unwrap();
    input.seek(SeekFrom::Start(crate::jobs::retained::HEADER_BYTES as u64)).unwrap();
    input.write_all(b"X").unwrap(); input.sync_all().unwrap(); drop(input);
    let mut output = fs::OpenOptions::new().append(true).open(root.0.join("results.spool")).unwrap();
    output.write_all(b"tail").unwrap(); output.sync_all().unwrap(); drop(output);
    for use_retained in [true, false] {
        let events = Rc::new(Events::default()); let processor = Processor::new(events.clone());
        let result = if use_retained {
            JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(), retention(),
                processor, TailPolicy::DiscardUncommitted, &mut control)
        } else {
            JobRunner::resume(&root.0, key(), JobId([7;16]), limits(), &population, processor,
                TailPolicy::DiscardUncommitted, &mut control)
        };
        assert_eq!(err(result), JobRunError::Storage(JobError::Authentication));
        assert!(events.executed.borrow().is_empty());
        assert_eq!(fs::read(root.0.join("results.spool")).unwrap(), b"tail");
    }
}

#[test]
fn retained_replay_keeps_failed_attempt_work_and_stable_delivery_coordinates() {
    let root = Root::new(); let population = inputs(); let events = Rc::new(Events::default());
    let mut processor = Processor::new(events.clone()); processor.fail_sequence = Some(1);
    let mut control = Control::default();
    let mut run = JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        retention(), &population, processor, &mut control).unwrap();
    assert!(run.step(&mut control).is_err()); assert_eq!(run.progress().attempts, 1); drop(run);
    let mut resumed = JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        retention(), Processor::new(events.clone()), TailPolicy::Refuse, &mut control).unwrap();
    let progress = resumed.run(&mut control).unwrap();
    assert_eq!(progress.attempts, 3);
    assert_eq!(progress.reserved_work, work().checked_add(work()).unwrap().checked_add(work()).unwrap());
    assert_eq!(*events.executed.borrow(), vec![1, 1, 2]);
}

#[test]
fn current_input_read_cap_can_tighten_without_rewriting_creation_consent() {
    let root = Root::new(); let population = inputs(); let mut control = Control::default();
    drop(JobRunner::create_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(), retention(),
        &population, Processor::new(Rc::new(Events::default())), &mut control).unwrap());
    let length = fs::metadata(root.0.join("inputs.spool")).unwrap().len();
    assert_eq!(err(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        InputRetentionLimits { max_spool_bytes: length - 1 }, Processor::new(Rc::new(Events::default())),
        TailPolicy::Refuse, &mut control)), JobRunError::Storage(JobError::Limit));
    drop(JobRunner::resume_with_retained_inputs(&root.0, key(), JobId([7;16]), limits(),
        InputRetentionLimits { max_spool_bytes: length }, Processor::new(Rc::new(Events::default())),
        TailPolicy::Refuse, &mut control).unwrap());
    let stored = crate::jobs::StoredJob::open(&root.0, key(), JobId([7;16]), limits(), &mut control).unwrap();
    let report = stored.status().unwrap().retained_inputs.unwrap();
    assert_eq!(report.spool_bytes, length); assert_eq!(report.max_spool_bytes, retention().max_spool_bytes);
}

#[test]
fn guards_survive_serialization_and_commit_checkpoints_then_release() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let mut control = GuardControl { events: events.clone(), cancel_held: false };
    let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(events.clone()), &mut control).unwrap();
    let progress = run.run(&mut control).unwrap();
    assert_eq!(progress.committed, 2); assert_eq!(progress.attempts, 2);
    assert_eq!(*events.executed.borrow(), vec![1,2]); assert!(!events.held.get());
    assert!(events.serialized.get() >= 4); assert!(events.commit_checkpoints.get() >= 2);
    assert_eq!(run.read_committed(0, &mut control).unwrap(), b"1");
    assert_eq!(run.read_committed(1, &mut control).unwrap(), b"2");
    assert!(run.materialize_ordered(&mut control).unwrap().materialized);
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), b"1\n2\n");
    assert!(run.step(&mut control).unwrap().is_none());
    assert_eq!(*events.executed.borrow(), vec![1,2]);
}
#[test]
fn interrupted_and_resumed_output_matches_uninterrupted_without_reexecution() {
    let root = Root::new(); let full = Root::new(); let inputs = inputs(); let mut control = Control::default();
    let seen = Rc::new(Events::default());
    let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(seen.clone()), &mut control).unwrap();
    assert_eq!(run.step(&mut control).unwrap().unwrap().committed, 1); drop(run);
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(seen.clone()), TailPolicy::Refuse, &mut control).unwrap();
    resumed.run(&mut control).unwrap(); resumed.materialize_ordered(&mut control).unwrap();
    assert_eq!(*seen.prepared.borrow(), vec!["alpha", "beta"]);
    assert_eq!(*seen.executed.borrow(), vec![1,2]);
    let mut baseline = JobRunner::create(&full.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(Rc::new(Events::default())), &mut control).unwrap();
    baseline.run(&mut control).unwrap(); baseline.materialize_ordered(&mut control).unwrap();
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), fs::read(full.0.join("materialized.ndjson")).unwrap());
}
#[test]
fn cancelled_output_is_not_committed_and_attempt_is_not_refunded() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let mut control = GuardControl { events: events.clone(), cancel_held: true };
    let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(events.clone()), &mut control).unwrap();
    assert_eq!(err(run.step(&mut control)), JobRunError::Storage(JobError::Cancelled(DecodeCancellationKind::Deadline)));
    assert!(!events.held.get()); assert!(run.is_stopped()); assert_eq!(run.progress().attempts, 1);
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().reserved_work, work());
    assert_eq!(err(run.step(&mut control)), JobRunError::Stopped); drop(run);
    let mut control = Control::default();
    let mut resumed = JobRunner::resume(&root.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(events.clone()), TailPolicy::DiscardUncommitted, &mut control).unwrap();
    let progress = resumed.run(&mut control).unwrap();
    assert_eq!(progress.attempts, 3); assert_eq!(progress.reserved_work, work().checked_add(work()).unwrap().checked_add(work()).unwrap());
    assert_eq!(*events.executed.borrow(), vec![1,1,2]);
}
#[test]
fn even_recoverable_batch_failure_stops_without_committing_error_or_next_item() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let mut processor = Processor::new(events.clone()); processor.fail_sequence = Some(1);
    let mut control = Control::default();
    let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs, processor, &mut control).unwrap();
    assert_eq!(err(run.run(&mut control)), JobRunError::Processor(BatchCode::Execution.into()));
    assert_eq!(run.progress().committed, 0); assert_eq!(run.progress().attempts, 1);
    assert_eq!(*events.prepared.borrow(), vec!["alpha"]); assert!(run.is_stopped());
}
#[test]
fn serialized_failure_drops_guard_but_never_authorizes_result() {
    let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
    let mut processor = Processor::new(events.clone()); processor.fail_serialization = true;
    let mut control = Control::default();
    let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs, processor, &mut control).unwrap();
    assert_eq!(err(run.step(&mut control)), JobRunError::Storage(JobError::Serialization));
    assert!(!events.held.get()); assert_eq!(run.progress().committed, 0); assert!(run.is_stopped());
}
#[test]
fn work_contract_and_known_output_bound_refuse_before_durable_debit() {
    for mismatched_work in [true, false] {
        let root = Root::new(); let inputs = inputs(); let events = Rc::new(Events::default());
        let mut processor = Processor::new(events.clone()); processor.bad_work = mismatched_work;
        let mut cap = limits(); if !mismatched_work { cap.max_result_bytes = 1; }
        let mut control = Control::default();
        let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), cap, &inputs, processor, &mut control).unwrap();
        assert_eq!(err(run.step(&mut control)), if mismatched_work { JobRunError::WorkContract } else { JobRunError::Storage(JobError::Limit) });
        assert_eq!(run.progress().attempts, 0); assert!(events.executed.borrow().is_empty());
    }
}
#[test]
fn all_six_lifetime_work_axes_remain_spent_across_reopen() {
    for axis in 0..6 {
        let root = Root::new(); let inputs = inputs(); let mut cap = limits();
        let w = work();
        match axis {
            0 => cap.max_work.model.forward_positions = w.model.forward_positions,
            1 => cap.max_work.model.projected_logits = w.model.projected_logits,
            2 => cap.max_work.model.attention_pairs = w.model.attention_pairs,
            3 => cap.max_work.model.projections.dot_products = w.model.projections.dot_products,
            4 => cap.max_work.model.projections.multiply_accumulates = w.model.projections.multiply_accumulates,
            _ => cap.max_work.mask_node_visits = w.mask_node_visits,
        }
        let events = Rc::new(Events::default()); let mut processor = Processor::new(events.clone()); processor.fail_sequence = Some(1);
        let mut control = Control::default();
        let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), cap, &inputs, processor, &mut control).unwrap();
        assert!(run.step(&mut control).is_err()); drop(run);
        let mut resumed = JobRunner::resume(&root.0, key(), JobId([7;16]), cap, &inputs,
            Processor::new(events.clone()), TailPolicy::Refuse, &mut control).unwrap();
        assert_eq!(err(resumed.step(&mut control)), JobRunError::Storage(JobError::WorkLimit));
        assert_eq!(*events.executed.borrow(), vec![1]); assert_eq!(resumed.progress().attempts, 1);
    }
}
#[test]
fn retry_count_is_also_nonrenewable() {
    let root = Root::new(); let inputs = inputs(); let mut cap = limits(); cap.max_attempts = 2;
    let mut control = Control::default();
    for attempt in 0..2 {
        let mut processor = Processor::new(Rc::new(Events::default())); processor.fail_sequence = Some(1);
        let mut run = if attempt == 0 {
            JobRunner::create(&root.0, key(), JobId([7;16]), cap, &inputs, processor, &mut control).unwrap()
        } else {
            JobRunner::resume(&root.0, key(), JobId([7;16]), cap, &inputs, processor, TailPolicy::Refuse, &mut control).unwrap()
        };
        assert!(run.step(&mut control).is_err());
    }
    let events = Rc::new(Events::default());
    let mut run = JobRunner::resume(&root.0, key(), JobId([7;16]), cap, &inputs,
        Processor::new(events.clone()), TailPolicy::Refuse, &mut control).unwrap();
    assert_eq!(err(run.step(&mut control)), JobRunError::Storage(JobError::WorkLimit));
    assert!(events.executed.borrow().is_empty());
}
#[test]
fn malformed_later_envelopes_refuse_before_files_or_any_planning() {
    for bytes in [br#"{"id":"beta","id":"beta","text":"x"}"#.as_slice(),
        br#"{"id":"other","text":"x"}"#, br#"{"flush":true}"#, br#"{"id":"beta","text":42}"#,
        br#"{"id":"beta","text":"x","unexpected":true}"#] {
        let root = Root::new(); let mut inputs = inputs();
        inputs[1] = JobInput { id: "beta", original: bytes, normalized: bytes };
        let events = Rc::new(Events::default());
        assert_eq!(err(JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs,
            Processor::new(events.clone()), &mut Control::default())), JobRunError::InvalidEnvelope);
        assert!(events.prepared.borrow().is_empty()); assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }
}
#[test]
fn resume_rejects_recipe_identity_limits_and_even_committed_input_changes_before_planning() {
    let root = Root::new(); let inputs = inputs(); let mut control = Control::default();
    let mut run = JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(Rc::new(Events::default())), &mut control).unwrap();
    run.step(&mut control).unwrap(); drop(run);
    for field in [MismatchField::Recipe, MismatchField::Execution, MismatchField::Limits, MismatchField::Population] {
        let events = Rc::new(Events::default()); let mut processor = Processor::new(events.clone());
        let mut cap = limits();
        let changed = br#"{"id":"alpha","text":"changed private input","task_args":{"v":1}}"#;
        let a = if field == MismatchField::Population { changed.as_slice() } else { inputs[0].original };
        let candidate = [JobInput { id: "alpha", original: a, normalized: a },
            JobInput { id: "beta", original: inputs[1].original, normalized: inputs[1].normalized }];
        match field {
            MismatchField::Recipe => processor.recipe.seed += 1,
            MismatchField::Execution => processor.identity.sampler_version.push_str("-changed"),
            MismatchField::Limits => cap.max_attempts += 1,
            _ => {},
        }
        assert_eq!(err(JobRunner::resume(&root.0, key(), JobId([7;16]), cap, &candidate,
            processor, TailPolicy::DiscardUncommitted, &mut control)), JobRunError::Storage(JobError::Mismatch(field)));
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
    }
}
#[test]
fn nonidentity_envelope_profile_refuses_without_storage() {
    let root = Root::new(); let mut inputs = inputs(); inputs[1].normalized = b"changed";
    assert_eq!(err(JobRunner::create(&root.0, key(), JobId([7;16]), limits(), &inputs,
        Processor::new(Rc::new(Events::default())), &mut Control::default())), JobRunError::InvalidEnvelope);
    assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
}
