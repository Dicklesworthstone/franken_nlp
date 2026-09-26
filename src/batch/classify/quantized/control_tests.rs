//! Pinned planning and cancellation/guard mechanics, not model-success fixtures.
use super::*;
use crate::{tasks::classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy},
    tokenizer::pinned_controls};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

struct StopAfter(usize);
impl DecodeStepControl for StopAfter {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.0 == 0 { Some(DecodeCancellationKind::Deadline) } else { self.0 -= 1; None }
    }
}
fn planner() -> ClassificationPlanner {
    ClassificationPlanner::pinned(pinned_controls::pinned().unwrap().template_controls(), 166101).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 100000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn compiler(p: &ClassificationPlanner) -> Int8ClassificationBatchPlanner<'_> {
    let d = Sha256Digest::of_bytes(b"classification-corpus-control-fixture");
    let id = ExecutionIdentity {
        schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: "classify-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    Int8ClassificationBatchPlanner::new(p, id, budget(), ClassificationLimits::default(), Some(ClassificationBatchArgs {
        labels: vec![ClassificationLabel { id: "yes".to_owned(), description: "positive".to_owned() },
            ClassificationLabel { id: "no".to_owned(), description: "negative".to_owned() }],
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: budget(),
    })).unwrap()
}
fn doc() -> BatchDocument<ClassificationBatchArgs> {
    BatchDocument { id: "record-1".to_owned(), text: "Great café".to_owned(), task_args: None }
}
#[test]
fn preexisting_cancellation_wins_before_cloning_or_compiling_a_request() {
    let p = planner(); let c = compiler(&p);
    let e = c.prepare_with_control(doc(), &mut StopAfter(0)).err().unwrap();
    assert!(e.stop); assert_eq!(e.fault.cancellation, Some(DecodeCancellationKind::Deadline));
}
#[test]
fn one_shared_preparation_quota_reaches_inner_head_compilation() {
    let p = planner(); let c = compiler(&p);
    for calls in [1, 2, 3] {
        let e = c.prepare_with_control(doc(), &mut StopAfter(calls)).err().unwrap();
        assert!(e.stop); assert_eq!(e.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    }
    let a = c.prepare(doc()).unwrap(); let b = c.prepare_with_control(doc(), &mut Continue).unwrap();
    assert_eq!(a.execution_identity(), b.execution_identity());
    assert_eq!(a.model_work(), b.model_work());
}
struct Guard(Arc<AtomicUsize>);
impl Drop for Guard { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }
struct Admission(Arc<AtomicUsize>);
impl Int8ClassificationAdmission for Admission {
    type Guard = Guard;
    fn admit(&mut self, id: &ExecutionIdentity, _: Int8Work) -> Result<(ExecutionIdentity, Guard), BatchItemFailure> {
        Ok((id.clone(), Guard(Arc::clone(&self.0))))
    }
}
#[test]
fn cancellation_after_callback_completion_discards_value_and_releases_guard() {
    let p = planner(); let c = compiler(&p); let prepared = c.prepare(doc()).unwrap();
    let drops = Arc::new(AtomicUsize::new(0)); let mut admission = Admission(Arc::clone(&drops));
    let mut called = false;
    let error = with_admission(&mut admission, &prepared.plan, &mut StopAfter(2), |_, _| {
        called = true; Ok(())
    }).err().unwrap();
    assert!(called); assert!(error.stop);
    assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
#[test]
fn successful_callback_keeps_guard_until_the_delivery_owner_drops() {
    let p = planner(); let c = compiler(&p); let prepared = c.prepare(doc()).unwrap();
    let drops = Arc::new(AtomicUsize::new(0)); let mut admission = Admission(Arc::clone(&drops));
    let output = with_admission(&mut admission, &prepared.plan, &mut Continue, |_, _| Ok(())).unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(output); assert_eq!(drops.load(Ordering::SeqCst), 1);
}
