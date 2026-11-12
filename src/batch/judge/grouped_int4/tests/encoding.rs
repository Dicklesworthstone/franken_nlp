//! Pinned planning and real corpus preflight using a tiny model identity.
//! Admission always refuses before any native engine is created: these tests
//! establish tokenizer/resource behavior, not numerical or quality evidence.
use super::*;
use std::{cell::Cell, rc::Rc};
use crate::{native_engine::strict_int4::weights::projection_fixture,
    tokenizer::untrusted::SafeBpeLimits};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct RefuseAdmission(Rc<Cell<usize>>);
impl Int4FaithfulnessDocumentBatchAdmission for RefuseAdmission {
    type Guard = ();
    fn admit(&mut self, request: Int4FaithfulnessDocumentAdmission<'_>)
        -> Result<(Vec<ExecutionIdentity>, Self::Guard), BatchItemFailure> {
        self.0.set(self.0.get() + 1);
        assert!(request.identities.len() > 1);
        assert!(request.head_count >= request.identities.len());
        assert_eq!(request.cohort_chunks, Some(2));
        Err(BatchItemFailure::fatal(BatchCode::Admission))
    }
}
fn encoding_limits() -> SafeBpeLimits {
    SafeBpeLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
        max_pair_checks: 1 << 26, max_pair_bytes: 1 << 29 }
}
fn source() -> BatchDocument<Int4FaithfulnessDocumentArgs> {
    BatchDocument { id: "complete".to_owned(),
        text: "Solar energy powers local homes. café 上海 𐐀\r\n<think> literal final tail. ".repeat(3),
        task_args: Some(Int4FaithfulnessDocumentArgs { claim: "Solar energy powers homes. 上海".to_owned() }) }
}
fn population_limits() -> Int4FaithfulnessMapLimits {
    let mut limits = mapping();
    limits.chunks.max_chunk_bytes = 32;
    limits.max_model_work = Int4Work::for_sequence(32, 0, 65_536, 512 * NANBEIGE_VOCAB_SIZE).unwrap();
    limits
}
fn evidence_policy() -> FaithfulnessPolicy {
    FaithfulnessPolicy { evidence_window_bytes: 16, max_evidence_windows: 4,
        max_evidence_spans: 4, ..policy() }
}
fn memory() -> Int4FaithfulnessMapMemoryLimits {
    Int4FaithfulnessMapMemoryLimits { context_tokens: 1024,
        preparation_reserve_bytes: 128 * 1024 * 1024, allocator_reserve_bytes: 1024 * 1024 }
}
fn batch(weights: &Int4ModelWeights, encoding: UntrustedEncoding, calls: Rc<Cell<usize>>)
    -> NativeInt4FaithfulnessDocumentBatch<'_, '_, RefuseAdmission> {
    let planner = Int4JudgePlanner::for_model(weights.identity(), task(), JudgeLimits::default()).unwrap()
        .with_document_encoding(encoding.clone()).unwrap();
    let limits = population_limits();
    let compiler = Int4FaithfulnessDocumentBatchPlanner::new(planner, evidence_policy(), limits,
        limits.max_model_work.checked_add(limits.max_model_work).unwrap()).unwrap()
        .with_max_heads(5).unwrap().with_chunk_cohorts(2).unwrap();
    assert_eq!(compiler.document_encoding(), &encoding);
    NativeInt4FaithfulnessDocumentBatch::new_cohorts(compiler, weights, PackedInt4Kernel::scalar(),
        memory(), RefuseAdmission(calls)).unwrap()
}
fn context() -> BatchRequestContext {
    BatchRequestContext { request_seq: 1, epoch: 1, input_line: 1, byte_offset: 0 }
}

#[test]
fn preparation_and_execution_share_every_tokenizer_axis_and_seal_only_the_pass_delta() {
    let (weights, _) = projection_fixture(32);
    let calls = Rc::new(Cell::new(0));
    let mut measure = batch(&weights, UntrustedEncoding::SafeBpeV1(encoding_limits()), Rc::clone(&calls));
    let prepared = measure.prepare_with_control(source(), &mut Continue).unwrap();
    let spent = prepared.encoding_work();
    assert!(prepared.chunk_count() > 1);
    assert!(spent.input_bytes() > source().text.len());
    assert!(spent.output_tokens() > 0 && spent.pair_checks() > 0 && spent.pair_bytes() > 0);
    let twice = spent.checked_add(spent).unwrap();
    let exact = SafeBpeLimits { max_input_bytes: twice.input_bytes(), max_output_tokens: twice.output_tokens(),
        max_pair_checks: twice.pair_checks(), max_pair_bytes: twice.pair_bytes() };
    exact.validate().unwrap();
    drop(prepared); drop(measure);
    for short_axis in [None, Some(0), Some(1), Some(2), Some(3)] {
        let mut limits = exact;
        if let Some(axis) = short_axis {
            match axis { 0 => limits.max_input_bytes -= 1, 1 => limits.max_output_tokens -= 1,
                2 => limits.max_pair_checks -= 1, _ => limits.max_pair_bytes -= 1 }
        }
        let calls = Rc::new(Cell::new(0));
        let mut processor = batch(&weights, UntrustedEncoding::SafeBpeV1(limits), Rc::clone(&calls));
        let before = processor.remaining_work();
        let prepared = processor.prepare_with_control(source(), &mut Continue).unwrap();
        assert_eq!(prepared.encoding_work(), spent);
        let native = prepared.planned_work();
        assert_eq!(processor.remaining_work(), before);
        let failure = processor.execute_with_context(prepared, context(), &mut Continue)
            .err().expect("admission or cumulative tokenizer work must refuse before native execution");
        assert!(failure.stop);
        assert_eq!(processor.remaining_work(), subtract(before, native).unwrap());
        assert!(processor.is_poisoned());
        if short_axis.is_some() {
            assert_eq!(failure.fault.code, BatchCode::Planning);
            assert_eq!(calls.get(), 0, "exhausted encoder cannot reach admission");
        } else {
            // Reaching admission proves reconstructed identities and preflight
            // metadata equal preparation, despite the enclosing ledger doubling.
            assert_eq!(failure.fault.code, BatchCode::Admission);
            assert_eq!(calls.get(), 1);
        }
        assert!(processor.prepare_with_control(source(), &mut Continue).is_err());
    }
}

#[test]
fn byte_population_keeps_zero_encoding_work_and_compact_backend_quotes_one_workspace() {
    let (weights, _) = projection_fixture(32);
    let mut byte = batch(&weights, UntrustedEncoding::ByteFallback, Rc::new(Cell::new(0)));
    let prepared = byte.prepare_with_control(source(), &mut Continue).unwrap();
    assert_eq!(prepared.encoding_work(), SafeBpeWork::default());
    let selected = UntrustedEncoding::SafeBpeV1(encoding_limits());
    let compact = batch(&weights, selected.clone(), Rc::new(Cell::new(0)));
    let baseline = memory().cohort_requirements(32, task(), JudgeLimits::default(),
        population_limits(), 5, 2).unwrap();
    let Backend::Cohorts { required, .. } = &compact.backend else { panic!("cohort backend required"); };
    let quote = baseline.memory.with_input_encoding(&selected).unwrap();
    assert_eq!(required.memory, quote);
    assert_eq!(required.memory.input_bytes - baseline.memory.input_bytes,
        encoding_limits().requirements().unwrap().workspace_bytes);
    assert_eq!(required.task_workspace_bytes, baseline.task_workspace_bytes);
    assert_eq!(required.max_cohort_output_bytes, baseline.max_cohort_output_bytes);
}

#[test]
fn original_planning_cancellation_stays_fatal_without_admission_or_native_debit() {
    struct Stop { calls: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.calls >= 6).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let (weights, _) = projection_fixture(32);
    let calls = Rc::new(Cell::new(0));
    let mut processor = batch(&weights, UntrustedEncoding::SafeBpeV1(encoding_limits()), Rc::clone(&calls));
    let before = processor.remaining_work();
    let failure = processor.prepare_with_control(source(), &mut Stop { calls: 0 }).err().unwrap();
    assert!(failure.stop);
    assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(processor.remaining_work(), before);
    assert_eq!(calls.get(), 0);
}
