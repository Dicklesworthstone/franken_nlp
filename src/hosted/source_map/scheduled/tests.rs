//! Focused admission and stop propagation; real-model parity is an explicit gate.
use super::*;
use crate::native_engine::decode::DecodeCancellationKind;
use crate::tasks::source_planning::quantized::long::{
    question::{SourceQuestion, Int8QuestionRun, synthesis::{
        SourceQuestionSynthesis, Int8QuestionSynthesisRun, hierarchy::Int8QuestionHierarchyRun}},
    summary::synthesis::{SourceSummarySynthesis, Int8SummarySynthesisRun, hierarchy::Int8SummaryHierarchyRun},
};

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65536 } }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn scheduling_retains_one_native_context_and_adds_coordinator_and_worker_capacity() {
    let serial = quote(native(), None, None).unwrap();
    assert_eq!(serial.scratch_bytes, serial.native.rope_bytes + serial.native.scratch_payload_bound
        + native().allocator_reserve_bytes);
    assert_eq!(serial.worker_bytes, 0); assert_eq!(serial.rows, 1);
    for rows in [1, 3, 64] {
        let schedule = prefill(rows);
        let packed = quote(native(), Some(schedule), None).unwrap();
        assert_eq!(packed.native, serial.native);
        assert_eq!(packed.scratch_bytes, serial.scratch_bytes + schedule.validate().unwrap());
        assert_eq!(packed.worker_bytes, 0);
        for width in [2, 7, 64] {
            let combined = quote(native(), Some(schedule), Some(width)).unwrap();
            assert_eq!(combined.native, serial.native);
            assert_eq!(combined.scratch_bytes, packed.scratch_bytes);
            assert_eq!(combined.rows, rows);
            assert_eq!(combined.worker_bytes, dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert!(combined.scratch_bytes + combined.worker_bytes > serial.scratch_bytes);
        }
    }
}

#[test]
fn invalid_width_prefill_capacity_and_overflow_refuse_before_owning_source_or_native_work() {
    for width in [0, 1, 65, usize::MAX] {
        assert!(quote(native(), Some(prefill(3)), Some(width)).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        let invalid = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
        assert!(quote(native(), Some(invalid), None).is_err());
        assert!(quote(native(), Some(invalid), Some(2)).is_err());
    }
    let mut short = prefill(3); short.max_extra_scratch_bytes -= 1;
    assert!(quote(native(), Some(short), None).is_err());
    assert!(quote(native(), Some(short), Some(2)).is_err());
    let mut overflow = native(); overflow.allocator_reserve_bytes = u64::MAX;
    assert!(quote(overflow, None, None).is_err());
    assert!(quote(overflow, Some(prefill(3)), Some(2)).is_err());
}

#[test]
fn every_original_cancellation_cause_survives_the_shared_document_boundary() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = poll(&mut Stop(cause)).err().unwrap();
        assert!(matches!(error, HostedError::Source(Int8SourceError::Cancelled(kind)) if kind == cause));
    }
}

#[test]
fn complete_source_and_question_owners_and_guarded_results_cross_the_physical_handoff() {
    fn send<T: Send + 'static>() {}
    send::<Owner<SourceQuestion>>();
    send::<Owner<SourceQuestionSynthesis>>();
    send::<Owner<SourceSummarySynthesis>>();
    send::<HostedInt8TeamResult<Int8QuestionRun>>();
    send::<HostedInt8TeamResult<Int8QuestionSynthesisRun>>();
    send::<HostedInt8TeamResult<Int8QuestionHierarchyRun>>();
    send::<HostedInt8TeamResult<Int8SummarySynthesisRun>>();
    send::<HostedInt8TeamResult<Int8SummaryHierarchyRun>>();
}

mod model;
