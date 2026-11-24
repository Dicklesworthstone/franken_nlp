//! Resource and cancellation contracts; numerical parity is a separate gate.
use super::*;
use crate::native_engine::decode::DecodeCancellationKind;

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn document_teams_price_all_coordinator_and_fixed_worker_buffers_together() {
    let limits = native(); let required = requirements(limits).unwrap();
    let ordinary = required.rope_bytes + required.scratch_payload_bound + limits.allocator_reserve_bytes;
    for width in [2, 7, 64] {
        let (scratch, rows) = workspace(limits, required, width, None).unwrap();
        assert_eq!(scratch, ordinary); assert_eq!(rows, 1);
        assert_eq!(dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap(),
            dispatch::int8_team_workspace_bytes(width).unwrap());
        for rows in [1, 3, 64] {
            let schedule = prefill(rows);
            let (scratch, capacity) = workspace(limits, required, width, Some(schedule)).unwrap();
            assert_eq!(capacity, rows);
            assert_eq!(scratch, ordinary + schedule.validate().unwrap());
            let workers = dispatch::int8_prefill_team_workspace_bytes(width, capacity).unwrap();
            assert!(workers >= dispatch::int8_team_workspace_bytes(width).unwrap());
            assert!(scratch + workers > ordinary);
            // KV is one resident context, not multiplied by chunks or workers.
            assert_eq!(requirements(limits).unwrap().kv_bytes, required.kv_bytes);
        }
    }
}

#[test]
fn invalid_width_underfunded_prompt_storage_and_checked_overflow_refuse() {
    let required = requirements(native()).unwrap();
    for width in [0, 1, 65, usize::MAX] {
        assert!(workspace(native(), required, width, Some(prefill(3))).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        assert!(workspace(native(), required, 2, Some(Int8PrefillLimits {
            max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX,
        })).is_err());
    }
    let mut short = prefill(3); short.max_extra_scratch_bytes -= 1;
    assert!(workspace(native(), required, 2, Some(short)).is_err());
    let mut overflow = native(); overflow.allocator_reserve_bytes = u64::MAX;
    assert!(workspace(overflow, required, 2, Some(prefill(3))).is_err());
}

#[test]
fn the_document_owner_and_results_can_survive_the_physical_completion_handoff() {
    fn send<T: Send + 'static>() {}
    send::<Admission>();
    send::<HostedInt8TeamResult<Int8SourceMapRun>>();
    send::<HostedInt8TeamResult<crate::tasks::source_planning::quantized::long::summary::Int8CorpusSummaryRun>>();
    send::<HostedInt8TeamResult<crate::tasks::source_planning::quantized::long::keyphrase_reduction::Int8CorpusKeyphraseRun>>();
}

#[test]
fn every_source_map_cancellation_cause_reaches_the_host_without_reclassification() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = poll(&mut Stop(cause)).err().unwrap();
        let HostedError::Source(error) = error else { panic!("wrong typed source boundary") };
        assert_eq!(error.cancellation(), Some(cause));
    }
}

mod model;
