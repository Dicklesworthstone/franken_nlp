//! Admission and ownership contracts. Real-model comparison remains ignored.
use super::*;
use crate::native_engine::decode::DecodeCancellationKind;
use crate::tasks::redact::long::{LongRedactionConfig, LongRedactionRun};

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
fn both_detector_stages_share_one_kv_while_prompt_and_worker_storage_coexist() {
    let limits = native();
    let required = requirements(limits).unwrap();
    let ordinary = required.rope_bytes + required.scratch_payload_bound + limits.allocator_reserve_bytes;
    assert_eq!(scratch_bytes(limits, required, None).unwrap(), ordinary);
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
            assert_eq!(scratch_bytes(limits, required, Some(schedule)).unwrap(), scratch,
                "serial prefill and team prefill retain the same coordinator workspace");
            let children = dispatch::int8_prefill_team_workspace_bytes(width, capacity).unwrap();
            assert!(children >= dispatch::int8_team_workspace_bytes(width).unwrap());
            assert!(scratch + children > ordinary);
            assert_eq!(requirements(limits).unwrap().kv_bytes, required.kv_bytes,
                "passes, chunks and workers never multiply resident KV");
        }
    }
}

#[test]
fn incomplete_schedule_allowances_and_overflow_refuse_before_team_work() {
    let required = requirements(native()).unwrap();
    for width in [0, 1, 65, usize::MAX] {
        assert!(workspace(native(), required, width, Some(prefill(3))).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        let invalid = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
        assert!(scratch_bytes(native(), required, Some(invalid)).is_err());
        assert!(workspace(native(), required, 2, Some(invalid)).is_err());
    }
    let mut short = prefill(3); short.max_extra_scratch_bytes -= 1;
    assert!(scratch_bytes(native(), required, Some(short)).is_err());
    assert!(workspace(native(), required, 2, Some(short)).is_err());
    let mut overflow = native(); overflow.allocator_reserve_bytes = u64::MAX;
    assert!(workspace(overflow, required, 2, Some(prefill(3))).is_err());
}

#[test]
fn complete_short_and_document_owners_cross_physical_completion_without_borrowed_secrets() {
    fn send<T: Send + 'static>() {}
    send::<Admission<Int8RedactionConfig>>();
    send::<Admission<LongRedactionConfig>>();
    send::<HostedInt8TeamResult<Int8RedactionRun>>();
    send::<HostedInt8TeamResult<LongRedactionRun>>();
}

#[test]
fn every_terminal_cause_preserves_the_exact_native_cancellation_kind() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        assert!(matches!(poll(&mut Stop(cause)),
            Err(HostedError::Native(StrictInt8Error::Cancelled(actual))) if actual == cause));
    }
}

#[test]
fn schedule_geometry_neither_replaces_nor_spends_the_compact_document_operation() {
    use crate::tokenizer::{embedded::EmbeddedTokenizer, pinned_controls,
        untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding}};
    let planner = SourceTaskPlanner::pinned(pinned_controls::pinned().unwrap().template_controls(),
        EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap()).unwrap()
        .with_map_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 256, max_output_tokens: 256, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }), SafeBpeWorkLimits { max_input_bytes: 4096, max_output_tokens: 4096,
            max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }).unwrap();
    planner.source_encoder().encode_with_control("Alice 上海", 256, 256, &mut || None).unwrap();
    let spent = planner.map_document_encoding_work().unwrap();
    let encoder_bytes = planner.map_document_encoding_workspace_bytes().unwrap();
    for schedule in [None, Some(prefill(3))] {
        workspace(native(), requirements(native()).unwrap(), 3, schedule).unwrap();
        check_encoding_preparation(&planner, encoder_bytes).unwrap();
        assert!(check_encoding_preparation(&planner, encoder_bytes - 1).is_err());
        assert_eq!(planner.map_document_encoding_work().unwrap(), spent);
    }
}

mod model;
