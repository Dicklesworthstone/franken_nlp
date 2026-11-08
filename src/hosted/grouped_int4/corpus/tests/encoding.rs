//! Checked structured-stream workspace arithmetic; no allocation or inference.
use super::*;
use crate::{
    batch::extract::grouped_int4::memory::Int4ExtractionCohortMemoryLimits,
    native_engine::strict_int4::prefill::Int4PrefillLimits,
    tokenizer::untrusted::SafeBpeLimits,
};

fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })
}
fn cohort(documents: usize) -> Int4ExtractionCohortMemoryLimits {
    Int4ExtractionCohortMemoryLimits { max_documents: documents, context_tokens: 32,
        preparation_reserve_bytes: 10_000, io_reserve_bytes: 4096, allocator_reserve_bytes: 8192 }
}
fn transport() -> BatchLimits {
    BatchLimits { max_line_bytes: 1024, max_document_bytes: 512, max_id_bytes: 16,
        max_epoch_ids: 17, max_epoch_id_bytes: 3072, max_output_line_bytes: 8192,
        ..BatchLimits::default() }
}
fn packed() -> Int4PrefillLimits {
    Int4PrefillLimits { max_batch_rows: 4,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() }
}

#[test]
fn serial_and_persistent_teams_charge_one_encoder_without_replacing_native_or_input_owners() {
    let encoding = policy();
    let UntrustedEncoding::SafeBpeV1(encoding_limits) = encoding else { unreachable!() };
    let expected = encoding_limits.requirements().unwrap().workspace_bytes;
    let encoding = UntrustedEncoding::SafeBpeV1(encoding_limits);
    assert_eq!(encoding_workspace_bytes(&UntrustedEncoding::ByteFallback).unwrap(), 0);
    assert_eq!(encoding_workspace_bytes(&encoding).unwrap(), expected);
    for group_width in [1, 32, 127, MAX_MODEL_K] {
        let byte = requirements(group_width, native_work(group_width), limits()).unwrap();
        let explicit_byte = requirements_with_document_encoding(group_width, native_work(group_width),
            limits(), &UntrustedEncoding::ByteFallback).unwrap();
        assert_eq!(byte, explicit_byte);
        let compact = requirements_with_document_encoding(group_width, native_work(group_width), limits(), &encoding).unwrap();
        assert_eq!(compact.native, byte.native);
        assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
        assert_eq!(compact.input_bytes, byte.input_bytes + expected);
        compact.check_team_workspace(None, group_width).unwrap();
        for threads in [2, 4, 64] {
            compact.check_team_workspace(Some(threads), group_width).unwrap();
            let workers = int4_task_team_workspace_bytes(threads, group_width).unwrap();
            assert!(workers > 0);
            assert_eq!(compact.input_bytes + compact.workspace_bytes + workers,
                byte.input_bytes + byte.workspace_bytes + workers + expected);
        }
    }
}

#[test]
fn cohort_lookahead_and_packed_or_selected_execution_keep_one_transient_encoder() {
    let encoding = policy(); let extra = encoding_workspace_bytes(&encoding).unwrap();
    for documents in [1, 3, 8] {
        for selected in [None, Some(Int4JsonSparseLimits { max_rows_per_step: 7 }),
            Some(Int4JsonSparseLimits { max_rows_per_step: NANBEIGE_VOCAB_SIZE })] {
            for prefill in [None, Some(packed())] {
                let rows = selected.map_or(NANBEIGE_VOCAB_SIZE, |policy| policy.max_rows_per_step);
                let mut work = native_work(32);
                work.max_model_work = Int4Work::for_sequence(32, 0, 256, 70 * rows).unwrap();
                let base = cohort(documents).requirements_for_execution(32, work, transport(), selected, prefill).unwrap();
                let compact = cohort(documents).requirements_for_execution_with_document_encoding(
                    32, work, transport(), selected, prefill, &encoding).unwrap();
                let byte = cohort(documents).requirements_for_execution_with_document_encoding(
                    32, work, transport(), selected, prefill, &UntrustedEncoding::ByteFallback).unwrap();
                assert_eq!(byte, base);
                assert_eq!(compact.native, base.native);
                assert_eq!(compact.workspace_bytes, base.workspace_bytes);
                assert_eq!(compact.input_bytes, base.input_bytes + extra);
                assert_eq!(compact.output_reservation_bytes, base.output_reservation_bytes);
                assert_eq!(compact.decode_workspace_bytes, base.decode_workspace_bytes);
                assert_eq!(compact.max_result_bytes, base.max_result_bytes);
                let mut retained = cohort(documents); retained.preparation_reserve_bytes += 1;
                let grown = retained.requirements_for_execution_with_document_encoding(
                    32, work, transport(), selected, prefill, &encoding).unwrap();
                assert_eq!(grown.input_bytes, compact.input_bytes + documents as u64 + 1);
                assert_eq!(grown.workspace_bytes, compact.workspace_bytes);
            }
        }
    }
}

#[test]
fn encoding_addition_is_checked_after_every_other_simultaneously_owned_input_byte() {
    let encoding = policy(); let extra = encoding_workspace_bytes(&encoding).unwrap();
    let mut serial = limits();
    let base = requirements(32, native_work(32), serial).unwrap();
    serial.preparation_reserve_bytes = u64::MAX - (base.input_bytes - serial.preparation_reserve_bytes) - extra;
    let exact = requirements_with_document_encoding(32, native_work(32), serial, &encoding).unwrap();
    assert_eq!(exact.input_bytes, u64::MAX);
    serial.preparation_reserve_bytes += 1;
    assert!(requirements(32, native_work(32), serial).is_ok());
    assert!(requirements_with_document_encoding(32, native_work(32), serial, &encoding).is_err());

    let mut cohort = cohort(3);
    let base = cohort.requirements(32, native_work(32), transport()).unwrap();
    cohort.io_reserve_bytes = u64::MAX - (base.input_bytes - cohort.io_reserve_bytes) - extra;
    let exact = cohort.requirements_for_execution_with_document_encoding(
        32, native_work(32), transport(), None, None, &encoding).unwrap();
    assert_eq!(exact.input_bytes, u64::MAX);
    cohort.io_reserve_bytes += 1;
    assert!(cohort.requirements(32, native_work(32), transport()).is_ok());
    assert!(cohort.requirements_for_execution_with_document_encoding(
        32, native_work(32), transport(), None, None, &encoding).is_err());
}

#[test]
fn every_invalid_encoder_axis_refuses_both_serial_and_cohort_quotes() {
    let UntrustedEncoding::SafeBpeV1(valid) = policy() else { unreachable!() };
    for axis in 0..8 {
        let mut changed = valid;
        match axis {
            0 => changed.max_input_bytes = 0,
            1 => changed.max_input_bytes = SafeBpeLimits::MAX_INPUT_BYTES + 1,
            2 => changed.max_output_tokens = 0,
            3 => changed.max_output_tokens = SafeBpeLimits::MAX_OUTPUT_TOKENS + 1,
            4 => changed.max_pair_checks = 0,
            5 => changed.max_pair_checks = SafeBpeLimits::MAX_PAIR_CHECKS + 1,
            6 => changed.max_pair_bytes = 0,
            _ => changed.max_pair_bytes = SafeBpeLimits::MAX_PAIR_BYTES + 1,
        }
        let encoding = UntrustedEncoding::SafeBpeV1(changed);
        assert_eq!(encoding_workspace_bytes(&encoding).unwrap_err().code, BatchCode::InvalidLimits);
        assert!(requirements_with_document_encoding(32, native_work(32), limits(), &encoding).is_err());
        assert!(cohort(3).requirements_for_execution_with_document_encoding(
            32, native_work(32), transport(), None, None, &encoding).is_err());
    }
}
