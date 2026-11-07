//! UNRUN lifetime and seal regression definitions, with no native executor.
use super::*;
use super::super::tests::{factory, document, Continue, SOURCE};
use crate::native_engine::portable_int8::ProjectionWork;

fn work(n: u64) -> DocumentWork {
    DocumentWork { model: Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }, mask_node_visits: n }
}
fn verification(n: usize) -> GroundingBudget { GroundingBudget { max_fields: n, max_matches: n, max_scan_steps: n as u64 } }
fn limits(n: u64) -> Int4SourceDocumentBatchLimits {
    Int4SourceDocumentBatchLimits { max_model_work: work(n).model, max_mask_visits: n,
        max_verification: verification(n as usize), max_summary_scan_steps: n, max_keyphrase_scan_work: n }
}
fn context(sequence: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }
}
#[test]
fn flush_epochs_do_not_renew_any_of_the_eleven_lifetime_axes() {
    for axis in 0..11 {
        let mut cap = limits(2);
        match axis {
            0 => cap.max_model_work.forward_positions = 1,
            1 => cap.max_model_work.projected_logits = 1,
            2 => cap.max_model_work.attention_pairs = 1,
            3 => cap.max_model_work.projections.dot_products = 1,
            4 => cap.max_model_work.projections.multiply_accumulates = 1,
            5 => cap.max_mask_visits = 1,
            6 => cap.max_verification.max_fields = 1,
            7 => cap.max_verification.max_matches = 1,
            8 => cap.max_verification.max_scan_steps = 1,
            9 => cap.max_summary_scan_steps = 1,
            _ => cap.max_keyphrase_scan_work = 1,
        }
        let mut ledger = Ledger::default();
        ledger.begin(work(1), verification(1), 1, 1, cap, context(1, 1)).unwrap();
        ledger.failed = false; // successful first attempt, then transport flush
        let error = ledger.begin(work(1), verification(1), 1, 1, cap, context(3, 2)).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::WorkLimit, "axis {axis}");
        assert!(error.stop);
        assert_eq!(ledger.work, work(1).model);
        assert_eq!(ledger.masks, 1);
        assert_eq!(ledger.verification, verification(1));
        assert_eq!((ledger.summary_scans, ledger.keyphrase_scans, ledger.sequence), (1, 1, 1));
        assert!(ledger.ready().is_err());
    }
}
#[test]
fn all_successful_counters_advance_atomically_and_failed_attempts_stay_charged() {
    let mut ledger = Ledger::default();
    ledger.begin(work(1), verification(1), 1, 1, limits(4), context(1, 1)).unwrap();
    // No successful completion: admission/cancellation/native/output refusal
    // keeps this full attempted reservation and permanently poisons the run.
    assert!(ledger.begin(work(1), verification(1), 1, 1, limits(4), context(2, 1)).is_err());
    assert_eq!(ledger.work, work(1).model); assert_eq!(ledger.masks, 1);
    assert_eq!(ledger.verification, verification(1));
    assert_eq!((ledger.summary_scans, ledger.keyphrase_scans), (1, 1));
    ledger.failed = false;
    ledger.begin(work(1), verification(1), 1, 1, limits(4), context(3, 2)).unwrap();
    assert_eq!(ledger.work, work(2).model); assert_eq!(ledger.masks, 2);
    assert_eq!(ledger.verification, verification(2));
    assert_eq!((ledger.summary_scans, ledger.keyphrase_scans, ledger.sequence), (2, 2, 3));
}
#[test]
fn overflow_and_invalid_request_order_fail_without_partial_debit() {
    for bad in [context(0, 1), context(1, 0), BatchRequestContext { input_line: 0, ..context(1, 1) }] {
        let mut ledger = Ledger::default();
        assert_eq!(ledger.begin(work(1), verification(1), 1, 1, limits(4), bad).unwrap_err().fault.code, BatchCode::InvalidExecution);
        assert_eq!(ledger.work, Int4Work::default()); assert_eq!(ledger.sequence, 0);
    }
    let mut ledger = Ledger::default();
    ledger.begin(work(u64::MAX), verification(1), 1, 1, limits(u64::MAX), context(1, 1)).unwrap();
    ledger.failed = false;
    assert_eq!(ledger.begin(work(1), verification(1), 1, 1, limits(u64::MAX), context(2, 1)).unwrap_err().fault.code, BatchCode::WorkLimit);
    assert_eq!(ledger.work, work(u64::MAX).model); assert_eq!(ledger.sequence, 1);
}
#[test]
fn reconstruction_checks_the_final_identity_full_work_source_and_factory_schedule() {
    let first = factory(5); let second = factory(5);
    let mut prepared = first.prepare_with_control(document(SOURCE), &mut Continue).unwrap();
    first.owns(&prepared).unwrap(); assert!(second.owns(&prepared).is_err());
    let config = first.configuration(); let plan = config.plan(SOURCE, &mut Continue).unwrap();
    plan.compare_receipt(prepared.expected, prepared.work, &prepared.identities, &mut Continue).unwrap();
    prepared.identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"changed final chunk");
    assert!(plan.compare_receipt(prepared.expected, prepared.work, &prepared.identities, &mut Continue).is_err());
    drop(plan);
    let prepared = first.prepare_with_control(document(SOURCE), &mut Continue).unwrap();
    let changed = first.configuration().plan("another original source", &mut Continue).unwrap();
    assert!(changed.compare_receipt(prepared.expected, prepared.work, &prepared.identities, &mut Continue).is_err());
    drop(changed);
    let mut prepared = first.prepare_with_control(document(SOURCE), &mut Continue).unwrap();
    prepared.work.model.projections.dot_products -= 1;
    let plan = first.configuration().plan(SOURCE, &mut Continue).unwrap();
    assert!(plan.compare_receipt(prepared.expected, prepared.work, &prepared.identities, &mut Continue).is_err());
    drop(plan);
    let changed = first.with_chunk_cohorts().unwrap();
    assert!(changed.owns(&prepared).is_err());
}
