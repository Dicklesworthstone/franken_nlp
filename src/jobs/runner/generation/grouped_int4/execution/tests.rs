//! UNRUN invocation-ledger arithmetic fixtures. Durable attempt persistence,
//! spool acknowledgement and recovery remain owned by the existing JobRunner.
use super::*;
use crate::native_engine::portable_int8::ProjectionWork;
fn work(n: u64) -> JobWork {
    JobWork::from_int4(Int4Work { forward_positions: n, projected_logits: n, attention_pairs: n,
        projections: ProjectionWork { dot_products: n, multiply_accumulates: n } }, 0)
}
fn limits() -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: 65_536, max_result_bytes: 65_536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 2, max_work: work(2) }
}
fn context(sequence: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: sequence, epoch: 1, input_line: sequence, byte_offset: 0 }
}
#[test]
fn failed_attempt_stays_charged_and_poisoned() {
    let mut ledger = Ledger::default();
    ledger.begin(work(1), context(7), limits()).unwrap();
    assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
    assert!(ledger.ready().is_err());
    assert!(ledger.begin(work(1), context(8), limits()).is_err());
    assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
}
#[test]
fn successful_early_finish_never_refunds_reserved_work_or_attempts() {
    let mut ledger = Ledger::default();
    for sequence in [7, 8] {
        ledger.begin(work(1), context(sequence), limits()).unwrap();
        ledger.failed = false; // Only fully validated native success clears it.
    }
    assert_eq!(ledger.work, work(2)); assert_eq!(ledger.attempts, 2);
    assert_eq!(ledger.begin(work(1), context(9), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
    assert!(ledger.ready().is_err());
}
#[test]
fn every_work_axis_and_overflow_refuse_before_model_admission() {
    for axis in 0..5 {
        let mut excess = work(1);
        match axis {
            0 => excess.model.forward_positions = 3, 1 => excess.model.projected_logits = 3,
            2 => excess.model.attention_pairs = 3, 3 => excess.model.projections.dot_products = 3,
            _ => excess.model.projections.multiply_accumulates = 3,
        }
        let mut ledger = Ledger::default();
        assert_eq!(ledger.begin(excess, context(1), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
        assert!(ledger.ready().is_err());
    }
    let mut ledger = Ledger { work: work(u64::MAX), ..Ledger::default() };
    let mut ceiling = limits(); ceiling.max_work = work(u64::MAX);
    assert_eq!(ledger.begin(work(1), context(1), ceiling).err().unwrap().fault.code, BatchCode::WorkLimit);
    assert!(ledger.ready().is_err());
}
#[test]
fn zero_or_repeated_delivery_coordinates_poison_before_admission() {
    for invalid in [context(0), BatchRequestContext { epoch: 0, ..context(1) },
        BatchRequestContext { input_line: 0, ..context(1) }] {
        let mut ledger = Ledger::default();
        assert!(ledger.begin(work(1), invalid, limits()).is_err()); assert!(ledger.ready().is_err());
    }
    let mut ledger = Ledger::default(); ledger.begin(work(1), context(7), limits()).unwrap();
    ledger.failed = false;
    assert!(ledger.begin(work(1), context(7), limits()).is_err()); assert!(ledger.ready().is_err());
}
