//! Invocation-ledger and whole-window admission fixtures; no native execution.
use super::*;
use crate::native_engine::{portable_int8::ProjectionWork, strict_int4::Int4MemoryBudget};

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
fn begun_failed_document_keeps_all_work_and_attempt_charge_and_poison() {
    let mut ledger = Ledger::default(); ledger.begin(work(1), context(7), limits()).unwrap();
    assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
    assert!(ledger.ready().is_err()); assert!(ledger.begin(work(1), context(8), limits()).is_err());
    assert_eq!(ledger.work, work(1)); assert_eq!(ledger.attempts, 1);
}
#[test]
fn successful_documents_never_replenish_lifetime_work_or_attempts() {
    let mut ledger = Ledger::default();
    for sequence in [7, 8] {
        ledger.begin(work(1), context(sequence), limits()).unwrap();
        ledger.failed = false; // Only completed native maps clear this state.
    }
    assert_eq!(ledger.work, work(2)); assert_eq!(ledger.attempts, 2);
    assert_eq!(ledger.begin(work(1), context(9), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
    assert!(ledger.ready().is_err());
}
#[test]
fn duplicate_invalid_delivery_and_mask_work_poison_before_admission() {
    for invalid in [context(0), BatchRequestContext { epoch: 0, ..context(1) },
        BatchRequestContext { input_line: 0, ..context(1) }] {
        let mut ledger = Ledger::default();
        assert!(ledger.begin(work(1), invalid, limits()).is_err()); assert!(ledger.ready().is_err());
    }
    let mut ledger = Ledger::default(); ledger.begin(work(1), context(7), limits()).unwrap();
    ledger.failed = false;
    assert!(ledger.begin(work(1), context(7), limits()).is_err()); assert!(ledger.ready().is_err());
    let mut masks = work(1); masks.mask_node_visits = 1;
    let mut ledger = Ledger::default();
    assert!(ledger.begin(masks, context(1), limits()).is_err()); assert!(ledger.ready().is_err());
}
#[test]
fn each_whole_document_work_axis_and_checked_addition_fail_closed() {
    for field in 0..5 {
        let mut excess = work(1);
        match field { 0 => excess.model.forward_positions = 3, 1 => excess.model.projected_logits = 3,
            2 => excess.model.attention_pairs = 3, 3 => excess.model.projections.dot_products = 3,
            _ => excess.model.projections.multiply_accumulates = 3 }
        let mut ledger = Ledger::default();
        assert_eq!(ledger.begin(excess, context(1), limits()).err().unwrap().fault.code, BatchCode::WorkLimit);
        assert!(ledger.ready().is_err());
    }
    let mut ledger = Ledger { work: work(u64::MAX), ..Ledger::default() };
    let mut maximum = limits(); maximum.max_work = work(u64::MAX);
    assert_eq!(ledger.begin(work(1), context(1), maximum).err().unwrap().fault.code, BatchCode::WorkLimit);
    assert!(ledger.ready().is_err());
}
#[test]
fn cohort_native_workspace_and_result_cap_are_admitted_together() {
    let native = Int4MemoryRequirement { kv_bytes: 500, rope_bytes: 600, scratch_payload_bound: 700 };
    let required = Int4ClassificationMapCohortMemoryRequirement {
        memory: Int4ClassificationMapMemoryRequirement { native, workspace_bytes: 1000,
            input_bytes: 2000, output_reservation_bytes: 3000, max_output_bytes: 4000 },
        task_workspace_bytes: 800, max_cohort_output_bytes: 900 };
    let budget = Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(work(1).int4_model_work()),
        memory: Int4MemoryBudget { max_kv_bytes: native.kv_bytes, max_rope_bytes: native.rope_bytes,
            max_scratch_payload_bytes: native.scratch_payload_bound },
        max_task_workspace_bytes: required.task_workspace_bytes, max_output_bytes: required.max_cohort_output_bytes };
    check_cohort_budget(required, work(1).int4_model_work(), budget).unwrap();
    for field in 0..10 {
        let mut under = budget;
        match field { 0 => under.memory.max_kv_bytes -= 1, 1 => under.memory.max_rope_bytes -= 1,
            2 => under.memory.max_scratch_payload_bytes -= 1, 3 => under.max_task_workspace_bytes -= 1,
            4 => under.max_output_bytes -= 1, 5 => under.native.max_forward_positions -= 1,
            6 => under.native.max_attention_pairs -= 1, 7 => under.native.max_projection_work.dot_products -= 1,
            8 => under.native.max_projection_work.multiply_accumulates -= 1, _ => under.max_output_bytes += 1 }
        let error = check_cohort_budget(required, work(1).int4_model_work(), under).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission, "field {field}");
    }
}
