use super::*;
use crate::native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work};

fn config() -> Int4ScoredCohortMemoryLimits {
    Int4ScoredCohortMemoryLimits { max_documents: 3, max_heads: 6, context_tokens: 48,
        preparation_reserve_bytes: 8 << 20, io_reserve_bytes: 65_536,
        allocator_reserve_bytes: 1 << 20 }
}
fn work(width: usize) -> Int4ScoredBatchLimits {
    let item = Int4Work::for_sequence(width, 0, 32, NANBEIGE_VOCAB_SIZE).unwrap();
    Int4ScoredBatchLimits { max_item_work: item,
        max_model_work: item.checked_add(item).unwrap().checked_add(item).unwrap() }
}

#[test]
fn native_peak_prices_every_possible_head_and_shared_scratch_once() {
    for width in [1, 3, 16, 32, 127, 128, 10752] {
        let cfg = config();
        let required = cfg.requirements(width, work(width), BatchLimits::default()).unwrap();
        let native = Int4MemoryRequirement::for_cohort(&[48; 6], width).unwrap();
        assert_eq!(required.native, native);
        assert_eq!(required.native.kv_bytes,
            6 * Int4MemoryRequirement::for_context(48, width).unwrap().kv_bytes);
        assert_eq!(required.workspace_bytes,
            native.rope_bytes + native.scratch_payload_bound + cfg.allocator_reserve_bytes);
    }
}

#[test]
fn one_lookahead_is_charged_even_when_the_window_is_full() {
    let cfg = config();
    let before = cfg.requirements(32, work(32), BatchLimits::default()).unwrap();
    let mut larger = cfg;
    larger.preparation_reserve_bytes += 8192;
    larger.io_reserve_bytes += 4096;
    let after = larger.requirements(32, work(32), BatchLimits::default()).unwrap();
    assert_eq!(after.input_bytes - before.input_bytes,
        (cfg.max_documents as u64 + 1) * 8192 + 4096);
    assert_eq!(after.native, before.native);
    assert_eq!(after.workspace_bytes, before.workspace_bytes);
    assert_eq!(after.output_reservation_bytes, before.output_reservation_bytes);
}

#[test]
fn retained_results_and_each_guard_survive_all_delivery() {
    let cfg = config(); let transport = BatchLimits::default();
    let required = cfg.requirements(3, work(3), transport).unwrap();
    assert_eq!(required.max_output_bytes,
        cfg.max_documents as u64 * transport.max_output_line_bytes as u64);
    assert_eq!(required.output_reservation_bytes,
        4 * required.max_output_bytes + cfg.max_documents as u64 * 4096);
    assert_eq!(required.task_workspace_bytes,
        required_task_workspace_bytes(work(3).max_model_work, cfg.max_heads,
            required.max_output_bytes).unwrap());
    let expected_input = (cfg.max_documents as u64 + 1)
        * (8 * transport.max_line_bytes as u64 + cfg.preparation_reserve_bytes)
        + 4 * transport.max_output_line_bytes as u64
        + 128 * transport.max_epoch_ids as u64 + transport.max_epoch_id_bytes as u64
        + cfg.io_reserve_bytes + required.task_workspace_bytes;
    assert_eq!(required.input_bytes, expected_input);
}

#[test]
fn explicit_document_and_output_limits_are_never_silently_reduced() {
    let mut cfg = config(); let transport = BatchLimits::default();
    cfg.max_documents = 16;
    assert_eq!(cfg.requirements(32, work(32), transport).unwrap().max_output_bytes,
        MAX_COHORT_OUTPUT_BYTES);
    cfg.max_documents = 17;
    assert!(cfg.requirements(32, work(32), transport).is_err());
    cfg.max_documents = MAX_GROUP_RECORDS;
    let smaller = BatchLimits { max_output_line_bytes: 65_536, ..transport };
    assert!(cfg.requirements(32, work(32), smaller).is_ok());
}

#[test]
fn invalid_and_overflowing_envelopes_fail_without_io_or_allocation() {
    for axis in 0..12 {
        let mut cfg = config();
        match axis {
            0 => cfg.max_documents = 0, 1 => cfg.max_documents = MAX_GROUP_RECORDS + 1,
            2 => cfg.max_heads = 0, 3 => cfg.max_heads = MAX_INT4_SEQUENCES + 1,
            4 => cfg.context_tokens = 0, 5 => cfg.context_tokens = usize::MAX,
            6 => cfg.preparation_reserve_bytes = 0, 7 => cfg.preparation_reserve_bytes = u64::MAX,
            8 => cfg.io_reserve_bytes = 0, 9 => cfg.io_reserve_bytes = u64::MAX,
            10 => cfg.allocator_reserve_bytes = 0, _ => cfg.allocator_reserve_bytes = u64::MAX,
        }
        assert!(cfg.requirements(32, work(32), BatchLimits::default()).is_err(), "axis {axis}");
    }
    let transport = BatchLimits { max_line_bytes: usize::MAX, ..BatchLimits::default() };
    assert!(config().requirements(32, work(32), transport).is_err());
    let mut huge = work(32);
    huge.max_model_work.forward_positions = u64::MAX;
    assert!(config().requirements(32, huge, BatchLimits::default()).is_err());
}

#[test]
fn each_native_work_counter_must_be_finite_and_nonzero() {
    for axis in 0..5 {
        let mut invalid = work(32);
        match axis {
            0 => invalid.max_item_work.forward_positions = 0,
            1 => invalid.max_item_work.projected_logits = 0,
            2 => invalid.max_item_work.attention_pairs = 0,
            3 => invalid.max_item_work.projections.dot_products = 0,
            _ => invalid.max_item_work.projections.multiply_accumulates = 0,
        }
        assert!(config().requirements(32, invalid, BatchLimits::default()).is_err());
    }
}
