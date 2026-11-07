//! Finite quotation fixtures: no model allocation, execution or RSS claim.
use super::*;
use crate::{batch::extract::{ExtractionMaskBudget, grouped_int4::Int4ExtractionCohortBatchLimits},
    grammar::mask::MaskWorkLimits,
    native_engine::{kv::KV_BYTES_PER_TOKEN, quant_algebra::MAX_MODEL_K,
        rope::DEFAULT_ADMITTED_CONTEXT_CAP, strict_int4::{Int4MemoryBudget, Int4Work}}};

const WIDTH: usize = 127;
fn limits() -> Int4ExtractionCohortMemoryLimits {
    Int4ExtractionCohortMemoryLimits { max_documents: 3, context_tokens: 32,
        preparation_reserve_bytes: 10_000, io_reserve_bytes: 4096, allocator_reserve_bytes: 8192 }
}
fn transport() -> BatchLimits {
    BatchLimits { max_line_bytes: 1024, max_document_bytes: 512, max_id_bytes: 16,
        max_epoch_ids: 17, max_epoch_id_bytes: 3072, max_output_line_bytes: 8192,
        ..BatchLimits::default() }
}
fn work(tokens: usize) -> Int4ExtractionBatchLimits {
    Int4ExtractionBatchLimits {
        max_model_work: Int4Work::for_sequence(WIDTH, 0, 256, tokens * NANBEIGE_VOCAB_SIZE).unwrap(),
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
            max_visits_per_item: 4000, max_visits_per_run: 1_000_000 },
    }
}
fn rejected(result: Result<Int4ExtractionCohortMemoryRequirement, BatchFault>) {
    assert_eq!(result.unwrap_err().code, BatchCode::InvalidLimits);
}

#[test]
fn full_peak_includes_independent_kv_shared_rope_and_all_live_input_and_output_owners() {
    let limits = limits(); let transport = transport();
    let quoted = limits.requirements(WIDTH, work(70), transport).unwrap();
    let native = Int4MemoryRequirement::for_cohort(&[32, 32, 32], WIDTH).unwrap();
    let serial = Int4MemoryRequirement::for_context(32, WIDTH).unwrap();
    assert_eq!(quoted.native, native);
    assert_eq!(quoted.native.kv_bytes, 3 * 32 * KV_BYTES_PER_TOKEN as u64);
    assert_eq!(quoted.native.rope_bytes, serial.rope_bytes);
    assert!(quoted.native.scratch_payload_bound > serial.scratch_payload_bound);
    assert_eq!(quoted.workspace_bytes, native.rope_bytes + native.scratch_payload_bound + 8192);

    let result_bytes = 3 * 8192 + 4096 + 3;
    let decoder = 64 * 70 + 8 * result_bytes + 65536 * 3 + NANBEIGE_VOCAB_SIZE.div_ceil(8) as u64;
    assert_eq!(quoted.max_result_bytes, result_bytes);
    assert_eq!(quoted.decode_workspace_bytes, decoder);
    // Three pending documents and one lookahead each retain input and their
    // preparation reserve; output framing and epoch identity storage stay live.
    assert_eq!(quoted.input_bytes,
        4 * (8 * 1024 + 10_000) + 4 * 8192 + 17 * 128 + 3072 + 4096 + decoder);
    // Tokens are charged at eight bytes each across the entire window, in
    // addition to four copies of each bounded result and each claim's overhead.
    assert_eq!(quoted.output_reservation_bytes, 4 * 3 * 8192 + 8 * 70 + 3 * 4096);
}

#[test]
fn lookahead_and_staging_growth_cannot_hide_in_the_native_workspace_reservation() {
    let base = limits().requirements(WIDTH, work(70), transport()).unwrap();
    for axis in 0..5 {
        let mut limits = limits(); let mut transport = transport();
        let (input_growth, workspace_growth) = match axis {
            0 => { limits.preparation_reserve_bytes += 1; (4, 0) }
            1 => { limits.io_reserve_bytes += 1; (1, 0) }
            2 => { limits.allocator_reserve_bytes += 1; (0, 1) }
            3 => { transport.max_line_bytes += 1; (8 * 4, 0) }
            _ => { transport.max_epoch_ids += 1; (128, 0) }
        };
        let quoted = limits.requirements(WIDTH, work(70), transport).unwrap();
        assert_eq!(quoted.input_bytes, base.input_bytes + input_growth, "axis {axis}");
        assert_eq!(quoted.workspace_bytes, base.workspace_bytes + workspace_growth, "axis {axis}");
        assert_eq!(quoted.native, base.native);
        assert_eq!(quoted.output_reservation_bytes, base.output_reservation_bytes);
    }
    let mut wider_output = transport(); wider_output.max_output_line_bytes += 1;
    let quoted = limits().requirements(WIDTH, work(70), wider_output).unwrap();
    assert_eq!(quoted.max_result_bytes, base.max_result_bytes + 3);
    assert_eq!(quoted.decode_workspace_bytes, base.decode_workspace_bytes + 8 * 3);
    assert_eq!(quoted.input_bytes, base.input_bytes + 4 + 8 * 3);
    assert_eq!(quoted.output_reservation_bytes, base.output_reservation_bytes + 4 * 3);
}

#[test]
fn whole_run_tokens_bound_output_even_when_fewer_than_all_declared_rows_can_execute() {
    let one = limits().requirements(WIDTH, work(1), transport()).unwrap();
    let seventy = limits().requirements(WIDTH, work(70), transport()).unwrap();
    assert_eq!(one.native, seventy.native);
    assert_eq!(one.max_result_bytes, seventy.max_result_bytes);
    assert_eq!(seventy.decode_workspace_bytes - one.decode_workspace_bytes, 64 * 69);
    assert_eq!(seventy.input_bytes - one.input_bytes, 64 * 69);
    assert_eq!(seventy.output_reservation_bytes - one.output_reservation_bytes, 8 * 69);

    let full = limits().requirements(WIDTH, work(96), transport()).unwrap();
    let longer_run = limits().requirements(WIDTH, work(192), transport()).unwrap();
    assert_eq!(full, longer_run); // Three contexts of 32 bound live tokens.
    let mut ceiling = work(70);
    ceiling.max_model_work.projected_logits += NANBEIGE_VOCAB_SIZE as u64 - 1;
    assert_eq!(limits().requirements(WIDTH, ceiling, transport()).unwrap(), seventy);
    ceiling.max_model_work.projected_logits = NANBEIGE_VOCAB_SIZE as u64 - 1;
    rejected(limits().requirements(WIDTH, ceiling, transport()));
    ceiling.max_model_work.projected_logits = u64::MAX;
    assert_eq!(limits().requirements(WIDTH, ceiling, transport()).unwrap(), full);
}

#[test]
fn the_quote_admits_its_exact_native_payload_and_refuses_each_underfunded_axis() {
    let quote = limits().requirements(WIDTH, work(70), transport()).unwrap();
    let exact = Int4ExtractionCohortBatchLimits { max_documents: 3, context_limit: 32,
        native_memory: Int4MemoryBudget { max_kv_bytes: quote.native.kv_bytes,
            max_rope_bytes: quote.native.rope_bytes,
            max_scratch_payload_bytes: quote.native.scratch_payload_bound },
        max_workspace_bytes: quote.decode_workspace_bytes, max_item_result_bytes: 8192,
        max_result_bytes: quote.max_result_bytes };
    exact.validate(WIDTH).unwrap();
    for axis in 0..3 {
        let mut insufficient = exact;
        match axis { 0 => insufficient.native_memory.max_kv_bytes -= 1,
            1 => insufficient.native_memory.max_rope_bytes -= 1,
            _ => insufficient.native_memory.max_scratch_payload_bytes -= 1 }
        assert_eq!(insufficient.validate(WIDTH).unwrap_err().code, BatchCode::InvalidLimits, "axis {axis}");
    }
}

#[test]
fn every_native_work_and_mask_axis_is_validated_before_memory_is_quoted() {
    for axis in 0..9 {
        let mut work = work(70);
        match axis {
            0 => work.max_model_work.forward_positions = 0,
            1 => work.max_model_work.projected_logits = 0,
            2 => work.max_model_work.attention_pairs = 0,
            3 => work.max_model_work.projections.dot_products = 0,
            4 => work.max_model_work.projections.multiply_accumulates = 0,
            5 => work.masks.per_mask.max_trie_node_visits = 0,
            6 => work.masks.per_mask.checkpoint_interval_nodes = 0,
            7 => work.masks.max_visits_per_item = 0,
            _ => work.masks.max_visits_per_run = work.masks.max_visits_per_item - 1,
        }
        rejected(limits().requirements(WIDTH, work, transport()));
    }
}

#[test]
fn row_context_partition_and_transport_bounds_refuse_before_native_allocation() {
    for axis in 0..9 {
        let mut limits = limits();
        match axis {
            0 => limits.max_documents = 0,
            1 => limits.max_documents = MAX_INT4_SEQUENCES + 1,
            2 => limits.max_documents = usize::MAX,
            3 => limits.context_tokens = 0,
            4 => limits.context_tokens = DEFAULT_ADMITTED_CONTEXT_CAP + 1,
            5 => limits.context_tokens = usize::MAX,
            6 => limits.preparation_reserve_bytes = 0,
            7 => limits.io_reserve_bytes = 0,
            _ => limits.allocator_reserve_bytes = 0,
        }
        rejected(limits.requirements(WIDTH, work(70), transport()));
    }
    for width in [0, MAX_MODEL_K + 1, usize::MAX] {
        rejected(limits().requirements(width, work(70), transport()));
    }
    for axis in 0..4 {
        let mut transport = transport();
        match axis { 0 => transport.max_requests = 0, 1 => transport.max_output_line_bytes = 0,
            2 => transport.max_epoch_id_bytes = transport.max_id_bytes - 1, _ => transport.max_json_depth = 65 }
        rejected(limits().requirements(WIDTH, work(70), transport));
    }
    let mut largest = limits(); largest.max_documents = MAX_INT4_SEQUENCES;
    assert!(largest.requirements(MAX_MODEL_K, work(70), transport()).is_ok());
}

#[test]
fn complete_result_envelope_accepts_the_exact_global_cap_and_refuses_one_byte_more() {
    let mut limits = limits(); limits.max_documents = 1;
    let mut transport = transport();
    transport.max_output_line_bytes = (MAX_COHORT_RESULT_BYTES - 4096 - 1) as usize;
    let exact = limits.requirements(WIDTH, work(70), transport).unwrap();
    assert_eq!(exact.max_result_bytes, MAX_COHORT_RESULT_BYTES);
    transport.max_output_line_bytes += 1;
    rejected(limits.requirements(WIDTH, work(70), transport));
    transport.max_output_line_bytes = usize::MAX;
    rejected(limits.requirements(WIDTH, work(70), transport));
}

#[test]
fn reservation_arithmetic_overflow_is_an_explicit_invalid_limits_failure() {
    for axis in 0..3 {
        let mut limits = limits();
        match axis { 0 => limits.preparation_reserve_bytes = u64::MAX,
            1 => limits.io_reserve_bytes = u64::MAX, _ => limits.allocator_reserve_bytes = u64::MAX }
        rejected(limits.requirements(WIDTH, work(70), transport()));
    }
    #[cfg(target_pointer_width = "64")]
    for axis in 0..3 {
        let mut transport = transport();
        match axis { 0 => transport.max_line_bytes = usize::MAX,
            1 => transport.max_epoch_ids = usize::MAX, _ => transport.max_epoch_id_bytes = usize::MAX }
        rejected(limits().requirements(WIDTH, work(70), transport));
    }
}

#[test]
fn packed_workspace_is_charged_once_without_changing_document_or_output_authority() {
    for documents in [1, 3, 64] {
        let mut limits = limits(); limits.max_documents = documents;
        let base = limits.requirements(WIDTH, work(70), transport()).unwrap();
        for rows in [1, 2, 64] {
            let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            // An oversized caller ceiling is not an allocation or permission
            // to multiply this workspace by every independent document.
            for cap in [extra, u64::MAX] {
                let mut packed = limits.requirements_packed(WIDTH, work(70), transport(),
                    Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: cap }).unwrap();
                assert_eq!(packed.workspace_bytes, base.workspace_bytes + extra);
                packed.workspace_bytes = base.workspace_bytes;
                assert_eq!(packed, base);
            }
        }
    }
}

#[test]
fn packed_cap_and_extra_scratch_refuse_before_any_native_allocation() {
    for rows in [0, 65, usize::MAX] {
        rejected(limits().requirements_packed(WIDTH, work(70), transport(),
            Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX }));
    }
    for rows in [1, 2, 64] {
        let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
        rejected(limits().requirements_packed(WIDTH, work(70), transport(),
            Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra - 1 }));
    }
    let mut limits = limits();
    let native = Int4MemoryRequirement::for_cohort(&[32, 32, 32], WIDTH).unwrap();
    limits.allocator_reserve_bytes = u64::MAX - native.rope_bytes - native.scratch_payload_bound;
    assert_eq!(limits.requirements(WIDTH, work(70), transport()).unwrap().workspace_bytes, u64::MAX);
    rejected(limits.requirements_packed(WIDTH, work(70), transport(),
        Int4PrefillLimits { max_batch_rows: 1, max_extra_scratch_bytes: u64::MAX }));
}

#[test]
fn selected_projection_quotes_full_token_storage_and_packed_scratch_once_per_window() {
    for documents in [1, 3, 64] {
        let mut limits = limits(); limits.max_documents = documents;
        let dense = limits.requirements(WIDTH, work(70), transport()).unwrap();
        for max_rows_per_step in [1, 7, NANBEIGE_VOCAB_SIZE] {
            let selected = Int4JsonSparseLimits { max_rows_per_step };
            let mut work = work(70);
            work.max_model_work = Int4Work::for_sequence(WIDTH, 0, 256, 70 * max_rows_per_step).unwrap();
            let base = limits.requirements_for_execution(WIDTH, work, transport(), Some(selected), None).unwrap();
            assert_eq!(base, dense); // A smaller projection never shrinks token, EOS or result storage.
            for rows in [1, 2, 64] {
                let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
                let mut packed = limits.requirements_for_execution(WIDTH, work, transport(), Some(selected),
                    Some(Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).unwrap();
                assert_eq!(packed.workspace_bytes, base.workspace_bytes + extra);
                packed.workspace_bytes = base.workspace_bytes;
                assert_eq!(packed, base);
            }
            work.max_model_work.projected_logits += max_rows_per_step as u64 - 1;
            assert_eq!(limits.requirements_for_execution(WIDTH, work, transport(), Some(selected), None).unwrap(), base);
        }
    }
    for max_rows_per_step in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        rejected(limits().requirements_for_execution(WIDTH, work(70), transport(),
            Some(Int4JsonSparseLimits { max_rows_per_step }), None));
    }
    let mut work = work(70); work.max_model_work.projected_logits = 6;
    rejected(limits().requirements_for_execution(WIDTH, work, transport(),
        Some(Int4JsonSparseLimits { max_rows_per_step: 7 }), None));
}
