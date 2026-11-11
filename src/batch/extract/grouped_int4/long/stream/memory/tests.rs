//! Pure ownership regressions; no model execution or extraction-quality claim.
use super::*;
use crate::{
    batch::extract::grouped_int4::long::Int4ExtractionMappingLimits,
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::mapreduce::{ChunkLimits, ExecutionLimits},
    validation::grounded_fields::GroundingBudget,
};

fn fixture() -> (Int4ExtractionDocumentMemoryLimits, TaskBudget, Int4ExtractionMapLimits, BatchLimits) {
    let memory = Int4ExtractionDocumentMemoryLimits { context_tokens: 128,
        allocator_reserve_bytes: 65_536, preparation_reserve_bytes: 32 * 1024 * 1024,
        io_reserve_bytes: 65_536, reduction_reserve_bytes: 65_536, cohort_chunks: None };
    let budget = TaskBudget { max_input_tokens: 64, max_output_tokens: 16,
        max_output_bytes: 4096, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(128, 32).unwrap().kv_bytes };
    let document = Int4ExtractionMapLimits { mapping: Int4ExtractionMappingLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 64,
            context_tokens: 128, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 3, reduce_fan_in: 8, max_reduction_levels: 8,
            max_task_calls: 1024, max_value_bytes: 65_536, max_live_value_bytes: 131_072,
            max_total_value_bytes: 1_048_576, max_result_bytes: 131_072 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 128 * NANBEIGE_VOCAB_SIZE).unwrap(),
        mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 8000 },
        verification: GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 100_000 }, root_array_union: None };
    (memory, budget, document, BatchLimits::default())
}

#[test]
fn serial_quote_owns_complete_document_results_and_all_frozen_chunk_descriptors() {
    let (memory, budget, document, transport) = fixture();
    let quote = memory.requirements(32, budget, document, transport).unwrap();
    let row = Int4MemoryRequirement::for_context(memory.context_tokens, 32).unwrap();
    assert_eq!(quote.native, row);
    assert_eq!(quote.workspace_bytes,
        row.rope_bytes + row.scratch_payload_bound + memory.allocator_reserve_bytes);
    let chunks = document.mapping.chunks.max_chunks as u64;
    let tokens = chunks * u64::from(budget.max_output_tokens);
    assert_eq!(quote.input_bytes,
        transport.max_line_bytes as u64 * 8 + transport.max_output_line_bytes as u64 * 4
        + transport.max_epoch_ids as u64 * 128 + transport.max_epoch_id_bytes as u64
        + memory.preparation_reserve_bytes + memory.io_reserve_bytes
        + chunks * EXTRACTION_DOCUMENT_CHUNK_METADATA_BYTES);
    assert_eq!(quote.max_output_bytes, document.mapping.reduction.max_result_bytes as u64);
    assert_eq!(quote.output_reservation_bytes, quote.max_output_bytes * 4 + tokens * 8 + 4096);
    let coordinates = std::mem::size_of::<VerifiedSourceSpan>() as u64;
    assert!(coordinates >= 4 * std::mem::size_of::<usize>() as u64);
    assert_eq!(quote.document_workspace_bytes,
        (budget.max_output_bytes * chunks + document.mapping.reduction.max_live_value_bytes as u64) * 4
        + tokens * 8 + document.verification.max_fields as u64 * std::mem::size_of::<SourceFieldEvidence>() as u64
        + document.verification.max_matches as u64 * coordinates * 2 + memory.reduction_reserve_bytes);
}

#[test]
fn union_workspace_is_added_once_without_changing_native_output_or_work_authority() {
    let (memory, budget, mut document, transport) = fixture();
    let plain = memory.requirements(32, budget, document, transport).unwrap();
    let policy = crate::tasks::extract::array_union::RootArrayUnionLimits {
        max_records: 8, max_origins: 32, max_input_bytes: 4096, max_json_bytes: 4096,
        max_nodes: 512, max_result_bytes: 65536 };
    document.root_array_union = Some(policy);
    let selected = memory.requirements(32, budget, document, transport).unwrap();
    assert_eq!(selected.document_workspace_bytes - plain.document_workspace_bytes, policy.requirements().unwrap().workspace_bytes);
    assert_eq!(selected.native, plain.native); assert_eq!(selected.workspace_bytes, plain.workspace_bytes);
    assert_eq!(selected.input_bytes, plain.input_bytes); assert_eq!(selected.output_reservation_bytes, plain.output_reservation_bytes);
    document.root_array_union.as_mut().unwrap().max_result_bytes = document.mapping.reduction.max_result_bytes + 1;
    assert!(memory.requirements(32, budget, document, transport).is_err());
}

#[test]
fn cohort_peak_changes_physical_rows_without_widening_a_chunk_or_final_result() {
    let (memory, budget, document, transport) = fixture();
    let serial = memory.requirements(32, budget, document, transport).unwrap();
    let grouped = Int4ExtractionDocumentMemoryLimits { cohort_chunks: Some(3), ..memory };
    let quote = grouped.requirements(32, budget, document, transport).unwrap();
    let cohort = grouped.cohort_requirements(32, budget, document).unwrap().unwrap();
    assert_eq!(quote.native, Int4MemoryRequirement::for_cohort(&[128; 3], 32).unwrap());
    assert_eq!(quote.native.kv_bytes, serial.native.kv_bytes * 3);
    assert_eq!(budget.max_kv_bytes, serial.native.kv_bytes);
    assert_eq!(quote.document_workspace_bytes, serial.document_workspace_bytes + cohort.workspace_bytes);
    assert_eq!(quote.input_bytes, serial.input_bytes);
    assert_eq!(quote.output_reservation_bytes, serial.output_reservation_bytes);
    assert_eq!(quote.max_output_bytes, serial.max_output_bytes);
    // A cohort's shared peak is not permission to enlarge an individual task.
    let mut widened = budget; widened.max_kv_bytes = quote.native.kv_bytes;
    assert!(grouped.requirements(32, widened, document, transport).is_err());
}

#[test]
fn compact_workspace_is_one_shared_allocation_for_longer_maps_root_arrays_and_cohorts() {
    let (memory, budget, document, transport) = fixture();
    let limits = crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 4096, max_pair_checks: 65_536, max_pair_bytes: 131_072,
    };
    let policy = UntrustedEncoding::SafeBpeV1(limits);
    let workspace = limits.requirements().unwrap().workspace_bytes;
    for cohorts in [None, Some(3)] {
        for chunks in [8, 16] {
            for array in [false, true] {
                let memory = Int4ExtractionDocumentMemoryLimits { cohort_chunks: cohorts, ..memory };
                let mut document = document; document.mapping.chunks.max_chunks = chunks;
                if array {
                    document.root_array_union = Some(crate::tasks::extract::array_union::RootArrayUnionLimits {
                        max_records: 8, max_origins: 32, max_input_bytes: 4096, max_json_bytes: 4096,
                        max_nodes: 512, max_result_bytes: 65_536,
                    });
                }
                let ordinary = memory.requirements(32, budget, document, transport).unwrap();
                assert_eq!(memory.requirements_with_encoding(32, budget, document, transport,
                    &UntrustedEncoding::ByteFallback).unwrap(), ordinary);
                let mut expected = ordinary; expected.input_bytes += workspace;
                assert_eq!(memory.requirements_with_encoding(32, budget, document, transport, &policy).unwrap(), expected);
            }
        }
    }
}

#[test]
fn compact_limits_and_total_live_memory_are_checked_before_input() {
    let (mut memory, budget, document, transport) = fixture();
    let limits = crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 4096, max_pair_checks: 65_536, max_pair_bytes: 131_072,
    };
    for axis in 0..4 {
        let mut invalid = limits;
        match axis { 0 => invalid.max_input_bytes = 0, 1 => invalid.max_output_tokens = 0,
            2 => invalid.max_pair_checks = 0, _ => invalid.max_pair_bytes = 0 }
        assert!(memory.requirements_with_encoding(32, budget, document, transport,
            &UntrustedEncoding::SafeBpeV1(invalid)).is_err(), "axis {axis}");
    }
    let ordinary = memory.requirements(32, budget, document, transport).unwrap();
    let total = ordinary.native.kv_bytes + ordinary.workspace_bytes + ordinary.input_bytes
        + ordinary.document_workspace_bytes + ordinary.output_reservation_bytes;
    memory.preparation_reserve_bytes += u64::MAX - total - 1024;
    assert!(memory.requirements(32, budget, document, transport).is_ok());
    assert!(memory.requirements_with_encoding(32, budget, document, transport,
        &UntrustedEncoding::SafeBpeV1(limits)).is_err());
}

#[test]
fn all_document_chunks_and_both_coordinate_copies_outlive_the_native_window() {
    let (mut memory, budget, mut document, transport) = fixture();
    memory.cohort_chunks = Some(3);
    let first = memory.requirements(32, budget, document, transport).unwrap();
    document.mapping.chunks.max_chunks += 1;
    let longer = memory.requirements(32, budget, document, transport).unwrap();
    assert_eq!(longer.native, first.native);
    assert_eq!(longer.input_bytes - first.input_bytes, EXTRACTION_DOCUMENT_CHUNK_METADATA_BYTES);
    assert_eq!(longer.document_workspace_bytes - first.document_workspace_bytes,
        budget.max_output_bytes * 4 + u64::from(budget.max_output_tokens) * 8);
    assert_eq!(longer.output_reservation_bytes - first.output_reservation_bytes,
        u64::from(budget.max_output_tokens) * 8);
    document.verification.max_fields += 1;
    document.verification.max_matches += 1;
    let evidence = memory.requirements(32, budget, document, transport).unwrap();
    assert_eq!(evidence.document_workspace_bytes - longer.document_workspace_bytes,
        std::mem::size_of::<SourceFieldEvidence>() as u64 + 2 * std::mem::size_of::<VerifiedSourceSpan>() as u64);
}

#[test]
fn complete_schedule_context_output_and_verification_must_fit_before_io() {
    for fault in 0..18 {
        let (mut memory, mut budget, mut document, mut transport) = fixture();
        match fault {
            0 => memory.context_tokens += 1,
            1 => memory.preparation_reserve_bytes = 0,
            2 => memory.io_reserve_bytes = 0,
            3 => memory.reduction_reserve_bytes = 0,
            4 => memory.allocator_reserve_bytes = 0,
            5 => budget.max_kv_bytes -= 1,
            6 => document.mapping.chunks.max_chunks = 0,
            7 => document.mapping.chunks.max_chunks = 257,
            8 => document.mapping.mask_visits_per_chunk = 0,
            9 => document.mapping.max_mask_visits = 999,
            10 => document.verification.max_fields = 0,
            11 => document.verification.max_matches = 0,
            12 => document.verification.max_scan_steps = 0,
            13 => transport.max_output_line_bytes = document.mapping.reduction.max_result_bytes - 1,
            14 => document.mapping.reduction.max_live_value_bytes = document.mapping.reduction.max_value_bytes - 1,
            15 => document.mapping.reduction.reduce_fan_in = 1,
            16 => memory.cohort_chunks = Some(2),
            _ => { memory.cohort_chunks = Some(9); document.mapping.reduction.map_batch_chunks = 9; },
        }
        assert!(memory.requirements(32, budget, document, transport).is_err(), "fault {fault}");
    }
    for chunks in [0, 65, usize::MAX] {
        let (mut memory, budget, mut document, transport) = fixture();
        memory.cohort_chunks = Some(chunks); document.mapping.reduction.map_batch_chunks = chunks;
        assert!(memory.requirements(32, budget, document, transport).is_err());
    }
}

#[test]
fn whole_native_work_and_reservation_arithmetic_never_silently_disappear() {
    for axis in 0..5 {
        let (memory, budget, mut document, transport) = fixture();
        let work = &mut document.mapping.max_model_work;
        match axis { 0 => work.forward_positions = 0, 1 => work.projected_logits = 0,
            2 => work.attention_pairs = 0, 3 => work.projections.dot_products = 0,
            _ => work.projections.multiply_accumulates = 0 }
        assert!(memory.requirements(32, budget, document, transport).is_err(), "axis {axis}");
    }
    for owner in 0..4 {
        let (mut memory, budget, document, transport) = fixture();
        match owner { 0 => memory.preparation_reserve_bytes = u64::MAX,
            1 => memory.io_reserve_bytes = u64::MAX, 2 => memory.allocator_reserve_bytes = u64::MAX,
            _ => memory.reduction_reserve_bytes = u64::MAX }
        assert!(memory.requirements(32, budget, document, transport).is_err(), "owner {owner}");
    }
}

#[test]
fn individually_representable_components_cannot_overflow_the_complete_peak() {
    let (mut memory, budget, document, transport) = fixture();
    let ordinary = memory.requirements(32, budget, document, transport).unwrap();
    let other_input = ordinary.input_bytes - memory.preparation_reserve_bytes;
    // Input itself fits u64, but native/reduction/output ownership cannot fit
    // alongside it. A caller must never receive a wrapped total reservation.
    memory.preparation_reserve_bytes = u64::MAX - other_input - 1024;
    assert!(memory.requirements(32, budget, document, transport).is_err());
}

#[test]
fn root_object_quote_owns_original_partial_schemas_and_reconciliation_through_delivery() {
    let (memory, budget, document, transport) = fixture();
    let compiler = CompileLimits { max_schema_bytes: 4096, max_states: 128, ..CompileLimits::default() };
    let object = RootObjectLimits { max_properties: 8, max_origins: 32,
        max_input_bytes: 4096, max_json_bytes: 4096, max_nodes: 512, max_result_bytes: 65_536 };
    for cohort_chunks in [None, Some(3)] {
        let memory = Int4ExtractionDocumentMemoryLimits { cohort_chunks, ..memory };
        let plain = memory.requirements(32, budget, document, transport).unwrap();
        assert_eq!(memory.requirements_with_root_object(32, budget, document, transport, compiler, None).unwrap(), plain);
        let complete = memory.requirements_with_root_object(32, budget, document, transport, compiler, Some(object)).unwrap();
        assert_eq!(memory.requirements_with_root_object_and_encoding(32, budget, document, transport,
            compiler, Some(object), &UntrustedEncoding::ByteFallback).unwrap(), complete);
        assert_eq!(complete.document_workspace_bytes - plain.document_workspace_bytes,
            object.workspace_bytes().unwrap() + RootObjectSchema::workspace_bytes(compiler).unwrap());
        assert_eq!(complete.native, plain.native);
        assert_eq!(complete.workspace_bytes, plain.workspace_bytes);
        assert_eq!(complete.input_bytes, plain.input_bytes);
        assert_eq!(complete.output_reservation_bytes, plain.output_reservation_bytes);
        assert_eq!(complete.max_output_bytes, plain.max_output_bytes);
    }
}

#[test]
fn root_object_quote_rejects_competing_reducers_and_incoherent_bounds_before_io() {
    for fault in 0..7 {
        let (memory, budget, mut document, transport) = fixture();
        let mut compiler = CompileLimits { max_schema_bytes: 4096, max_states: 128, ..CompileLimits::default() };
        let mut object = RootObjectLimits { max_properties: 8, max_origins: 32,
            max_input_bytes: 4096, max_json_bytes: 4096, max_nodes: 512, max_result_bytes: 65_536 };
        match fault {
            0 => document.root_array_union = Some(crate::tasks::extract::array_union::RootArrayUnionLimits {
                max_records: 8, max_origins: 32, max_input_bytes: 4096,
                max_json_bytes: 4096, max_nodes: 512, max_result_bytes: 65_536 }),
            1 => object.max_properties = 0,
            2 => object.max_origins = 1,
            3 => object.max_json_bytes = document.mapping.reduction.max_result_bytes + 1,
            4 => object.max_result_bytes = document.mapping.reduction.max_result_bytes + 1,
            5 => compiler.max_schema_bytes = 0,
            _ => compiler.max_states = usize::MAX,
        }
        assert!(memory.requirements_with_root_object(32, budget, document, transport, compiler, Some(object)).is_err(),
            "fault {fault}");
    }
}

#[test]
fn root_object_extra_frontier_cannot_overflow_a_previously_representable_total() {
    let (mut memory, budget, document, transport) = fixture();
    let compiler = CompileLimits { max_schema_bytes: 4096, max_states: 128, ..CompileLimits::default() };
    let object = RootObjectLimits { max_properties: 8, max_origins: 32,
        max_input_bytes: 4096, max_json_bytes: 4096, max_nodes: 512, max_result_bytes: 65_536 };
    let ordinary = memory.requirements(32, budget, document, transport).unwrap();
    let ordinary_total = ordinary.native.kv_bytes + ordinary.workspace_bytes + ordinary.input_bytes
        + ordinary.document_workspace_bytes + ordinary.output_reservation_bytes;
    memory.preparation_reserve_bytes += u64::MAX - ordinary_total - 1024;
    assert!(memory.requirements(32, budget, document, transport).is_ok());
    assert!(memory.requirements_with_root_object(32, budget, document, transport, compiler, Some(object)).is_err());
}
