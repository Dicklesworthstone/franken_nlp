//! UNRUN source fixtures for complete schema-extraction admission and ownership.
//! Synthetic work/identity data do not establish native inference or durability.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::extract::grouped_int4::long::Int4ExtractionMappingLimits,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::RunLimits,
    jobs::{JobWork, population::PopulationReadLimits},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::mapreduce::{ChunkLimits, ExecutionLimits},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan},
};
use std::{io::Cursor, time::Duration};

const WIDTH: usize = 32;
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-extraction-document-host-admission");
    ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH,
        )).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "fixture".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "constrained-json".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }
}
fn work(copies: u64) -> Int4Work {
    let one = Int4Work::for_sequence(WIDTH, 0, 64, 16 * NANBEIGE_VOCAB_SIZE).unwrap();
    Int4Work { forward_positions: one.forward_positions * copies,
        projected_logits: one.projected_logits * copies, attention_pairs: one.attention_pairs * copies,
        projections: ProjectionWork { dot_products: one.projections.dot_products * copies,
            multiply_accumulates: one.projections.multiply_accumulates * copies } }
}
fn fixture() -> (TaskBudget, Int4ExtractionMapLimits, JobLimits, JobHostLimits) {
    let native = Int4MemoryRequirement::for_context(64, WIDTH).unwrap();
    let budget = TaskBudget { max_input_tokens: 48, max_output_tokens: 16,
        max_output_bytes: 4096, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
    let document = Int4ExtractionMapLimits {
        root_array_union: None,
        mapping: Int4ExtractionMappingLimits {
            chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 1024,
                max_chunk_tokens: 48, context_tokens: 64, reserved_tokens: 16,
                max_chunks: 4, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 4,
                max_reduction_levels: 8, max_task_calls: 64, max_value_bytes: 8192,
                max_live_value_bytes: 65536, max_total_value_bytes: 1 << 20, max_result_bytes: 65536 },
            max_model_work: work(4),
            mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
            mask_visits_per_chunk: 4000, max_mask_visits: 16000,
        },
        verification: GroundingBudget { max_fields: 64, max_matches: 128, max_scan_steps: 65536 },
    };
    let job = JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(document.mapping.max_model_work, document.mapping.max_mask_visits) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 131072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (budget, document, job, host)
}
fn request(ids: &[ExecutionIdentity], document: Int4ExtractionMapLimits, kv: u64)
    -> Int4ExtractionDocumentAdmission<'_> {
    Int4ExtractionDocumentAdmission { identities: ids, model_work: work(2), mask_node_visits: 8000,
        document_encoding: UntrustedEncoding::ByteFallback,
        mask_limits: document.mapping.mask_limits, kv_reservation_bytes: kv,
        max_result_bytes: 65536, max_output_tokens: 32, chunk_count: 2 }
}

#[test]
fn whole_population_two_schema_owners_and_maximum_frontier_are_admitted_together() {
    let (budget, document, job, host) = fixture();
    let frontier = document_workspace_bytes(budget, document, 4, 65536).unwrap();
    let need = requirements(WIDTH, budget, document, 65536, job, host, frontier, 8192,
        JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).unwrap();
    assert_eq!(need.input_bytes, host.required_buffer_bytes(job).unwrap() + 2 * 8192);
    assert_eq!(need.document_workspace_bytes, frontier);
    assert_eq!(need.native.kv_bytes, budget.max_kv_bytes);
    assert_eq!(need.native_workspace_bytes, need.native.rope_bytes + need.native.scratch_payload_bound
        + host.native.allocator_reserve_bytes);
}

#[test]
fn compact_encoder_is_reserved_once_before_input_independent_of_chunks_cohorts_and_input_mode() {
    use crate::{batch::extract::grouped_int4::long::cohort::Int4ExtractionMapCohortMemoryLimits,
        tokenizer::untrusted::SafeBpeLimits};
    let (budget, document, job, host) = fixture();
    let compact = SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 32768,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let encoder_bytes = compact.requirements().unwrap().workspace_bytes;
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        for chunks in [1, 4] {
            let mut document = document; document.mapping.chunks.max_chunks = chunks;
            let frontier = document_workspace_bytes(budget, document, chunks, 65536).unwrap();
            let byte = requirements(WIDTH, budget, document, 65536, job, host, frontier, 8192,
                input, &UntrustedEncoding::ByteFallback).unwrap();
            let compact = requirements(WIDTH, budget, document, 65536, job, host, frontier, 8192,
                input, &UntrustedEncoding::SafeBpeV1(compact)).unwrap();
            assert_eq!(compact.input_bytes - byte.input_bytes, encoder_bytes);
            assert_eq!(compact.document_workspace_bytes, frontier);
            assert_eq!(compact.native, byte.native);
            assert_eq!(compact.native_workspace_bytes, byte.native_workspace_bytes);
            for rows in [1, chunks] {
                let quote = Int4ExtractionMapCohortMemoryLimits { max_chunks: rows, context_tokens: 64,
                    max_output_tokens_per_chunk: 16, max_result_bytes_per_chunk: 4096 }.requirements(WIDTH).unwrap();
                let mut grouped_byte = byte; let mut grouped_compact = compact;
                grouped_byte.use_cohorts(quote, host.native.allocator_reserve_bytes).unwrap();
                grouped_compact.use_cohorts(quote, host.native.allocator_reserve_bytes).unwrap();
                assert_eq!(grouped_compact.input_bytes - grouped_byte.input_bytes, encoder_bytes);
                assert_eq!(grouped_compact.document_workspace_bytes, frontier);
                assert_eq!(grouped_compact.native, grouped_byte.native);
                assert_eq!(grouped_compact.native_workspace_bytes, grouped_byte.native_workspace_bytes);
            }
        }
    }
}

#[test]
fn every_compact_encoder_cap_is_validated_before_population_ownership() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let (budget, document, job, host) = fixture();
    for axis in 0..8 {
        let mut caps = SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 32768,
            max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
        match axis { 0 => caps.max_input_bytes = 0, 1 => caps.max_output_tokens = 0,
            2 => caps.max_pair_checks = 0, 3 => caps.max_pair_bytes = 0,
            4 => caps.max_input_bytes = SafeBpeLimits::MAX_INPUT_BYTES + 1,
            5 => caps.max_output_tokens = SafeBpeLimits::MAX_OUTPUT_TOKENS + 1,
            6 => caps.max_pair_checks = SafeBpeLimits::MAX_PAIR_CHECKS + 1,
            _ => caps.max_pair_bytes = SafeBpeLimits::MAX_PAIR_BYTES + 1 }
        assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 8192,
            JobInputMemory::CompletePopulation, &UntrustedEncoding::SafeBpeV1(caps)).is_err(), "axis {axis}");
    }
}

#[test]
fn native_admission_binds_the_complete_encoding_policy_that_owns_the_workspace() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let (budget, document, _, _) = fixture(); let ids = [identity(), identity()];
    let mut claim = request(&ids, document, budget.max_kv_bytes);
    check_encoding(&UntrustedEncoding::ByteFallback, &claim).unwrap();
    let compact = SafeBpeLimits { max_input_bytes: 65536, max_output_tokens: 32768,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let expected = UntrustedEncoding::SafeBpeV1(compact);
    assert!(check_encoding(&expected, &claim).is_err());
    claim.document_encoding = expected.clone();
    check_encoding(&expected, &claim).unwrap();
    assert!(check_encoding(&UntrustedEncoding::ByteFallback, &claim).is_err());
    for axis in 0..4 {
        let mut changed = compact;
        match axis { 0 => changed.max_input_bytes += 1, 1 => changed.max_output_tokens += 1,
            2 => changed.max_pair_checks += 1, _ => changed.max_pair_bytes += 1 }
        claim.document_encoding = UntrustedEncoding::SafeBpeV1(changed);
        let error = check_encoding(&expected, &claim).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission, "axis {axis}");
    }
    let invalid = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_pair_checks: 0, ..compact });
    claim.document_encoding = invalid.clone();
    assert!(check_encoding(&invalid, &claim).is_err());
}

#[test]
fn complete_object_frontier_includes_schema_and_reducer_ownership_without_native_renewal() {
    use crate::{grammar::CompileLimits, tasks::extract::object_reconcile::{RootObjectLimits, RootObjectSchema}};
    let (budget, document, job, host) = fixture();
    let compiler = CompileLimits::default();
    let object = RootObjectLimits { max_properties: 32, max_origins: 128,
        max_input_bytes: 1 << 20, max_json_bytes: 32768, max_nodes: 4096, max_result_bytes: 65536 };
    let ordinary = complete_workspace_bytes(budget, document, 65536, compiler, None).unwrap();
    let complete = complete_workspace_bytes(budget, document, 65536, compiler, Some(object)).unwrap();
    assert_eq!(complete - ordinary, object.workspace_bytes().unwrap() + RootObjectSchema::workspace_bytes(compiler).unwrap());
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let before = requirements(WIDTH, budget, document, 65536, job, host, ordinary, 8192,
            input, &UntrustedEncoding::ByteFallback).unwrap();
        let after = requirements(WIDTH, budget, document, 65536, job, host, complete, 8192,
            input, &UntrustedEncoding::ByteFallback).unwrap();
        assert_eq!(before.input_bytes, after.input_bytes);
        assert_eq!(before.native, after.native);
        assert_eq!(before.native_workspace_bytes, after.native_workspace_bytes);
        assert_eq!(after.document_workspace_bytes - before.document_workspace_bytes, complete - ordinary);
    }
    let larger = CompileLimits { max_schema_bytes: compiler.max_schema_bytes + 1, ..compiler };
    assert!(complete_workspace_bytes(budget, document, 65536, larger, Some(object)).unwrap() > complete);
}

#[test]
fn object_frontier_refuses_conflicting_reducers_and_schema_workspace_overflow() {
    use crate::{grammar::CompileLimits, tasks::extract::{object_reconcile::RootObjectLimits,
        array_union::RootArrayUnionLimits}};
    let (budget, document, _, _) = fixture();
    let object = RootObjectLimits { max_properties: 32, max_origins: 128,
        max_input_bytes: 1 << 20, max_json_bytes: 32768, max_nodes: 4096, max_result_bytes: 65536 };
    let mut conflict = document;
    conflict.root_array_union = Some(RootArrayUnionLimits { max_records: 32, max_origins: 128,
        max_input_bytes: 1 << 20, max_json_bytes: 32768, max_nodes: 4096, max_result_bytes: 65536 });
    assert!(complete_workspace_bytes(budget, conflict, 65536, CompileLimits::default(), Some(object)).is_err());
    for compiler in [CompileLimits { max_schema_bytes: usize::MAX, ..CompileLimits::default() },
        CompileLimits { max_states: usize::MAX, ..CompileLimits::default() },
        CompileLimits { max_schema_bytes: 0, ..CompileLimits::default() }] {
        assert!(complete_workspace_bytes(budget, document, 65536, compiler, Some(object)).is_err());
    }
    let too_large = RootObjectLimits { max_result_bytes: 65537, ..object };
    assert!(complete_workspace_bytes(budget, document, 65536, CompileLimits::default(), Some(too_large)).is_err());
}

#[test]
fn exact_resident_context_kv_and_nonzero_schema_are_required_before_input() {
    for axis in 0..5 {
        let (mut budget, mut document, job, mut host) = fixture();
        match axis { 0 => budget.max_kv_bytes -= 1, 1 => budget.max_kv_bytes += 1,
            2 => document.mapping.chunks.context_tokens -= 1, 3 => host.native.context_tokens += 1,
            _ => host.native.allocator_reserve_bytes = 0 }
        assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 128,
            JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).is_err(), "axis {axis}");
    }
    let (budget, document, job, host) = fixture();
    assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 0,
        JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).is_err());
}

#[test]
fn cohort_reservation_keeps_both_schema_owners_and_the_complete_document_frontier() {
    use crate::batch::extract::grouped_int4::long::cohort::Int4ExtractionMapCohortMemoryLimits;
    let (budget, document, job, host) = fixture();
    let frontier = document_workspace_bytes(budget, document, 4, 65536).unwrap();
    let serial = requirements(WIDTH, budget, document, 65536, job, host, frontier, 8192,
        JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).unwrap();
    for chunks in [1, 2, 4] {
        let quote = Int4ExtractionMapCohortMemoryLimits { max_chunks: chunks, context_tokens: 64,
            max_output_tokens_per_chunk: 16, max_result_bytes_per_chunk: 4096 }.requirements(WIDTH).unwrap();
        let mut grouped = serial;
        grouped.use_cohorts(quote, host.native.allocator_reserve_bytes).unwrap();
        assert_eq!(grouped.native, quote.native);
        assert_eq!(grouped.native.kv_bytes, chunks as u64 * budget.max_kv_bytes);
        assert_eq!(grouped.input_bytes, serial.input_bytes);
        assert_eq!(grouped.document_workspace_bytes, frontier);
        assert_eq!(grouped.native_workspace_bytes, quote.native.rope_bytes
            + quote.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
        assert!(quote.workspace_bytes > 0);
        assert_eq!(budget.max_kv_bytes, serial.native.kv_bytes);
    }
}

#[test]
fn incomplete_native_or_decoder_declarations_refuse_before_job_execution() {
    use crate::batch::extract::grouped_int4::long::cohort::Int4ExtractionMapCohortMemoryLimits;
    let (budget, document, job, host) = fixture();
    let serial = requirements(WIDTH, budget, document, 65536, job, host, 65536, 8192,
        JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).unwrap();
    let quote = Int4ExtractionMapCohortMemoryLimits { max_chunks: 4, context_tokens: 64,
        max_output_tokens_per_chunk: 16, max_result_bytes_per_chunk: 4096 }.requirements(WIDTH).unwrap();
    for axis in 0..5 {
        let mut short = quote;
        match axis { 0 => short.native.kv_bytes = serial.native.kv_bytes - 1,
            1 => short.native.rope_bytes = serial.native.rope_bytes - 1,
            2 => short.native.scratch_payload_bound = serial.native.scratch_payload_bound - 1,
            3 => short.workspace_bytes = 0, _ => short.max_result_bytes = 0 }
        let mut required = serial;
        assert!(required.use_cohorts(short, host.native.allocator_reserve_bytes).is_err());
    }
    let mut required = serial;
    assert!(required.use_cohorts(quote, u64::MAX).is_err());
}

#[test]
fn cohort_admission_retains_whole_window_kv_with_exact_semantic_result_tokens() {
    let (budget, document, _, _) = fixture(); let base = identity(); let ids = [identity(), identity()];
    let peak = 4 * budget.max_kv_bytes;
    let mut claim = request(&ids, document, peak);
    check_request(WIDTH, &base, peak, document, 65536, 16, None, &claim).unwrap();
    claim.kv_reservation_bytes = budget.max_kv_bytes;
    assert!(check_request(WIDTH, &base, peak, document, 65536, 16, None, &claim).is_err());
    claim.kv_reservation_bytes = peak; claim.max_output_tokens += 1;
    assert!(check_request(WIDTH, &base, peak, document, 65536, 16, None, &claim).is_err());
}

#[test]
fn every_document_work_axis_and_masks_fit_the_persisted_lifetime_budget() {
    for axis in 0..6 {
        let (budget, document, mut job, host) = fixture();
        match axis { 0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1, 2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            _ => job.max_work.mask_node_visits -= 1 }
        assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 128,
            JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).is_err(), "axis {axis}");
    }
}

#[test]
fn many_native_sequences_can_exceed_one_context_without_changing_the_group_work() {
    let (budget, document, _, _) = fixture(); let base = identity(); let ids = [identity(), identity()];
    let req = request(&ids, document, budget.max_kv_bytes);
    assert!(req.model_work.forward_positions > document.mapping.chunks.context_tokens as u64);
    assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16, None, &req).is_ok());
}

#[test]
fn the_last_chunk_must_have_the_same_task_pinned_assets_and_profile() {
    let (budget, document, _, _) = fixture(); let base = identity();
    for axis in 0..5 {
        let mut ids = [identity(), identity()];
        match axis { 0 => ids[1].task_spec = "ner-v1".to_owned(),
            1 => ids[1].numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => ids[1].quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=128"),
            3 => ids[1].tokenizer_digest = Sha256Digest::of_bytes(b"different-tokenizer"),
            _ => ids[1].template_digest = Sha256Digest::of_bytes(b"different-template") }
        assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16, None,
            &request(&ids, document, budget.max_kv_bytes)).is_err(), "axis {axis}");
    }
    let ids = [identity()];
    assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16, None,
        &request(&ids, document, budget.max_kv_bytes)).is_err());
}

#[test]
fn chunk_count_mask_token_and_result_claims_cannot_diverge_from_complete_preparation() {
    let (budget, document, _, _) = fixture(); let base = identity(); let ids = [identity(), identity()];
    for axis in 0..13 {
        let mut req = request(&ids, document, budget.max_kv_bytes);
        match axis { 0 => req.chunk_count = 0, 1 => req.chunk_count = 5,
            2 => req.mask_node_visits -= 1, 3 => req.mask_limits.max_trie_node_visits += 1,
            4 => req.mask_limits.checkpoint_interval_nodes += 1, 5 => req.max_output_tokens -= 1,
            6 => req.max_result_bytes -= 1, 7 => req.kv_reservation_bytes -= 1,
            8 => req.model_work.forward_positions = document.mapping.max_model_work.forward_positions + 1,
            9 => req.model_work.projected_logits = document.mapping.max_model_work.projected_logits + 1,
            10 => req.model_work.attention_pairs = document.mapping.max_model_work.attention_pairs + 1,
            11 => req.model_work.projections.dot_products = document.mapping.max_model_work.projections.dot_products + 1,
            _ => req.model_work.projections.multiply_accumulates = document.mapping.max_model_work.projections.multiply_accumulates + 1 }
        assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16, None, &req).is_err(), "axis {axis}");
    }
}

#[test]
fn shared_workspace_quote_prices_all_native_values_and_both_evidence_coordinate_copies() {
    let (budget, mut document, _, _) = fixture();
    let before = document_workspace_bytes(budget, document, 3, 65536).unwrap();
    let four = document_workspace_bytes(budget, document, 4, 65536).unwrap();
    assert_eq!(four - before, 4 * budget.max_output_bytes + 8 * u64::from(budget.max_output_tokens));
    document.verification.max_fields += 1;
    assert_eq!(document_workspace_bytes(budget, document, 3, 65536).unwrap() - before,
        std::mem::size_of::<SourceFieldEvidence>() as u64);
    let before = document_workspace_bytes(budget, document, 3, 65536).unwrap();
    document.verification.max_matches += 1;
    assert_eq!(document_workspace_bytes(budget, document, 3, 65536).unwrap() - before,
        2 * std::mem::size_of::<VerifiedSourceSpan>() as u64);
}

#[test]
fn zero_reserves_overflow_and_small_stored_result_caps_refuse_before_execution() {
    let (budget, document, job, host) = fixture();
    assert!(requirements(WIDTH, budget, document, 65536, job, host, 0, 128,
        JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).is_err());
    assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, usize::MAX,
        JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).is_err());
    assert!(requirements(WIDTH, budget, document, job.max_result_bytes as u64 + 1,
        job, host, 65536, 128, JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).is_err());
    for chunks in [0, 5, 257, usize::MAX] {
        assert!(document_workspace_bytes(budget, document, chunks, 65536).is_err());
    }
    for axis in 0..4 {
        let mut host = host;
        match axis { 0 => host.preparation_reserve_bytes = 0, 1 => host.io_reserve_bytes = 0,
            2 => host.journal_reserve_bytes = 0, _ => host.serialization_reserve_bytes = 0 }
        assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 128,
            JobInputMemory::CompletePopulation, &UntrustedEncoding::ByteFallback).is_err(), "axis {axis}");
    }
}

#[test]
fn the_private_factory_vocabulary_population_and_reader_cross_one_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4DocumentExtractionJobPlanner>();
    send::<ExtractionDocumentInput<Cursor<Vec<u8>>>>();
    send::<Charged<ExtractionDocumentInput<Cursor<Vec<u8>>>>>();
    send::<ExtractionDocumentExecution<Cursor<Vec<u8>>>>();
    let _entry = NlpEngine::job_int4_extract_document::<Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_extract_document_with_threads::<Cursor<Vec<u8>>>;
}

#[test]
fn durable_numerical_teams_cannot_acquire_unauthenticated_chunk_cohorts() {
    let (_, mut document, _, _) = fixture();
    for chunks in [1, 4] {
        document.mapping.reduction.map_batch_chunks = chunks;
        check_team_schedule(None, false, document).unwrap();
        check_team_schedule(None, true, document).unwrap();
        for width in [2, 4, 64] {
            assert!(matches!(check_team_schedule(Some(width), true, document), Err(HostedError::Limits(_))));
            assert_eq!(check_team_schedule(Some(width), false, document).is_ok(), chunks == 1);
        }
    }
}

mod team {
    use super::*;
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding},
        grammar::{CompileLimits, runtime::SourceRuntimeLimits},
        hosted::corpus::jobs::JobOpenMode, jobs::{JobId, JobSecret},
        native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};
    use std::{io::{self, BufRead, Read}, sync::atomic::{AtomicU64, AtomicUsize, Ordering}};
    const PREPARATION: u64 = 128 << 20;
    #[derive(Clone, Copy)]
    enum Action { Refuse, Cancel(DecodeCancellationKind), Panic }
    #[derive(Default)]
    struct Observation { reads: AtomicUsize, drops: AtomicUsize,
        input_at_drop: AtomicU64, workers_at_read: AtomicU64 }
    struct Reader { resources: Arc<EngineResources>, observed: Arc<Observation>,
        action: Action, stop: CancellationToken }
    impl Drop for Reader {
        fn drop(&mut self) {
            let snapshot = self.resources.memory_snapshot();
            self.observed.input_at_drop.store(snapshot.by_class.get(&MemoryClass::JobBuffers)
                .map_or(0, |value| value.committed_bytes + value.reserved_bytes), Ordering::SeqCst);
            self.observed.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl Read for Reader {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { self.fill_buf()?; Ok(0) }
    }
    impl BufRead for Reader {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.observed.reads.fetch_add(1, Ordering::SeqCst);
            self.observed.workers_at_read.store(self.resources.memory_snapshot().by_class
                .get(&MemoryClass::ActivationScratch).map_or(0, |v| v.committed_bytes), Ordering::SeqCst);
            match self.action {
                Action::Refuse => panic!("unadmitted document job read private input"),
                Action::Panic => panic!("injected document job reader panic"),
                Action::Cancel(cause) => self.stop.cancel(cause),
            }
            Ok(&[])
        }
        fn consume(&mut self, _: usize) {}
    }
    fn planner(model: &ResidentInt4) -> (Int4DocumentExtractionJobPlanner, SourceJobRequest, JobHostLimits) {
        let (budget, document, job, mut limits) = fixture();
        limits.preparation_reserve_bytes = PREPARATION;
        limits.native.run.max_elapsed = Duration::from_secs(60);
        limits.native.run.max_checkpoints = 100_000;
        let verification = document.verification;
        let lifetime = GroundingBudget {
            max_fields: verification.max_fields * job.max_attempts as usize,
            max_matches: verification.max_matches * job.max_attempts as usize,
            max_scan_steps: verification.max_scan_steps * job.max_attempts,
        };
        let request = ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(),
            grounding: ExtractionBatchGrounding::Structural, budget };
        let planner = Int4DocumentExtractionJobPlanner::for_model(model.model_identity(), budget,
            CompileLimits::default(), SourceRuntimeLimits::default(), request, document, job, lifetime).unwrap();
        let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-extraction-job-team-refusal"),
            key: JobSecret::from_bytes([3; 32]), job_id: JobId([4; 16]), limits: job,
            mode: JobOpenMode::Create, materialize: false };
        (planner, request, limits)
    }
    fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
        assert_eq!(&host.resources().memory_snapshot(), baseline);
        let closures = host.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
    }
    #[test]
    #[ignore = "controller-only fresh runtime: tiny weights prove admission/drain, never native or durable-result parity"]
    fn extraction_job_teams_keep_population_charged_through_refusal_cancel_and_panic() {
        assert!(crate::installed_process_resources().is_none());
        let host = NlpEngine::builder().resource_config(ResourceHostConfig {
            runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
            max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
            helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 1024 << 20,
            leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
        }).build().unwrap();
        let idle = host.resources().memory_snapshot();
        let lease = host.resources().acquire_lease();
        let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
        let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
            || Ok(projection_fixture(WIDTH).0)).unwrap();
        let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
        let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
        let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls()).unwrap());
        let baseline = host.resources().memory_snapshot();
        use DecodeCancellationKind::*;
        for width in [2, 4] {
            for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
                ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
                let (planner, request, limits) = planner(&model);
                let charge = limits.required_buffer_bytes(request.limits).unwrap();
                let stop = CancellationToken::default(); stop.cancel(cause);
                let observed = Arc::new(Observation::default());
                let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                    action: Action::Refuse, stop: stop.clone() };
                let error = host.job_int4_extract_document_with_threads(&model, planner, Arc::clone(&vocabulary),
                    request, limits, 65536, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
                let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                    else { panic!("extraction pre-cancellation lost its typed cause"); };
                assert_eq!(stop.kind, cause); assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
                assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
                drained(&host, &baseline);
            }
        }
        for width in [0, 1, 5, 65, usize::MAX] {
            let (planner, request, limits) = planner(&model);
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action: Action::Refuse, stop: stop.clone() };
            let error = host.job_int4_extract_document_with_threads(&model, planner, Arc::clone(&vocabulary),
                request, limits, 65536, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
            assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            drained(&host, &baseline);
        }
        for action in [Action::Cancel(Deadline), Action::Panic] {
            let (planner, request, limits) = planner(&model);
            let charge = limits.required_buffer_bytes(request.limits).unwrap();
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action, stop: stop.clone() };
            let error = host.job_int4_extract_document_with_threads(&model, planner, Arc::clone(&vocabulary),
                request, limits, 65536, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
            match action {
                Action::Cancel(cause) => {
                    let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                        else { panic!("extraction reader cancellation lost its typed cause"); };
                    assert_eq!(stop.kind, cause);
                }
                Action::Panic => assert!(matches!(error,
                    HostedJobError::Host(HostedError::Panicked | HostedError::Scope { .. }))),
                Action::Refuse => unreachable!(),
            }
            assert_eq!(observed.reads.load(Ordering::SeqCst), 1);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            assert!(observed.workers_at_read.load(Ordering::SeqCst) >= int4_task_team_workspace_bytes(4, WIDTH).unwrap());
            assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
            drained(&host, &baseline);
        }
        drop(vocabulary); drop(model); drop(preparation);
        drained(&host, &idle);
    }
}


#[test]
fn selected_document_jobs_preserve_the_exact_planned_ceiling_across_serial_and_cohort_admission() {
    let (budget, document, _, _) = fixture(); let base = identity(); let ids = [identity(), identity()];
    for slots in [1, 4] {
        let peak = slots * budget.max_kv_bytes;
        let dense = request(&ids, document, peak);
        check_request(WIDTH, &base, peak, document, 65536, 16, None, &dense).unwrap();
        for cap in [1, 7, 32, 65, NANBEIGE_VOCAB_SIZE] {
            let selected = Some(Int4JsonSparseLimits { max_rows_per_step: cap });
            let one = Int4Work::for_sequence(WIDTH, 0, 64, 16 * cap).unwrap();
            let mut claim = request(&ids, document, peak);
            claim.model_work = one.checked_add(one).unwrap();
            assert_eq!(claim.model_work.projected_logits, claim.max_output_tokens * cap as u64);
            assert!(claim.model_work.forward_positions > document.mapping.chunks.context_tokens as u64);
            assert_eq!(claim.max_result_bytes, dense.max_result_bytes);
            assert_eq!(claim.max_output_tokens, dense.max_output_tokens);
            if cap == 32 { assert_ne!(claim.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0); }
            check_request(WIDTH, &base, peak, document, 65536, 16, selected, &claim).unwrap();
            if cap != NANBEIGE_VOCAB_SIZE {
                assert_eq!(check_request(WIDTH, &base, peak, document, 65536, 16, None, &claim)
                    .unwrap_err().fault.code, BatchCode::InvalidExecution);
                assert_eq!(check_request(WIDTH, &base, peak, document, 65536, 16, selected, &dense)
                    .unwrap_err().fault.code, BatchCode::InvalidExecution);
            } else { assert_eq!(claim.model_work, dense.model_work); }
            let other = Some(Int4JsonSparseLimits { max_rows_per_step: if cap == 1 { 2 } else { cap - 1 } });
            assert!(check_request(WIDTH, &base, peak, document, 65536, 16, other, &claim).is_err());
        }
    }
}

#[test]
fn selected_job_admission_refuses_short_divisible_work_overflow_and_renewed_output_authority() {
    let (budget, document, _, _) = fixture(); let base = identity(); let ids = [identity(), identity()];
    let selected = Some(Int4JsonSparseLimits { max_rows_per_step: 32 });
    let one = Int4Work::for_sequence(WIDTH, 0, 64, 16 * 32).unwrap();
    for slots in [1, 4] {
        let peak = slots * budget.max_kv_bytes;
        for delta in [-32_i64, -1, 1, 32] {
            let mut claim = request(&ids, document, peak); claim.model_work = one.checked_add(one).unwrap();
            claim.model_work.projected_logits = claim.model_work.projected_logits.checked_add_signed(delta).unwrap();
            if delta == -32 { claim.max_output_tokens -= 1; }
            let error = check_request(WIDTH, &base, peak, document, 65536, 16, selected, &claim).unwrap_err();
            assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
        }
        let mut claim = request(&ids, document, peak); claim.model_work = one.checked_add(one).unwrap();
        for cap in [0, NANBEIGE_VOCAB_SIZE + 1] {
            let invalid = Some(Int4JsonSparseLimits { max_rows_per_step: cap });
            assert_eq!(check_request(WIDTH, &base, peak, document, 65536, 16, invalid, &claim)
                .unwrap_err().fault.code, BatchCode::Admission);
        }
        for tokens in [0, u64::MAX, u64::MAX / 2] {
            assert_eq!(check_request(WIDTH, &base, peak, document, 65536, tokens, selected, &claim)
                .unwrap_err().fault.code, BatchCode::InvalidExecution);
        }
        claim.max_output_tokens -= 1;
        assert_eq!(check_request(WIDTH, &base, peak, document, 65536, 16, selected, &claim)
            .unwrap_err().fault.code, BatchCode::OutputLineLimit);
        let mut dense = request(&ids, document, peak); dense.model_work.projected_logits -= NANBEIGE_VOCAB_SIZE as u64;
        assert_eq!(check_request(WIDTH, &base, peak, document, 65536, 16, None, &dense)
            .unwrap_err().fault.code, BatchCode::InvalidExecution);
    }
}
