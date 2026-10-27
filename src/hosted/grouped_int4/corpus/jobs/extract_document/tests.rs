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
        mask_limits: document.mapping.mask_limits, kv_reservation_bytes: kv,
        max_result_bytes: 65536, max_output_tokens: 32, chunk_count: 2 }
}

#[test]
fn whole_population_two_schema_owners_and_maximum_frontier_are_admitted_together() {
    let (budget, document, job, host) = fixture();
    let frontier = document_workspace_bytes(budget, document, 4, 65536).unwrap();
    let need = requirements(WIDTH, budget, document, 65536, job, host, frontier, 8192).unwrap();
    assert_eq!(need.input_bytes, host.required_buffer_bytes(job).unwrap() + 2 * 8192);
    assert_eq!(need.document_workspace_bytes, frontier);
    assert_eq!(need.native.kv_bytes, budget.max_kv_bytes);
    assert_eq!(need.native_workspace_bytes, need.native.rope_bytes + need.native.scratch_payload_bound
        + host.native.allocator_reserve_bytes);
}

#[test]
fn exact_resident_context_kv_and_nonzero_schema_are_required_before_input() {
    for axis in 0..5 {
        let (mut budget, mut document, job, mut host) = fixture();
        match axis { 0 => budget.max_kv_bytes -= 1, 1 => budget.max_kv_bytes += 1,
            2 => document.mapping.chunks.context_tokens -= 1, 3 => host.native.context_tokens += 1,
            _ => host.native.allocator_reserve_bytes = 0 }
        assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 128).is_err(), "axis {axis}");
    }
    let (budget, document, job, host) = fixture();
    assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 0).is_err());
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
        assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 128).is_err(), "axis {axis}");
    }
}

#[test]
fn many_native_sequences_can_exceed_one_context_without_changing_the_group_work() {
    let (budget, document, _, _) = fixture(); let base = identity(); let ids = [identity(), identity()];
    let req = request(&ids, document, budget.max_kv_bytes);
    assert!(req.model_work.forward_positions > document.mapping.chunks.context_tokens as u64);
    assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16, &req).is_ok());
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
        assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16,
            &request(&ids, document, budget.max_kv_bytes)).is_err(), "axis {axis}");
    }
    let ids = [identity()];
    assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16,
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
        assert!(check_request(WIDTH, &base, budget.max_kv_bytes, document, 65536, 16, &req).is_err(), "axis {axis}");
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
    assert!(requirements(WIDTH, budget, document, 65536, job, host, 0, 128).is_err());
    assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, usize::MAX).is_err());
    assert!(requirements(WIDTH, budget, document, job.max_result_bytes as u64 + 1,
        job, host, 65536, 128).is_err());
    for chunks in [0, 5, 257, usize::MAX] {
        assert!(document_workspace_bytes(budget, document, chunks, 65536).is_err());
    }
    for axis in 0..4 {
        let mut host = host;
        match axis { 0 => host.preparation_reserve_bytes = 0, 1 => host.io_reserve_bytes = 0,
            2 => host.journal_reserve_bytes = 0, _ => host.serialization_reserve_bytes = 0 }
        assert!(requirements(WIDTH, budget, document, 65536, job, host, 65536, 128).is_err(), "axis {axis}");
    }
}

#[test]
fn the_private_factory_vocabulary_population_and_reader_cross_one_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4DocumentExtractionJobPlanner>();
    send::<ExtractionDocumentInput<Cursor<Vec<u8>>>>();
    send::<Charged<ExtractionDocumentInput<Cursor<Vec<u8>>>>>();
    let _entry = NlpEngine::job_int4_extract_document::<Cursor<Vec<u8>>>;
}
