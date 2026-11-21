//! Pure ownership/refusal definitions. Runtime tests require a fresh process;
//! no fixture here manufactures a successful native extraction or review.
use super::*;
use crate::{
    batch::extract::{ExtractionBatchGrounding, grouped_int4::long::{Int4ExtractionMapLimits,
        Int4ExtractionMappingLimits}},
    grammar::source_index::SourceLanguageLimits,
    native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork},
    tasks::{extract::{object_reconcile::RootObjectLimits, semantic::SemanticVerificationSpec},
        judge::{FaithfulnessAggregationPolicy, FaithfulnessPolicy},
        mapreduce::{ChunkLimits, ExecutionLimits}},
    tokenizer::untrusted::SafeBpeLimits,
    validation::grounded_fields::GroundingBudget,
};

const WIDTH: usize = 32;
const MIB: u64 = 1 << 20;
fn maximum() -> Int4Work {
    Int4Work { forward_positions: u64::MAX / 8, projected_logits: u64::MAX / 8,
        attention_pairs: u64::MAX / 8, projections: ProjectionWork {
            dot_products: u64::MAX / 8, multiply_accumulates: u64::MAX / 8 } }
}
fn fixture() -> (ExtractionBatchArgs, Int4SemanticOptions, Int4SemanticObjectDocumentHostLimits) {
    let kv = Int4MemoryRequirement::for_context(256, WIDTH).unwrap().kv_bytes;
    let budget = TaskBudget { max_input_tokens: 224, max_output_tokens: 32,
        max_output_bytes: 4096, max_grammar_states: 64, max_kv_bytes: kv };
    let judge_budget = TaskBudget { max_input_tokens: 240, max_output_tokens: 8,
        max_output_bytes: 16_384, max_grammar_states: 64, max_kv_bytes: kv };
    let options = Int4SemanticOptions {
        specification: SemanticVerificationSpec { schema_version: 1, experimental_opt_in: true,
            revision: "host-complete-object-review".to_owned(), claims: vec![], judge_budget,
            judge_policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: 0,
                evidence_window_bytes: 64, max_evidence_windows: 2, max_evidence_spans: 2 } },
        limits: SemanticLimits { max_fields: 3, max_claim_bytes: 128, max_total_claim_bytes: 1024,
            max_template_bytes: 1024, max_walk_nodes: 128, max_input_json_bytes: 8192,
            max_total_prompt_tokens: 8192, max_total_projected_logits: 1_000_000,
            max_output_bytes: 131_072 }, max_model_work: maximum(),
    };
    let chunks = ChunkLimits { max_input_bytes: 512, max_chunk_bytes: 64, max_chunk_tokens: 64,
        context_tokens: 256, reserved_tokens: 32, max_chunks: 4, max_tokenizer_calls: 128 };
    let grounding = GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 10_000 };
    let limits = Int4SemanticObjectDocumentHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 1_000_000,
                cleanup_reserve_bytes: 65_536 } },
        compiler: CompileLimits { max_schema_bytes: 1024, max_string_bytes: 256, max_array_items: 8,
            max_output_bytes: 4096, max_states: 64, max_transitions: 256, max_mask_bytes: 65_536 },
        source: SourceRuntimeLimits { index: SourceLanguageLimits { max_source_bytes: 64,
            max_index_bytes: 65_536, max_build_steps: 1_000_000, max_matches: 128 }, verification: grounding },
        extraction: Int4ObjectExtractionMapLimits {
            mapping: Int4ExtractionMapLimits { mapping: Int4ExtractionMappingLimits { chunks,
                reduction: ExecutionLimits { map_batch_chunks: 1, max_value_bytes: 8192,
                    max_live_value_bytes: 32768, max_total_value_bytes: 65536, max_result_bytes: 32768,
                    ..ExecutionLimits::default() }, max_model_work: maximum(),
                mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
                mask_visits_per_chunk: 32_000, max_mask_visits: 128_000 },
                verification: grounding, root_array_union: None },
            reconciliation: RootObjectLimits { max_properties: 16, max_origins: 64,
                max_input_bytes: 65536, max_json_bytes: 4096, max_nodes: 1024, max_result_bytes: 16384 },
        },
        judge_planning: JudgeLimits { max_total_prompt_tokens: 2048, max_total_candidate_tokens: 256,
            max_total_projected_logits: 1_000_000, max_output_bytes: 16_384, ..JudgeLimits::default() },
        review: Int4FaithfulnessMapLimits { chunks: ChunkLimits { max_input_bytes: 640, ..chunks },
            max_model_work: maximum(), max_retained_result_bytes: 65536, max_result_bytes: 65536,
            aggregation: FaithfulnessAggregationPolicy::default() },
        preparation_reserve_bytes: 128 * MIB, reduction_reserve_bytes: 65_536,
    };
    (ExtractionBatchArgs { schema: r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}"#.to_owned(),
        grounding: ExtractionBatchGrounding::Structural, budget }, options, limits)
}
fn quote(task: TaskBudget, options: &Int4SemanticOptions, limits: Int4SemanticObjectDocumentHostLimits)
    -> Result<Int4SemanticObjectDocumentRequirements, HostedError> {
    int4_semantic_object_document_requirements(WIDTH, task, options, &UntrustedEncoding::ByteFallback, limits, None)
}

#[test]
fn every_field_retains_its_complete_source_map_while_the_native_cache_is_shared() {
    let (request, mut options, limits) = fixture();
    options.limits.max_fields = 1;
    let one = quote(request.budget, &options, limits).unwrap();
    options.limits.max_fields = 4;
    let four = quote(request.budget, &options, limits).unwrap();
    let field = Int4FaithfulnessMapMemoryLimits { context_tokens: limits.native.context_tokens,
        preparation_reserve_bytes: limits.preparation_reserve_bytes,
        allocator_reserve_bytes: limits.native.allocator_reserve_bytes }
        .requirements_with_max_heads(WIDTH, options.specification.judge_budget, limits.judge_planning,
            limits.review, options.specification.judge_policy.max_evidence_windows as usize + 1).unwrap();
    let pointers = 2 * options.limits.max_input_json_bytes as u64;
    assert_eq!(four.preparation_bytes - one.preparation_bytes,
        3 * (field.input_bytes - limits.preparation_reserve_bytes + field.output_reservation_bytes + pointers));
    assert_eq!(one.native, four.native);
    assert_eq!(one.workspace_bytes, four.workspace_bytes);
    assert_eq!(one.output_reservation_bytes, four.output_reservation_bytes);
    assert!(four.preparation_bytes > 4 * field.output_reservation_bytes);
}

#[test]
fn retained_chunk_programs_do_not_recycle_one_partial_schema_or_source_index() {
    let (request, options, mut limits) = fixture();
    limits.extraction.mapping.mapping.chunks.max_chunks = 1;
    let one = quote(request.budget, &options, limits).unwrap();
    limits.extraction.mapping.mapping.chunks.max_chunks = 4;
    let four = quote(request.budget, &options, limits).unwrap();
    let per_chunk_floor = RootObjectSchema::workspace_bytes(limits.compiler).unwrap()
        + limits.source.index.max_index_bytes as u64 + limits.compiler.max_mask_bytes as u64;
    assert!(four.preparation_bytes - one.preparation_bytes >= 3 * per_chunk_floor);
    assert_eq!(one.native, four.native);
    assert_eq!(one.workspace_bytes, four.workspace_bytes);
    assert_eq!(four.output_reservation_bytes - one.output_reservation_bytes,
        3 * u64::from(request.budget.max_output_tokens) * 8);
}

#[test]
fn tiny_object_output_cannot_hide_large_optional_schema_owners() {
    let (request, options, mut limits) = fixture();
    limits.extraction.reconciliation.max_json_bytes = 2;
    limits.extraction.reconciliation.max_result_bytes = 2;
    let small = quote(request.budget, &options, limits).unwrap();
    limits.compiler.max_schema_bytes *= 2;
    let large = quote(request.budget, &options, limits).unwrap();
    assert!(large.preparation_bytes > small.preparation_bytes);
    assert_eq!(small.output_reservation_bytes, large.output_reservation_bytes);
    assert_eq!(small.native, large.native);
}

#[test]
fn one_encoder_and_one_persistent_team_add_to_all_retained_plans() {
    let (request, mut options, mut limits) = fixture();
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 16384,
        max_output_tokens: 8192, max_pair_checks: 1_000_000, max_pair_bytes: 4_000_000 });
    let UntrustedEncoding::SafeBpeV1(caps) = &encoding else { unreachable!() };
    for fields in [1, 3, 32] {
        for chunks in [1, 4, 16] {
            options.limits.max_fields = fields;
            limits.extraction.mapping.mapping.chunks.max_chunks = chunks;
            let byte = quote(request.budget, &options, limits).unwrap();
            let compact = int4_semantic_object_document_requirements(WIDTH, request.budget,
                &options, &encoding, limits, None).unwrap();
            assert_eq!(compact.preparation_bytes - byte.preparation_bytes, caps.requirements().unwrap().workspace_bytes);
            assert_eq!(compact.native, byte.native);
            assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
            assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
            for threads in [2, 4] {
                let team = int4_semantic_object_document_requirements(WIDTH, request.budget,
                    &options, &encoding, limits, Some(threads)).unwrap();
                let scratch = int4_task_team_workspace_bytes(threads, WIDTH).unwrap();
                assert_eq!(team.team_workspace_bytes, scratch);
                assert_eq!(team.total_bytes().unwrap() - compact.total_bytes().unwrap(), scratch);
                assert_eq!(team.preparation_bytes, compact.preparation_bytes);
            }
        }
    }
}

#[test]
fn both_stages_require_full_resident_kv_even_when_each_head_fits_less() {
    for stage in 0..2 {
        let (mut request, mut options, limits) = fixture();
        if stage == 0 { request.budget.max_kv_bytes -= 1; }
        else { options.specification.judge_budget.max_kv_bytes -= 1; }
        assert!(quote(request.budget, &options, limits).is_err());
    }
    let (request, options, limits) = fixture();
    assert_eq!(quote(request.budget, &options, limits).unwrap().native.kv_bytes, request.budget.max_kv_bytes);
    assert!(int4_semantic_object_document_requirements(0, request.budget, &options,
        &UntrustedEncoding::ByteFallback, limits, None).is_err());
}

#[test]
fn incomplete_source_review_unsupported_schedules_and_missing_reserves_refuse_purely() {
    for axis in 0..15 {
        let (request, mut options, mut limits) = fixture();
        match axis {
            0 => limits.review.chunks.max_input_bytes -= 1,
            1 => limits.extraction.mapping.mapping.reduction.map_batch_chunks = 2,
            2 => limits.preparation_reserve_bytes = 0,
            3 => limits.reduction_reserve_bytes = 0,
            4 => limits.native.allocator_reserve_bytes = 0,
            5 => limits.compiler.max_schema_bytes = 0,
            6 => limits.source.index.max_index_bytes = 0,
            7 => limits.extraction.mapping.mapping.mask_limits.checkpoint_interval_nodes = 0,
            8 => limits.extraction.mapping.mapping.max_mask_visits = 1,
            9 => options.specification.experimental_opt_in = false,
            10 => options.limits.max_fields = 0,
            11 => limits.judge_planning.max_total_candidate_tokens = 0,
            12 => options.max_model_work.forward_positions = 0,
            13 => options.limits.max_output_bytes = 2,
            _ => options.limits.max_input_json_bytes = 2,
        }
        assert!(quote(request.budget, &options, limits).is_err(), "axis {axis}");
    }
}

#[test]
fn independent_stage_caps_can_exceed_the_intersecting_whole_request_cap() {
    let (request, mut options, limits) = fixture();
    options.max_model_work.forward_positions = 1;
    // Only the real prepared reservation can decide whether execution fits.
    // Declaring a larger per-stage cap cannot grant another unit of work.
    assert!(quote(request.budget, &options, limits).is_ok());
}

#[test]
fn actual_private_policy_capacity_and_checked_arithmetic_cannot_escape_the_quote() {
    let (request, mut options, mut limits) = fixture();
    let small = quote(request.budget, &options, limits).unwrap();
    options.specification.revision.reserve(65_536);
    let capacity = quote(request.budget, &options, limits).unwrap();
    assert!(capacity.preparation_bytes > small.preparation_bytes);
    assert_eq!(capacity.output_reservation_bytes, small.output_reservation_bytes);
    limits.preparation_reserve_bytes = u64::MAX;
    assert!(quote(request.budget, &options, limits).is_err());
    let invalid = Int4SemanticObjectDocumentRequirements { preparation_bytes: u64::MAX, ..small };
    assert!(invalid.total_bytes().is_err());
}

#[test]
fn original_cancellation_cause_and_private_safe_error_chain_survive_host_mapping() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let error = document_error(Int4SemanticObjectDocumentError::Cancelled(cause));
        let HostedError::Int4(error) = error else { panic!("wrong host classification"); };
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
        let Int4HostError::SemanticObjectDocument(inner) = error else { panic!("lost typed document cause"); };
        assert_eq!(inner.cancellation(), Some(cause));
    }
}

#[test]
fn complete_real_inputs_and_result_are_owned_across_the_single_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<DocumentInput>();
    send::<DocumentExecution>();
    send::<DeferredDocument>();
    send::<Int4SemanticObjectDocumentRun>();
    let _serial = NlpEngine::extract_int4_semantic_object_document;
    let _team = NlpEngine::extract_int4_semantic_object_document_with_threads;
}

mod lifecycle;
