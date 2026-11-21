//! Declaration, admission and ownership regression definitions; no model
//! output or successful durable execution is synthesized by these fixtures.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::extract::{ExtractionBatchGrounding, grouped_int4::long::{
        Int4ExtractionMapLimits, Int4ExtractionMappingLimits}},
    execution_identity::{NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::source_index::SourceLanguageLimits,
    native_engine::{constrained_int4::planned_work, generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::{ProjectionWork, int4::convert::GROUPED_INT4_RECIPE},
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{extract::{object_reconcile::RootObjectLimits, semantic::SemanticVerificationSpec},
        judge::{FaithfulnessAggregationPolicy, FaithfulnessPolicy},
        mapreduce::{ChunkLimits, ExecutionLimits}},
    tokenizer::untrusted::SafeBpeLimits,
    validation::grounded_fields::GroundingBudget,
};
use std::{io::Cursor, time::Duration};

const WIDTH: usize = 32;
const MIB: u64 = 1 << 20;
fn maximum() -> Int4Work {
    Int4Work { forward_positions: u64::MAX / 8, projected_logits: u64::MAX / 8,
        attention_pairs: u64::MAX / 8, projections: ProjectionWork {
            dot_products: u64::MAX / 8, multiply_accumulates: u64::MAX / 8 } }
}
fn document_fixture() -> (ExtractionBatchArgs, Int4SemanticOptions, Int4SemanticObjectDocumentHostLimits) {
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

fn fixture() -> (Int4SemanticObjectDocumentJobConfig, JobLimits, Int4SemanticObjectDocumentJobHostLimits) {
    let (extraction, options, document) = document_fixture();
    let per = document.extraction.mapping.verification;
    let work = Int4SemanticObjectDocumentBatchLimits {
        max_model_work: maximum().checked_add(maximum()).unwrap(),
        max_mask_node_visits: document.extraction.mapping.mapping.max_mask_visits * 4,
        max_verification: GroundingBudget { max_fields: per.max_fields * 4,
            max_matches: per.max_matches * 4, max_scan_steps: per.max_scan_steps * 4 },
    };
    let config = Int4SemanticObjectDocumentJobConfig { extraction,
        compiler_limits: document.compiler, source_limits: document.source, options,
        judge_planning: document.judge_planning, encoding: UntrustedEncoding::ByteFallback,
        extraction_limits: document.extraction, review_limits: document.review, work };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 131_072, max_spool_bytes: MIB,
        max_materialized_bytes: MIB, max_journal_bytes: MIB, max_attempts: 4,
        max_work: JobWork::from_int4(work.max_model_work, work.max_mask_node_visits) };
    let host = Int4SemanticObjectDocumentJobHostLimits { document,
        transport: PopulationReadLimits { max_stream_bytes: MIB, max_lines: 100 },
        io_reserve_bytes: 131_072, journal_reserve_bytes: MIB, serialization_reserve_bytes: MIB };
    (config, job, host)
}
fn quote(config: &Int4SemanticObjectDocumentJobConfig, job: JobLimits,
    host: Int4SemanticObjectDocumentJobHostLimits, input: JobInputMemory, threads: Option<usize>)
    -> Result<Int4SemanticObjectDocumentJobRequirements, HostedJobError> {
    int4_semantic_object_job_requirements(WIDTH, config.extraction.budget, &config.options,
        &config.encoding, config.work, host, job, input, threads)
}

#[test]
fn every_document_owner_and_durable_staging_coexist_without_double_charging_shared_preparation() {
    let (config, job, host) = fixture();
    let document = int4_semantic_object_document_requirements(WIDTH, config.extraction.budget,
        &config.options, &config.encoding, host.document, None).unwrap();
    let overlap = 2 * host.document.compiler.max_schema_bytes as u64
        + 2 * options_storage_bytes(&config.options).unwrap()
        + 2 * SEMANTIC_OBJECT_JOB_RECIPE_BYTES as u64 + 1;
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let required = quote(&config, job, host, input, None).unwrap();
        assert_eq!(required.native, document.native);
        assert_eq!(required.workspace_bytes, document.workspace_bytes);
        assert_eq!(required.preparation_bytes, document.preparation_bytes + overlap);
        assert_eq!(required.input_bytes, host.storage().required_buffer_bytes_for_input(job, input).unwrap()
            - host.document.preparation_reserve_bytes + document.preparation_bytes + overlap);
        assert_eq!(required.output_reservation_bytes, document.output_reservation_bytes);
        assert!(required.output_reservation_bytes >= 8 * config.options.limits.max_output_bytes);
        assert_eq!(required.team_workspace_bytes, 0);
        assert_eq!(required.total_bytes().unwrap(), required.native.kv_bytes + required.workspace_bytes
            + required.input_bytes + required.output_reservation_bytes);
    }
    let complete = quote(&config, job, host, JobInputMemory::CompletePopulation, None).unwrap();
    let retained = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    assert_eq!(complete.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(complete.preparation_bytes, retained.preparation_bytes);
    assert_eq!(complete.output_reservation_bytes, retained.output_reservation_bytes);
}

#[test]
fn one_encoder_and_one_persistent_team_do_not_multiply_with_fields_or_chunks() {
    let (mut config, job, mut host) = fixture();
    let encoding = SafeBpeLimits { max_input_bytes: 16384, max_output_tokens: 8192,
        max_pair_checks: 1_000_000, max_pair_bytes: 4_000_000 };
    for fields in [1, 3, 16] {
        for chunks in [1, 4, 16] {
            config.options.limits.max_fields = fields;
            host.document.extraction.mapping.mapping.chunks.max_chunks = chunks;
            config.encoding = UntrustedEncoding::ByteFallback;
            let byte = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
            config.encoding = UntrustedEncoding::SafeBpeV1(encoding);
            let compact = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
            let workspace = encoding.requirements().unwrap().workspace_bytes;
            assert_eq!(compact.preparation_bytes - byte.preparation_bytes, workspace);
            assert_eq!(compact.input_bytes - byte.input_bytes, workspace);
            assert_eq!(compact.native, byte.native);
            assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
            for threads in [2, 4, 64] {
                let team = quote(&config, job, host, JobInputMemory::RetainedRecord, Some(threads)).unwrap();
                let extra = int4_task_team_workspace_bytes(threads, WIDTH).unwrap();
                assert_eq!(team.team_workspace_bytes, extra);
                assert_eq!(team.input_bytes, compact.input_bytes);
                assert_eq!(team.total_bytes().unwrap() - compact.total_bytes().unwrap(), extra);
            }
        }
    }
}

#[test]
fn all_native_mask_and_every_attempt_grounding_ceilings_refuse_before_private_io() {
    let (config, job, host) = fixture();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        for axis in 0..6 {
            let mut altered = job;
            match axis {
                0 => altered.max_work.model.forward_positions += 1,
                1 => altered.max_work.model.projected_logits += 1,
                2 => altered.max_work.model.attention_pairs += 1,
                3 => altered.max_work.model.projections.dot_products += 1,
                4 => altered.max_work.model.projections.multiply_accumulates += 1,
                _ => altered.max_work.mask_node_visits += 1,
            }
            assert!(quote(&config, altered, host, input, None).is_err(), "axis {axis}");
        }
        for axis in 0..3 {
            let mut altered = config.clone();
            match axis {
                0 => altered.work.max_verification.max_fields -= 1,
                1 => altered.work.max_verification.max_matches -= 1,
                _ => altered.work.max_verification.max_scan_steps -= 1,
            }
            // Still enough for one item; the missing unit belongs to a later
            // permitted attempt and must not be renewed by authenticated resume.
            altered.work.validate_item(altered.options.max_model_work,
                altered.extraction_limits.mapping.mapping.max_mask_visits,
                altered.extraction_limits.mapping.verification).unwrap();
            assert!(quote(&altered, job, host, input, None).is_err(), "grounding axis {axis}");
            assert!(check_durable_limits(&altered, job).is_err());
        }
        let mut too_small = job; too_small.max_result_bytes -= 1;
        assert!(quote(&config, too_small, host, input, None).is_err());
        let mut retry = job; retry.max_attempts += 1;
        assert!(quote(&config, retry, host, input, None).is_err());
        let mut overflow = job; overflow.max_attempts = u64::MAX;
        assert!(quote(&config, overflow, host, input, None).is_err());
    }
}

#[test]
fn fixed_factory_configuration_must_match_the_entire_admitted_host() {
    let (config, job, host) = fixture();
    check_config(&config, host, job).unwrap();
    for axis in 0..10 {
        let mut altered = config.clone();
        match axis {
            0 => altered.compiler_limits.max_schema_bytes += 1,
            1 => altered.source_limits.index.max_index_bytes += 1,
            2 => altered.source_limits.verification.max_scan_steps += 1,
            3 => altered.extraction_limits.mapping.mapping.chunks.max_chunks += 1,
            4 => altered.extraction_limits.mapping.mapping.reduction.max_result_bytes += 1,
            5 => altered.extraction_limits.reconciliation.max_origins += 1,
            6 => altered.review_limits.chunks.max_input_bytes += 1,
            7 => altered.review_limits.aggregation.minimum_classified_coverage_ppm += 1,
            8 => altered.judge_planning.per_head.max_nodes += 1,
            _ => altered.extraction.schema.clear(),
        }
        assert!(check_config(&altered, host, job).is_err(), "axis {axis}");
    }
}

#[test]
fn explicit_reserves_and_checked_schema_recipe_overlap_cannot_be_waived() {
    let (config, job, host) = fixture();
    for axis in 0..10 {
        let mut altered = host;
        match axis {
            0 => altered.document.preparation_reserve_bytes = 0,
            1 => altered.io_reserve_bytes = 0,
            2 => altered.journal_reserve_bytes = 0,
            3 => altered.serialization_reserve_bytes = 0,
            4 => altered.document.native.allocator_reserve_bytes = 0,
            5 => altered.document.preparation_reserve_bytes = u64::MAX,
            6 => altered.io_reserve_bytes = u64::MAX,
            7 => altered.journal_reserve_bytes = u64::MAX,
            8 => altered.serialization_reserve_bytes = u64::MAX,
            _ => altered.document.review.chunks.max_input_bytes -= 1,
        }
        assert!(quote(&config, job, altered, JobInputMemory::RetainedRecord, None).is_err(), "axis {axis}");
    }
    assert!(preparation_with_recipe(u64::MAX, config.compiler_limits, &config.options).is_err());
    let floor = preparation_with_recipe(0, config.compiler_limits, &config.options).unwrap();
    assert_eq!(preparation_with_recipe(u64::MAX - floor, config.compiler_limits, &config.options).unwrap(), u64::MAX);
    assert!(preparation_with_recipe(u64::MAX - floor + 1, config.compiler_limits, &config.options).is_err());
    for threads in [0, 65, usize::MAX] {
        assert!(quote(&config, job, host, JobInputMemory::RetainedRecord, Some(threads)).is_err());
    }
    let mut capacity = config.clone();
    let small = quote(&capacity, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    capacity.options.specification.revision.reserve(65_536);
    let large = quote(&capacity, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    assert!(large.preparation_bytes > small.preparation_bytes);
    assert_eq!(large.output_reservation_bytes, small.output_reservation_bytes);
}

fn identity(task: &str) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"semantic-object-job-admission-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, WIDTH)).unwrap()),
        tokenizer_digest: digest, template_digest: digest, task_spec: task.to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "fixture".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}

#[test]
fn complete_document_admission_counts_chunks_without_confusing_total_work_with_one_context() {
    let (config, job, host) = fixture();
    let required = quote(&config, job, host, JobInputMemory::RetainedRecord, None).unwrap();
    let identities = vec![identity("extract-v1"); 4];
    let judge = identity("judge-v1");
    let one = planned_work(WIDTH, 200, 8).unwrap();
    let work = one.checked_add(one).unwrap().checked_add(one).unwrap().checked_add(one).unwrap();
    assert!(work.forward_positions > host.document.native.context_tokens as u64);
    let make = || Int4SemanticObjectDocumentAdmission {
        identities: &identities, judge_identity: &judge, group_width: WIDTH,
        model_work: config.options.max_model_work, extraction_reserved_work: work,
        mask_node_visits: config.extraction_limits.mapping.mapping.max_mask_visits,
        mask_limits: config.extraction_limits.mapping.mapping.mask_limits,
        verification_budget: config.extraction_limits.mapping.verification,
        kv_reservation_bytes: required.native.kv_bytes,
        max_result_bytes: config.options.limits.max_output_bytes,
        max_output_tokens: 4 * u64::from(config.extraction.budget.max_output_tokens),
        chunk_count: 4, source_bytes: config.extraction_limits.mapping.mapping.chunks.max_input_bytes,
        extraction_budget: config.extraction.budget, judge_budget: config.options.specification.judge_budget,
        extraction_limits: config.extraction_limits, review_limits: config.review_limits,
        semantic_limits: config.options.limits, judge_planning_limits: config.judge_planning,
        input_encoding: &config.encoding,
    };
    check_request(WIDTH, &config, host, job, required, &make()).unwrap();
    for axis in 0..23 {
        let mut request = make();
        match axis {
            0 => request.source_bytes += 1,
            1 => request.model_work.forward_positions -= 1,
            2 => request.mask_node_visits += 1,
            3 => request.mask_limits.checkpoint_interval_nodes += 1,
            4 => request.verification_budget.max_scan_steps += 1,
            5 => request.kv_reservation_bytes -= 1,
            6 => request.max_result_bytes -= 1,
            7 => request.max_output_tokens -= 1,
            8 => request.chunk_count -= 1,
            9 => request.identities = &identities[..3],
            10 => request.extraction_budget.max_output_tokens -= 1,
            11 => request.judge_budget.max_input_tokens -= 1,
            12 => request.extraction_limits.reconciliation.max_origins += 1,
            13 => request.review_limits.max_retained_result_bytes += 1,
            14 => request.semantic_limits.max_fields += 1,
            15 => request.judge_planning_limits.per_head.max_nodes += 1,
            16 => request.source_bytes = 0,
            17 => request.extraction_reserved_work.forward_positions = 0,
            18 => request.extraction_reserved_work.projected_logits = 0,
            19 => request.extraction_reserved_work.attention_pairs = 0,
            20 => request.extraction_reserved_work.projections.dot_products = 0,
            21 => request.extraction_reserved_work.projections.multiply_accumulates = 0,
            _ => request.group_width *= 2,
        }
        assert!(check_request(WIDTH, &config, host, job, required, &request).is_err(), "axis {axis}");
    }
    let foreign = vec![identity("judge-v1"); 4];
    let mut request = make(); request.identities = &foreign;
    assert!(check_request(WIDTH, &config, host, job, required, &request).is_err());
    for axis in 0..3 {
        let mut short = required;
        match axis {
            0 => short.preparation_bytes -= 1,
            1 => short.output_reservation_bytes -= 1,
            _ => short.workspace_bytes -= 1,
        }
        assert!(check_request(WIDTH, &config, host, job, short, &make()).is_err(), "owner {axis}");
    }
}

#[test]
fn actual_inputs_compilers_and_complete_result_claims_cross_the_owned_boundary() {
    fn send<T: Send + 'static>() {}
    send::<JobInput<Cursor<Vec<u8>>>>();
    send::<JobExecution<Cursor<Vec<u8>>>>();
    send::<DeferredDocumentJob<Cursor<Vec<u8>>>>();
    send::<Arc<JobOwners>>();
    send::<Arc<Int4SemanticObjectDocumentBatchPlanner>>();
    send::<JobPopulation>();
    let _serial = NlpEngine::job_int4_semantic_object_documents::<Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_semantic_object_documents_with_threads::<Cursor<Vec<u8>>>;
}

mod lifecycle;
