//! Pure durable-host contracts and owned type boundaries. No native model,
//! fabricated model authority, execution parity or storage qualification.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    jobs::{JobWork, population::PopulationReadLimits},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::grouped_int4::long::question::synthesis::{QuestionSynthesisLimits,
            hierarchy::QuestionHierarchyLimits}},
    validation::grounded_fields::GroundingBudget,
};
use std::{io::Cursor, time::Duration};

const WIDTH: usize = 127;
fn fixture(mode: usize) -> (Int4QuestionDocumentBatchConfig, JobLimits, JobHostLimits) {
    let row = constrained_int4::planned_work(WIDTH, 64, 64).unwrap();
    let mut item = row;
    for _ in 1..8 { item = item.checked_add(row).unwrap(); }
    let mut run = item;
    for _ in 1..4 { run = run.checked_add(item).unwrap(); }
    let verification = GroundingBudget { max_fields: 32, max_matches: 128, max_scan_steps: 1_000_000 };
    let lifetime = GroundingBudget { max_fields: 128, max_matches: 512, max_scan_steps: 4_000_000 };
    let config = Int4QuestionDocumentBatchConfig {
        options: Default::default(),
        budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 64, max_output_bytes: 65_536,
            max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(128, WIDTH).unwrap().kv_bytes },
        planning: SourcePlanningLimits { max_input_bytes: 8192, max_context_tokens: 128,
            max_passages: 4, ..Default::default() },
        mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 128, max_chunk_tokens: 64,
                context_tokens: 128, reserved_tokens: 64, max_chunks: 3, ..Default::default() },
            reduction: ExecutionLimits { map_batch_chunks: 1, max_value_bytes: 65_536,
                max_live_value_bytes: 131_072, max_result_bytes: 262_144, ..Default::default() },
            max_model_work: item, mask_limits: Default::default(), mask_visits_per_chunk: 1000,
            max_mask_visits: 8000, verification,
        },
        verification, synthesis: (mode > 0).then_some(QuestionSynthesisLimits {
            max_evidence_passages: 4, max_evidence_bytes: 1024 }),
        hierarchy: (mode > 1).then_some(QuestionHierarchyLimits { max_passes: 4,
            ..QuestionHierarchyLimits::default() }),
        max_question_bytes: 4096, max_model_work: run, max_mask_visits: 32_000,
        max_map_verification: lifetime, max_verification: lifetime,
    };
    let job = JobLimits { max_items: 3, max_id_bytes: 4096, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 1 << 20, max_spool_bytes: 8 << 20,
        max_materialized_bytes: 8 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(run, 32_000) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 131_072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (config, job, host)
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"question-document-job-host-metadata-only-fixture");
    ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH,
        )).unwrap()), tokenizer_digest: d, template_digest: d, task_spec: ANSWER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }
}
fn request<'a>(identities: &'a [ExecutionIdentity], capacity: QuestionCapacity) -> Int4QuestionDocumentAdmission<'a> {
    let contexts = identities.len() + capacity.config.continuation_passes();
    let row = constrained_int4::planned_work(WIDTH, 64, 64).unwrap();
    let mut work = Int4Work::default();
    for _ in 0..contexts { work = work.checked_add(row).unwrap(); }
    Int4QuestionDocumentAdmission { identities, model_work: work,
        mask_node_visits: capacity.config.mapping.mask_visits_per_chunk * contexts as u64,
        mask_limits: capacity.config.mapping.mask_limits, kv_reservation_bytes: capacity.kv_bytes,
        max_result_bytes: capacity.max_result_bytes, max_output_tokens: capacity.max_output_tokens,
        chunk_count: 3, native_chunks: identities.len() }
}

#[test]
fn shared_quote_matches_job_host_population_and_independent_native_phase_peaks() {
    let row = Int4MemoryRequirement::for_context(128, WIDTH).unwrap();
    for mode in 0..3 {
        for chunks in [None, Some(1), Some(3)] {
            for groups in [None, Some(1), Some(2), Some(4)] {
                if mode != 2 && groups.is_some() { continue; }
                let (mut config, job, host) = fixture(mode);
                config.mapping.reduction.map_batch_chunks = chunks.unwrap_or(1);
                let actual = int4_question_document_job_requirements(WIDTH, config, job, host,
                    1 << 20, chunks, groups, JobInputMemory::CompletePopulation).unwrap();
                let expected = Int4QuestionDocumentJobMemoryLimits {
                    document: Int4QuestionDocumentMemoryLimits { context_tokens: host.native.context_tokens,
                        allocator_reserve_bytes: host.native.allocator_reserve_bytes,
                        preparation_reserve_bytes: host.preparation_reserve_bytes,
                        io_reserve_bytes: host.io_reserve_bytes, reduction_reserve_bytes: 1 << 20,
                        cohort_chunks: chunks, hierarchy_groups: groups },
                    transport: host.transport, journal_reserve_bytes: host.journal_reserve_bytes,
                    serialization_reserve_bytes: host.serialization_reserve_bytes,
                }.requirements(WIDTH, config, job, JobInputMemory::CompletePopulation).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual.population_buffer_bytes, host.required_buffer_bytes(job).unwrap());
                assert_eq!(actual.document.native.kv_bytes,
                    row.kv_bytes * chunks.unwrap_or(1).max(groups.unwrap_or(1)) as u64);
                assert!(actual.document.input_bytes > actual.population_buffer_bytes);
                assert_eq!(actual.document.max_output_bytes, config.mapping.reduction.max_result_bytes as u64);
            }
        }
    }
}

#[test]
fn durable_output_admission_keeps_every_hierarchy_transcript_and_exact_result_cap() {
    for mode in 0..3 {
        let (mut config, job, host) = fixture(mode); config.mapping.reduction.map_batch_chunks = 3;
        let groups = (mode == 2).then_some(2);
        let quote = int4_question_document_job_requirements(WIDTH, config, job, host,
            1 << 20, Some(3), groups, JobInputMemory::CompletePopulation).unwrap().document;
        let capacity = QuestionCapacity { config, kv_bytes: quote.native.kv_bytes,
            max_result_bytes: quote.max_output_bytes, max_output_tokens: quote.max_output_tokens };
        let base = identity(); let identities = [base.clone()];
        check_request(WIDTH, &base, capacity, &request(&identities, capacity)).unwrap();
        assert_eq!(quote.max_output_tokens, if mode == 2 { 4 * 64 } else { 0 });
        assert_eq!(quote.output_reservation_bytes, quote.max_output_bytes * 4 + quote.max_output_tokens * 8 + 4096);
        for field in 0..3 {
            let mut changed = request(&identities, capacity);
            match field { 0 => changed.max_result_bytes += 4096, 1 => changed.max_output_tokens += 1,
                _ => changed.kv_reservation_bytes = config.budget.max_kv_bytes }
            let failure = check_request(WIDTH, &base, capacity, &changed).unwrap_err();
            assert!(failure.stop);
            assert_eq!(failure.fault.code, if field == 2 { BatchCode::Admission } else { BatchCode::OutputLineLimit });
        }
        let mut wrong = base.clone(); wrong.task_spec = "ner-v1".to_owned();
        let wrong = [wrong];
        assert_eq!(check_request(WIDTH, &base, capacity, &request(&wrong, capacity)).unwrap_err().fault.code,
            BatchCode::Admission);
    }
}

#[test]
fn durable_quote_rejects_work_verification_and_run_limits_before_population_io() {
    let (config, job, host) = fixture(2);
    for field in 0..6 {
        let mut changed = job;
        match field { 0 => changed.max_work.model.forward_positions -= 1,
            1 => changed.max_work.model.projected_logits -= 1,
            2 => changed.max_work.model.attention_pairs -= 1,
            3 => changed.max_work.model.projections.dot_products -= 1,
            4 => changed.max_work.model.projections.multiply_accumulates -= 1,
            _ => changed.max_work.mask_node_visits -= 1 }
        assert!(int4_question_document_job_requirements(WIDTH, config, changed, host, 1 << 20, None, None, JobInputMemory::CompletePopulation).is_err());
    }
    for field in 0..6 {
        let mut changed = config;
        let pool = if field < 3 { &mut changed.max_map_verification } else { &mut changed.max_verification };
        match field % 3 { 0 => pool.max_fields -= 1, 1 => pool.max_matches -= 1, _ => pool.max_scan_steps -= 1 }
        assert!(int4_question_document_job_requirements(WIDTH, changed, job, host, 1 << 20, None, None, JobInputMemory::CompletePopulation).is_err());
    }
    let mut changed = job; changed.max_attempts += 1;
    assert!(int4_question_document_job_requirements(WIDTH, config, changed, host, 1 << 20, None, None, JobInputMemory::CompletePopulation).is_err());
    let mut changed = host; changed.native.run.max_checkpoints = 0;
    assert!(int4_question_document_job_requirements(WIDTH, config, job, changed, 1 << 20, None, None, JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn separate_decoded_maxima_do_not_replace_encoded_population_or_host_ownership_caps() {
    let (mut config, job, host) = fixture(1);
    config.mapping.chunks.max_input_bytes = job.max_input_bytes_per_item;
    config.max_question_bytes = job.max_input_bytes_per_item;
    let original = int4_question_document_job_requirements(WIDTH, config, job, host, 1 << 20, None, None, JobInputMemory::CompletePopulation).unwrap();
    let mut larger = job; larger.max_snapshot_bytes += 4096;
    let larger = int4_question_document_job_requirements(WIDTH, config, larger, host, 1 << 20, None, None, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(larger.document.input_bytes - original.document.input_bytes, 8192);
    assert_eq!(larger.population_buffer_bytes - original.population_buffer_bytes, 8192);
    assert_eq!(larger.document.native, original.document.native);
    for field in 0..7 {
        let mut changed = host;
        match field { 0 => changed.native.context_tokens -= 1, 1 => changed.journal_reserve_bytes = 0,
            2 => changed.serialization_reserve_bytes = 0, 3 => changed.preparation_reserve_bytes = u64::MAX,
            4 => changed.io_reserve_bytes = u64::MAX, 5 => changed.journal_reserve_bytes = u64::MAX,
            _ => changed.transport.max_stream_bytes = 0 }
        assert!(int4_question_document_job_requirements(WIDTH, config, job, changed, 1 << 20, None, None, JobInputMemory::CompletePopulation).is_err());
    }
}

#[test]
fn private_planner_request_reader_and_guards_cross_the_owned_send_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4QuestionDocumentJobPlanner>();
    send::<SourceJobRequest>();
    send::<QuestionJobInput<Cursor<Vec<u8>>>>();
    send::<Charged<QuestionJobInput<Cursor<Vec<u8>>>>>();
    let _host = NlpEngine::job_int4_question_documents::<Cursor<Vec<u8>>>;
}
