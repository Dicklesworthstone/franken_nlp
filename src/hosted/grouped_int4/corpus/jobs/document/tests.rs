//! Source-only fixtures for durable document admission and physical ownership.
//! UNRUN: no native inference, compiler or storage execution was performed.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::RunLimits,
    jobs::{JobWork, population::PopulationReadLimits},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{answer::AnswerOptions, mapreduce::{ChunkLimits, ExecutionLimits},
        source_planning::grouped_int4::long::question::{SourceQuestion,
            synthesis::{QuestionSynthesisLimits, SourceQuestionSynthesis,
                hierarchy::QuestionHierarchyLimits}}},
    validation::grounded_fields::GroundingBudget,
};
use std::{io::Cursor, time::Duration};

const WIDTH: usize = 32;
fn identity() -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"document-job-host-synthetic-admission");
    ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, WIDTH,
        )).unwrap()),
        tokenizer_digest: digest, template_digest: digest, task_spec: "answer-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "fixture".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "constrained-json".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None,
        compiler_identity: None,
    }
}
fn multiply_work(work: Int4Work, copies: u64) -> Int4Work {
    Int4Work { forward_positions: work.forward_positions * copies,
        projected_logits: work.projected_logits * copies,
        attention_pairs: work.attention_pairs * copies,
        projections: crate::native_engine::portable_int8::ProjectionWork {
            dot_products: work.projections.dot_products * copies,
            multiply_accumulates: work.projections.multiply_accumulates * copies,
        } }
}
fn fixture() -> (TaskBudget, SourcePlanningLimits, Int4SourceMapLimits, JobLimits, JobHostLimits) {
    let native = Int4MemoryRequirement::for_context(64, WIDTH).unwrap();
    let budget = TaskBudget { max_input_tokens: 48, max_output_tokens: 16,
        max_output_bytes: 4096, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
    let planning = SourcePlanningLimits { max_context_tokens: 64, ..Default::default() };
    let one = Int4Work::for_sequence(WIDTH, 0, 64, 16 * NANBEIGE_VOCAB_SIZE).unwrap();
    let mapping = Int4SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 1024,
            max_chunk_tokens: 48, context_tokens: 64, reserved_tokens: 16,
            max_chunks: 4, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 4,
            max_reduction_levels: 8, max_task_calls: 64, max_value_bytes: 8192,
            max_live_value_bytes: 65536, max_total_value_bytes: 1 << 20, max_result_bytes: 65536 },
        max_model_work: multiply_work(one, 8),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
        mask_visits_per_chunk: 4000, max_mask_visits: 32000,
        verification: GroundingBudget { max_fields: 64, max_matches: 128, max_scan_steps: 65536 },
    };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(mapping.max_model_work, mapping.max_mask_visits) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 131072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (budget, planning, mapping, job, host)
}
fn request<'a>(ids: &'a [ExecutionIdentity], mapping: Int4SourceMapLimits, kv: u64)
    -> Int4SourceDocumentAdmission<'a> {
    Int4SourceDocumentAdmission {
        identities: ids,
        model_work: multiply_work(Int4Work::for_sequence(WIDTH, 0, 64, 16 * NANBEIGE_VOCAB_SIZE).unwrap(), 2),
        mask_node_visits: 8000, mask_limits: mapping.mask_limits, kv_reservation_bytes: kv,
        max_result_bytes: 65536, max_output_tokens: 32, chunk_count: 2,
    }
}

#[test]
fn complete_population_question_frontier_and_native_workspace_are_reserved_together() {
    let (budget, planning, mapping, job, host) = fixture();
    let need = document_requirements(WIDTH, budget, planning, mapping, 65536,
        job, host, 123456, 8192, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(need.input_bytes, host.required_buffer_bytes(job).unwrap() + 8192);
    assert_eq!(need.document_workspace_bytes, 123456);
    assert_eq!(need.native.kv_bytes, budget.max_kv_bytes);
    assert_eq!(need.native_workspace_bytes, need.native.rope_bytes
        + need.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
}

#[test]
fn context_and_kv_must_match_the_immutable_factory_before_population_access() {
    for axis in 0..5 {
        let (mut budget, mut planning, mapping, job, mut host) = fixture();
        match axis {
            0 => budget.max_kv_bytes -= 1, 1 => budget.max_kv_bytes += 1,
            2 => planning.max_context_tokens -= 1, 3 => host.native.context_tokens += 1,
            _ => host.native.allocator_reserve_bytes = 0,
        }
        assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
            job, host, 65536, 0, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn cohort_phase_reservation_keeps_population_question_and_complete_frontier_charges() {
    use crate::tasks::source_planning::grouped_int4::long::cohort::Int4SourceMapCohortMemoryLimits;
    let (budget, planning, mapping, job, host) = fixture();
    let serial = document_requirements(WIDTH, budget, planning, mapping, 65536,
        job, host, 123456, 8192, JobInputMemory::CompletePopulation).unwrap();
    for chunks in [1, 2, 4] {
        let quote = Int4SourceMapCohortMemoryLimits { max_chunks: chunks, context_tokens: 64,
            max_output_tokens_per_chunk: 16, max_result_bytes_per_chunk: 4096 }.requirements(WIDTH).unwrap()
            .with_synthesis_context(64, WIDTH, budget.max_kv_bytes).unwrap();
        let mut grouped = serial;
        grouped.use_cohorts(quote, host.native.allocator_reserve_bytes).unwrap();
        assert_eq!(grouped.native, quote.native);
        assert_eq!(grouped.native.kv_bytes, chunks as u64 * budget.max_kv_bytes);
        assert_eq!(grouped.input_bytes, serial.input_bytes);
        assert_eq!(grouped.document_workspace_bytes, serial.document_workspace_bytes);
        assert_eq!(grouped.native_workspace_bytes, quote.native.rope_bytes
            + quote.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
        assert!(quote.workspace_bytes > 0); // Separately retained decoder reservation.
        assert_eq!(budget.max_kv_bytes, serial.native.kv_bytes);
    }
}

#[test]
fn partial_cohort_peaks_cannot_replace_complete_job_native_authority() {
    use crate::tasks::source_planning::grouped_int4::long::cohort::Int4SourceMapCohortMemoryLimits;
    let (budget, planning, mapping, job, host) = fixture();
    let serial = document_requirements(WIDTH, budget, planning, mapping, 65536,
        job, host, 123456, 8192, JobInputMemory::CompletePopulation).unwrap();
    let quote = Int4SourceMapCohortMemoryLimits { max_chunks: 4, context_tokens: 64,
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
fn cohort_admission_prices_the_reserved_window_even_for_a_short_document() {
    let (budget, _, mapping, _, _) = fixture(); let ids = [identity(), identity()];
    let peak = 4 * budget.max_kv_bytes;
    let mut claim = request(&ids, mapping, peak);
    check_document_request(WIDTH, "answer-v1", peak, mapping, 65536, 64, &claim).unwrap();
    claim.kv_reservation_bytes = budget.max_kv_bytes;
    assert!(check_document_request(WIDTH, "answer-v1", peak, mapping, 65536, 64, &claim).is_err());
}

#[test]
fn every_native_axis_and_masks_must_fit_the_persisted_lifetime_ceiling() {
    for axis in 0..6 {
        let (budget, planning, mapping, mut job, host) = fixture();
        match axis {
            0 => job.max_work.model.forward_positions -= 1,
            1 => job.max_work.model.projected_logits -= 1,
            2 => job.max_work.model.attention_pairs -= 1,
            3 => job.max_work.model.projections.dot_products -= 1,
            4 => job.max_work.model.projections.multiply_accumulates -= 1,
            _ => job.max_work.mask_node_visits -= 1,
        }
        assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
            job, host, 65536, 0, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn selected_summary_can_have_a_smaller_final_envelope_than_each_native_map() {
    let (budget, planning, mapping, mut job, host) = fixture();
    // The input population may choose a stricter independent source bound.
    job.max_input_bytes_per_item = 4096;
    assert!(document_requirements(WIDTH, budget, planning, mapping, 1024,
        job, host, 65536, 0, JobInputMemory::CompletePopulation).is_ok());
    assert!(document_requirements(WIDTH, budget, planning, mapping, job.max_result_bytes as u64 + 1,
        job, host, 65536, 0, JobInputMemory::CompletePopulation).is_err());
    assert!(document_requirements(WIDTH, budget, planning, mapping, 0,
        job, host, 65536, 0, JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn aggregate_document_work_is_not_mistaken_for_one_native_context() {
    let (budget, _, mapping, _, _) = fixture();
    let ids = [identity(), identity()];
    let req = request(&ids, mapping, budget.max_kv_bytes);
    assert!(req.model_work.forward_positions > 64);
    assert!(check_document_request(WIDTH, "answer-v1", budget.max_kv_bytes,
        mapping, 65536, 64, &req).is_ok());
}

#[test]
fn ranked_keyphrases_keep_complete_chunk_work_without_retained_native_tokens() {
    let (budget, _, mapping, _, _) = fixture();
    let mut id = identity(); id.task_spec = "keyphrases-v1".to_owned();
    let ids = [id.clone(), id];
    let mut req = request(&ids, mapping, budget.max_kv_bytes);
    req.max_output_tokens = 0;
    assert!(req.model_work.forward_positions > mapping.chunks.context_tokens as u64);
    assert!(check_document_request(WIDTH, "keyphrases-v1", budget.max_kv_bytes,
        mapping, 65536, 0, &req).is_ok());
    req.max_output_tokens = 1;
    assert!(check_document_request(WIDTH, "keyphrases-v1", budget.max_kv_bytes,
        mapping, 65536, 0, &req).is_err());
}

#[test]
fn every_initial_identity_is_checked_and_blank_qa_chunks_need_no_fake_identity() {
    let (budget, _, mapping, _, _) = fixture();
    let ids = [identity()];
    let req = request(&ids, mapping, budget.max_kv_bytes);
    assert!(check_document_request(WIDTH, "answer-v1", budget.max_kv_bytes,
        mapping, 65536, 64, &req).is_ok());
    for axis in 0..3 {
        let mut ids = [identity(), identity()];
        match axis {
            0 => ids[1].task_spec = "ner-v1".to_owned(),
            1 => ids[1].numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            _ => ids[1].quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=128"),
        }
        assert!(check_document_request(WIDTH, "answer-v1", budget.max_kv_bytes,
            mapping, 65536, 64, &request(&ids, mapping, budget.max_kv_bytes)).is_err(), "axis {axis}");
    }
    let empty = request(&[], mapping, budget.max_kv_bytes);
    assert!(check_document_request(WIDTH, "answer-v1", budget.max_kv_bytes,
        mapping, 65536, 64, &empty).is_err());
}

#[test]
fn forged_work_mask_output_and_chunk_claims_refuse_before_result_allocation() {
    let (budget, _, mapping, _, _) = fixture();
    let ids = [identity(), identity()];
    for axis in 0..12 {
        let mut req = request(&ids, mapping, budget.max_kv_bytes);
        match axis {
            0 => req.model_work.forward_positions = mapping.max_model_work.forward_positions + 1,
            1 => req.model_work.projected_logits = mapping.max_model_work.projected_logits + 1,
            2 => req.model_work.attention_pairs = mapping.max_model_work.attention_pairs + 1,
            3 => req.model_work.projections.dot_products = mapping.max_model_work.projections.dot_products + 1,
            4 => req.model_work.projections.multiply_accumulates = mapping.max_model_work.projections.multiply_accumulates + 1,
            5 => req.mask_node_visits = mapping.max_mask_visits + 1,
            6 => req.mask_limits.max_trie_node_visits += 1,
            7 => req.mask_limits.checkpoint_interval_nodes += 1,
            8 => req.max_output_tokens = 65,
            9 => req.max_result_bytes -= 1,
            10 => req.chunk_count = 1,
            _ => req.kv_reservation_bytes -= 1,
        }
        assert!(check_document_request(WIDTH, "answer-v1", budget.max_kv_bytes,
            mapping, 65536, 64, &req).is_err(), "axis {axis}");
    }
}

fn question() -> SourceQuestion {
    let mut text = String::with_capacity(8192); text.push_str("Who founded the company?");
    SourceQuestion { question: text, options: AnswerOptions::default(),
        verification: GroundingBudget { max_fields: 64, max_matches: 128, max_scan_steps: 65536 } }
}
fn config<T>(task: T) -> Int4SourceMapConfig<T> {
    let (budget, planning, mapping, _, host) = fixture();
    Int4SourceMapConfig { identity: identity(), task, budget, planning, mapping, native: host.native,
        preparation_reserve_bytes: host.preparation_reserve_bytes, reduction_reserve_bytes: 65536 }
}

#[test]
fn borrowed_private_questions_use_the_same_existing_workspace_calculations() {
    let owned = config(question());
    let borrowed = config(&owned.task);
    assert_eq!(borrowed.question_workspace_bytes(4).unwrap(), owned.question_workspace_bytes(4).unwrap());
    let owned = config(SourceQuestionSynthesis { question: question(),
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 1024 } });
    let borrowed = config(&owned.task);
    let hierarchy = QuestionHierarchyLimits::default();
    assert_eq!(borrowed.question_synthesis_workspace_bytes(4).unwrap(),
        owned.question_synthesis_workspace_bytes(4).unwrap());
    assert_eq!(borrowed.question_hierarchy_workspace_bytes(4, hierarchy).unwrap(),
        owned.question_hierarchy_workspace_bytes(4, hierarchy).unwrap());
    assert!(borrowed.question_hierarchy_workspace_bytes(4, hierarchy).unwrap()
        > borrowed.question_synthesis_workspace_bytes(4).unwrap());
}

#[test]
fn question_capacity_and_all_passes_are_owned_until_physical_completion() {
    let question = question(); let capacity = question.question.capacity() as u64;
    let task = Int4SourceDocumentTask::QuestionHierarchy {
        request: SourceQuestionSynthesis { question,
            limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 1024 } },
        limits: QuestionHierarchyLimits::default(),
    };
    assert_eq!(private_question_bytes(&task), capacity);
    fn send<T: Send + 'static>() {}
    send::<Int4SourceDocumentJobPlanner>();
    send::<DocumentJobInput<Cursor<Vec<u8>>>>();
    send::<Charged<DocumentJobInput<Cursor<Vec<u8>>>>>();
    let _entry = NlpEngine::job_int4_source_document::<Cursor<Vec<u8>>>;
}

#[test]
fn zero_reserves_and_overflow_cannot_create_small_job_memory_claims() {
    let (budget, planning, mapping, job, host) = fixture();
    assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
        job, host, 0, 0, JobInputMemory::CompletePopulation).is_err());
    assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
        job, host, 65536, u64::MAX, JobInputMemory::CompletePopulation).is_err());
    for axis in 0..4 {
        let mut host = host;
        match axis { 0 => host.preparation_reserve_bytes = 0, 1 => host.io_reserve_bytes = 0,
            2 => host.journal_reserve_bytes = 0, _ => host.serialization_reserve_bytes = 0 }
        assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
            job, host, 65536, 0, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn retained_document_and_cohort_hosts_preserve_question_frontier_and_whole_work_authority() {
    use crate::tasks::source_planning::grouped_int4::long::cohort::Int4SourceMapCohortMemoryLimits;
    let (budget, planning, mapping, job, host) = fixture();
    let complete = document_requirements(WIDTH, budget, planning, mapping, 65536,
        job, host, 123456, 8192, JobInputMemory::CompletePopulation).unwrap();
    let retained = document_requirements(WIDTH, budget, planning, mapping, 65536,
        job, host, 123456, 8192, JobInputMemory::RetainedRecord).unwrap();
    assert_eq!(complete.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(retained.input_bytes, host.required_buffer_bytes_for_input(job, JobInputMemory::RetainedRecord).unwrap() + 8192);
    assert_eq!(retained.native, complete.native);
    assert_eq!(retained.native_workspace_bytes, complete.native_workspace_bytes);
    assert_eq!(retained.document_workspace_bytes, complete.document_workspace_bytes);
    for chunks in [1, 2, 4] {
        let quote = Int4SourceMapCohortMemoryLimits { max_chunks: chunks, context_tokens: 64,
            max_output_tokens_per_chunk: 16, max_result_bytes_per_chunk: 4096 }.requirements(WIDTH).unwrap()
            .with_synthesis_context(64, WIDTH, budget.max_kv_bytes).unwrap();
        let mut full_cohort = complete; let mut retained_cohort = retained;
        full_cohort.use_cohorts(quote, host.native.allocator_reserve_bytes).unwrap();
        retained_cohort.use_cohorts(quote, host.native.allocator_reserve_bytes).unwrap();
        assert_eq!(full_cohort.input_bytes - retained_cohort.input_bytes, 2 * job.max_snapshot_bytes);
        assert_eq!(retained_cohort.native, full_cohort.native);
        assert_eq!(retained_cohort.native_workspace_bytes, full_cohort.native_workspace_bytes);
        assert_eq!(retained_cohort.document_workspace_bytes, full_cohort.document_workspace_bytes);
    }
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let mut short = job; short.max_work.model.attention_pairs -= 1;
        assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
            short, host, 123456, 8192, input).is_err());
        let mut missing = host; missing.native.allocator_reserve_bytes = 0;
        assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
            job, missing, 123456, 8192, input).is_err());
        assert!(document_requirements(WIDTH, budget, planning, mapping, 65536,
            job, host, 0, 8192, input).is_err());
    }
}

#[test]
fn compact_source_job_funds_one_encoder_within_its_authenticated_input_owner() {
    use crate::{
        batch::source::grouped_int4::document::Int4SourceDocumentLifetime,
        native_engine::strict_int4::weights::projection_fixture,
        tasks::source_planning::SourceTaskPlanner,
        tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding},
    };
    let (budget, planning, mapping, job, host) = fixture();
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let source_planner = SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap();
    let (weights, _) = projection_fixture(WIDTH);
    let lifetime = Int4SourceDocumentLifetime {
        verification: GroundingBudget { max_fields: 1_000_000, max_matches: 1_000_000,
            max_scan_steps: 1_000_000_000_000 },
        summary_scan_steps: 1_000_000_000_000,
    };
    let task = Int4SourceDocumentTask::Map(
        crate::tasks::source_planning::grouped_int4::long::SourceMapTask::Ner(Default::default()));
    let planner = Int4SourceDocumentJobPlanner::for_model(weights.identity(), Arc::new(source_planner),
        task, budget, planning, mapping, job, lifetime).unwrap();
    assert_eq!(planner.encoding_workspace_bytes().unwrap(), 0);
    check_encoding_preparation(0, host.preparation_reserve_bytes).unwrap();
    let call = SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 };
    let complete = SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let planner = planner.with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), complete).unwrap();
    let bytes = planner.encoding_workspace_bytes().unwrap();
    assert_eq!(bytes, call.requirements().unwrap().workspace_bytes + 512);
    assert_eq!(planner.map_document_encoding_limits(), Some(complete));
    let frontier = planner.workspace_bytes(host.native, 65_536).unwrap();
    for ownership in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let original = document_requirements(WIDTH, budget, planning, mapping, planner.max_result_bytes(),
            job, host, frontier, private_question_bytes(planner.task()), ownership).unwrap();
        let mut exact_host = host;
        exact_host.preparation_reserve_bytes = bytes;
        check_encoding_preparation(bytes, exact_host.preparation_reserve_bytes).unwrap();
        let exact = document_requirements(WIDTH, budget, planning, mapping, planner.max_result_bytes(),
            job, exact_host, frontier, private_question_bytes(planner.task()), ownership).unwrap();
        assert_eq!(exact.input_bytes - bytes, original.input_bytes - host.preparation_reserve_bytes);
        assert_eq!(exact.native, original.native);
        assert_eq!(exact.native_workspace_bytes, original.native_workspace_bytes);
        assert_eq!(exact.document_workspace_bytes, original.document_workspace_bytes);
        exact_host.preparation_reserve_bytes -= 1;
        assert!(check_encoding_preparation(bytes, exact_host.preparation_reserve_bytes).is_err());
        let short = document_requirements(WIDTH, budget, planning, mapping, planner.max_result_bytes(),
            job, exact_host, frontier, private_question_bytes(planner.task()), ownership).unwrap();
        assert_eq!(exact.input_bytes - short.input_bytes, 1);
    }
}
