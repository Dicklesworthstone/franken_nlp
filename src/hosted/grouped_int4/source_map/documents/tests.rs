//! Pure reservation and admission fixtures; no native model is executed here.
use super::*;
use crate::{
    batch::{BatchLimits, BatchWork},
    tasks::{ner::{EntityType, NerOptions},
        source_planning::grouped_int4::long::{
            question::{SourceQuestion, synthesis::{QuestionSynthesisLimits, SourceQuestionSynthesis,
                hierarchy::QuestionHierarchyLimits}},
            summary::synthesis::{SourceSummarySynthesis, SummarySynthesisLimits,
                hierarchy::SummaryHierarchyLimits},
        }},
    validation::grounded_fields::GroundingBudget,
};

const WIDTH: usize = 32;
fn fixture() -> (Int4SourceMapConfig, Int4SourceDocumentBatchLimits, CorpusLimits) {
    let (_, base) = super::super::tests::fixture(SourceMapTask::Ner(Default::default()), WIDTH);
    let run = Int4SourceDocumentBatchLimits {
        max_model_work: base.mapping.max_model_work, max_mask_visits: base.mapping.max_mask_visits,
        max_verification: GroundingBudget { max_fields: 1_000_000, max_matches: 1_000_000,
            max_scan_steps: 1_000_000_000_000 },
        max_summary_scan_steps: 1_000_000_000_000, max_keyphrase_scan_work: 1_000_000_000_000,
    };
    let host = CorpusLimits {
        native: base.native, preparation_reserve_bytes: base.preparation_reserve_bytes,
        io_reserve_bytes: 131072,
        transport: BatchLimits { max_document_bytes: base.mapping.chunks.max_input_bytes,
            max_work: BatchWork { forward_positions: run.max_model_work.forward_positions,
                projected_logits: run.max_model_work.projected_logits }, ..Default::default() },
    };
    (base, run, host)
}
fn quote(base: &Int4SourceMapConfig, task: &Int4SourceDocumentTask,
    run: Int4SourceDocumentBatchLimits, host: CorpusLimits, discovery: bool,
    hierarchy: Option<Int4HierarchyCohortSchedule>) -> Result<Int4SourceDocumentCorpusRequirement, HostedError> {
    int4_source_document_corpus_requirements(WIDTH, task, base.budget, base.planning, base.mapping,
        run, host, base.reduction_reserve_bytes, discovery, hierarchy)
}
fn question(mapping: Int4SourceMapLimits, capacity: usize) -> SourceQuestion {
    let mut text = String::with_capacity(capacity); text.push_str("Who founded the company?");
    SourceQuestion { question: text, options: Default::default(), verification: mapping.verification }
}
fn summary(mapping: Int4SourceMapLimits) -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: Default::default(), synthesis_options: Default::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 32, max_evidence_bytes: 1024,
            verification: mapping.verification } }
}
fn qa(mapping: Int4SourceMapLimits, capacity: usize) -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: question(mapping, capacity),
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 1024 } }
}

#[test]
fn pure_quote_keeps_source_stream_frozen_metadata_and_complete_frontier_together() {
    let (base, run, host) = fixture();
    let task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
    let need = quote(&base, &task, run, host, false, None).unwrap();
    assert_eq!(need.input_bytes, host.reservation_bytes().unwrap()
        + base.mapping.chunks.max_input_bytes as u64
        + base.mapping.chunks.max_chunks as u64 * SOURCE_DOCUMENT_CHUNK_METADATA_BYTES
        + task.retained_bytes().unwrap());
    assert_eq!(need.document_workspace_bytes, base.workspace_bytes(base.mapping.chunks.max_chunks).unwrap());
    assert_eq!(need.native.kv_bytes, base.budget.max_kv_bytes);
    assert_eq!(need.native_workspace_bytes, need.native.rope_bytes
        + need.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    assert_eq!(need.output_reservation_bytes, 4 * base.mapping.reduction.max_result_bytes as u64
        + 8 * base.mapping.chunks.max_chunks as u64 * u64::from(base.budget.max_output_tokens) + 4096);
    assert_eq!(need.total_bytes, need.native.kv_bytes + need.native_workspace_bytes
        + need.input_bytes + need.document_workspace_bytes + need.output_reservation_bytes);
    assert!(need.cohort.is_none());
}

#[test]
fn retained_ner_vector_capacity_is_priced_separately_from_its_semantic_contents() {
    let (base, run, host) = fixture();
    let small = Int4SourceDocumentTask::Map(SourceMapTask::Ner(NerOptions::default()));
    let mut options = NerOptions::default();
    let mut types = Vec::with_capacity(4096); types.extend_from_slice(&options.types); options.types = types;
    let large = Int4SourceDocumentTask::Map(SourceMapTask::Ner(options));
    let a = quote(&base, &small, run, host, false, None).unwrap();
    let b = quote(&base, &large, run, host, false, None).unwrap();
    assert_eq!(b.input_bytes - a.input_bytes, large.retained_bytes().unwrap() - small.retained_bytes().unwrap());
    assert!(b.input_bytes - a.input_bytes >= 4093 * std::mem::size_of::<EntityType>() as u64);
    assert_eq!(a.document_workspace_bytes, b.document_workspace_bytes);
    assert_eq!(a.native, b.native);
    assert_eq!(a.output_reservation_bytes, b.output_reservation_bytes);
}

#[test]
fn every_question_mode_retains_private_capacity_and_only_its_public_transcripts() {
    let (base, run, host) = fixture();
    for mode in 0..3 {
        let task = |capacity| match mode {
            0 => Int4SourceDocumentTask::Question(question(base.mapping, capacity)),
            1 => Int4SourceDocumentTask::QuestionSynthesis(qa(base.mapping, capacity)),
            _ => Int4SourceDocumentTask::QuestionHierarchy { request: qa(base.mapping, capacity),
                limits: QuestionHierarchyLimits { max_passes: 4, ..Default::default() } },
        };
        let small = task(32); let large = task(8192);
        let a = quote(&base, &small, run, host, false, None).unwrap();
        let b = quote(&base, &large, run, host, false, None).unwrap();
        assert_eq!(b.input_bytes - a.input_bytes, large.retained_bytes().unwrap() - small.retained_bytes().unwrap());
        let tokens = if mode == 2 { 4 * u64::from(base.budget.max_output_tokens) } else { 0 };
        assert_eq!(a.output_reservation_bytes, 4 * base.mapping.reduction.max_result_bytes as u64 + 8 * tokens + 4096);
        assert_eq!(a.document_workspace_bytes, b.document_workspace_bytes);
    }
}

#[test]
fn summary_hierarchy_keeps_discovery_and_every_permitted_native_transcript() {
    let (base, run, host) = fixture();
    let flat = Int4SourceDocumentTask::SummarySynthesis(summary(base.mapping));
    let hierarchy = Int4SourceDocumentTask::SummaryHierarchy { request: summary(base.mapping),
        limits: SummaryHierarchyLimits { max_passes: 4, max_levels: 3, ..Default::default() } };
    let a = quote(&base, &flat, run, host, false, None).unwrap();
    let b = quote(&base, &hierarchy, run, host, false, None).unwrap();
    assert_eq!(b.output_reservation_bytes - a.output_reservation_bytes,
        3 * u64::from(base.budget.max_output_tokens) * 8);
    assert_eq!(b.document_workspace_bytes - a.document_workspace_bytes,
        3 * (4 * base.budget.max_output_bytes + 8 * u64::from(base.budget.max_output_tokens))
            + 4 * (8192 + 256) + 3 * 128);
}

#[test]
fn independent_hierarchy_and_discovery_phases_keep_one_full_native_peak() {
    let (mut base, run, host) = fixture(); base.mapping.reduction.map_batch_chunks = 2;
    let task = Int4SourceDocumentTask::QuestionHierarchy { request: qa(base.mapping, 8192),
        limits: QuestionHierarchyLimits { max_passes: 4, ..Default::default() } };
    let serial = quote(&base, &task, run, host, false, None).unwrap();
    for discovery_cohorts in [false, true] {
        let schedule = Int4HierarchyCohortSchedule { max_groups: 3, discovery_cohorts };
        let grouped = quote(&base, &task, run, host, discovery_cohorts, Some(schedule)).unwrap();
        assert_eq!(grouped.native.kv_bytes, 3 * base.budget.max_kv_bytes);
        assert_eq!(grouped.input_bytes, serial.input_bytes + 2 * host.preparation_reserve_bytes);
        assert_eq!(grouped.document_workspace_bytes, serial.document_workspace_bytes);
        assert_eq!(grouped.output_reservation_bytes, serial.output_reservation_bytes);
        let decoder = grouped.cohort.unwrap();
        assert_eq!(grouped.total_bytes, grouped.native.kv_bytes + grouped.native_workspace_bytes
            + grouped.input_bytes + grouped.document_workspace_bytes + decoder.workspace_bytes
            + grouped.output_reservation_bytes);
        assert!(quote(&base, &task, run, host, !discovery_cohorts, Some(schedule)).is_err());
    }
    let wrong_task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
    assert!(quote(&base, &wrong_task, run, host, false,
        Some(Int4HierarchyCohortSchedule { max_groups: 3, discovery_cohorts: false })).is_err());
}

#[test]
fn partial_memory_and_lifetime_declarations_refuse_before_stream_or_model_access() {
    for axis in 0..15 {
        let (mut base, mut run, mut host) = fixture();
        let task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
        match axis {
            0 => base.budget.max_kv_bytes -= 1,
            1 => base.budget.max_kv_bytes += 1,
            2 => base.planning.max_context_tokens -= 1,
            3 => host.native.allocator_reserve_bytes = 0,
            4 => host.preparation_reserve_bytes = 0,
            5 => host.io_reserve_bytes = 0,
            6 => base.reduction_reserve_bytes = 0,
            7 => host.transport.max_output_line_bytes = base.mapping.reduction.max_result_bytes - 1,
            8 => run.max_model_work.forward_positions -= 1,
            9 => run.max_model_work.projected_logits -= 1,
            10 => run.max_model_work.attention_pairs -= 1,
            11 => run.max_model_work.projections.dot_products -= 1,
            12 => run.max_model_work.projections.multiply_accumulates -= 1,
            13 => run.max_mask_visits -= 1,
            _ => run.max_verification.max_matches = base.mapping.verification.max_matches - 1,
        }
        assert!(quote(&base, &task, run, host, false, None).is_err(), "axis {axis}");
    }
    let (base, run, mut host) = fixture();
    let task = Int4SourceDocumentTask::Map(SourceMapTask::Ner(Default::default()));
    host.io_reserve_bytes = u64::MAX;
    assert!(quote(&base, &task, run, host, false, None).is_err());
    assert!(output_bytes(u64::MAX, 0).is_err());
    assert!(output_bytes(1, u64::MAX).is_err());
}

#[test]
fn admission_keeps_complete_sequence_work_and_frozen_physical_peak() {
    let (base, _, _) = fixture();
    let identities = [base.identity.clone(), base.identity.clone()];
    let request = || Int4SourceDocumentAdmission { identities: &identities,
        model_work: base.mapping.max_model_work, mask_node_visits: base.mapping.max_mask_visits,
        mask_limits: base.mapping.mask_limits, kv_reservation_bytes: 3 * base.budget.max_kv_bytes,
        max_result_bytes: base.mapping.reduction.max_result_bytes as u64,
        max_output_tokens: 2 * u64::from(base.budget.max_output_tokens), chunk_count: 2 };
    let check = |claim: &Int4SourceDocumentAdmission<'_>| admission::check_document_request(WIDTH, "ner-v1",
        3 * base.budget.max_kv_bytes, base.mapping, base.mapping.reduction.max_result_bytes as u64,
        base.mapping.chunks.max_chunks as u64 * u64::from(base.budget.max_output_tokens), claim);
    check(&request()).unwrap();
    for axis in 0..8 {
        let mut claim = request();
        match axis {
            0 => claim.kv_reservation_bytes = base.budget.max_kv_bytes,
            1 => claim.model_work.attention_pairs += 1,
            2 => claim.model_work.projections.dot_products += 1,
            3 => claim.model_work.projections.multiply_accumulates += 1,
            4 => claim.mask_node_visits += 1,
            5 => claim.mask_limits.checkpoint_interval_nodes += 1,
            6 => claim.max_result_bytes -= 1,
            _ => claim.chunk_count = 1,
        }
        assert!(check(&claim).is_err(), "axis {axis}");
    }
}
