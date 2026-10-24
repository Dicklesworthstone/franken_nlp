//! Pure admission, retained ownership and complete-receipt contracts. These
//! fixtures construct no resident weights, native engine or execution permit.
use super::*;
use crate::{
    batch::source::SourceMaskBudget,
    corpus::{entities::EntityVerificationWork,
        entities_int4::{Int4EntityChunkReceipt, Int4EntityDocumentReceipt},
        native_resolve::NativeResolveLimits, resolve::ResolveError},
    native_engine::decode::DecodeCancellationKind,
    tasks::{ir::TaskBudget, source_planning::{SourcePlanningLimits,
        grouped_int4::Int4SourceError}},
    validation::grounded_fields::VerifiedSourceSpan,
};

fn config(width: usize) -> Int4EntityConfig {
    let budget = TaskBudget { max_input_tokens: 64, max_output_tokens: 16,
        max_output_bytes: 4096, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(128, width).unwrap().kv_bytes };
    let mut source_planning = SourcePlanningLimits::default();
    source_planning.max_context_tokens = 128;
    let mut planning = NativeResolveLimits::default();
    planning.max_context_tokens = 128;
    planning.per_head = budget;
    Int4EntityConfig { ner: Default::default(), ner_budget: budget, source_planning,
        masks: SourceMaskBudget { per_mask: Default::default(), max_visits_per_item: 100,
            max_visits_per_run: 1000 },
        resolution: Default::default(), graph: Default::default(),
        scoring: Int4ResolveLimits { planning, max_model_work: Int4Work::default() },
        verification: Default::default(), max_model_work: Int4Work::default(),
        max_result_bytes: 4096, chunks: None, max_snapshot_chunks: 256 }
}

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
        run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100,
            cleanup_reserve_bytes: 4096 } }
}

#[test]
fn whole_allocated_kv_must_fit_both_stages_for_the_actual_group() {
    for width in [16, 32, 128] {
        let c = config(width);
        let (required, scratch) = requirements(&c, 80, native(), 4096, 8192, width).unwrap();
        assert_eq!(required, Int4MemoryRequirement::for_context(128, width).unwrap());
        assert_eq!(scratch, required.rope_bytes + required.scratch_payload_bound + 4096);
        assert_eq!(required.kv_bytes, c.ner_budget.max_kv_bytes);
        assert_eq!(required.kv_bytes, c.scoring.planning.per_head.max_kv_bytes);
        for stage in 0..2 {
            let mut capped = c.clone();
            if stage == 0 { capped.ner_budget.max_kv_bytes -= 1; }
            else { capped.scoring.planning.per_head.max_kv_bytes -= 1; }
            assert!(requirements(&capped, 80, native(), 4096, 8192, width).is_err(), "stage {stage}");
        }
    }
    let c = config(32);
    let (_, small) = requirements(&c, 80, native(), 4096, 8192, 16).unwrap();
    let (_, large) = requirements(&c, 80, native(), 4096, 8192, 128).unwrap();
    assert_eq!(large - small, 128 - 16);
}

#[test]
fn ner_and_pair_contexts_cannot_exceed_the_resident_capacity() {
    let c = config(32);
    assert!(requirements(&c, 129, native(), 4096, 8192, 32).is_err());
    let mut c = c;
    c.source_planning.max_context_tokens = 129;
    assert!(requirements(&c, 80, native(), 4096, 8192, 32).is_err());
    let mut c = config(32);
    c.scoring.planning.max_context_tokens = 129;
    assert!(requirements(&c, 80, native(), 4096, 8192, 32).is_err());
    let c = config(32);
    let mut n = native(); n.context_tokens = 0;
    assert!(requirements(&c, 0, n, 4096, 8192, 32).is_err());
    for width in [0, crate::native_engine::quant_algebra::MAX_MODEL_K + 1] {
        assert!(requirements(&c, 80, native(), 4096, 8192, width).is_err());
    }
}

#[test]
fn every_host_reserve_and_runtime_cleanup_bound_is_explicit() {
    for axis in 0..10 {
        let mut c = config(32); let mut n = native();
        let (mut preparation, mut graph) = (4096, 8192);
        match axis {
            0 => preparation = 0,
            1 => graph = 0,
            2 => n.allocator_reserve_bytes = 0,
            3 => n.allocator_reserve_bytes = u64::MAX,
            4 => n.run.max_elapsed = Duration::ZERO,
            5 => n.run.max_checkpoints = 1,
            6 => n.run.cleanup_reserve_bytes = 0,
            7 => c.ner_budget.max_output_bytes = 0,
            8 => c.scoring.planning.per_head.max_output_tokens = 1,
            _ => c.max_snapshot_chunks = 0,
        }
        assert!(requirements(&c, 80, n, preparation, graph, 32).is_err(), "axis {axis}");
    }
}

#[test]
fn intermediate_storage_prices_decoded_bytes_tokens_and_graph_with_checked_arithmetic() {
    let mut c = config(32);
    assert_eq!(temporary_bytes(&c, 8192).unwrap(), 4 * 4096 + 8 * 16 + 8192);
    c.ner_budget.max_output_tokens = u32::MAX;
    assert_eq!(temporary_bytes(&c, 8192).unwrap(), 4 * 4096 + 8 * u64::from(u32::MAX) + 8192);
    c.ner_budget.max_output_bytes = u64::MAX / 4 + 1;
    assert!(temporary_bytes(&c, 8192).is_err());
    let c = config(32);
    assert!(temporary_bytes(&c, u64::MAX).is_err());
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

// Honest empty candidate graphs provide the result shape; small synthetic NER
// receipts exercise the host's completion checks without claiming inference.
fn receipt(chunks: usize, chunked: bool) -> (Int4EntityRun, Expected) {
    assert!(chunks <= 2);
    let documents = if chunks == 0 { Vec::new() } else {
        vec![ResolutionDocument { id: "source".to_owned(), text: "é Beta".to_owned(), mentions: Vec::new() }]
    };
    let plan = ResolutionPlan::prepare(&documents, ResolveOptions::default(),
        ResolveLimits::default(), &mut Continue).unwrap();
    assert_eq!(plan.candidate_count(), 0);
    let resolution = Int4ResolutionRun { schema_version: 1, execution: INT4_RESOLUTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE,
        scoring_execution: crate::native_engine::strict_int4::scoring::INT4_SCORING_EXECUTION,
        group_width: 32, model_evaluated: false, head_count: 0,
        planned_model_work: Int4Work::default(), model_work: Int4Work::default(), rewound_positions: 0,
        result: plan.finalize(Vec::new(), &mut Continue).unwrap() };
    let per_chunk = Int4Work::for_sequence(32, 0, 2, 8).unwrap();
    let reserved_chunk = Int4Work::for_sequence(32, 0, 8, 32).unwrap();
    let (mut actual, mut reserved) = (Int4Work::default(), Int4Work::default());
    let mut chunk_receipts = Vec::new();
    for index in 0..chunks {
        actual = actual.checked_add(per_chunk).unwrap();
        reserved = reserved.checked_add(reserved_chunk).unwrap();
        let span = match (chunks, index) {
            (1, _) => VerifiedSourceSpan { byte_start: 0, byte_end: 7, scalar_start: 0, scalar_end: 6 },
            (2, 0) => VerifiedSourceSpan { byte_start: 0, byte_end: 3, scalar_start: 0, scalar_end: 2 },
            _ => VerifiedSourceSpan { byte_start: 3, byte_end: 7, scalar_start: 2, scalar_end: 6 },
        };
        chunk_receipts.push(Int4EntityChunkReceipt { chunk_id: index, source_span: span,
            proposed_entities: 0, anchored_mentions: 0, model_work: per_chunk, mask_node_visits: 10 });
    }
    let expected = Expected { width: 32, documents: documents.len(), chunks, ner_work: reserved,
        masks: chunks as u64 * 100, max_model_work: reserved, chunked };
    let receipts = if chunks == 0 { Vec::new() } else {
        vec![Int4EntityDocumentReceipt { document_id: "source".to_owned(), proposed_entities: 0,
            anchored_mentions: 0, model_work: actual, mask_node_visits: chunks as u64 * 10,
            chunks: chunk_receipts }]
    };
    (Int4EntityRun { schema_version: 1, execution: INT4_ENTITY_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: 32,
        extraction_task: crate::tasks::ner::NER_TASK_VERSION,
        chunk_profile: chunked.then_some(crate::tasks::mapreduce::CHUNK_PROFILE),
        chunk_count: chunks, documents: receipts, ner_reserved_work: reserved, ner_model_work: actual,
        reserved_model_work: reserved, model_work: actual, reserved_mask_node_visits: expected.masks,
        mask_node_visit_charge: chunks as u64 * 10, verification_used: EntityVerificationWork::default(),
        resolution, warnings: ["ner_recall_and_resolution_quality_not_established",
            "ner_partition_boundaries_may_split_entities"] }, expected)
}

fn counts(work: Int4Work) -> [u64; 5] {
    [work.forward_positions, work.projected_logits, work.attention_pairs,
        work.projections.dot_products, work.projections.multiply_accumulates]
}
fn set_count(work: &mut Int4Work, axis: usize, count: u64) {
    match axis {
        0 => work.forward_positions = count,
        1 => work.projected_logits = count,
        2 => work.attention_pairs = count,
        3 => work.projections.dot_products = count,
        4 => work.projections.multiply_accumulates = count,
        _ => panic!("unknown work axis"),
    }
}

#[test]
fn empty_snapshots_complete_without_inventing_model_or_mask_work() {
    for chunked in [false, true] {
        let (result, expected) = receipt(0, chunked);
        check_result(&result, expected).unwrap();
        assert!(result.documents.is_empty());
        assert!(result.resolution.result.judgments.is_empty());
        assert!(!result.resolution.model_evaluated);
        assert_eq!(result.reserved_model_work, Int4Work::default());
        assert_eq!(result.model_work, Int4Work::default());
        assert_eq!(result.mask_node_visit_charge, 0);
    }
}

#[test]
fn complete_source_and_nested_resolution_receipts_must_match_the_admission() {
    for axis in 0..14 {
        let (mut result, expected) = receipt(1, false);
        match axis {
            0 => result.schema_version = 2,
            1 => result.execution = "foreign-source-executor",
            2 => result.numerics_profile = "strict-quantized-v1",
            3 => result.group_width = 16,
            4 => result.extraction_task = "other-task",
            5 => result.chunk_profile = Some(crate::tasks::mapreduce::CHUNK_PROFILE),
            6 => result.documents.clear(),
            7 => result.resolution.result.document_count += 1,
            8 => result.chunk_count += 1,
            9 => result.ner_reserved_work.forward_positions += 1,
            10 => result.reserved_mask_node_visits += 1,
            11 => result.mask_node_visit_charge = result.reserved_mask_node_visits + 1,
            12 => result.reserved_model_work.projected_logits += 1,
            _ => result.model_work.projected_logits += 1,
        }
        assert!(check_result(&result, expected).is_err(), "source axis {axis}");
    }
    for axis in 0..6 {
        let (mut result, expected) = receipt(1, false);
        match axis {
            0 => result.resolution.execution = "foreign-pair-executor",
            1 => result.resolution.scoring_execution = "foreign-score-space",
            2 => result.resolution.group_width = 16,
            3 => result.resolution.head_count = 2,
            4 => result.resolution.rewound_positions = 2,
            _ => result.resolution.model_evaluated = true,
        }
        assert!(check_result(&result, expected).is_err(), "resolution axis {axis}");
    }
}

#[test]
fn chunked_completion_preserves_one_document_and_the_declared_partition_profile() {
    let (result, expected) = receipt(2, true);
    check_result(&result, expected).unwrap();
    assert_eq!(result.documents.len(), 1);
    assert_eq!(result.resolution.result.document_count, 1);
    assert_eq!(result.documents[0].chunks.len(), 2);
    assert_eq!(result.documents[0].chunks[1].source_span.byte_start, 3);
    assert_eq!(result.documents[0].chunks[1].source_span.scalar_start, 2);
    let mut other = expected; other.chunked = false;
    assert!(check_result(&result, other).is_err());
    let mut other = expected; other.chunks = 1;
    assert!(check_result(&result, other).is_err());
}

#[test]
fn early_eos_cannot_refund_any_reserved_ner_axis_to_the_total_allowance() {
    let (result, expected) = receipt(1, false);
    check_result(&result, expected).unwrap();
    for axis in 0..5 {
        assert!(counts(result.model_work)[axis] < counts(result.reserved_model_work)[axis]);
        let mut reduced = expected;
        set_count(&mut reduced.max_model_work, axis, counts(result.model_work)[axis]);
        // Actual work fits, but the immutable NER reservation must still fit.
        super::super::check_work(result.model_work, reduced.max_model_work).unwrap();
        assert!(check_result(&result, reduced).is_err(), "axis {axis}");
    }
}

#[test]
fn actual_ner_work_cannot_exceed_its_reservation_on_any_axis() {
    for axis in 0..5 {
        let (mut result, expected) = receipt(1, false);
        let limit = counts(expected.ner_work)[axis];
        set_count(&mut result.ner_model_work, axis, limit + 1);
        result.model_work = result.ner_model_work;
        assert!(check_result(&result, expected).is_err(), "axis {axis}");
    }
}

#[test]
fn complete_stage_work_addition_refuses_overflow_on_every_axis() {
    for axis in 0..5 {
        let (mut result, expected) = receipt(1, false);
        set_count(&mut result.resolution.planned_model_work, axis, u64::MAX);
        result.resolution.model_work = result.resolution.planned_model_work;
        assert!(check_result(&result, expected).is_err(), "axis {axis}");
    }
}

#[test]
fn all_eleven_cancellation_kinds_survive_every_entity_stage_and_private_diagnostics() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stage in [Int4EntityError::Source(Int4SourceError::Cancelled(cause)),
            Int4EntityError::Graph(ResolveError::Cancelled(cause)),
            Int4EntityError::Resolution(Int4ResolveError::Native(StrictInt4Error::Cancelled(cause))),
            Int4EntityError::Native(StrictInt4Error::Cancelled(cause))] {
            let error = execution_error(stage);
            let HostedError::Int4(Int4HostError::Entities(inner)) = &error else { panic!("wrong stage"); };
            assert_eq!(inner.cancellation(), Some(cause));
            assert_eq!(error.to_string(), format!("{error:?}"));
            assert_eq!(inner.to_string(), format!("{inner:?}"));
            assert!(error.source().is_some());
            assert!(inner.source().is_some());
        }
    }
}

#[test]
fn consumed_preparation_and_guarded_complete_results_own_the_physical_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Input>(); send::<PreparedInt4EntityCorpus>(); send::<Int4EntityRun>();
    send::<HostedOutput<Int4EntityRun>>();
    let _host = NlpEngine::execute_int4_entities;
}
