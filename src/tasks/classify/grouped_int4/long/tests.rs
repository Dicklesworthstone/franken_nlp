//! Real pinned prompt compilation and original finalizers with private
//! synthetic logits. No model-present or task-quality result is inferred.
use super::*;
use std::sync::OnceLock;
use crate::{native_engine::{portable_int8::ProjectionWork,
        strict_int4::scoring::cohort::task::fixtures},
    tasks::{classify::ClassificationLabel, ir::PromptSegmentKind},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls}};
use fixtures::Continue;

fn fixture() -> (&'static ClassificationPlanner, &'static ExecutionIdentity) {
    static FIXTURE: OnceLock<(ClassificationPlanner, ExecutionIdentity)> = OnceLock::new();
    let (planner, identity) = FIXTURE.get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let planner = ClassificationPlanner::pinned(controls.template_controls(), eos).unwrap();
        let identity = fixtures::identity("classify-v1", planner.tokenizer_digest(), *planner.template_digest());
        (planner, identity)
    });
    (planner, identity)
}
fn request(mode: ClassificationMode) -> ClassificationRequest {
    ClassificationRequest { document: "é\r\n𐐀 one <think> literal data and tail".to_owned(),
        labels: ["a", "b", "c"].map(|id| ClassificationLabel { id: id.to_owned(),
            description: format!("Description of {id} with \"quotes\" and é") }).to_vec(),
        mode, policy: ClassificationPolicy::default(), budget: fixtures::budget() }
}
fn mapping() -> Int4ClassificationMapLimits {
    Int4ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8,
        max_chunk_tokens: 1024, context_tokens: 1024, reserved_tokens: 1, max_chunks: 64, max_tokenizer_calls: 4096 },
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        max_retained_result_bytes: 4 * 1024 * 1024, max_result_bytes: 8 * 1024 * 1024,
        aggregation: ClassificationAggregationPolicy::default() }
}
fn prepare<'a>(request: &'a ClassificationRequest, limits: Int4ClassificationMapLimits)
    -> Result<PreparedInt4ClassificationMap<'a>, Int4ClassificationMapError> {
    let (planner, identity) = fixture(); let planning = ClassificationLimits::default();
    let capacity = planner.source_capacity(request, request.budget, planning, limits.chunks.context_tokens)?;
    PreparedInt4ClassificationMap::compile(request, limits, capacity, 32, &mut Continue, |row, control|
        planner.compile_int4(row, &PlanContext::new(identity, request.budget).unwrap(), planning, 32, control))
}
fn run(plan: &PreparedInt4Classification) -> Int4ClassificationRun {
    plan.execute_heads(&mut Continue, |_, head, schedule, _| Ok(Int4CandidateRun {
        schema_version: 1, execution: INT4_SCORING_EXECUTION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        group_width: 32, scores: head.classifier.scorer.score(&mut fixtures::Constant, ScoringMode::FullVocabulary).unwrap(),
        model_work: schedule.model, rewound_positions: schedule.rewound })).unwrap()
}
fn finish(plan: &PreparedInt4ClassificationMap<'_>) -> Result<Int4ClassificationMapRun, Int4ClassificationMapError> {
    plan.collect_with_control(&mut Continue, |_, row, _| Ok(run(row)))
}

#[test]
fn exact_source_partition_retains_unicode_crlf_and_literal_control_text() {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let controls = pinned_controls::pinned().unwrap();
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let request = request(mode); let plan = prepare(&request, mapping()).unwrap();
        assert!(plan.chunk_count() > 1);
        let mut reconstructed = String::new();
        for (source, compiled) in plan.chunks.chunks().iter().zip(&plan.plans) {
            reconstructed.push_str(source.text());
            assert_eq!(source.tokens(), source.text().len());
            assert!(!source.text().ends_with('\r'));
            for head in &compiled.inner.heads {
                let document = head.task.ir().prompt_segments().iter()
                    .filter(|segment| segment.kind() == PromptSegmentKind::Document).nth(1).unwrap();
                assert_eq!(tokenizer.tokenizer().decode_bytes(document.token_ids()).unwrap(), source.text().as_bytes());
                assert!(document.token_ids().iter().all(|&id| !controls.template_controls().contains(id)));
            }
        }
        assert_eq!(reconstructed, request.document);
        let result = finish(&plan).unwrap();
        plan.preflight_metadata().verify_completed(&result).unwrap();
        assert_eq!(result.source_span.byte_end, request.document.len());
        assert_eq!(result.source_span.scalar_end, request.document.chars().count());
        assert!(!result.single_context_equivalence);
    }
}
#[test]
fn every_work_axis_covers_all_chunks_and_all_heads_before_execution() {
    let request = request(ClassificationMode::MultiLabel); let plan = prepare(&request, mapping()).unwrap();
    let work = plan.plans.iter().try_fold(Int4Work::default(), |sum, chunk| sum.checked_add(chunk.planned_work())).unwrap();
    assert_eq!(work, plan.planned_work()); assert_eq!(plan.head_count(), request.labels.len());
    for axis in 0..5 {
        let mut limits = mapping(); limits.max_model_work = work;
        match axis { 0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1, 2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            _ => limits.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(matches!(prepare(&request, limits), Err(Int4ClassificationMapError::Work)));
    }
    let mut limits = mapping(); limits.max_model_work = work;
    assert_eq!(prepare(&request, limits).unwrap().planned_work(), work);
}
#[test]
fn late_failure_or_cancel_never_returns_an_earlier_successful_prefix() {
    let request = request(ClassificationMode::Exclusive); let plan = prepare(&request, mapping()).unwrap();
    let mut calls = 0;
    let error = plan.collect_with_control(&mut Continue, |index, chunk, _| {
        calls += 1;
        if index == 1 { return Err(ClassificationPlanningError::Cancelled(DecodeCancellationKind::Deadline).into()); }
        Ok(run(chunk))
    }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls, 2);
    struct Stop(bool);
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.0.then_some(DecodeCancellationKind::User) } }
    let mut calls = 0;
    let error = plan.collect_with_control(&mut Stop(false), |_, chunk, control| {
        calls += 1; let result = run(chunk); control.0 = true; Ok(result)
    }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::User)); assert_eq!(calls, 1);
}
#[test]
fn retained_frontier_and_complete_document_caps_are_independent_and_exact() {
    let request = request(ClassificationMode::Exclusive); let mut plan = prepare(&request, mapping()).unwrap();
    let baseline = finish(&plan).unwrap(); let bytes = canonjson::canonical_bytes(&baseline).unwrap().len() as u64;
    plan.limits.max_retained_result_bytes = baseline.retained_result_bytes;
    plan.limits.max_result_bytes = bytes;
    assert_eq!(finish(&plan).unwrap(), baseline);
    plan.limits.max_result_bytes -= 1;
    assert!(matches!(finish(&plan), Err(Int4ClassificationMapError::Output)));
    plan.limits.max_result_bytes = bytes; plan.limits.max_retained_result_bytes -= 1;
    assert!(matches!(finish(&plan), Err(Int4ClassificationMapError::Output)));
}
#[test]
fn metadata_detects_missing_reordered_chunks_and_native_work_drift() {
    let request = request(ClassificationMode::MultiLabel); let plan = prepare(&request, mapping()).unwrap();
    let baseline = finish(&plan).unwrap(); let expected = plan.preflight_metadata();
    for fault in 0..7 {
        let mut changed = baseline.clone();
        match fault { 0 => { changed.chunks.pop(); }, 1 => changed.chunks.swap(0, 1),
            2 => changed.chunks[1].source_span.byte_start += 1, 3 => changed.chunks[1].run.head_count += 1,
            4 => changed.model_work.projected_logits += 1, 5 => changed.single_context_equivalence = true,
            _ => changed.aggregate.classification_policy.minimum_margin_ppm += 1 }
        assert!(expected.verify_completed(&changed).is_err());
    }
}
#[test]
fn whole_source_chunk_and_tokenizer_limits_fail_without_truncation() {
    let request = request(ClassificationMode::Exclusive);
    for axis in 0..3 {
        let mut limits = mapping();
        match axis { 0 => limits.chunks.max_input_bytes = request.document.len() - 1,
            1 => limits.chunks.max_chunks = 1, _ => limits.chunks.max_tokenizer_calls = 1 }
        assert!(prepare(&request, limits).is_err());
    }
    let mut limits = mapping(); limits.chunks.max_chunk_tokens = 1;
    assert!(matches!(prepare(&request, limits), Err(Int4ClassificationMapError::Chunk(MapReduceError::TokenBudget))));
}
