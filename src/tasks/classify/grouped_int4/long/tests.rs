//! Real pinned prompt compilation and original finalizers with private
//! synthetic logits. No model-present or task-quality result is inferred.
use super::*;
use std::sync::OnceLock;
use crate::{native_engine::{portable_int8::ProjectionWork,
        strict_int4::scoring::cohort::task::fixtures},
    tasks::{classify::ClassificationLabel, ir::PromptSegmentKind},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls}};
use fixtures::Continue;

mod cohorts;
mod model;

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

fn cohort_budget(plan: &PreparedInt4ClassificationMap<'_>) -> Int4ClassificationMapCohortBudget {
    use crate::native_engine::strict_int4::Int4MemoryBudget;
    Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()),
        memory: Int4MemoryBudget { max_kv_bytes: u64::MAX, max_rope_bytes: u64::MAX,
            max_scratch_payload_bytes: u64::MAX }, max_task_workspace_bytes: u64::MAX,
        max_output_bytes: 4 * 1024 * 1024 }
}
fn finish_window(plan: &PreparedInt4ClassificationMap<'_>, admitted: &[ExecutionIdentity], window: &cohort::Window)
    -> Result<crate::native_engine::strict_int4::scoring::cohort::Int4ScoredCohort<Vec<Int4ClassificationRun>>,
        Int4ClassificationMapError> {
    use crate::native_engine::strict_int4::scoring::cohort::{Int4ScoringCohortBudget, task};
    let documents: Vec<_> = (window.start..window.end).map(|index| (&plan.plans[index], &admitted[index])).collect();
    let budget = Int4ScoringCohortBudget { native: Int4RunBudget::exact(window.work),
        max_kv_bytes: u64::MAX, max_output_bytes: 4 * 1024 * 1024 };
    let (schedules, work) = task::document_layout(&documents, &window.contexts, budget)?;
    let native = fixtures::run(documents.iter().flat_map(|(plan, _)| plan.inner.heads.iter().zip(&plan.schedules)
        .map(|(head, &schedule)| (head.classifier.scorer.score(&mut fixtures::Constant,
            ScoringMode::FullVocabulary).unwrap(), schedule))));
    Ok(task::finish_documents(&documents, native, &schedules, work, budget.max_output_bytes, &mut Continue)?)
}

#[test]
fn chunk_cohorts_keep_exact_heads_and_the_same_complete_aggregate_in_both_modes() {
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let request = request(mode); let plan = prepare(&request, mapping()).unwrap();
        let admitted = plan.identities().unwrap(); let expected = finish(&plan).unwrap();
        for chunks in [1, 2, plan.chunk_count() - 1] {
            let windows = plan.cohort_windows(&admitted, chunks, cohort_budget(&plan), &mut Continue).unwrap();
            assert_eq!(windows.len(), plan.chunk_count().div_ceil(chunks));
            assert_eq!(windows.first().unwrap().start, 0);
            assert_eq!(windows.last().unwrap().end, plan.chunk_count());
            assert!(windows.windows(2).all(|pair| pair[0].end == pair[1].start));
            let work = windows.iter().fold(Int4Work::default(), |total, window| {
                assert_eq!(window.contexts.len(), (window.end - window.start) * plan.head_count());
                let expected: Vec<_> = plan.plans[window.start..window.end].iter()
                    .flat_map(|plan| plan.cohort_contexts().unwrap()).collect();
                assert_eq!(window.contexts, expected);
                total.checked_add(window.work).unwrap()
            });
            assert_eq!(work, plan.planned_work());
            let actual = plan.collect_cohorts_with_control(&windows, &mut Continue,
                |window, _| finish_window(&plan, &admitted, window)).unwrap();
            assert_eq!(actual.output, expected);
            assert_eq!(actual.chunk_cohorts, windows.len());
            assert_eq!(actual.execution, INT4_CLASSIFICATION_MAP_COHORT_EXECUTION);
            plan.preflight_metadata().verify_completed(&actual.output).unwrap();
        }
    }
}

#[test]
fn all_future_windows_validate_identity_and_nonrenewable_work_before_any_cohort() {
    use crate::execution_identity::Sha256Digest;
    let request = request(ClassificationMode::MultiLabel); let mut plan = prepare(&request, mapping()).unwrap();
    let admitted = plan.identities().unwrap(); let budget = cohort_budget(&plan);
    let windows = plan.cohort_windows(&admitted, 2, budget, &mut Continue).unwrap();
    assert!(windows.len() > 1);
    let mut last = admitted.clone(); last.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"foreign tail");
    assert!(plan.cohort_windows(&last, 2, budget, &mut Continue).is_err());
    let mut shortened = budget; shortened.native = Int4RunBudget::exact(windows[0].work);
    assert!(plan.cohort_windows(&admitted, 2, shortened, &mut Continue).is_err());
    let original = plan.limits.max_model_work;
    for axis in 0..5 {
        plan.limits.max_model_work = plan.planned_work();
        match axis { 0 => plan.limits.max_model_work.forward_positions -= 1,
            1 => plan.limits.max_model_work.projected_logits -= 1,
            2 => plan.limits.max_model_work.attention_pairs -= 1,
            3 => plan.limits.max_model_work.projections.dot_products -= 1,
            _ => plan.limits.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(matches!(plan.cohort_windows(&admitted, 2, budget, &mut Continue), Err(Int4ClassificationMapError::Work)));
    }
    plan.limits.max_model_work = original;
    for chunks in [0, 22, 65, usize::MAX] {
        assert!(matches!(plan.cohort_windows(&admitted, chunks, budget, &mut Continue), Err(Int4ClassificationMapError::Limits)));
    }
    assert!(plan.cohort_windows(&admitted[..admitted.len() - 1], 2, budget, &mut Continue).is_err());
}

#[test]
fn every_window_prices_simultaneous_kv_rope_scratch_and_task_workspace() {
    use crate::native_engine::strict_int4::{Int4MemoryRequirement,
        scoring::cohort::required_task_workspace_bytes};
    let request = request(ClassificationMode::MultiLabel); let plan = prepare(&request, mapping()).unwrap();
    let admitted = plan.identities().unwrap(); let mut budget = cohort_budget(&plan);
    let windows = plan.cohort_windows(&admitted, 2, budget, &mut Continue).unwrap();
    let requirements: Vec<_> = windows.iter().map(|window|
        Int4MemoryRequirement::for_cohort(&window.contexts, plan.group_width()).unwrap()).collect();
    budget.memory.max_kv_bytes = requirements.iter().map(|need| need.kv_bytes).max().unwrap();
    budget.memory.max_rope_bytes = requirements.iter().map(|need| need.rope_bytes).max().unwrap();
    budget.memory.max_scratch_payload_bytes = requirements.iter().map(|need| need.scratch_payload_bound).max().unwrap();
    budget.max_task_workspace_bytes = windows.iter().map(|window| required_task_workspace_bytes(window.work,
        window.contexts.len(), budget.max_output_bytes).unwrap()).max().unwrap();
    assert!(plan.cohort_windows(&admitted, 2, budget, &mut Continue).is_ok());
    for axis in 0..4 {
        let mut small = budget;
        match axis { 0 => small.memory.max_kv_bytes -= 1, 1 => small.memory.max_rope_bytes -= 1,
            2 => small.memory.max_scratch_payload_bytes -= 1, _ => small.max_task_workspace_bytes -= 1 }
        assert!(matches!(plan.cohort_windows(&admitted, 2, small, &mut Continue), Err(Int4ClassificationMapError::Admission)));
    }
    let mut narrow_request = request.clone();
    let maximum = plan.plans.iter().map(|chunk| chunk.cohort_contexts().unwrap().iter().sum::<usize>()).max().unwrap();
    narrow_request.budget.max_kv_bytes = maximum as u64 * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64 - 1;
    let narrow = prepare(&narrow_request, mapping()).unwrap();
    let error = narrow.cohort_windows(&narrow.identities().unwrap(), 2, cohort_budget(&narrow), &mut Continue).err().unwrap();
    assert!(matches!(error, Int4ClassificationMapError::Classification(Int4ClassificationError::Scoring(
        Int4ScoringError::Native(StrictInt4Error::Memory)))));
}

#[test]
fn a_late_window_failure_or_cancellation_publishes_no_complete_document() {
    let request = request(ClassificationMode::Exclusive); let plan = prepare(&request, mapping()).unwrap();
    let admitted = plan.identities().unwrap();
    let windows = plan.cohort_windows(&admitted, 2, cohort_budget(&plan), &mut Continue).unwrap();
    let mut calls = 0;
    let error = plan.collect_cohorts_with_control(&windows, &mut Continue, |window, _| {
        calls += 1;
        if window.start != 0 { return Err(ClassificationPlanningError::Cancelled(DecodeCancellationKind::Deadline).into()); }
        finish_window(&plan, &admitted, window)
    }).unwrap_err();
    assert_eq!(calls, 2); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    struct Stop(bool);
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.0.then_some(DecodeCancellationKind::User) } }
    let mut calls = 0;
    let error = plan.collect_cohorts_with_control(&windows, &mut Stop(false), |window, control| {
        calls += 1; let run = finish_window(&plan, &admitted, window)?; control.0 = true; Ok(run)
    }).unwrap_err();
    assert_eq!(calls, 1); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::User));
}

#[test]
fn malformed_native_window_and_exact_document_output_caps_fail_closed() {
    let request = request(ClassificationMode::Exclusive); let mut plan = prepare(&request, mapping()).unwrap();
    let admitted = plan.identities().unwrap();
    let windows = plan.cohort_windows(&admitted, 2, cohort_budget(&plan), &mut Continue).unwrap();
    for fault in 0..5 {
        assert!(plan.collect_cohorts_with_control(&windows, &mut Continue, |window, _| {
            let mut native = finish_window(&plan, &admitted, window)?;
            match fault { 0 => { native.output.pop(); }, 1 => native.execution.push('x'),
                2 => native.group_steps = 0, 3 => native.projection_groups = u64::MAX,
                _ => native.output.last_mut().unwrap().model_work.projected_logits += 1 }
            Ok(native)
        }).is_err());
    }
    assert!(plan.collect_cohorts_with_control(&windows[..windows.len() - 1], &mut Continue,
        |window, _| finish_window(&plan, &admitted, window)).is_err());
    let expected = plan.collect_cohorts_with_control(&windows, &mut Continue,
        |window, _| finish_window(&plan, &admitted, window)).unwrap();
    plan.limits.max_result_bytes = canonjson::canonical_bytes(&expected.output).unwrap().len() as u64;
    plan.limits.max_retained_result_bytes = expected.output.retained_result_bytes;
    assert_eq!(plan.collect_cohorts_with_control(&windows, &mut Continue,
        |window, _| finish_window(&plan, &admitted, window)).unwrap(), expected);
    plan.limits.max_result_bytes -= 1;
    assert!(matches!(plan.collect_cohorts_with_control(&windows, &mut Continue,
        |window, _| finish_window(&plan, &admitted, window)), Err(Int4ClassificationMapError::Output)));
}

#[test]
#[ignore = "requires controller-admitted pinned source and full-model INT4 document/cohort parity"]
fn native_document_cohorts_match_serial_chunks_and_aggregates_for_both_classification_modes() {
    use crate::{artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input},
        native_engine::{int4::PackedInt4Kernel,
            strict_int4::{Int4ModelWeights, Int4ModelLoadBudget, Int4ModelLoadRequirement,
                Int4MemoryRequirement, Int4MemoryBudget}}};
    let directory = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
    let source = prepare_nanbeige42_input(directory, ConversionSourceManifest::load_pinned(manifest).unwrap(), false, 65536).unwrap();
    let load = Int4ModelLoadRequirement::for_group_width(32).unwrap();
    let weights = Int4ModelWeights::from_prepared(&source, 32, Int4ModelLoadBudget {
        max_payload_bytes: load.peak_payload_bytes().unwrap(), max_source_values: load.source_values,
        max_groups: load.groups, max_shard_revalidation_bytes: load.shard_revalidation_bytes,
    }, &mut Continue).unwrap();
    let row = Int4MemoryRequirement::for_context(1024, 32).unwrap();
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let mut request = request(mode);
        request.document = "é first.\nSecond part.".to_owned();
        request.labels.truncate(2);
        request.budget.max_kv_bytes = row.kv_bytes * request.labels.len() as u64;
        let planner = Int4ClassificationPlanner::for_model(weights.identity(), request.budget,
            ClassificationLimits::default()).unwrap();
        let mut limits = mapping(); limits.chunks.max_chunk_bytes = 12; limits.chunks.max_chunks = 8;
        let plan = planner.plan_document_with_control(&request, limits, &mut Continue).unwrap();
        let admitted = plan.identities().unwrap();
        assert!(plan.chunk_count() > 1);
        let mut serial = StrictInt4Engine::new(&weights, 1024, Int4MemoryBudget {
            max_kv_bytes: row.kv_bytes, max_rope_bytes: row.rope_bytes,
            max_scratch_payload_bytes: row.scratch_payload_bound }).unwrap();
        let expected = plan.execute_with_control(&admitted, &mut serial, Int4ScoringBudget {
            native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: row.kv_bytes }, &mut Continue).unwrap();
        assert!(!serial.is_poisoned()); assert!(serial.kv_cache().all_slots_have_len(0)); drop(serial);
        let peak = Int4MemoryRequirement::for_cohort(&vec![1024; 2 * plan.head_count()], 32).unwrap();
        let mut budget = cohort_budget(&plan);
        budget.memory = Int4MemoryBudget { max_kv_bytes: peak.kv_bytes, max_rope_bytes: peak.rope_bytes,
            max_scratch_payload_bytes: peak.scratch_payload_bound };
        budget.max_task_workspace_bytes = crate::native_engine::strict_int4::scoring::cohort::required_task_workspace_bytes(
            plan.planned_work(), 2 * plan.head_count(), budget.max_output_bytes).unwrap();
        let actual = plan.execute_cohorts_with_control(&admitted, &weights, PackedInt4Kernel::scalar(), 2, budget,
            &mut Continue).unwrap();
        assert_eq!(actual.output, expected);
        assert_eq!(actual.chunk_cohorts, plan.chunk_count().div_ceil(2));
        assert!(actual.group_steps < plan.planned_work().forward_positions);
        plan.preflight_metadata().verify_completed(&actual.output).unwrap();
    }
}
