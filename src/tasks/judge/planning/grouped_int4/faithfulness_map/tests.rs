//! Pinned compilation and original finalizers driven by private synthetic
//! logits. These regressions do not establish native parity or model quality.
use super::*;
use std::sync::OnceLock;
use crate::{native_engine::{lmhead::scoring::{CandidateLogits, ProjectionRows},
        portable_int8::ProjectionWork, strict_int4::{Int4MemoryBudget, Int4MemoryRequirement,
            scoring::cohort::{self as native_cohort, task::fixtures}}},
    tasks::{ir::PromptSegmentKind, judge::{FaithfulnessRelation, FaithfulnessAggregateAbstention}},
    tokenizer::pinned_controls};
use fixtures::Continue;

mod binding;

fn fixture() -> (&'static JudgePlanner, &'static ExecutionIdentity) {
    static FIXTURE: OnceLock<(JudgePlanner, ExecutionIdentity)> = OnceLock::new();
    let (planner, identity) = FIXTURE.get_or_init(|| {
        let controls = pinned_controls::pinned().unwrap();
        let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let planner = JudgePlanner::pinned(controls.template_controls(), eos).unwrap();
        let identity = fixtures::identity("judge-v1", planner.tokenizer_digest(), *planner.template_digest());
        (planner, identity)
    });
    (planner, identity)
}
fn request() -> JudgeRequest {
    JudgeRequest::Faithfulness { source: "é\r\n𐐀 one <think> literal. Second chunk.\nTail.".to_owned(),
        claim: "The claim <|im_start|> remains data.".to_owned(),
        policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 600000, minimum_margin_milli: 100,
            evidence_window_bytes: 8, max_evidence_windows: 31, max_evidence_spans: 31 }, budget: fixtures::budget() }
}
fn mapping() -> Int4FaithfulnessMapLimits {
    Int4FaithfulnessMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 16,
        max_chunk_tokens: 1024, context_tokens: 1024, reserved_tokens: 1, max_chunks: 64, max_tokenizer_calls: 4096 },
        max_model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        max_retained_result_bytes: 4 * 1024 * 1024, max_result_bytes: 8 * 1024 * 1024,
        aggregation: FaithfulnessAggregationPolicy::default() }
}
fn prepare<'a>(request: &'a JudgeRequest, limits: Int4FaithfulnessMapLimits)
    -> Result<PreparedInt4FaithfulnessMap<'a>, Int4FaithfulnessMapError> {
    let (planner, identity) = fixture(); let planning = JudgeLimits::default();
    let JudgeRequest::Faithfulness { claim, policy, budget, .. } = request else { return Err(Int4FaithfulnessMapError::Limits); };
    let capacity = planner.faithfulness.source_capacity(claim, *policy, *budget, *budget,
        planning, limits.chunks.context_tokens)?;
    PreparedInt4FaithfulnessMap::compile(request, limits, capacity, 32, &mut Continue, |row, control|
        planner.compile_int4(row, &PlanContext::new(identity, *budget).unwrap(), planning, 32, control))
}
struct Favor(u32);
impl CandidateLogits for Favor {
    type Error = &'static str;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { return Err("complete denominator required"); };
        let mut values = vec![0.0; vocabulary_size];
        if prefix.is_empty() { values[self.0 as usize] = 12.0; }
        Ok(values)
    }
}
fn run(plan: &PreparedInt4Judge, favorite: Option<&str>) -> Int4JudgeRun {
    plan.evaluate_heads(&mut Continue, |_, head, schedule, _| {
        let scores = if let Some(favorite) = favorite {
            let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else { panic!("finite language"); };
            let token = candidates.iter().find(|c| c.id() == favorite).unwrap().continuation().token_ids()[0];
            head.scorer.score(&mut Favor(token), ScoringMode::FullVocabulary).unwrap()
        } else { head.scorer.score(&mut fixtures::Constant, ScoringMode::FullVocabulary).unwrap() };
        Ok(Int4CandidateRun { schema_version: 1, execution: INT4_SCORING_EXECUTION.to_owned(),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: 32,
            scores, model_work: schedule.model, rewound_positions: schedule.rewound })
    }).unwrap()
}
fn finish(plan: &PreparedInt4FaithfulnessMap<'_>) -> Result<Int4FaithfulnessMapRun, Int4FaithfulnessMapError> {
    plan.collect_with_control(&mut Continue, |_, row, _| Ok(run(row, None)))
}
fn cohort_budget(plan: &PreparedInt4FaithfulnessMap<'_>) -> Int4FaithfulnessMapCohortBudget {
    Int4FaithfulnessMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()),
        memory: Int4MemoryBudget { max_kv_bytes: u64::MAX, max_rope_bytes: u64::MAX,
            max_scratch_payload_bytes: u64::MAX }, max_task_workspace_bytes: u64::MAX,
        max_output_bytes: 8 * 1024 * 1024 }
}
fn finish_window(plan: &PreparedInt4FaithfulnessMap<'_>, admitted: &[ExecutionIdentity], window: &cohort::Window)
    -> Result<native_cohort::Int4ScoredCohort<Vec<Int4JudgeRun>>, Int4FaithfulnessMapError> {
    let documents: Vec<_> = (window.start..window.end).map(|index| (&plan.plans[index], &admitted[index])).collect();
    let budget = native_cohort::Int4ScoringCohortBudget { native: Int4RunBudget::exact(window.work),
        max_kv_bytes: u64::MAX, max_output_bytes: 8 * 1024 * 1024 };
    let (schedules, work) = native_cohort::task::document_layout(&documents, &window.contexts, budget)?;
    let native = fixtures::run(documents.iter().flat_map(|(plan, _)| plan.inner.executable.bundle().heads.iter()
        .zip(&plan.schedules).map(|(head, &schedule)| (head.scorer.score(&mut fixtures::Constant,
            ScoringMode::FullVocabulary).unwrap(), schedule))));
    Ok(native_cohort::task::finish_documents(&documents, native, &schedules, work, budget.max_output_bytes, &mut Continue)?)
}

#[test]
fn claim_scaffold_and_verbalizer_reduce_exact_source_capacity() {
    let (planner, _) = fixture(); let mut request = request();
    let JudgeRequest::Faithfulness { source, claim, policy, budget } = &mut request else { unreachable!() };
    policy.evidence_window_bytes = 4096;
    let limits = JudgeLimits::default();
    let first = planner.faithfulness.source_capacity(claim, *policy, *budget, *budget, limits, 1024).unwrap();
    claim.push_str("é");
    let smaller = planner.faithfulness.source_capacity(claim, *policy, *budget, *budget, limits, 1024).unwrap();
    assert_eq!(first - smaller, "é".len());
    let overhead: usize = planner.faithfulness.fragments.iter().map(Vec::len).sum();
    assert_eq!(smaller + claim.len() + overhead + 1, 1024);
    source.clear(); source.push_str(&"x".repeat(smaller + 11));
    let mut limits = mapping(); limits.chunks.max_chunk_bytes = 4096;
    let plan = prepare(&request, limits).unwrap();
    assert_eq!(plan.chunks.limits().max_chunk_tokens, smaller);
    assert!(plan.chunks.chunks().iter().all(|chunk| chunk.tokens() <= smaller));
    assert!(plan.plans.iter().all(|p| p.required_context() <= 1024));
    assert_eq!(plan.source.len(), smaller + 11);
}

#[test]
fn source_partition_and_every_claim_preserve_original_utf8_and_literal_controls() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap();
    let JudgeRequest::Faithfulness { source, claim, .. } = &request else { unreachable!() };
    let (compiler, _) = fixture(); let mut rebuilt = String::new();
    assert!(plan.chunk_count() > 1); assert!(plan.max_head_count() > 1);
    assert_eq!(plan.head_count(), plan.plans.iter().map(PreparedInt4Judge::head_count).sum::<usize>());
    for (chunk, row) in plan.chunks.chunks().iter().zip(&plan.plans) {
        rebuilt.push_str(chunk.text()); assert_eq!(chunk.tokens(), chunk.text().len());
        assert!(!chunk.text().ends_with('\r'));
        for (index, head) in row.inner.executable.bundle().heads.iter().enumerate() {
            let documents: Vec<_> = head.ir.prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document).collect();
            assert_eq!(documents.len(), 2);
            if index == 0 {
                assert_eq!(compiler.tokenizer.tokenizer().decode_bytes(documents[0].token_ids()).unwrap(), chunk.text().as_bytes());
            }
            assert_eq!(compiler.tokenizer.tokenizer().decode_bytes(documents[1].token_ids()).unwrap(), claim.as_bytes());
            assert!(documents.iter().flat_map(|s| s.token_ids()).all(|id| !compiler.controls.contains(*id)));
        }
    }
    assert_eq!(&rebuilt, source);
    let result = finish(&plan).unwrap(); plan.preflight_metadata().verify_completed(&result).unwrap();
    assert_eq!(result.source_span.byte_end, source.len()); assert_eq!(result.source_span.scalar_end, source.chars().count());
    assert!(!result.single_context_equivalence); assert_eq!(result.aggregate.relation, None);
    assert_eq!(result.aggregate.abstention, Some(FaithfulnessAggregateAbstention::InsufficientClassifiedCoverage));
}

#[test]
fn decisive_chunk_evidence_is_lifted_without_averaging_and_conflict_abstains() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap();
    let accepted = plan.collect_with_control(&mut Continue, |index, row, _|
        Ok(run(row, Some(if index == 0 { "entailed" } else { "unsupported" })))).unwrap();
    assert_eq!(accepted.aggregate.relation, Some(FaithfulnessRelation::Entailed));
    assert!(!accepted.aggregate.evidence.is_empty());
    for evidence in &accepted.aggregate.evidence {
        assert_eq!(&plan.source[evidence.span.byte_start..evidence.span.byte_end], evidence.quote);
        assert_eq!(evidence.span.scalar_start, plan.source[..evidence.span.byte_start].chars().count());
    }
    let conflict = plan.collect_with_control(&mut Continue, |index, row, _|
        Ok(run(row, Some(if index == 0 { "entailed" } else { "contradicted" })))).unwrap();
    assert_eq!(conflict.aggregate.relation, None);
    assert_eq!(conflict.aggregate.abstention, Some(FaithfulnessAggregateAbstention::ConflictingEvidence));
    assert!(conflict.aggregate.evidence.iter().any(|e| e.relation == FaithfulnessRelation::Contradicted));
}

#[test]
fn all_five_work_axes_cover_every_source_and_evidence_head_without_renewal() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap();
    let work = plan.plans.iter().try_fold(Int4Work::default(), |sum, row| sum.checked_add(row.planned_work())).unwrap();
    assert_eq!(work, plan.planned_work());
    for axis in 0..5 {
        let mut limits = mapping(); limits.max_model_work = work;
        match axis { 0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1, 2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            _ => limits.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(matches!(prepare(&request, limits), Err(Int4FaithfulnessMapError::Work)));
    }
    let mut exact = mapping(); exact.max_model_work = work;
    assert_eq!(prepare(&request, exact).unwrap().planned_work(), work);
}

#[test]
fn chunk_and_evidence_limits_refuse_tail_loss_and_excessive_combined_input() {
    let request = request();
    for axis in 0..3 {
        let mut limits = mapping();
        match axis { 0 => limits.chunks.max_input_bytes = 1, 1 => limits.chunks.max_chunks = 1,
            _ => limits.chunks.max_tokenizer_calls = 1 }
        assert!(prepare(&request, limits).is_err());
    }
    let mut wrong = request.clone();
    if let JudgeRequest::Faithfulness { policy, .. } = &mut wrong { policy.max_evidence_windows = 0; }
    assert!(prepare(&wrong, mapping()).is_err());
}

#[test]
fn punctuation_and_utf8_cannot_exhaust_a_valid_chunks_evidence_window_budget() {
    let mut request = request();
    if let JudgeRequest::Faithfulness { source, policy, .. } = &mut request {
        *source = "abcd.efgh.ijklmno.é.𐐀.xy.末尾.".to_owned();
        policy.evidence_window_bytes = 8; policy.max_evidence_windows = 2; policy.max_evidence_spans = 2;
    }
    let mut limits = mapping(); limits.chunks.max_chunk_bytes = 64;
    let plan = prepare(&request, limits).unwrap();
    assert!(plan.chunk_count() > 1);
    for row in &plan.plans {
        let Executable::Faithfulness(faithfulness) = &row.inner.executable else { unreachable!() };
        assert!(faithfulness.bundle.heads.len() <= 3);
    }
    let JudgeRequest::Faithfulness { source, .. } = &request else { unreachable!() };
    assert_eq!(plan.chunks.chunks().iter().map(|chunk| chunk.text()).collect::<String>(), *source);
}

#[test]
fn late_failure_and_cancellation_never_return_partial_document_relations() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap(); let mut calls = 0;
    let error = plan.collect_with_control(&mut Continue, |index, row, _| {
        calls += 1; if index == 1 { return Err(Int4JudgeError::Cancelled(DecodeCancellationKind::Deadline).into()); }
        Ok(run(row, Some("entailed")))
    }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls, 2);
    struct Stop(bool);
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.0.then_some(DecodeCancellationKind::User) } }
    let error = plan.collect_with_control(&mut Stop(false), |_, row, control| {
        let run = run(row, Some("entailed")); control.0 = true; Ok(run)
    }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::User));
}

#[test]
fn retained_chunk_bytes_and_complete_outer_result_have_separate_exact_caps() {
    let request = request(); let mut plan = prepare(&request, mapping()).unwrap(); let baseline = finish(&plan).unwrap();
    let bytes = canonjson::canonical_bytes(&baseline).unwrap().len() as u64;
    plan.limits.max_retained_result_bytes = baseline.retained_result_bytes; plan.limits.max_result_bytes = bytes;
    assert_eq!(finish(&plan).unwrap(), baseline);
    plan.limits.max_result_bytes -= 1;
    assert!(matches!(finish(&plan), Err(Int4FaithfulnessMapError::Output)));
    plan.limits.max_result_bytes = bytes; plan.limits.max_retained_result_bytes -= 1;
    assert!(matches!(finish(&plan), Err(Int4FaithfulnessMapError::Output)));
}

#[test]
fn real_task_cohort_finalizers_preserve_serial_complete_document_results() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap(); let admitted = plan.identities().unwrap();
    let expected = finish(&plan).unwrap();
    for chunks in [1, 2, plan.chunk_count()] {
        let windows = plan.cohort_windows(&admitted, chunks, cohort_budget(&plan), &mut Continue).unwrap();
        for window in &windows {
            let expected_heads: usize = plan.plans[window.start..window.end].iter().map(PreparedInt4Judge::head_count).sum();
            assert_eq!(window.contexts.len(), expected_heads);
        }
        let actual = plan.collect_cohorts_with_control(&windows, &mut Continue,
            |window, _| finish_window(&plan, &admitted, window)).unwrap();
        assert_eq!(actual.output, expected); assert_eq!(actual.chunk_cohorts, plan.chunk_count().div_ceil(chunks));
        plan.preflight_metadata().verify_completed_cohorts(&actual, chunks).unwrap();
    }
}

#[test]
fn every_future_window_checks_identity_work_and_ragged_head_ceiling() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap(); let admitted = plan.identities().unwrap();
    let mut changed = admitted.clone(); changed.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"changed tail");
    assert!(plan.cohort_windows(&changed, 2, cohort_budget(&plan), &mut Continue).is_err());
    let windows = plan.cohort_windows(&admitted, 2, cohort_budget(&plan), &mut Continue).unwrap();
    let mut short = cohort_budget(&plan); short.native = Int4RunBudget::exact(windows[0].work);
    assert!(plan.cohort_windows(&admitted, 2, short, &mut Continue).is_err());
    let mut many = request.clone();
    if let JudgeRequest::Faithfulness { source, policy, .. } = &mut many {
        *source = "x".repeat(128); policy.evidence_window_bytes = 1;
    }
    let many = prepare(&many, mapping()).unwrap();
    assert_eq!(many.max_head_count(), 17);
    assert!(many.validate_cohort_chunks(3).is_ok());
    assert!(matches!(many.validate_cohort_chunks(4), Err(Int4FaithfulnessMapError::Limits)));
    for chunks in [0, 65, usize::MAX] { assert!(many.validate_cohort_chunks(chunks).is_err()); }
}

#[test]
fn simultaneous_kv_rope_scratch_and_task_workspace_are_all_preflighted() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap(); let admitted = plan.identities().unwrap();
    let mut budget = cohort_budget(&plan);
    let windows = plan.cohort_windows(&admitted, 2, budget, &mut Continue).unwrap();
    let required: Vec<_> = windows.iter().map(|w| Int4MemoryRequirement::for_cohort(&w.contexts, 32).unwrap()).collect();
    budget.memory.max_kv_bytes = required.iter().map(|r| r.kv_bytes).max().unwrap();
    budget.memory.max_rope_bytes = required.iter().map(|r| r.rope_bytes).max().unwrap();
    budget.memory.max_scratch_payload_bytes = required.iter().map(|r| r.scratch_payload_bound).max().unwrap();
    budget.max_task_workspace_bytes = windows.iter().map(|w| native_cohort::required_task_workspace_bytes(
        w.work, w.contexts.len(), budget.max_output_bytes).unwrap()).max().unwrap();
    assert!(plan.cohort_windows(&admitted, 2, budget, &mut Continue).is_ok());
    for axis in 0..4 {
        let mut short = budget;
        match axis { 0 => short.memory.max_kv_bytes -= 1, 1 => short.memory.max_rope_bytes -= 1,
            2 => short.memory.max_scratch_payload_bytes -= 1, _ => short.max_task_workspace_bytes -= 1 }
        assert!(plan.cohort_windows(&admitted, 2, short, &mut Continue).is_err());
    }
}

#[test]
fn malformed_late_native_window_never_becomes_an_unsupported_relation() {
    let request = request(); let plan = prepare(&request, mapping()).unwrap(); let admitted = plan.identities().unwrap();
    let windows = plan.cohort_windows(&admitted, 2, cohort_budget(&plan), &mut Continue).unwrap();
    for fault in 0..4 {
        assert!(plan.collect_cohorts_with_control(&windows, &mut Continue, |window, _| {
            let mut run = finish_window(&plan, &admitted, window)?;
            if window.end == plan.chunk_count() {
                match fault { 0 => { run.output.pop(); }, 1 => run.group_steps = 0,
                    2 => run.execution.push('x'), _ => run.output.last_mut().unwrap().model_work.projected_logits += 1 }
            }
            Ok(run)
        }).is_err());
    }
}

#[test]
#[ignore = "requires controller-admitted pinned source and full-model INT4 faithfulness map parity"]
fn native_faithfulness_chunk_cohorts_match_serial_complete_document_and_evidence() {
    use crate::{artifact::converter::{ConversionSourceManifest, prepare_nanbeige42_input},
        native_engine::{int4::PackedInt4Kernel, strict_int4::{Int4ModelWeights,
            Int4ModelLoadBudget, Int4ModelLoadRequirement}}};
    let directory = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
    let source = prepare_nanbeige42_input(directory, ConversionSourceManifest::load_pinned(manifest).unwrap(), false, 65536).unwrap();
    let load = Int4ModelLoadRequirement::for_group_width(32).unwrap();
    let weights = Int4ModelWeights::from_prepared(&source, 32, Int4ModelLoadBudget {
        max_payload_bytes: load.peak_payload_bytes().unwrap(), max_source_values: load.source_values,
        max_groups: load.groups, max_shard_revalidation_bytes: load.shard_revalidation_bytes,
    }, &mut Continue).unwrap();
    let memory = Int4MemoryRequirement::for_context(1024, 32).unwrap();
    let mut request = request();
    if let JudgeRequest::Faithfulness { source, claim, budget, .. } = &mut request {
        *source = "Alice left.\nBob stayed.\nCafé open.".to_owned(); *claim = "Alice left.".to_owned();
        budget.max_kv_bytes = memory.kv_bytes * MAX_FAITHFULNESS_MAP_HEADS as u64;
    }
    let planner = Int4JudgePlanner::for_model(weights.identity(), request.budget(), JudgeLimits::default()).unwrap();
    let mut limits = mapping(); limits.chunks.max_chunk_bytes = 12; limits.chunks.max_chunks = 8;
    let plan = planner.plan_faithfulness_document_with_control(&request, limits, &mut Continue).unwrap();
    let admitted = plan.identities().unwrap(); assert!(plan.chunk_count() > 1);
    let mut engine = StrictInt4Engine::new(&weights, 1024, Int4MemoryBudget {
        max_kv_bytes: memory.kv_bytes, max_rope_bytes: memory.rope_bytes,
        max_scratch_payload_bytes: memory.scratch_payload_bound }).unwrap();
    let expected = plan.execute_with_control(&admitted, &mut engine, Int4ScoringBudget {
        native: Int4RunBudget::exact(plan.planned_work()), max_kv_bytes: memory.kv_bytes }, &mut Continue).unwrap();
    assert!(!engine.is_poisoned()); assert!(engine.kv_cache().all_slots_have_len(0)); drop(engine);
    let heads = 2 * plan.max_head_count();
    let native = Int4MemoryRequirement::for_cohort(&vec![1024; heads], 32).unwrap();
    let mut budget = cohort_budget(&plan);
    budget.memory = Int4MemoryBudget { max_kv_bytes: native.kv_bytes, max_rope_bytes: native.rope_bytes,
        max_scratch_payload_bytes: native.scratch_payload_bound };
    budget.max_task_workspace_bytes = native_cohort::required_task_workspace_bytes(plan.planned_work(),
        heads, budget.max_output_bytes).unwrap();
    let actual = plan.execute_cohorts_with_control(&admitted, &weights, PackedInt4Kernel::scalar(), 2,
        budget, &mut Continue).unwrap();
    assert_eq!(actual.output, expected); assert_eq!(actual.chunk_cohorts, plan.chunk_count().div_ceil(2));
    assert!(actual.group_steps < plan.planned_work().forward_positions);
    plan.preflight_metadata().verify_completed_cohorts(&actual, 2).unwrap();
}
