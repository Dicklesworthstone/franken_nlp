//! Real pinned plans plus PRIVATE scripted corruption tests. Not neural or
//! recognition/summary-quality evidence; no public fake backend is provided.
use super::*;
use std::{cell::Cell, collections::VecDeque, rc::Rc};
use crate::{
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{ir::ScoreSpace, summarize::{CitationGuarantee, CitedBullet, SourceCitation, SummarySemanticSupport}},
    tokenizer::pinned_controls,
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, scan_occurrences},
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> SourceTaskPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    SourceTaskPlanner::pinned(controls, eos).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(p: &SourceTaskPlanner, task: &SourceMapTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"summary-int8-model-free-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: task.request(String::new(), budget()).task().spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn mapping(fan_in: usize) -> Int8SourceMapLimits {
    Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
            context_tokens: 8192, reserved_tokens: 1024, max_chunks: 64, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: fan_in,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work::for_sequence(0, 16_384, 16_384 * 166_144).unwrap(),
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000,
    }
}
fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, fan_in: usize) -> PreparedInt8SourceMap<'s> {
    let task = SourceMapTask::Summarize(SummaryOptions::default());
    let id = identity(p, &task);
    p.plan_int8_map_with_control(source, &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping(fan_in), &mut Continue).unwrap()
}
fn limits() -> Int8SummaryLimits { Int8SummaryLimits { max_bullets: 1, ..Int8SummaryLimits::default() } }
fn raw(chunk: &SourceChunk<'_>) -> SummaryResult {
    let quote = chunk.text().chars().next().unwrap().to_string();
    let spans = scan_occurrences(chunk.text(), &quote, &mut GroundingBudget::default()).unwrap();
    let citation = SourceCitation { quote, occurrence: if spans.len() == 1 { SourceOccurrence::Anchored }
        else { SourceOccurrence::Ambiguous }, spans };
    SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        bullets: vec![CitedBullet { text: format!("local-{}", chunk.id()), citations: vec![citation.clone()] },
            CitedBullet { text: "common".to_owned(), citations: vec![citation] }],
        generated_token_ids: vec![1, 0], forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 }
}
struct Script {
    outputs: VecDeque<SummaryResult>, calls: Rc<Cell<usize>>, checkpoints: Rc<Cell<usize>>,
    corrupt_axis: Option<usize>, cancel_at: Option<usize>,
}
fn script(prepared: &PreparedInt8SourceMap<'_>) -> Script {
    Script { outputs: prepared.chunks.chunks().iter().map(raw).collect(), calls: Rc::new(Cell::new(0)),
        checkpoints: Rc::new(Cell::new(0)), corrupt_axis: None, cancel_at: None }
}
impl SourceDriver for Script {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> {
        let n = self.checkpoints.get() + 1; self.checkpoints.set(n);
        if self.cancel_at == Some(n) { return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        Ok(())
    }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        self.calls.set(self.calls.get() + 1);
        let mut result = self.outputs.pop_front().ok_or(Int8SourceError::InvalidResult)?;
        let work = constrained_int8::planned_work(plan.prompt_tokens(), result.generated_token_ids.len()).unwrap();
        result.forward_positions = work.forward_positions; result.projected_logits = work.projected_logits;
        let mut model_work = work;
        if let Some(axis) = self.corrupt_axis {
            match axis { 0 => model_work.forward_positions += 1, 1 => model_work.projected_logits += 1,
                2 => model_work.attention_pairs += 1, 3 => model_work.projections.dot_products += 1,
                _ => model_work.projections.multiply_accumulates += 1 }
        }
        Ok(Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
            result: SourceTaskResult::Summarize(result), model_work })
    }
}
fn run(prepared: PreparedInt8SourceMap<'_>, limits: Int8SummaryLimits, driver: Script)
    -> Result<Int8CorpusSummaryRun, Int8CorpusSummaryError> {
    let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    prepared.execute_summary_with_driver(&admitted, limits, driver)
}

#[test]
fn global_top_k_keeps_late_support_and_every_original_unicode_occurrence() {
    let p = planner(); let source = "éAéAéAéA"; let prepared = prepare(&p, source, 2);
    assert_eq!(prepared.chunk_count(), 2);
    let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    let result = run(prepared, limits(), driver).unwrap();
    assert_eq!(calls.get(), 2); assert_eq!(result.summary.bullets.len(), 1);
    let bullet = &result.summary.bullets[0];
    assert_eq!(bullet.text, "common"); assert_eq!(bullet.rank_sum, 4); assert_eq!(bullet.evidence.len(), 2);
    assert_eq!(result.summary.omitted_bullets, 2); assert_eq!(result.summary.mapped_chunks, 2);
    assert_eq!(result.source_span.byte_end, source.len()); assert_eq!(result.source_span.scalar_end, source.chars().count());
    let second = &bullet.evidence[1].citations[0];
    assert_eq!(second.spans[0].byte_start, 6); assert_eq!(second.spans[0].scalar_start, 4);
    for evidence in &bullet.evidence { for citation in &evidence.citations { for span in &citation.spans {
        assert_eq!(&source[span.byte_start..span.byte_end], citation.quote);
        assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
    } } }
    assert_eq!(result.mask_node_visit_charge, 40); assert_eq!(result.reserved_mask_node_visits, 2000);
    assert!(within(result.model_work, result.planned_model_work));
    assert_eq!(result.summary.projected_logits, result.model_work.projected_logits);
    assert_eq!(result.model_work.projected_logits, 4 * 166_144);
}
#[test]
fn final_selection_is_identical_across_fanins_and_multi_level_trees() {
    let p = planner(); let source = "éAéAéAéAéAéAéAéA";
    let mut baseline = None;
    for fan_in in [2, 3, 8] {
        let prepared = prepare(&p, source, fan_in); let driver = script(&prepared);
        let result = run(prepared, limits(), driver).unwrap();
        let json = crate::canonjson::canonical_string(&result.summary).unwrap();
        if let Some(expected) = &baseline { assert_eq!(&json, expected); } else { baseline = Some(json); }
        if fan_in == 2 { assert!(result.reduction_levels > 1); }
    }
}
#[test]
fn duplicate_bullets_never_inflate_chunk_votes_and_all_proposals_are_verified() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared);
    for result in &mut driver.outputs { result.bullets.push(result.bullets[1].clone()); }
    let result = run(prepared, limits(), driver).unwrap();
    assert_eq!(result.summary.bullets[0].evidence.len(), 2); assert_eq!(result.summary.bullets[0].rank_sum, 4);
    let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared);
    let last = driver.outputs.back_mut().unwrap(); let mut corrupt = last.bullets[1].clone();
    corrupt.citations[0].spans.pop(); last.bullets.push(corrupt);
    assert!(run(prepared, limits(), driver).is_err());
}
#[test]
fn missing_or_malformed_final_citations_abort_the_whole_document() {
    let p = planner();
    for axis in 0..4 {
        let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        let citation = &mut driver.outputs.back_mut().unwrap().bullets[1].citations[0];
        match axis { 0 => { citation.spans.pop(); }, 1 => citation.spans[0].scalar_end += 1,
            2 => citation.occurrence = SourceOccurrence::Anchored, _ => citation.quote = "absent".to_owned() }
        assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 2);
    }
}
#[test]
fn last_identity_and_all_five_receipt_axes_are_checked() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    let calls = Rc::clone(&driver.calls); let mut admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    admitted.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"foreign last plan");
    assert!(prepared.execute_summary_with_driver(&admitted, limits(), driver).is_err()); assert_eq!(calls.get(), 0);
    for axis in 0..5 {
        let prepared = prepare(&p, "éAéA", 2); let mut driver = script(&prepared); driver.corrupt_axis = Some(axis);
        assert!(run(prepared, limits(), driver).is_err());
    }
}
#[test]
fn eager_profile_cannot_be_relabelled_as_int8() {
    let p = planner(); let prepared = prepare(&p, "éAéA", 2); let mut driver = script(&prepared);
    driver.outputs[0].numerics_profile = crate::native_engine::hf_bf16_eager::HF_BF16_EAGER_PROFILE.to_owned();
    assert!(run(prepared, limits(), driver).is_err());
}
#[test]
fn scan_work_is_shared_and_top_k_does_not_hide_an_oversized_complete_union() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    let scan_steps = run(prepared, limits(), driver).unwrap().verification_scan_steps;
    assert!(scan_steps > 1);
    for axis in 0..3 {
        let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared); let mut config = limits();
        match axis { 0 => config.aggregation.max_scan_steps = scan_steps - 1,
            1 => config.aggregation.max_unique_bullets = 2, _ => config.aggregation.max_citations = 3 }
        assert!(run(prepared, config, driver).is_err());
    }
}
#[test]
fn final_cancellation_preserves_kind_after_every_map_and_reduction() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    let checkpoints = Rc::clone(&driver.checkpoints);
    run(prepared, limits(), driver).unwrap(); let final_checkpoint = checkpoints.get();
    let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared);
    driver.cancel_at = Some(final_checkpoint); let calls = Rc::clone(&driver.calls);
    let error = match run(prepared, limits(), driver) { Err(error) => error, Ok(_) => panic!("must cancel") };
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls.get(), 2);
}
#[test]
fn singleton_and_empty_bullet_maps_preserve_complete_source_coverage() {
    let p = planner(); let prepared = prepare(&p, "éAéA", 2); let driver = script(&prepared);
    let result = run(prepared, limits(), driver).unwrap();
    assert_eq!(result.reduce_calls, 0); assert_eq!(result.reduction_levels, 0);
    let prepared = prepare(&p, "éAéAéAéA", 2); let mut driver = script(&prepared);
    for output in &mut driver.outputs { output.bullets.clear(); }
    let result = run(prepared, limits(), driver).unwrap();
    assert!(result.summary.bullets.is_empty()); assert_eq!(result.summary.mapped_chunks, 2);
    assert_eq!(result.source_span.byte_end, 12); assert!(result.model_work.forward_positions > 0);
}
#[test]
fn non_summary_plans_and_mixed_options_are_refused_before_native_calls() {
    let p = planner(); let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&p, &task);
    let prepared = p.plan_int8_map_with_control("éAéA", &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping(2), &mut Continue).unwrap();
    let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 0);
    let mut prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    let Finalizer::Summarize(options) = &mut prepared.plans.last_mut().unwrap().finalizer else { panic!("summary") };
    options.max_bullets += 1;
    assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 0);
}
#[test]
fn whole_envelope_is_bounded_and_no_private_prompt_or_token_transcript_escapes() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    let result = run(prepared, limits(), driver).unwrap(); let bytes = crate::canonjson::canonical_bytes(&result).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    for field in ["generated_token_ids", "prompt_digest", "logical_model_digest"] { assert!(!text.contains(field)); }
    let prepared = prepare(&p, "éAéAéAéA", 2); let driver = script(&prepared);
    assert!(run(prepared, Int8SummaryLimits { max_result_bytes: bytes.len() - 1, ..limits() }, driver).is_err());
}
#[test]
fn invalid_summary_limits_are_rejected_before_any_native_call() {
    let p = planner();
    for axis in 0..4 {
        let prepared = prepare(&p, "éAéA", 2); let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        let mut config = limits();
        match axis { 0 => config.max_bullets = 0, 1 => config.max_result_bytes = 0,
            2 => config.aggregation.max_value_bytes = 8 << 20, _ => config.max_result_bytes = 8 << 20 }
        assert!(run(prepared, config, driver).is_err()); assert_eq!(calls.get(), 0);
    }
}
