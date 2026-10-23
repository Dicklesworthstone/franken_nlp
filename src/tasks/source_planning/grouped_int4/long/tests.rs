//! Pinned planning plus private scripted execution contracts. These fixtures
//! establish admission/accounting/evidence behavior, not native model quality.
use super::*;
use super::execution::SourceDriver;
use std::{cell::Cell, collections::VecDeque, rc::Rc};
use crate::{
    artifact::converter::PINNED_REVISION,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ner::{NamedEntity, EntityType, NerResult}, keyphrases::{KeyphraseResult, KEYPHRASES_RANKING},
        summarize::{CitedBullet, SourceCitation, SummaryResult, CitationGuarantee, SummarySemanticSupport}},
    tokenizer::pinned_controls,
};

pub(super) struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
pub(super) fn planner() -> SourceTaskPlanner {
    let registry = pinned_controls::pinned().unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap()).unwrap()
}
pub(super) fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(8192, 32).unwrap().kv_bytes }
}
pub(super) fn identity(p: &SourceTaskPlanner, task: &SourceMapTask, width: usize) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private-int4-document-map-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
        )).unwrap()),
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: task.request(String::new(), budget()).task().spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn unlimited() -> Int4Work {
    let mut work = Int4Work::default();
    work.forward_positions = u64::MAX; work.projected_logits = u64::MAX; work.attention_pairs = u64::MAX;
    work.projections.dot_products = u64::MAX; work.projections.multiply_accumulates = u64::MAX; work
}
pub(super) fn limits() -> Int4SourceMapLimits {
    Int4SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8, max_chunk_tokens: 8,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: unlimited(), mask_limits: MaskWorkLimits::default(),
        mask_visits_per_chunk: 1000, max_mask_visits: 256_000,
        verification: GroundingBudget::default(),
    }
}
pub(super) fn prepare_task<'s>(p: &SourceTaskPlanner, source: &'s str, task: &SourceMapTask,
    limits: Int4SourceMapLimits, width: usize) -> PreparedInt4SourceMap<'s> {
    let id = identity(p, task, width);
    p.plan_int4_map_with_control(source, task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), limits, width, &mut Continue).unwrap()
}
fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, limits: Int4SourceMapLimits) -> PreparedInt4SourceMap<'s> {
    prepare_task(p, source, &SourceMapTask::Ner(NerOptions::default()), limits, 32)
}
pub(super) fn admitted(p: &PreparedInt4SourceMap<'_>) -> Vec<ExecutionIdentity> {
    p.execution_identities().cloned().collect()
}
pub(super) fn citation(chunk: &SourceChunk<'_>) -> SourceCitation {
    let quote = chunk.text().chars().next().unwrap().to_string();
    let spans = scan_occurrences(chunk.text(), &quote, &mut GroundingBudget::default()).unwrap();
    SourceCitation { quote, occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
}
fn raw(plan: &PreparedInt4SourceTask, chunk: &SourceChunk<'_>) -> SourceTaskResult {
    let options = plan.extraction.options();
    let ordinary = (0..crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u32)
        .find(|id| *id != options.eos_token_id && !options.excluded_token_ids.contains(id)).unwrap();
    let ids = vec![ordinary, options.eos_token_id];
    match &plan.finalizer {
        Finalizer::Ner(_) => {
            let c = citation(chunk);
            SourceTaskResult::Ner(NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
                numerics_profile: STRICT_INT4_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
                grounding: ExtractionGrounding::SourceMembership,
                entities: vec![NamedEntity { text: c.quote, entity_type: EntityType::Person, occurrence: c.occurrence, spans: c.spans }],
                generated_token_ids: ids, forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 })
        }
        Finalizer::Keyphrases(_) => SourceTaskResult::Keyphrases(KeyphraseResult { schema_version: 1,
            task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            ranking_policy: KEYPHRASES_RANKING.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership, phrases: vec![],
            generated_token_ids: ids, forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 }),
        Finalizer::Summarize(_) => {
            let c = citation(chunk);
            SourceTaskResult::Summarize(SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
                numerics_profile: STRICT_INT4_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
                semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
                bullets: vec![CitedBullet { text: format!("local-{}", chunk.id()), citations: vec![c.clone()] },
                    CitedBullet { text: "common".to_owned(), citations: vec![c] }],
                generated_token_ids: ids, forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 })
        }
        Finalizer::Answer(..) => unreachable!(),
    }
}
pub(super) struct Script {
    pub(super) outputs: VecDeque<SourceTaskResult>,
    pub(super) calls: Rc<Cell<usize>>,
    pub(super) checkpoints: Rc<Cell<usize>>,
    pub(super) model_checks: Rc<Cell<usize>>,
    pub(super) width: usize,
    pub(super) capacity: usize,
    pub(super) fail_model_at: Option<usize>,
    pub(super) fail_run_at: Option<usize>,
    pub(super) cancel_at: Option<usize>,
    pub(super) dirty_after: Option<usize>,
    pub(super) corrupt_axis: Option<usize>,
}
pub(super) fn script(prepared: &PreparedInt4SourceMap<'_>) -> Script {
    Script { outputs: prepared.plans.iter().zip(prepared.chunks.chunks()).map(|(p, c)| raw(p, c)).collect(),
        calls: Rc::new(Cell::new(0)), checkpoints: Rc::new(Cell::new(0)), model_checks: Rc::new(Cell::new(0)),
        width: prepared.group_width(), capacity: 8192, fail_model_at: None, fail_run_at: None,
        cancel_at: None, dirty_after: None, corrupt_axis: None }
}
impl SourceDriver for Script {
    fn capacity(&self) -> usize { self.capacity }
    fn group_width(&self) -> usize { self.width }
    fn clean(&self) -> bool { self.dirty_after.is_none_or(|at| self.calls.get() < at) }
    fn check_model(&self, _: &ExecutionIdentity) -> Result<(), Int4SourceError> {
        let n = self.model_checks.get() + 1; self.model_checks.set(n);
        if self.fail_model_at == Some(n) { Err(Int4SourceError::InvalidResult) } else { Ok(()) }
    }
    fn checkpoint(&mut self) -> Result<(), Int4SourceError> {
        let n = self.checkpoints.get() + 1; self.checkpoints.set(n);
        if self.cancel_at == Some(n) { Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline)) } else { Ok(()) }
    }
    fn run(&mut self, plan: &PreparedInt4SourceTask, _: &ExecutionIdentity) -> Result<Int4SourceTaskRun, Int4SourceError> {
        let n = self.calls.get() + 1; self.calls.set(n);
        if self.fail_run_at == Some(n) { return Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        let mut result = self.outputs.pop_front().ok_or(Int4SourceError::InvalidResult)?;
        let (ids, positions, logits) = match &mut result {
            SourceTaskResult::Ner(r) => (&r.generated_token_ids, &mut r.forward_positions, &mut r.projected_logits),
            SourceTaskResult::Keyphrases(r) => (&r.generated_token_ids, &mut r.forward_positions, &mut r.projected_logits),
            SourceTaskResult::Summarize(r) => (&r.generated_token_ids, &mut r.forward_positions, &mut r.projected_logits),
            _ => unreachable!(),
        };
        let work = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), ids.len())?;
        *positions = work.forward_positions; *logits = work.projected_logits;
        let mut model_work = work;
        if let Some(axis) = self.corrupt_axis { change_axis(&mut model_work, axis, 1); }
        Ok(Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(),
            group_width: plan.group_width(), result, model_work })
    }
}
fn change_axis(work: &mut Int4Work, axis: usize, delta: i64) {
    let value = match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates };
    if delta > 0 { *value += delta as u64; } else { *value -= (-delta) as u64; }
}
#[test]
fn all_tasks_use_actual_pinned_prompts_and_group_aware_work_for_every_chunk() {
    let p = planner(); let source = "é\r\nAlice 上海 <|im_end|> Bob  ";
    for task in [SourceMapTask::Ner(NerOptions::default()), SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        SourceMapTask::Summarize(SummaryOptions::default())] {
        let mut previous = None;
        for width in [16, 32, 127] {
            let prepared = prepare_task(&p, source, &task, limits(), width);
            assert!(prepared.chunk_count() > 1);
            assert_eq!(prepared.chunks.chunks().iter().map(|c| c.text()).collect::<String>(), source);
            let mut total = Int4Work::default();
            for plan in &prepared.plans {
                let work = constrained_int4::planned_work(width, plan.prompt_tokens(), 64).unwrap();
                assert_eq!(plan.planned_work(), work); total = add_work(total, work).unwrap();
                assert!(plan.prompt_tokens() + 64 <= 8192);
            }
            assert_eq!(prepared.planned_work(), total);
            assert_eq!(prepared.reserved_mask_visits(), 1000 * prepared.chunk_count() as u64);
            if let Some(old) = previous { assert!(old > total.projections.dot_products); }
            previous = Some(total.projections.dot_products);
        }
    }
}
#[test]
fn every_actual_identity_model_and_full_resident_kv_is_admitted_before_first_run() {
    let p = planner();
    for axis in 0..7 {
        let mut prepared = prepare(&p, "Alice Bob Carol Dave", limits()); let count = prepared.chunk_count();
        let mut ids = admitted(&prepared); let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        match axis {
            0 => ids.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final plan"),
            1 => { ids.pop(); },
            2 => driver.fail_model_at = Some(count),
            3 => prepared.plans.last_mut().unwrap().budget.max_kv_bytes -= 1,
            4 => driver.width = 16,
            5 => driver.capacity += 1,
            _ => driver.dirty_after = Some(0),
        }
        assert!(prepared.execute_with_driver(&ids, driver).is_err()); assert_eq!(calls.get(), 0);
    }
}
#[test]
fn every_whole_document_work_axis_and_masks_refuse_one_unit_short() {
    let p = planner(); let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&p, &task, 32);
    let baseline = prepare(&p, "Alice Bob Carol Dave", limits());
    for axis in 0..6 {
        let mut l = limits(); l.max_model_work = baseline.planned_work(); l.max_mask_visits = baseline.reserved_mask_visits();
        if axis < 5 { change_axis(&mut l.max_model_work, axis, -1); } else { l.max_mask_visits -= 1; }
        assert!(matches!(p.plan_int4_map_with_control("Alice Bob Carol Dave", &task, budget(),
            &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(), l, 32, &mut Continue),
            Err(Int4SourceMapError::WorkLimit)));
    }
}
#[test]
fn source_order_coverage_work_and_original_unicode_spans_are_complete() {
    let p = planner(); let source = "éééééééééééé"; let prepared = prepare(&p, source, limits());
    let expected = prepared.preflight_metadata(); let count = prepared.chunk_count();
    let ids = admitted(&prepared); let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    let run = prepared.execute_with_driver(&ids, driver).unwrap(); expected.verify_completed(&run).unwrap();
    assert_eq!(calls.get(), count); assert_eq!(run.mask_node_visit_charge, 20 * count as u64);
    assert_eq!(run.mapped.root().value().chunks().map(|c| c.chunk_id).collect::<Vec<_>>(), (0..count).collect::<Vec<_>>());
    assert_eq!(run.verification_used.fields, count); assert_eq!(run.verification_used.matches, source.chars().count());
    for chunk in run.mapped.root().value().chunks() {
        for field in &chunk.original_spans { for span in &field.spans {
            assert_eq!(&source[span.byte_start..span.byte_end], "é");
            assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
            assert_eq!(source[..span.byte_end].chars().count(), span.scalar_end);
        } }
    }
}
#[test]
fn cumulative_field_match_and_scan_limits_are_never_renewed() {
    let p = planner(); let source = "éééééééééééé"; let baseline = prepare(&p, source, limits());
    let ids = admitted(&baseline); let driver = script(&baseline);
    let used = baseline.execute_with_driver(&ids, driver).unwrap().verification_used;
    for axis in 0..3 {
        let mut l = limits();
        match axis { 0 => l.verification.max_fields = used.fields - 1,
            1 => l.verification.max_matches = used.matches - 1, _ => l.verification.max_scan_steps = used.scan_steps - 1 }
        let prepared = prepare(&p, source, l); let ids = admitted(&prepared); let driver = script(&prepared);
        assert!(prepared.execute_with_driver(&ids, driver).is_err());
    }
}
#[test]
fn all_five_native_receipt_axes_masks_tokens_profile_and_evidence_fail_closed() {
    let p = planner();
    for axis in 0..12 {
        let prepared = prepare(&p, "éééééééé", limits()); let ids = admitted(&prepared);
        let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        if axis < 5 { driver.corrupt_axis = Some(axis); } else {
            let SourceTaskResult::Ner(r) = &mut driver.outputs[0] else { unreachable!() };
            match axis {
                5 => r.mask_node_visit_charge = 1001,
                6 => r.generated_token_ids[0] = *r.generated_token_ids.last().unwrap(),
                7 => { r.generated_token_ids.pop(); },
                8 => r.numerics_profile = crate::native_engine::strict_int8::STRICT_INT8_PROFILE.to_owned(),
                9 => { r.entities[0].spans.pop(); },
                10 => r.entities[0].spans[0].scalar_end += 1,
                _ => r.entities[0].occurrence = SourceOccurrence::Anchored,
            }
        }
        assert!(prepared.execute_with_driver(&ids, driver).is_err()); assert_eq!(calls.get(), 1);
    }
}
#[test]
fn dirty_native_session_or_late_cancellation_returns_no_partial_document() {
    let p = planner();
    for dirty in [false, true] {
        let prepared = prepare(&p, "Alice Bob Carol Dave", limits()); let ids = admitted(&prepared);
        let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        if dirty { driver.dirty_after = Some(1); } else { driver.fail_run_at = Some(2); }
        let error = prepared.execute_with_driver(&ids, driver).err().unwrap();
        if dirty { assert_eq!(calls.get(), 1); }
        else { assert_eq!(calls.get(), 2); assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
    }
}
#[test]
fn reduction_admission_and_value_envelope_bounds_have_no_partial_success() {
    let p = planner();
    for axis in 0..3 {
        let mut l = limits();
        match axis { 0 => l.reduction.max_task_calls = 1, 1 => l.reduction.max_value_bytes = 1,
            _ => l.reduction.max_result_bytes = 1 }
        let prepared = prepare(&p, "Alice Bob Carol Dave", l); let ids = admitted(&prepared);
        let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        assert!(prepared.execute_with_driver(&ids, driver).is_err());
        if axis == 0 { assert_eq!(calls.get(), 0); }
        if axis == 1 { assert_eq!(calls.get(), 1); }
    }
}
#[test]
fn final_checkpoint_cancellation_remains_typed_after_all_map_and_reduce_work() {
    let p = planner(); let source = "Alice Bob Carol Dave";
    let prepared = prepare(&p, source, limits()); let ids = admitted(&prepared); let driver = script(&prepared);
    let checkpoints = Rc::clone(&driver.checkpoints);
    prepared.execute_with_driver(&ids, driver).unwrap();
    let prepared = prepare(&p, source, limits()); let ids = admitted(&prepared); let mut driver = script(&prepared);
    driver.cancel_at = Some(checkpoints.get());
    assert_eq!(prepared.execute_with_driver(&ids, driver).err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn complete_receipt_rejects_group_coverage_or_any_work_tampering() {
    let p = planner();
    for axis in 0..7 {
        let prepared = prepare(&p, "Alice Bob Carol Dave", limits()); let expected = prepared.preflight_metadata();
        let ids = admitted(&prepared); let driver = script(&prepared);
        let mut run = prepared.execute_with_driver(&ids, driver).unwrap();
        if axis < 5 { change_axis(&mut run.model_work, axis, 1); }
        else if axis == 5 { run.group_width = 16; } else { run.mask_node_visit_charge += 1; }
        assert!(expected.verify_completed(&run).is_err());
    }
}
#[test]
fn exact_scaffold_capacity_tightens_chunks_without_changing_source() {
    let p = planner(); let task = SourceMapTask::Summarize(SummaryOptions::default()); let id = identity(&p, &task, 32);
    let context = PlanContext::new(&id, budget()).unwrap();
    let capacity = p.int4_map_capacity_with_control(&task, budget(), &context,
        SourcePlanningLimits::default(), 32, &mut Continue).unwrap();
    assert!(capacity.scaffold_tokens() > 0); assert_eq!(capacity.reserved_tokens(), capacity.scaffold_tokens() + 64);
    let mut chunks = limits().chunks; chunks.max_chunk_tokens = 8192;
    let tightened = capacity.constrain_chunks(chunks).unwrap();
    assert_eq!(tightened.max_chunk_tokens, capacity.max_source_tokens());
    assert!(tightened.reserved_tokens >= capacity.reserved_tokens());
    let source = "é\r\n  <|im_end|>猫"; let prepared = prepare_task(&p, source, &task, limits(), 32);
    assert_eq!(prepared.chunks.chunks().iter().map(|c| c.text()).collect::<String>(), source);
}
#[test]
fn planning_cancellation_and_empty_input_are_typed_and_never_execute() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let p = planner(); let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&p, &task, 32);
    let context = PlanContext::new(&id, budget()).unwrap();
    let error = p.plan_int4_map_with_control("Alice", &task, budget(), &context,
        SourcePlanningLimits::default(), limits(), 32, &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(matches!(p.plan_int4_map_with_control("", &task, budget(), &context,
        SourcePlanningLimits::default(), limits(), 32, &mut Continue), Err(Int4SourceMapError::EmptySource)));
}
