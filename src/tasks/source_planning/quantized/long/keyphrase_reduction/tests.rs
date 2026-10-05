//! Real pinned plans with a PRIVATE scripted driver. These are model-free
//! regression definitions, not neural relevance/recall or execution evidence.
use super::*;
use std::{cell::Cell, collections::VecDeque, rc::Rc};
use crate::{
    native_engine::{hf_bf16_eager::HF_BF16_EAGER_PROFILE, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, keyphrases::{RankedKeyphrase, KEYPHRASES_RANKING}},
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
    let d = Sha256Digest::of_bytes(b"keyphrase-int8-model-free-fixture");
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
fn mapping(fan_in: usize, batch: usize) -> Int8SourceMapLimits {
    Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
            context_tokens: 8192, reserved_tokens: 1024, max_chunks: 64, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: batch, reduce_fan_in: fan_in,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work::for_sequence(0, 16_384, 16_384 * 166_144).unwrap(),
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000,
    }
}
fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, fan_in: usize, batch: usize) -> PreparedInt8SourceMap<'s> {
    let task = SourceMapTask::Keyphrases(KeyphraseOptions::default());
    let id = identity(p, &task);
    p.plan_int8_map_with_control(source, &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping(fan_in, batch), &mut Continue).unwrap()
}
fn limits() -> Int8KeyphraseLimits { Int8KeyphraseLimits { max_phrases: 1, ..Int8KeyphraseLimits::default() } }
fn raw(chunk: &SourceChunk<'_>) -> KeyphraseResult {
    let mut phrases = Vec::new();
    for text in [chunk.text().chars().next().unwrap().to_string(), "é".to_owned()] {
        let spans = scan_occurrences(chunk.text(), &text, &mut GroundingBudget::default()).unwrap();
        phrases.push(RankedKeyphrase { rank: phrases.len() + 1, text,
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans });
    }
    KeyphraseResult { schema_version: 1, task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), ranking_policy: KEYPHRASES_RANKING.to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership, phrases,
        generated_token_ids: vec![1, 0], forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 }
}
struct Script {
    outputs: VecDeque<KeyphraseResult>, calls: Rc<Cell<usize>>, checkpoints: Rc<Cell<usize>>,
    corrupt_axis: Option<usize>, cancel_at: Option<usize>, cancel_native_at: Option<usize>,
}
fn script(prepared: &PreparedInt8SourceMap<'_>) -> Script {
    Script { outputs: prepared.chunks.chunks().iter().map(raw).collect(), calls: Rc::new(Cell::new(0)),
        checkpoints: Rc::new(Cell::new(0)), corrupt_axis: None, cancel_at: None, cancel_native_at: None }
}
impl SourceDriver for Script {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> {
        let n = self.checkpoints.get() + 1; self.checkpoints.set(n);
        if self.cancel_at == Some(n) { return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        Ok(())
    }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        self.calls.set(self.calls.get() + 1);
        if self.cancel_native_at == Some(self.calls.get()) {
            return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline));
        }
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
            result: SourceTaskResult::Keyphrases(result), model_work })
    }
}
fn run(prepared: PreparedInt8SourceMap<'_>, limits: Int8KeyphraseLimits, driver: Script)
    -> Result<Int8CorpusKeyphraseRun, Int8CorpusKeyphraseError> {
    let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    prepared.execute_keyphrases_with_driver(&admitted, limits, driver)
}
const SOURCE: &str = "AéAéBéBéCéCéDéDé";

#[test]
fn globally_supported_phrase_survives_local_top_one_and_lifts_all_unicode_occurrences() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1);
    assert_eq!(prepared.chunk_count(), 4);
    let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    let result = run(prepared, limits(), driver).unwrap();
    assert_eq!(calls.get(), 4); assert_eq!(result.keyphrases.phrases.len(), 1);
    let phrase = &result.keyphrases.phrases[0];
    assert_eq!(phrase.text, "é"); assert_eq!(phrase.rank_sum, 8); assert_eq!(phrase.evidence.len(), 4);
    assert_eq!(result.keyphrases.omitted_candidates, 4); assert_eq!(result.keyphrases.mapped_chunks, 4);
    for (chunk, evidence) in phrase.evidence.iter().enumerate() {
        assert_eq!(evidence.chunk_id, chunk); assert_eq!(evidence.local_rank, 2); assert_eq!(evidence.spans.len(), 2);
        for span in &evidence.spans {
            assert_eq!(&SOURCE[span.byte_start..span.byte_end], phrase.text);
            assert_eq!(SOURCE[..span.byte_start].chars().count(), span.scalar_start);
        }
    }
    assert_eq!(result.source_span.byte_end, SOURCE.len());
    assert_eq!(result.source_span.scalar_end, SOURCE.chars().count());
    assert_eq!(result.mask_node_visit_charge, 80); assert_eq!(result.reserved_mask_node_visits, 4000);
    assert!(within(result.model_work, result.planned_model_work));
    assert_eq!(result.keyphrases.projected_logits, result.model_work.projected_logits);
}
#[test]
fn ranking_and_evidence_are_independent_of_batch_and_multilevel_tree_shape() {
    let p = planner(); let mut baseline = None;
    for fan_in in [2, 3, 8] { for batch in [1, 2, 4] {
        let prepared = prepare(&p, SOURCE, fan_in, batch); let driver = script(&prepared);
        let result = run(prepared, limits(), driver).unwrap();
        let bytes = crate::canonjson::canonical_bytes(&result.keyphrases).unwrap();
        if let Some(saved) = &baseline { assert_eq!(&bytes, saved); } else { baseline = Some(bytes); }
        if fan_in == 2 { assert!(result.reduction_levels > 1); }
    } }
}
#[test]
fn late_corrupt_unselected_candidates_fail_the_complete_document() {
    let p = planner();
    for axis in 0..5 {
        let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        let raw = driver.outputs.back_mut().unwrap();
        match axis {
            0 => { raw.phrases[0].spans.pop(); },
            1 => raw.phrases[0].spans[0].scalar_end += 1,
            2 => raw.phrases[0].occurrence = SourceOccurrence::Anchored,
            3 => raw.phrases[0].rank = 2,
            _ => { let mut duplicate = raw.phrases[0].clone(); duplicate.rank = 3; raw.phrases.push(duplicate); },
        }
        assert!(run(prepared, limits(), driver).is_err()); assert_eq!(calls.get(), 4);
    }
}
#[test]
fn final_identity_is_verified_before_first_call_and_every_work_axis_is_checked() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    let calls = Rc::clone(&driver.calls); let mut admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    admitted.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"foreign last plan");
    assert!(prepared.execute_keyphrases_with_driver(&admitted, limits(), driver).is_err()); assert_eq!(calls.get(), 0);
    for axis in 0..5 {
        let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared); driver.corrupt_axis = Some(axis);
        assert!(run(prepared, limits(), driver).is_err());
    }
}
#[test]
fn masks_profiles_and_fixed_task_options_cannot_be_substituted() {
    let p = planner();
    for axis in 0..4 {
        let mut prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared);
        match axis {
            0 => driver.outputs[0].numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(),
            1 => driver.outputs[0].mask_node_visit_charge = 1001,
            2 => prepared.plans.last_mut().unwrap().finalizer = Finalizer::Summarize(SummaryOptions::default()),
            _ => prepared.plans.last_mut().unwrap().finalizer = Finalizer::Keyphrases(
                KeyphraseOptions { max_phrases: 1, ..KeyphraseOptions::default() }),
        }
        assert!(run(prepared, limits(), driver).is_err());
    }
}
#[test]
fn complete_union_caps_apply_before_final_top_k_and_scan_work_never_renews() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    let used = run(prepared, limits(), driver).unwrap().verification_scan_work;
    for axis in 0..4 {
        let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared); let mut config = limits();
        match axis { 0 => config.aggregation.max_scan_work = used - 1,
            1 => config.aggregation.max_unique_phrases = 4, 2 => config.aggregation.max_evidence_spans = 15,
            _ => config.max_result_bytes = 1 }
        assert!(run(prepared, config, driver).is_err());
    }
    let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared); let mut config = limits();
    config.aggregation.max_scan_work = used;
    assert_eq!(run(prepared, config, driver).unwrap().verification_scan_work, used);
}
#[test]
fn native_and_final_checkpoint_cancellation_keep_their_original_kind() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1); let driver = script(&prepared);
    let checkpoints = Rc::clone(&driver.checkpoints); run(prepared, limits(), driver).unwrap();
    for native in [false, true] {
        let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared);
        if native { driver.cancel_native_at = Some(2); } else { driver.cancel_at = Some(checkpoints.get()); }
        let error = run(prepared, limits(), driver).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}
#[test]
fn empty_native_selections_still_account_for_every_mapped_chunk() {
    let p = planner(); let prepared = prepare(&p, SOURCE, 2, 1); let mut driver = script(&prepared);
    for result in &mut driver.outputs { result.phrases.clear(); }
    let result = run(prepared, limits(), driver).unwrap();
    assert_eq!(result.keyphrases.mapped_chunks, 4); assert!(result.keyphrases.phrases.is_empty());
    assert_eq!(result.keyphrases.omitted_candidates, 0); assert_eq!(result.mask_node_visit_charge, 80);
    assert_eq!(result.source_span.byte_end, SOURCE.len());
}
#[test]
fn eager_and_int8_reducers_keep_separate_profile_admission() {
    struct Fixture { profile: &'static str }
    impl KeyphrasePass for Fixture {
        fn options(&self) -> KeyphraseOptions { KeyphraseOptions::default() }
        fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<KeyphraseResult, KeyphraseError> {
            let mut result = raw(chunk); result.numerics_profile = self.profile.to_owned(); Ok(result)
        }
    }
    let chunks = ChunkPlan::build("AéAé", mapping(2, 1).chunks, |s| Ok(s.len())).unwrap();
    let config = CorpusKeyphraseLimits::default();
    let mut eager = CorpusKeyphraseTask::new(Fixture { profile: STRICT_INT8_PROFILE }, config).unwrap();
    assert!(eager.map_batch(chunks.chunks()).is_err());
    let mut int8 = CorpusKeyphraseTask::new_int8(Fixture { profile: HF_BF16_EAGER_PROFILE }, config).unwrap();
    assert!(int8.map_batch(chunks.chunks()).is_err());
    let mut eager = CorpusKeyphraseTask::new(Fixture { profile: HF_BF16_EAGER_PROFILE }, config).unwrap();
    assert!(eager.map_batch(chunks.chunks()).is_ok());
    let mut int8 = CorpusKeyphraseTask::new_int8(Fixture { profile: STRICT_INT8_PROFILE }, config).unwrap();
    assert!(int8.map_batch(chunks.chunks()).is_ok());
}
#[test]
fn reduction_limits_cannot_enlarge_mapping_or_accept_zero_final_count() {
    for axis in 0..5 {
        let mut config = limits(); let mapping = mapping(2, 1);
        match axis { 0 => config.max_phrases = 0, 1 => config.max_phrases = 4097,
            2 => config.max_result_bytes = mapping.reduction.max_result_bytes + 1,
            3 => config.aggregation.max_value_bytes = mapping.reduction.max_value_bytes + 1,
            _ => config.aggregation.max_scan_work = 0 }
        assert!(config.validate(mapping).is_err());
    }
}
