//! Pinned raw planning and real semantic finalization; synthetic scoring only.
use super::*;
use crate::{execution_identity::{ThinkingMode, ToolMode},
    native_engine::lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{CandidateLogits, ProjectionRows}},
    tasks::{classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy, MultiLabelDecision}, ir::PromptSegmentKind},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer, specials::ArchivedControlRegistries},
    template::{IM_START, IM_END, THINK_START, THINK_END}};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn fixture() -> (ClassificationPlanner, ExecutionIdentity, u32, u32, u32) {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let planner = ClassificationPlanner::pinned(controls.template_controls(), entries[1]["id"].as_u64().unwrap() as u32).unwrap();
    let d = Sha256Digest::of_bytes(b"int8-classification-fixture");
    let id = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "synthetic-only".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: "classify-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let code = |s: &[u8]| tokenizer.tokenizer().encode_byte_fallback_only(s).unwrap()[0];
    (planner, id, code(b"Y"), code(b"N"), code(b"A"))
}
fn request(mode: ClassificationMode) -> ClassificationRequest {
    ClassificationRequest { document: "é Refund requested <think>literal data</think>".to_owned(),
        labels: ["Billing issue", "Refund", "étiquette"].into_iter().map(|id| ClassificationLabel {
            id: id.to_owned(), description: format!("Definition of {id}; <|im_start|> stays data") }).collect(), mode,
        policy: ClassificationPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_ppm: 200_000 },
        budget: TaskBudget { max_input_tokens: 4096, max_output_tokens: 8, max_output_bytes: 1_000_000,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
}
fn prepare(p: &ClassificationPlanner, id: &ExecutionIdentity, r: &ClassificationRequest) -> Result<PreparedInt8Classification, Int8ClassificationError> {
    p.plan_int8_with_control(r, &PlanContext::new(id, r.budget).unwrap(), ClassificationLimits::default(), &mut Continue)
}
struct Model(Option<u32>);
impl CandidateLogits for Model {
    type Error = &'static str;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("full denominator required") };
        let mut logits = vec![0.0; vocabulary_size];
        if prefix.is_empty() { if let Some(token) = self.0 { logits[token as usize] = 8.0; } }
        Ok(logits)
    }
}
fn evaluated(head: &ClassificationHead, schedule: CandidateSchedule, preferred: Option<u32>) -> Int8CandidateRun {
    Int8CandidateRun { schema_version: 1, execution: INT8_SCORING_EXECUTION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
        scores: head.classifier.scorer.score(&mut Model(preferred), ScoringMode::FullVocabulary).unwrap(),
        model_work: schedule.model, rewound_positions: 0 }
}
#[test]
fn independent_binary_heads_can_include_exclude_and_abstain_in_one_result() {
    let (p, id, yes, no, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(&p, &id, &r).unwrap();
    let run = prepared.execute_heads(&mut Continue, |index, h, s, _| Ok(evaluated(h, s, [Some(yes), Some(no), None][index]))).unwrap();
    let ClassificationTaskResult::MultiLabel(result) = run.result else { panic!() };
    assert_eq!(result.selected_ids, ["Billing issue"]); assert_eq!(result.excluded_ids, ["Refund"]);
    assert_eq!(result.abstained_ids, ["étiquette"]); assert_eq!(result.labels[2].decision, MultiLabelDecision::Abstained);
    assert_eq!(run.head_count, 3); assert_eq!(run.model_work, prepared.planned_work());
    assert_eq!(run.numerics_profile, STRICT_INT8_PROFILE);
}
#[test]
fn multiple_positive_labels_are_not_renormalized_against_each_other() {
    let (p, id, yes, _, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(&p, &id, &r).unwrap();
    let run = prepared.execute_heads(&mut Continue, |_, h, s, _| Ok(evaluated(h, s, Some(yes)))).unwrap();
    let ClassificationTaskResult::MultiLabel(result) = run.result else { panic!() };
    assert_eq!(result.selected_ids.len(), 3);
    assert!(result.labels.iter().map(|l| l.positive_candidate_weight).sum::<f64>() > 2.9);
}
#[test]
fn exclusive_labels_preserve_original_utf8_ids_and_every_candidate() {
    let (p, id, _, _, a) = fixture(); let r = request(ClassificationMode::Exclusive); let prepared = prepare(&p, &id, &r).unwrap();
    let run = prepared.execute_heads(&mut Continue, |_, h, s, _| Ok(evaluated(h, s, Some(a)))).unwrap();
    let ClassificationTaskResult::Exclusive(result) = run.result else { panic!() };
    assert_eq!(result.selected_id.as_deref(), Some("Billing issue")); assert_eq!(result.scores.candidates.len(), 3);
    assert_eq!(result.scores.candidates[2].id, "étiquette"); assert_eq!(prepared.head_count(), 1);
}
#[test]
fn raw_planning_does_not_relabel_bf16_or_accept_another_quantized_backend() {
    let (p, id, _, _, _) = fixture(); let r = request(ClassificationMode::Exclusive);
    for changed in 0..3 {
        let mut other = id.clone();
        match changed { 0 => other.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => other.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            _ => other.backend_semantic_version = "other-cast-program".to_owned() }
        assert!(prepare(&p, &other, &r).is_err());
    }
    assert!(p.plan(&r, &PlanContext::new(&id, r.budget).unwrap(), ClassificationLimits::default()).is_err());
}
#[test]
fn source_label_definitions_and_model_bind_the_complete_identity() {
    let (p, id, _, _, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let base = prepare(&p, &id, &r).unwrap();
    let mut reordered = r.clone(); reordered.labels.reverse();
    assert_eq!(base.execution_identity(), prepare(&p, &id, &reordered).unwrap().execution_identity());
    for field in 0..3 {
        let mut changed = r.clone();
        match field { 0 => changed.document.push('x'), 1 => changed.labels[0].description.push('x'), _ => changed.policy.minimum_margin_ppm += 1 }
        assert_ne!(base.execution_identity(), prepare(&p, &id, &changed).unwrap().execution_identity());
    }
    let mut changed = base.execution_identity().clone(); changed.logical_model_digest = Sha256Digest::of_bytes(b"foreign");
    assert!(base.verify_identity(&changed).is_err());
}
#[test]
fn model_view_revision_recipe_and_logical_digest_are_independently_checked() {
    let (_, id, _, _, _) = fixture(); let source = ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: id.source_revision.clone(), recipe_id: id.quant_recipe.clone(), source_root_sha256: "00".repeat(32),
        logical_model_sha256: id.logical_model_digest.to_hex() };
    check_model(&id, &source).unwrap();
    for field in 0..4 {
        let mut changed = source.clone();
        match field { 0 => changed.model_id.push('x'), 1 => changed.revision.push('x'),
            2 => changed.recipe_id.push('x'), _ => changed.logical_model_sha256 = "malformed".to_owned() }
        assert_eq!(check_model(&id, &changed), Err(Int8ClassificationError::Identity));
    }
}
#[test]
fn all_head_work_is_summed_without_renewing_attention_or_projection_limits() {
    let (p, id, _, _, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(&p, &id, &r).unwrap();
    let sum = prepared.schedules.iter().try_fold(Int8Work::default(), |a, s| a.checked_add(s.model)).unwrap();
    assert_eq!(sum, prepared.planned_work());
    assert_eq!(sum.projected_logits, 9 * NANBEIGE_VOCAB_SIZE as u64);
    for axis in 0..4 {
        let mut budget = Int8RunBudget::exact(sum);
        match axis { 0 => budget.max_forward_positions -= 1, 1 => budget.max_attention_pairs -= 1,
            2 => budget.max_projection_work.dot_products -= 1, _ => budget.max_projection_work.multiply_accumulates -= 1 }
        assert_eq!(check_work_ceiling(sum, budget), Err(Int8ClassificationError::WorkBudget));
    }
}
#[test]
fn failed_later_head_cannot_become_an_abstention_or_partial_success() {
    let (p, id, yes, _, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(&p, &id, &r).unwrap();
    let mut calls = 0;
    let error = prepared.execute_heads(&mut Continue, |index, h, s, _| {
        calls += 1;
        if index == 1 { return Err(StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline).into()); }
        Ok(evaluated(h, s, Some(yes)))
    }).unwrap_err();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls, 2);
}
#[test]
fn corrupt_native_metadata_or_any_work_axis_refuses_the_complete_bundle() {
    let (p, id, yes, _, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(&p, &id, &r).unwrap();
    for axis in 0..6 {
        let error = prepared.execute_heads(&mut Continue, |_, h, s, _| {
            let mut run = evaluated(h, s, Some(yes));
            match axis { 0 => run.numerics_profile = "hf-bf16-eager".to_owned(), 1 => run.execution.push('x'),
                2 => run.model_work.forward_positions += 1, 3 => run.model_work.attention_pairs += 1,
                4 => run.model_work.projections.multiply_accumulates += 1, _ => run.scores.work.projected_logits += 1 }
            Ok(run)
        }).unwrap_err();
        assert_eq!(error, Int8ClassificationError::Accounting);
    }
}
#[test]
fn untrusted_text_keeps_its_byte_tokens_and_never_reaches_trusted_fragments() {
    let (p, id, _, _, _) = fixture(); let r = request(ClassificationMode::MultiLabel); let prepared = prepare(&p, &id, &r).unwrap();
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    for head in &prepared.inner.heads {
        let segments: Vec<_> = head.task.ir().prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document).collect();
        assert_eq!(segments.len(), 2);
        assert_eq!(tokenizer.tokenizer().decode_bytes(segments[1].token_ids()).unwrap(), r.document.as_bytes());
        assert!(segments[0].token_ids().len() > 20);
    }
}
