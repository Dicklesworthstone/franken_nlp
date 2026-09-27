//! Actual pinned planning and keyed manifests only; not inference or quality evidence.
use super::*;
use crate::{
    batch::BatchDocument, canonjson,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    jobs::{FrozenManifest, JobContract, JobId, JobInput, JobLimits, JobSecret, JobWork, MismatchField, JobError},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl}, lmhead::scoring::ScoringMode,
        strict_int8::STRICT_INT8_EXECUTION},
    tasks::{classify::{ClassificationPlanner, ClassificationLabel, ClassificationMode, ClassificationPolicy},
        sentiment::{SentimentPlanner, SentimentOptions, SentimentPolicy, SentimentAxis}},
    tokenizer::pinned_controls,
};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn work() -> Int8Work { Int8Work::for_sequence(0, 8192, 100_000_000).unwrap() }
fn identity(task: &str, template: Sha256Digest, tokenizer: Sha256Digest) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"durable-score-planning-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture-int8".to_owned(), packing_set_digest: d,
        tokenizer_digest: tokenizer, template_digest: template, task_spec: task.to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn classifier() -> ClassificationPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let eos = registry.template_controls().entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    ClassificationPlanner::pinned(registry.template_controls(), eos).unwrap()
}
fn defaults() -> ClassificationBatchArgs {
    ClassificationBatchArgs { labels: vec![
        ClassificationLabel { id: "legal".to_owned(), description: "private legal category".to_owned() },
        ClassificationLabel { id: "other".to_owned(), description: "other subject".to_owned() }],
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: budget() }
}
fn classify_factory(p: &ClassificationPlanner, args: Option<ClassificationBatchArgs>) -> Int8ClassificationJobPlanner<'_> {
    Int8ClassificationJobPlanner::new(p, identity("classify-v1", *p.template_digest(), p.tokenizer_digest()),
        budget(), ClassificationLimits::default(), args, work()).unwrap()
}
fn sentiment(policy: SentimentPolicy, mode: ScoringMode) -> SentimentPlanner {
    let registry = pinned_controls::pinned().unwrap();
    let eos = registry.template_controls().entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    SentimentPlanner::pinned(registry.template_controls(), SentimentOptions { mode, eos_token_id: eos, policy }).unwrap()
}
fn policy() -> SentimentPolicy { SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 } }
fn sentiment_config(p: &SentimentPlanner) -> SentimentBatchConfig {
    SentimentBatchConfig { identity: identity("sentiment-v1", *p.template_digest(), p.tokenizer_digest()),
        task_ceiling: budget(), planning: SentimentLimits::default(),
        defaults: Some(SentimentBatchArgs { axes: SentimentAxis::ALL.to_vec(), budget: budget() }),
        max_item_work: work(), max_model_work: work() }
}
fn document<A>(args: Option<A>) -> BatchDocument<A> {
    BatchDocument { id: "item".to_owned(), text: "Alice <tool_call> é 上海".to_owned(), task_args: args }
}
fn freeze<R: Serialize>(identity: &ExecutionIdentity, recipe: &R, input: &[u8], secret: u8) -> FrozenManifest {
    let limits = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192, max_snapshot_bytes: 65536,
        max_result_bytes: 65536, max_spool_bytes: 1 << 20, max_materialized_bytes: 1 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 4, max_work: JobWork { model: work(), mask_node_visits: 0 } };
    FrozenManifest::freeze(&JobSecret::from_bytes([secret; 32]), JobContract { job_id: JobId([9; 16]),
        execution: identity, recipe, limits }, [JobInput { id: "item", original: input, normalized: input }], &mut Continue).unwrap()
}
const INPUT: &[u8] = br#"{"id":"item","text":"Alice"}"#;

#[test]
fn classification_recipe_changes_reject_resume_even_when_records_are_identical() {
    let p = classifier(); let a = classify_factory(&p, Some(defaults()));
    let frozen = freeze(a.execution_identity(), a.job_recipe(), INPUT, 7);
    frozen.binding.compare(&freeze(a.execution_identity(), a.job_recipe(), INPUT, 7).binding).unwrap();
    for axis in 0..4 {
        let mut args = defaults();
        match axis { 0 => args.labels[0].description.push('!'), 1 => args.labels.swap(0, 1),
            2 => args.mode = ClassificationMode::MultiLabel, _ => args.policy.minimum_margin_ppm += 1 }
        let changed = classify_factory(&p, Some(args));
        assert_eq!(frozen.binding.compare(&freeze(changed.execution_identity(), changed.job_recipe(), INPUT, 7).binding),
            Err(JobError::Mismatch(MismatchField::Recipe)));
    }
    assert_eq!(frozen.binding.compare(&freeze(a.execution_identity(), a.job_recipe(), br#"{"id":"item","text":"Bob"}"#, 7).binding),
        Err(JobError::Mismatch(MismatchField::Population)));
    assert!(frozen.binding.compare(&freeze(a.execution_identity(), a.job_recipe(), INPUT, 8).binding).is_err());
}
#[test]
fn every_classification_scorer_planning_and_native_axis_is_frozen() {
    let original = ClassificationJobRecipe::new(budget(), ClassificationLimits::default(), None, work()).unwrap();
    let before = canonjson::canonical_bytes(&original).unwrap();
    for axis in 0..20 {
        let mut l = ClassificationLimits::default(); let mut w = work();
        match axis { 0 => l.max_labels += 1, 1 => l.max_input_bytes += 1, 2 => l.max_label_id_bytes += 1,
            3 => l.max_label_description_bytes += 1, 4 => l.max_total_label_bytes += 1,
            5 => l.max_context_tokens += 1, 6 => l.max_total_prompt_tokens += 1,
            7 => l.max_work.forward_positions += 1, 8 => l.max_work.projected_logits += 1,
            9 => l.scoring.max_candidates += 1, 10 => l.scoring.max_total_tokens += 1,
            11 => l.scoring.max_nodes += 1, 12 => l.scoring.max_depth += 1,
            13 => l.scoring.max_candidate_id_bytes += 1, 14 => l.scoring.max_projected_logits += 1,
            15 => w.forward_positions += 1, 16 => w.projected_logits += 1, 17 => w.attention_pairs += 1,
            18 => w.projections.dot_products += 1, _ => w.projections.multiply_accumulates += 1 }
        assert_ne!(before, canonjson::canonical_bytes(&ClassificationJobRecipe::new(budget(), l, None, w).unwrap()).unwrap(), "{axis}");
    }
}
#[test]
fn actual_native_classification_plans_preserve_exclusive_and_independent_heads() {
    let p = classifier(); let factory = classify_factory(&p, Some(defaults()));
    let before = canonjson::canonical_bytes(factory.job_recipe()).unwrap();
    let exclusive = factory.prepare_with_control(document(None), &mut Continue).unwrap();
    assert_eq!(exclusive.head_count(), 1);
    let mut args = defaults(); args.mode = ClassificationMode::MultiLabel;
    let multi = factory.prepare_with_control(document(Some(args)), &mut Continue).unwrap();
    assert_eq!(multi.head_count(), 2);
    assert_ne!(exclusive.execution_identity().taskir_digest, multi.execution_identity().taskir_digest);
    assert_eq!(exclusive.execution_identity().logical_model_digest, factory.execution_identity().logical_model_digest);
    assert!(multi.model_work().projections.multiply_accumulates > 0);
    let again = factory.prepare_with_control(document(None), &mut Continue).unwrap();
    assert_eq!(exclusive.execution_identity(), again.execution_identity());
    assert_eq!(before, canonjson::canonical_bytes(factory.job_recipe()).unwrap());
}
#[test]
fn sentiment_policy_and_default_axes_are_part_of_the_authenticated_contract() {
    let p = sentiment(policy(), ScoringMode::FullVocabulary);
    let original = Int8SentimentJobPlanner::new(&p, sentiment_config(&p)).unwrap();
    let first = freeze(original.execution_identity(), original.job_recipe(), INPUT, 7);
    let mut changed = sentiment_config(&p); changed.defaults.as_mut().unwrap().axes.pop();
    let changed = Int8SentimentJobPlanner::new(&p, changed).unwrap();
    assert_eq!(first.binding.compare(&freeze(changed.execution_identity(), changed.job_recipe(), INPUT, 7).binding),
        Err(JobError::Mismatch(MismatchField::Recipe)));
    let mut new_policy = policy(); new_policy.minimum_peak_weight_ppm = 123;
    let other = sentiment(new_policy, ScoringMode::FullVocabulary);
    assert_ne!(p.template_digest(), other.template_digest());
    let changed = Int8SentimentJobPlanner::new(&other, sentiment_config(&other)).unwrap();
    assert!(first.binding.compare(&freeze(changed.execution_identity(), changed.job_recipe(), INPUT, 7).binding).is_err());
}
#[test]
fn every_sentiment_scorer_bundle_and_both_native_ceiling_sets_are_frozen() {
    let p = sentiment(policy(), ScoringMode::FullVocabulary);
    let before = canonjson::canonical_bytes(&SentimentJobRecipe::new(&sentiment_config(&p)).unwrap()).unwrap();
    for axis in 0..21 {
        let mut c = sentiment_config(&p);
        match axis { 0 => c.planning.per_axis.max_candidates += 1, 1 => c.planning.per_axis.max_total_tokens += 1,
            2 => c.planning.per_axis.max_nodes += 1, 3 => c.planning.per_axis.max_depth += 1,
            4 => c.planning.per_axis.max_candidate_id_bytes += 1, 5 => c.planning.per_axis.max_projected_logits += 1,
            6 => c.planning.max_total_prompt_tokens += 1, 7 => c.planning.max_total_candidate_tokens += 1,
            8 => c.planning.max_total_nodes += 1, 9 => c.planning.max_total_projected_logits += 1,
            10 => c.planning.max_output_bytes += 1, 11 => c.max_item_work.forward_positions += 1,
            12 => c.max_item_work.projected_logits += 1, 13 => c.max_item_work.attention_pairs += 1,
            14 => c.max_item_work.projections.dot_products += 1, 15 => c.max_item_work.projections.multiply_accumulates += 1,
            16 => c.max_model_work.forward_positions += 1, 17 => c.max_model_work.projected_logits += 1,
            18 => c.max_model_work.attention_pairs += 1, 19 => c.max_model_work.projections.dot_products += 1,
            _ => c.max_model_work.projections.multiply_accumulates += 1 }
        assert_ne!(before, canonjson::canonical_bytes(&SentimentJobRecipe::new(&c).unwrap()).unwrap(), "{axis}");
    }
}
#[test]
fn sentiment_override_is_one_complete_bundle_and_does_not_mutate_defaults() {
    let p = sentiment(policy(), ScoringMode::FullVocabulary);
    let factory = Int8SentimentJobPlanner::new(&p, sentiment_config(&p)).unwrap();
    let before = canonjson::canonical_bytes(factory.job_recipe()).unwrap();
    let all = factory.prepare_with_control(document(None), &mut Continue).unwrap();
    assert_eq!(all.head_count(), SentimentAxis::ALL.len());
    let single = factory.prepare_with_control(document(Some(SentimentBatchArgs {
        axes: vec![SentimentAxis::ALL[0]], budget: budget() })), &mut Continue).unwrap();
    assert_eq!(single.head_count(), 1);
    assert!(all.model_work().forward_positions > single.model_work().forward_positions);
    let again = factory.prepare_with_control(document(None), &mut Continue).unwrap();
    assert_eq!(again.execution_identity(), all.execution_identity());
    assert_eq!(before, canonjson::canonical_bytes(factory.job_recipe()).unwrap());
}
#[test]
fn shared_deadline_is_preserved_by_both_model_free_factories() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        Some(DecodeCancellationKind::Deadline)
    } }
    let c = classifier(); let cp = classify_factory(&c, Some(defaults()));
    let s = sentiment(policy(), ScoringMode::FullVocabulary);
    let sp = Int8SentimentJobPlanner::new(&s, sentiment_config(&s)).unwrap();
    for failure in [cp.prepare_with_control(document(None), &mut Stop).err().unwrap(),
        sp.prepare_with_control(document(None), &mut Stop).err().unwrap()] {
        assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    }
}
#[test]
fn malformed_defaults_and_each_unfunded_native_axis_are_rejected_without_an_engine() {
    for axis in 0..5 {
        let mut w = work();
        match axis { 0 => w.forward_positions = 0, 1 => w.projected_logits = 0, 2 => w.attention_pairs = 0,
            3 => w.projections.dot_products = 0, _ => w.projections.multiply_accumulates = 0 }
        assert!(ClassificationJobRecipe::new(budget(), ClassificationLimits::default(), None, w).is_err());
    }
    for axis in 0..5 {
        let mut args = defaults();
        match axis { 0 => args.labels.clear(), 1 => args.labels[1].id = args.labels[0].id.clone(),
            2 => args.labels[0].id = "\n".to_owned(), 3 => args.policy.minimum_margin_ppm = 1_000_001,
            _ => args.budget.max_output_bytes += 1 }
        assert!(ClassificationJobRecipe::new(budget(), ClassificationLimits::default(), Some(args), work()).is_err());
    }
    let p = classifier(); let missing = classify_factory(&p, None);
    assert!(missing.prepare_with_control(document(None), &mut Continue).is_err());
}
