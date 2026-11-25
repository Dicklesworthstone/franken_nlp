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


mod encoding {
    use super::*;
    use crate::tokenizer::{embedded::EmbeddedTokenizer,
        untrusted::{SafeBpeLimits, UntrustedDocumentEncoder}};

    fn call() -> SafeBpeLimits {
        SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 2048,
            max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 }
    }
    fn total() -> SafeBpeWorkLimits {
        SafeBpeWorkLimits { max_input_bytes: 65536, max_output_tokens: 32768,
            max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }
    }
    fn classifier(call: SafeBpeLimits) -> ClassificationPlanner {
        super::classifier().with_document_encoding(UntrustedEncoding::SafeBpeV1(call)).unwrap()
    }
    fn sentiment(call: SafeBpeLimits) -> SentimentPlanner {
        super::sentiment(policy(), ScoringMode::FullVocabulary)
            .with_document_encoding(UntrustedEncoding::SafeBpeV1(call)).unwrap()
    }
    fn classify_factory(p: &ClassificationPlanner, args: ClassificationBatchArgs,
        total: SafeBpeWorkLimits) -> Int8ClassificationJobPlanner<'_> {
        Int8ClassificationJobPlanner::new_with_document_encoding(p,
            identity("classify-v1", *p.template_digest(), p.tokenizer_digest()),
            budget(), ClassificationLimits::default(), Some(args), work(), Some(total)).unwrap()
    }
    fn sentiment_factory(p: &SentimentPlanner, total: SafeBpeWorkLimits) -> Int8SentimentJobPlanner<'_> {
        Int8SentimentJobPlanner::new_with_document_encoding(p, sentiment_config(p), Some(total)).unwrap()
    }

    #[test]
    fn byte_recipes_remain_exact_and_compact_requires_an_explicit_invocation_allowance() {
        let c = super::classifier();
        let original = super::classify_factory(&c, Some(defaults()));
        let explicit = Int8ClassificationJobPlanner::new_with_document_encoding(&c,
            identity("classify-v1", *c.template_digest(), c.tokenizer_digest()),
            budget(), ClassificationLimits::default(), Some(defaults()), work(), None).unwrap();
        assert_eq!(canonjson::canonical_bytes(original.job_recipe()).unwrap(),
            canonjson::canonical_bytes(explicit.job_recipe()).unwrap());
        let s = super::sentiment(policy(), ScoringMode::FullVocabulary);
        let original_s = Int8SentimentJobPlanner::new(&s, sentiment_config(&s)).unwrap();
        let explicit_s = Int8SentimentJobPlanner::new_with_document_encoding(&s, sentiment_config(&s), None).unwrap();
        assert_eq!(canonjson::canonical_bytes(original_s.job_recipe()).unwrap(),
            canonjson::canonical_bytes(explicit_s.job_recipe()).unwrap());
        for value in [serde_json::to_value(explicit.job_recipe()).unwrap(),
            serde_json::to_value(explicit_s.job_recipe()).unwrap()] {
            assert!(value.get("input_encoding").is_none());
            assert!(value.get("encoding_work_limits").is_none());
        }
        assert_eq!(explicit.encoding_workspace_bytes().unwrap(), 0);
        assert_eq!(explicit_s.encoding_workspace_bytes().unwrap(), 0);
        assert!(Int8ClassificationJobPlanner::new_with_document_encoding(&c,
            identity("classify-v1", *c.template_digest(), c.tokenizer_digest()),
            budget(), ClassificationLimits::default(), Some(defaults()), work(), Some(total())).is_err());
        assert!(Int8SentimentJobPlanner::new_with_document_encoding(&s, sentiment_config(&s), Some(total())).is_err());
        let compact_c = classifier(call()); let compact_s = sentiment(call());
        assert!(Int8ClassificationJobPlanner::new(&compact_c,
            identity("classify-v1", *compact_c.template_digest(), compact_c.tokenizer_digest()),
            budget(), ClassificationLimits::default(), Some(defaults()), work()).is_err());
        assert!(Int8SentimentJobPlanner::new(&compact_s, sentiment_config(&compact_s)).is_err());
        let c = classify_factory(&compact_c, defaults(), total());
        let s = sentiment_factory(&compact_s, total());
        for actual in [c.encoding_workspace_bytes().unwrap(), s.encoding_workspace_bytes().unwrap()] {
            assert_eq!(actual, call().requirements().unwrap().workspace_bytes + 1024);
        }
        for axis in 0..4 {
            let mut invalid = total();
            match axis { 0 => invalid.max_input_bytes = 0, 1 => invalid.max_output_tokens = 0,
                2 => invalid.max_pair_checks = 0, _ => invalid.max_pair_bytes = 0 }
            assert!(EncodingWorkScope::new(&UntrustedEncoding::SafeBpeV1(call()), Some(invalid)).is_err());
        }
    }

    #[test]
    fn each_original_call_and_invocation_cap_is_authenticated_by_both_job_recipes() {
        let c = classifier(call()); let s = sentiment(call());
        let cp = classify_factory(&c, defaults(), total()); let sp = sentiment_factory(&s, total());
        let cf = freeze(cp.execution_identity(), cp.job_recipe(), INPUT, 7);
        let sf = freeze(sp.execution_identity(), sp.job_recipe(), INPUT, 7);
        for value in [serde_json::to_value(cp.job_recipe()).unwrap(), serde_json::to_value(sp.job_recipe()).unwrap()] {
            assert_eq!(value["input_encoding"], serde_json::to_value(UntrustedEncoding::SafeBpeV1(call())).unwrap());
            assert_eq!(value["encoding_work_limits"], serde_json::to_value(total()).unwrap());
        }
        for axis in 0..8 {
            let mut call = call(); let mut total = total();
            match axis { 0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
                2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
                4 => total.max_input_bytes -= 1, 5 => total.max_output_tokens -= 1,
                6 => total.max_pair_checks -= 1, _ => total.max_pair_bytes -= 1 }
            let changed_c = classifier(call); let changed_s = sentiment(call);
            let cc = classify_factory(&changed_c, defaults(), total);
            let ss = sentiment_factory(&changed_s, total);
            assert_ne!(canonjson::canonical_bytes(cp.job_recipe()).unwrap(),
                canonjson::canonical_bytes(cc.job_recipe()).unwrap(), "classification axis {axis}");
            assert_ne!(canonjson::canonical_bytes(sp.job_recipe()).unwrap(),
                canonjson::canonical_bytes(ss.job_recipe()).unwrap(), "sentiment axis {axis}");
            for mismatch in [
                cf.binding.compare(&freeze(cc.execution_identity(), cc.job_recipe(), INPUT, 7).binding),
                sf.binding.compare(&freeze(ss.execution_identity(), ss.job_recipe(), INPUT, 7).binding),
            ] {
                assert!(mismatch.is_err(), "axis {axis}");
                if axis >= 4 { assert_eq!(mismatch, Err(JobError::Mismatch(MismatchField::Recipe))); }
            }
        }
    }

    #[test]
    fn classification_source_and_all_label_metadata_share_one_nonrenewable_allowance() {
        let p = classifier(call());
        for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
            let mut args = defaults(); args.mode = mode;
            let baseline = classify_factory(&p, args.clone(), total());
            baseline.prepare_with_control(document(None), &mut Continue).unwrap();
            let one = baseline.encoding_work().unwrap();
            assert!(one.input_bytes() > document::<ClassificationBatchArgs>(None).text.len(),
                "canonical private label records must also pay tokenizer work");
            let mut cap = total(); cap.max_input_bytes = 2 * one.input_bytes() as u64;
            let factory = classify_factory(&p, args, cap);
            let recipe = canonjson::canonical_bytes(factory.job_recipe()).unwrap();
            let first = factory.prepare_with_control(document(None), &mut Continue).unwrap();
            assert_eq!(factory.encoding_work().unwrap(), one);
            assert_eq!(first.head_count(), if mode == ClassificationMode::Exclusive { 1 } else { 2 });
            let moved = factory;
            let second = moved.prepare_with_control(document(None), &mut Continue).unwrap();
            assert_eq!(first.execution_identity(), second.execution_identity(),
                "a smaller remaining execution ceiling cannot change the original prompt identity");
            assert_eq!(first.model_work(), second.model_work());
            assert_eq!(moved.document_encoding(), &UntrustedEncoding::SafeBpeV1(call()));
            assert_eq!(moved.encoding_work_limits(), Some(cap));
            assert_eq!(moved.encoding_work().unwrap(), one.checked_add(one).unwrap());
            let before = moved.encoding_work().unwrap();
            let error = moved.prepare_with_control(document(None), &mut Continue).err().unwrap();
            assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
            assert_eq!(moved.encoding_work().unwrap(), before);
            assert_eq!(canonjson::canonical_bytes(moved.job_recipe()).unwrap(), recipe);
        }
    }

    #[test]
    fn all_sentiment_axes_encode_the_source_once_and_reprepare_cannot_renew_work() {
        let p = sentiment(call());
        let baseline = sentiment_factory(&p, total());
        baseline.prepare_with_control(document(None), &mut Continue).unwrap();
        let one = baseline.encoding_work().unwrap();
        assert_eq!(one.input_bytes(), document::<SentimentBatchArgs>(None).text.len());
        let mut cap = total(); cap.max_input_bytes = 2 * one.input_bytes() as u64;
        let factory = sentiment_factory(&p, cap);
        let recipe = canonjson::canonical_bytes(factory.job_recipe()).unwrap();
        let first = factory.prepare_with_control(document(None), &mut Continue).unwrap();
        assert_eq!(first.head_count(), SentimentAxis::ALL.len());
        assert_eq!(factory.encoding_work().unwrap(), one);
        let moved = factory;
        let second = moved.prepare_with_control(document(None), &mut Continue).unwrap();
        assert_eq!(first.execution_identity(), second.execution_identity());
        assert_eq!(first.model_work(), second.model_work());
        assert_eq!(moved.document_encoding(), &UntrustedEncoding::SafeBpeV1(call()));
        assert_eq!(moved.encoding_work_limits(), Some(cap));
        let spent = one.checked_add(one).unwrap();
        assert_eq!(moved.encoding_work().unwrap(), spent);
        let error = moved.prepare_with_control(document(None), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(moved.encoding_work().unwrap(), spent);
        assert_eq!(canonjson::canonical_bytes(moved.job_recipe()).unwrap(), recipe);
    }

    #[test]
    fn tokenizer_failure_and_later_native_work_rejection_both_spend_completed_encoding() {
        let mut tight = call(); tight.max_output_tokens = 1;
        let c = classifier(tight); let s = sentiment(tight);
        let cp = classify_factory(&c, defaults(), total()); let sp = sentiment_factory(&s, total());
        assert!(cp.prepare_with_control(document(None), &mut Continue).is_err());
        assert!(sp.prepare_with_control(document(None), &mut Continue).is_err());
        let cfirst = cp.encoding_work().unwrap(); let sfirst = sp.encoding_work().unwrap();
        for first in [cfirst, sfirst] { assert!(first.input_bytes() > 0 && first.output_tokens() > 0); }
        assert!(cp.prepare_with_control(document(None), &mut Continue).is_err());
        assert!(sp.prepare_with_control(document(None), &mut Continue).is_err());
        assert_eq!(cp.encoding_work().unwrap(), cfirst.checked_add(cfirst).unwrap());
        assert_eq!(sp.encoding_work().unwrap(), sfirst.checked_add(sfirst).unwrap());

        let c = classifier(call()); let s = sentiment(call());
        let mut planning = ClassificationLimits::default(); planning.max_work.forward_positions = 1;
        let cp = Int8ClassificationJobPlanner::new_with_document_encoding(&c,
            identity("classify-v1", *c.template_digest(), c.tokenizer_digest()),
            budget(), planning, Some(defaults()), work(), Some(total())).unwrap();
        let mut config = sentiment_config(&s); config.max_item_work.forward_positions = 1;
        let sp = Int8SentimentJobPlanner::new_with_document_encoding(&s, config, Some(total())).unwrap();
        assert_eq!(cp.prepare_with_control(document(None), &mut Continue).err().unwrap().fault.code, BatchCode::WorkLimit);
        assert_eq!(sp.prepare_with_control(document(None), &mut Continue).err().unwrap().fault.code, BatchCode::WorkLimit);
        let cfirst = cp.encoding_work().unwrap(); let sfirst = sp.encoding_work().unwrap();
        for first in [cfirst, sfirst] { assert!(first.input_bytes() > 0 && first.output_tokens() > 0); }
        assert!(cp.prepare_with_control(document(None), &mut Continue).is_err());
        assert!(sp.prepare_with_control(document(None), &mut Continue).is_err());
        assert_eq!(cp.encoding_work().unwrap(), cfirst.checked_add(cfirst).unwrap());
        assert_eq!(sp.encoding_work().unwrap(), sfirst.checked_add(sfirst).unwrap());
    }

    #[test]
    fn every_cancellation_cause_survives_and_prepared_encoding_is_never_refunded() {
        struct Count { seen: usize, stop: usize, cause: DecodeCancellationKind }
        impl DecodeStepControl for Count {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
                self.seen += 1; (self.seen == self.stop).then_some(self.cause)
            }
        }
        let c = classifier(call()); let s = sentiment(call());
        let cp = classify_factory(&c, defaults(), total()); let sp = sentiment_factory(&s, total());
        let mut cc = Count { seen: 0, stop: usize::MAX, cause: DecodeCancellationKind::User };
        let mut sc = Count { seen: 0, stop: usize::MAX, cause: DecodeCancellationKind::User };
        cp.prepare_with_control(document(None), &mut cc).unwrap();
        sp.prepare_with_control(document(None), &mut sc).unwrap();
        let cspent = cp.encoding_work().unwrap(); let sspent = sp.encoding_work().unwrap();
        let mut ccap = total(); ccap.max_input_bytes = cspent.input_bytes() as u64;
        let mut scap = total(); scap.max_input_bytes = sspent.input_bytes() as u64;
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let cp = classify_factory(&c, defaults(), ccap); let sp = sentiment_factory(&s, scap);
            for error in [
                cp.prepare_with_control(document(None), &mut Count { seen: 0, stop: cc.seen, cause }).err().unwrap(),
                sp.prepare_with_control(document(None), &mut Count { seen: 0, stop: sc.seen, cause }).err().unwrap(),
            ] { assert!(error.stop); assert_eq!(error.fault.cancellation, Some(cause)); }
            assert_eq!(cp.encoding_work().unwrap(), cspent); assert_eq!(sp.encoding_work().unwrap(), sspent);
            for error in [cp.prepare_with_control(document(None), &mut Continue).err().unwrap(),
                sp.prepare_with_control(document(None), &mut Continue).err().unwrap()] {
                assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
            }
            assert_eq!(cp.encoding_work().unwrap(), cspent); assert_eq!(sp.encoding_work().unwrap(), sspent);
        }
    }

    #[test]
    fn preparation_owner_is_shared_by_alias_only_and_unwind_cannot_restore_it() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let encoding = UntrustedEncoding::SafeBpeV1(call());
        let owner = EncodingWorkScope::new(&encoding, Some(total())).unwrap();
        let alias = Arc::clone(&owner);
        let other = EncodingWorkScope::new(&encoding, Some(total())).unwrap();
        owner.verify(&alias).unwrap();
        let wrong = owner.verify(&other).err().unwrap();
        assert!(wrong.stop); assert_eq!(wrong.fault.code, BatchCode::Admission);
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        let controls = pinned_controls::pinned().unwrap();
        let encoder = UntrustedDocumentEncoder::new(tokenizer.tokenizer(), controls.template_controls());
        let unwind = catch_unwind(AssertUnwindSafe(|| {
            owner.prepare::<(), _>(&mut Continue, |ceiling, work, _| {
                encoder.encode_with_policy(b"Alice private text", &UntrustedEncoding::SafeBpeV1(ceiling.unwrap()),
                    work, &mut || true).unwrap();
                assert!(work.input_bytes() > 0);
                let blocked = alias.prepare::<(), _>(&mut Continue, |_, _, _| Ok(())).err().unwrap();
                assert!(blocked.stop); assert_eq!(blocked.fault.code, BatchCode::InvalidExecution);
                panic!("injected preparation unwind after actual tokenization");
            })
        }));
        assert!(unwind.is_err());
        assert!(owner.work().is_err()); assert!(alias.work().is_err());
        let refused = alias.prepare::<(), _>(&mut Continue, |_, _, _| Ok(())).err().unwrap();
        assert!(refused.stop); assert_eq!(refused.fault.code, BatchCode::InvalidExecution);
        assert_eq!(other.work().unwrap(), SafeBpeWork::default());
    }
}
