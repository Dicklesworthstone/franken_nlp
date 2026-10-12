//! Model-free planning and accounting regressions, not native inference tests.
use super::*;
use crate::{canonjson, corpus::summarize::CorpusSummaryLimits,
    tasks::{keyphrases::KeyphraseOptions, summarize::SummaryOptions,
        source_planning::quantized::Int8SourceError},
    tokenizer::pinned_controls, validation::grounded_fields::VerifiedSourceSpan};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn factory(summary: bool) -> Int8SourceDocumentJobPlanner {
    let old = super::super::tests::factory();
    let task = if summary { SourceMapTask::Summarize(SummaryOptions::default()) }
        else { SourceMapTask::Keyphrases(KeyphraseOptions::default()) };
    let mut identity = old.identity;
    identity.task_spec = if summary { "summarize-v1" } else { "keyphrases-v1" }.into();
    let registry = pinned_controls::pinned().unwrap();
    Int8SourceDocumentJobPlanner::pinned(registry.template_controls(), 166101, identity, task,
        old.budget, old.planning, old.mapping, old.limits, &mut Continue).unwrap()
}
fn policy(summary: bool) -> SourceDocumentReduction {
    if summary {
        SourceDocumentReduction::Summary { limits: Int8SummaryLimits {
            aggregation: CorpusSummaryLimits { max_value_bytes: 512 * 1024, max_scan_steps: 100_000,
                ..CorpusSummaryLimits::default() }, max_bullets: 8, max_result_bytes: 2 * 1024 * 1024 },
            max_lifetime_scan_work: 400_000 }
    } else {
        SourceDocumentReduction::Keyphrases { limits: Int8KeyphraseLimits {
            aggregation: CorpusKeyphraseLimits { max_value_bytes: 512 * 1024, max_scan_work: 100_000,
                ..CorpusKeyphraseLimits::default() }, max_phrases: 8, max_result_bytes: 2 * 1024 * 1024 },
            max_lifetime_scan_work: 400_000 }
    }
}
fn document() -> BatchDocument<SourceDocumentArgs> {
    BatchDocument { id: "doc".into(), text: "é Alice Bob 上海 Carol".into(), task_args: None }
}
#[test]
fn ordinary_maps_keep_the_legacy_recipe_projection() {
    let p = factory(true); let value = serde_json::to_value(p.job_recipe()).unwrap();
    assert!(p.reduction_policy().is_none()); assert_eq!(value["version"], 1);
    assert_eq!(value["execution"], INT8_SOURCE_MAP_EXECUTION);
    assert!(!value.as_object().unwrap().contains_key("reduction_task"));
    assert_eq!(value.as_object().unwrap().len(), 10);
}
#[test]
fn both_reducers_preserve_native_work_and_change_recipe_and_prepared_owner() {
    for summary in [false, true] {
        let p = factory(summary);
        let original = p.prepare_with_control(document(), &mut Continue).unwrap();
        let old_recipe = canonjson::canonical_bytes(p.job_recipe()).unwrap();
        let p = p.with_reduction(policy(summary)).unwrap();
        assert!(p.owns(&original).is_err());
        let prepared = p.prepare_with_control(document(), &mut Continue).unwrap();
        assert!(prepared.chunk_count() > 1);
        assert_eq!(prepared.work, original.work); assert_eq!(prepared.identities, original.identities);
        p.owns(&prepared).unwrap();
        assert_eq!(p.recipe.version, 2); assert_eq!(p.recipe.execution, policy(summary).execution());
        assert_ne!(old_recipe, canonjson::canonical_bytes(p.job_recipe()).unwrap());
        let again = factory(summary).with_reduction(policy(summary)).unwrap();
        assert_eq!(canonjson::canonical_bytes(p.job_recipe()).unwrap(), canonjson::canonical_bytes(again.job_recipe()).unwrap());
    }
}
#[test]
fn different_reduction_policy_invalidates_prepared_ownership() {
    let p = factory(true).with_reduction(policy(true)).unwrap();
    let prepared = p.prepare_with_control(document(), &mut Continue).unwrap();
    let before = canonjson::canonical_bytes(p.job_recipe()).unwrap();
    let SourceDocumentReduction::Summary { mut limits, max_lifetime_scan_work } = policy(true) else { unreachable!() };
    limits.max_bullets += 1;
    let p = p.with_reduction(SourceDocumentReduction::Summary { limits, max_lifetime_scan_work }).unwrap();
    assert!(p.owns(&prepared).is_err()); assert_ne!(before, canonjson::canonical_bytes(p.job_recipe()).unwrap());
}
#[test]
fn task_mismatch_is_refused_before_execution() {
    for summary in [false, true] {
        let p = factory(summary); let text = document().text;
        let plan = p.plan(&text, &mut Continue).unwrap();
        policy(summary).check_plan(&plan).unwrap();
        assert!(policy(!summary).check_plan(&plan).is_err());
        assert!(policy(!summary).validate(&p.task, p.mapping, p.limits).is_err());
    }
    assert!(super::super::tests::factory().with_reduction(policy(true)).is_err());
}
#[test]
fn scan_allowance_covers_all_attempts_and_rejects_overflow() {
    assert!(scans_cover_attempts(100, 400, 4));
    for (per, max, attempts) in [(100, 399, 4), (100, 200, 4), (0, 400, 4), (100, 400, 0), (u64::MAX, u64::MAX, 2)] {
        assert!(!scans_cover_attempts(per, max, attempts));
    }
    for summary in [false, true] {
        let p = factory(summary); let mut job = p.limits;
        policy(summary).validate(&p.task, p.mapping, job).unwrap(); job.max_attempts += 1;
        assert!(policy(summary).validate(&p.task, p.mapping, job).is_err());
    }
}
#[test]
fn every_summary_limit_changes_the_contract() {
    let original = canonjson::canonical_bytes(&policy(true)).unwrap();
    for axis in 0..8 {
        let SourceDocumentReduction::Summary { mut limits, mut max_lifetime_scan_work } = policy(true) else { unreachable!() };
        match axis { 0 => limits.aggregation.max_unique_bullets += 1, 1 => limits.aggregation.max_citations += 1,
            2 => limits.aggregation.max_evidence_spans += 1, 3 => limits.aggregation.max_value_bytes += 1,
            4 => limits.aggregation.max_scan_steps += 1, 5 => limits.max_bullets += 1,
            6 => limits.max_result_bytes += 1, _ => max_lifetime_scan_work += 1 }
        assert_ne!(original, canonjson::canonical_bytes(&SourceDocumentReduction::Summary { limits, max_lifetime_scan_work }).unwrap());
    }
}
#[test]
fn every_keyphrase_limit_changes_the_contract() {
    let original = canonjson::canonical_bytes(&policy(false)).unwrap();
    for axis in 0..7 {
        let SourceDocumentReduction::Keyphrases { mut limits, mut max_lifetime_scan_work } = policy(false) else { unreachable!() };
        match axis { 0 => limits.aggregation.max_unique_phrases += 1, 1 => limits.aggregation.max_evidence_spans += 1,
            2 => limits.aggregation.max_value_bytes += 1, 3 => limits.aggregation.max_scan_work += 1,
            4 => limits.max_phrases += 1, 5 => limits.max_result_bytes += 1, _ => max_lifetime_scan_work += 1 }
        assert_ne!(original, canonjson::canonical_bytes(&SourceDocumentReduction::Keyphrases { limits, max_lifetime_scan_work }).unwrap());
    }
}
#[test]
fn final_top_k_does_not_waive_aggregate_or_output_limits() {
    let p = factory(true);
    let SourceDocumentReduction::Summary { mut limits, max_lifetime_scan_work } = policy(true) else { unreachable!() };
    limits.max_bullets = 1; limits.aggregation.max_value_bytes = p.mapping.reduction.max_value_bytes + 1;
    assert!(SourceDocumentReduction::Summary { limits, max_lifetime_scan_work }.validate(&p.task, p.mapping, p.limits).is_err());
    limits.aggregation.max_value_bytes = 512 * 1024; limits.max_result_bytes = p.limits.max_result_bytes + 1;
    assert!(SourceDocumentReduction::Summary { limits, max_lifetime_scan_work }.validate(&p.task, p.mapping, p.limits).is_err());
}
#[test]
fn completion_covers_unicode_source_chunks_and_every_work_axis() {
    let p = factory(false).with_reduction(policy(false)).unwrap();
    let prepared = p.prepare_with_control(document(), &mut Continue).unwrap();
    let span = VerifiedSourceSpan { byte_start: 0, byte_end: prepared.source.len(), scalar_start: 0, scalar_end: prepared.source.chars().count() };
    let check = |planned, observed, chunks, span, scans| execution::check_accounting(&prepared, planned, observed, chunks, span, scans, 100_000);
    check(prepared.work, prepared.work, prepared.chunk_count(), span, 100_000).unwrap();
    assert!(check(prepared.work, prepared.work, prepared.chunk_count() - 1, span, 0).is_err());
    assert!(check(prepared.work, prepared.work, prepared.chunk_count(), span, 100_001).is_err());
    for axis in 0..4 {
        let mut bad = span;
        match axis { 0 => bad.byte_start += 1, 1 => bad.byte_end -= 1, 2 => bad.scalar_start += 1, _ => bad.scalar_end -= 1 }
        assert!(check(prepared.work, prepared.work, prepared.chunk_count(), bad, 0).is_err());
    }
    for axis in 0..6 {
        let mut bad = prepared.work;
        match axis { 0 => bad.model.forward_positions += 1, 1 => bad.model.projected_logits += 1,
            2 => bad.model.attention_pairs += 1, 3 => bad.model.projections.dot_products += 1,
            4 => bad.model.projections.multiply_accumulates += 1, _ => bad.mask_node_visits += 1 }
        assert!(check(bad, prepared.work, prepared.chunk_count(), span, 0).is_err());
        assert!(check(prepared.work, bad, prepared.chunk_count(), span, 0).is_err());
    }
}
#[test]
fn cancellation_cause_is_preserved_for_both_reducers() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        let failure = summary_failure(Int8CorpusSummaryError::Source(Int8SourceError::Cancelled(cause)));
        assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(cause));
        let failure = keyphrase_failure(Int8CorpusKeyphraseError::Source(Int8SourceError::Cancelled(cause)));
        assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(cause));
    }
}
#[test]
fn native_processor_implements_the_durable_interface() {
    struct Refuse;
    impl SourceDocumentBatchAdmission for Refuse {
        type Guard = ();
        fn admit(&mut self, _: SourceDocumentAdmission<'_>) -> Result<(Vec<ExecutionIdentity>, ()), BatchItemFailure> {
            Err(BatchItemFailure::fatal(BatchCode::Admission))
        }
    }
    fn durable<P: DurableBatchProcessor<Recipe = Int8SourceDocumentJobRecipe>>() {}
    durable::<Int8ReducedSourceDocumentJobProcessor<'static, 'static, 'static, 'static, Refuse>>();
}
