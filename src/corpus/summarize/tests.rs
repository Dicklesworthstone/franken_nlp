use super::*;
use crate::{tasks::{mapreduce::{self, ChunkLimits, ChunkPlan, ExecutionLimits}, summarize::CitedBullet},
    validation::{SourceSpan, validate_source_span}};

fn plan(source: &str) -> ChunkPlan<'_> {
    ChunkPlan::build(source, ChunkLimits { max_chunk_bytes: 4, max_chunk_tokens: 4,
        ..ChunkLimits::default() }, |s| Ok(s.len())).unwrap()
}
fn raw(chunk: &SourceChunk<'_>, entries: &[(&str, &[&str])]) -> SummaryResult {
    let bullets = entries.iter().map(|(text, quotes)| CitedBullet { text: (*text).to_owned(),
        citations: quotes.iter().map(|quote| {
            let spans = scan_occurrences(chunk.text(), quote, &mut GroundingBudget::default()).unwrap();
            SourceCitation { quote: (*quote).to_owned(), occurrence: if spans.len() == 1 {
                SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }
        }).collect() }).collect();
    SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
        numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        bullets, generated_token_ids: vec![1, 0], forward_positions: 7, projected_logits: 13, mask_node_visit_charge: 17 }
}
fn map(chunk: &SourceChunk<'_>, value: SummaryResult) -> SummaryAggregate {
    map_value(chunk, value, SummaryOptions::default(), CorpusSummaryLimits::default(), &mut u64::MAX).unwrap()
}
#[test]
fn duplicate_text_and_quotes_merge_without_double_voting_or_lost_evidence() {
    let p = plan("abab"); let c = &p.chunks()[0];
    let value = map(c, raw(c, &[("same", &["a", "a"]), ("same", &["b"])]));
    assert_eq!(value.bullets.len(), 1); let b = &value.bullets[0];
    assert_eq!(b.rank_sum, 1); assert_eq!(b.evidence.len(), 1);
    assert_eq!(b.evidence[0].citations.len(), 2);
    assert_eq!(b.evidence[0].citations[0].spans.len(), 2);
}
#[test]
fn citations_lift_over_unicode_chunks_and_keep_overlapping_occurrences() {
    let source = "éaaaaaaaaaa"; let p = plan(source);
    for chunk in p.chunks() {
        let quote = if chunk.id() == 0 { "é" } else { "aa" };
        let value = map(chunk, raw(chunk, &[("claim", &[quote])]));
        for citation in &value.bullets[0].evidence[0].citations { for s in &citation.spans {
            validate_source_span(source, &citation.quote, SourceSpan::new(s.byte_start, s.byte_end,
                s.scalar_start, s.scalar_end)).unwrap();
        } }
    }
}
#[test]
fn forged_missing_reordered_and_incomplete_citations_fail_independent_recheck() {
    let p = plan("aaaa"); let c = &p.chunks()[0];
    for mode in 0..6 {
        let mut value = raw(c, &[("claim", &["aa"])]);
        let cite = &mut value.bullets[0].citations[0];
        match mode { 0 => cite.spans[0].byte_start += 1, 1 => { cite.spans.pop(); },
            2 => cite.spans.reverse(), 3 => cite.occurrence = SourceOccurrence::Anchored,
            4 => cite.quote = "zz".to_owned(), _ => cite.spans[0].scalar_end += 1 }
        assert!(map_value(c, value, SummaryOptions::default(), CorpusSummaryLimits::default(), &mut u64::MAX).is_err());
    }
}
#[test]
fn contradictory_prose_is_retained_without_semantic_certification() {
    let p = plan("aaaa"); let c = &p.chunks()[0];
    let result = map(c, raw(c, &[("True", &["a"]), ("False", &["a"])])).into_ranked(8, 10000).unwrap();
    assert_eq!(result.bullets.len(), 2); assert_eq!(result.semantic_support, SummarySemanticSupport::NotAssessed);
    assert!(result.warnings.contains(&SummaryWarning::SemanticConflictsNotReconciled));
}
struct Fixture;
impl SummaryPass for Fixture {
    type Error = SummaryError;
    fn options(&self) -> SummaryOptions { SummaryOptions::default() }
    fn run(&mut self, c: &SourceChunk<'_>) -> Result<SummaryResult, SummaryError> {
        Ok(raw(c, &[(if c.id() % 2 == 0 { "local even" } else { "local odd" }, &["a"]), ("global", &["a"])]))
    }
}
#[test]
fn global_winner_survives_every_tree_shape_and_map_batch_size() {
    let p = plan("aaaaaaaaaaaaaaaaaaaaaaaaaaaa"); let mut expected = None;
    for fan in 2..=8 { for batch in 1..=7 {
        let mut task = CorpusSummaryTask::new(Fixture, CorpusSummaryLimits::default()).unwrap();
        let result = mapreduce::execute(&p, &mut task, ExecutionLimits { reduce_fan_in: fan, map_batch_chunks: batch,
            ..ExecutionLimits::default() }, || Ok(())).unwrap().into_value().into_ranked(1, 100000).unwrap();
        assert_eq!(result.bullets[0].text, "global"); assert_eq!(result.bullets[0].evidence.len(), 7);
        assert_eq!(result.omitted_bullets, 2); assert_eq!(result.forward_positions, 49);
        let bytes = canonjson::canonical_bytes(&result).unwrap();
        if let Some(ref old) = expected { assert_eq!(&bytes, old); } else { expected = Some(bytes); }
    } }
}
#[test]
fn scan_budget_is_shared_across_maps_and_failure_poisoning_prevents_retry() {
    let p = plan("aaaaaaaa");
    let mut task = CorpusSummaryTask::new(Fixture, CorpusSummaryLimits { max_scan_steps: 96,
        ..CorpusSummaryLimits::default() }).unwrap();
    assert!(task.map_batch(&p.chunks()[..1]).is_ok());
    assert_eq!(task.scan_steps_remaining(), 0);
    assert!(task.map_batch(&p.chunks()[1..]).is_err());
    assert!(matches!(task.map_batch(&p.chunks()[1..]), Err(CorpusSummaryError::Poisoned)));
}
#[test]
fn limits_refuse_complete_results_instead_of_truncating_citations() {
    let p = plan("aaaa"); let c = &p.chunks()[0];
    for axis in 0..4 {
        let mut limits = CorpusSummaryLimits::default();
        match axis { 0 => limits.max_unique_bullets = 1, 1 => limits.max_citations = 1,
            2 => limits.max_evidence_spans = 1, _ => limits.max_value_bytes = 1 }
        assert!(map_value(c, raw(c, &[("one", &["a"]), ("two", &["a"])]),
            SummaryOptions::default(), limits, &mut u64::MAX).is_err());
    }
}
#[test]
fn reduction_checks_complete_ranges_and_checked_work_totals() {
    let p = plan("aaaaaaaaaaaa");
    let a = map(&p.chunks()[0], raw(&p.chunks()[0], &[]));
    let mut b = map(&p.chunks()[1], raw(&p.chunks()[1], &[]));
    let c = map(&p.chunks()[2], raw(&p.chunks()[2], &[]));
    assert!(merge_values([&a, &c], CorpusSummaryLimits::default()).is_err());
    assert!(merge_values([&b, &a], CorpusSummaryLimits::default()).is_err());
    assert!(merge_values([&a, &a], CorpusSummaryLimits::default()).is_err());
    b.forward_positions = u64::MAX;
    assert!(merge_values([&a, &b], CorpusSummaryLimits::default()).is_err());
}
#[test]
fn empty_maps_still_count_toward_complete_source_lineage() {
    let p = plan("aaaaaaaa");
    let a = map(&p.chunks()[0], raw(&p.chunks()[0], &[]));
    let b = map(&p.chunks()[1], raw(&p.chunks()[1], &[]));
    let result = merge_values([&a, &b], CorpusSummaryLimits::default()).unwrap().into_ranked(2, 10000).unwrap();
    assert_eq!(result.mapped_chunks, 2); assert!(result.bullets.is_empty()); assert_eq!(result.forward_positions, 14);
}
#[test]
fn final_complete_envelope_has_exact_byte_boundary() {
    let p = plan("aaaa"); let c = &p.chunks()[0];
    let make = || map(c, raw(c, &[("é\n\"", &["a"])]));
    let n = canonjson::canonical_bytes(&make().into_ranked(1, 10000).unwrap()).unwrap().len();
    assert!(make().into_ranked(1, n).is_ok()); assert!(make().into_ranked(1, n - 1).is_err());
}
#[test]
fn duplicate_chunk_admission_is_rejected_even_when_it_has_no_bullets() {
    let p = plan("aaaa"); let mut task = CorpusSummaryTask::new(Fixture, CorpusSummaryLimits::default()).unwrap();
    assert!(task.map_batch(p.chunks()).is_ok()); assert!(task.map_batch(p.chunks()).is_err());
}
