//! Aggregate admission uses complete native and evidence frontiers, not top-k.
use super::*;
use super::super::super::tests::fixture;
use crate::{corpus::summarize::CorpusSummaryLimits,
    tasks::corpus_keyphrases::CorpusKeyphraseLimits};

fn summary() -> Int4SummaryLimits {
    Int4SummaryLimits { aggregation: CorpusSummaryLimits { max_value_bytes: 65_536, ..Default::default() },
        max_bullets: 3, max_result_bytes: 65_536 }
}
fn keyphrases() -> Int4KeyphraseLimits {
    Int4KeyphraseLimits { aggregation: CorpusKeyphraseLimits { max_value_bytes: 65_536, ..Default::default() },
        max_phrases: 3, max_result_bytes: 65_536 }
}

#[test]
fn aggregate_kind_and_limits_cannot_be_substituted_by_an_ordinary_map() {
    for task in [SourceMapTask::Ner(Default::default()), SourceMapTask::Keyphrases(Default::default()),
        SourceMapTask::Summarize(Default::default())] {
        let summary_task = matches!(&task, SourceMapTask::Summarize(_));
        let keyphrase_task = matches!(&task, SourceMapTask::Keyphrases(_));
        let (_, config) = fixture(task, 32);
        assert_eq!(DocumentAggregation::validate(summary(), &config).is_ok(), summary_task);
        assert_eq!(DocumentAggregation::validate(keyphrases(), &config).is_ok(), keyphrase_task);
    }
    let (_, config) = fixture(SourceMapTask::Summarize(Default::default()), 32);
    let mut invalid = summary(); invalid.max_bullets = 0;
    assert!(DocumentAggregation::validate(invalid, &config).is_err());
    let mut invalid = summary(); invalid.max_result_bytes = usize::MAX;
    assert!(DocumentAggregation::validate(invalid, &config).is_err());
    let (_, config) = fixture(SourceMapTask::Keyphrases(Default::default()), 32);
    let mut invalid = keyphrases(); invalid.aggregation.max_scan_work = 0;
    assert!(DocumentAggregation::validate(invalid, &config).is_err());
}

#[test]
fn final_selection_does_not_reduce_the_reserved_native_or_evidence_population() {
    let (_, mut config) = fixture(SourceMapTask::Summarize(Default::default()), 32);
    config.mapping.reduction.map_batch_chunks = 4;
    let quote = config.cohort_requirements(32).unwrap();
    let whole = DocumentAggregation::workspace_bytes(summary(), &config, 7).unwrap();
    let mut one = summary(); one.max_bullets = 1;
    assert_eq!(DocumentAggregation::workspace_bytes(one, &config, 7).unwrap(), whole);
    assert_eq!(quote.native.kv_bytes, config.budget.max_kv_bytes * 4);
    assert_eq!(quote, config.cohort_requirements(32).unwrap());
    assert!(DocumentAggregation::workspace_bytes(one, &config, 1).unwrap() < whole);

    let (_, mut config) = fixture(SourceMapTask::Keyphrases(Default::default()), 32);
    config.mapping.reduction.map_batch_chunks = 4;
    let native = config.cohort_requirements(32).unwrap();
    let raw = config.workspace_bytes(7).unwrap();
    let ranked = DocumentAggregation::workspace_bytes(keyphrases(), &config, 7).unwrap();
    let mut one = keyphrases(); one.max_phrases = 1;
    assert!(ranked > raw);
    assert_eq!(DocumentAggregation::workspace_bytes(one, &config, 7).unwrap(), ranked);
    assert_eq!(native, config.cohort_requirements(32).unwrap());
}

#[test]
fn independent_union_evidence_storage_remains_charged_beside_the_map_frontier() {
    let (_, config) = fixture(SourceMapTask::Keyphrases(Default::default()), 32);
    let limits = keyphrases();
    let full = DocumentAggregation::workspace_bytes(limits, &config, 5).unwrap();
    let mut extra = limits; extra.aggregation.max_evidence_spans += 1;
    assert_eq!(DocumentAggregation::workspace_bytes(extra, &config, 5).unwrap() - full,
        2 * std::mem::size_of::<VerifiedSourceSpan>() as u64);
    let mut extra = limits; extra.aggregation.max_unique_phrases += 1;
    assert_eq!(DocumentAggregation::workspace_bytes(extra, &config, 5).unwrap() - full, 512);
    assert!(DocumentAggregation::workspace_bytes(limits, &config, 0).is_err());
    assert!(DocumentAggregation::workspace_bytes(limits, &config, 9).is_err());
}

#[test]
fn output_claim_covers_the_complete_aggregate_and_physical_envelope() {
    assert_eq!(DocumentAggregation::output_bytes(summary()).unwrap(), 65_536 + 4096);
    assert_eq!(DocumentAggregation::output_bytes(keyphrases()).unwrap(), 65_536 + 4096);
    let (_, mut config) = fixture(SourceMapTask::Summarize(Default::default()), 32);
    config.reduction_reserve_bytes = u64::MAX;
    assert!(DocumentAggregation::workspace_bytes(summary(), &config, 1).is_err());
}

#[test]
fn complete_aggregate_ownership_crosses_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4CorpusSummaryCohortRun>();
    send::<Int4CorpusKeyphraseCohortRun>();
    send::<HostedOutput<Int4CorpusSummaryCohortRun>>();
    send::<HostedOutput<Int4CorpusKeyphraseCohortRun>>();
}
