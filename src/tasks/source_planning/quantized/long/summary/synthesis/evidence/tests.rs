//! Synthetic private evidence fixtures, not passing neural or quality receipts.
use super::*;
use std::sync::Arc;
struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop).then_some(DecodeCancellationKind::Deadline)
    }
}
fn continuing() -> Control { Control { calls: 0, stop: usize::MAX } }
fn limits() -> SummarySynthesisLimits {
    SummarySynthesisLimits { max_evidence_segments: 64, max_evidence_bytes: 4096, verification: GroundingBudget::default() }
}
fn citation(text: &str, quote: &str) -> SourceCitation {
    let spans = scan_occurrences(text, quote, &mut GroundingBudget::default()).unwrap();
    SourceCitation { quote: quote.to_owned(), occurrence: occurrence(spans.len()), spans }
}
fn summary(text: &str, quotes: &[&str]) -> SummaryResult {
    SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        bullets: quotes.iter().map(|q| CitedBullet { text: "UNTRUSTED_GENERATED_ASSERTION".to_owned(),
            citations: vec![citation(text, q)] }).collect(), generated_token_ids: vec![1, 0],
        forward_positions: 1, projected_logits: 166_144, mask_node_visit_charge: 1 }
}
fn value(parts: &[(&str, &[&str])]) -> (String, SourceMapValue) {
    let mut source = String::new(); let mut chunks = Vec::new(); let mut scalar = 0;
    for (id, (text, quotes)) in parts.iter().enumerate() {
        let span = VerifiedSourceSpan { byte_start: source.len(), byte_end: source.len() + text.len(),
            scalar_start: scalar, scalar_end: scalar + text.chars().count() };
        source.push_str(text); scalar = span.scalar_end;
        let raw = summary(text, quotes);
        let original_spans = raw.bullets.iter().enumerate().map(|(bullet, b)| OriginalSourceSpans {
            field: SourceMapField::SummaryCitation { bullet, citation: 0 },
            spans: b.citations[0].spans.iter().map(|&s| lift(s, span).unwrap()).collect() }).collect();
        chunks.push(Arc::new(MappedSourceChunk { chunk_id: id, source_span: span, original_spans,
            native: Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
                model_work: Int8Work::default(), result: SourceTaskResult::Summarize(raw) } }));
    }
    (source, SourceMapValue { chunks })
}
fn collect_value(source: &str, value: &SourceMapValue, remaining: &mut GroundingBudget) -> Collection {
    collect(source, value, SummaryOptions::default(), limits(), remaining, &mut continuing()).unwrap()
}
#[test]
fn only_exact_quotes_enter_evidence_and_equal_quotes_keep_distinct_chunk_origins() {
    let (source, value) = value(&[("éA", &["é"]), ("éA", &["é"])]);
    let mut budget = limits().verification;
    let evidence = collect_value(&source, &value, &mut budget);
    assert_eq!(evidence.text, "é\n\né"); assert_eq!(evidence.segment_count(), 2);
    assert!(!evidence.text.contains("UNTRUSTED_GENERATED_ASSERTION"));
    let raw = summary(&evidence.text, &["é"]);
    let lifted = lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).unwrap();
    let spans = &lifted[0].citations[0].spans;
    assert_eq!(spans.len(), 2); assert_eq!(spans[1].byte_start, 3); assert_eq!(spans[1].scalar_start, 2);
    assert_eq!(lifted[0].citations[0].occurrence, SourceOccurrence::Ambiguous);
    for span in spans { assert_eq!(&source[span.byte_start..span.byte_end], "é"); }
}
#[test]
fn subquotes_lift_through_every_origin_without_claiming_unselected_occurrences() {
    let (source, value) = value(&[("上海上海", &["上海"]), ("上海", &[])]);
    let mut budget = limits().verification;
    let evidence = collect_value(&source, &value, &mut budget);
    assert_eq!(evidence.text, "上海");
    let raw = summary(&evidence.text, &["海"]);
    let lifted = lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).unwrap();
    assert_eq!(lifted[0].citations[0].spans.iter().map(|s| s.byte_start).collect::<Vec<_>>(), vec![3, 9]);
    assert_eq!(lifted[0].citations[0].spans.iter().map(|s| s.scalar_start).collect::<Vec<_>>(), vec![1, 3]);
}
#[test]
fn join_crossing_quotes_fail_even_if_the_same_text_exists_in_the_original() {
    let (source, value) = value(&[("a\n\nb", &["a", "b"])]);
    let mut budget = limits().verification;
    let evidence = collect_value(&source, &value, &mut budget);
    assert_eq!(evidence.text, source);
    for quote in ["a\n\nb", "\n", "a\n", "\nb"] {
        let raw = summary(&evidence.text, &[quote]);
        assert!(lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).is_err());
    }
}
#[test]
fn corrupt_duplicate_or_late_coordinates_cannot_hide_behind_deduplication() {
    for axis in 0..4 {
        let (source, mut value) = value(&[("éé", &["é", "é"])]);
        let chunk = Arc::get_mut(&mut value.chunks[0]).unwrap();
        match axis {
            0 => chunk.original_spans[1].spans[0].scalar_start += 1,
            1 => chunk.original_spans[1].field = SourceMapField::SummaryCitation { bullet: 0, citation: 0 },
            2 => { chunk.original_spans[1].spans.pop(); },
            _ => { let SourceTaskResult::Summarize(raw) = &mut chunk.native.result else { panic!() };
                raw.bullets[1].citations[0].occurrence = SourceOccurrence::Anchored; }
        }
        assert!(collect(&source, &value, SummaryOptions::default(), limits(), &mut limits().verification, &mut continuing()).is_err());
    }
}
#[test]
fn no_final_top_k_can_hide_evidence_count_or_framed_byte_overflow() {
    let (source, value) = value(&[("ab", &["a", "b"])]);
    for axis in 0..2 {
        let mut cap = limits();
        if axis == 0 { cap.max_evidence_segments = 1; } else { cap.max_evidence_bytes = 3; }
        assert!(collect(&source, &value, SummaryOptions::default(), cap, &mut cap.verification, &mut continuing()).is_err());
    }
    let mut cap = limits(); cap.max_evidence_bytes = 4;
    assert_eq!(collect(&source, &value, SummaryOptions::default(), cap, &mut cap.verification, &mut continuing()).unwrap().text, "a\n\nb");
}
#[test]
fn repeated_and_overlapping_evidence_deduplicates_original_offsets_after_charging_fanout() {
    let (source, value) = value(&[("éé", &["é", "éé"])]);
    let mut budget = limits().verification;
    let evidence = collect_value(&source, &value, &mut budget);
    let before = budget.max_matches;
    let raw = summary(&evidence.text, &["é"]);
    let result = lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).unwrap();
    assert_eq!(result[0].citations[0].spans.len(), 2);
    assert_eq!(before - budget.max_matches, 7); // Three native matches plus four origin lifts.
}
#[test]
fn collection_and_final_lift_share_fields_matches_and_scan_work_without_refresh() {
    let (source, value) = value(&[("éé", &["é"])]);
    let mut initial = limits().verification;
    let evidence = collect_value(&source, &value, &mut initial);
    let raw = summary(&evidence.text, &["é"]);
    for axis in 0..3 {
        let mut budget = initial;
        match axis { 0 => budget.max_fields = 0, 1 => budget.max_matches = 2, _ => budget.max_scan_steps = 0 }
        assert!(lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).is_err());
    }
    let mut budget = limits().verification; budget.max_fields = 1;
    let evidence = collect_value(&source, &value, &mut budget); assert_eq!(budget.max_fields, 0);
    assert!(lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).is_err());
}
#[test]
fn omitted_native_match_and_forged_profile_are_refused() {
    let (source, value) = value(&[("ab", &["a"]), ("ab", &["a"])]);
    let mut budget = limits().verification;
    let evidence = collect_value(&source, &value, &mut budget);
    let mut raw = summary(&evidence.text, &["a"]);
    raw.bullets[0].citations[0].spans.pop(); raw.bullets[0].citations[0].occurrence = SourceOccurrence::Anchored;
    assert!(lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).is_err());
    let mut raw = summary(&evidence.text, &["a"]); raw.numerics_profile = "hf-bf16-eager".to_owned();
    assert!(lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut continuing()).is_err());
}
#[test]
fn complete_source_coverage_including_crlf_and_empty_map_summaries_is_mandatory() {
    let (source, mut value) = value(&[("é\r\n", &["é"]), ("\r\n", &[])]);
    let mut budget = limits().verification;
    assert_eq!(collect_value(&source, &value, &mut budget).text, "é");
    value.chunks.pop();
    assert!(collect(&source, &value, SummaryOptions::default(), limits(), &mut limits().verification, &mut continuing()).is_err());
    let (source, value) = self::value(&[("\r\n", &[])]);
    assert_eq!(collect_value(&source, &value, &mut limits().verification).segment_count(), 0);
}
#[test]
fn cancellation_during_collection_or_final_lift_never_returns_partial_bullets() {
    let (source, value) = value(&[("éA", &["é"]), ("éA", &["é"])]);
    let mut control = continuing(); let mut budget = limits().verification;
    let evidence = collect(&source, &value, SummaryOptions::default(), limits(), &mut budget, &mut control).unwrap();
    let last = control.calls;
    for stop in [1, last] {
        let error = collect(&source, &value, SummaryOptions::default(), limits(), &mut limits().verification,
            &mut Control { calls: 0, stop }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
    let raw = summary(&evidence.text, &["é"]); let mut control = continuing();
    lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut budget, &mut control).unwrap();
    for stop in [1, control.calls] {
        let error = lift_summary(&source, &raw, &evidence, SummaryOptions::default(), &mut limits().verification,
            &mut Control { calls: 0, stop }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}
