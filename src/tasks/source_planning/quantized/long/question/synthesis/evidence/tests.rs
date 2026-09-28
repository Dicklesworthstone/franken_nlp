//! Multi-passage citation mechanics only, not model-success or QA-quality evidence.
use super::*;
use crate::{native_engine::decode::DecodeCancellationKind, tasks::answer::PassageOccurrence};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn mapped(parts: &[&str]) -> (String, QuestionValue) {
    let mut source = String::new(); let mut chunks = Vec::new(); let mut scalar = 0;
    for (id, &text) in parts.iter().enumerate() {
        let byte = source.len(); source.push_str(text); let end = scalar + text.chars().count();
        let span = VerifiedSourceSpan { byte_start: byte, byte_end: source.len(), scalar_start: scalar, scalar_end: end };
        chunks.push(Arc::new(QuestionChunk { chunk_id: id, source_span: span, status: QuestionChunkStatus::Answered,
            answer: Some("untrusted proposal".to_owned()), citations: vec![SourceCitation {
                quote: text.to_owned(), occurrence: SourceOccurrence::Anchored, spans: vec![span] }],
            model_work: Int8Work::default(), mask_node_visit_charge: 0 }));
        scalar = end;
    }
    (source, QuestionValue { chunks })
}
fn fixture(parts: &[&str]) -> (String, Collection) {
    let (source, value) = mapped(parts);
    let evidence = collect(&source, &value, QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 8192 },
        &mut GroundingBudget::default(), &mut Continue).unwrap();
    (source, evidence)
}
fn raw(evidence: &Collection, quotes: &[&str]) -> AnswerResult {
    let citations = quotes.iter().map(|&quote| {
        let mut spans = Vec::new();
        for passage in &evidence.passages {
            match scan_occurrences(&passage.text, quote, &mut GroundingBudget::default()) {
                Ok(found) => spans.extend(found.into_iter().map(|span| PassageOccurrence { passage_id: passage.id.clone(), span })),
                Err(FieldGroundingError::Absent) => {},
                Err(error) => panic!("fixture scan: {error}"),
            }
        }
        PassageCitation { quote: quote.to_owned(), occurrence: occurrence(spans.len()), spans }
    }).collect();
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
        status: AnswerStatus::Answered, answerable: true, answer: Some("answer".to_owned()), citations,
        calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        untrusted_fields: ["answer".to_owned(), "citations".to_owned()], generated_token_ids: vec![1, 0],
        forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 0 }
}
fn finish(source: &str, evidence: &Collection, result: AnswerResult) -> Result<SynthesisAnswer, Int8SourceMapError> {
    final_answer(source, result, evidence, AnswerOptions::default(), &mut GroundingBudget::default(), &mut Continue)
}
#[test]
fn citations_can_name_only_the_first_middle_or_last_of_distinct_evidence_passages() {
    let (source, evidence) = fixture(&["é Alice ", "上海 Bob ", "😀 Carol"]);
    for quote in ["Alice", "Bob", "Carol"] {
        let answer = finish(&source, &evidence, raw(&evidence, &[quote])).unwrap();
        let citation = &answer.citations[0]; assert_eq!(citation.occurrence, SourceOccurrence::Anchored);
        assert_eq!(citation.spans.len(), 1); let span = citation.spans[0];
        assert_eq!(&source[span.byte_start..span.byte_end], quote);
        assert_eq!(span.scalar_start, source[..span.byte_start].chars().count());
        assert_eq!(span.scalar_end, source[..span.byte_end].chars().count());
    }
}
#[test]
fn one_answer_can_cite_different_passages_without_requiring_every_quote_in_every_passage() {
    let (source, evidence) = fixture(&["Alice ", "Bob ", "Carol"]);
    let answer = finish(&source, &evidence, raw(&evidence, &["Carol", "Alice"])).unwrap();
    assert_eq!(answer.citations.len(), 2);
    assert_eq!(answer.citations[0].spans[0].byte_start, 10);
    assert_eq!(answer.citations[1].spans[0].byte_start, 0);
}
#[test]
fn nonmatching_passages_do_not_stop_later_occurrence_verification() {
    let (source, evidence) = fixture(&["Alice ", "Bob ", "Alice"]);
    let result = raw(&evidence, &["Alice"]);
    let answer = finish(&source, &evidence, result.clone()).unwrap();
    assert_eq!(answer.citations[0].occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(answer.citations[0].spans.len(), 2);
    for axis in 0..4 {
        let mut bad = result.clone(); let citation = &mut bad.citations[0];
        match axis {
            0 => { citation.spans.pop(); },
            1 => citation.spans[1].passage_id = evidence.passages[1].id.clone(),
            2 => citation.spans.swap(0, 1),
            _ => citation.spans[1].span.scalar_end += 1,
        }
        assert!(finish(&source, &evidence, bad).is_err());
    }
}
#[test]
fn absence_in_all_passages_and_matches_only_across_a_join_still_refuse() {
    let (source, evidence) = fixture(&["Alice", "Bob"]);
    for quote in ["missing", "AliceBob", "Alice\n\nBob"] {
        assert!(finish(&source, &evidence, raw(&evidence, &[quote])).is_err());
    }
    let mut fabricated = raw(&evidence, &["Alice"]);
    fabricated.citations[0].quote = "missing".to_owned();
    assert!(finish(&source, &evidence, fabricated).is_err());
}
#[test]
fn no_match_scans_remain_charged_without_consuming_or_refunding_occurrence_allowances() {
    let (source, evidence) = fixture(&["Alice", "Bob", "Carol"]);
    let quote = "Alice";
    let cost: u64 = evidence.passages.iter().map(|p| (p.text.len() as u64 + quote.len() as u64 + 1) * 8).sum();
    let mut exact = GroundingBudget { max_fields: 1, max_matches: 2, max_scan_steps: cost };
    final_answer(&source, raw(&evidence, &[quote]), &evidence, AnswerOptions::default(), &mut exact, &mut Continue).unwrap();
    assert_eq!(exact, GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 });
    // The last passage does not match, but its scan cannot exceed the ledger.
    for (fields, matches, steps) in [(1, 2, cost - 1), (1, 1, cost), (0, 2, cost)] {
        let mut budget = GroundingBudget { max_fields: fields, max_matches: matches, max_scan_steps: steps };
        assert!(matches!(final_answer(&source, raw(&evidence, &[quote]), &evidence, AnswerOptions::default(),
            &mut budget, &mut Continue), Err(Int8SourceMapError::WorkLimit)));
    }
}
#[test]
fn the_collection_boundary_still_requires_a_quote_in_its_declared_original_chunk() {
    let (source, mut value) = mapped(&["Alice", "Bob"]);
    Arc::get_mut(&mut value.chunks[0]).unwrap().citations[0].quote = "Bob".to_owned();
    assert!(collect(&source, &value, QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 8192 },
        &mut GroundingBudget::default(), &mut Continue).is_err());
}
#[test]
fn cancellation_at_every_boundary_including_after_a_nonmatching_passage_prevents_completion() {
    struct Count { calls: usize, stop: Option<usize> }
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let (source, evidence) = fixture(&["Bob", "Alice", "Carol"]);
    let mut count = Count { calls: 0, stop: None };
    final_answer(&source, raw(&evidence, &["Alice"]), &evidence, AnswerOptions::default(),
        &mut GroundingBudget::default(), &mut count).unwrap();
    for stop in 1..=count.calls {
        let error = final_answer(&source, raw(&evidence, &["Alice"]), &evidence, AnswerOptions::default(),
            &mut GroundingBudget::default(), &mut Count { calls: 0, stop: Some(stop) }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}
