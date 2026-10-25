//! Adversarial multipassage evidence mechanics only, not model quality proof.
use super::*;
use crate::tasks::answer::PassageOccurrence;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn limits() -> QuestionSynthesisLimits { QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 8192 } }
fn mapped(parts: &[(&str, Option<&str>)]) -> (String, QuestionValue) {
    let mut source = String::new(); let mut chunks = Vec::new(); let mut scalar = 0;
    for (id, &(text, quote)) in parts.iter().enumerate() {
        let byte = source.len(); source.push_str(text); let end = scalar + text.chars().count();
        let span = VerifiedSourceSpan { byte_start: byte, byte_end: source.len(), scalar_start: scalar, scalar_end: end };
        let citations = quote.into_iter().map(|quote| {
            let local = scan_occurrences(text, quote, &mut GroundingBudget::default()).unwrap();
            SourceCitation { quote: quote.to_owned(), occurrence: occurrence(local.len()),
                spans: local.into_iter().map(|local| lift(local, span).unwrap()).collect() }
        }).collect();
        chunks.push(Arc::new(QuestionChunk { chunk_id: id, source_span: span,
            status: if quote.is_some() { QuestionChunkStatus::Answered } else if has_text(text) {
                QuestionChunkStatus::Abstained } else { QuestionChunkStatus::WhitespaceOnly },
            answer: quote.map(|_| "UNTRUSTED MODEL ASSERTION".to_owned()), citations,
            model_work: Int4Work::default(), mask_node_visit_charge: 0 }));
        scalar = end;
    }
    (source, QuestionValue { chunks })
}
fn fixture(parts: &[&str]) -> (String, Collection) {
    let parts = parts.iter().map(|text| (*text, Some(*text))).collect::<Vec<_>>();
    let (source, value) = mapped(&parts);
    let evidence = collect(&source, &value, limits(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    (source, evidence)
}
fn raw(evidence: &Collection, quotes: &[&str]) -> AnswerResult {
    let citations = quotes.iter().map(|&quote| {
        let mut spans = Vec::new();
        for passage in &evidence.passages {
            match scan_occurrences(&passage.text, quote, &mut GroundingBudget::default()) {
                Ok(found) => spans.extend(found.into_iter().map(|span| PassageOccurrence { passage_id: passage.id.clone(), span })),
                Err(FieldGroundingError::Absent) => (), Err(error) => panic!("{error}"),
            }
        }
        PassageCitation { quote: quote.to_owned(), occurrence: occurrence(spans.len()), spans }
    }).collect();
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        status: AnswerStatus::Answered, answerable: true, answer: Some("answer".to_owned()), citations,
        calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        untrusted_fields: ["answer".to_owned(), "citations".to_owned()], generated_token_ids: vec![1, 0],
        forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 }
}
fn finish(source: &str, evidence: &Collection, raw: AnswerResult) -> Result<SynthesisAnswer, Int4SourceMapError> {
    final_answer(source, raw, evidence, AnswerOptions::default(), &mut GroundingBudget::default(), &mut Continue)
}
#[test]
fn only_verified_quotes_enter_the_passages_while_all_generated_answers_remain_inspectable() {
    let (source, value) = mapped(&[("é Alice", Some("Alice")), (" Bob", Some("Bob"))]);
    let evidence = collect(&source, &value, limits(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    let passages = evidence.copy_passages().unwrap();
    assert_eq!(passages[0].text, "Alice"); assert_eq!(passages[1].text, "Bob");
    assert_eq!(passages[0].id, "evidence_0"); assert_eq!(passages[1].id, "evidence_1");
    assert!(!canonjson::canonical_string(&passages).unwrap().contains("UNTRUSTED MODEL"));
    assert!(value.chunks().all(|chunk| chunk.answer.as_deref() == Some("UNTRUSTED MODEL ASSERTION")));
}
#[test]
fn citations_can_select_first_middle_or_last_evidence_passage_with_unicode_offsets() {
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
fn one_answer_cites_different_passages_without_requiring_each_quote_everywhere() {
    let (source, evidence) = fixture(&["Alice ", "Bob ", "Carol"]);
    let answer = finish(&source, &evidence, raw(&evidence, &["Carol", "Alice"])).unwrap();
    assert_eq!(answer.citations.len(), 2); assert_eq!(answer.citations[0].spans[0].byte_start, 10);
    assert_eq!(answer.citations[1].spans[0].byte_start, 0);
}
#[test]
fn nonmatching_passages_do_not_skip_later_occurrences_or_passage_id_verification() {
    let (source, evidence) = fixture(&["Alice ", "Bob ", "Alice"]); let result = raw(&evidence, &["Alice"]);
    assert_eq!(finish(&source, &evidence, result.clone()).unwrap().citations[0].spans.len(), 2);
    for axis in 0..5 {
        let mut bad = result.clone(); let citation = &mut bad.citations[0];
        match axis { 0 => { citation.spans.pop(); }, 1 => citation.spans[1].passage_id = "question".to_owned(),
            2 => citation.spans.swap(0, 1), 3 => citation.spans[1].span.scalar_end += 1,
            _ => citation.occurrence = SourceOccurrence::Anchored }
        assert!(finish(&source, &evidence, bad).is_err());
    }
}
#[test]
fn absent_quotes_synthetic_joins_and_question_only_claims_are_refused() {
    let (source, evidence) = fixture(&["Alice", "Bob"]);
    for quote in ["missing", "AliceBob", "Alice\n\nBob", "Who is mentioned?"] {
        assert!(finish(&source, &evidence, raw(&evidence, &[quote])).is_err());
    }
}
#[test]
fn no_match_scans_and_each_original_byte_comparison_consume_the_same_exact_allowance() {
    let (source, evidence) = fixture(&["Alice", "Bob", "Carol"]); let quote = "Alice";
    let scans: u64 = evidence.passages.iter().map(|p| (p.text.len() as u64 + quote.len() as u64 + 1) * 8).sum();
    let cost = scans + quote.len() as u64;
    let mut exact = GroundingBudget { max_fields: 1, max_matches: 2, max_scan_steps: cost };
    final_answer(&source, raw(&evidence, &[quote]), &evidence, AnswerOptions::default(), &mut exact, &mut Continue).unwrap();
    assert_eq!(exact, GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 });
    for (fields, matches, steps) in [(1, 2, cost - 1), (1, 1, cost), (0, 2, cost)] {
        let mut budget = GroundingBudget { max_fields: fields, max_matches: matches, max_scan_steps: steps };
        assert!(matches!(final_answer(&source, raw(&evidence, &[quote]), &evidence, AnswerOptions::default(),
            &mut budget, &mut Continue), Err(Int4SourceMapError::WorkLimit)));
    }
}
#[test]
fn repeated_subquotes_reach_only_selected_originals_and_charge_fanout_before_dedup() {
    let (source, value) = mapped(&[("éAlice éAlice", Some("éAlice")), (" Alice", None)]);
    let evidence = collect(&source, &value, limits(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    let mut budget = GroundingBudget::default(); let before = budget.max_matches;
    let answer = final_answer(&source, raw(&evidence, &["Alice"]), &evidence, AnswerOptions::default(), &mut budget, &mut Continue).unwrap();
    assert_eq!(before - budget.max_matches, 3);
    assert_eq!(answer.citations[0].spans.iter().map(|s| s.byte_start).collect::<Vec<_>>(), vec![2, 10]);
    assert_eq!(answer.citations[0].spans.iter().map(|s| s.scalar_start).collect::<Vec<_>>(), vec![1, 8]);
}
#[test]
fn overlapping_evidence_does_not_manufacture_additional_source_occurrences() {
    let (source, mut value) = mapped(&[("é Alice", Some("é Alice"))]);
    Arc::get_mut(&mut value.chunks[0]).unwrap().citations.push(SourceCitation { quote: "Alice".to_owned(),
        occurrence: SourceOccurrence::Anchored,
        spans: vec![VerifiedSourceSpan { byte_start: 3, byte_end: 8, scalar_start: 2, scalar_end: 7 }] });
    let evidence = collect(&source, &value, limits(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    let mut budget = GroundingBudget::default(); let before = budget.max_matches;
    let answer = final_answer(&source, raw(&evidence, &["Alice"]), &evidence, AnswerOptions::default(), &mut budget, &mut Continue).unwrap();
    assert_eq!(before - budget.max_matches, 4); assert_eq!(answer.citations[0].spans.len(), 1);
    assert_eq!(answer.citations[0].occurrence, SourceOccurrence::Anchored);
}
#[test]
fn duplicate_quotes_are_verified_before_deduplication_and_limits_never_top_k() {
    let (source, mut value) = mapped(&[("Alice", Some("Alice"))]);
    let chunk = Arc::get_mut(&mut value.chunks[0]).unwrap(); chunk.citations.push(chunk.citations[0].clone());
    assert_eq!(collect(&source, &value, limits(), &mut GroundingBudget::default(), &mut Continue).unwrap().passages.len(), 1);
    Arc::get_mut(&mut value.chunks[0]).unwrap().citations[1].spans[0].scalar_end += 1;
    assert!(collect(&source, &value, limits(), &mut GroundingBudget::default(), &mut Continue).is_err());
    let (source, value) = mapped(&[("Alice", Some("Alice")), ("Bob", Some("Bob"))]);
    for cap in [QuestionSynthesisLimits { max_evidence_passages: 1, ..limits() },
        QuestionSynthesisLimits { max_evidence_bytes: 7, ..limits() }] {
        assert!(matches!(collect(&source, &value, cap, &mut GroundingBudget::default(), &mut Continue), Err(Int4SourceMapError::WorkLimit)));
    }
}
#[test]
fn collection_refuses_missing_source_coverage_late_geometry_and_forged_whitespace_work() {
    for axis in 0..4 {
        let (source, mut value) = mapped(&[("Alice", Some("Alice")), ("\r\n", None)]);
        match axis { 0 => { value.chunks.pop(); }, 1 => Arc::get_mut(&mut value.chunks[1]).unwrap().source_span.scalar_start += 1,
            2 => Arc::get_mut(&mut value.chunks[1]).unwrap().model_work.forward_positions = 1,
            _ => Arc::get_mut(&mut value.chunks[1]).unwrap().status = QuestionChunkStatus::Abstained }
        assert!(collect(&source, &value, limits(), &mut GroundingBudget::default(), &mut Continue).is_err());
    }
}
#[test]
fn actual_final_abstention_cannot_hide_answer_text_or_citations() {
    let (source, evidence) = fixture(&["Alice"]); let mut result = raw(&evidence, &["Alice"]);
    result.status = AnswerStatus::Abstained; result.answerable = false; result.answer = None; result.citations.clear();
    assert_eq!(finish(&source, &evidence, result.clone()).unwrap().status, SynthesisStatus::Abstained);
    result.answer = Some("hidden answer".to_owned()); assert!(finish(&source, &evidence, result).is_err());
}
#[test]
fn foreign_profile_or_claims_cannot_be_promoted_by_valid_citation_bytes() {
    let (source, evidence) = fixture(&["Alice"]);
    let mut result = raw(&evidence, &["Alice"]); result.numerics_profile = "strict-quantized-v1".to_owned();
    assert!(finish(&source, &evidence, result).is_err());
    let mut result = raw(&evidence, &["Alice"]); result.untrusted_fields[0] = "trusted_answer".to_owned();
    assert!(finish(&source, &evidence, result).is_err());
}
#[test]
fn cancellation_at_every_final_lift_boundary_returns_no_partial_answer() {
    struct Count { calls: usize, stop: Option<usize> }
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.stop == Some(self.calls)).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let (source, evidence) = fixture(&["Bob", "Alice", "Carol"]); let mut count = Count { calls: 0, stop: None };
    final_answer(&source, raw(&evidence, &["Alice"]), &evidence, AnswerOptions::default(),
        &mut GroundingBudget::default(), &mut count).unwrap();
    for stop in 1..=count.calls {
        let error = final_answer(&source, raw(&evidence, &["Alice"]), &evidence, AnswerOptions::default(),
            &mut GroundingBudget::default(), &mut Count { calls: 0, stop: Some(stop) }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}
