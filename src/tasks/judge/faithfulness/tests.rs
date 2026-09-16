//! Source-only regression cases; no model-quality claims.
use super::*;
fn policy() -> FaithfulnessPolicy { FaithfulnessPolicy { minimum_candidate_weight_ppm: 500000,
    minimum_margin_milli: 100, evidence_window_bytes: 8, max_evidence_windows: 31, max_evidence_spans: 31 } }
fn assessment(relation: FaithfulnessRelation) -> FaithfulnessAssessment {
    FaithfulnessAssessment { top_relation: relation, candidate_conditional_weight: 0.9, log_score_margin: 3.0, passes_policy: true }
}
fn window(relation: FaithfulnessRelation) -> FaithfulnessWindow {
    FaithfulnessWindow { span: VerifiedSourceSpan { byte_start: 0, byte_end: 1, scalar_start: 0, scalar_end: 1 },
        score_head: 1, assessment: assessment(relation) }
}
#[test]
fn unicode_windows_cover_every_original_byte_and_verify_independently() {
    let source = "é 上海.\nA😀B\r\nLast";
    let spans = partition_evidence(source, policy()).unwrap();
    let mut rebuilt = String::new();
    for span in spans {
        assert!(span.byte_end - span.byte_start <= 8);
        let text = &source[span.byte_start..span.byte_end]; rebuilt.push_str(text);
        validate_source_span(source, text, SourceSpan::new(span.byte_start, span.byte_end,
            span.scalar_start, span.scalar_end)).unwrap();
    }
    assert_eq!(rebuilt, source);
}
#[test]
fn over_budget_tail_and_too_narrow_utf8_windows_refuse_instead_of_truncating() {
    let mut p = policy(); p.max_evidence_windows = 1; p.max_evidence_spans = 1;
    assert!(partition_evidence("123456789", p).is_err());
    p.evidence_window_bytes = 1; p.max_evidence_windows = 4;
    assert!(partition_evidence("😀", p).is_err());
}
#[test]
fn whole_source_support_without_window_support_abstains() {
    use FaithfulnessRelation::*;
    assert_eq!(decide(&assessment(Entailed), &[window(Unsupported)]), (None, Some(FaithfulnessAbstention::InsufficientEvidence)));
    assert_eq!(decide(&assessment(Contradicted), &[window(Unsupported)]), (None, Some(FaithfulnessAbstention::InsufficientEvidence)));
    assert_eq!(decide(&assessment(Unsupported), &[window(Unsupported)]), (Some(Unsupported), None));
}
#[test]
fn conflicts_are_not_hidden_by_selected_supporting_quotes() {
    use FaithfulnessRelation::*;
    for global in [Entailed, Contradicted, Unsupported] {
        assert_eq!(decide(&assessment(global), &[window(Entailed), window(Contradicted)]),
            (None, Some(FaithfulnessAbstention::ConflictingEvidence)));
    }
}
#[test]
fn uncertain_whole_source_cannot_be_overridden_by_a_confident_window() {
    let mut global = assessment(FaithfulnessRelation::Entailed); global.passes_policy = false;
    assert_eq!(decide(&global, &[window(FaithfulnessRelation::Entailed)]),
        (None, Some(FaithfulnessAbstention::AmbiguousDistribution)));
}
