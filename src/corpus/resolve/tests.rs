use super::*;
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn document(id: &str, surface: &str) -> ResolutionDocument {
    ResolutionDocument { id: id.to_owned(), text: surface.to_owned(), mentions: vec![MentionInput {
        entity_type: "PERSON".to_owned(), surface: surface.to_owned(), span: VerifiedSourceSpan {
            byte_start: 0, byte_end: surface.len(), scalar_start: 0, scalar_end: surface.chars().count() } }] }
}
fn plan(documents: &[ResolutionDocument]) -> ResolutionPlan<'_> {
    ResolutionPlan::prepare(documents, ResolveOptions::default(), ResolveLimits::default(), &mut Continue).unwrap()
}
fn probabilities(decision: PairDecision) -> PairLogProbabilities {
    match decision { PairDecision::Same => PairLogProbabilities { same: -0.1, different: -4.0, uncertain: -5.0 },
        PairDecision::Different => PairLogProbabilities { same: -4.0, different: -0.1, uncertain: -5.0 },
        PairDecision::Uncertain => PairLogProbabilities { same: -4.0, different: -5.0, uncertain: -0.1 } }
}
fn scores(decision: PairDecision) -> BidirectionalScores {
    BidirectionalScores { forward: probabilities(decision), reverse: probabilities(decision) }
}
fn finish(p: &ResolutionPlan<'_>, f: impl Fn(CandidatePair) -> PairDecision) -> ResolutionResult {
    let results = p.pairs().map(|ticket| { let decision = f(ticket.indices()); ticket.finish(scores(decision)) }).collect();
    p.finalize(results, &mut Continue).unwrap()
}
#[test]
fn arrival_and_completion_order_do_not_change_canonical_output() {
    let a = vec![document("z", "Alice Smith"), document("a", "Smith"), document("m", "A Smith")];
    let mut b = a.clone(); b.reverse();
    let p = plan(&a); let q = plan(&b);
    let expected = finish(&p, |_| PairDecision::Same);
    let mut completed: Vec<_> = q.pairs().map(|t| t.finish(scores(PairDecision::Same))).collect(); completed.reverse();
    let observed = q.finalize(completed, &mut Continue).unwrap();
    assert_eq!(canonjson::canonical_bytes(&expected).unwrap(), canonjson::canonical_bytes(&observed).unwrap());
    assert_eq!(observed.clusters[0].mentions, vec![0, 1, 2]);
}
#[test]
fn lexical_similarity_and_same_names_do_not_authorize_merges() {
    let docs = vec![document("a", "John Smith"), document("b", "John Smith")];
    let p = plan(&docs); assert_eq!(p.candidate_count(), 1);
    let result = finish(&p, |_| PairDecision::Uncertain); assert_eq!(result.clusters.len(), 2);
    assert_eq!(result.calibration, "uncalibrated");
}
#[test]
fn contradictory_triangle_cannot_become_one_entity() {
    let docs = vec![document("a", "Smith"), document("b", "Alice Smith"), document("c", "Bob Smith")];
    let p = plan(&docs);
    let result = finish(&p, |pair| if pair == (CandidatePair { left: 0, right: 2 }) { PairDecision::Different } else { PairDecision::Same });
    assert_eq!(result.clusters, vec![EntityCluster { id: 0, mentions: vec![0, 1] }, EntityCluster { id: 1, mentions: vec![2] }]);
    assert_eq!(result.blocked_merges, 1);
}
#[test]
fn missing_triangle_edge_also_blocks_transitive_guess() {
    let docs = vec![document("a", "Alpha"), document("b", "Alpha Beta"), document("c", "Beta")];
    let p = plan(&docs); assert_eq!(p.candidate_count(), 2);
    assert_eq!(finish(&p, |_| PairDecision::Same).clusters.len(), 2);
}
#[test]
fn order_disagreement_and_margin_ties_abstain() {
    let a = probabilities(PairDecision::Same); let b = probabilities(PairDecision::Different);
    assert_eq!(decide(BidirectionalScores { forward: a, reverse: b }, 1).unwrap().0, PairDecision::Uncertain);
    let tied = PairLogProbabilities { same: -3.0, different: -3.0, uncertain: -4.0 };
    assert_eq!(decide(BidirectionalScores { forward: tied, reverse: tied }, 1).unwrap().0, PairDecision::Uncertain);
    assert_eq!(decide(scores(PairDecision::Same), 3900).unwrap().0, PairDecision::Same);
    assert_eq!(decide(scores(PairDecision::Same), 3901).unwrap().0, PairDecision::Uncertain);
}
#[test]
fn nonfinite_positive_and_impossible_probability_mass_are_rejected() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.01] {
        let mut s = scores(PairDecision::Same); s.forward.same = invalid;
        assert_eq!(decide(s, 1000), Err(ResolveError::InvalidScores));
    }
    let p = PairLogProbabilities { same: -0.1, different: -0.1, uncertain: -0.1 };
    assert_eq!(classify_order(p, 1.0), Err(ResolveError::InvalidScores));
}
#[test]
fn full_snapshot_and_policy_bind_scores_even_when_names_match() {
    let a = vec![document("a", "Smith"), document("b", "Smith")];
    let mut b = a.clone(); b[0].text.push_str(" changed context");
    let p = plan(&a); let q = plan(&b);
    let foreign = p.pairs().map(|t| t.finish(scores(PairDecision::Same))).collect();
    assert!(matches!(q.finalize(foreign, &mut Continue), Err(ResolveError::ForeignScores)));
    let q = ResolutionPlan::prepare(&a, ResolveOptions { minimum_margin_milli: 2000, ..ResolveOptions::default() },
        ResolveLimits::default(), &mut Continue).unwrap();
    let foreign = p.pairs().map(|t| t.finish(scores(PairDecision::Same))).collect();
    assert!(matches!(q.finalize(foreign, &mut Continue), Err(ResolveError::ForeignScores)));
}
#[test]
fn every_candidate_is_required_exactly_once() {
    let docs = vec![document("a", "Smith"), document("b", "Smith"), document("c", "Smith")]; let p = plan(&docs);
    assert!(matches!(p.finalize(Vec::new(), &mut Continue), Err(ResolveError::IncompleteScores)));
    let mut duplicate: Vec<_> = p.pairs().map(|t| t.finish(scores(PairDecision::Same))).collect();
    duplicate[2] = p.pairs().next().unwrap().finish(scores(PairDecision::Same));
    assert!(matches!(p.finalize(duplicate, &mut Continue), Err(ResolveError::IncompleteScores)));
}
#[test]
fn unicode_anchors_are_exact_and_context_preserves_boundaries() {
    let mut d = document("zh", "上海"); d.text = "é 上海 tail".to_owned();
    d.mentions[0].span = VerifiedSourceSpan { byte_start: 3, byte_end: 9, scalar_start: 2, scalar_end: 4 };
    let docs = vec![d.clone()]; let p = plan(&docs); assert_eq!(p.mentions()[0].context(), "é 上海 tail");
    d.mentions[0].span.scalar_start = 3;
    assert!(matches!(ResolutionPlan::prepare(&[d], ResolveOptions::default(), ResolveLimits::default(), &mut Continue), Err(ResolveError::InvalidAnchor)));
    assert_ne!(lexical_keys("É", BlockingPolicy::AsciiWordOverlap, 32).unwrap(), lexical_keys("é", BlockingPolicy::AsciiWordOverlap, 32).unwrap());
}
#[test]
fn types_do_not_cross_blocks_and_acronyms_are_only_candidates() {
    let mut docs = vec![document("a", "International Business Machines"), document("b", "IBM")];
    assert_eq!(plan(&docs).candidate_count(), 1);
    docs[1].mentions[0].entity_type = "ORG".to_owned(); assert_eq!(plan(&docs).candidate_count(), 0);
    assert_ne!(lexical_keys("Alice", BlockingPolicy::ExactSurface, 1).unwrap(), lexical_keys("alice", BlockingPolicy::ExactSurface, 1).unwrap());
}
#[test]
fn overflow_limits_refuse_instead_of_pruning_pairs() {
    let docs = vec![document("a", "Alice Smith"), document("b", "Alice Smith"), document("c", "Alice Smith")];
    for axis in 0..6 {
        let mut limits = ResolveLimits::default();
        match axis { 0 => limits.max_mentions = 1, 1 => limits.max_block_members = 2,
            2 => limits.max_candidate_pairs = 2, 3 => limits.max_pair_visits = 3,
            4 => limits.max_scan_steps = 1, _ => limits.max_keys_per_mention = 1 }
        assert!(ResolutionPlan::prepare(&docs, ResolveOptions::default(), limits, &mut Continue).is_err());
    }
    let p = ResolutionPlan::prepare(&docs, ResolveOptions::default(), ResolveLimits { max_cluster_checks: 1, ..ResolveLimits::default() }, &mut Continue).unwrap();
    let records = p.pairs().map(|t| t.finish(scores(PairDecision::Same))).collect();
    assert!(matches!(p.finalize(records, &mut Continue), Err(ResolveError::ClusterBudget)));
}
#[test]
fn duplicate_documents_and_mentions_are_not_double_counted() {
    let d = document("a", "Smith");
    assert!(ResolutionPlan::prepare(&[d.clone(), d.clone()], ResolveOptions::default(), ResolveLimits::default(), &mut Continue).is_err());
    let mut d = d; d.mentions.push(d.mentions[0].clone());
    assert!(ResolutionPlan::prepare(&[d], ResolveOptions::default(), ResolveLimits::default(), &mut Continue).is_err());
}
#[test]
fn empty_corpus_singletons_and_output_privacy() {
    assert!(finish(&plan(&[]), |_| PairDecision::Same).clusters.is_empty());
    let docs = vec![document("a", "Alice"), document("b", "Bob")]; let p = plan(&docs);
    let result = finish(&p, |_| PairDecision::Same); assert_eq!(result.clusters.len(), 2);
    let json = canonjson::canonical_string(&result).unwrap();
    for absent in ["binding", "digest", "confidence\":", "context\":"] { assert!(!json.contains(absent)); }
    let size = canonjson::canonical_bytes(&result).unwrap().len();
    assert!(check_output(&result, size).is_ok()); assert_eq!(check_output(&result, size - 1), Err(ResolveError::OutputBudget));
}
#[test]
fn cancellation_never_returns_a_prefix_snapshot() {
    struct Cancel;
    impl DecodeStepControl for Cancel { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let docs = vec![document("a", "Alice")];
    assert!(matches!(ResolutionPlan::prepare(&docs, ResolveOptions::default(), ResolveLimits::default(), &mut Cancel), Err(ResolveError::Cancelled(DecodeCancellationKind::Deadline))));
    assert!(matches!(plan(&docs).finalize(vec![], &mut Cancel), Err(ResolveError::Cancelled(DecodeCancellationKind::Deadline))));
}
