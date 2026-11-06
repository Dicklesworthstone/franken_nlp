//! Exercise the reducer with original judge finalizers and deterministic
//! synthetic logits. These fixtures make no neural-quality claim.
use super::*;
use crate::{execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::lmhead::scoring::ProjectionRows,
    tasks::{extract::SourceDocumentEncoder, ir::{PlanContext, PromptSegment, TaskBudget},
        judge::{JudgePlanner, JudgeRequest, JudgeResult, JudgeLimits, JudgeLogits}}, tokenizer::pinned_controls};
use FaithfulnessRelation::{Contradicted, Entailed, Unsupported};

struct Go;
impl DecodeStepControl for Go { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn policy() -> FaithfulnessPolicy { FaithfulnessPolicy { minimum_candidate_weight_ppm: 500000,
    minimum_margin_milli: 100, evidence_window_bytes: 128, max_evidence_windows: 31, max_evidence_spans: 31 } }
struct Logits { labels: Vec<Option<FaithfulnessRelation>>, tokens: [u32; 3] }
impl JudgeLogits for Logits {
    type Error = &'static str;
    fn project(&mut self, head: usize, _: &[PromptSegment], prefix: &[u32], rows: ProjectionRows<'_>)
        -> Result<Vec<f32>, Self::Error> {
        let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("full denominator required"); };
        let mut values = vec![0.0; vocabulary_size];
        if prefix.is_empty() {
            if let Some(relation) = self.labels[head] {
                let index = match relation { Contradicted => 0, Entailed => 1, Unsupported => 2 };
                values[self.tokens[index] as usize] = 12.0;
            }
        }
        Ok(values)
    }
}
fn local(source: &str, labels: &[Option<FaithfulnessRelation>], policy: FaithfulnessPolicy) -> FaithfulnessResult {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == "<|im_end|>").unwrap().id;
    let planner = JudgePlanner::pinned(controls, eos).unwrap();
    let digest = Sha256Digest::of_bytes(b"synthetic-aggregation-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: digest, artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(),
        packing_set_digest: digest, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "judge-v1".to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "none".to_owned(), schema_digest: digest, numerics_profile: NumericsProfile::HfBf16Eager,
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None };
    let budget = TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 1_000_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
    let request = JudgeRequest::Faithfulness { source: source.to_owned(), claim: "A fixed claim".to_owned(), policy, budget };
    let prepared = planner.plan(&request, &PlanContext::new(&identity, budget).unwrap(), JudgeLimits::default()).unwrap();
    let encoder = SourceDocumentEncoder::pinned(controls).unwrap();
    let tokens = ["C", "E", "U"].map(|text| {
        let encoded = encoder.encode(text, 8, 8).unwrap();
        assert_eq!(encoded.token_ids().len(), 1); encoded.token_ids()[0]
    });
    let mut model = Logits { labels: labels.to_vec(), tokens };
    let JudgeResult::Faithfulness(result) = prepared.execute(prepared.execution_identity(), &mut model).unwrap()
        else { panic!("faithfulness finalizer required"); };
    result
}
fn spans(parts: &[&str]) -> Vec<VerifiedSourceSpan> {
    let mut byte = 0; let mut scalar = 0;
    parts.iter().map(|part| {
        let span = VerifiedSourceSpan { byte_start: byte, byte_end: byte + part.len(),
            scalar_start: scalar, scalar_end: scalar + part.chars().count() };
        byte = span.byte_end; scalar = span.scalar_end; span
    }).collect()
}
fn aggregate(parts: &[&str], results: &[FaithfulnessResult], floor: u32)
    -> Result<FaithfulnessAggregate, FaithfulnessAggregationError> {
    let spans = spans(parts);
    let source = parts.concat();
    aggregate_with_control(results.iter().enumerate().map(|(chunk_id, result)| FaithfulnessChunkView {
        chunk_id, span: spans[chunk_id], result }), parts.len(), &source, results[0].policy,
        FaithfulnessAggregationPolicy { minimum_classified_coverage_ppm: floor }, &mut Go)
}

#[test]
fn complete_unicode_chunks_lift_original_evidence_and_keep_exact_native_scores() {
    let parts = ["é one\r\n", "上海.\n", "尾😀"];
    let results = [local(parts[0], &[Some(Unsupported)], policy()), local(parts[1], &[Some(Entailed)], policy()),
        local(parts[2], &[Some(Unsupported)], policy())];
    let before = serde_json::to_vec(&results).unwrap();
    let run = aggregate(&parts, &results, SCALE).unwrap();
    assert_eq!(run.relation, Some(Entailed)); assert_eq!(run.abstention, None);
    assert_eq!(run.classified_coverage, 1.0); assert_eq!(run.evidence.len(), 1);
    let evidence = &run.evidence[0];
    assert_eq!(evidence.chunk_id, 1); assert_eq!(evidence.score_head, 0);
    assert_eq!(evidence.span, spans(&parts)[1]); assert_eq!(evidence.quote, parts[1]);
    let source = parts.concat();
    crate::validation::validate_source_span(&source, &evidence.quote, crate::validation::SourceSpan::new(
        evidence.span.byte_start, evidence.span.byte_end, evidence.span.scalar_start, evidence.span.scalar_end)).unwrap();
    assert_eq!(serde_json::to_vec(&results).unwrap(), before);
    run.verify_metadata(FaithfulnessAggregationPolicy::default(), policy(), 3, source.len(), source.chars().count()).unwrap();
}

#[test]
fn contrary_witnesses_are_retained_and_cannot_be_hidden_by_relaxed_coverage() {
    let parts = ["support", "😀 contradiction"];
    let results = [local(parts[0], &[Some(Entailed)], policy()), local(parts[1], &[Some(Contradicted)], policy())];
    for floor in [0, 500000, SCALE] {
        let run = aggregate(&parts, &results, floor).unwrap();
        assert_eq!(run.relation, None);
        assert_eq!(run.abstention, Some(FaithfulnessAggregateAbstention::ConflictingEvidence));
        assert_eq!(run.evidence.len(), 2);
        assert_eq!(run.evidence[0].relation, Entailed); assert_eq!(run.evidence[1].relation, Contradicted);
        assert_eq!(run.evidence[1].span, spans(&parts)[1]);
    }
}

#[test]
fn coverage_uses_original_scalars_and_exact_integer_thresholds() {
    let parts = ["é", "好x", "?"];
    let results = [local(parts[0], &[Some(Entailed)], policy()), local(parts[1], &[Some(Unsupported)], policy()),
        local(parts[2], &[None], policy())];
    let accepted = aggregate(&parts, &results, 750000).unwrap();
    assert_eq!(accepted.classified_scalars, 3); assert_eq!(accepted.source_scalars, 4);
    assert_eq!(accepted.classified_coverage, 0.75); assert_eq!(accepted.relation, Some(Entailed));
    for floor in [750001, SCALE] {
        assert_eq!(aggregate(&parts, &results, floor).unwrap().abstention,
            Some(FaithfulnessAggregateAbstention::InsufficientClassifiedCoverage));
    }
    let uncertain = [local("?", &[None], policy())];
    assert_eq!(aggregate(&["?"], &uncertain, 0).unwrap().abstention,
        Some(FaithfulnessAggregateAbstention::InsufficientClassifiedCoverage));
}

#[test]
fn local_conflict_keeps_every_decisive_window_even_when_original_evidence_is_empty() {
    let policy = FaithfulnessPolicy { evidence_window_bytes: 4, ..policy() };
    let result = local("abcdefgh", &[Some(Entailed), Some(Entailed), Some(Contradicted)], policy);
    assert!(result.evidence.is_empty()); assert_eq!(result.abstention, Some(FaithfulnessAbstention::ConflictingEvidence));
    let run = aggregate(&["abcdefgh"], &[result], 0).unwrap();
    assert_eq!(run.abstention, Some(FaithfulnessAggregateAbstention::ConflictingEvidence));
    assert_eq!(run.evidence.iter().map(|e| e.quote.as_str()).collect::<Vec<_>>(), ["abcd", "efgh"]);
    assert_eq!(run.evidence.iter().map(|e| e.score_head).collect::<Vec<_>>(), [1, 2]);
}

#[test]
fn uncertain_chunk_support_does_not_become_an_unsupported_document_verdict() {
    let policy = FaithfulnessPolicy { evidence_window_bytes: 4, ..policy() };
    let results = [local("abcdefgh", &[None, Some(Entailed), Some(Unsupported)], policy),
        local("ijkl", &[Some(Unsupported)], policy)];
    let run = aggregate(&["abcdefgh", "ijkl"], &results, 0).unwrap();
    assert_eq!(run.relation, None); assert_eq!(run.abstention, Some(FaithfulnessAggregateAbstention::UnresolvedEvidence));
    assert_eq!(run.evidence.len(), 1); assert_eq!(run.evidence[0].quote, "abcd");
    let all_unsupported = [local("abcdefgh", &[Some(Unsupported); 3], policy)];
    assert_eq!(aggregate(&["abcdefgh"], &all_unsupported, SCALE).unwrap().relation, Some(Unsupported));
}

#[test]
fn gaps_overlaps_tail_loss_wrong_order_and_false_unicode_coordinates_refuse() {
    let parts = ["é", "上海"];
    let results = [local(parts[0], &[Some(Entailed)], policy()), local(parts[1], &[Some(Unsupported)], policy())];
    for fault in 0..6 {
        let mut spans = spans(&parts);
        match fault { 0 => spans[1].byte_start -= 1, 1 => spans[1].byte_start += 1,
            2 => spans[1].byte_end -= 3, 3 => spans[1].scalar_start += 1,
            4 => spans[0].byte_end = 1, _ => spans[1].scalar_end += 1 }
        assert!(aggregate_with_control(results.iter().enumerate().map(|(chunk_id, result)|
            FaithfulnessChunkView { chunk_id, span: spans[chunk_id], result }), 2, &parts.concat(), policy(),
            FaithfulnessAggregationPolicy::default(), &mut Go).is_err());
    }
    let spans = spans(&parts);
    assert!(aggregate_with_control(results.iter().enumerate().map(|(i, result)|
        FaithfulnessChunkView { chunk_id: 1 - i, span: spans[i], result }), 2, &parts.concat(), policy(),
        FaithfulnessAggregationPolicy::default(), &mut Go).is_err());
    assert!(aggregate_with_control([FaithfulnessChunkView { chunk_id: 0, span: spans[0], result: &results[0] }],
        2, &parts.concat(), policy(), FaithfulnessAggregationPolicy::default(), &mut Go).is_err());
}

#[test]
fn changed_native_scores_policy_decision_work_and_quotes_refuse() {
    let original = local("exact source", &[Some(Entailed)], policy());
    for fault in 0..10 {
        let mut changed = original.clone();
        match fault {
            0 => changed.heads[0].candidates[0].candidate_weight = 1.0,
            1 => changed.heads[0].eos_rule = "unscored".to_owned(),
            2 => changed.heads[0].candidates.swap(0, 1),
            3 => changed.global.passes_policy = false,
            4 => changed.windows[0].span.scalar_end += 1,
            5 => changed.evidence[0].quote = "other source".to_owned(),
            6 => changed.relation = Some(Unsupported),
            7 => changed.work.projected_logits += 1,
            8 => changed.calibration = "certified".to_owned(),
            _ => changed.evidence.push(changed.evidence[0].clone()),
        }
        assert!(aggregate(&["exact source"], &[changed], SCALE).is_err(), "fault {fault}");
    }
    assert!(aggregate(&["exact source"], &[original], SCALE + 1).is_err());
}

#[test]
fn positive_full_vocabulary_logprob_receipts_refuse_even_with_consistent_softmax() {
    let result = local("exact source", &[Some(Entailed)], policy());
    let mut scores = result.heads[0].clone();
    for candidate in &mut scores.candidates { candidate.sequence_score += 1000.0; }
    let maximum = scores.candidates.iter().map(|candidate| candidate.sequence_score).fold(f64::NEG_INFINITY, f64::max);
    let mass: f64 = scores.candidates.iter().map(|candidate| (candidate.sequence_score - maximum).exp()).sum();
    for candidate in &mut scores.candidates {
        candidate.candidate_weight = (candidate.sequence_score - maximum).exp() / mass;
    }
    assert!(validate_scores(&scores, &mut None).is_err());
}

#[test]
fn cancellation_before_and_during_evidence_keeps_its_exact_cause() {
    struct Stop { calls: usize, at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.at).then_some(self.cause)
    } }
    let result = local("exact source", &[Some(Entailed)], policy());
    let span = spans(&["exact source"])[0];
    for (at, cause) in [(1, DecodeCancellationKind::Deadline), (4, DecodeCancellationKind::Shutdown),
        (5, DecodeCancellationKind::User)] {
        let mut control = Stop { calls: 0, at, cause };
        assert_eq!(aggregate_with_control([FaithfulnessChunkView { chunk_id: 0, span, result: &result }],
            1, "exact source", policy(), FaithfulnessAggregationPolicy::default(), &mut control).unwrap_err(),
            FaithfulnessAggregationError::Cancelled(cause));
        assert_eq!(control.calls, at);
    }
}
