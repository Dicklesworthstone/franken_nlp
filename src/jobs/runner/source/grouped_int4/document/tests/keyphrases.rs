//! UNRUN pinned-planning, replay-contract and refusal fixtures. Constructed
//! completion metadata below exercises validation only, never native inference.
use super::*;
use crate::{
    native_engine::strict_int4::STRICT_INT4_PROFILE,
    tasks::{
        corpus_keyphrases::{CorpusKeyphraseResult, CorpusKeyphraseWarning, CorpusPhrase,
            ChunkPhraseEvidence, CORPUS_KEYPHRASE_POLICY},
        extract::ExtractionGrounding,
        ir::ScoreSpace,
        keyphrases::KEYPHRASES_TASK_VERSION,
        source_planning::grouped_int4::{Int4SourceError,
            long::{SourceMapVerification, keyphrases::INT4_CORPUS_KEYPHRASE_SEMANTICS}},
    },
    validation::grounded_fields::VerifiedSourceSpan,
};

fn keyphrase_task() -> Int4SourceDocumentTask {
    let keyphrases = Int4KeyphraseLimits::default();
    Int4SourceDocumentTask::Keyphrases { options: KeyphraseOptions::default(),
        max_lifetime_scan_work: keyphrases.aggregation.max_scan_work * limits().max_attempts + 4096,
        limits: keyphrases }
}
fn ranked() -> Int4SourceDocumentJobPlanner {
    build(keyphrase_task(), mapping(), limits(), lifetime()).unwrap()
}

#[test]
fn global_ranking_prepares_every_real_chunk_without_extra_native_work_or_retained_tokens() {
    let ranked = ranked();
    let raw = build(Int4SourceDocumentTask::Map(SourceMapTask::Keyphrases(KeyphraseOptions::default())),
        mapping(), limits(), lifetime()).unwrap();
    let prepared = prepare(&ranked, SOURCE); let independent = prepare(&raw, SOURCE);
    assert!(matches!(prepared.expected, planning::Expected::Keyphrases(_)));
    assert_eq!(prepared.chunk_count(), 2);
    assert_eq!(prepared.identities, independent.identities);
    assert_eq!(prepared.work, independent.work);
    assert_eq!(prepared.work.mask_node_visits, 2000);
    assert_eq!(ranked.output_token_ceiling().unwrap(), 0);
    assert_eq!(raw.output_token_ceiling().unwrap(), 8 * 64);
    ranked.plan(SOURCE, &mut Continue).unwrap().compare(&prepared, &mut Continue).unwrap();
    assert!(freeze(&ranked).binding.compare(&freeze(&raw).binding).is_err());
}

#[test]
fn ranked_mode_cannot_cross_raw_map_factory_source_or_final_chunk_seals() {
    let ranked = ranked();
    let raw = build(Int4SourceDocumentTask::Map(SourceMapTask::Keyphrases(KeyphraseOptions::default())),
        mapping(), limits(), lifetime()).unwrap();
    let prepared = prepare(&ranked, SOURCE);
    assert!(raw.owns(&prepared).is_err());
    assert!(raw.plan(SOURCE, &mut Continue).unwrap().compare(&prepared, &mut Continue).is_err());
    let mut changed = prepare(&ranked, SOURCE);
    changed.identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"changed final keyphrase chunk");
    assert!(ranked.plan(SOURCE, &mut Continue).unwrap().compare(&changed, &mut Continue).is_err());
    assert!(ranked.plan("different source", &mut Continue).unwrap().compare(&prepared, &mut Continue).is_err());
}

#[test]
fn every_global_policy_option_cap_and_lifetime_scan_allowance_is_keyed() {
    let baseline = freeze(&ranked());
    for axis in 0..9 {
        let mut task = keyphrase_task();
        let Int4SourceDocumentTask::Keyphrases { options, limits, max_lifetime_scan_work } = &mut task else { unreachable!() };
        match axis {
            0 => options.max_phrases += 1,
            1 => options.max_phrase_scalars += 1,
            2 => limits.aggregation.max_unique_phrases += 1,
            3 => limits.aggregation.max_evidence_spans += 1,
            4 => limits.aggregation.max_value_bytes -= 1,
            5 => limits.aggregation.max_scan_work += 1,
            6 => limits.max_phrases += 1,
            7 => limits.max_result_bytes -= 1,
            _ => *max_lifetime_scan_work += 1,
        }
        let changed = build(task, mapping(), super::limits(), lifetime()).unwrap();
        assert!(baseline.binding.compare(&freeze(&changed).binding).is_err(), "axis {axis}");
    }
    let projection = serde_json::to_value(ranked().job_recipe()).unwrap();
    assert_eq!(projection["execution"], INT4_CORPUS_KEYPHRASE_EXECUTION);
    assert_eq!(projection["task"]["mode"], "keyphrases");
    assert_eq!(projection["task"]["limits"]["policy"], CORPUS_KEYPHRASE_POLICY);
    assert!(projection["task"]["max_lifetime_scan_work"].as_u64().is_some());
}

#[test]
fn existing_raw_and_summary_recipes_keep_their_original_lifetime_shape() {
    for task in [Int4SourceDocumentTask::Map(SourceMapTask::Keyphrases(KeyphraseOptions::default())), task(1)] {
        let factory = build(task, mapping(), limits(), lifetime()).unwrap();
        let recipe = serde_json::to_value(factory.job_recipe()).unwrap();
        assert_eq!(recipe["lifetime"].as_object().unwrap().len(), 2);
        assert_eq!(recipe["lifetime"]["summary_scan_steps"], lifetime().summary_scan_steps);
        assert!(recipe["task"].get("max_lifetime_scan_work").is_none());
        assert!(recipe["lifetime"].get("keyphrase_scan_work").is_none());
    }
}

#[test]
fn reducer_scan_authority_covers_all_attempts_independently_of_source_and_summary_budgets() {
    let mut task = keyphrase_task();
    let Int4SourceDocumentTask::Keyphrases { limits: keyphrases, max_lifetime_scan_work, .. } = &mut task else { unreachable!() };
    *max_lifetime_scan_work = keyphrases.aggregation.max_scan_work * limits().max_attempts;
    let mut source_lifetime = lifetime(); source_lifetime.summary_scan_steps = 0;
    assert!(build(task, mapping(), limits(), source_lifetime).is_ok());
    for axis in 0..3 {
        let mut task = keyphrase_task();
        let Int4SourceDocumentTask::Keyphrases { limits: keyphrases, max_lifetime_scan_work, .. } = &mut task else { unreachable!() };
        match axis {
            0 => *max_lifetime_scan_work = keyphrases.aggregation.max_scan_work * limits().max_attempts - 1,
            1 => *max_lifetime_scan_work = 0,
            _ => { keyphrases.aggregation.max_scan_work = u64::MAX; *max_lifetime_scan_work = u64::MAX; }
        }
        assert!(build(task, mapping(), limits(), lifetime()).is_err(), "axis {axis}");
    }
    for axis in 0..3 {
        let mut cap = lifetime();
        match axis { 0 => cap.verification.max_fields -= 1, 1 => cap.verification.max_matches -= 1,
            _ => cap.verification.max_scan_steps -= 1 }
        assert!(build(keyphrase_task(), mapping(), limits(), cap).is_err(), "source axis {axis}");
    }
    let factory = ranked(); let mut changed = limits(); changed.max_attempts += 1;
    assert!(factory.check_job_limits(changed).is_err());
}

#[test]
fn final_ranked_result_can_have_a_smaller_cap_than_native_chunk_and_map_values() {
    let mut job = limits(); job.max_result_bytes = 32_768;
    let mut task = keyphrase_task();
    let Int4SourceDocumentTask::Keyphrases { limits, .. } = &mut task else { unreachable!() };
    limits.max_result_bytes = job.max_result_bytes;
    let factory = build(task, mapping(), job, lifetime()).unwrap();
    assert_eq!(factory.max_result_bytes(), 32_768);
    assert_eq!(factory.output_token_ceiling().unwrap(), 0);
    assert!(factory.task_budget().max_output_bytes > factory.max_result_bytes());
    assert!(factory.document_limits().reduction.max_result_bytes as u64 > factory.max_result_bytes());
    assert!(build(keyphrase_task(), mapping(), job, lifetime()).is_err());
}

#[test]
fn invalid_union_top_k_and_native_options_refuse_before_complete_document_preparation() {
    for axis in 0..8 {
        let mut task = keyphrase_task();
        let Int4SourceDocumentTask::Keyphrases { options, limits: keyphrases, .. } = &mut task else { unreachable!() };
        match axis {
            0 => options.max_phrases = 0,
            1 => options.max_phrase_scalars = 0,
            2 => keyphrases.max_phrases = 0,
            3 => keyphrases.aggregation.max_unique_phrases = 0,
            4 => keyphrases.aggregation.max_evidence_spans = 0,
            5 => keyphrases.aggregation.max_value_bytes = mapping().reduction.max_value_bytes + 1,
            6 => keyphrases.aggregation.max_scan_work = 0,
            _ => keyphrases.max_result_bytes = mapping().reduction.max_result_bytes + 1,
        }
        assert!(build(task, mapping(), super::limits(), lifetime()).is_err(), "axis {axis}");
    }
}

// These public metadata values are only inputs to negative validator tests.
// They are not produced by a mock native backend or claimed as model output.
fn receipt_fixture(factory: &Int4SourceDocumentJobPlanner) -> (planning::Expected, Int4CorpusKeyphraseRun) {
    let prepared = prepare(factory, SOURCE);
    let planning::Expected::Keyphrases(expected) = prepared.expected else { unreachable!() };
    let Int4SourceDocumentTask::Keyphrases { limits, .. } = factory.task() else { unreachable!() };
    let work = expected.planned_work();
    let result = CorpusKeyphraseResult { schema_version: 1, task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(),
        ranking_policy: CORPUS_KEYPHRASE_POLICY.to_owned(), score_space: ScoreSpace::NotComputed,
        grounding: ExtractionGrounding::SourceMembership, mapped_chunks: expected.chunk_count(),
        omitted_candidates: 0, phrases: Vec::new(), forward_positions: work.forward_positions,
        projected_logits: work.projected_logits, mask_node_visit_charge: 0,
        warnings: [CorpusKeyphraseWarning::ChunkBoundariesMaySplitPhrases,
            CorpusKeyphraseWarning::SingleContextEquivalenceNotEstablished,
            CorpusKeyphraseWarning::ModelSelectionIsNotRecallGuarantee] };
    let run = Int4CorpusKeyphraseRun { schema_version: 1, execution: INT4_CORPUS_KEYPHRASE_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: WIDTH, chunk_profile: CHUNK_PROFILE,
        semantics: INT4_CORPUS_KEYPHRASE_SEMANTICS, source_span: expected.source_span(),
        map_batches: expected.chunk_count(), reduce_calls: 1, reduction_levels: 1,
        requested_max_phrases: limits.max_phrases, planned_model_work: work, model_work: work,
        reserved_mask_node_visits: expected.reserved_mask_visits(), mask_node_visit_charge: 0,
        verification_used: SourceMapVerification { fields: 0, matches: 0, scan_steps: 0 },
        verification_scan_work: 0, keyphrases: result };
    (prepared.expected, run)
}

#[test]
fn durable_completion_rejects_wrong_native_receipt_requested_rank_or_reducer_scan_charge() {
    let factory = ranked();
    let (expected, run) = receipt_fixture(&factory);
    expected.verify(&Int4SourceDocumentRun::Keyphrases(run), factory.task()).unwrap();
    for axis in 0..8 {
        let (expected, mut run) = receipt_fixture(&factory);
        match axis {
            0 => run.execution = "wrong-execution",
            1 => run.group_width += 1,
            2 => run.model_work.projections.dot_products += 1,
            3 => run.reserved_mask_node_visits -= 1,
            4 => run.source_span.byte_end -= 1,
            5 => run.keyphrases.mapped_chunks -= 1,
            6 => run.requested_max_phrases -= 1,
            _ => run.verification_scan_work = Int4KeyphraseLimits::default().aggregation.max_scan_work + 1,
        }
        assert!(expected.verify(&Int4SourceDocumentRun::Keyphrases(run), factory.task()).is_err(), "axis {axis}");
    }
    let (expected, run) = receipt_fixture(&factory);
    assert!(expected.verify(&Int4SourceDocumentRun::Keyphrases(run), &task(0)).is_err());
}

#[test]
fn retained_and_omitted_candidates_and_all_selected_evidence_fit_the_fixed_union_caps() {
    for evidence_overflow in [false, true] {
        let mut task = keyphrase_task();
        let Int4SourceDocumentTask::Keyphrases { limits, .. } = &mut task else { unreachable!() };
        limits.max_phrases = 1; limits.aggregation.max_unique_phrases = 2;
        limits.aggregation.max_evidence_spans = 1;
        let factory = build(task, mapping(), super::limits(), lifetime()).unwrap();
        let (expected, mut run) = receipt_fixture(&factory);
        let span = VerifiedSourceSpan { byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 };
        run.keyphrases.phrases.push(CorpusPhrase { text: "é".to_owned(), rank_sum: 1,
            evidence: vec![ChunkPhraseEvidence { chunk_id: 0, local_rank: 1,
                spans: if evidence_overflow { vec![span, span] } else { vec![span] } }] });
        run.verification_used.fields = 3; run.verification_used.matches = 2;
        run.keyphrases.omitted_candidates = if evidence_overflow { 0 } else { 2 };
        assert!(expected.verify(&Int4SourceDocumentRun::Keyphrases(run), factory.task()).is_err());
    }
}

#[test]
fn complete_result_byte_cap_is_checked_before_durable_commit() {
    let mut task = keyphrase_task();
    let Int4SourceDocumentTask::Keyphrases { limits, .. } = &mut task else { unreachable!() };
    limits.max_result_bytes = 1;
    let factory = build(task, mapping(), super::limits(), lifetime()).unwrap();
    let (expected, run) = receipt_fixture(&factory);
    let error = expected.verify(&Int4SourceDocumentRun::Keyphrases(run), factory.task()).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
}

#[test]
fn typed_native_cancellation_survives_the_durable_keyphrase_error_boundary() {
    let failure = keyphrase_failure(Int4CorpusKeyphraseError::Source(
        Int4SourceError::Cancelled(DecodeCancellationKind::Deadline)));
    assert!(failure.stop); assert_eq!(failure.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(keyphrase_failure(Int4CorpusKeyphraseError::InvalidLimits).fault.code, BatchCode::InvalidLimits);
}
