//! Pinned preflight and private delivery-corruption fixtures, not neural success.
use super::*;
use crate::{candidate_cli::redact::tests::command,
    native_engine::decode::DecodeCancellationKind,
    tasks::redact::{actions, union::DetectedDocument, long::LongRedactionStage}};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "long-redaction-planning-fixture".to_owned(), source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn assets(command: &RedactCommand) -> (SourceTaskPlanner, ExecutionIdentity, LongRedactionConfig) {
    let (_, limits) = command.validate().unwrap();
    let (planner, _) = planner().unwrap();
    let identity = source_identity(&facts(), &planner, BuiltInTask::Ner).unwrap();
    let config = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
    (planner, identity, config)
}
#[test]
fn actual_pinned_long_preflight_is_deterministic_and_covers_original_unicode() {
    let cmd = command(&["--chunked", "--max-ner-chunk-bytes", "64", "--max-new-tokens", "64"]);
    let (planner, identity, config) = assets(&cmd);
    let source = "Alice é 上海\r\n".repeat(160);
    assert!(source.len() > cmd.host.context_tokens);
    let redactor = Int8DocumentRedactor::new(&planner, identity, config.clone()).unwrap();
    let a = redactor.preflight(&source, &mut Continue).unwrap();
    let b = redactor.preflight(&source, &mut Continue).unwrap();
    assert_eq!(a, b); assert!(a.chunks > 1); assert_eq!(a.source_bytes, source.len());
    assert_eq!(a.source_scalars, source.chars().count());
    assert_eq!(a.reserved_mask_node_visits, a.chunks as u64 * config.mapping.mask_visits_per_chunk);
    assert!(fits(a.planned_model_work, config.mapping.max_model_work));
}
#[test]
fn whole_native_axes_and_original_preparation_cancellation_refuse_before_weights() {
    struct Stop;
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
    let cmd = command(&["--chunked", "--max-ner-chunk-bytes", "6", "--max-new-tokens", "64"]);
    let (p, id, config) = assets(&cmd); let source = "éAéAéAéA";
    let redactor = Int8DocumentRedactor::new(&p, id.clone(), config.clone()).unwrap();
    let exact = redactor.preflight(source, &mut Continue).unwrap();
    assert_eq!(redactor.preflight(source, &mut Stop).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
    for axis in 0..6 {
        let mut c = config.clone(); c.mapping.max_model_work = exact.planned_model_work; c.mapping.max_mask_visits = exact.reserved_mask_node_visits;
        match axis { 0 => c.mapping.max_model_work.forward_positions -= 1, 1 => c.mapping.max_model_work.projected_logits -= 1,
            2 => c.mapping.max_model_work.attention_pairs -= 1, 3 => c.mapping.max_model_work.projections.dot_products -= 1,
            4 => c.mapping.max_model_work.projections.multiply_accumulates -= 1, _ => c.mapping.max_mask_visits -= 1 }
        assert!(Int8DocumentRedactor::new(&p, id.clone(), c).unwrap().preflight(source, &mut Continue).is_err(), "{axis}");
    }
}
// A local verifier fixture, not produced by the native host and never exported
// as execution evidence. Its text comes from genuine model-free rule editing.
fn receipt(verify: bool) -> (LongRedactionPreflight, LongRedactionRun, RedactionRequest, LongRedactionConfig) {
    let cmd = command(&["--chunked"]); let (_, limits) = cmd.validate().unwrap();
    let config = cmd.long.config(&cmd.host, limits, NerOptions::default()).unwrap();
    let mut request = cmd.request(); request.verify = verify;
    let doc = DetectedDocument::rules_only("***", &request.rules, request.rule_budget).unwrap();
    let mut result = actions::apply(&doc, &request.actions, None, request.edit_budget).unwrap();
    result.verification = if verify { VerificationStatus::CleanDeclaredUnion } else { VerificationStatus::NotRequested };
    let work = constrained_int8::planned_work(30, 2).unwrap();
    let expected = LongRedactionPreflight { source_bytes: 3, source_scalars: 3, chunks: 1,
        planned_model_work: work, reserved_mask_node_visits: config.mapping.mask_visits_per_chunk };
    let stage = || LongRedactionStage { preflight: expected, model_work: work, mask_node_visit_charge: 20 };
    let total = if verify { work.checked_add(work).unwrap() } else { work };
    let run = LongRedactionRun { schema_version: 1, execution: LONG_REDACTION_EXECUTION,
        numerics_profile: STRICT_INT8_PROFILE, detector_scope: "whole-source-rules-independent-ner-chunks-v1",
        result, original: stage(), verification: verify.then(stage), reserved_model_work: total, model_work: total,
        reserved_mask_node_visits: expected.reserved_mask_node_visits * (1 + u64::from(verify)),
        mask_node_visit_charge: if verify { 40 } else { 20 }, verification_scan_steps: 0,
        warnings: ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
            "clean_scan_and_pseudonyms_are_not_anonymization"] };
    (expected, run, request, config)
}
#[test]
fn completed_output_refuses_partial_source_substituted_profile_or_any_work_axis() {
    let (expected, run, request, config) = receipt(false); check_completed(expected, &run, &request, &config).unwrap();
    for axis in 0..10 {
        let (expected, mut run, request, config) = receipt(false);
        match axis { 0 => run.original.preflight.source_bytes -= 1, 1 => run.numerics_profile = "hf-bf16-eager",
            2 => run.model_work.forward_positions += 1, 3 => run.model_work.projected_logits += 1,
            4 => run.model_work.attention_pairs += 1, 5 => run.model_work.projections.dot_products += 1,
            6 => run.model_work.projections.multiply_accumulates += 1,
            7 => run.mask_node_visit_charge += 1, 8 => run.reserved_mask_node_visits += 1,
            _ => run.verification_scan_steps = request.grounding_budget.max_scan_steps + 1 }
        assert!(check_completed(expected, &run, &request, &config).is_err(), "{axis}");
    }
}
#[test]
fn verification_receipt_must_cover_actual_transformed_text_and_exact_stage_totals() {
    let (expected, run, request, config) = receipt(true); check_completed(expected, &run, &request, &config).unwrap();
    for axis in 0..5 {
        let (expected, mut run, request, config) = receipt(true);
        match axis { 0 => run.verification.as_mut().unwrap().preflight.source_bytes += 1,
            1 => run.verification.as_mut().unwrap().preflight.source_scalars += 1,
            2 => run.verification.as_mut().unwrap().preflight.chunks += 1,
            3 => run.result.verification = VerificationStatus::NotRequested, _ => run.verification = None }
        assert!(check_completed(expected, &run, &request, &config).is_err(), "{axis}");
    }
}
