//! Receipt/publication fixtures only: rules are real, native receipts are synthetic.
//! These tests do not establish native execution, NER recall or model parity.
use super::*;
use super::super::tests::{command, options};
use crate::{native_engine::constrained_int4,
    tasks::{redact::redact_rules, source_planning::grouped_int4::{Int4SourceError, long::Int4SourceMapError}}};

fn fixture(verify: bool, include_map: bool)
    -> (Int4LongRedactionPreflight, Int4LongRedactionRun, RedactionRequest, Int4LongRedactionConfig) {
    let mut flags = vec!["--rules", "email", "--action", "placeholder"];
    if !verify { flags.push("--no-verify"); }
    if include_map { flags.push("--include-map"); }
    let command = command(&flags); let options = options(&command);
    let limits = options.validate(&command.args).unwrap();
    let request = options.request(&command.args, limits);
    let detector = options.detector(&command.args, limits, NerOptions::default());
    let source = "é a@example.org 上海";
    let result = redact_rules(source, &request, None).unwrap();
    let per_chunk = constrained_int4::planned_work(128, 7, 8).unwrap();
    let work = per_chunk.checked_add(per_chunk).unwrap();
    let masks = detector.mapping.mask_visits_per_chunk * 2;
    let map_used = SourceMapVerification { fields: 1, matches: 1, scan_steps: 11 };
    let detection_used = SourceMapVerification { fields: 1, matches: 1, scan_steps: 13 };
    let expected = Int4LongRedactionPreflight { source_bytes: source.len(), source_scalars: source.chars().count(),
        chunks: 2, group_width: 128, planned_model_work: work, reserved_mask_node_visits: masks };
    let original = Int4LongRedactionStage { preflight: expected, model_work: work, mask_node_visit_charge: masks,
        map_verification_used: map_used, detection_verification_used: detection_used };
    let verification = verify.then(|| Int4LongRedactionStage {
        preflight: Int4LongRedactionPreflight { source_bytes: result.text().len(),
            source_scalars: result.text().chars().count(), ..expected },
        model_work: work, mask_node_visit_charge: masks,
        map_verification_used: map_used, detection_verification_used: detection_used,
    });
    let stages = 1 + u64::from(verify);
    let total_work = if verify { work.checked_add(work).unwrap() } else { work };
    let run = Int4LongRedactionRun { schema_version: 1, execution: INT4_LONG_REDACTION_EXECUTION,
        numerics_profile: STRICT_INT4_PROFILE, group_width: 128,
        detector_scope: "whole-source-rules-independent-ner-chunks-v1", result, original, verification,
        reserved_model_work: total_work, model_work: total_work,
        reserved_mask_node_visits: masks * stages, mask_node_visit_charge: masks * stages,
        map_verification_used: SourceMapVerification { fields: stages as usize, matches: stages as usize, scan_steps: 11 * stages },
        detection_verification_used: SourceMapVerification { fields: stages as usize, matches: stages as usize, scan_steps: 13 * stages },
        verification_scan_steps: 13 * stages,
        warnings: ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
            "clean_scan_and_pseudonyms_are_not_anonymization"] };
    (expected, run, request, detector)
}

#[test]
fn complete_receipt_requires_selected_verification_and_actual_transformed_geometry() {
    for verify in [false, true] {
        let (expected, run, request, detector) = fixture(verify, false);
        assert!(check_completed(expected, &run, &request, &detector).is_ok());
        assert!(run.result.model_types().is_empty(), "rules-only fixture is not native NER success");
        if verify {
            let second = &run.verification.as_ref().unwrap().preflight;
            assert_ne!(second.source_bytes, expected.source_bytes, "placeholder expansion must be reflected");
            assert_eq!(second.source_bytes, run.result.text().len());
            assert_eq!(second.source_scalars, run.result.text().chars().count());
        }
        let mut changed = request.clone(); changed.verify = !verify;
        assert_eq!(check_completed(expected, &run, &changed, &detector), Err(CandidateError::Execution));
    }
}

#[test]
fn forged_stage_aggregate_group_or_geometry_never_passes_complete_receipt_checks() {
    for fault in 0..18 {
        let (expected, mut run, request, detector) = fixture(true, false);
        match fault {
            0 => run.reserved_model_work.forward_positions += 1,
            1 => run.reserved_model_work.projected_logits += 1,
            2 => run.reserved_model_work.attention_pairs += 1,
            3 => run.reserved_model_work.projections.dot_products += 1,
            4 => run.reserved_model_work.projections.multiply_accumulates += 1,
            5 => run.original.model_work.projections.dot_products += 1,
            6 => run.original.preflight.reserved_mask_node_visits += 1,
            7 => run.mask_node_visit_charge -= 1,
            8 => run.verification.as_mut().unwrap().preflight.source_bytes = expected.source_bytes,
            9 => run.verification.as_mut().unwrap().preflight.source_scalars += 1,
            10 => run.verification.as_mut().unwrap().preflight.group_width = 32,
            11 => run.group_width = 32,
            12 => run.verification = None,
            13 => run.original.preflight.chunks = 0,
            14 => run.map_verification_used.matches += 1,
            15 => run.detection_verification_used.scan_steps += 1,
            16 => run.verification_scan_steps += 1,
            _ => run.detector_scope = "rules-only",
        }
        assert_eq!(check_completed(expected, &run, &request, &detector), Err(CandidateError::Execution), "fault {fault}");
    }
    let (expected, run, mut request, detector) = fixture(true, true);
    request.actions.include_map = false;
    assert_eq!(check_completed(expected, &run, &request, &detector), Err(CandidateError::Execution));
}

#[test]
fn five_native_caps_and_each_grounding_pool_apply_to_the_complete_two_stage_operation() {
    for axis in 0..5 {
        let (expected, run, request, mut detector) = fixture(true, false);
        let cap = &mut detector.mapping.max_model_work;
        *cap = run.reserved_model_work;
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        assert_eq!(check_completed(expected, &run, &request, &detector), Err(CandidateError::Execution));
    }
    for pool in 0..2 {
        for axis in 0..3 {
            let (expected, run, mut request, mut detector) = fixture(true, false);
            let used = if pool == 0 { run.map_verification_used } else { run.detection_verification_used };
            let cap = if pool == 0 { &mut detector.mapping.verification } else { &mut request.grounding_budget };
            match axis { 0 => cap.max_fields = used.fields - 1, 1 => cap.max_matches = used.matches - 1,
                _ => cap.max_scan_steps = used.scan_steps - 1 }
            assert_eq!(check_completed(expected, &run, &request, &detector), Err(CandidateError::Execution));
        }
    }
    let one = SourceMapVerification { fields: 1, matches: 1, scan_steps: 1 };
    for overflow in [SourceMapVerification { fields: usize::MAX, ..one },
        SourceMapVerification { matches: usize::MAX, ..one }, SourceMapVerification { scan_steps: u64::MAX, ..one }] {
        assert_eq!(add_usage(one, overflow), Err(CandidateError::Execution));
    }
}

#[test]
fn map_and_native_cancellation_stay_failures_with_their_typed_deadline_semantics() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(planning_error(Int4LongRedactionError::Map(
            Int4SourceMapError::Source(Int4SourceError::Cancelled(cause)))), CandidateError::Timeout);
        assert_eq!(planning_error(Int4LongRedactionError::Redaction(Int4RedactionError::Cancelled(cause))),
            CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::SourceMap(
            Int4SourceMapError::Source(Int4SourceError::Cancelled(cause))))), CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::Redaction(
            Int4RedactionError::Cancelled(cause)))), CandidateError::Timeout);
    }
    assert_eq!(planning_error(Int4LongRedactionError::Redaction(Int4RedactionError::Identity)), CandidateError::Identity);
    assert_eq!(planning_error(Int4LongRedactionError::Map(Int4SourceMapError::WorkLimit)), CandidateError::Planning);
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::Redaction(
        Int4RedactionError::Cancelled(DecodeCancellationKind::PollQuota)))), CandidateError::Execution);
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(expired.checkpoint(0), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn bounded_publication_keeps_final_maps_and_accounting_without_private_source_or_prompts() {
    let (_, run, _, _) = fixture(true, true);
    let output = serde_json::json!({"fixture_kind":"synthetic-receipts-real-rules", "run":run});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-redact-document-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
        redaction_mode: "complete_document", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
        source_root_sha256: "fixture-source", model_content_sha256: "fixture-model",
        candidate_format: "fixture-format", quant_recipe: "fixture:group=128", group_width: 128,
        integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 16384, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-redact-document-v1");
    assert_eq!(value["output"]["run"]["result"]["text"], "é [redacted:email] 上海");
    assert!(value["output"]["run"]["result"]["edits"].is_array());
    assert!(value["output"]["run"]["verification"].is_object());
    assert_eq!(value["output"]["run"]["map_verification_used"]["scan_steps"], 22);
    assert_eq!(value["output"]["run"]["detection_verification_used"]["scan_steps"], 26);
    for absent in ["document", "source", "prompt", "prompt_digest", "ner_transcript", "key", "namespace",
        "source_dir", "source_manifest", "snapshot_path"] { assert!(value.get(absent).is_none()); }
    assert!(!std::str::from_utf8(&bytes).unwrap().contains("a@example.org"));
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn flush_failure_is_reported_after_complete_staging() {
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush failure")) }
    }
    let (_, run, _, _) = fixture(true, false);
    let complete = serde_json::json!({"fixture_kind":"synthetic-receipts-real-rules", "run":run});
    let mut writer = FlushFailure(Vec::new());
    assert_eq!(publish(&complete, 16384, &mut writer), Err(CandidateError::Output));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&writer.0).unwrap(), complete);
}
