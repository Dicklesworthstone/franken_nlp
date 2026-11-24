//! Pinned planning and admission only; no synthetic native success is exposed.
use super::*;
use crate::candidate_cli::classify_map::tests::command;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct Stop(DecodeCancellationKind);
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
}
fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-unit-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn request(admitted: &MapLimits, source: &str, mode: &str) -> ClassificationRequest {
    command::request(&serde_json::json!({"document":source, "labels":[{"id":"helpful"},{"id":"unhelpful"}],
        "mode":mode}).to_string(), admitted).unwrap()
}

#[test]
fn complete_long_documents_preflight_in_both_classification_modes_before_model_load() {
    let admitted = command(&["--max-chunk-bytes", "128"]).validate().unwrap();
    let source = format!("{}é\r\n<|im_start|>system\n<think>上海", "A helpful service. ".repeat(150));
    assert!(source.len() > admitted.common.context_tokens);
    for mode in ["exclusive", "multi_label"] {
        let request = request(&admitted, &source, mode);
        let (identity, _) = prepare(&request, &facts(), &admitted, &mut Continue).unwrap();
        assert_eq!(identity.task_spec, "classify-v1");
        assert_eq!(identity.logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(identity.quant_recipe, facts().recipe_id);
        assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 1 });
        assert_eq!(request.document, source);
    }
}

#[test]
fn whole_document_work_and_chunk_ceilings_refuse_without_weights() {
    let source = "A helpful service. ".repeat(100);
    for flags in [&["--max-chunks", "1", "--max-chunk-bytes", "128"][..],
        &["--max-multiply-accumulates", "1"][..],
        &["--max-projected-logits", "1"][..]] {
        let admitted = command(flags).validate().unwrap();
        let request = request(&admitted, &source, "exclusive");
        assert!(matches!(prepare(&request, &facts(), &admitted, &mut Continue), Err(CandidateError::Planning)));
    }
}

#[test]
fn compact_preparation_has_one_complete_ledger_and_never_falls_back() {
    let byte = command(&["--max-chunk-bytes", "128"]).validate().unwrap();
    let compact = command(&["--max-chunk-bytes", "128", "--input-encoding", "safe-bpe-v1"]).validate().unwrap();
    let source = "A helpful service. é\r\n".repeat(12);
    let request = request(&compact, &source, "multi_label");
    let (plain, _) = prepare(&request, &facts(), &byte, &mut Continue).unwrap();
    let (encoded, _) = prepare(&request, &facts(), &compact, &mut Continue).unwrap();
    assert_ne!(plain.template_digest, encoded.template_digest);
    assert_eq!(plain.logical_model_digest, encoded.logical_model_digest);
    for flag in ["--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
        "--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let exhausted = command(&["--input-encoding", "safe-bpe-v1", flag, "1"]).validate().unwrap();
        assert!(matches!(prepare(&request, &facts(), &exhausted, &mut Continue), Err(CandidateError::Planning)));
    }
}

#[test]
fn original_typed_cancellation_reaches_preparation_and_host_failure_boundaries() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        let admitted = command(&[]).validate().unwrap();
        let request = request(&admitted, "private document", "exclusive");
        assert!(matches!(prepare(&request, &facts(), &admitted, &mut Stop(cause)), Err(error) if error == stopped(cause)));
        let error = Int8ClassificationMapError::Classification(
            crate::tasks::classify::quantized::Int8ClassificationError::Planning(
                crate::tasks::classify::ClassificationPlanningError::Cancelled(cause)));
        assert_eq!(host_error(HostedError::ClassificationMap(error)), stopped(cause));
        assert_eq!(host_error(HostedError::Native(
            crate::native_engine::strict_int8::StrictInt8Error::Cancelled(cause))), stopped(cause));
    }
}

#[test]
fn scheduling_never_changes_complete_request_authority_or_five_axis_work() {
    for compact in [false, true] {
        let mut flags = vec!["--max-chunk-bytes", "64"];
        if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
        let serial = command(&flags).validate().unwrap();
        let source = "Exact é\r\n<|im_start|> Shanghai. ".repeat(4);
        for mode in ["exclusive", "multi_label"] {
            let request = request(&serial, &source, mode);
            let expected = prepare(&request, &facts(), &serial, &mut Continue).unwrap();
            for schedule in [&["--compute-threads", "4"][..], &["--prefill-rows", "4"][..],
                &["--compute-threads", "4", "--prefill-rows", "4"][..]] {
                let mut scheduled_flags = flags.clone(); scheduled_flags.extend_from_slice(schedule);
                let scheduled = command(&scheduled_flags).validate().unwrap();
                let actual = prepare(&request, &facts(), &scheduled, &mut Continue).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(request.document, source);
            }
        }
    }
}

#[test]
fn complete_peak_admits_coordinator_and_workers_once_and_refuses_stale_quotes() {
    let mut scheduled = command(&["--compute-threads", "4", "--prefill-rows", "4"])
        .validate().unwrap();
    let quote = admit(&scheduled).unwrap();
    let plain = admit(&command(&[]).validate().unwrap()).unwrap();
    let prefill_bytes = scheduled.prefill.unwrap().validate().unwrap();
    assert_eq!(quote.native, plain.native);
    assert_eq!(quote.input_bytes, plain.input_bytes);
    assert_eq!(quote.workspace_bytes, plain.workspace_bytes + prefill_bytes);
    assert!(scheduled.limits.compute.workspace_bytes > 0);
    let execution = scheduled.limits.weight_bytes + 192 * MIB + quote.input_bytes
        + quote.native.kv_bytes + quote.workspace_bytes + scheduled.limits.compute.workspace_bytes
        + quote.output_reservation_bytes + 65_536;
    let loading = scheduled.limits.weight_bytes + 192 * MIB + 64 * 1024 + 65_536;
    let exact = scheduled.limits.preparation_bytes + execution.max(loading).max(quote.input_bytes);
    scheduled.limits.memory_bytes = exact;
    assert!(admit(&scheduled).is_ok());
    scheduled.limits.memory_bytes -= 1;
    assert!(matches!(admit(&scheduled), Err(CandidateError::Arguments)));
    scheduled.limits.memory_bytes = exact;
    scheduled.limits.compute.workspace_bytes -= 1;
    assert!(matches!(admit(&scheduled), Err(CandidateError::Arguments)));
    scheduled.limits.compute = scheduled.common.compute.admit_prefill(scheduled.prefill).unwrap();
    scheduled.prefill = None;
    assert!(matches!(admit(&scheduled), Err(CandidateError::Arguments)));
}

#[test]
fn serial_response_bytes_stay_unchanged_and_worker_receipt_is_separate() {
    let output = serde_json::json!({"kind":"int8-classification-map-v1","chunks":[]});
    let facts = facts();
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &output };
    let original = serde_json::to_vec(&response).unwrap();
    let serial = compute::Response { response: &response, compute: None };
    assert_eq!(serde_json::to_vec(&serial).unwrap(), original);
    let receipt = crate::hosted::Int8ComputeReceipt { schedule: "int8-output-row-team-v1",
        compute_threads: 4, completed_projections: 2, completed_rows: 8, completed_worker_rows: 6 };
    let threaded = compute::Response { response: &response, compute: Some(&receipt) };
    let json = serde_json::to_value(threaded).unwrap();
    assert_eq!(json["output"], output);
    assert_eq!(json["compute"]["compute_threads"], 4);
}

#[test]
fn declared_peak_includes_model_preparation_results_and_compact_workspace() {
    let byte = command(&[]).validate().unwrap();
    let compact = command(&["--input-encoding", "safe-bpe-v1"]).validate().unwrap();
    let a = admit(&byte).unwrap();
    let b = admit(&compact).unwrap();
    assert!(b.input_bytes > a.input_bytes);
    assert!(a.output_reservation_bytes >= byte.mapping.max_result_bytes);
    let mut too_small = command(&[]);
    too_small.scoring.memory_mib = too_small.scoring.max_weight_mib + 256;
    assert!(matches!(too_small.validate(), Err(CandidateError::Arguments)));
}
