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
    }
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
