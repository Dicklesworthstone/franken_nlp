//! Capacity and identity checks require no model and perform no corpus IO.
use super::*;
use crate::candidate_cli::scored_documents::tests::command;

fn facts() -> ArtifactIdentity {
    ArtifactIdentity {
        model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "metadata-only-unit-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32),
        logical_model_sha256: "cd".repeat(32),
    }
}

#[test]
fn whole_population_memory_has_an_exact_boundary_for_every_schedule() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        for flags in [&[][..], &["--prefill-rows", "4"][..],
            &["--compute-threads", "2"][..],
            &["--compute-threads", "2", "--prefill-rows", "4"][..]] {
            let mut admitted = command(name, flags).validate().unwrap();
            let peak = admit(&admitted).unwrap();
            assert!(peak > admitted.limits.weight_bytes + admitted.limits.preparation_bytes);
            admitted.limits.memory_bytes = peak;
            assert_eq!(admit(&admitted).unwrap(), peak);
            admitted.limits.memory_bytes = peak - 1;
            assert!(matches!(admit(&admitted), Err(CandidateError::Arguments)));
        }
    }
}

#[test]
fn transferred_transport_and_map_frontiers_are_both_priced() {
    for name in ["classify-corpus", "sentiment-corpus"] {
        let short = command(name, &["--max-input-bytes", "65536", "--max-chunks", "2", "--max-line-bytes", "65536"]).validate().unwrap();
        let transport = command(name, &["--max-input-bytes", "65536", "--max-chunks", "2", "--max-line-bytes", "1048576"]).validate().unwrap();
        let frontier = command(name, &["--max-input-bytes", "65536", "--max-chunks", "64", "--max-line-bytes", "65536"]).validate().unwrap();
        assert!(admit(&transport).unwrap() > admit(&short).unwrap());
        assert!(admit(&frontier).unwrap() > admit(&short).unwrap());
        let mut forged = command(name, &["--compute-threads", "2", "--prefill-rows", "4"]).validate().unwrap();
        forged.limits.compute.workspace_bytes = 0;
        assert!(matches!(admit(&forged), Err(CandidateError::Arguments)));
    }
}

#[test]
fn classification_corpus_binds_pinned_identity_policy_and_whole_run_work() {
    for encoding in ["byte-fallback", "safe-bpe-v1"] {
        let command = command("classify-corpus", &["--input-encoding", encoding]);
        let admitted = command.validate().unwrap();
        let defaults = command.parse_defaults(Some(
            r#"{"labels":[{"id":"yes","description":"supported"},{"id":"no","description":"unsupported"}]}"#),
            admitted.task).unwrap();
        let Corpus::Classify(planner, config) = configure(&admitted, &facts(), defaults).unwrap()
            else { panic!("classification corpus"); };
        assert_eq!(config.identity.task_spec, "classify-v1");
        assert_eq!(config.identity.logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(config.identity.template_digest, *planner.template_digest());
        assert_eq!(config.identity.tokenizer_digest, planner.tokenizer_digest());
        assert_eq!(config.task_ceiling, admitted.task);
        assert_eq!(config.max_model_work, admitted.mapping.work());
        assert_eq!(config.max_tokenizer_work, admitted.max_tokenizer_work);
        assert_eq!(planner.document_encoding(), &admitted.encoding);
        assert_eq!(config.defaults.unwrap().labels.len(), 2);
    }
}

#[test]
fn sentiment_corpus_retains_fixed_axes_and_refuses_cross_task_defaults() {
    let command = command("sentiment-corpus", &[]);
    let admitted = command.validate().unwrap();
    let defaults = command.parse_defaults(None, admitted.task).unwrap();
    let Corpus::Sentiment(planner, config) = configure(&admitted, &facts(), defaults).unwrap()
        else { panic!("sentiment corpus"); };
    assert_eq!(config.identity.task_spec, "sentiment-v1");
    assert_eq!(config.identity.template_digest, *planner.template_digest());
    assert_eq!(config.task_ceiling, admitted.task);
    assert_eq!(config.max_model_work, admitted.mapping.work());
    assert_eq!(config.defaults.unwrap().axes, crate::tasks::sentiment::SentimentAxis::ALL.to_vec());
    assert!(matches!(configure(&admitted, &facts(), Defaults::Classify(None)), Err(CandidateError::Arguments)));
}

#[test]
fn provenance_reports_complete_document_and_declared_invocation_scope() {
    let admitted = command("classify-corpus", &["--input-encoding", "safe-bpe-v1",
        "--compute-threads", "2", "--prefill-rows", "4"]).validate().unwrap();
    let facts = facts();
    let provenance = Provenance {
        protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256,
        logical_model_sha256: &facts.logical_model_sha256, quant_recipe: &facts.recipe_id,
        task: admitted.mapping.task_name(), complete_document: true,
        input_encoding: input_encoding::reported_policy(&admitted.encoding),
        invocation_encoding_work_limits: admitted.max_tokenizer_work,
        compute_threads: Some(2), prefill_rows: Some(4),
    };
    let json = serde_json::to_value(&provenance).unwrap();
    assert_eq!(json["complete_document"], true);
    assert_eq!(json["evidence"], "non_authoritative");
    assert_eq!(json["compute_threads"], 2);
    assert_eq!(json["prefill_rows"], 4);
    assert!(json["invocation_encoding_work_limits"].is_object());
    assert!(json.get("document").is_none());
    assert!(json.get("confidence").is_none());
    CandidateWriter::new(Vec::<u8>::new(), &provenance,
        admitted.transport.max_output_line_bytes, admitted.output_bytes).unwrap();
}

#[test]
fn hosted_terminal_causes_keep_timeout_policy_at_the_cli_boundary() {
    for cause in [DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline] {
        assert!(matches!(stopped(cause), CandidateError::Timeout));
    }
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::Shutdown] {
        assert!(matches!(stopped(cause), CandidateError::Execution));
    }
}
