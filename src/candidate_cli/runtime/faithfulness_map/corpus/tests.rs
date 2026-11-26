//! Pure memory/identity checks against the actual pinned faithfulness compiler.
use super::*;
use crate::candidate_cli::faithfulness_map::corpus::tests::{command, POLICY};

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "faithfulness-corpus-cli-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}

#[test]
fn complete_population_memory_has_an_exact_boundary_for_all_schedules() {
    for compact in [false, true] {
        for mut flags in [vec![], vec!["--prefill-rows", "4"], vec!["--compute-threads", "2"],
            vec!["--compute-threads", "2", "--prefill-rows", "4"]] {
            if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
            let mut admitted = command(&flags).validate().unwrap();
            let peak = admit(&admitted).unwrap();
            assert!(peak > admitted.document.limits.weight_bytes + admitted.document.limits.preparation_bytes);
            admitted.document.limits.memory_bytes = peak;
            assert_eq!(admit(&admitted).unwrap(), peak);
            admitted.document.limits.memory_bytes -= 1;
            assert!(matches!(admit(&admitted), Err(CandidateError::Arguments)));
        }
    }
}

#[test]
fn retained_chunks_transport_and_persistent_workers_are_admitted_together() {
    let small = command(&["--max-input-bytes", "65536", "--max-chunks", "2",
        "--max-line-bytes", "65536"]).validate().unwrap();
    let transport = command(&["--max-input-bytes", "65536", "--max-chunks", "2",
        "--max-line-bytes", "1048576"]).validate().unwrap();
    let chunks = command(&["--max-input-bytes", "65536", "--max-chunks", "64",
        "--max-line-bytes", "65536"]).validate().unwrap();
    assert!(admit(&transport).unwrap() > admit(&small).unwrap());
    assert!(admit(&chunks).unwrap() > admit(&small).unwrap());
    let mut stale = command(&["--compute-threads", "2", "--prefill-rows", "4"]).validate().unwrap();
    stale.document.limits.compute.workspace_bytes = 0;
    assert!(matches!(admit(&stale), Err(CandidateError::Arguments)));
}

#[test]
fn pinned_config_binds_fixed_policy_encoding_and_all_work_axes() {
    for encoding in ["byte-fallback", "safe-bpe-v1"] {
        let admitted = command(&["--input-encoding", encoding]).validate().unwrap();
        let policy = command::parse_policy(POLICY).unwrap();
        let (planner, config) = configure(&admitted, &facts(), policy).unwrap();
        assert_eq!(config.identity.task_spec, "judge-v1");
        assert_eq!(config.identity.logical_model_digest.to_hex(), facts().logical_model_sha256);
        assert_eq!(config.identity.template_digest, *planner.template_digest());
        assert_eq!(config.identity.tokenizer_digest, planner.tokenizer_digest());
        assert_eq!(config.policy, policy);
        assert_eq!(config.task_ceiling, admitted.document.task);
        assert_eq!(config.mapping.chunks, admitted.document.mapping.chunks);
        assert_eq!(config.mapping.max_result_bytes, admitted.document.mapping.max_result_bytes);
        assert_eq!(config.mapping.max_retained_result_bytes, admitted.document.mapping.max_retained_result_bytes);
        assert_eq!(config.mapping.aggregation.minimum_classified_coverage_ppm,
            admitted.document.mapping.aggregation.minimum_classified_coverage_ppm);
        assert_eq!(config.max_model_work, admitted.document.mapping.max_model_work);
        assert_eq!(config.max_tokenizer_work, admitted.max_tokenizer_work);
        assert_eq!(planner.document_encoding(), &admitted.document.encoding);
    }
}

#[test]
fn provenance_marks_complete_source_and_reports_only_declared_execution_metadata() {
    let admitted = command(&["--input-encoding", "safe-bpe-v1",
        "--compute-threads", "2", "--prefill-rows", "4"]).validate().unwrap();
    let facts = facts();
    let provenance = Provenance { protocol: "fnlp-candidate-batch-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, task: "judge", judgment_mode: "faithfulness", complete_document: true,
        input_encoding: input_encoding::reported_policy(&admitted.document.encoding),
        invocation_encoding_work_limits: admitted.max_tokenizer_work,
        compute_threads: Some(2), prefill_rows: Some(4) };
    let value = serde_json::to_value(&provenance).unwrap();
    assert_eq!(value["judgment_mode"], "faithfulness");
    assert_eq!(value["complete_document"], true);
    assert_eq!(value["evidence"], "non_authoritative");
    assert!(value["invocation_encoding_work_limits"].is_object());
    for private in ["source", "claim", "policy", "confidence", "scores"] {
        assert!(value.get(private).is_none(), "{private}");
    }
    CandidateWriter::new(Vec::<u8>::new(), &provenance,
        admitted.transport.max_output_line_bytes, admitted.output_bytes).unwrap();
}
