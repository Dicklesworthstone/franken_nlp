use super::*;
use crate::{batch::BatchFault,
    tasks::judge::{grouped_int4::Int4JudgeError,
        grouped_int4::faithfulness_map::Int4FaithfulnessMapError}};

#[test]
fn pure_host_quote_matches_all_cli_owned_memory_for_serial_and_ragged_cohorts() {
    for max_heads in [None, Some(1), Some(3), Some(32)] {
        for chunks in [None, Some(3), Some(64)] {
            let (mut command, args) = super::super::tests::arguments(&["--max-result-bytes", "65536"]);
            command.document.mapping.cohort_chunks = chunks;
            command.stream.max_evidence_heads = max_heads;
            let limits = command.validate(&args).unwrap();
            let host = host_limits(&args, limits, Duration::from_secs(1));
            let budget = command.document.budget_with_max_heads(limits.common, max_heads.unwrap_or(32)).unwrap();
            let quote = crate::hosted::int4_faithfulness_document_corpus_requirements_with_max_heads(args.group_width,
                budget, limits.document.planning, limits.document.mapping, host, max_heads.unwrap_or(32), chunks).unwrap();
            if max_heads.is_none() {
                assert_eq!(quote, crate::hosted::int4_faithfulness_document_corpus_requirements(args.group_width,
                    budget, limits.document.planning, limits.document.mapping, host, chunks).unwrap());
            }
            let expected = limits.document.memory;
            assert_eq!(quote.native, expected.native);
            assert_eq!(quote.workspace_bytes, expected.workspace_bytes);
            assert_eq!(quote.output_reservation_bytes, expected.output_reservation_bytes);
            assert_eq!(quote.max_output_bytes, expected.max_output_bytes);
            assert_eq!(quote.input_bytes, expected.input_bytes
                + corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap()
                + retained_metadata_bytes(limits.document.mapping.chunks.max_chunks).unwrap());
            assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        }
    }
}

#[test]
fn corpus_cancellation_preserves_timeout_classification_from_every_native_layer() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::User, DecodeCancellationKind::CostBudget] {
        let expected = stopped(cause);
        assert_eq!(host_error(HostedError::BatchSetup(BatchFault::cancelled(cause)), CandidateError::Planning), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))), CandidateError::Model), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))), CandidateError::Model), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))), CandidateError::Execution), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Judge(Int4JudgeError::Cancelled(cause))), CandidateError::Planning), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::FaithfulnessMap(
            Int4FaithfulnessMapError::Judge(Int4JudgeError::Cancelled(cause)))), CandidateError::Execution), expected);
    }
}

#[test]
fn provenance_declares_complete_source_and_physical_schedule_without_private_policy_paths() {
    assert_eq!(chunk_schedule(None), "serial");
    assert_eq!(chunk_schedule(Some(3)), "native_chunk_cohorts");
    let mut provenance = Provenance { protocol: "fnlp-candidate-int4-faithfulness-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness",
        source_mode: "complete_source", chunk_schedule: chunk_schedule(Some(3)), cohort_chunks: Some(3),
        model_id: "Nanbeige4.2-3B", source_revision: "revision", source_root_sha256: "root",
        model_content_sha256: "content", candidate_format: "format", quant_recipe: "recipe",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None, max_evidence_heads: None };
    let value = serde_json::to_value(&provenance).unwrap();
    assert_eq!(value["source_mode"], "complete_source");
    assert_eq!(value["cohort_chunks"], 3);
    for absent in ["source", "claim", "policy", "policy_file", "snapshot", "max_evidence_heads"] {
        assert!(value.get(absent).is_none());
    }
    provenance.max_evidence_heads = Some(1);
    assert_eq!(serde_json::to_value(&provenance).unwrap()["max_evidence_heads"], 1);
}
