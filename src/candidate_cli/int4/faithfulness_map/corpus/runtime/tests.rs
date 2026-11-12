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
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None, max_evidence_heads: None,
        compute_schedule: None, input_encoding: None };
    let value = serde_json::to_value(&provenance).unwrap();
    assert_eq!(value["source_mode"], "complete_source");
    assert_eq!(value["cohort_chunks"], 3);
    for absent in ["source", "claim", "policy", "policy_file", "snapshot", "max_evidence_heads", "compute_schedule", "input_encoding"] {
        assert!(value.get(absent).is_none());
    }
    provenance.max_evidence_heads = Some(1);
    assert_eq!(serde_json::to_value(&provenance).unwrap()["max_evidence_heads"], 1);
}

#[test]
fn row_team_provenance_is_optional_and_does_not_change_inner_events_or_head_policy() {
    // A framing-only fixture exercises exact number preservation; it makes
    // no claim about any native judgment or successful model execution.
    let event = br#"{"event":"wire_fixture","number":12345678901234567890123456789012345678}"#;
    let mut expected = None;
    for threads in [1, 2, 4, 64] {
        let provenance = Provenance { protocol: "fnlp-candidate-int4-faithfulness-corpus-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness",
            source_mode: "complete_source", chunk_schedule: chunk_schedule(None), cohort_chunks: None,
            model_id: "Nanbeige4.2-3B", source_revision: "revision", source_root_sha256: "root",
            model_content_sha256: "content", candidate_format: "format", quant_recipe: "recipe",
            group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None, max_evidence_heads: Some(3),
            compute_schedule: compute::Schedule::for_threads(threads), input_encoding: None };
        let mut value = serde_json::to_value(&provenance).unwrap();
        if threads == 1 { assert!(value.get("compute_schedule").is_none()); }
        else {
            assert_eq!(value["compute_schedule"]["execution"], "int4-output-row-team-v1");
            assert_eq!(value["compute_schedule"]["declared_compute_threads"], threads);
        }
        assert!(value.get("compute").is_none());
        value.as_object_mut().unwrap().remove("compute_schedule");
        if let Some(expected) = &expected { assert_eq!(&value, expected); }
        else { expected = Some(value); }
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 8192, 65536).unwrap();
            writer.write_all(event).unwrap(); writer.write_all(b"\n").unwrap(); writer.flush().unwrap();
        }
        let mut framed = serde_json::to_vec(&provenance).unwrap(); framed.pop();
        framed.extend_from_slice(b",\"record\":"); framed.extend_from_slice(event); framed.extend_from_slice(b"}\n");
        assert_eq!(output, framed);
    }
}

#[test]
fn team_runtime_rechecks_conflicts_and_prices_the_admitted_process_width() {
    for threads in [1, 2, 4, 64] {
        let width = threads.to_string();
        let (command, args) = super::super::tests::arguments(&["--compute-threads", width.as_str()]);
        let limits = command.validate(&args).unwrap(); let config = compute::resource_config(limits.common);
        assert_eq!(config.scoped_cpu_children_per_coordinator, threads - 1);
        assert_eq!(config.thread_ceiling, threads + 1);
        assert_eq!(config.memory_ceiling_bytes, limits.common.memory_bytes);
    }
    let (mut command, args) = super::super::tests::arguments(&["--compute-threads", "4"]);
    let admitted = command.validate(&args).unwrap(); command.document.mapping.cohort_chunks = Some(1);
    assert_eq!(execute(args, command, admitted, super::super::tests::NoIo, super::super::tests::NoIo),
        Err(CandidateError::Arguments));
}


#[test]
fn selected_compact_policy_matches_host_memory_and_is_present_on_every_stream_record() {
    for chunks in [None, Some(3)] {
        let (mut command, args) = super::super::tests::arguments(&[
            "--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "65536",
            "--max-tokenizer-output-tokens", "32768", "--max-result-bytes", "65536"]);
        command.document.mapping.cohort_chunks = chunks;
        command.stream.max_evidence_heads = Some(3);
        let selected = command.document.input_encoding(&args).unwrap();
        let limits = command.validate(&args).unwrap();
        let budget = command.document.budget_with_max_heads(limits.common, 3).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(1));
        let base = crate::hosted::int4_faithfulness_document_corpus_requirements_with_max_heads(
            args.group_width, budget, limits.document.planning, limits.document.mapping, host, 3, chunks).unwrap();
        let quote = base.with_input_encoding(&selected).unwrap();
        assert_eq!(quote.input_bytes, limits.document.memory.input_bytes
            + corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap()
            + retained_metadata_bytes(limits.document.mapping.chunks.max_chunks).unwrap());
        assert_eq!(quote.native, limits.document.memory.native);
        assert_eq!(quote.workspace_bytes, limits.document.memory.workspace_bytes);
        let provenance = Provenance { protocol: "fnlp-candidate-int4-faithfulness-corpus-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness",
            source_mode: "complete_source", chunk_schedule: chunk_schedule(chunks), cohort_chunks: chunks,
            model_id: "Nanbeige4.2-3B", source_revision: "revision", source_root_sha256: "root",
            model_content_sha256: "content", candidate_format: "format", quant_recipe: "recipe",
            group_width: args.group_width, integer_kernel: "s_scalar_i32", snapshot: None,
            max_evidence_heads: Some(3), compute_schedule: None,
            input_encoding: input_encoding::reported_policy(&selected) };
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 8192, 65536).unwrap();
            for event in ["doc_result", "doc_error", "flush", "run_complete"] {
                // Transport-only frames, never fabricated native evidence.
                let frame = format!("{{\"event\":\"{event}\",\"fixture\":true}}\n");
                writer.write_all(frame.as_bytes()).unwrap();
                writer.flush().unwrap();
            }
        }
        for (index, line) in output.split(|byte| *byte == b'\n').filter(|line| !line.is_empty()).enumerate() {
            let value: serde_json::Value = serde_json::from_slice(line).unwrap();
            assert_eq!(value["input_encoding"], serde_json::to_value(&selected).unwrap());
            assert_eq!(value["record"]["event"], ["doc_result", "doc_error", "flush", "run_complete"][index]);
            assert_eq!(value["record"]["fixture"], true);
            for private in ["source", "claim", "policy", "policy_file"] { assert!(value.get(private).is_none()); }
        }
    }
}

#[test]
fn compact_corpus_runtime_revalidates_snapshot_refusal_before_any_private_io() {
    let (command, mut args) = super::super::tests::arguments(&["--input-encoding", "safe-bpe-v1"]);
    let admitted = command.validate(&args).unwrap();
    args.save_snapshot = Some(PathBuf::from("must-not-be-created"));
    assert_eq!(execute(args, command, admitted, super::super::tests::NoIo, super::super::tests::NoIo),
        Err(CandidateError::Arguments));
}
