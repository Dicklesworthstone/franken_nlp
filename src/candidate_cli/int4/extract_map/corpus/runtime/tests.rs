//! Shared admission, exact frame preservation and typed terminal failures.
use super::*;
use super::super::tests::{arguments, NoIo};
use crate::{
    batch::{BatchFault, BatchRunError, BatchSummary},
    native_engine::constrained::JsonDecodeError,
    tasks::extract::{ExtractError, grouped_int4::Int4ExtractError},
    batch::extract::grouped_int4::long::Int4ExtractionMapError,
};

#[test]
fn pure_host_quote_matches_cli_for_every_admitted_native_document_schedule() {
    for width in [32, 128] {
        for chunks in [None, Some(1), Some(3), Some(64)] {
            let (mut command, mut args) = arguments(&["--max-result-bytes", "65536"]);
            args.group_width = width; args.max_load_mib = 65_536;
            command.document.cohort_chunks = chunks;
            let limits = command.validate(&args).unwrap();
            let budget = command.document.extraction.budget(&args, limits.common);
            let host = host_limits(&args, limits, Duration::from_secs(7));
            let quote = crate::hosted::int4_extraction_document_corpus_requirements(width, budget,
                limits.document.mapping, host, limits.document.reduction_reserve_bytes, chunks).unwrap();
            assert_eq!(quote, limits.memory);
            assert_eq!(quote.max_output_bytes as usize + FRAME_ALLOWANCE as usize,
                limits.transport.max_output_line_bytes);
            assert_eq!(host.native.run.max_elapsed, Duration::from_secs(7));
            assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
            assert_eq!(host.transport.max_work, limits.transport.max_work);
            assert_eq!(quote.native.kv_bytes, budget.max_kv_bytes * chunks.unwrap_or(1) as u64);
            let expected_frontier = limits.document.workspace_bytes
                + limits.document.cohort.map_or(0, |cohort| cohort.workspace_bytes);
            assert_eq!(quote.document_workspace_bytes, expected_frontier);
        }
    }
}

#[test]
fn runtime_rechecks_programmatic_refusal_before_schema_or_population_io() {
    let (mut command, args) = arguments(&[]);
    let admitted = command.validate(&args).unwrap();
    command.stream.max_run_extraction_fields = 0;
    assert_eq!(execute(args, command, admitted, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn typed_cancellation_survives_loading_planning_batching_and_nested_failures() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = stopped(cause);
        let extraction = || Int4ExtractError::Extraction(ExtractError::Decode(JsonDecodeError::Cancelled(cause)));
        for error in [HostedError::BatchSetup(BatchFault::cancelled(cause)),
            HostedError::Batch(BatchRunError { fault: BatchFault::cancelled(cause), summary: BatchSummary::default() }),
            HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Extraction(extraction())),
            HostedError::Int4(Int4HostError::ExtractionMap(Int4ExtractionMapError::Extraction(extraction()))),
            HostedError::Scope { source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
                physical_error: Some(Box::new(HostedError::BatchSetup(BatchFault::cancelled(cause)))) }] {
            assert_eq!(host_error(error, CandidateError::Batch), expected);
        }
        assert!(diagnostic(expected).contains("earlier completed frames may exist"));
    }
    assert_eq!(host_error(HostedError::ModelIdentity, CandidateError::Batch), CandidateError::Identity);
    assert_eq!(host_error(HostedError::ResourceDomain, CandidateError::Batch), CandidateError::Identity);
    assert_eq!(host_error(HostedError::Reservation(crate::ReservationError::ArithmeticOverflow), CandidateError::Batch),
        CandidateError::Memory);
    assert_eq!(completed(BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
    assert_eq!(completed(BatchSummary { succeeded: 2, ..BatchSummary::default() }), Ok(()));
}

#[test]
fn corpus_frame_keeps_exact_chunk_json_and_reveals_only_public_outer_metadata() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    for (operation, cohort_chunks) in [(None, None), (Some("saved"), Some(1)), (Some("loaded"), Some(3))] {
        let provenance = Provenance {
            protocol: "fnlp-candidate-int4-extract-corpus-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "extract",
            source_mode: "complete_document", chunk_schedule: chunk_schedule(cohort_chunks), cohort_chunks,
            root_array_union: None,
            model_id: "Nanbeige4.2-3B", source_revision: "public-revision", source_root_sha256: "public-source",
            model_content_sha256: "public-model", candidate_format: "candidate-int4", quant_recipe: "recipe",
            group_width: 128, integer_kernel: "s_scalar_i32",
            snapshot: operation.map(|name| SnapshotRecord::new(name, need)),
        };
        // A transport fixture only; no fabricated native result enters a planner.
        let inner = "{\"event\":\"fixture\",\"result\":{\"chunks\":[{\"json\":\"{\\\"n\\\":12345678901234567890123456789012345678}\",\"text\":\"café 上海\"}]}}\n";
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let text = std::str::from_utf8(&output).unwrap();
        assert!(text.ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["source_mode"], "complete_document");
        assert_eq!(value["evidence"], "non_authoritative");
        if let Some(chunks) = cohort_chunks {
            assert_eq!(value["chunk_schedule"], "native_chunk_cohorts"); assert_eq!(value["cohort_chunks"], chunks);
        } else {
            assert_eq!(value["chunk_schedule"], "serial"); assert!(value.get("cohort_chunks").is_none());
        }
        if let Some(operation) = operation {
            assert_eq!(value["snapshot"]["operation"], operation);
            assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
        } else { assert!(value.get("snapshot").is_none()); }
        for absent in ["source", "schema", "schema_path", "source_dir", "source_manifest",
            "snapshot_path", "schema_digest", "taskir_digest", "document"] {
            assert!(value.get(absent).is_none());
        }
        assert!(matches!(CandidateWriter::new(NoIo, &provenance, 4096, 1), Err(CandidateError::Output)));
    }
}
