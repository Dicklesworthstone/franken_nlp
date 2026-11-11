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
fn compact_host_and_cli_quotes_charge_one_encoder_workspace_for_every_schedule() {
    for width in [32, 128] {
        for chunks in [None, Some(1), Some(3), Some(64)] {
            let (mut plain, mut args) = arguments(&["--max-result-bytes", "65536"]);
            let (mut compact, _) = arguments(&["--max-result-bytes", "65536", "--input-encoding", "safe-bpe-v1",
                "--max-tokenizer-input-bytes", "262144", "--max-tokenizer-output-tokens", "131072",
                "--max-tokenizer-pair-checks", "8192", "--max-tokenizer-pair-bytes", "65536"]);
            args.group_width = width; args.max_load_mib = 65_536;
            plain.document.cohort_chunks = chunks; compact.document.cohort_chunks = chunks;
            let before = plain.validate(&args).unwrap();
            let after = compact.validate(&args).unwrap();
            let encoding = compact.document.encoding.policy().unwrap();
            let crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(policy) = &encoding
                else { panic!("compact policy required"); };
            let budget = compact.document.extraction.budget(&args, after.common);
            let quote = crate::hosted::int4_extraction_document_object_corpus_requirements_with_encoding(width,
                budget, after.document.mapping, host_limits(&args, after, Duration::from_secs(7)),
                after.document.reduction_reserve_bytes, chunks, compact.document.extraction.compiler(),
                after.root_object, &encoding).unwrap();
            assert_eq!(quote, after.memory);
            assert_eq!(quote.input_bytes - before.memory.input_bytes, policy.requirements().unwrap().workspace_bytes);
            assert_eq!(quote.document_workspace_bytes, before.memory.document_workspace_bytes);
            assert_eq!(quote.document_workspace_bytes, after.document.workspace_bytes
                + after.document.cohort.map_or(0, |cohort| cohort.workspace_bytes));
            assert_eq!(quote.native, before.memory.native);
            assert_eq!(quote.workspace_bytes, before.memory.workspace_bytes);
            assert_eq!(quote.output_reservation_bytes, before.memory.output_reservation_bytes);
            assert_eq!(quote.native.kv_bytes, budget.max_kv_bytes * chunks.unwrap_or(1) as u64);
            assert_eq!(after.cli_preparation_bytes, before.cli_preparation_bytes);
            assert_eq!(after.common.preparation_bytes, before.common.preparation_bytes);
        }
    }
}

#[test]
fn root_object_host_and_cli_quotes_include_the_same_complete_reconciliation_ownership() {
    for width in [32, 128] {
        for chunks in [None, Some(1), Some(3), Some(64)] {
            let (mut command, mut args) = arguments(&["--reconcile-root-object", "--max-result-bytes", "65536",
                "--max-selected-rows", "32"]);
            args.group_width = width; args.max_load_mib = 65_536;
            command.document.cohort_chunks = chunks;
            let limits = command.validate(&args).unwrap();
            let budget = command.document.extraction.budget(&args, limits.common);
            let host = host_limits(&args, limits, Duration::from_secs(7));
            let quote = crate::hosted::int4_extraction_document_object_corpus_requirements(width, budget,
                limits.document.mapping, host, limits.document.reduction_reserve_bytes, chunks,
                command.document.extraction.compiler(), limits.root_object).unwrap();
            assert_eq!(quote, limits.memory);
            assert_eq!(quote.document_workspace_bytes, limits.document.workspace_bytes
                + limits.document.cohort.map_or(0, |cohort| cohort.workspace_bytes)
                + limits.object_preparation_bytes);
            assert_eq!(quote.max_output_bytes as usize + FRAME_ALLOWANCE as usize,
                limits.transport.max_output_line_bytes);
            assert_eq!(quote.native.kv_bytes, budget.max_kv_bytes * chunks.unwrap_or(1) as u64);
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
            HostedError::Int4(Int4HostError::ObjectExtractionMap(
                crate::batch::extract::grouped_int4::long::object::Int4ObjectExtractionMapError::Reconciliation(
                    crate::tasks::extract::object_reconcile::RootObjectError::Cancelled(cause)))),
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
            root_array_union: None, root_object_reconciliation: None, max_selected_rows: None, input_encoding: None,
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
        assert!(value.get("max_selected_rows").is_none());
        assert!(value.get("root_object_reconciliation").is_none());
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
        let selected = Provenance { max_selected_rows: Some(8192), ..provenance };
        let mut selected_output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut selected_output, &selected, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let selected_value: serde_json::Value = serde_json::from_slice(&selected_output).unwrap();
        assert_eq!(selected_value["max_selected_rows"], 8192);
        assert_eq!(selected_value["record"], value["record"]);
        assert!(std::str::from_utf8(&selected_output).unwrap()
            .ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        let root = RootObjectLimits { max_properties: 16, max_origins: 64,
            max_input_bytes: 65_536, max_json_bytes: 4096, max_nodes: 4096, max_result_bytes: 8192 };
        let reconciled = Provenance { root_object_reconciliation: Some(RootObjectSelection::new(root)), ..selected };
        let mut object_output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut object_output, &reconciled, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let object_value: serde_json::Value = serde_json::from_slice(&object_output).unwrap();
        assert_eq!(object_value["record"], value["record"]);
        assert_eq!(object_value["root_object_reconciliation"]["policy"],
            crate::tasks::extract::object_reconcile::ROOT_OBJECT_RECONCILIATION);
        assert_eq!(object_value["root_object_reconciliation"]["schema_derivation"],
            crate::tasks::extract::object_reconcile::ROOT_OBJECT_DERIVATION);
        assert_eq!(object_value["root_object_reconciliation"]["map_prompt"],
            crate::batch::extract::grouped_int4::long::object::ROOT_OBJECT_MAP_PROMPT);
        assert_eq!(object_value["root_object_reconciliation"]["limits"], serde_json::to_value(root).unwrap());
        assert!(std::str::from_utf8(&object_output).unwrap()
            .ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
    }
}

#[test]
fn compact_provenance_preserves_inner_bytes_and_every_empty_stream_frame_including_completion() {
    use crate::{batch::{BatchDocument, BatchItemFailure, BatchLimits, BatchProcessor, BatchWork},
        tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};
    struct Empty;
    impl BatchProcessor for Empty {
        type Args = serde_json::Value;
        type Prepared = ();
        type Output = ();
        fn prepare(&mut self, _: BatchDocument<Self::Args>) -> Result<(), BatchItemFailure> {
            panic!("empty population must not prepare a document");
        }
        fn planned_work(&self, _: &()) -> BatchWork { panic!("empty population has no model work"); }
        fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<(), BatchItemFailure> {
            panic!("empty population must not execute native inference");
        }
    }
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let transport = BatchLimits { max_output_line_bytes: 4096, ..BatchLimits::default() };
    let mut inner = Vec::new();
    let summary = crate::batch::run_ndjson(&mut io::Cursor::new(b""), &mut inner,
        &mut Empty, transport, &mut Continue).unwrap();
    assert_eq!(summary.requests, 0); assert_eq!(summary.succeeded, 0); assert_eq!(summary.failed, 0);
    let inner_lines = inner.split_inclusive(|byte| *byte == b'\n').collect::<Vec<_>>();
    let footer: serde_json::Value = serde_json::from_slice(inner_lines.last().unwrap()).unwrap();
    assert_eq!(footer["event"], "run_complete");
    assert_eq!(footer["summary"]["requests"], 0);
    let limits = SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 131_072,
        max_pair_checks: 8192, max_pair_bytes: 65_536 };
    for encoding in [UntrustedEncoding::ByteFallback, UntrustedEncoding::SafeBpeV1(limits)] {
        let provenance = Provenance {
            protocol: "fnlp-candidate-int4-extract-corpus-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "extract",
            source_mode: "complete_document", chunk_schedule: "serial", cohort_chunks: None,
            root_array_union: None, root_object_reconciliation: None, max_selected_rows: None,
            input_encoding: input_encoding::reported_policy(&encoding),
            model_id: "Nanbeige4.2-3B", source_revision: "public-revision", source_root_sha256: "public-source",
            model_content_sha256: "public-model", candidate_format: "candidate-int4", quant_recipe: "recipe",
            group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None,
        };
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 16_384).unwrap();
            assert_eq!(crate::batch::run_ndjson(&mut io::Cursor::new(b""), &mut writer,
                &mut Empty, transport, &mut Continue).unwrap(), summary);
        }
        let frames = output.split_inclusive(|byte| *byte == b'\n').collect::<Vec<_>>();
        assert_eq!(frames.len(), inner_lines.len());
        for (frame, original) in frames.iter().zip(&inner_lines) {
            let value: serde_json::Value = serde_json::from_slice(frame).unwrap();
            let original_text = std::str::from_utf8(original).unwrap().trim_end_matches('\n');
            assert!(std::str::from_utf8(frame).unwrap().ends_with(&format!("\"record\":{original_text}}}\n")));
            if encoding == UntrustedEncoding::ByteFallback {
                assert!(value.get("input_encoding").is_none());
            } else {
                assert_eq!(value["input_encoding"], serde_json::to_value(&encoding).unwrap());
                assert_eq!(value["input_encoding"]["kind"], "safe_bpe_v1");
                assert_eq!(value["input_encoding"]["limits"], serde_json::to_value(limits).unwrap());
            }
        }
        // Raw precision and whitespace remain untouched under the same policy
        // envelope; this transport fixture is not an injected native result.
        let exact = b"{\"event\":\"fixture\",\"result\": {\"n\":12345678901234567890123456789012345678}}\n";
        let mut framed = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut framed, &provenance, 4096, 8192).unwrap();
            writer.write_all(exact).unwrap(); writer.flush().unwrap();
        }
        let mut suffix = b"\"record\":".to_vec();
        suffix.extend_from_slice(&exact[..exact.len() - 1]); suffix.extend_from_slice(b"}\n");
        assert!(framed.ends_with(&suffix));
    }
}
