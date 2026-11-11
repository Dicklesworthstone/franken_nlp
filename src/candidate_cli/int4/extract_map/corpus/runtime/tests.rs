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
#[ignore = "controller-only real-model gate: requires independently pinned snapshot, explicit memory and fresh process"]
fn persistent_document_corpus_teams_preserve_native_values_lineage_and_work_across_flushes() {
    use std::sync::Mutex;
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let schema = r#"{"type":"array","minItems":1,"maxItems":1,"items":{"type":"string","maxLength":5,"const":"Alice","x-fnlp-source":"verbatim"}}"#;
    let source = "é Alice\n".repeat(6);
    let record = serde_json::to_string(&serde_json::json!({"id":"same-document","text":source,"task_args":{}})).unwrap();
    let input = format!("{record}\n{{\"flush\":true}}\n{record}\n").into_bytes();
    let command = |threads: usize, selected: bool| {
        let width = threads.to_string();
        let mut argv = vec!["candidate", "source-int4", "extract-corpus", "--snapshot", &snapshot,
            "--expected-model-sha256", &pin, "--memory-mib", &memory,
            "--schema", "controller-inline-schema.json", "--source-membership", "--root-array-union",
            "--context-tokens", "1024", "--max-new-tokens", "32", "--max-chunk-bytes", "9",
            "--compute-threads", &width, "--timeout-seconds", "86400"];
        if selected { argv.extend(["--max-selected-rows", "166144"]); }
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("explicit extraction corpus required"); };
        command
    };
    let base = command(4, false);
    let Int4Task::ExtractCorpus(options) = &base.task else { panic!("corpus arguments required"); };
    let limits = options.validate(&base.args).unwrap();
    assert!(crate::installed_process_resources().is_none(), "model gate requires its own fresh process");
    let engine = NlpEngine::builder().resource_config(compute::resource_config(limits.common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes).unwrap().commit().unwrap();
    let model = crate::candidate_cli::int4::runtime::load_model(&engine, &base.args, limits.common,
        Duration::from_secs(3600), CancellationToken::default()).unwrap();
    let resident = engine.resources().memory_snapshot();
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls()).unwrap());
    for selected in [false, true] {
        let mut expected = None;
        for threads in [1, 2, 4] {
            let command = command(threads, selected);
            let Int4Task::ExtractCorpus(options) = command.task else { panic!("corpus task required"); };
            let limits = options.validate(&command.args).unwrap();
            let budget = options.document.extraction.budget(&command.args, limits.common);
            let request = options.load_defaults_with_control(schema.to_owned(), budget, &mut Continue).unwrap();
            let planner = Int4ExtractionDocumentBatchPlanner::for_model(model.model_identity(), budget,
                options.document.extraction.compiler(), options.document.extraction.source_limits(),
                Int4ExtractionDocumentBatchConfig { request, limits: limits.document.mapping,
                    max_model_work: limits.document.mapping.mapping.max_model_work,
                    max_mask_visits: limits.max_run_mask_visits,
                    max_verification: limits.max_run_verification }).unwrap();
            let planner = if selected {
                planner.with_selected_rows(options.document.extraction.selected_rows().unwrap().unwrap()).unwrap()
            } else { planner };
            let output = Arc::new(Mutex::new(Vec::new()));
            let host = host_limits(&command.args, limits, Duration::from_secs(86_400));
            let summary = if threads == 1 {
                engine.batch_int4_extract_documents(&model, planner, Arc::clone(&vocabulary), host,
                    limits.document.reduction_reserve_bytes, limits.common.kernel,
                    std::io::Cursor::new(input.clone()), Capture(Arc::clone(&output)), CancellationToken::default())
            } else {
                engine.batch_int4_extract_documents_with_threads(&model, planner, Arc::clone(&vocabulary), host,
                    limits.document.reduction_reserve_bytes, limits.common.kernel, threads,
                    std::io::Cursor::new(input.clone()), Capture(Arc::clone(&output)), CancellationToken::default())
            }.unwrap();
            assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 0); assert_eq!(summary.flushes, 2);
            assert!(summary.reserved_work.forward_positions > 0);
            let bytes = output.lock().unwrap().clone();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert_eq!(text.matches("\"event\":\"run_complete\"").count(), 1);
            assert!(text.contains("\"root_array_union\""));
            assert!(text.contains("\"source_span\""));
            let completed = (bytes, serde_json::to_value(summary).unwrap());
            if let Some(expected) = &expected { assert_eq!(&completed, expected); }
            else { expected = Some(completed); }
            assert_eq!(engine.resources().memory_snapshot(), resident);
            let closures = engine.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
        }
    }
    drop(vocabulary); drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
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
            compute_schedule: None,
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
        assert!(value.get("compute_schedule").is_none());
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
        for threads in [2, 4, 64] {
            let threaded = Provenance {
                    protocol: "fnlp-candidate-int4-extract-corpus-v1", schema_version: 1,
                    scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "extract",
                    source_mode: "complete_document", chunk_schedule: "serial", cohort_chunks: None,
                    root_array_union: None, root_object_reconciliation: None, max_selected_rows: Some(32),
                    input_encoding: None, compute_schedule: compute::Schedule::for_threads(threads),
                    model_id: "Nanbeige4.2-3B", source_revision: "public-revision", source_root_sha256: "public-source",
                    model_content_sha256: "public-model", candidate_format: "candidate-int4", quant_recipe: "recipe",
                    group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None,
                };
            let mut bytes = Vec::new();
            {
                let mut writer = CandidateWriter::new(&mut bytes, &threaded, 4096, 8192).unwrap();
                writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
            }
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["compute_schedule"], serde_json::json!({
                "execution": "int4-output-row-team-v1", "declared_compute_threads": threads }));
            assert_eq!(value["record"]["result"]["chunks"][0]["json"],
                "{\"n\":12345678901234567890123456789012345678}");
            assert!(std::str::from_utf8(&bytes).unwrap()
                .ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        }
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
            compute_schedule: None,
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
