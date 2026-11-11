//! Owned host boundaries, exact memory quotes and publication provenance.
use super::*;
use super::super::tests::{command, NoIo};
use crate::batch::{BatchCode, BatchFault, BatchRunError, BatchSummary};

#[test]
fn owned_runtime_revalidates_controls_before_defaults_model_or_document_io() {
    let command = command(&["--defaults", "never-open-private-axes.json"]);
    let Int4Command { task: Int4Task::SentimentCorpus(options), mut args, batch: None, job: None } = command
        else { panic!("sentiment corpus required"); };
    let admitted = options.validate(&args).unwrap();
    args.seed = Some("ab".repeat(32));
    assert_eq!(execute(args, options, admitted, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn host_quote_preserves_native_geometry_and_adds_exact_transport_and_frozen_metadata() {
    for width in [3, 128, 10752] {
        for (axes, chunks, cohort_chunks) in [(4, 1, None), (4, 8, Some(2)),
            (4, 64, None), (1, 8, Some(2)), (2, 8, Some(1))] {
            let axes_text = axes.to_string(); let chunks_text = chunks.to_string();
            let cohort_text = cohort_chunks.map(|count| count.to_string());
            let mut flags = vec!["--max-axes", axes_text.as_str(), "--max-chunks", chunks_text.as_str()];
            if let Some(count) = &cohort_text { flags.extend(["--cohort-chunks", count.as_str()]); }
            let command = command(&flags);
            let Int4Task::SentimentCorpus(options) = command.task else { panic!("sentiment corpus required"); };
            let mut args = command.args;
            args.group_width = width; args.max_load_mib = 65536;
            let limits = options.validate(&args).unwrap();
            let task = options.document.budget_with_max_axes(limits.common, axes).unwrap();
            let host = host_limits(&args, limits, Duration::from_secs(30));
            let quote = crate::hosted::int4_sentiment_document_corpus_requirements(width, task,
                limits.document.planning, limits.document.mapping, host, axes, cohort_chunks).unwrap();
            let map = limits.document.memory;
            assert_eq!(quote.native, map.native);
            assert_eq!(quote.workspace_bytes, map.workspace_bytes);
            assert_eq!(quote.output_reservation_bytes, map.output_reservation_bytes);
            assert_eq!(quote.task_workspace_bytes, map.task_workspace_bytes);
            assert_eq!(quote.max_cohort_output_bytes, map.max_cohort_output_bytes);
            assert_eq!(quote.max_output_bytes, map.max_output_bytes);
            assert_eq!(quote.max_output_bytes + FRAME_ALLOWANCE, limits.transport.max_output_line_bytes as u64);
            assert_eq!(quote.input_bytes, map.input_bytes
                + corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap()
                + retained_metadata_bytes(chunks).unwrap());
            let execution = quote.native.kv_bytes + quote.workspace_bytes + quote.input_bytes + quote.output_reservation_bytes;
            let resident = limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES;
            assert!(limits.cli_preparation_bytes + resident + execution + CLEANUP_BYTES
                <= limits.common.minimum_memory_bytes);
            let rows = cohort_chunks.map_or(1, |chunks| chunks * axes);
            assert_eq!(quote.native.kv_bytes, limits.common.native.kv_bytes * rows as u64);
            assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
            assert_eq!(host.transport.max_work, limits.transport.max_work);
        }
    }
}

#[test]
fn cancellation_preserves_actual_causes_and_never_claims_earlier_frames_were_unpublished() {
    use DecodeCancellationKind::*;
    use crate::tasks::sentiment::grouped_int4::{Int4SentimentError, long::Int4SentimentMapError};
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = if matches!(cause, Timeout | Deadline) { CandidateError::Timeout } else { CandidateError::Execution };
        assert_eq!(host_error(HostedError::BatchSetup(BatchFault::cancelled(cause)), CandidateError::Batch), expected);
        assert_eq!(host_error(HostedError::Batch(BatchRunError {
            fault: BatchFault::cancelled(cause), summary: BatchSummary::default(),
        }), CandidateError::Batch), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::SentimentMap(
            Int4SentimentMapError::Sentiment(Int4SentimentError::Cancelled(cause)))), CandidateError::Batch), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Sentiment(Int4SentimentError::Cancelled(cause))),
            CandidateError::Planning), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            CandidateError::Model), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            CandidateError::Model), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            CandidateError::Batch), expected);
        let nested = HostedError::Scope {
            source: asupersync::cx::ScopedCpuError::WorkerCapExceeded { cap: 1 },
            physical_error: Some(Box::new(HostedError::BatchSetup(BatchFault::cancelled(cause)))),
        };
        assert_eq!(host_error(nested, CandidateError::Batch), expected);
        assert!(diagnostic(expected).contains("earlier completed frames may exist"));
    }
    assert_eq!(host_error(HostedError::BatchSetup(BatchCode::InvalidLimits.into()), CandidateError::Batch), CandidateError::Batch);
    assert_eq!(host_error(HostedError::ModelIdentity, CandidateError::Batch), CandidateError::Identity);
    assert_eq!(host_error(HostedError::ResourceDomain, CandidateError::Batch), CandidateError::Identity);
    assert_eq!(host_error(HostedError::Reservation(crate::ReservationError::ArithmeticOverflow), CandidateError::Batch), CandidateError::Memory);
    assert_eq!(completed(BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
    assert_eq!(completed(BatchSummary { succeeded: 2, ..BatchSummary::default() }), Ok(()));
}

#[test]
fn candidate_frames_preserve_native_bytes_and_report_only_the_declared_public_schedule() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    for (operation, cohort_chunks, threads) in [(None, None, 1), (Some("saved"), None, 1), (Some("loaded"), None, 1),
        (None, Some(1), 1), (Some("saved"), Some(4), 1), (Some("loaded"), Some(4), 1),
        (None, None, 2), (Some("saved"), None, 4), (Some("loaded"), None, 64)] {
        let provenance = Provenance {
            protocol: "fnlp-candidate-int4-sentiment-corpus-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment",
            sentiment_mode: "complete_document", chunk_schedule: chunk_schedule(cohort_chunks), max_axes: 4, cohort_chunks,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fixture-format", quant_recipe: "fixture-recipe:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32",
            snapshot: operation.map(|name| SnapshotRecord::new(name, need)),
            input_encoding: None, compute_schedule: compute::Schedule::for_threads(threads),
        };
        // Framing bytes are a transport fixture, never invented native scores.
        let inner = "{\"event\":\"fixture\",\"result\":{\"label\":\"café 上海\",\"single_context_equivalence\":false,\"exact\":12345678901234567890123456789012345678}}\n";
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let bytes = std::str::from_utf8(&output).unwrap();
        assert!(bytes.ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["protocol"], "fnlp-candidate-int4-sentiment-corpus-v1");
        assert_eq!(value["sentiment_mode"], "complete_document");
        assert_eq!(value["max_axes"], 4);
        if let Some(chunks) = cohort_chunks {
            assert_eq!(value["chunk_schedule"], "native_chunk_cohorts");
            assert_eq!(value["cohort_chunks"], chunks);
        } else {
            assert_eq!(value["chunk_schedule"], "serial");
            assert!(value.get("cohort_chunks").is_none());
        }
        assert_eq!(value["evidence"], "non_authoritative");
        if threads == 1 { assert!(value.get("compute_schedule").is_none()); }
        else {
            assert_eq!(value["compute_schedule"]["execution"], "int4-output-row-team-v1");
            assert_eq!(value["compute_schedule"]["declared_compute_threads"], threads);
        }
        assert!(value.get("compute").is_none(), "no native projection receipt is fabricated");
        assert_eq!(value["record"]["result"]["single_context_equivalence"], false);
        if let Some(operation) = operation {
            assert_eq!(value["snapshot"]["operation"], operation);
            assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
            assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
        } else { assert!(value.get("snapshot").is_none()); }
        for absent in ["document", "prompt", "source_dir", "source_manifest", "snapshot_path",
            "defaults_path", "prompt_digest", "taskir_digest", "policy", "eos_token_id"] {
            assert!(value.get(absent).is_none());
        }
        assert!(matches!(CandidateWriter::new(NoIo, &provenance, 4096, 1), Err(CandidateError::Output)));
        assert!(value.get("input_encoding").is_none());
        let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 262_144,
            max_output_tokens: 131_072, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 });
        let compact = Provenance { input_encoding: Some(&encoding), ..provenance };
        let compact = serde_json::to_value(&compact).unwrap();
        assert_eq!(compact["input_encoding"], serde_json::to_value(&encoding).unwrap());
    }
}

mod model;

#[test]
fn compact_host_and_cli_quotes_share_one_encoder_reservation() {
    for schedule in [vec![], vec!["--cohort-chunks", "2", "--max-axes", "2"],
        vec!["--compute-threads", "4"], vec!["--compute-threads", "64"]] {
        let mut flags = schedule;
        flags.extend(["--input-encoding", "safe-bpe-v1", "--max-tokenizer-input-bytes", "262144",
            "--max-tokenizer-output-tokens", "131072"]);
        let command = command(&flags);
        let Int4Task::SentimentCorpus(options) = command.task else { panic!("sentiment corpus required"); };
        let args = command.args; let limits = options.validate(&args).unwrap();
        let encoding = options.document.input_encoding(&args).unwrap();
        let task = options.document.budget_with_max_axes(limits.common, options.stream.max_axes).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(30));
        let quote = crate::hosted::int4_sentiment_document_corpus_requirements(args.group_width, task,
            limits.document.planning, limits.document.mapping, host, options.stream.max_axes,
            options.document.mapping.cohort_chunks).unwrap().with_input_encoding(&encoding).unwrap();
        assert_eq!(quote.input_bytes, limits.document.memory.input_bytes
            + corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap()
            + retained_metadata_bytes(limits.document.mapping.chunks.max_chunks).unwrap());
        assert_eq!(quote.native, limits.document.memory.native);
        assert_eq!(quote.workspace_bytes, limits.document.memory.workspace_bytes);
        let config = compute::resource_config(limits.common);
        assert_eq!(config.scoped_cpu_children_per_coordinator, limits.common.compute.threads - 1);
        assert_eq!(config.thread_ceiling, limits.common.compute.threads + 1);
        let execution = quote.native.kv_bytes + quote.workspace_bytes + quote.input_bytes
            + quote.output_reservation_bytes + limits.common.compute.workspace_bytes;
        assert!(limits.cli_preparation_bytes + limits.common.load.weight_payload_bytes
            + ALLOCATOR_BYTES + execution + CLEANUP_BYTES <= limits.common.minimum_memory_bytes);
    }
}

#[test]
fn runtime_rechecks_team_cohort_conflicts_before_private_io() {
    let command = command(&["--compute-threads", "4"]);
    let Int4Command { task: Int4Task::SentimentCorpus(mut options), args, batch: None, job: None } = command
        else { panic!("sentiment corpus required"); };
    let admitted = options.validate(&args).unwrap();
    options.document.mapping.cohort_chunks = Some(1);
    assert_eq!(execute(args, options, admitted, NoIo, NoIo), Err(CandidateError::Arguments));
}
