//! Owned host boundaries and publication contracts; model parity is explicit.
use super::*;
use super::super::tests::{command, NoIo};
use crate::batch::{BatchCode, BatchFault, BatchRunError, BatchSummary};

#[test]
fn owned_runtime_revalidates_controls_before_defaults_model_or_document_io() {
    let command = command(&["--defaults", "never-open-private-labels.json"]);
    let Int4Command { task: Int4Task::ClassifyCorpus(options), mut args, batch: None, job: None } = command
        else { panic!("classification corpus required"); };
    let admitted = options.validate(&args).unwrap();
    args.seed = Some("ab".repeat(32));
    assert_eq!(execute(args, options, admitted, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn host_quote_keeps_the_full_map_and_exact_transport_metadata_envelope() {
    for width in [3, 128, 10752] {
        for (chunks, cohort_chunks, max_labels) in [(1, None, 128), (1, Some(1), 128),
            (8, None, 128), (8, Some(4), 128), (64, None, 128), (64, Some(4), 128), (8, Some(2), 2)] {
            let chunks_text = chunks.to_string();
            let cohort_text = cohort_chunks.map(|count| count.to_string());
            let labels_text = max_labels.to_string();
            let mut flags = vec!["--max-chunks", chunks_text.as_str(), "--max-labels", labels_text.as_str()];
            if let Some(count) = &cohort_text { flags.extend(["--cohort-chunks", count.as_str()]); }
            let command = command(&flags);
            let Int4Task::ClassifyCorpus(options) = command.task else { panic!("classification corpus required"); };
            let mut args = command.args;
            args.group_width = width; args.max_load_mib = 65536;
            let limits = options.validate(&args).unwrap();
            let task = options.document.budget(limits.common, limits.document.planning.max_labels).unwrap();
            let host = host_limits(&args, limits, Duration::from_secs(30));
            let quote = memory_quote(width, task, limits, host, cohort_chunks).unwrap();
            assert_eq!(quote.native, limits.document.memory.native);
            assert_eq!(quote.workspace_bytes, limits.document.memory.workspace_bytes);
            assert_eq!(quote.output_reservation_bytes, limits.document.memory.output_reservation_bytes);
            assert_eq!(quote.max_output_bytes, limits.document.memory.max_output_bytes);
            assert_eq!(quote.max_output_bytes + FRAME_ALLOWANCE, limits.transport.max_output_line_bytes as u64);
            assert_eq!(quote.input_bytes, limits.document.memory.input_bytes
                + corpus_buffer_bytes(limits.transport, limits.common.preparation_bytes, limits.io_bytes).unwrap()
                + retained_metadata_bytes(chunks).unwrap());
            let execution = quote.native.kv_bytes + quote.workspace_bytes + quote.input_bytes + quote.output_reservation_bytes;
            let resident = limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES;
            assert!(limits.cli_preparation_bytes + resident + execution + CLEANUP_BYTES
                <= limits.common.minimum_memory_bytes,
                "every simultaneous host and CLI owner must fit the pre-IO ledger floor");
            assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
            assert_eq!(host.transport.max_work, limits.transport.max_work);
        }
    }
}

#[test]
fn cancellation_and_partial_corpus_diagnostics_retain_their_actual_scope() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = if matches!(cause, Timeout | Deadline) { CandidateError::Timeout } else { CandidateError::Execution };
        assert_eq!(host_error(HostedError::BatchSetup(BatchFault::cancelled(cause)), CandidateError::Batch), expected);
        assert_eq!(host_error(HostedError::Batch(BatchRunError {
            fault: BatchFault::cancelled(cause), summary: BatchSummary::default(),
        }), CandidateError::Batch), expected);
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
fn candidate_frames_preserve_complete_native_record_bytes_and_private_paths_stay_absent() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    for (operation, cohort_chunks) in [(None, None), (Some("saved"), None), (Some("loaded"), None),
        (None, Some(1)), (Some("saved"), Some(4)), (Some("loaded"), Some(4))] {
        let provenance = Provenance {
            protocol: "fnlp-candidate-int4-classify-corpus-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "classify",
            classification_mode: "complete_document", chunk_schedule: chunk_schedule(cohort_chunks), cohort_chunks,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fixture-format", quant_recipe: "fixture-recipe:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32",
            snapshot: operation.map(|name| SnapshotRecord::new(name, need)),
        };
        // These bytes test framing, not fabricated native model behavior.
        let inner = "{\"event\":\"fixture\",\"result\":{\"chunks\":[{\"label\":\"café 上海\"}],\"single_context_equivalence\":false,\"exact\":12345678901234567890123456789012345678}}\n";
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let bytes = std::str::from_utf8(&output).unwrap();
        assert!(bytes.ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["protocol"], "fnlp-candidate-int4-classify-corpus-v1");
        assert_eq!(value["classification_mode"], "complete_document");
        if let Some(chunks) = cohort_chunks {
            assert_eq!(value["chunk_schedule"], "native_chunk_cohorts");
            assert_eq!(value["cohort_chunks"], chunks);
        } else {
            assert_eq!(value["chunk_schedule"], "serial");
            assert!(value.get("cohort_chunks").is_none(), "serial provenance stays wire-compatible");
        }
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["record"]["result"]["single_context_equivalence"], false);
        if let Some(operation) = operation {
            assert_eq!(value["snapshot"]["operation"], operation);
            assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
            assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
        } else { assert!(value.get("snapshot").is_none()); }
        for absent in ["document", "prompt", "source_dir", "source_manifest", "snapshot_path",
            "defaults_path", "prompt_digest", "taskir_digest", "cohort_schedule"] {
            assert!(value.get(absent).is_none());
        }
        assert!(matches!(CandidateWriter::new(NoIo, &provenance, 4096, 1), Err(CandidateError::Output)));
    }
}

#[derive(Clone, Default)]
struct CapturedStream(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
impl Write for CapturedStream {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("fixture output lock").extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
impl CapturedStream {
    fn events(&self) -> Vec<serde_json::Value> {
        let bytes = self.0.lock().expect("fixture output lock");
        assert!(bytes.ends_with(b"\n"));
        bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap()).collect()
    }
}

#[test]
#[ignore = "requires controller-admitted pinned snapshot and native complete-document corpus/map equivalence"]
fn model_present_resident_documents_match_maps_and_flush_does_not_renew_work() {
    const CHILD: &str = "FNLP_INT4_CLASSIFICATION_CORPUS_CHILD";
    const COMPLETE: &str = "FNLP_INT4_CLASSIFICATION_CORPUS_COMPLETE";
    let full_name = concat!(module_path!(),
        "::model_present_resident_documents_match_maps_and_flush_does_not_renew_work");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated native fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated model fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let snapshot = std::env::var("FNLP_INT4_CLASSIFICATION_CORPUS_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_CLASSIFICATION_CORPUS_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required")
        .parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be supported");
    let configure = |args: &mut Int4Args| {
        args.source_dir = None; args.source_manifest = None;
        args.snapshot = Some(PathBuf::from(&snapshot)); args.expected_model_sha256 = Some(pin.clone());
        args.memory_mib = memory; args.kernel = kernel; args.context_tokens = 2048;
        args.timeout_seconds = 86_400;
    };
    let mut serial_results = Vec::new();
    for cohort_chunks in [None, Some(2)] {
        assert_native_corpus_schedule(&configure, cohort_chunks, &pin, &mut serial_results);
    }
    println!("{COMPLETE}");
}

fn assert_native_corpus_schedule(configure: &impl Fn(&mut Int4Args), cohort_chunks: Option<usize>,
    pin: &str, serial_results: &mut Vec<Vec<u8>>) {
    let mut flags = vec!["--max-chunks", "8", "--max-chunk-bytes", "64"];
    let cohort_text = cohort_chunks.map(|chunks| chunks.to_string());
    if let Some(chunks) = &cohort_text { flags.extend(["--cohort-chunks", chunks.as_str()]); }
    let standalone_flags = flags.clone();
    flags.extend(["--max-labels", "2"]);
    let command = command(&flags);
    let Int4Task::ClassifyCorpus(options) = command.task else { panic!("classification corpus required"); };
    let mut args = command.args; configure(&mut args);
    let limits = options.validate(&args).expect("explicit memory must admit complete maps and transport");
    let budget = options.document.budget(limits.common, limits.document.planning.max_labels).unwrap();
    let documents = [
        ("The invoice needs a refund. café 上海. <|im_start|> is data. ".repeat(2), "exclusive"),
        ("The package arrived late. The support team answered quickly and corrected the invoice. ".repeat(2), "multi_label"),
    ];
    let requests: Vec<_> = documents.iter().map(|(text, mode)| serde_json::json!({
        "document":text, "mode":mode,
        "labels":[{"id":"Billing issue","description":"Invoices"}, {"id":"Delivery issue","description":"Packages"}],
        "policy":{"minimum_candidate_weight_ppm":0,"minimum_margin_ppm":0},
    })).collect();
    let mut input = Vec::new();
    let mut rows = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        if index == 1 { input.extend_from_slice(b"\n{\"flush\":true}\n"); }
        let row = serde_json::json!({"id":"reused", "text":request["document"], "task_args":{
            "labels":request["labels"], "mode":request["mode"], "policy":request["policy"], "budget":budget,
        }});
        let bytes = serde_json::to_vec(&row).unwrap();
        input.extend_from_slice(&bytes); rows.push(bytes);
    }
    // EOF must finish the final record even without a newline.
    let output = CapturedStream::default();
    options.execute_owned(args, io::Cursor::new(input.clone()), output.clone()).unwrap();
    let events = output.events();
    let results: Vec<_> = events.iter().filter(|value| value["record"]["event"] == "doc").collect();
    assert_eq!(results.len(), 2);
    for (index, value) in results.iter().enumerate() {
        assert_eq!(value["model_content_sha256"], pin);
        assert_eq!(value["classification_mode"], "complete_document");
        assert_eq!(value["record"]["caller_id"], "reused");
        assert_eq!(value["record"]["epoch"], index + 1);
        assert_eq!(value["record"]["request_seq"], 2 * index + 1);
        assert_eq!(value["chunk_schedule"], chunk_schedule(cohort_chunks));
        let wrapper = &value["record"]["result"];
        let result = if cohort_chunks.is_some() { &wrapper["output"] } else { wrapper };
        assert!(result["chunks"].as_array().unwrap().len() > 1);
        assert_eq!(result["single_context_equivalence"], false);
        let mut standalone = crate::candidate_cli::int4::classify_map::tests::command(&standalone_flags);
        configure(&mut standalone.args);
        let mut bytes = Vec::new();
        standalone.execute(&mut io::Cursor::new(serde_json::to_vec(&requests[index]).unwrap()), &mut bytes).unwrap();
        let standalone: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(canonjson::canonical_bytes(result).unwrap(),
            canonjson::canonical_bytes(&standalone["output"]).unwrap());
        if let Some(chunks) = cohort_chunks {
            assert_eq!(value["cohort_chunks"], chunks);
            assert_eq!(wrapper["execution"], "grouped-int4-classification-chunk-cohorts-v1");
            assert_eq!(wrapper["chunk_cohorts"], result["chunks"].as_array().unwrap().len().div_ceil(chunks));
            assert!(wrapper["group_steps"].as_u64().unwrap() > 0);
            assert!(wrapper["projection_groups"].as_u64().unwrap() > 0);
            for field in ["schema_version", "execution", "chunk_cohorts", "group_steps", "projection_groups"] {
                assert_eq!(wrapper[field], standalone["cohort"][field]);
            }
            assert_eq!(canonjson::canonical_bytes(result).unwrap(), serial_results[index],
                "shared layers must preserve every original chunk score and aggregate");
        } else {
            assert!(value.get("cohort_chunks").is_none());
            assert!(wrapper.get("output").is_none());
            serial_results.push(canonjson::canonical_bytes(result).unwrap());
        }
    }
    let footer = &events.last().unwrap()["record"];
    assert_eq!(footer["event"], "run_complete");
    assert_eq!(footer["summary"]["succeeded"], 2);
    assert_eq!(footer["summary"]["failed"], 0);
    assert_eq!(footer["summary"]["flushes"], 2);

    // Set every native axis to the first document's exact work. The drained
    // flush can reuse the ID, but it cannot fund the second document. A later
    // malformed row remains a separate ordered error and never erases doc 1.
    let first = &results[0]["record"]["result"];
    let first = if cohort_chunks.is_some() { &first["output"] } else { first };
    let first = &first["planned_model_work"];
    let work: crate::native_engine::strict_int4::Int4Work = serde_json::from_value(first.clone()).unwrap();
    let values = [work.forward_positions.to_string(), work.projected_logits.to_string(),
        work.attention_pairs.to_string(), work.projections.dot_products.to_string(),
        work.projections.multiply_accumulates.to_string()];
    let mut limited_flags = flags.to_vec();
    for (flag, value) in ["--max-forward-positions", "--max-projected-logits", "--max-attention-pairs",
        "--max-dot-products", "--max-multiply-accumulates"].into_iter().zip(&values) {
        limited_flags.extend([flag, value.as_str()]);
    }
    let limited = super::super::tests::command(&limited_flags);
    let Int4Task::ClassifyCorpus(options) = limited.task else { panic!("classification corpus required"); };
    let mut args = limited.args; configure(&mut args);
    // Repeat the SAME first document so both items individually fit all five
    // exact caps. Only the spent whole-run ledger can reject the second one.
    let mut limited_input = rows[0].clone();
    limited_input.extend_from_slice(b"\n{\"flush\":true}\n");
    limited_input.extend_from_slice(&rows[0]);
    limited_input.extend_from_slice(b"\n{\"id\":\"malformed\",\"text\":false}\n");
    let output = CapturedStream::default();
    assert_eq!(options.execute_owned(args, io::Cursor::new(limited_input), output.clone()), Err(CandidateError::Batch));
    let events = output.events();
    assert_eq!(events.iter().filter(|value| value["record"]["event"] == "doc").count(), 1);
    let errors: Vec<_> = events.iter().filter(|value| value["record"]["event"] == "doc_error").collect();
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0]["record"]["error"]["code"], "work_limit");
    let footer = &events.last().unwrap()["record"];
    assert_eq!(footer["event"], "run_complete");
    assert_eq!(footer["summary"]["succeeded"], 1);
    assert_eq!(footer["summary"]["failed"], 2);
    assert_eq!(footer["summary"]["reserved_work"]["forward_positions"], work.forward_positions);
    assert_eq!(footer["summary"]["reserved_work"]["projected_logits"], work.projected_logits);
    let resources = crate::installed_process_resources().unwrap();
    assert_eq!(resources.outstanding_closure_snapshot().active_closures, 0);
}
