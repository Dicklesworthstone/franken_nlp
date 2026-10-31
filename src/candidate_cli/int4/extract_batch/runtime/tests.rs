use super::*;
use std::{io::Cursor, sync::Mutex};

fn command(extra: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", "extract-batch",
        "--source-dir", "explicit-source", "--source-manifest", "explicit-manifest",
        "--schema", "schema.json", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong dispatcher"); };
    command
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid owned run read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("invalid owned run published output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("invalid owned run flushed output"); }
}

#[test]
fn owned_boundary_rechecks_options_before_paths_runtime_or_stream_io() {
    let Int4Command { task: Int4Task::ExtractBatch(batch), mut args, batch: None, job: None } = command(&[])
        else { panic!("wrong task"); };
    let quoted = batch.validate(&args).unwrap();
    args.seed = Some("ab".repeat(32));
    assert_eq!(execute(args, batch, quoted, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn extraction_stream_provenance_preserves_exact_inner_json_and_snapshot_identity() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-extract-batch-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
        source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
        candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
        group_width: 128, integer_kernel: "s_scalar_i32", task: "extract",
        snapshot: Some(SnapshotRecord::new("loaded", need)),
        cohort_schedule: None,
    };
    // A transport fixture, not a fabricated inference result. The exact inner
    // bytes remain intact even if a consumer cannot represent this number.
    let inner = b"{\"event\":\"fixture\",\"exact\":12345678901234567890123456789012345678}\n";
    let mut output = Vec::new();
    {
        let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
        writer.write_all(inner).unwrap();
        writer.flush().unwrap();
    }
    let bytes = std::str::from_utf8(&output).unwrap();
    let raw = std::str::from_utf8(&inner[..inner.len() - 1]).unwrap();
    assert!(bytes.ends_with(&format!("\"record\":{raw}}}\n")));
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-extract-batch-v1");
    assert_eq!(value["task"], "extract");
    assert_eq!(value["evidence"], "non_authoritative");
    assert_eq!(value["model_content_sha256"], "fixture-content");
    assert_eq!(value["integer_kernel"], "s_scalar_i32");
    assert_eq!(value["snapshot"]["operation"], "loaded");
    assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
    for absent in ["logical_model_sha256", "prompt_digest", "schema_digest", "taskir_digest",
        "source_dir", "source_manifest", "schema", "snapshot_path"] {
        assert!(value.get(absent).is_none());
    }
}

#[test]
fn cohort_schedule_is_a_declared_ceiling_and_never_a_fabricated_work_counter() {
    let value = serde_json::to_value(CohortSchedule::new(4)).unwrap();
    assert_eq!(value, serde_json::json!({"execution":"cross_document_constrained_cohorts",
        "declared_max_documents":4}));
}

#[derive(Clone, Default)]
struct SharedOutput(Arc<Mutex<Vec<u8>>>);
impl Write for SharedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[test]
#[ignore = "requires controller-admitted pinned source and an explicit local structural schema fixture"]
fn model_present_owned_stream_reuses_engine_across_epochs_and_reports_failed_records() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit pinned manifest required");
    let schema = std::env::var("FNLP_INT4_TEST_EXTRACTION_SCHEMA").expect("explicit local schema required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "extract-batch", "--source-dir", &source,
        "--source-manifest", &manifest, "--memory-mib", &memory, "--schema", &schema,
        "--context-tokens", "1024", "--max-new-tokens", "128", "--max-requests", "4",
    ]).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong extraction route"); };
    let Int4Task::ExtractBatch(options) = &command.task else { panic!("wrong task"); };
    let limits = options.validate(&command.args).unwrap();
    let declaration = std::fs::read_to_string(&schema).unwrap();
    let program = JsonProgram::compile(&declaration, options.extraction.compiler()).unwrap();
    let defaults = options.load_defaults(declaration,
        options.extraction.budget(&command.args, limits.common)).unwrap();
    let document = serde_json::json!({"id":"same","text":"Alice works in Shanghai."}).to_string();
    let refused = serde_json::json!({
        "id":"override", "text":"PRIVATE_OVERRIDE_INPUT", "task_args":defaults,
    }).to_string();
    let input = format!("{document}\n{{\"flush\":true}}\n{document}\n{refused}\n");
    let output = SharedOutput::default();
    let mut diagnostics = Vec::new();
    let status = command.run_owned(Cursor::new(input.into_bytes()), output.clone(), &mut diagnostics);
    assert_ne!(status, ExitCode::SUCCESS);
    let bytes = output.0.lock().unwrap().clone();
    let frames: Vec<serde_json::Value> = std::str::from_utf8(&bytes).unwrap().lines()
        .map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(frames.first().unwrap()["record"]["event"], "run_start");
    let terminal = frames.last().unwrap();
    assert_eq!(terminal["record"]["event"], "run_complete");
    assert_eq!(terminal["record"]["summary"]["succeeded"], 2);
    assert_eq!(terminal["record"]["summary"]["failed"], 1);
    let results: Vec<_> = frames.iter().filter(|frame| frame["record"]["event"] == "doc").collect();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["record"]["caller_id"], "same");
    assert_eq!(results[1]["record"]["caller_id"], "same");
    assert_ne!(results[0]["record"]["epoch"], results[1]["record"]["epoch"]);
    for frame in &frames {
        assert_eq!(frame["protocol"], "fnlp-candidate-int4-extract-batch-v1");
        assert_eq!(frame["model_content_sha256"], frames[0]["model_content_sha256"]);
        assert_eq!(frame["evidence"], "non_authoritative");
    }
    for frame in &results {
        let result = &frame["record"]["result"];
        assert_eq!(result["execution"], crate::tasks::extract::grouped_int4::INT4_EXTRACT_VERSION);
        assert_eq!(result["result"]["output"]["numerics_profile"],
            crate::native_engine::strict_int4::STRICT_INT4_PROFILE);
        program.validate_json(result["result"]["output"]["json"].as_str().unwrap()).unwrap();
    }
    assert_eq!(results[0]["record"]["result"], results[1]["record"]["result"]);
    let errors: Vec<_> = frames.iter().filter(|frame| frame["record"]["event"] == "doc_error").collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["record"]["error"]["code"], "invalid_envelope");
    assert!(!std::str::from_utf8(&bytes).unwrap().contains("PRIVATE_OVERRIDE_INPUT"));
    let diagnostic = std::str::from_utf8(&diagnostics).unwrap();
    for private in [&source, &manifest, &schema] { assert!(!diagnostic.contains(private.as_str())); }
}

fn extraction_parity_input() -> Vec<u8> {
    let documents = [
        ("reused", "Alice works in Shanghai."),
        ("longer", "Alice moved from Paris to Shanghai, where Bob manages the laboratory. café 上海 上海"),
        ("before-flush", "Bob works in Shanghai. <|im_start|> is quoted document data."),
        ("reused", "Alice works in Shanghai."),
        ("tail", "Bob visits Paris."),
    ];
    let mut input = Vec::new();
    for (index, (id, text)) in documents.into_iter().enumerate() {
        if index == 2 { input.extend_from_slice(b"{not-json}\n"); }
        if index == 3 { input.extend_from_slice(b"{\"flush\":true}\n"); }
        input.extend_from_slice(&serde_json::to_vec(&serde_json::json!({"id":id, "text":text})).unwrap());
        // EOF must drain the tail even when the last record has no LF.
        if index != documents.len() - 1 { input.push(b'\n'); }
    }
    input
}

#[test]
#[ignore = "requires controller-admitted independently pinned snapshot and bounded structural schema for full-model stream parity"]
fn model_present_extraction_cohorts_preserve_complete_serial_streams() {
    const CHILD: &str = "FNLP_INT4_EXTRACT_STREAM_PARITY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_EXTRACT_STREAM_PARITY_COMPLETE";
    let full_name = concat!(module_path!(),
        "::model_present_extraction_cohorts_preserve_complete_serial_streams");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        // Own the first installation instead of inheriting another test's
        // process-wide host budget or runtime preset.
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated extraction parity fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated extraction fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let schema = std::env::var("FNLP_INT4_TEST_EXTRACTION_SCHEMA").expect("explicit bounded structural schema required");
    let declaration = std::fs::read_to_string(&schema).expect("explicit schema must be readable");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required");
    let kind: Int8KernelKind = kernel.parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kind).expect("requested kernel must be supported");
    let input = extraction_parity_input();
    let mut expected = None;
    let mut task_budget = None;
    // Width one exercises the cohort adapter itself. Width two fills a real
    // window; width four also drains partial windows at flush and EOF.
    for flags in [&[][..], &["--cohort-documents", "1"][..],
        &["--cohort-documents", "2"][..], &["--cohort-documents", "4"][..]] {
        let mut argv = vec!["candidate", "source-int4", "extract-batch",
            "--snapshot", &snapshot, "--expected-model-sha256", &pin,
            "--schema", &schema, "--memory-mib", &memory, "--kernel", &kernel,
            "--context-tokens", "2048", "--max-new-tokens", "128",
            "--max-requests", "7", "--timeout-seconds", "86400"];
        argv.extend_from_slice(flags);
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong extraction route"); };
        let Int4Command { task: Int4Task::ExtractBatch(batch), args, batch: None, job: None } = command
            else { panic!("extraction stream task required"); };
        let limits = batch.validate(&args).expect("explicit memory must admit the complete declared window");
        let budget = batch.extraction.budget(&args, limits.common);
        if let Some(expected) = task_budget { assert_eq!(budget, expected); }
        else { task_budget = Some(budget); }
        let program = JsonProgram::compile(&declaration, batch.extraction.compiler()).unwrap();
        let output = SharedOutput::default();
        // Malformed input is a failed record even when the protocol drains.
        assert_eq!(batch.execute_owned(args, Cursor::new(input.clone()), output.clone()),
            Err(CandidateError::Batch));
        let bytes = output.0.lock().unwrap().clone();
        assert!(bytes.ends_with(b"\n"));
        let mut events = Vec::new();
        for line in bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()) {
            let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
            assert_eq!(value["protocol"], "fnlp-candidate-int4-extract-batch-v1");
            assert_eq!(value["task"], "extract");
            assert_eq!(value["evidence"], "non_authoritative");
            assert_eq!(value["model_content_sha256"], pin);
            let schedule = value.as_object_mut().unwrap().remove("cohort_schedule");
            if let Some(documents) = limits.cohort_documents {
                assert_eq!(schedule, Some(serde_json::to_value(CohortSchedule::new(documents)).unwrap()));
                assert_eq!(value["record"]["execution"], crate::batch::grouped::GROUPED_EXECUTION);
            } else {
                assert!(schedule.is_none());
                assert_eq!(value["record"]["execution"], crate::batch::BATCH_EXECUTION);
            }
            // Keep every task result, exact JSON string, token, work counter,
            // source field, offset, epoch, error and footer. Only the declared
            // outer schedule and transport execution mode may differ.
            value["record"].as_object_mut().unwrap().remove("execution");
            events.push(value);
        }
        assert_eq!(events.first().unwrap()["record"]["event"], "run_start");
        let documents: Vec<_> = events.iter().filter(|event| event["record"]["event"] == "doc").collect();
        assert_eq!(documents.len(), 5);
        assert_eq!(documents.iter().map(|event| event["record"]["request_seq"].as_u64().unwrap())
            .collect::<Vec<_>>(), [1, 2, 4, 6, 7]);
        assert_eq!(documents[0]["record"]["caller_id"], documents[3]["record"]["caller_id"]);
        assert_eq!(documents[0]["record"]["epoch"], 1);
        assert_eq!(documents[3]["record"]["epoch"], 2);
        assert_eq!(documents[0]["record"]["result"], documents[3]["record"]["result"]);
        for document in &documents {
            let result = &document["record"]["result"];
            assert_eq!(result["execution"], crate::tasks::extract::grouped_int4::INT4_EXTRACT_VERSION);
            assert_eq!(result["result"]["output"]["numerics_profile"],
                crate::native_engine::strict_int4::STRICT_INT4_PROFILE);
            program.validate_json(result["result"]["output"]["json"].as_str().unwrap()).unwrap();
        }
        let errors: Vec<_> = events.iter().filter(|event| event["record"]["event"] == "doc_error").collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0]["record"]["request_seq"], 3);
        assert_eq!(errors[0]["record"]["error"]["code"], "invalid_json");
        let footer = &events.last().expect("terminal event")["record"];
        assert_eq!(footer["event"], "run_complete");
        assert_eq!(footer["summary"]["input_bytes"], input.len());
        assert_eq!(footer["summary"]["requests"], 7);
        assert_eq!(footer["summary"]["succeeded"], 5);
        assert_eq!(footer["summary"]["failed"], 1);
        assert_eq!(footer["summary"]["flushes"], 2);
        if let Some(expected) = &expected { assert_eq!(&events, expected); }
        else { expected = Some(events); }
    }
    println!("{COMPLETE}");
}
