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
