use super::*;

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid owned source run read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("invalid owned source run wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("invalid owned source run flushed output"); }
}

#[test]
fn owned_boundary_rechecks_all_options_before_runtime_paths_or_io() {
    let command = super::super::tests::command("ner-batch", &[]);
    let Int4Command { task: Int4Task::SourceBatch(batch), mut args, batch: None, job: None } = command
        else { panic!("wrong source batch task"); };
    let quote = batch.validate(&args).unwrap();
    args.seed = Some("ab".repeat(32));
    assert_eq!(execute(args, batch, quote, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn every_source_task_stream_preserves_inner_spans_and_explicit_snapshot_provenance() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let provenance = Provenance {
            protocol: "fnlp-candidate-int4-source-batch-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32", task,
            snapshot: Some(SnapshotRecord::new("loaded", need)),
        };
        // Publication fixture only: it is not neural inference authority.
        let inner = "{\"event\":\"fixture\",\"text\":\"上海\",\"byte_start\":4,\"scalar_start\":3,\"exact\":12345678901234567890123456789012345678}\n";
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let bytes = std::str::from_utf8(&output).unwrap();
        assert!(bytes.ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["task"], task);
        assert_eq!(value["record"]["text"], "上海");
        assert_eq!(value["record"]["byte_start"], 4);
        assert_eq!(value["record"]["scalar_start"], 3);
        assert_eq!(value["protocol"], "fnlp-candidate-int4-source-batch-v1");
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["model_content_sha256"], "fixture-content");
        assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
        assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
        for absent in ["source_dir", "source_manifest", "snapshot_path", "defaults_path", "prompt_digest", "taskir_digest"] {
            assert!(value.get(absent).is_none());
        }
    }
}
