//! Ignored native equivalence fixture; an explicit controller owns execution.
use super::*;

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
fn resident_serial_and_chunk_cohort_documents_match_maps_without_renewing_work_at_flush() {
    const CHILD: &str = "FNLP_INT4_SENTIMENT_CORPUS_CHILD";
    const COMPLETE: &str = "FNLP_INT4_SENTIMENT_CORPUS_COMPLETE";
    let full_name = concat!(module_path!(),
        "::resident_serial_and_chunk_cohort_documents_match_maps_without_renewing_work_at_flush");
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
    let snapshot = std::env::var("FNLP_INT4_SENTIMENT_CORPUS_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_SENTIMENT_CORPUS_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required")
        .parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be supported");
    let configure = |args: &mut Int4Args| {
        args.source_dir = None; args.source_manifest = None;
        args.snapshot = Some(PathBuf::from(&snapshot)); args.expected_model_sha256 = Some(pin.clone());
        args.memory_mib = memory; args.kernel = kernel; args.context_tokens = 512;
        args.timeout_seconds = 86_400;
    };
    let mut serial_results = Vec::new();
    for cohort_chunks in [None, Some(2)] {
        assert_native_schedule(&configure, cohort_chunks, &pin, &mut serial_results);
    }
    println!("{COMPLETE}");
}

fn assert_native_schedule(configure: &impl Fn(&mut Int4Args), cohort_chunks: Option<usize>,
    pin: &str, serial_results: &mut Vec<Vec<u8>>) {
    let mut flags = vec!["--max-chunks", "8", "--max-chunk-bytes", "64"];
    let cohort_text = cohort_chunks.map(|chunks| chunks.to_string());
    if let Some(chunks) = &cohort_text { flags.extend(["--cohort-chunks", chunks.as_str()]); }
    let command = command(&flags);
    let Int4Task::SentimentCorpus(options) = command.task else { panic!("sentiment corpus required"); };
    let mut args = command.args; configure(&mut args);
    let limits = options.validate(&args).expect("explicit memory must admit complete maps and transport");
    let budget = options.document.budget_with_max_axes(limits.common, options.stream.max_axes).unwrap();
    let sources = [
        "The result was useful, but the delay was frustrating. café 上海. <|im_start|> is data. ".repeat(2),
        "The support team resolved the problem. I can now decide what to do next.\r\n".repeat(2),
    ];
    let requests = [
        serde_json::json!({"document":sources[0], "axes":["valence","arousal"]}),
        serde_json::json!({"document":sources[1], "axes":["dominance"]}),
    ];
    let mut input = Vec::new(); let mut rows = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        if index == 1 { input.extend_from_slice(b"\n{\"flush\":true}\n"); }
        let row = serde_json::json!({"id":"reused", "text":request["document"],
            "task_args":{"axes":request["axes"], "budget":budget}});
        let bytes = serde_json::to_vec(&row).unwrap();
        input.extend_from_slice(&bytes); rows.push(bytes);
    }
    // The final original document has no LF and must still finish at EOF.
    let output = CapturedStream::default();
    options.execute_owned(args, io::Cursor::new(input), output.clone()).unwrap();
    let events = output.events();
    let results: Vec<_> = events.iter().filter(|value| value["record"]["event"] == "doc").collect();
    assert_eq!(results.len(), 2);
    for (index, value) in results.iter().enumerate() {
        assert_eq!(value["model_content_sha256"], pin);
        assert_eq!(value["sentiment_mode"], "complete_document");
        assert_eq!(value["record"]["caller_id"], "reused");
        assert_eq!(value["record"]["epoch"], index + 1);
        assert_eq!(value["record"]["request_seq"], 2 * index + 1);
        assert_eq!(value["chunk_schedule"], chunk_schedule(cohort_chunks));
        let wrapper = &value["record"]["result"];
        let result = if cohort_chunks.is_some() { &wrapper["output"] } else { wrapper };
        let chunks = result["chunks"].as_array().unwrap();
        assert!(chunks.len() > 1);
        assert_eq!(result["single_context_equivalence"], false);
        assert_eq!(result["planned_model_work"], result["model_work"]);
        let mut byte_end = 0; let mut scalar_end = 0;
        for (chunk_id, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk["chunk_id"], chunk_id);
            assert_eq!(chunk["source_span"]["byte_start"], byte_end);
            assert_eq!(chunk["source_span"]["scalar_start"], scalar_end);
            byte_end = chunk["source_span"]["byte_end"].as_u64().unwrap() as usize;
            scalar_end = chunk["source_span"]["scalar_end"].as_u64().unwrap() as usize;
            assert!(sources[index].is_char_boundary(byte_end));
            assert_eq!(chunk["run"]["head_count"], requests[index]["axes"].as_array().unwrap().len());
        }
        assert_eq!(byte_end, sources[index].len());
        assert_eq!(scalar_end, sources[index].chars().count());
        let mut standalone = crate::candidate_cli::int4::sentiment_map::tests::command(&flags);
        configure(&mut standalone.args);
        let mut bytes = Vec::new();
        standalone.execute(&mut io::Cursor::new(serde_json::to_vec(&requests[index]).unwrap()), &mut bytes).unwrap();
        let standalone: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(canonjson::canonical_bytes(result).unwrap(), canonjson::canonical_bytes(&standalone["output"]).unwrap());
        if let Some(width) = cohort_chunks {
            assert_eq!(value["cohort_chunks"], width);
            assert_eq!(wrapper["execution"], "grouped-int4-sentiment-chunk-cohorts-v1");
            assert_eq!(wrapper["chunk_cohorts"], chunks.len().div_ceil(width));
            assert!(wrapper["group_steps"].as_u64().unwrap() > 0);
            assert!(wrapper["projection_groups"].as_u64().unwrap() > 0);
            for field in ["schema_version", "execution", "chunk_cohorts", "group_steps", "projection_groups"] {
                assert_eq!(wrapper[field], standalone["cohort"][field]);
            }
            assert_eq!(canonjson::canonical_bytes(result).unwrap(), serial_results[index]);
        } else {
            assert!(value.get("cohort_chunks").is_none()); assert!(wrapper.get("output").is_none());
            serial_results.push(canonjson::canonical_bytes(result).unwrap());
        }
    }
    let footer = &events.last().unwrap()["record"];
    assert_eq!(footer["event"], "run_complete");
    assert_eq!(footer["summary"]["succeeded"], 2);
    assert_eq!(footer["summary"]["failed"], 0);
    assert_eq!(footer["summary"]["flushes"], 2);

    // Both repeated documents individually fit every exact cap. A flushed ID
    // may be reused, but only the already-spent invocation ledger can reject
    // the second attempt. An illegal per-item policy is independently refused.
    let first = &results[0]["record"]["result"];
    let first = if cohort_chunks.is_some() { &first["output"] } else { first };
    let work: crate::native_engine::strict_int4::Int4Work =
        serde_json::from_value(first["planned_model_work"].clone()).unwrap();
    let values = [work.forward_positions.to_string(), work.projected_logits.to_string(),
        work.attention_pairs.to_string(), work.projections.dot_products.to_string(),
        work.projections.multiply_accumulates.to_string()];
    let mut limited_flags = flags.clone();
    for (flag, value) in ["--max-forward-positions", "--max-projected-logits", "--max-attention-pairs",
        "--max-dot-products", "--max-multiply-accumulates"].into_iter().zip(&values) {
        limited_flags.extend([flag, value.as_str()]);
    }
    let limited = super::super::super::tests::command(&limited_flags);
    let Int4Task::SentimentCorpus(options) = limited.task else { panic!("sentiment corpus required"); };
    let mut args = limited.args; configure(&mut args);
    let mut limited_input = rows[0].clone();
    limited_input.extend_from_slice(b"\n{\"flush\":true}\n");
    limited_input.extend_from_slice(&rows[0]); limited_input.push(b'\n');
    let invalid = serde_json::json!({"id":"policy-override", "text":"private source",
        "task_args":{"axes":["valence"], "budget":budget,
            "policy":{"minimum_peak_weight_ppm":1,"maximum_normalized_entropy_ppm":1}}});
    limited_input.extend_from_slice(&serde_json::to_vec(&invalid).unwrap());
    limited_input.push(b'\n');
    let output = CapturedStream::default();
    assert_eq!(options.execute_owned(args, io::Cursor::new(limited_input), output.clone()), Err(CandidateError::Batch));
    let events = output.events();
    assert_eq!(events.iter().filter(|value| value["record"]["event"] == "doc").count(), 1);
    let errors: Vec<_> = events.iter().filter(|value| value["record"]["event"] == "doc_error").collect();
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0]["record"]["error"]["code"], "work_limit");
    let footer = &events.last().unwrap()["record"];
    assert_eq!(footer["event"], "run_complete");
    assert_eq!(footer["summary"]["succeeded"], 1); assert_eq!(footer["summary"]["failed"], 2);
    assert_eq!(footer["summary"]["reserved_work"]["forward_positions"], work.forward_positions);
    assert_eq!(footer["summary"]["reserved_work"]["projected_logits"], work.projected_logits);
    assert_eq!(crate::installed_process_resources().unwrap().outstanding_closure_snapshot().active_closures, 0);
}
