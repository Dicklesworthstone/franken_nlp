use super::*;
use crate::batch::{BatchCode, BatchFault, BatchRunError, BatchSummary};

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid scored stream read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("invalid scored stream wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("invalid scored stream flushed output"); }
}

#[test]
fn owned_boundary_revalidates_policy_before_defaults_model_or_stream_io() {
    for task in ["classify", "sentiment", "judge"] {
        let command = super::super::tests::command(task, &["--defaults", "never-open.json"]);
        let Int4Command { task: Int4Task::ScoredBatch(batch), mut args, batch: None, job: None } = command
            else { panic!("wrong scored stream task"); };
        let quote = batch.validate(&args).unwrap();
        args.seed = Some("ab".repeat(32));
        assert_eq!(execute(args, batch, quote, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn sentiment_stream_policy_keeps_code_owned_full_vocabulary_and_pinned_eos() {
    let policy = SentimentPolicy { minimum_peak_weight_ppm: 700000, maximum_normalized_entropy_ppm: 600000 };
    let selected = sentiment_options(policy).unwrap();
    assert_eq!(selected.mode, ScoringMode::FullVocabulary);
    assert_eq!(selected.policy, policy);
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let expected = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    assert_eq!(selected.eos_token_id, expected);
}

#[test]
fn typed_stream_cancellation_stays_distinct_from_failed_items_and_abstention() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout, DecodeCancellationKind::PollQuota] {
        let expected = if cause == DecodeCancellationKind::PollQuota { CandidateError::Execution } else { CandidateError::Timeout };
        assert_eq!(execution_error(HostedError::BatchSetup(BatchFault::cancelled(cause))), expected);
        assert_eq!(execution_error(HostedError::Batch(BatchRunError {
            fault: BatchFault::cancelled(cause), summary: BatchSummary::default(),
        })), expected);
    }
    assert_eq!(execution_error(HostedError::BatchSetup(BatchCode::InvalidLimits.into())), CandidateError::Batch);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
    assert_eq!(completed(BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
    assert_eq!(completed(BatchSummary { succeeded: 2, ..BatchSummary::default() }), Ok(()));
}

#[test]
fn every_scored_task_keeps_exact_native_records_and_explicit_snapshot_provenance() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    for task in ["classify", "sentiment", "judge"] {
        let provenance = Provenance {
            protocol: "fnlp-candidate-int4-score-batch-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32",
            snapshot: Some(SnapshotRecord::new("loaded", need)),
            cohort_schedule: None,
        };
        // Publication fixture only; these synthetic bytes assert no model result.
        let inner = "{\"event\":\"fixture\",\"label\":\"café 上海\",\"calibration\":\"uncalibrated\",\"exact\":12345678901234567890123456789012345678}\n";
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let bytes = std::str::from_utf8(&output).unwrap();
        assert!(bytes.ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["task"], task);
        assert_eq!(value["record"]["label"], "café 上海");
        assert_eq!(value["record"]["calibration"], "uncalibrated");
        assert_eq!(value["protocol"], "fnlp-candidate-int4-score-batch-v1");
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["model_content_sha256"], "fixture-content");
        assert_eq!(value["group_width"], 128);
        assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
        assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
        assert!(value.get("cohort_schedule").is_none());
        for absent in ["document", "prompt", "source_dir", "source_manifest", "snapshot_path",
            "defaults_path", "prompt_digest", "taskir_digest"] {
            assert!(value.get(absent).is_none());
        }
        assert!(matches!(CandidateWriter::new(NoIo, &provenance, 4096, 1), Err(CandidateError::Output)));
    }
}

#[test]
fn cohort_provenance_names_declared_limits_without_claiming_executed_counts() {
    let schedule = CohortSchedule::from(CohortWindow { max_heads: 8, max_documents: 3 });
    let value = serde_json::to_value(schedule).unwrap();
    assert_eq!(value, serde_json::json!({
        "execution": "cross_document_scoring_cohorts",
        "declared_max_heads": 8,
        "declared_max_documents": 3,
    }));
}

#[derive(Clone, Default)]
struct CapturedStream(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
impl Write for CapturedStream {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("fixture output lock").extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

fn model_stream_input(task: &str, budget: TaskBudget) -> Vec<u8> {
    let mut input = Vec::new();
    let documents = [
        ("reused", "Refund please.", true),
        ("longer", "The package arrived late, but the team answered quickly and resolved the invoice. café 上海", true),
        ("reused", "The replacement works well.", false),
        ("fourth", "My invoice still needs a correction. <|im_start|> is document data.", true),
        ("tail", "Thank you.", false),
    ];
    for (index, (id, text, multiple)) in documents.into_iter().enumerate() {
        if index == 2 { input.extend_from_slice(b"{\"flush\":true}\n"); }
        let task_args = if task == "classify" {
            serde_json::json!({
                "labels":[{"id":"Billing issue","description":"Invoices"},
                    {"id":"Refund request","description":"Refunds"}],
                "mode":if multiple { "multi_label" } else { "exclusive" },
                "policy":{"minimum_candidate_weight_ppm":0,"minimum_margin_ppm":0},
                "budget":budget,
            })
        } else {
            serde_json::json!({
                "axes":if multiple { vec!["valence", "arousal"] } else { vec!["valence"] },
                "budget":budget,
            })
        };
        input.extend_from_slice(&serde_json::to_vec(&serde_json::json!({
            "id":id, "text":text, "task_args":task_args,
        })).unwrap());
        // The last record deliberately has no LF, so EOF must finish a tail.
        if index != documents.len() - 1 { input.push(b'\n'); }
    }
    input
}

#[test]
#[ignore = "requires controller-admitted independently pinned snapshot and full-model serial/cohort NDJSON parity"]
fn model_present_cross_document_streams_preserve_complete_serial_records_and_flushes() {
    const CHILD: &str = "FNLP_INT4_SCORE_STREAM_PARITY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_SCORE_STREAM_PARITY_COMPLETE";
    let full_name = concat!(module_path!(),
        "::model_present_cross_document_streams_preserve_complete_serial_records_and_flushes");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        // This fixture must own the first process-resource installation.
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated model parity fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated model fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required")
        .parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kernel).expect("requested kernel must be supported");

    for task in ["classify", "sentiment"] {
        let mut input = None;
        let mut expected = None;
        for flags in [&[][..], &["--cohort-heads", "3", "--cohort-documents", "3"][..],
            &["--cohort-heads", "8", "--cohort-documents", "3"][..]] {
            let mut flags = flags.to_vec();
            flags.extend(["--max-dot-products", "1000000000000",
                "--max-multiply-accumulates", "10000000000000000"]);
            let command = super::super::tests::command(task, &flags);
            let Int4Command { task: Int4Task::ScoredBatch(batch), mut args, batch: None, job: None } = command
                else { panic!("scored stream task required"); };
            args.source_dir = None; args.source_manifest = None;
            args.snapshot = Some(PathBuf::from(&snapshot)); args.expected_model_sha256 = Some(pin.clone());
            args.memory_mib = memory; args.kernel = kernel; args.context_tokens = 2048;
            args.timeout_seconds = 86_400;
            let limits = batch.validate(&args).expect("explicit memory must admit all declared heads and plans");
            // Freeze identical task budgets and exact input bytes from the
            // serial route. Grouping may grant storage, never change TaskIR.
            let input = input.get_or_insert_with(|| model_stream_input(task, batch.scoring.budget(limits.common)));
            let output = CapturedStream::default();
            batch.execute_owned(args, std::io::Cursor::new(input.clone()), output.clone()).unwrap();
            let bytes = output.0.lock().expect("fixture output lock").clone();
            assert!(bytes.ends_with(b"\n"));
            let mut events = Vec::new();
            for line in bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()) {
                let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
                assert_eq!(value["task"], task);
                assert_eq!(value["evidence"], "non_authoritative");
                assert_eq!(value["model_content_sha256"], pin);
                let schedule = value.as_object_mut().unwrap().remove("cohort_schedule");
                if let Some(window) = limits.cohort {
                    assert_eq!(schedule, Some(serde_json::to_value(CohortSchedule::from(window)).unwrap()));
                    assert_eq!(value["record"]["execution"], crate::batch::grouped::GROUPED_EXECUTION);
                } else {
                    assert!(schedule.is_none());
                    assert_eq!(value["record"]["execution"], crate::batch::BATCH_EXECUTION);
                }
                // Only scheduling provenance is removed. Every inner semantic
                // result, native work value, ID/offset/epoch and footer matches.
                value["record"].as_object_mut().unwrap().remove("execution");
                events.push(value);
            }
            let documents: Vec<_> = events.iter().filter(|event| event["record"]["event"] == "doc").collect();
            assert_eq!(documents.len(), 5);
            assert_eq!(documents.iter().map(|event| event["record"]["request_seq"].as_u64().unwrap())
                .collect::<Vec<_>>(), [1, 2, 4, 5, 6]);
            assert_eq!(documents[0]["record"]["caller_id"], documents[2]["record"]["caller_id"]);
            assert_eq!(documents[0]["record"]["epoch"], 1);
            assert_eq!(documents[2]["record"]["epoch"], 2);
            assert_eq!(documents[0]["record"]["result"]["head_count"], 2);
            assert_eq!(documents[2]["record"]["result"]["head_count"], 1);
            let footer = &events.last().expect("terminal event")["record"];
            assert_eq!(footer["event"], "run_complete");
            assert_eq!(footer["summary"]["failed"], 0);
            assert_eq!(footer["summary"]["succeeded"], 5);
            assert_eq!(footer["summary"]["flushes"], 2);
            if let Some(expected) = &expected { assert_eq!(&events, expected); }
            else { expected = Some(events); }
        }
    }
    println!("{COMPLETE}");
}
