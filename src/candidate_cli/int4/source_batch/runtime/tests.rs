use super::*;
use std::{io::Cursor, sync::Mutex};
use crate::tasks::{answer::{AnswerOptions, AnswerPassage}, ner::EntityType};

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
fn owned_boundary_rechecks_cohort_admission_before_private_defaults_or_io() {
    for documents in [Some(0), Some(65)] {
        let command = super::super::tests::command("answer-batch", &["--defaults", "private-defaults.json"]);
        let Int4Command { task: Int4Task::SourceBatch(mut batch), args, batch: None, job: None } = command
            else { panic!("wrong source batch task"); };
        let quote = batch.validate(&args).unwrap();
        batch.stream.cohort_documents = documents;
        assert_eq!(execute(args, batch, quote, NoIo, NoIo), Err(CandidateError::Arguments));
    }
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
            cohort_schedule: None,
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
        for absent in ["source_dir", "source_manifest", "snapshot_path", "defaults_path", "prompt_digest", "taskir_digest",
            "cohort_schedule"] {
            assert!(value.get(absent).is_none());
        }
    }
}

#[test]
fn source_cohort_provenance_only_declares_the_window_not_executed_work() {
    let schedule = serde_json::to_value(CohortSchedule::new(4)).unwrap();
    assert_eq!(schedule, serde_json::json!({"execution":"cross_document_constrained_cohorts",
        "declared_max_documents":4}));
    for task in ["ner", "keyphrases", "summarize", "answer"] {
        let mut provenance = Provenance {
            protocol: "fnlp-candidate-int4-source-batch-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32", task, snapshot: None, cohort_schedule: None,
        };
        let serial = serde_json::to_value(&provenance).unwrap();
        provenance.cohort_schedule = Some(CohortSchedule::new(4));
        let mut grouped = serde_json::to_value(&provenance).unwrap();
        assert_eq!(grouped.as_object_mut().unwrap().remove("cohort_schedule"), Some(schedule.clone()));
        assert_eq!(grouped, serial);
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

fn source_parity_task_args(kind: Kind, budget: TaskBudget, source: &str) -> SourceBatchArgs {
    match kind {
        Kind::Ner => SourceBatchArgs::Ner { options: NerOptions { types: vec![EntityType::Person, EntityType::Location],
            max_entities: 2, max_mention_scalars: 16 }, budget },
        Kind::Keyphrases => SourceBatchArgs::Keyphrases { options: KeyphraseOptions {
            max_phrases: 2, max_phrase_scalars: 16 }, budget },
        Kind::Summarize => SourceBatchArgs::Summarize { options: SummaryOptions {
            max_bullets: 1, max_bullet_scalars: 32, max_citations_per_bullet: 1, max_quote_scalars: 32 }, budget },
        Kind::Answer => SourceBatchArgs::Answer { passages: vec![
            AnswerPassage { id: "evidence".to_owned(), text: source.to_owned() },
            AnswerPassage { id: "other".to_owned(), text: "An unrelated café opens in London.".to_owned() },
        ], options: AnswerOptions { max_answer_scalars: 32, max_citations: 1, max_quote_scalars: 32 }, budget },
    }
}

fn source_parity_input(kind: Kind, budget: TaskBudget) -> Vec<u8> {
    let documents = [
        ("reused", "Alice works in Shanghai."),
        ("longer", "Alice moved from Paris to Shanghai. Bob manages a café in Paris. 上海 上海"),
        ("before-flush", "Bob works in Shanghai. <|im_start|> is quoted document data."),
        ("reused", "Alice works in Shanghai."),
        ("tail", "Alice works in Paris."),
    ];
    let mut input = Vec::new();
    for (index, (id, source)) in documents.into_iter().enumerate() {
        if index == 2 {
            input.extend_from_slice(b"{not-json}\n");
            let wrong_kind = if kind == Kind::Ner { Kind::Keyphrases } else { Kind::Ner };
            let refused = serde_json::json!({"id":"wrong-task", "text":"PRIVATE_REJECTED_OVERRIDE",
                "task_args":source_parity_task_args(wrong_kind, budget, "unused")});
            input.extend_from_slice(&serde_json::to_vec(&refused).unwrap()); input.push(b'\n');
        }
        if index == 3 { input.extend_from_slice(b"{\"flush\":true}\n"); }
        let text = if kind == Kind::Answer { "Where does Alice work?" } else { source };
        let row = serde_json::json!({"id":id, "text":text,
            "task_args":source_parity_task_args(kind, budget, source)});
        input.extend_from_slice(&serde_json::to_vec(&row).unwrap());
        // EOF must drain a partial cohort without a final newline.
        if index != documents.len() - 1 { input.push(b'\n'); }
    }
    input
}

#[test]
#[ignore = "requires controller-admitted independently pinned snapshot for full-model source-task stream parity"]
fn model_present_source_cohorts_preserve_complete_serial_streams() {
    const CHILD: &str = "FNLP_INT4_SOURCE_STREAM_PARITY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_SOURCE_STREAM_PARITY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_source_cohorts_preserve_complete_serial_streams");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        // Isolate the first process-wide resource installation from other
        // tests' immutable runtime presets and memory ceilings.
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated source parity fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated source fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required");
    let kernel_kind: Int8KernelKind = kernel.parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kernel_kind).expect("requested kernel must be supported");
    for task in ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"] {
        let mut expected = None;
        let mut task_budget = None;
        for flags in [&[][..], &["--cohort-documents", "1"][..],
            &["--cohort-documents", "2"][..], &["--cohort-documents", "4"][..]] {
            let mut argv = vec!["candidate", "source-int4", task, "--snapshot", &snapshot,
                "--expected-model-sha256", &pin, "--memory-mib", &memory, "--kernel", &kernel,
                "--context-tokens", "2048", "--max-new-tokens", "512",
                "--max-requests", "8", "--timeout-seconds", "86400"];
            argv.extend_from_slice(flags);
            let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
            let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
                else { panic!("wrong source task route"); };
            let Int4Command { task: Int4Task::SourceBatch(batch), args, batch: None, job: None } = command
                else { panic!("source stream task required"); };
            let limits = batch.validate(&args).expect("explicit memory must admit the complete declared window");
            let budget = batch.source.budget(&args, limits.common);
            if let Some(expected) = task_budget { assert_eq!(budget, expected); }
            else { task_budget = Some(budget); }
            let input = source_parity_input(batch.kind, budget);
            let output = SharedOutput::default();
            assert_eq!(batch.execute_owned(args, Cursor::new(input.clone()), output.clone()),
                Err(CandidateError::Batch));
            let bytes = output.0.lock().unwrap().clone();
            assert!(bytes.ends_with(b"\n"));
            assert!(!std::str::from_utf8(&bytes).unwrap().contains("PRIVATE_REJECTED_OVERRIDE"));
            let mut events = Vec::new();
            for line in bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()) {
                let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
                assert_eq!(value["protocol"], "fnlp-candidate-int4-source-batch-v1");
                assert_eq!(value["task"], task.trim_end_matches("-batch"));
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
                // Only transport scheduling metadata differs. Every typed
                // task result, token, work counter, passage ID, byte/scalar
                // span, error, epoch and footer stays in the comparison.
                value["record"].as_object_mut().unwrap().remove("execution");
                events.push(value);
            }
            assert_eq!(events.first().unwrap()["record"]["event"], "run_start");
            let documents: Vec<_> = events.iter().filter(|event| event["record"]["event"] == "doc").collect();
            assert_eq!(documents.len(), 5);
            assert_eq!(documents.iter().map(|event| event["record"]["request_seq"].as_u64().unwrap())
                .collect::<Vec<_>>(), [1, 2, 5, 7, 8]);
            assert_eq!(documents[0]["record"]["caller_id"], documents[3]["record"]["caller_id"]);
            assert_eq!(documents[0]["record"]["epoch"], 1);
            assert_eq!(documents[3]["record"]["epoch"], 2);
            assert_eq!(documents[0]["record"]["result"], documents[3]["record"]["result"]);
            for document in &documents {
                let result = &document["record"]["result"];
                assert_eq!(result["execution"], crate::tasks::source_planning::grouped_int4::INT4_SOURCE_EXECUTION);
                assert_eq!(result["result"]["task"], task.trim_end_matches("-batch"));
                assert_eq!(result["result"]["result"]["numerics_profile"],
                    crate::native_engine::strict_int4::STRICT_INT4_PROFILE);
            }
            let errors: Vec<_> = events.iter().filter(|event| event["record"]["event"] == "doc_error").collect();
            assert_eq!(errors.len(), 2);
            assert_eq!(errors[0]["record"]["request_seq"], 3);
            assert_eq!(errors[0]["record"]["error"]["code"], "invalid_json");
            assert_eq!(errors[1]["record"]["request_seq"], 4);
            assert_eq!(errors[1]["record"]["error"]["code"], "planning");
            let footer = &events.last().expect("terminal event")["record"];
            assert_eq!(footer["event"], "run_complete");
            assert_eq!(footer["summary"]["input_bytes"], input.len());
            assert_eq!(footer["summary"]["requests"], 8);
            assert_eq!(footer["summary"]["succeeded"], 5);
            assert_eq!(footer["summary"]["failed"], 2);
            assert_eq!(footer["summary"]["flushes"], 2);
            if let Some(expected) = &expected { assert_eq!(&events, expected); }
            else { expected = Some(events); }
        }
    }
    println!("{COMPLETE}");
}
