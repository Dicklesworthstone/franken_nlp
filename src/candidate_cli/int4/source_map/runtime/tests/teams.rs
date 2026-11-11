//! Explicit owned-model parity gate; never a substitute for a DSR receipt.
use super::*;
use crate::{MemorySnapshot, hosted::Int4ComputeReceipt};

#[test]
#[ignore = "requires controller-admitted independently pinned INT4 snapshot and explicit kernel/memory; compares serial/2/4-thread complete documents"]
fn model_present_source_document_teams_match_serial_and_release_all_guards() {
    const CHILD: &str = "FNLP_INT4_SOURCE_DOCUMENT_TEAM_CHILD";
    const COMPLETE: &str = "FNLP_INT4_SOURCE_DOCUMENT_TEAM_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_source_document_teams_match_serial_and_release_all_guards");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated source-document team gate must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated source-document team gate failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact filter ran no completed model gate");
        return;
    }
    assert!(crate::installed_process_resources().is_none(), "model gate requires a fresh process");
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB")
        .expect("explicit memory ceiling required").parse().expect("memory MiB must be an integer");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit hardware-supported kernel required");
    let width = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").unwrap_or_else(|_| "128".to_owned());
    let make_command = |kind, mode: &[&str]| {
        let mut flags = vec!["--compute-threads", "4", "--context-tokens", "2048",
            "--max-new-tokens", "512", "--max-chunk-bytes", "32", "--max-chunks", "16",
            "--group-width", &width, "--kernel", &kernel, "--timeout-seconds", "86400"];
        flags.extend_from_slice(mode);
        let mut command = crate::candidate_cli::int4::source_map::tests::command(kind, &flags);
        command.args.source_dir = None; command.args.source_manifest = None;
        command.args.snapshot = Some(PathBuf::from(&snapshot));
        command.args.expected_model_sha256 = Some(pin.clone());
        command.args.memory_mib = memory;
        command
    };
    let command = make_command(Kind::Ner, &[]);
    let Int4Task::SourceMap(options) = command.task else { panic!("source map required"); };
    let common = options.validate(&command.args).expect("explicit memory must admit the complete document").common;
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .unwrap().commit().unwrap();
    let model = crate::candidate_cli::int4::runtime::load_model(&engine, &command.args, common,
        Duration::from_secs(3600), CancellationToken::default()).unwrap();
    let (planner, vocabulary) = source::planner().unwrap();
    let planner = Arc::new(planner); let vocabulary = Arc::new(vocabulary);
    let resident = engine.resources().memory_snapshot();
    let document = "Alice met Bob in 上海. Café Paris welcomed Alice.\r\n".repeat(2);

    for (kind, mode) in [(Kind::Ner, &[][..]), (Kind::Keyphrases, &[][..]),
        (Kind::Keyphrases, &["--reduce-keyphrases", "--document-keyphrases", "3"][..]),
        (Kind::Summarize, &["--max-summary-bullets", "3"][..])] {
        let command = make_command(kind, mode);
        let Int4Task::SourceMap(options) = command.task else { panic!("source map required"); };
        let limits = options.validate(&command.args).unwrap();
        assert_eq!(limits.common.compute.threads, 4);
        assert_eq!(limits.common.preparation_bytes, common.preparation_bytes);
        let budget = options.source.budget(&command.args, limits.common);
        let identity = source::identity(model.model_identity(), &planner, source::task(kind)).unwrap();
        let config = || {
            let (_, task) = options.request(document.clone(), None, budget, command.args.max_input_bytes).unwrap();
            configuration(identity.clone(), task, &command.args, &options, limits, Duration::from_secs(86_400))
        };
        let expected = if let Some(aggregation) = limits.keyphrases {
            keyphrase_preflight(&document, &planner, &config(), model.model_identity().group_width(),
                aggregation, &mut continuing()).unwrap()
        } else {
            preflight(&document, &planner, &config(), model.model_identity().group_width(),
                limits.summary, &mut continuing()).unwrap()
        };
        assert!(expected.chunk_count() > 1);
        assert_eq!(expected.source_span().byte_end, document.len());
        assert_eq!(expected.source_span().scalar_end, document.chars().count());
        let mut serial_bytes = None;
        for threads in [1, 2, 4] {
            let bytes = if let Some(aggregation) = limits.keyphrases {
                let output = if threads == 1 {
                    compute::Completion::serial(engine.reduce_int4_keyphrases_document(&model, document.clone(),
                        Arc::clone(&planner), Arc::clone(&vocabulary), config(), aggregation, common.kernel,
                        CancellationToken::default()).unwrap())
                } else {
                    compute::Completion::Team(engine.reduce_int4_keyphrases_document_with_threads(&model,
                        document.clone(), Arc::clone(&planner), Arc::clone(&vocabulary), config(), aggregation,
                        common.kernel, threads, CancellationToken::default()).unwrap())
                };
                expected.verify_keyphrases_completed_with_limits(output.result(), aggregation).unwrap();
                assert_eq!(output.result().requested_max_phrases, 3);
                inspect_completion(&engine, output, expected, &document, threads, &resident)
            } else if let Some(summary) = limits.summary {
                let output = if threads == 1 {
                    compute::Completion::serial(engine.summarize_int4_source(&model, document.clone(),
                        Arc::clone(&planner), Arc::clone(&vocabulary), config(), summary, common.kernel,
                        CancellationToken::default()).unwrap())
                } else {
                    compute::Completion::Team(engine.summarize_int4_source_with_threads(&model, document.clone(),
                        Arc::clone(&planner), Arc::clone(&vocabulary), config(), summary, common.kernel,
                        threads, CancellationToken::default()).unwrap())
                };
                expected.verify_summary_completed_with_limits(output.result(), summary).unwrap();
                inspect_completion(&engine, output, expected, &document, threads, &resident)
            } else {
                let output = if threads == 1 {
                    compute::Completion::serial(engine.map_int4_source(&model, document.clone(), Arc::clone(&planner),
                        Arc::clone(&vocabulary), config(), common.kernel, CancellationToken::default()).unwrap())
                } else {
                    compute::Completion::Team(engine.map_int4_source_with_threads(&model, document.clone(),
                        Arc::clone(&planner), Arc::clone(&vocabulary), config(), common.kernel, threads,
                        CancellationToken::default()).unwrap())
                };
                expected.verify_completed(output.result()).unwrap();
                inspect_completion(&engine, output, expected, &document, threads, &resident)
            };
            // No semantic field is scrubbed: native tokens/work, all original
            // spans, reducer counts, ranks and verification charges must match.
            if let Some(serial) = &serial_bytes { assert_eq!(&bytes, serial); }
            else { serial_bytes = Some(bytes); }
        }
    }
    // QA has a distinct complete-document seal and retains blank source
    // ranges without inventing native work for them. Exercise that driver
    // separately, under the same installed four-lane process envelope.
    let command = make_command(Kind::Answer, &["--question", "private-question.txt"]);
    let Int4Task::SourceMap(options) = command.task else { panic!("document question required"); };
    let limits = options.validate(&command.args).unwrap();
    assert_eq!(limits.common.preparation_bytes, common.preparation_bytes);
    let identity = source::identity(model.model_identity(), &planner, source::task(Kind::Answer)).unwrap();
    let config = || configuration(identity.clone(), options.question.as_ref().unwrap()
        .task("Where did Alice meet Bob?".to_owned(), None).unwrap(),
        &command.args, &options, limits, Duration::from_secs(86_400));
    let document = format!("{document}{}Bob visits Paris.\r\n", " ".repeat(96));
    let expected = question_preflight(&document, &planner, &config(), model.model_identity().group_width(),
        &mut continuing()).unwrap();
    assert!(expected.native_chunks() > 1);
    assert!(expected.whitespace_chunks() > 0);
    assert_eq!(expected.source_span().byte_end, document.len());
    assert_eq!(expected.source_span().scalar_end, document.chars().count());
    let mut serial_bytes = None;
    for threads in [1, 2, 4] {
        let output = if threads == 1 {
            compute::Completion::serial(engine.answer_int4_document(&model, document.clone(),
                Arc::clone(&planner), Arc::clone(&vocabulary), config(), common.kernel,
                CancellationToken::default()).unwrap())
        } else {
            compute::Completion::Team(engine.answer_int4_document_with_threads(&model, document.clone(),
                Arc::clone(&planner), Arc::clone(&vocabulary), config(), common.kernel, threads,
                CancellationToken::default()).unwrap())
        };
        expected.verify_completed(output.result()).unwrap();
        let held = engine.resources().memory_snapshot();
        assert!(held.committed_bytes > resident.committed_bytes);
        let bytes = serde_json::to_vec(output.result()).unwrap();
        if let Some(serial) = &serial_bytes { assert_eq!(&bytes, serial); } else { serial_bytes = Some(bytes); }
        if threads == 1 { assert!(output.compute().is_none()); }
        else {
            let receipt = output.compute().unwrap();
            assert_eq!(receipt.compute_threads, threads);
            assert!(receipt.completed_projections > expected.native_chunks() as u64);
            assert!(receipt.completed_worker_rows > 0);
            assert!(receipt.completed_rows > receipt.completed_worker_rows);
        }
        assert_eq!(engine.resources().memory_snapshot(), held);
        drop(output);
        assert_eq!(engine.resources().memory_snapshot(), resident);
        assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
    }
    for cause in [DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::PollQuota] {
        let cancellation = CancellationToken::default();
        let mut config = config();
        if cause == DecodeCancellationKind::PollQuota { config.native.run.max_checkpoints = 2; }
        else { cancellation.cancel(cause); }
        let error = engine.answer_int4_document_with_threads(&model, document.clone(),
            Arc::clone(&planner), Arc::clone(&vocabulary), config, common.kernel, 4, cancellation)
            .err().expect("stopped QA must not return an answer, abstention or completed prefix");
        assert_eq!(stopped_kind(error), cause);
        assert_eq!(engine.resources().memory_snapshot(), resident);
        assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
    }
    drop(vocabulary); drop(planner); drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
    println!("{COMPLETE}");
}

fn stopped_kind(error: HostedError) -> DecodeCancellationKind {
    match error {
        HostedError::Stopped { stop, .. } => stop.kind,
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => stopped_kind(*error),
        _ => panic!("expected the original hosted cancellation cause"),
    }
}

fn inspect_completion<T: Serialize>(engine: &NlpEngine, output: compute::Completion<T>,
    expected: Int4SourceMapPreflight, source: &str, threads: usize, resident: &MemorySnapshot) -> Vec<u8> {
    let held = engine.resources().memory_snapshot();
    assert!(held.committed_bytes > resident.committed_bytes, "complete output must retain its memory guard");
    let bytes = serde_json::to_vec(output.result()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["planned_model_work"], serde_json::to_value(expected.planned_work()).unwrap());
    assert!(value["model_work"]["forward_positions"].as_u64().unwrap() > 0);
    assert!(value["model_work"]["forward_positions"].as_u64().unwrap() <= expected.planned_work().forward_positions);
    let span = value.get("source_span").unwrap_or(&value["mapped"]["root"]["source_span"]);
    assert_eq!(span["byte_start"], 0); assert_eq!(span["scalar_start"], 0);
    assert_eq!(span["byte_end"], source.len()); assert_eq!(span["scalar_end"], source.chars().count());
    if let Some(mapped) = value.get("mapped") {
        let chunks = mapped["root"]["value"].as_array().unwrap();
        assert_eq!(chunks.len(), expected.chunk_count());
        let (mut bytes, mut scalars) = (0, 0);
        for (index, chunk) in chunks.iter().enumerate() {
            let span = &chunk["source_span"];
            assert_eq!(chunk["chunk_id"], index);
            assert_eq!(span["byte_start"], bytes); assert_eq!(span["scalar_start"], scalars);
            bytes = span["byte_end"].as_u64().unwrap() as usize;
            scalars = span["scalar_end"].as_u64().unwrap() as usize;
            assert!(source.is_char_boundary(bytes));
            assert_eq!(scalars, source[..bytes].chars().count());
        }
        assert_eq!((bytes, scalars), (source.len(), source.chars().count()));
    }
    if threads == 1 { assert!(output.compute().is_none()); }
    else {
        let receipt = output.compute().expect("actual projection counters required");
        assert_eq!(receipt.schedule, "int4-output-row-team-v1");
        assert_eq!(receipt.compute_threads, threads);
        assert!(receipt.completed_projections > expected.chunk_count() as u64);
        assert!(receipt.completed_worker_rows > 0);
        assert!(receipt.completed_rows > receipt.completed_worker_rows);
    }
    assert_eq!(engine.resources().memory_snapshot(), held, "serialization cannot release the output owner");
    drop(output);
    assert_eq!(&engine.resources().memory_snapshot(), resident, "all per-document and team charges must drain");
    bytes
}

#[test]
fn optional_compute_metadata_preserves_source_output_and_complete_publication_cap() {
    let (planner, _, _) = fixture(Kind::Ner, 32);
    let semantic = serde_json::json!({"fixture_only":true,"source_span":{
        "byte_start":0,"byte_end":17,"scalar_start":0,"scalar_end":11},"chunks":2});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-source-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "ner",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: INT4_SOURCE_FORMAT,
        quant_recipe: "fixture:group=32", group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let receipt = Int4ComputeReceipt { schedule: "int4-output-row-team-v1", compute_threads: 4,
        completed_projections: 31, completed_rows: 1000, completed_worker_rows: 750 };
    let mut serial = Vec::new(); let mut threaded = Vec::new();
    publish_map(&provenance, &semantic, None, &planner, 4096, &mut serial, &|| Ok(())).unwrap();
    publish_team_map(&provenance, &semantic, &receipt, &planner, 4096, &mut threaded, &|| Ok(())).unwrap();
    let serial_value: serde_json::Value = serde_json::from_slice(&serial).unwrap();
    let mut threaded_value: serde_json::Value = serde_json::from_slice(&threaded).unwrap();
    assert_eq!(threaded_value["output"], semantic);
    assert_eq!(threaded_value.as_object_mut().unwrap().remove("compute").unwrap(),
        serde_json::to_value(&receipt).unwrap());
    assert_eq!(threaded_value, serial_value);
    for absent in ["compute", "input_encoding", "cohort_schedule", "prompt_digest", "snapshot_path"] {
        assert!(serial_value.get(absent).is_none());
    }
    assert_eq!(threaded.last(), Some(&b'\n'));
    let mut exact = Vec::new();
    publish_team_map(&provenance, &semantic, &receipt, &planner, threaded.len(), &mut exact, &|| Ok(())).unwrap();
    assert_eq!(exact, threaded);
    let mut refused = Vec::new();
    assert_eq!(publish_team_map(&provenance, &semantic, &receipt, &planner,
        threaded.len() - 1, &mut refused, &|| Ok(())), Err(CandidateError::Output));
    assert!(refused.is_empty());
}

#[test]
fn stale_question_team_admission_rejects_new_schedules_before_runtime_or_private_io() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid question team read source"); }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("invalid question team wrote output"); }
        fn flush(&mut self) -> io::Result<()> { panic!("invalid question team flushed output"); }
    }
    let mut command = crate::candidate_cli::int4::source_map::tests::command(Kind::Answer,
        &["--question", "private-question.txt", "--compute-threads", "2", "--input-encoding", "safe-bpe-v1"]);
    command.args.memory_mib = 262_144;
    let Int4Task::SourceMap(mut options) = command.task else { panic!("document question required"); };
    let admitted = options.validate(&command.args).unwrap();
    options.cohort_chunks = Some(2);
    assert_eq!(execute(command.args.clone(), options.clone(), admitted, &mut NoIo, &mut NoIo),
        Err(CandidateError::Arguments));
    options.cohort_chunks = None;
    let replacement = crate::candidate_cli::int4::source_map::tests::command(Kind::Answer,
        &["--question", "private-question.txt", "--synthesize-answer"]);
    let Int4Task::SourceMap(replacement) = replacement.task else { panic!("document question required"); };
    options.question = replacement.question;
    assert_eq!(execute(command.args, options, admitted, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[test]
fn controlled_question_verification_keeps_timeout_distinct_from_invalid_results() {
    use crate::tasks::source_planning::grouped_int4::Int4SourceError;
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(verification_error(Int4SourceError::Cancelled(cause).into()), CandidateError::Timeout);
    }
    for cause in [DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::PollQuota] {
        assert_eq!(verification_error(Int4SourceError::Cancelled(cause).into()), CandidateError::Execution);
    }
    assert_eq!(verification_error(Int4SourceError::InvalidResult.into()), CandidateError::Execution);
}
