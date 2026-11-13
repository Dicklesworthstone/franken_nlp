//! Typed transport/native failure policy and full process admission.
use super::*;
use crate::{candidate_cli::int4::resolve::corpus::tests::{args, NoIo}, hosted::RunStop,
    ResourceHostConfig, RuntimePreset, LeakResponsePolicy};

#[test]
fn all_eleven_cancellation_causes_survive_loading_discovery_and_transport() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = match cause {
            Timeout | Deadline | PollQuota | CostBudget => ErrorCode::BudgetOrTimeout,
            User | ParentCancelled | Shutdown => ErrorCode::Cancelled,
            ResourceUnavailable => ErrorCode::AdmissionOrResourceLimit,
            FailFast | LinkedExit | RaceLost => ErrorCode::Generic,
        };
        for error in [
            Int4EntityCorpusError::Host(HostedError::Stopped { stop: RunStop {
                kind: cause, runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }),
            Int4EntityCorpusError::Host(HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))),
            Int4EntityCorpusError::Host(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))),
            Int4EntityCorpusError::Transport(EntityStreamError::Resolution(ResolveError::Cancelled(cause))),
            Int4EntityCorpusError::Transport(EntityStreamError::Handler(HostedError::Int4(Int4HostError::Entities(
                Int4EntityError::Native(StrictInt4Error::Cancelled(cause)))))),
        ] {
            let failure = corpus_failure(error);
            assert_eq!(failure.exit, expected);
            assert_eq!(failure.cancellation, Some(cause));
        }
    }
}

#[test]
fn record_errors_transport_exhaustion_and_unknown_delivery_stay_distinct() {
    for error in [EntityStreamError::InputIo, EntityStreamError::InvalidRecord] {
        assert_eq!(corpus_failure(Int4EntityCorpusError::Transport(error)).exit, ErrorCode::InputDecodeOrParse);
    }
    for error in [EntityStreamError::InputLimit, EntityStreamError::LineLimit, EntityStreamError::RecordLimit,
        EntityStreamError::SnapshotLimit, EntityStreamError::OutputLimit] {
        let failure = corpus_failure(Int4EntityCorpusError::Transport(error));
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.code, "entity_corpus_transport_budget");
    }
    for error in [Int4EntityCorpusError::OutputIo,
        Int4EntityCorpusError::Transport(EntityStreamError::OutputIo)] {
        let failure = corpus_failure(error);
        assert_eq!(failure.exit, ErrorCode::Generic);
        assert_eq!(failure.code, "entity_corpus_output_delivery_unknown");
    }
    assert_eq!(corpus_failure(Int4EntityCorpusError::Completion).exit, ErrorCode::Generic);
}

#[test]
fn failed_fixed_policy_native_accounting_and_supervision_never_count_as_epochs() {
    assert_eq!(entity_failure(Int4EntityError::WorkBudget).exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(entity_failure(Int4EntityError::InvalidInput).exit, ErrorCode::InputDecodeOrParse);
    assert_eq!(entity_failure(Int4EntityError::Identity).exit, ErrorCode::ArtifactIntegrityOrFormatOrVersion);
    assert_eq!(entity_failure(Int4EntityError::Accounting).exit, ErrorCode::Generic);
    assert_eq!(entity_failure(Int4EntityError::Allocation).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(entity_failure(Int4EntityError::Source(
        crate::tasks::source_planning::grouped_int4::Int4SourceError::InvalidResult)).exit,
        ErrorCode::StructuredTaskNoResult);
    for cause in [DecodeCancellationKind::FailFast, DecodeCancellationKind::LinkedExit] {
        let failure = host_failure(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Panicked)),
        });
        assert_eq!(failure.exit, ErrorCode::Generic);
        assert_eq!(failure.cancellation, Some(cause));
    }
    assert_eq!(host_failure(HostedError::Panicked).exit, ErrorCode::Generic);
    assert_eq!(host_failure(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Integrity))).exit,
        ErrorCode::ArtifactIntegrityOrFormatOrVersion);
}

#[test]
fn process_quote_includes_both_preparation_owners_native_storage_and_stream_buffers() {
    let (command, args) = args(&["--chunked"]);
    let limits = command.validate(&args).unwrap();
    let task = command.configuration(&args, limits.common,
        Policy { ner: NerOptions::default(), options: ResolveOptions::default() }).unwrap();
    let need = Int4EntityCorpusHostRequirements::for_config(&task, limits.transport,
        native(&args, Duration::from_secs(args.timeout_seconds)), limits.common.preparation_bytes,
        command.resolution.graph_bytes().unwrap(), limits.io_bytes, args.group_width).unwrap();
    let quote = limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES
        + limits.common.preparation_bytes + need.total_bytes().unwrap();
    assert!(limits.common.minimum_memory_bytes >= quote);
    let without_stream = command.resolution.validate(&args).unwrap();
    assert!(limits.common.minimum_memory_bytes >= without_stream.minimum_memory_bytes);
}

#[test]
fn runtime_revalidates_mutated_options_before_policy_model_or_corpus_io() {
    for which in 0..3 {
        let (mut command, mut args) = args(&[]);
        let admitted = command.validate(&args).unwrap();
        match which {
            0 => command.stream.max_records = 0,
            1 => command.resolution.discovery.discover_entities = false,
            _ => args.max_checkpoints = 0,
        }
        assert_eq!(execute(args, command, admitted, NoIo, NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
}

#[test]
#[ignore = "requires controller-admitted INT4 snapshot, independent pin, fixed policy file and real native execution"]
fn real_cli_reuses_epochs_and_refuses_late_input_or_lifetime_mask_exhaustion() {
    use std::sync::Mutex;
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let snapshot = std::env::var("FNLP_INT4_ENTITY_CORPUS_SNAPSHOT").expect("explicit existing INT4 snapshot required");
    let pin = std::env::var("FNLP_INT4_ENTITY_CORPUS_MODEL_SHA256").expect("independent model-content pin required");
    let policy = std::env::var("FNLP_INT4_ENTITY_CORPUS_POLICY").expect("explicit fixed {ner,options} policy file required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit process memory ceiling required");
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2,
        memory_ceiling_bytes: memory.parse::<u64>().unwrap().checked_mul(MIB).unwrap(),
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let make = |extra: &[&str]| {
        let mut argv = vec!["candidate", "source-int4", "resolve-corpus", "--discover-entities", "--chunked",
            "--policy", policy.as_str(), "--snapshot", snapshot.as_str(),
            "--expected-model-sha256", pin.as_str(), "--memory-mib", memory.as_str()];
        argv.extend_from_slice(extra);
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("INT4 corpus route required"); };
        command
    };
    let run = |wire: &[u8], extra: &[&str]| {
        let output = Capture::default(); let observer = output.clone();
        let mut diagnostics = Vec::new();
        let status = make(extra).run_owned(io::Cursor::new(wire.to_vec()), output, &mut diagnostics);
        let bytes = observer.0.lock().unwrap().clone();
        let frames: Vec<serde_json::Value> = bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap()).collect();
        assert_eq!(engine.resources().memory_snapshot(), idle);
        let closures = engine.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0);
        assert_eq!(closures.wrapper_cancelled_closures, 0);
        (status, frames, diagnostics)
    };
    let document = serde_json::json!({"id":"repeat","text":"Ada works at Acme in Zürich."}).to_string();
    let success = format!("{{\"flush\":true}}\n{document}\n{{\"flush\":true}}\n{document}\n");
    let (status, frames, diagnostics) = run(success.as_bytes(), &[]);
    assert_eq!(status, ExitCode::SUCCESS);
    assert!(diagnostics.is_empty());
    assert_eq!(frames.len(), 4);
    for frame in &frames { assert_eq!(frame["protocol"], ENTITY_STREAM_PROTOCOL); }
    for frame in &frames[..3] { assert_eq!(frame["event"], "corpus"); }
    assert_eq!(frames[0]["result"]["output"]["documents"], serde_json::json!([]));
    assert!(frames[1]["result"]["output"]["model_work"]["forward_positions"].as_u64().unwrap() > 0);
    assert_eq!(frames[1]["result"]["output"], frames[2]["result"]["output"]);
    assert_eq!(frames[1]["result"]["cluster_scope"], "epoch_local");
    assert_eq!(frames[1]["eof"], false);
    assert_eq!(frames[2]["eof"], true);
    assert_eq!(frames[3]["event"], "run_complete");
    assert_eq!(frames[3]["summary"]["transport"]["snapshots"], 3);
    assert_eq!(frames[3]["summary"]["transport"]["documents"], 2);
    assert_eq!(frames[3]["summary"]["transport"]["records"], 4);

    let malformed = format!("{document}\n{{\"flush\":true}}\n{{\"id\":\"last\",\"text\":\"private\",\"model\":\"forbidden\"}}\n");
    let (status, frames, diagnostics) = run(malformed.as_bytes(), &[]);
    assert_eq!(status, ErrorCode::InputDecodeOrParse.as_process_exit());
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["event"], "corpus");
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["event"], "run_error");
    assert!(!String::from_utf8(diagnostics).unwrap().contains("private"));

    let exhausted = format!("{document}\n{{\"flush\":true}}\n{document}\n");
    let (status, frames, diagnostics) = run(exhausted.as_bytes(), &["--max-snapshot-mask-node-visits", "1000000000"]);
    assert_eq!(status, ErrorCode::BudgetOrTimeout.as_process_exit());
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["event"], "corpus");
    let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
    assert_eq!(report["event"], "run_error");
    assert_eq!(report["status"], "incomplete");

    // A partial write and a failed flush both consume their writer. Neither
    // delivery state permits retry, another epoch, or a success footer.
    struct Broken {
        capture: Capture, calls: Arc<Mutex<[usize; 2]>>, fail_write: bool,
    }
    impl Write for Broken {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let mut calls = self.calls.lock().unwrap(); calls[0] += 1;
            if self.fail_write && calls[0] > 1 { return Err(io::Error::other("private output failure")); }
            let count = if self.fail_write { bytes.len().min(7) } else { bytes.len() };
            self.capture.write(&bytes[..count])
        }
        fn flush(&mut self) -> io::Result<()> {
            self.calls.lock().unwrap()[1] += 1;
            Err(io::Error::other("private flush failure"))
        }
    }
    for fail_write in [true, false] {
        let capture = Capture::default(); let observer = capture.clone();
        let calls = Arc::new(Mutex::new([0; 2]));
        let writer = Broken { capture, calls: Arc::clone(&calls), fail_write };
        let mut diagnostics = Vec::new();
        let status = make(&[]).run_owned(io::Cursor::new(format!("{document}\n").into_bytes()), writer, &mut diagnostics);
        assert_eq!(status, ErrorCode::Generic.as_process_exit());
        let bytes = observer.0.lock().unwrap().clone();
        assert!(!bytes.windows(b"run_complete".len()).any(|window| window == b"run_complete"));
        assert_eq!(*calls.lock().unwrap(), if fail_write { [2, 0] } else { [1, 1] });
        if fail_write { assert_eq!(bytes.len(), 7); }
        let report: serde_json::Value = serde_json::from_slice(&diagnostics).unwrap();
        assert_eq!(report["code"], "entity_corpus_output_delivery_unknown");
        assert!(!String::from_utf8(diagnostics).unwrap().contains("private"));
        assert_eq!(engine.resources().memory_snapshot(), idle);
        assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
    }

    // The public hosted boundary also refuses a cancelled invocation before
    // consuming a corpus reader or writer, while retaining the loaded model.
    let command = make(&[]);
    let Int4Task::ResolveCorpus(options) = command.task else { panic!("corpus route required"); };
    let args = command.args;
    let limits = options.validate(&args).unwrap();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, limits.common.preparation_bytes).unwrap().commit().unwrap();
    let policy_text = read_input(&mut crate::local_io::open_document(&options.stream.policy).unwrap(), POLICY_BYTES).unwrap();
    let task = options.configuration(&args, limits.common, Policy::parse(&policy_text).unwrap()).unwrap();
    let model = engine.load_int4_snapshot(args.snapshot.clone().unwrap(), limits.common.snapshot_pin.unwrap(),
        limits.common.snapshot_budget(true).unwrap(), ALLOCATOR_BYTES,
        RunLimits { max_elapsed: Duration::from_secs(args.timeout_seconds), max_checkpoints: args.snapshot_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES }, CancellationToken::default()).unwrap();
    let (source, vocabulary) = crate::candidate_cli::int4::source::planner().unwrap();
    let source_identity = crate::candidate_cli::int4::source::identity(model.model_identity(), &source, BuiltInTask::Ner).unwrap();
    let (resolver, resolution_identity) = crate::candidate_cli::int4::resolve::runtime::planner(model.model_identity()).unwrap();
    let config = Int4EntityCorpusHostConfig { source_identity, resolution_identity, task,
        transport: limits.transport, native: native(&args, Duration::from_secs(args.timeout_seconds)),
        preparation_reserve_bytes: limits.common.preparation_bytes,
        graph_reserve_bytes: options.resolution.graph_bytes().unwrap(), io_reserve_bytes: limits.io_bytes,
        schedule: None };
    let loaded = engine.resources().memory_snapshot();
    let cancellation = CancellationToken::default(); cancellation.cancel(DecodeCancellationKind::User);
    let error = engine.execute_int4_entity_corpus(&model, Arc::new(source), Arc::new(resolver), Arc::new(vocabulary),
        config, limits.common.kernel, BufReader::new(NoIo), NoIo, cancellation).err().expect("cancelled run must fail");
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::User));
    assert_eq!(engine.resources().memory_snapshot(), loaded);
    assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
    drop(policy_text); drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}

#[test]
fn corpus_team_process_quote_adds_one_independent_worker_allocation() {
    for threads in ["2", "3", "64"] {
        let (command, mut args) = args(&["--chunked", "--compute-threads", threads]);
        args.memory_mib = 131072;
        let limits = command.validate(&args).unwrap();
        let task = command.configuration(&args, limits.common,
            Policy { ner: NerOptions::default(), options: ResolveOptions::default() }).unwrap();
        let required = Int4EntityCorpusHostRequirements::for_config(&task, limits.transport,
            native(&args, Duration::from_secs(args.timeout_seconds)), limits.common.preparation_bytes,
            command.resolution.graph_bytes().unwrap(), limits.io_bytes, args.group_width).unwrap();
        let expected = limits.common.load.weight_payload_bytes + ALLOCATOR_BYTES
            + limits.common.preparation_bytes + required.total_bytes().unwrap()
            + limits.common.compute.workspace_bytes;
        assert_eq!(memory_floor(&args, &command, limits.common, limits.transport, limits.io_bytes, &task).unwrap(),
            expected);
        assert!(limits.common.minimum_memory_bytes >= expected);
        args.memory_mib = (limits.common.minimum_memory_bytes - 1) / MIB;
        let refused = command.clone().execute_owned(args.clone(), NoIo, NoIo).err().unwrap();
        assert!(matches!(refused.exit, ErrorCode::Usage | ErrorCode::AdmissionOrResourceLimit));
        args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
        assert!(command.validate(&args).is_ok());
    }
}

#[test]
#[ignore = "requires controller-admitted INT4 snapshot, independent pin, fixed NER policy and real model parity"]
fn persistent_entity_teams_match_all_three_scalar_apis_and_drain_before_delivery() {
    use crate::{corpus::{entities::EntityDocument, resolve::ResolutionDocument},
        hosted::{Int4ComputeReceipt, Int4ResolveConfig}};
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    impl Capture {
        fn frames(&self) -> Vec<serde_json::Value> {
            self.0.lock().unwrap().split(|&b| b == b'\n').filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap()).collect()
        }
    }
    let snapshot = std::env::var("FNLP_INT4_ENTITY_CORPUS_SNAPSHOT").expect("explicit INT4 snapshot required");
    let pin = std::env::var("FNLP_INT4_ENTITY_CORPUS_MODEL_SHA256").expect("independent model-content pin required");
    let policy = std::env::var("FNLP_INT4_ENTITY_CORPUS_POLICY").expect("fixed person/organization NER policy required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit process memory ceiling required");
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "resolve-corpus", "--discover-entities", "--chunked",
        "--max-ner-chunk-bytes", "24", "--compute-threads", "4",
        "--policy", &policy, "--snapshot", &snapshot, "--expected-model-sha256", &pin,
        "--memory-mib", &memory,
    ]).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("INT4 corpus route required"); };
    let Int4Task::ResolveCorpus(options) = command.task else { panic!("entity corpus required"); };
    let args = command.args;
    let limits = options.validate(&args).unwrap();
    let common = limits.common;
    // One immutable process host admits scalar execution and widths 2..=4.
    // Never compare different CLI process envelopes in the same process.
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes).unwrap().commit().unwrap();
    let policy_text = read_input(&mut crate::local_io::open_document(&options.stream.policy).unwrap(), POLICY_BYTES).unwrap();
    let task = options.configuration(&args, common, Policy::parse(&policy_text).unwrap()).unwrap();
    let model = engine.load_int4_snapshot(args.snapshot.clone().unwrap(), common.snapshot_pin.unwrap(),
        common.snapshot_budget(true).unwrap(), ALLOCATOR_BYTES,
        RunLimits { max_elapsed: Duration::from_secs(args.timeout_seconds), max_checkpoints: args.snapshot_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES }, CancellationToken::default()).unwrap();
    let (source, vocabulary) = crate::candidate_cli::int4::source::planner().unwrap();
    let source_identity = crate::candidate_cli::int4::source::identity(model.model_identity(), &source, BuiltInTask::Ner).unwrap();
    let (resolver, resolution_identity) = crate::candidate_cli::int4::resolve::runtime::planner(model.model_identity()).unwrap();
    let (source, resolver, vocabulary) = (Arc::new(source), Arc::new(resolver), Arc::new(vocabulary));
    let loaded = engine.resources().memory_snapshot();
    let assert_drained = || {
        let closures = engine.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0);
        assert_eq!(closures.wrapper_cancelled_closures, 0);
    };
    let assert_compute = |compute: &Int4ComputeReceipt, threads| {
        assert_eq!(compute.compute_threads, threads);
        assert_eq!(compute.schedule, "int4-output-row-team-v1");
        assert!(compute.completed_projections > 0);
        assert!(compute.completed_rows > 0);
        assert!(compute.completed_worker_rows > 0);
        assert!(compute.completed_worker_rows < compute.completed_rows);
    };
    let make_pair_config = || Int4ResolveConfig {
        identity: resolution_identity.clone(), options: task.resolution, graph: task.graph,
        scoring: task.scoring, native: native(&args, Duration::from_secs(args.timeout_seconds)),
        preparation_reserve_bytes: common.preparation_bytes,
        graph_reserve_bytes: options.resolution.graph_bytes().unwrap(),
    };
    let pair_documents = || -> Vec<ResolutionDocument> {
        serde_json::from_value(serde_json::json!([
            {"id":"a","text":"Ada works here.","mentions":[{"entity_type":"person","surface":"Ada",
                "span":{"byte_start":0,"byte_end":3,"scalar_start":0,"scalar_end":3}}]},
            {"id":"b","text":"Ada works there.","mentions":[{"entity_type":"person","surface":"Ada",
                "span":{"byte_start":0,"byte_end":3,"scalar_start":0,"scalar_end":3}}]},
        ])).unwrap()
    };
    let scalar = engine.resolve_int4(&model, pair_documents(), Arc::clone(&resolver), make_pair_config(),
        common.kernel, CancellationToken::default()).unwrap();
    assert_eq!(scalar.result().head_count, 2);
    assert_eq!(scalar.result().result.judgments.len(), 1);
    assert!(scalar.result().model_evaluated);
    let pair_expected = crate::canonjson::canonical_bytes(scalar.result()).unwrap();
    drop(scalar);
    for threads in [2, 4] {
        let result = engine.resolve_int4_with_threads(&model, pair_documents(), Arc::clone(&resolver),
            make_pair_config(), common.kernel, threads, CancellationToken::default()).unwrap();
        assert_eq!(crate::canonjson::canonical_bytes(result.result()).unwrap(), pair_expected);
        assert_compute(result.compute(), threads);
        assert_drained();
        assert!(engine.resources().memory_snapshot().committed_bytes > loaded.committed_bytes);
        drop(result);
        assert_eq!(engine.resources().memory_snapshot(), loaded);
    }

    let raw_documents = || -> Vec<EntityDocument> {
        serde_json::from_value(serde_json::json!([
            {"id":"a","text":"Ada works at Acme in Zürich."},
            {"id":"b","text":"Ada works at Acme in Zürich."},
        ])).unwrap()
    };
    let prepare = || prepare_int4_entities(raw_documents(), Arc::clone(&source), source_identity.clone(),
        Arc::clone(&resolver), resolution_identity.clone(), task.clone(), model.model_identity(),
        &mut DeadlineControl { started: Instant::now(), deadline: Duration::from_secs(args.timeout_seconds) }).unwrap();
    let scalar = engine.execute_int4_entities(&model, prepare(), Arc::clone(&vocabulary),
        native(&args, Duration::from_secs(args.timeout_seconds)), common.preparation_bytes,
        options.resolution.graph_bytes().unwrap(), common.kernel, CancellationToken::default()).unwrap();
    assert!(scalar.result().chunk_count > 2, "fixture must execute multiple chunks per source");
    assert!(scalar.result().ner_model_work.forward_positions > 0);
    assert!(scalar.result().resolution.head_count > 0, "fixture must exercise both pair orders after discovery");
    let entity_expected = crate::canonjson::canonical_bytes(scalar.result()).unwrap();
    drop(scalar);
    let mut entity_compute = None;
    for threads in [2, 4] {
        let result = engine.execute_int4_entities_with_threads(&model, prepare(), Arc::clone(&vocabulary),
            native(&args, Duration::from_secs(args.timeout_seconds)), common.preparation_bytes,
            options.resolution.graph_bytes().unwrap(), common.kernel, threads, CancellationToken::default()).unwrap();
        assert_eq!(crate::canonjson::canonical_bytes(result.result()).unwrap(), entity_expected);
        assert_compute(result.compute(), threads);
        entity_compute = Some(*result.compute());
        assert_drained();
        drop(result);
        assert_eq!(engine.resources().memory_snapshot(), loaded);
    }
    let empty_prepared = prepare_int4_entities(Vec::new(), Arc::clone(&source), source_identity.clone(),
        Arc::clone(&resolver), resolution_identity.clone(), task.clone(), model.model_identity(),
        &mut DeadlineControl { started: Instant::now(), deadline: Duration::from_secs(args.timeout_seconds) }).unwrap();
    let empty = engine.execute_int4_entities_with_threads(&model, empty_prepared, Arc::clone(&vocabulary),
        native(&args, Duration::from_secs(args.timeout_seconds)), common.preparation_bytes,
        options.resolution.graph_bytes().unwrap(), common.kernel, 4, CancellationToken::default()).unwrap();
    assert!(empty.result().documents.is_empty());
    assert_eq!(empty.compute().completed_projections, 0);
    assert_drained();
    drop(empty);
    assert_eq!(engine.resources().memory_snapshot(), loaded);
    let cancelled = CancellationToken::default(); cancelled.cancel(DecodeCancellationKind::ParentCancelled);
    let error = engine.execute_int4_entities_with_threads(&model, prepare(), Arc::clone(&vocabulary),
        native(&args, Duration::from_secs(args.timeout_seconds)), common.preparation_bytes,
        options.resolution.graph_bytes().unwrap(), common.kernel, 4, cancelled).err().unwrap();
    assert!(matches!(error, HostedError::Stopped { stop: RunStop {
        kind: DecodeCancellationKind::ParentCancelled, .. }, .. }));
    assert_drained();
    assert_eq!(engine.resources().memory_snapshot(), loaded);

    let make_corpus_config = || Int4EntityCorpusHostConfig {
        source_identity: source_identity.clone(), resolution_identity: resolution_identity.clone(), task: task.clone(),
        transport: limits.transport, native: native(&args, Duration::from_secs(args.timeout_seconds)),
        preparation_reserve_bytes: common.preparation_bytes,
        graph_reserve_bytes: options.resolution.graph_bytes().unwrap(), io_reserve_bytes: limits.io_bytes, schedule: None,
    };
    let epoch = raw_documents().iter().map(|document| serde_json::to_string(document).unwrap())
        .collect::<Vec<_>>().join("\n") + "\n";
    let wire = format!("{{\"flush\":true}}\n{epoch}{{\"flush\":true}}\n{epoch}");
    let capture = Capture::default();
    let scalar = engine.execute_int4_entity_corpus(&model, Arc::clone(&source), Arc::clone(&resolver),
        Arc::clone(&vocabulary), make_corpus_config(), common.kernel, io::Cursor::new(wire.clone().into_bytes()),
        capture.clone(), CancellationToken::default()).unwrap();
    assert_drained();
    let expected_summary = scalar.finish().unwrap();
    assert_eq!(expected_summary.transport.snapshots, 3);
    assert_eq!(expected_summary.transport.documents, 4);
    assert!(expected_summary.compute.is_none());
    let expected_frames = capture.frames();
    assert_eq!(expected_frames.len(), 4);
    assert_eq!(crate::canonjson::canonical_bytes(&expected_frames[1]["result"]["output"]).unwrap(), entity_expected);
    assert_eq!(expected_frames[1]["result"]["output"], expected_frames[2]["result"]["output"]);
    assert_eq!(engine.resources().memory_snapshot(), loaded);
    for threads in [2, 4] {
        let capture = Capture::default();
        let completed = engine.execute_int4_entity_corpus_with_threads(&model, Arc::clone(&source),
            Arc::clone(&resolver), Arc::clone(&vocabulary), make_corpus_config(), common.kernel, threads,
            io::Cursor::new(wire.clone().into_bytes()), capture.clone(), CancellationToken::default()).unwrap();
        assert_drained();
        assert_eq!(capture.frames().len(), 3, "footer must wait for explicit completion delivery");
        assert!(engine.resources().memory_snapshot().committed_bytes > loaded.committed_bytes);
        let mut summary = completed.finish().unwrap();
        let compute = summary.compute.take().unwrap();
        assert_compute(&compute, threads);
        let one = entity_compute.unwrap();
        assert_eq!(compute.completed_projections, one.completed_projections * 2);
        assert_eq!(compute.completed_rows, one.completed_rows * 2);
        assert_eq!(summary, expected_summary);
        let mut frames = capture.frames();
        assert_eq!(frames.last_mut().unwrap()["summary"].as_object_mut().unwrap().remove("compute"),
            Some(serde_json::to_value(compute).unwrap()));
        assert_eq!(frames, expected_frames);
        assert_eq!(engine.resources().memory_snapshot(), loaded);
    }

    // No epochs still authenticates the pinned recipe and owns/joins one team.
    let capture = Capture::default();
    let empty = engine.execute_int4_entity_corpus_with_threads(&model, Arc::clone(&source), Arc::clone(&resolver),
        Arc::clone(&vocabulary), make_corpus_config(), common.kernel, 4, io::Cursor::new(Vec::<u8>::new()),
        capture.clone(), CancellationToken::default()).unwrap();
    assert_drained();
    let summary = empty.finish().unwrap();
    assert_eq!(summary.transport.snapshots, 0);
    assert_eq!(summary.compute.unwrap().completed_projections, 0);
    assert_eq!(capture.frames().len(), 1);
    assert_eq!(engine.resources().memory_snapshot(), loaded);

    let cancellation = CancellationToken::default();
    cancellation.cancel(DecodeCancellationKind::ParentCancelled);
    let error = engine.execute_int4_entity_corpus_with_threads(&model, Arc::clone(&source), Arc::clone(&resolver),
        Arc::clone(&vocabulary), make_corpus_config(), common.kernel, 4, BufReader::new(NoIo), NoIo,
        cancellation).err().expect("pre-cancelled team must refuse before private IO");
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::ParentCancelled));
    assert_drained();
    assert_eq!(engine.resources().memory_snapshot(), loaded);

    // A late parser failure must join the team, preserve only the acknowledged
    // epoch and consume all physical reservations without a success footer.
    let capture = Capture::default();
    let malformed = format!("{epoch}{{\"flush\":true}}\n{{\"id\":\"late\",\"text\":\"private\",\"model\":true}}\n");
    let failure = engine.execute_int4_entity_corpus_with_threads(&model, Arc::clone(&source), Arc::clone(&resolver),
        Arc::clone(&vocabulary), make_corpus_config(), common.kernel, 4, io::Cursor::new(malformed.into_bytes()),
        capture.clone(), CancellationToken::default()).err().expect("late invalid record must fail");
    assert_eq!(corpus_failure(failure).exit, ErrorCode::InputDecodeOrParse);
    assert_eq!(capture.frames().len(), 1);
    assert_eq!(capture.frames()[0]["event"], "corpus");
    assert_drained();
    assert_eq!(engine.resources().memory_snapshot(), loaded);
    drop(source); drop(resolver); drop(vocabulary); drop(task); drop(policy_text);
    drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}
