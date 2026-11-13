//! Controller-only full-model serial/team parity, not execution evidence.
use super::*;

fn command(task: &str, threads: usize, encoding: &str, snapshot: &str, pin: &str,
    memory: u64, kernel: Int8KernelKind) -> (BatchArgs, Int4Args, BatchLimits) {
    let width = threads.to_string();
    let command = super::super::super::tests::command(task, &["--compute-threads", &width,
        "--input-encoding", encoding, "--max-dot-products", "1000000000000",
        "--max-multiply-accumulates", "10000000000000000"]);
    let Int4Command { task: Int4Task::ScoredBatch(batch), mut args, batch: None, job: None } = command
        else { panic!("scored stream task required"); };
    args.source_dir = None; args.source_manifest = None;
    args.snapshot = Some(PathBuf::from(snapshot)); args.expected_model_sha256 = Some(pin.to_owned());
    args.memory_mib = memory; args.kernel = kernel; args.context_tokens = 2048;
    args.timeout_seconds = 86_400;
    let limits = batch.validate(&args).expect("explicit memory must admit the complete stream and team");
    (batch, args, limits)
}

#[allow(clippy::too_many_arguments)]
fn execute_stream(engine: &NlpEngine, model: &ResidentInt4, batch: &BatchArgs, args: &Int4Args,
    limits: BatchLimits, task: &'static str, input: Vec<u8>, output: CapturedStream) {
    let budget = batch.scoring.budget(limits.common);
    let encoding = batch.input_encoding(args, limits.common).unwrap();
    let reader: Box<dyn BufRead + Send> = Box::new(std::io::Cursor::new(input));
    let cancellation = CancellationToken::default();
    let started = Instant::now(); let deadline = Duration::from_secs(args.timeout_seconds);
    match batch.load_defaults(None, budget, args).unwrap() {
        Defaults::Classify(defaults) => {
            let planner = Int4ClassificationPlanner::for_model(model.model_identity(), budget,
                batch.scoring.classification_limits(args)).unwrap()
                .with_document_encoding(encoding.clone()).unwrap();
            let planner = Int4ScoredBatchPlanner::new_with_document_encoding(planner, defaults, limits.native).unwrap();
            stream(engine, model, planner, task, args, limits, encoding, started, deadline, reader, output, cancellation).unwrap();
        }
        Defaults::Sentiment { args: defaults, policy } => {
            let planner = Int4SentimentPlanner::for_model(model.model_identity(), sentiment_options(policy).unwrap(), budget,
                batch.scoring.sentiment_limits(args)).unwrap().with_document_encoding(encoding.clone()).unwrap();
            let planner = Int4ScoredBatchPlanner::new_with_document_encoding(planner, Some(defaults), limits.native).unwrap();
            stream(engine, model, planner, task, args, limits, encoding, started, deadline, reader, output, cancellation).unwrap();
        }
        Defaults::Judge(defaults) => {
            let planner = Int4JudgePlanner::for_model(model.model_identity(), budget,
                batch.scoring.judge_limits(budget)).unwrap().with_document_encoding(encoding.clone()).unwrap();
            let planner = Int4ScoredBatchPlanner::new_with_document_encoding(planner, defaults, limits.native).unwrap();
            stream(engine, model, planner, task, args, limits, encoding, started, deadline, reader, output, cancellation).unwrap();
        }
    }
}

#[test]
#[ignore = "run alone with controller-admitted snapshot, independent pin and real multi-document serial/team INT4 inference"]
fn snapshot_score_stream_teams_match_every_semantic_record_work_and_flush_and_release_charges() {
    assert!(crate::installed_process_resources().is_none(), "fixture requires a fresh process");
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required").parse().unwrap();
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required").parse().unwrap();
    PackedInt4Kernel::detect(kernel).unwrap();
    let cases = [("classify", "byte-fallback"), ("classify", "safe-bpe-v1"),
        ("sentiment", "byte-fallback"), ("sentiment", "safe-bpe-v1"),
        ("judge", "byte-fallback"), ("judge", "safe-bpe-v1")];
    let preparation_bytes = cases.iter().map(|&(task, encoding)|
        command(task, 4, encoding, &snapshot, &pin, memory, kernel).2.common.preparation_bytes).max().unwrap();
    let (_, base_args, base_limits) = command("classify", 4, "safe-bpe-v1", &snapshot, &pin, memory, kernel);
    let engine = NlpEngine::builder().resource_config(resource_config(base_limits.common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    // Fixture inputs, captured output and comparison values remain inside this
    // explicit outer preparation reserve while each invocation owns its IO.
    let preparation = lease.reserve(MemoryClass::JobBuffers, preparation_bytes).unwrap().commit().unwrap();
    let model = crate::candidate_cli::int4::runtime::load_model(&engine, &base_args, base_limits.common,
        Duration::from_secs(3600), CancellationToken::default()).unwrap();
    let resident = engine.resources().memory_snapshot();
    for (task, encoding) in cases {
        let mut input = None;
        let mut expected = None;
        for threads in [1, 2, 4] {
            let (batch, args, limits) = command(task, threads, encoding, &snapshot, &pin, memory, kernel);
            let input = input.get_or_insert_with(|| model_stream_input(task, batch.scoring.budget(limits.common)));
            let output = CapturedStream::default();
            execute_stream(&engine, &model, &batch, &args, limits, task, input.clone(), output.clone());
            let bytes = output.0.lock().unwrap().clone();
            let mut records = Vec::new();
            for line in bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()) {
                let mut record: serde_json::Value = serde_json::from_slice(line).unwrap();
                let schedule = record.as_object_mut().unwrap().remove("compute_schedule");
                assert_eq!(schedule, ComputeSchedule::for_threads(threads).map(serde_json::to_value).transpose().unwrap());
                assert!(record.get("cohort_schedule").is_none());
                records.push(record);
            }
            let footer = &records.last().unwrap()["record"];
            assert_eq!(footer["event"], "run_complete");
            assert_eq!(footer["summary"]["succeeded"], 5);
            assert_eq!(footer["summary"]["failed"], 0);
            assert_eq!(footer["summary"]["flushes"], 2);
            if let Some(expected) = &expected { assert_eq!(&records, expected, "{task} {encoding} threads={threads}"); }
            else { expected = Some(records); }
            assert_eq!(engine.resources().memory_snapshot(), resident);
            let closures = engine.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
        }
    }
    drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}
