//! Controller-only full-model stream parity; fixture source is not execution evidence.
use super::*;

fn command(threads: usize, selected: bool, snapshot: &str, pin: &str, schema: &str,
    memory: &str, kernel: &str) -> (BatchArgs, Int4Args, BatchLimits) {
    let width = threads.to_string();
    let mut argv = vec!["candidate", "source-int4", "extract-batch",
        "--snapshot", snapshot, "--expected-model-sha256", pin, "--schema", schema,
        "--memory-mib", memory, "--kernel", kernel, "--compute-threads", &width,
        "--context-tokens", "2048", "--max-new-tokens", "128",
        "--max-requests", "7", "--timeout-seconds", "86400"];
    if selected { argv.extend(["--max-selected-rows", "166144"]); }
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong extraction route"); };
    let Int4Command { task: Int4Task::ExtractBatch(batch), args, batch: None, job: None } = command
        else { panic!("extraction stream task required"); };
    let limits = batch.validate(&args).expect("explicit memory must admit the complete stream and team");
    (batch, args, limits)
}

#[test]
#[ignore = "run alone with controller-admitted independently pinned snapshot and bounded structural schema for full-model serial/team parity"]
fn snapshot_extraction_teams_preserve_every_record_work_policy_and_flush_and_release_charges() {
    assert!(crate::installed_process_resources().is_none(), "fixture requires a fresh process");
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let schema = std::env::var("FNLP_INT4_TEST_EXTRACTION_SCHEMA").expect("explicit bounded structural schema required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required");
    PackedInt4Kernel::detect(kernel.parse::<Int8KernelKind>().unwrap()).unwrap();
    let (base_batch, base_args, base_limits) = command(4, false, &snapshot, &pin, &schema, &memory, &kernel);
    let preparation_bytes = [false, true].into_iter().map(|selected|
        command(4, selected, &snapshot, &pin, &schema, &memory, &kernel).2.common.preparation_bytes)
        .max().unwrap();
    let engine = NlpEngine::builder().resource_config(compute::resource_config(base_limits.common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    // Retain the production outer preparation reserve for the bounded schema,
    // input, captures and comparison values, independently of each hosted run's
    // preparation/IO reservations. No width reinstalls the process host.
    let preparation = lease.reserve(MemoryClass::JobBuffers, preparation_bytes).unwrap().commit().unwrap();
    {
        let declaration = read_input(&mut File::open(&schema).unwrap(), SCHEMA_BYTES).unwrap();
        let program = JsonProgram::compile(&declaration, base_batch.extraction.compiler()).unwrap();
        let input = extraction_parity_input();
        let model = crate::candidate_cli::int4::runtime::load_model(&engine, &base_args, base_limits.common,
            Duration::from_secs(3600), CancellationToken::default()).unwrap();
        let resident = engine.resources().memory_snapshot();
        for selected in [false, true] {
            let mut expected: Option<Vec<u8>> = None;
            let mut expected_summary = None;
            let (_, _, serial_limits) = command(1, selected, &snapshot, &pin, &schema, &memory, &kernel);
            for threads in [1, 2, 4] {
                let (batch, args, limits) = command(threads, selected, &snapshot, &pin, &schema, &memory, &kernel);
                assert_eq!(limits.transport, serial_limits.transport);
                assert_eq!(limits.native.max_model_work, serial_limits.native.max_model_work);
                assert_eq!(limits.native.masks.max_visits_per_run, serial_limits.native.masks.max_visits_per_run);
                assert_eq!(limits.selected_rows.map(|rows| rows.max_rows_per_step), selected.then_some(166144));
                let budget = batch.extraction.budget(&args, limits.common);
                let defaults = batch.load_defaults(declaration.clone(), budget).unwrap();
                let planner = Int4ExtractionBatchPlanner::for_model(model.model_identity(), budget,
                    batch.extraction.compiler(), batch.extraction.source_limits(), Some(defaults))
                    .and_then(Int4ExtractionBatchPlanner::require_shared_defaults).unwrap();
                let planner = batch.configure_planner(planner).unwrap();
                let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
                let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls()).unwrap());
                let corpus = CorpusLimits {
                    native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                        run: RunLimits { max_elapsed: Duration::from_secs(args.timeout_seconds),
                            max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
                    transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
                    io_reserve_bytes: limits.io_bytes,
                };
                let output = SharedOutput::default();
                let reader = Cursor::new(input.clone());
                let cancellation = CancellationToken::default();
                let summary = if threads == 1 {
                    engine.batch_int4_extract(&model, planner, vocabulary, limits.native, corpus,
                        limits.common.kernel, reader, output.clone(), cancellation)
                } else {
                    engine.batch_int4_extract_with_threads(&model, planner, vocabulary, limits.native, corpus,
                        limits.common.kernel, threads, reader, output.clone(), cancellation)
                }.expect("malformed records must be reported while the protocol still drains");
                let bytes = output.0.lock().unwrap().clone();
                assert!(bytes.ends_with(b"\n"));
                let records: Vec<serde_json::Value> = bytes.split(|&byte| byte == b'\n')
                    .filter(|line| !line.is_empty()).map(|line| serde_json::from_slice(line).unwrap()).collect();
                assert_eq!(records.first().unwrap()["event"], "run_start");
                for record in &records { assert_eq!(record["execution"], crate::batch::BATCH_EXECUTION); }
                let documents: Vec<_> = records.iter().filter(|record| record["event"] == "doc").collect();
                assert_eq!(documents.len(), 5);
                assert_eq!(documents.iter().map(|record| record["request_seq"].as_u64().unwrap())
                    .collect::<Vec<_>>(), [1, 2, 4, 6, 7]);
                assert_eq!(documents[0]["caller_id"], documents[3]["caller_id"]);
                assert_eq!(documents[0]["epoch"], 1); assert_eq!(documents[3]["epoch"], 2);
                assert_eq!(documents[0]["result"], documents[3]["result"]);
                for document in &documents {
                    let result = &document["result"];
                    assert_eq!(result["execution"], if selected {
                        crate::tasks::extract::grouped_int4::INT4_SPARSE_EXTRACT_VERSION
                    } else { crate::tasks::extract::grouped_int4::INT4_EXTRACT_VERSION });
                    assert_eq!(result["result"]["output"]["numerics_profile"],
                        crate::native_engine::strict_int4::STRICT_INT4_PROFILE);
                    program.validate_json(result["result"]["output"]["json"].as_str().unwrap()).unwrap();
                }
                let errors: Vec<_> = records.iter().filter(|record| record["event"] == "doc_error").collect();
                assert_eq!(errors.len(), 1);
                assert_eq!(errors[0]["request_seq"], 3); assert_eq!(errors[0]["error"]["code"], "invalid_json");
                let footer = records.last().unwrap();
                assert_eq!(footer["event"], "run_complete");
                assert_eq!(footer["summary"], serde_json::to_value(summary).unwrap());
                assert_eq!(summary.input_bytes, input.len() as u64);
                assert_eq!(summary.requests, 7); assert_eq!(summary.succeeded, 5);
                assert_eq!(summary.failed, 1); assert_eq!(summary.flushes, 2);
                // Compare the canonical bytes, including every output token,
                // observed/reserved work counter, source offset, error and epoch.
                // Each head policy has its own serial authority; no field is removed.
                if let Some(expected) = &expected {
                    assert!(bytes.as_slice() == expected.as_slice(), "selected={selected}, threads={threads}");
                    assert_eq!(Some(summary), expected_summary);
                } else { expected = Some(bytes); expected_summary = Some(summary); }
                assert_eq!(engine.resources().memory_snapshot(), resident);
                let closures = engine.resources().outstanding_closure_snapshot();
                assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
            }
        }
        drop(model);
    }
    drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}
