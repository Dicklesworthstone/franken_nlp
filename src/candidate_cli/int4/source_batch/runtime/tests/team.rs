//! Controller-only full-model stream parity; fixture source is not execution evidence.
use super::*;

fn command(task: &str, threads: usize, selected: bool, snapshot: &str, pin: &str,
    memory: &str, kernel: &str) -> (BatchArgs, Int4Args, BatchLimits) {
    let width = threads.to_string();
    let mut argv = vec!["candidate", "source-int4", task, "--snapshot", snapshot,
        "--expected-model-sha256", pin, "--memory-mib", memory, "--kernel", kernel,
        "--compute-threads", &width, "--context-tokens", "2048", "--max-new-tokens", "512",
        "--max-requests", "8", "--timeout-seconds", "86400"];
    if selected { argv.extend(["--max-selected-rows", "166144"]); }
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong source route"); };
    let Int4Command { task: Int4Task::SourceBatch(batch), args, batch: None, job: None } = command
        else { panic!("source stream task required"); };
    let limits = batch.validate(&args).expect("explicit memory must admit the complete stream and team");
    (batch, args, limits)
}

#[test]
#[ignore = "run alone with controller-admitted independently pinned snapshot for all four full-model source serial/team streams"]
fn snapshot_source_teams_preserve_all_tasks_evidence_work_errors_and_flushes_and_release_charges() {
    assert!(crate::installed_process_resources().is_none(), "fixture requires a fresh process");
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required");
    PackedInt4Kernel::detect(kernel.parse::<Int8KernelKind>().unwrap()).unwrap();
    let tasks = ["ner-batch", "keyphrases-batch", "summarize-batch", "answer-batch"];
    let preparation_bytes = tasks.iter().flat_map(|task| [false, true].map(|selected|
        command(task, 4, selected, &snapshot, &pin, &memory, &kernel).2.common.preparation_bytes))
        .max().unwrap();
    let (_, base_args, base_limits) = command(tasks[0], 4, false, &snapshot, &pin, &memory, &kernel);
    let engine = NlpEngine::builder().resource_config(compute::resource_config(base_limits.common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    // This production-sized outer reserve owns fixture inputs, pinned planners,
    // captured streams and comparison values until model and IO cleanup finish.
    // Each invocation retains its original, separately charged corpus ceilings.
    let preparation = lease.reserve(MemoryClass::JobBuffers, preparation_bytes).unwrap().commit().unwrap();
    {
        let model = crate::candidate_cli::int4::runtime::load_model(&engine, &base_args, base_limits.common,
            Duration::from_secs(3600), CancellationToken::default()).unwrap();
        let resident = engine.resources().memory_snapshot();
        for task in tasks {
            for selected in [false, true] {
                let mut expected: Option<Vec<u8>> = None;
                let mut expected_summary = None;
                let (serial_batch, serial_args, serial_limits) = command(task, 1, selected, &snapshot, &pin, &memory, &kernel);
                let task_budget = serial_batch.source.budget(&serial_args, serial_limits.common);
                let input = source_parity_input(serial_batch.kind, task_budget);
                for threads in [1, 2, 4] {
                    let (batch, args, limits) = command(task, threads, selected, &snapshot, &pin, &memory, &kernel);
                    let budget = batch.source.budget(&args, limits.common);
                    assert_eq!(budget, task_budget);
                    assert_eq!(limits.transport, serial_limits.transport);
                    assert_eq!(limits.native.max_model_work, serial_limits.native.max_model_work);
                    assert_eq!(limits.native.masks.max_visits_per_run, serial_limits.native.masks.max_visits_per_run);
                    assert_eq!(limits.selected_rows.map(|rows| rows.max_rows_per_step), selected.then_some(166144));
                    let defaults = batch.load_defaults(None, budget, args.max_input_bytes).unwrap();
                    let (planner, vocabulary) = source::planner().unwrap();
                    let identity = source::identity(model.model_identity(), &planner, source::task(batch.kind)).unwrap();
                    let config = Int4SourceCorpusConfig { identity, task_ceiling: budget,
                        planning: batch.source.planning(&args), defaults, native_work: limits.native,
                        selected_rows: limits.selected_rows };
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
                        engine.batch_int4_source(&model, Arc::new(planner), Arc::new(vocabulary), config,
                            corpus, limits.common.kernel, reader, output.clone(), cancellation)
                    } else {
                        engine.batch_int4_source_with_threads(&model, Arc::new(planner), Arc::new(vocabulary), config,
                            corpus, limits.common.kernel, threads, reader, output.clone(), cancellation)
                    }.expect("invalid records must be reported while the protocol still drains");
                    let bytes = output.0.lock().unwrap().clone();
                    assert!(bytes.ends_with(b"\n"));
                    assert!(!std::str::from_utf8(&bytes).unwrap().contains("PRIVATE_REJECTED_OVERRIDE"));
                    let records: Vec<serde_json::Value> = bytes.split(|&byte| byte == b'\n')
                        .filter(|line| !line.is_empty()).map(|line| serde_json::from_slice(line).unwrap()).collect();
                    assert_eq!(records.first().unwrap()["event"], "run_start");
                    for record in &records { assert_eq!(record["execution"], crate::batch::BATCH_EXECUTION); }
                    let documents: Vec<_> = records.iter().filter(|record| record["event"] == "doc").collect();
                    assert_eq!(documents.len(), 5);
                    assert_eq!(documents.iter().map(|record| record["request_seq"].as_u64().unwrap())
                        .collect::<Vec<_>>(), [1, 2, 5, 7, 8]);
                    assert_eq!(documents[0]["caller_id"], documents[3]["caller_id"]);
                    assert_eq!(documents[0]["epoch"], 1); assert_eq!(documents[3]["epoch"], 2);
                    assert_eq!(documents[0]["result"], documents[3]["result"]);
                    for document in &documents {
                        let result = &document["result"];
                        assert_eq!(result["execution"], if selected {
                            crate::tasks::source_planning::grouped_int4::INT4_SPARSE_SOURCE_EXECUTION
                        } else { crate::tasks::source_planning::grouped_int4::INT4_SOURCE_EXECUTION });
                        assert_eq!(result["result"]["task"], task.trim_end_matches("-batch"));
                        assert_eq!(result["result"]["result"]["numerics_profile"],
                            crate::native_engine::strict_int4::STRICT_INT4_PROFILE);
                    }
                    let errors: Vec<_> = records.iter().filter(|record| record["event"] == "doc_error").collect();
                    assert_eq!(errors.len(), 2);
                    assert_eq!(errors[0]["request_seq"], 3); assert_eq!(errors[0]["error"]["code"], "invalid_json");
                    assert_eq!(errors[1]["request_seq"], 4); assert_eq!(errors[1]["error"]["code"], "planning");
                    let footer = records.last().unwrap();
                    assert_eq!(footer["event"], "run_complete");
                    assert_eq!(footer["summary"], serde_json::to_value(summary).unwrap());
                    assert_eq!(summary.input_bytes, input.len() as u64);
                    assert_eq!(summary.requests, 8); assert_eq!(summary.succeeded, 5);
                    assert_eq!(summary.failed, 2); assert_eq!(summary.flushes, 2);
                    // Raw byte equality retains task-finalized evidence, passage
                    // IDs, UTF-8/scalar offsets, tokens, exact work and all errors.
                    // Dense and selected-row policies each retain their own baseline.
                    if let Some(expected) = &expected {
                        assert!(bytes.as_slice() == expected.as_slice(), "{task}, selected={selected}, threads={threads}");
                        assert_eq!(Some(summary), expected_summary);
                    } else { expected = Some(bytes); expected_summary = Some(summary); }
                    assert_eq!(engine.resources().memory_snapshot(), resident);
                    let closures = engine.resources().outstanding_closure_snapshot();
                    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
                }
            }
        }
        drop(model);
    }
    drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}
