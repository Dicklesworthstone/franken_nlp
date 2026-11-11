//! Explicitly ignored complete CLI serial/cohort comparison on an owned model.
use super::*;

#[test]
#[ignore = "run alone with controller-admitted independently pinned INT4 snapshot and serial/2/4-thread complete-document inference"]
fn document_teams_match_serial_multiaxis_results_and_release_charges() {
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model pin required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit hardware-supported kernel required");
    let width = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").unwrap_or_else(|_| "128".to_owned());
    let mut command = command(&["--compute-threads", "4", "--context-tokens", "512",
        "--max-chunk-bytes", "32", "--max-chunks", "16", "--group-width", &width,
        "--kernel", &kernel, "--timeout-seconds", "86400"]);
    command.args.source_dir = None; command.args.source_manifest = None;
    command.args.snapshot = Some(PathBuf::from(snapshot)); command.args.expected_model_sha256 = Some(pin);
    command.args.memory_mib = memory.parse().unwrap();
    let Int4Task::SentimentMap(options) = command.task else { panic!("sentiment map required"); };
    let limits = options.validate_with_threads(&command.args).unwrap();
    let common = limits.common;
    assert!(crate::installed_process_resources().is_none(), "model gate requires its own fresh process");
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common)).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let lease = engine.resources().acquire_lease();
    let preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .unwrap().commit().unwrap();
    let model = crate::candidate_cli::int4::runtime::load_model(&engine, &command.args, common,
        Duration::from_secs(3600), CancellationToken::default()).unwrap();
    let resident = engine.resources().memory_snapshot();
    let source = "é上海 positive then negative.\r\n".repeat(2);
    let wire = serde_json::to_string(&serde_json::json!({"document":source,"axes":["valence","arousal"]})).unwrap();
    let budget = options.budget(common).unwrap();
    let (request, policy) = sentiment::parse_request(&wire, budget, command.args.max_input_bytes).unwrap();
    options.admit_request(&request).unwrap();
    let config = Int4SentimentMapHostConfig {
        native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: Duration::from_secs(86_400),
                max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
        planning: limits.planning, mapping: limits.mapping, options: sentiment_options(policy).unwrap(),
        preparation_reserve_bytes: common.preparation_bytes,
    };
    let expected = {
        let _preflight = lease.reserve(MemoryClass::JobBuffers,
            sum(&[limits.memory.input_bytes, limits.memory.workspace_bytes]).unwrap()).unwrap().commit().unwrap();
        let planner = Int4SentimentPlanner::for_model(model.model_identity(), config.options, budget, limits.planning).unwrap();
        planner.plan_document_with_control(&request, limits.mapping,
            &mut DeadlineControl { started: Instant::now(), deadline: Duration::from_secs(3600) })
            .unwrap().preflight_metadata()
    };
    assert!(expected.chunk_count() > 1);
    let mut serial_bytes = None;
    for threads in [1, 2, 4] {
        let output = if threads == 1 {
            compute::Completion::serial(engine.map_int4_sentiment(&model, request.clone(), config,
                common.kernel, CancellationToken::default()).unwrap())
        } else {
            compute::Completion::Team(engine.map_int4_sentiment_with_threads(&model, request.clone(), config,
                common.kernel, threads, CancellationToken::default()).unwrap())
        };
        expected.verify_completed(output.result()).unwrap();
        assert_eq!(output.result().source_span.byte_end, source.len());
        assert_eq!(output.result().source_span.scalar_end, source.chars().count());
        assert_eq!(output.result().model_work, output.result().planned_model_work);
        assert!(output.result().chunks.iter().all(|chunk| chunk.run.head_count == request.axes.len()));
        let bytes = serde_json::to_vec(output.result()).unwrap();
        if let Some(serial) = &serial_bytes { assert_eq!(&bytes, serial); }
        else { serial_bytes = Some(bytes); }
        if threads > 1 {
            let receipt = output.compute().unwrap();
            assert_eq!(receipt.compute_threads, threads);
            assert!(receipt.completed_worker_rows > 0);
            assert!(receipt.completed_rows > receipt.completed_worker_rows);
            assert!(receipt.completed_projections > expected.chunk_count() as u64);
        } else { assert!(output.compute().is_none()); }
        drop(output);
        assert_eq!(engine.resources().memory_snapshot(), resident);
    }
    drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}

#[test]
#[ignore = "requires controller-admitted pinned INT4 snapshot, explicit kernel and memory; no task-quality claim"]
fn complete_document_cli_preserves_utf8_lineage_serial_cohort_results_and_whole_work_refusal() {
    const CHILD: &str = "FNLP_INT4_SENTIMENT_MAP_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact",
                "candidate_cli::int4::sentiment_map::runtime::tests::model::complete_document_cli_preserves_utf8_lineage_serial_cohort_results_and_whole_work_refusal",
                "--nocapture"])
            .env(CHILD, "1").status().unwrap();
        assert!(status.success()); return;
    }
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model pin required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit process memory required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit hardware-supported kernel required");
    let width = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").unwrap_or_else(|_| "128".to_owned());
    let source = "é上海 private-map-fixture positive then negative.\r\n".repeat(4);
    let wire = serde_json::to_vec(&serde_json::json!({"document":source,"axes":["valence","arousal"]})).unwrap();
    let make = |extra: &[&str]| {
        let mut flags = vec!["--context-tokens", "512", "--max-chunks", "16", "--max-chunk-bytes", "32",
            "--group-width", &width, "--kernel", &kernel];
        flags.extend_from_slice(extra);
        let mut command = command(&flags);
        command.args.source_dir = None; command.args.source_manifest = None;
        command.args.snapshot = Some(PathBuf::from(&snapshot)); command.args.expected_model_sha256 = Some(pin.clone());
        command.args.memory_mib = memory.parse().unwrap(); command
    };
    let run = |command: Int4Command| {
        let mut output = Vec::new(); command.execute(&mut io::Cursor::new(&wire), &mut output).unwrap();
        serde_json::from_slice::<serde_json::Value>(&output).unwrap()
    };
    let serial = run(make(&[])); let grouped = run(make(&["--cohort-chunks", "3"]));
    assert_eq!(serial["output"], grouped["output"]);
    assert_eq!(serial["protocol"], "fnlp-candidate-int4-sentiment-map-v1");
    assert_eq!(serial["evidence"], "non_authoritative"); assert!(serial.get("cohort").is_none());
    let output = &serial["output"];
    assert_eq!(output["single_context_equivalence"], false);
    assert_eq!(output["planned_model_work"], output["model_work"]);
    let chunks = output["chunks"].as_array().unwrap(); assert!(chunks.len() > 3);
    let mut bytes = 0; let mut scalars = 0;
    for (index, chunk) in chunks.iter().enumerate() {
        let span = &chunk["source_span"];
        assert_eq!(chunk["chunk_id"], index); assert_eq!(span["byte_start"], bytes); assert_eq!(span["scalar_start"], scalars);
        bytes = span["byte_end"].as_u64().unwrap() as usize;
        scalars = span["scalar_end"].as_u64().unwrap() as usize;
        assert!(source.is_char_boundary(bytes)); assert_eq!(chunk["run"]["head_count"], 2);
    }
    assert_eq!(bytes, source.len()); assert_eq!(scalars, source.chars().count());
    assert_eq!(output["aggregate"]["source_bytes"], bytes); assert_eq!(output["aggregate"]["source_scalars"], scalars);
    assert_eq!(grouped["cohort"]["chunk_cohorts"], chunks.len().div_ceil(3));
    assert!(grouped["cohort"]["group_steps"].as_u64().unwrap() > 0);
    assert!(grouped["cohort"]["projection_groups"].as_u64().unwrap() > 0);
    let private = serde_json::to_string(&grouped).unwrap(); assert!(!private.contains("private-map-fixture"));
    for forbidden in ["prompt_digest", "taskir_digest", "source_dir", "snapshot_path"] { assert!(!private.contains(forbidden)); }
    let actual = output["model_work"]["forward_positions"].as_u64().unwrap(); assert!(actual > 1);
    let cap = (actual - 1).to_string(); let mut refused = Vec::new();
    let result = make(&["--max-forward-positions", &cap]).execute(&mut io::Cursor::new(&wire), &mut refused);
    assert!(matches!(result, Err(CandidateError::Planning | CandidateError::Execution)));
    assert!(refused.is_empty(), "whole-document failure must not publish a completed prefix");
}
