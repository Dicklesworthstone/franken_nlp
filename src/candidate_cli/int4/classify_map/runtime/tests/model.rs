//! Explicit owned-model gate; never run by an unarmed code checkpoint.
use super::*;

#[test]
#[ignore = "run alone with controller-admitted independently pinned INT4 snapshot and serial/2/4-thread complete-document inference"]
fn document_teams_match_serial_exclusive_multilabel_and_release_charges() {
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
    let Int4Task::ClassifyMap(options) = command.task else { panic!("classification map required"); };
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
    let budget = options.budget(common, limits.planning.max_labels).unwrap();
    let config = Int4ClassificationMapHostConfig {
        native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: Duration::from_secs(86_400),
                max_checkpoints: command.args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
        planning: limits.planning, mapping: limits.mapping, preparation_reserve_bytes: common.preparation_bytes,
    };
    let source = "é上海 billing then refund.\r\n".repeat(2);
    for (mode, heads) in [("exclusive", 1), ("multi_label", 2)] {
        let wire = serde_json::to_string(&serde_json::json!({"document":source,"mode":mode,
            "labels":[{"id":"billing","description":"Invoices"},{"id":"refund","description":"Refunds"}]})).unwrap();
        let request = classify::parse_request(&wire, budget, command.args.max_input_bytes).unwrap();
        options.admit_request(&request).unwrap();
        let expected = {
            let _preflight = lease.reserve(MemoryClass::JobBuffers,
                sum(&[limits.memory.input_bytes, limits.memory.workspace_bytes]).unwrap()).unwrap().commit().unwrap();
            let planner = Int4ClassificationPlanner::for_model(model.model_identity(), budget, limits.planning).unwrap();
            planner.plan_document_with_control(&request, limits.mapping,
                &mut DeadlineControl { started: Instant::now(), deadline: Duration::from_secs(3600) })
                .unwrap().preflight_metadata()
        };
        assert!(expected.chunk_count() > 1);
        let mut serial_bytes = None;
        for threads in [1, 2, 4] {
            let output = if threads == 1 {
                compute::Completion::serial(engine.map_int4_classify(&model, request.clone(), config,
                    common.kernel, CancellationToken::default()).unwrap())
            } else {
                compute::Completion::Team(engine.map_int4_classify_with_threads(&model, request.clone(), config,
                    common.kernel, threads, CancellationToken::default()).unwrap())
            };
            expected.verify_completed(output.result()).unwrap();
            assert_eq!(output.result().source_span.byte_end, source.len());
            assert_eq!(output.result().source_span.scalar_end, source.chars().count());
            assert_eq!(output.result().model_work, output.result().planned_model_work);
            assert!(output.result().chunks.iter().all(|chunk| chunk.run.head_count == heads));
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
    }
    drop(model); drop(preparation); drop(lease);
    assert_eq!(engine.resources().memory_snapshot(), idle);
}

#[test]
#[ignore = "requires controller-admitted pinned sources, a NEW snapshot, explicit memory and supported kernel; no quality claim"]
fn complete_classification_map_reaches_sources_and_snapshot_without_losing_any_source_chunk() {
    const CHILD: &str = "FNLP_INT4_CLASSIFY_MAP_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact",
                "candidate_cli::int4::classify_map::runtime::tests::model::complete_classification_map_reaches_sources_and_snapshot_without_losing_any_source_chunk",
                "--nocapture"])
            .env(CHILD, "1").status().unwrap();
        assert!(status.success()); return;
    }
    let source_dir = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let snapshot = std::env::var("FNLP_INT4_TEST_CLASSIFICATION_MAP_SNAPSHOT").expect("explicit NEW snapshot required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit process memory required").parse().unwrap();
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit hardware-supported kernel required");
    let width = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").unwrap_or_else(|_| "128".to_owned());
    assert!(!std::path::Path::new(&snapshot).exists(), "an existing path is never removed or overwritten");
    let source = "é上海 private-map-fixture billing then refund.\r\n".repeat(4);
    let wire = |mode| serde_json::to_vec(&serde_json::json!({"document":source,"mode":mode,
        "labels":[{"id":"billing","description":"Invoices"},{"id":"refund","description":"Refunds"}]})).unwrap();
    let make_source = |extra: &[&str]| {
        let mut flags = vec!["--context-tokens", "512", "--max-chunk-bytes", "32",
            "--group-width", &width, "--kernel", &kernel];
        flags.extend_from_slice(extra);
        let mut command = command(&flags);
        command.args.source_dir = Some(PathBuf::from(&source_dir));
        command.args.source_manifest = Some(PathBuf::from(&manifest));
        command.args.memory_mib = memory; command
    };
    let exclusive = wire("exclusive");
    // A known complete-document planning refusal must precede the requested
    // persistent side effect. The same untouched destination is used below.
    let mut refused = Vec::new();
    assert_eq!(make_source(&["--max-chunks", "1", "--save-snapshot", &snapshot])
        .execute(&mut io::Cursor::new(&exclusive), &mut refused), Err(CandidateError::Planning));
    assert!(refused.is_empty()); assert!(!std::path::Path::new(&snapshot).exists());
    let run = |command: Int4Command, wire: &[u8]| {
        let mut output = Vec::new(); command.execute(&mut io::Cursor::new(wire), &mut output).unwrap();
        serde_json::from_slice::<serde_json::Value>(&output).unwrap()
    };
    let original = run(make_source(&["--max-chunks", "16", "--cohort-chunks", "3", "--save-snapshot", &snapshot]), &exclusive);
    assert_eq!(original["snapshot"]["operation"], "saved");
    assert_eq!(original["cohort"]["max_chunks"], 3);
    // This pin comes from successful independently checked source conversion,
    // never from the snapshot file's own unauthenticated header.
    let pin = original["model_content_sha256"].as_str().unwrap().to_owned();
    let make_snapshot = |extra: &[&str]| {
        let mut flags = vec!["--max-chunks", "16"]; flags.extend_from_slice(extra);
        let mut command = make_source(&flags);
        command.args.source_dir = None; command.args.source_manifest = None;
        command.args.snapshot = Some(PathBuf::from(&snapshot)); command.args.expected_model_sha256 = Some(pin.clone());
        command
    };
    for (mode, heads) in [("exclusive", 1), ("multi_label", 2)] {
        let output = run(make_snapshot(&[]), &wire(mode));
        let grouped = run(make_snapshot(&["--cohort-chunks", "3"]), &wire(mode));
        assert_eq!(output["protocol"], "fnlp-candidate-int4-classify-map-v1");
        assert_eq!(output["evidence"], "non_authoritative"); assert_eq!(output["snapshot"]["operation"], "loaded");
        assert_eq!(output["model_content_sha256"], pin);
        let semantic = &output["output"];
        assert_eq!(semantic, &grouped["output"]);
        assert_eq!(canonjson::canonical_bytes(semantic).unwrap(), canonjson::canonical_bytes(&grouped["output"]).unwrap());
        assert!(output.get("cohort").is_none()); assert!(grouped["output"].get("output").is_none());
        assert_eq!(semantic["single_context_equivalence"], false);
        assert_eq!(semantic["planned_model_work"], semantic["model_work"]);
        assert_eq!(semantic["aggregate"]["result"]["mode"], mode);
        let chunks = semantic["chunks"].as_array().unwrap(); assert!(chunks.len() > 1);
        assert_eq!(grouped["cohort"]["max_chunks"], 3);
        assert_eq!(grouped["cohort"]["chunk_cohorts"], chunks.len().div_ceil(3));
        assert!(grouped["cohort"]["group_steps"].as_u64().unwrap() > 0);
        assert!(grouped["cohort"]["projection_groups"].as_u64().unwrap() > 0);
        let mut bytes = 0; let mut scalars = 0;
        for (index, chunk) in chunks.iter().enumerate() {
            let span = &chunk["source_span"];
            assert_eq!(chunk["chunk_id"], index); assert_eq!(span["byte_start"], bytes); assert_eq!(span["scalar_start"], scalars);
            bytes = span["byte_end"].as_u64().unwrap() as usize;
            scalars = span["scalar_end"].as_u64().unwrap() as usize;
            assert!(source.is_char_boundary(bytes)); assert_eq!(chunk["run"]["head_count"], heads);
            assert_eq!(chunk["run"]["result"]["mode"], mode);
        }
        assert_eq!(bytes, source.len()); assert_eq!(scalars, source.chars().count());
        assert_eq!(semantic["aggregate"]["source_bytes"], bytes); assert_eq!(semantic["aggregate"]["source_scalars"], scalars);
        if mode == "exclusive" { assert_eq!(semantic, &original["output"]); }
        let serialized = serde_json::to_string(&grouped).unwrap(); assert!(!serialized.contains("private-map-fixture"));
        for forbidden in ["prompt_digest", "taskir_digest", "source_dir", "snapshot_path"] { assert!(!serialized.contains(forbidden)); }
    }
    let actual = original["output"]["model_work"]["forward_positions"].as_u64().unwrap(); assert!(actual > 1);
    let cap = (actual - 1).to_string();
    for extra in [vec!["--max-forward-positions", cap.as_str()],
        vec!["--max-forward-positions", cap.as_str(), "--cohort-chunks", "3"]] {
        let mut refused = Vec::new();
        let result = make_snapshot(&extra).execute(&mut io::Cursor::new(&exclusive), &mut refused);
        assert!(matches!(result, Err(CandidateError::Planning | CandidateError::Execution)));
        assert!(refused.is_empty(), "whole-document failure must not publish a completed prefix");
    }
    // The explicitly requested snapshot remains intact after the fixture.
}
