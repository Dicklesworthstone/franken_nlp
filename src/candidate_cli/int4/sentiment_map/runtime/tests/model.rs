//! Explicitly ignored complete CLI serial/cohort comparison on an owned model.
use super::*;

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
