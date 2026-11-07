//! Opt-in native parity definition; requires independently pinned model data.
use super::*;

#[test]
#[ignore = "requires controller-admitted INT4 snapshot and a bounded source extraction schema"]
fn model_present_selected_maps_and_chunk_cohorts_preserve_complete_semantic_output() {
    const CHILD: &str = "FNLP_INT4_SELECTED_MAP_PARITY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_SELECTED_MAP_PARITY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_selected_maps_and_chunk_cohorts_preserve_complete_semantic_output");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated selected-map fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "selected-map fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "fixture filter must execute the complete comparison");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model pin required");
    let schema = std::env::var("FNLP_INT4_TEST_EXTRACTION_SCHEMA")
        .expect("bounded source schema satisfiable in every chunk required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit supported kernel required");
    PackedInt4Kernel::detect(kernel.parse::<Int8KernelKind>().expect("canonical kernel")).unwrap();
    let source = "Alice visits Shanghai. Bob sees café 上海 上海. <tool_call> is source text.\r\n".repeat(3);
    let mut expected = None;
    let mut expected_work = None;
    for selected in [false, true] {
        for cohort in [None, Some("1"), Some("2"), Some("4")] {
            let mut argv = vec!["candidate", "source-int4", "extract-map", "--snapshot", &snapshot,
                "--expected-model-sha256", &pin, "--memory-mib", &memory, "--kernel", &kernel,
                "--schema", &schema, "--source-membership", "--context-tokens", "1024",
                "--max-new-tokens", "128", "--max-chunk-bytes", "32", "--timeout-seconds", "86400"];
            if selected { argv.extend(["--max-selected-rows", "166144"]); }
            if let Some(width) = cohort { argv.extend(["--cohort-chunks", width]); }
            let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
            let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
                else { panic!("native INT4 route required"); };
            let mut output = Vec::new();
            command.execute(&mut std::io::Cursor::new(source.as_bytes()), &mut output)
                .expect("every native chunk and the final document must complete");
            let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(value["model_content_sha256"], pin);
            assert_eq!(value["evidence"], "non_authoritative");
            let document = &value["output"];
            assert_eq!(document["execution"], if selected {
                crate::batch::extract::grouped_int4::long::INT4_EXTRACTION_MAP_SELECTED_EXECUTION
            } else { crate::batch::extract::grouped_int4::long::INT4_EXTRACTION_MAP_EXECUTION });
            assert_eq!(document["max_selected_rows"].as_u64(), selected.then_some(166144));
            if let Some(expected) = &expected_work { assert_eq!(&document["planned_model_work"], expected); }
            else { expected_work = Some(document["planned_model_work"].clone()); }
            let chunks = document["mapped"]["root"]["value"].as_array().unwrap();
            assert!(chunks.len() > 4);
            let semantic: Vec<_> = chunks.iter().map(|chunk| {
                let native = &chunk["native"];
                assert_eq!(native["execution"], if selected {
                    crate::tasks::extract::grouped_int4::INT4_SPARSE_EXTRACT_VERSION
                } else { crate::tasks::extract::grouped_int4::INT4_EXTRACT_VERSION });
                let mut result = native["result"].clone();
                result["output"].as_object_mut().unwrap().remove("projected_logits");
                serde_json::json!({"chunk_id":chunk["chunk_id"], "source_span":chunk["source_span"],
                    "original_fields":chunk["original_fields"], "result":result})
            }).collect();
            if let Some(expected) = &expected { assert_eq!(&semantic, expected); }
            else { expected = Some(semantic); }
        }
    }
    println!("{COMPLETE}");
}
