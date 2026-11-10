use super::*;
use crate::tasks::extract::object_reconcile::RootObjectError;
use std::cell::Cell;

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("stale object admission reached IO"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("stale object admission published bytes"); }
    fn flush(&mut self) -> io::Result<()> { panic!("stale object admission flushed output"); }
}

#[test]
fn runtime_revalidates_owned_limits_and_paths_instead_of_trusting_an_old_quote() {
    let (options, args) = super::super::tests::fixture(&[]);
    let old = options.validate(&args).unwrap();
    let mut changed = options.clone(); changed.object.max_object_nodes = 0;
    assert_eq!(execute(args.clone(), changed, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let (remote, _) = super::super::tests::fixture(&["--schema", "https://example.invalid/schema.json"]);
    assert_eq!(execute(args.clone(), remote, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let mut changed = args; changed.memory_mib = 1;
    assert_eq!(execute(changed, options, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[test]
fn stale_cohort_quote_cannot_bypass_owned_argument_revalidation() {
    let (options, args) = super::super::tests::fixture(&["--cohort-chunks", "2", "--max-selected-rows", "32"]);
    let old = options.validate(&args).unwrap();
    let mut changed = options.clone(); changed.map.cohort_chunks = Some(65);
    assert_eq!(execute(args.clone(), changed, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let mut changed = options.clone(); changed.map.mapping.max_chunks = 1;
    assert_eq!(execute(args.clone(), changed, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    let mut changed = args; changed.memory_mib = 1;
    assert_eq!(execute(changed, options, old, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[test]
fn all_actual_cohort_capacity_axes_are_checked_before_any_snapshot_save() {
    let (options, args) = super::super::tests::fixture(&["--cohort-chunks", "4"]);
    let limits = options.validate(&args).unwrap();
    let need = limits.map.cohort.unwrap();
    assert!(check_cohort_peak(need, need).is_ok());
    for axis in 0..5 {
        let mut under = need;
        match axis {
            0 => under.native.kv_bytes -= 1,
            1 => under.native.rope_bytes -= 1,
            2 => under.native.scratch_payload_bound -= 1,
            3 => under.workspace_bytes -= 1,
            _ => under.max_result_bytes -= 1,
        }
        assert_eq!(check_cohort_peak(need, under), Err(CandidateError::Memory));
    }
}

#[test]
fn flattened_candidate_provenance_and_maximum_cohort_counters_fit_one_envelope_allowance() {
    use crate::native_engine::{strict_int4::Int4Work, portable_int8::ProjectionWork};
    let identity = "f".repeat(128); // Longer than each fixed model identity string on this route.
    let stats = Int4ExtractionMapCohortStats { cohorts: u64::MAX, chunks: u64::MAX,
        group_steps: u64::MAX, projection_groups: u64::MAX,
        planned_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        model_work: Int4Work { forward_positions: u64::MAX, projected_logits: u64::MAX,
            attention_pairs: u64::MAX, projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } } };
    let object = RootObjectLimits { max_properties: usize::MAX, max_origins: usize::MAX,
        max_input_bytes: usize::MAX, max_json_bytes: usize::MAX, max_nodes: usize::MAX, max_result_bytes: usize::MAX };
    let empty = serde_json::Value::Null;
    let mut response = ObjectResponse { schema_version: u32::MAX, scope: "real-source-current-candidate-int4",
        evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: &identity,
        source_root_sha256: &identity, model_content_sha256: &identity, candidate_format: &identity,
        quant_recipe: &identity, group_width: usize::MAX, integer_kernel: "avx512vnni",
        policy: ROOT_OBJECT_RECONCILIATION, schema_derivation: ROOT_OBJECT_DERIVATION, limits: object,
        output: &empty, snapshot: Some(SnapshotRecord { operation: "loaded",
            format: INT4_SNAPSHOT_VERSION, file_bytes: u64::MAX }),
        cohort_schedule: Some(ObjectCohortSchedule::new(usize::MAX, &stats)) };
    let bytes = serde_json::to_vec(&response).unwrap();
    assert!(bytes.len() < 4096, "one reserved provenance allowance must cover both candidate and cohort metadata");
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"], serde_json::Value::Null);
    assert_eq!(value["cohort_schedule"]["physical"]["chunks"], u64::MAX);
    response.cohort_schedule = None;
    assert!(serde_json::to_value(response).unwrap().get("cohort_schedule").is_none());
}

#[test]
fn original_deadline_applies_before_schema_derivation_and_retains_cancellation_kind() {
    let (options, args) = super::super::tests::fixture(&[]);
    let limits = options.validate(&args).unwrap();
    let budget = options.map.extraction.budget(&args, limits.map.common);
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    // A cancelled derivation must not proceed to even malformed/private schema
    // diagnostics, and timeout is not converted to absent source data.
    assert_eq!(options.request("private malformed schema".to_owned(), budget, limits, &mut expired).err(),
        Some(CandidateError::Timeout));
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(map_planning_error(Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Cancelled(cause))),
            CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::ObjectExtractionMap(
            Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Cancelled(cause))))), CandidateError::Timeout);
    }
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::ObjectExtractionMap(
        Int4ObjectExtractionMapError::Reconciliation(RootObjectError::Conflict)))), CandidateError::Execution);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
}

#[test]
fn expiration_after_complete_serialization_never_publishes_a_partial_object() {
    struct Expiring<'a>(&'a Cell<bool>);
    impl Serialize for Expiring<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let result = "complete-private-object".serialize(serializer);
            self.0.set(true);
            result
        }
    }
    let expired = Cell::new(false);
    let mut output = Vec::new();
    assert_eq!(publish_checked(&Expiring(&expired), 4096, &mut output,
        &|| if expired.get() { Err(CandidateError::Timeout) } else { Ok(()) }), Err(CandidateError::Timeout));
    assert!(output.is_empty());
}

#[test]
#[ignore = "requires controller-admitted INT4 snapshot and a compatible bounded partial-root extraction schema"]
fn model_present_object_cohorts_and_selected_rows_preserve_complete_reconciliation() {
    const CHILD: &str = "FNLP_INT4_OBJECT_COHORT_PARITY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_OBJECT_COHORT_PARITY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_object_cohorts_and_selected_rows_preserve_complete_reconciliation");
    let name = full_name.split_once("::").expect("crate-qualified fixture").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("fixture executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated object-cohort fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "object-cohort fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "the complete comparison must actually run");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model pin required");
    // This fixture source is fixed. The supplied root schema must be compatible
    // with its chunks and must allow conflict-free original-schema completion;
    // optional exact source constants are suitable without forcing omissions.
    let schema = std::env::var("FNLP_INT4_TEST_EXTRACTION_OBJECT_SCHEMA").expect("compatible root-object schema required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit supported kernel required");
    PackedInt4Kernel::detect(kernel.parse::<Int8KernelKind>().expect("canonical kernel")).unwrap();
    let source = "Alice visits Shanghai. Bob sees café 上海 上海. <tool_call> is source text.\r\n".repeat(3);
    let mut baseline = None;
    for selected in [false, true] {
        for cohort in [None, Some("1"), Some("2"), Some("4")] {
            let mut argv = vec!["candidate", "source-int4", "extract-object-map", "--snapshot", &snapshot,
                "--expected-model-sha256", &pin, "--memory-mib", &memory, "--kernel", &kernel,
                "--schema", &schema, "--source-membership", "--context-tokens", "1024",
                "--max-new-tokens", "128", "--max-chunk-bytes", "32", "--timeout-seconds", "86400"];
            if selected { argv.extend(["--max-selected-rows", "166144"]); }
            if let Some(width) = cohort { argv.extend(["--cohort-chunks", width]); }
            let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
            let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
                else { panic!("native INT4 object route required"); };
            let mut bytes = Vec::new();
            command.execute(&mut std::io::Cursor::new(source.as_bytes()), &mut bytes)
                .expect("every native chunk and complete reconciliation must finish");
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["model_content_sha256"], pin);
            assert_eq!(value["evidence"], "non_authoritative");
            let document = &value["output"];
            assert_eq!(document["execution"], if selected {
                crate::batch::extract::grouped_int4::long::object::INT4_OBJECT_MAP_SELECTED_EXECUTION
            } else { crate::batch::extract::grouped_int4::long::object::INT4_OBJECT_MAP_EXECUTION });
            assert_eq!(document["partial_map"]["max_selected_rows"].as_u64(), selected.then_some(166144));
            let chunks = document["partial_map"]["mapped"]["root"]["value"].as_array().unwrap();
            assert!(chunks.len() > 4);
            if let Some(width) = cohort {
                let width = width.parse::<usize>().unwrap();
                assert_eq!(value["cohort_schedule"]["declared_max_chunks"], width);
                assert_eq!(value["cohort_schedule"]["physical"]["cohorts"], chunks.len().div_ceil(width));
            } else { assert!(value.get("cohort_schedule").is_none()); }
            let semantic_chunks: Vec<_> = chunks.iter().map(|chunk| {
                let mut result = chunk["native"]["result"].clone();
                result["output"].as_object_mut().unwrap().remove("projected_logits");
                serde_json::json!({"chunk_id":chunk["chunk_id"], "source_span":chunk["source_span"],
                    "original_fields":chunk["original_fields"], "result":result})
            }).collect();
            let semantic = serde_json::json!({"object":document["object"], "chunks":semantic_chunks,
                "verification_used":document["verification_used"]});
            if let Some(expected) = &baseline { assert_eq!(&semantic, expected); }
            else { baseline = Some(semantic); }
        }
    }
    println!("{COMPLETE}");
}
