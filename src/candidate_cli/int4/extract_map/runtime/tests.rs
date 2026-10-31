//! Real pinned schema/chunk preparation; no fixture claims model execution.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

fn fixture(width: usize) -> (Int4ExtractionBatchPlanner, ExtractionBatchArgs, Int4ExtractionMapLimits) {
    let width_arg = width.to_string();
    let command = super::super::tests::command(&["--max-chunk-bytes", "16", "--max-new-tokens", "16",
        "--group-width", &width_arg]);
    let Int4Task::ExtractMap(options) = command.task else { panic!("wrong map task"); };
    let limits = options.validate(&command.args).unwrap();
    let budget = options.extraction.budget(&command.args, limits.common);
    let d = Sha256Digest::of_bytes(b"metadata-only-INT4-map-CLI-fixture");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, width,
    )).unwrap());
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: packing, tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "uncompiled".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, width,
        budget, options.extraction.compiler(), options.extraction.source_limits(), None).unwrap();
    let request = options.request(r#"{"type":"string","maxLength":32}"#.to_owned(), budget).unwrap();
    (planner, request, limits.mapping)
}
struct Control { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop_at).then_some(self.cause)
    }
}
fn continuing() -> Control { Control { calls: 0, stop_at: usize::MAX, cause: DecodeCancellationKind::PollQuota } }
const SOURCE: &str = "é\r\n上海 source <tool_call> and another paragraph\r\n";

#[test]
fn complete_source_geometry_and_group_work_survive_preflight_and_host_replanning() {
    let (planner, request, mapping) = fixture(32);
    let expected = preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1);
    let span = expected.source_span();
    assert_eq!((span.byte_start, span.byte_end), (0, SOURCE.len()));
    assert_eq!((span.scalar_start, span.scalar_end), (0, SOURCE.chars().count()));
    assert_eq!(expected.group_width(), 32);
    let second = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
    assert_eq!(expected, second.preflight_metadata());
    assert_eq!(second.execution_identities().len(), expected.chunk_count());
    assert_eq!(expected.reserved_mask_visits(), mapping.mapping.mask_visits_per_chunk * expected.chunk_count() as u64);
    let (other, request, mapping) = fixture(128);
    let other = preflight(SOURCE, &request, &other, mapping, &mut continuing()).unwrap();
    assert_eq!(expected.planned_work().forward_positions, other.planned_work().forward_positions);
    assert_eq!(expected.planned_work().projections.multiply_accumulates, other.planned_work().projections.multiply_accumulates);
    assert!(expected.planned_work().projections.dot_products > other.planned_work().projections.dot_products);
}

#[test]
fn chunk_cohort_scheduling_preserves_all_prepared_identities_and_full_document_preflight() {
    let (planner, request, mapping) = fixture(32);
    let serial = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
    let expected = serial.preflight_metadata();
    let identities: Vec<_> = serial.execution_identities().cloned().collect();
    for chunks in [1, 2, 4, 64] {
        let mut scheduled = mapping;
        scheduled.mapping.reduction.map_batch_chunks = chunks;
        let grouped = planner.plan_document_with_control(SOURCE, &request, scheduled, &mut continuing()).unwrap();
        assert_eq!(grouped.preflight_metadata(), expected);
        assert_eq!(grouped.execution_identities().cloned().collect::<Vec<_>>(), identities);
        assert_eq!(grouped.max_kv_bytes(), request.budget.max_kv_bytes);
        assert_eq!(grouped.max_result_bytes(), serial.max_result_bytes());
    }
}

#[test]
fn every_native_work_axis_and_mask_cap_refuses_the_complete_preflight() {
    let (planner, request, mapping) = fixture(32);
    let expected = preflight(SOURCE, &request, &planner, mapping, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut cap = mapping;
        cap.mapping.max_model_work = expected.planned_work();
        cap.mapping.max_mask_visits = expected.reserved_mask_visits();
        match axis { 0 => cap.mapping.max_model_work.forward_positions -= 1,
            1 => cap.mapping.max_model_work.projected_logits -= 1,
            2 => cap.mapping.max_model_work.attention_pairs -= 1,
            3 => cap.mapping.max_model_work.projections.dot_products -= 1,
            4 => cap.mapping.max_model_work.projections.multiply_accumulates -= 1,
            _ => cap.mapping.max_mask_visits -= 1 }
        assert_eq!(preflight(SOURCE, &request, &planner, cap, &mut continuing()), Err(CandidateError::Planning));
    }
    let mut exact = mapping;
    exact.mapping.max_model_work = expected.planned_work(); exact.mapping.max_mask_visits = expected.reserved_mask_visits();
    assert_eq!(preflight(SOURCE, &request, &planner, exact, &mut continuing()).unwrap(), expected);
}

#[test]
fn exact_schema_decimals_and_whitespace_bind_each_chunk_without_rewriting() {
    let (planner, mut request, mapping) = fixture(32);
    for raw in [r#"{"type":"number","const":0.12345678901234567890123456789012345678}"#,
        " {\"type\":\"number\",\"const\":0.12345678901234567890123456789012345679}\n"] {
        request.schema = raw.to_owned();
        let prepared = planner.plan_document_with_control(SOURCE, &request, mapping, &mut continuing()).unwrap();
        for identity in prepared.execution_identities() {
            assert_eq!(identity.schema_digest, Sha256Digest::of_bytes(raw.as_bytes()));
            assert_eq!(identity.task_spec, "extract-v1");
            assert_eq!(identity.numerics_profile, NumericsProfile::StrictQuantized { version: 2 });
        }
        assert_eq!(request.schema, raw);
    }
    for raw in ["{", r#"{"type":"boolean","type":"string"}"#, r#"{"$ref":"remote"}"#] {
        request.schema = raw.to_owned();
        assert!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()).is_err());
    }
}

#[test]
fn whole_original_source_cannot_be_truncated_and_blank_chunks_are_retained() {
    let (planner, mut request, mut mapping) = fixture(32);
    assert_eq!(preflight("", &request, &planner, mapping, &mut continuing()), Err(CandidateError::Input));
    mapping.mapping.chunks.max_chunks = 1;
    assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut continuing()), Err(CandidateError::Planning));
    mapping.mapping.chunks.max_chunks = 64;
    request.grounding = crate::batch::extract::ExtractionBatchGrounding::SourceMembership;
    request.schema = r#"{"type":"string","maxLength":32,"x-fnlp-source":"verbatim"}"#.to_owned();
    let text = "é\r\n                 上海";
    let expected = preflight(text, &request, &planner, mapping, &mut continuing()).unwrap();
    assert!(expected.chunk_count() > 1); assert_eq!(expected.source_span().byte_end, text.len());
    request.schema = r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#.to_owned();
    assert!(preflight(text, &request, &planner, mapping, &mut continuing()).is_err());
}

#[test]
fn caller_cancellation_survives_final_preflight_and_never_gets_a_fresh_budget() {
    let (planner, request, mapping) = fixture(32);
    let mut complete = continuing();
    preflight(SOURCE, &request, &planner, mapping, &mut complete).unwrap();
    assert!(complete.calls > 2);
    for stop_at in [1, 2, complete.calls] {
        let mut control = Control { stop_at, ..continuing() };
        assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut control), Err(CandidateError::Execution));
    }
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        let mut control = Control { calls: 0, stop_at: 1, cause };
        assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut control), Err(CandidateError::Timeout));
    }
    complete.stop_at = complete.calls + 1;
    assert_eq!(preflight(SOURCE, &request, &planner, mapping, &mut complete), Err(CandidateError::Execution));
}

#[test]
fn cli_workspace_quote_matches_actual_host_for_complete_occurrence_vectors() {
    let command = super::super::tests::command(&[]);
    let Int4Task::ExtractMap(options) = command.task else { panic!("wrong map task"); };
    let limits = options.validate(&command.args).unwrap();
    let budget = options.extraction.budget(&command.args, limits.common);
    let config = Int4ExtractionMapConfig {
        request: options.request(r#"{"type":"boolean"}"#.to_owned(), budget).unwrap(), limits: limits.mapping,
        native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.common.preparation_bytes, reduction_reserve_bytes: limits.reduction_reserve_bytes,
    };
    for chunks in [1, limits.mapping.mapping.chunks.max_chunks] {
        assert_eq!(config.workspace_bytes(chunks).unwrap(),
            workspace_bytes(budget, limits.mapping, chunks, limits.reduction_reserve_bytes).unwrap());
    }
    assert_eq!(config.workspace_bytes(limits.mapping.mapping.chunks.max_chunks).unwrap(), limits.workspace_bytes);
}

#[test]
fn cli_cohort_quote_matches_actual_host_without_widening_each_chunk() {
    for chunks in [1, 2, 4, 64] {
        let value = chunks.to_string();
        let mut command = super::super::tests::command(&["--cohort-chunks", &value, "--max-result-bytes", "65536"]);
        command.args.memory_mib = 262_144;
        let Int4Task::ExtractMap(options) = command.task else { panic!("wrong map task"); };
        let limits = options.validate(&command.args).unwrap();
        let budget = options.extraction.budget(&command.args, limits.common);
        let config = Int4ExtractionMapConfig {
            request: options.request(r#"{"type":"boolean"}"#.to_owned(), budget).unwrap(), limits: limits.mapping,
            native: NativeLimits { context_tokens: command.args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: CLEANUP_BYTES } },
            preparation_reserve_bytes: limits.common.preparation_bytes, reduction_reserve_bytes: limits.reduction_reserve_bytes,
        };
        let quote = config.cohort_requirements(command.args.group_width).unwrap();
        assert_eq!(Some(quote), limits.cohort);
        assert_eq!(quote.native.kv_bytes, chunks as u64 * budget.max_kv_bytes);
        assert_eq!(config.workspace_bytes(limits.mapping.mapping.chunks.max_chunks).unwrap(), limits.workspace_bytes);
    }
}

#[test]
fn changed_cohort_options_are_rechecked_before_private_files_or_runtime_access() {
    struct NoIo;
    impl Read for NoIo {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused map read private input"); }
    }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused map published output"); }
        fn flush(&mut self) -> io::Result<()> { panic!("refused map flushed output"); }
    }
    for chunks in [0, 65, usize::MAX] {
        let command = super::super::tests::command(&[]);
        let Int4Task::ExtractMap(mut options) = command.task else { panic!("wrong map task"); };
        let quote = options.validate(&command.args).unwrap();
        options.mapping.cohort_chunks = Some(chunks);
        assert_eq!(execute(command.args, options, quote, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn publication_preserves_independent_json_strings_and_original_coordinates() {
    let exact = "{\"number\":12345678901234567890123456789012345678}";
    let output = serde_json::json!({"execution":"fixture-only", "chunks":[
        {"json":exact,"byte_start":0,"byte_end":10,"scalar_start":0,"scalar_end":6},
        {"json":"{}","byte_start":10,"byte_end":14,"scalar_start":6,"scalar_end":10}
    ]});
    let response = SourceResponse { schema_version: 1, scope: "real-source-current-candidate-int4",
        evidence: "non_authoritative", model_id: "Nanbeige4.2-3B", source_revision: "fixture",
        source_root_sha256: "fixture-source", model_content_sha256: "fixture-model",
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: "fixture:group=32", group_width: 32,
        integer_kernel: "s_scalar_i32", output: &output, snapshot: None, cohort_schedule: None };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"]["chunks"][0]["json"], exact);
    assert_eq!(value["output"]["chunks"][1]["byte_start"], 10);
    assert_eq!(value["output"]["chunks"].as_array().unwrap().len(), 2);
    assert_eq!(value["evidence"], "non_authoritative");
    for absent in ["source_dir", "source_manifest", "schema", "snapshot_path", "prompt_digest", "cohort_schedule"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new(); assert!(publish(&response, 1, &mut refused).is_err());
    assert!(refused.is_empty());
}

#[test]
#[ignore = "requires controller-admitted independently pinned snapshot and a bounded chunk-compatible extraction schema"]
fn model_present_chunk_cohorts_preserve_complete_document_values_and_lineage() {
    const CHILD: &str = "FNLP_INT4_EXTRACTION_MAP_PARITY_CHILD";
    const COMPLETE: &str = "FNLP_INT4_EXTRACTION_MAP_PARITY_COMPLETE";
    let full_name = concat!(module_path!(), "::model_present_chunk_cohorts_preserve_complete_document_values_and_lineage");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated extraction map parity fixture must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated extraction map fixture failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact fixture filter ran no completed fixture");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    let snapshot = std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit existing snapshot required");
    let pin = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin required");
    let schema = std::env::var("FNLP_INT4_TEST_EXTRACTION_SCHEMA")
        .expect("explicit bounded schema satisfiable for every independent source chunk required");
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required");
    let kernel = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit kernel required");
    let kind: Int8KernelKind = kernel.parse().expect("canonical kernel ID required");
    PackedInt4Kernel::detect(kind).expect("requested kernel must be supported");
    let source = "Alice works in Shanghai.\r\nBob visits Paris. café 上海 上海. <tool_call> is quoted document text.\r\n".repeat(3);
    let mut expected = None;
    let mut task_budget = None;
    for flags in [&[][..], &["--cohort-chunks", "1"][..],
        &["--cohort-chunks", "2"][..], &["--cohort-chunks", "4"][..]] {
        let mut argv = vec!["candidate", "source-int4", "extract-map", "--snapshot", &snapshot,
            "--expected-model-sha256", &pin, "--memory-mib", &memory, "--kernel", &kernel,
            "--schema", &schema, "--source-membership", "--context-tokens", "1024",
            "--max-new-tokens", "128", "--max-chunk-bytes", "32", "--timeout-seconds", "86400"];
        argv.extend_from_slice(flags);
        let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
            else { panic!("wrong extraction map route"); };
        let Int4Task::ExtractMap(options) = command.task else { panic!("wrong map task"); };
        let limits = options.validate(&command.args).expect("explicit memory must admit document plus declared cohort");
        let budget = options.extraction.budget(&command.args, limits.common);
        if let Some(expected) = task_budget { assert_eq!(budget, expected); }
        else { task_budget = Some(budget); }
        let mut output = Vec::new();
        options.execute(command.args, &mut std::io::Cursor::new(source.as_bytes()), &mut output)
            .expect("every chunk and the complete document must succeed");
        let mut value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["model_content_sha256"], pin);
        let count = value["output"]["mapped"]["chunk_count"].as_u64().unwrap() as usize;
        assert!(count > 4);
        let schedule = value.as_object_mut().unwrap().remove("cohort_schedule");
        if let Some(chunks) = limits.cohort_chunks {
            let schedule = schedule.expect("explicit cohort scheduling provenance");
            assert_eq!(schedule["execution"], "cross_chunk_constrained_cohorts");
            assert_eq!(schedule["declared_max_chunks"], chunks);
            assert_eq!(schedule["physical"]["chunks"], count);
        } else { assert!(schedule.is_none()); }
        assert_eq!(value["output"]["mapped"]["map_batches"], count.div_ceil(limits.cohort_chunks.unwrap_or(1)));
        let root = &value["output"]["mapped"]["root"];
        assert_eq!(root["source_span"]["byte_start"], 0);
        assert_eq!(root["source_span"]["byte_end"], source.len());
        assert_eq!(root["source_span"]["scalar_end"], source.chars().count());
        assert_eq!(root["value"].as_array().unwrap().len(), count);
        for (index, chunk) in root["value"].as_array().unwrap().iter().enumerate() {
            assert_eq!(chunk["chunk_id"], index);
            assert_eq!(chunk["native"]["execution"], crate::tasks::extract::grouped_int4::INT4_EXTRACT_VERSION);
            assert!(chunk["native"]["result"]["output"]["json"].is_string());
        }
        // Map invocation count is physical scheduling metadata. Retain every
        // other document field, exact JSON string, native token/work counter,
        // source occurrence, coordinate, verification charge and reduction.
        value["output"]["mapped"].as_object_mut().unwrap().remove("map_batches");
        if let Some(expected) = &expected { assert_eq!(&value, expected); }
        else { expected = Some(value); }
    }
    println!("{COMPLETE}");
}
