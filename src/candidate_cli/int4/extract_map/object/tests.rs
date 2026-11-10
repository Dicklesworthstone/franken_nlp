use super::*;
use crate::native_engine::decode::DecodeCancellationKind;

pub(super) fn fixture(extra: &[&str]) -> (ObjectMapArgs, Int4Args) {
    let mut argv = vec!["candidate", "source-int4", "extract-object-map", "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "262144"];
    if !extra.contains(&"--schema") { argv.extend_from_slice(&["--schema", "object-schema.json"]); }
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong source dispatcher"); };
    let Int4Task::ObjectMap(options) = command.task else { panic!("wrong explicit object-map route"); };
    (options, command.args)
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused object command read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused object command wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused object command flushed output"); }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

#[test]
fn actual_dispatcher_exposes_separate_object_contract_and_rejects_unbound_modes() {
    let (options, args) = fixture(&[]);
    let limits = options.validate(&args).unwrap();
    assert!(limits.map.mapping.root_array_union.is_none());
    assert!(limits.map.cohort.is_none());
    assert_eq!(limits.map.mapping.mapping.reduction.map_batch_chunks, 1);
    for flags in [vec!["--root-array-union"], vec!["--input-encoding", "safe-bpe-v1"],
        vec!["--compute-threads", "2"], vec!["--prefill-rows", "2"], vec!["--seed", "3"],
        vec!["--temperature-milli", "900"]] {
        let mut argv = vec!["candidate", "source-int4", "extract-object-map", "--schema", "schema.json",
            "--source-dir", "explicit-source", "--source-manifest", "manifest.json", "--memory-mib", "262144"];
        argv.extend(flags);
        if let Ok(matches) = crate::candidate_cli::definition().try_get_matches_from(argv) {
            assert!(CandidateCommand::from_matches(&matches).is_err(), "unsupported mode parsed into an executable command");
        }
    }
}

#[test]
fn object_cohorts_price_all_simultaneous_chunks_and_keep_original_task_and_reducer_bounds() {
    let (serial, args) = fixture(&[]);
    let baseline = serial.validate(&args).unwrap();
    for width in [1, 2, 64] {
        let value = width.to_string();
        let (grouped, args) = fixture(&["--cohort-chunks", &value]);
        let limits = grouped.validate(&args).unwrap();
        let quote = limits.map.cohort.unwrap();
        assert_eq!(limits.map.cohort_chunks, Some(width));
        assert_eq!(limits.map.mapping.mapping.reduction.map_batch_chunks, width);
        assert_eq!(quote.native.kv_bytes, baseline.map.common.native.kv_bytes * width as u64);
        assert!(quote.workspace_bytes > 0);
        assert_eq!(limits.object, baseline.object);
        assert_eq!(limits.object_workspace_bytes, baseline.object_workspace_bytes);
        assert_eq!(limits.map.workspace_bytes, baseline.map.workspace_bytes);
        assert_eq!(limits.map.mapping.verification, baseline.map.mapping.verification);
        assert_eq!(limits.map.mapping.mapping.max_model_work, baseline.map.mapping.mapping.max_model_work);
        assert_eq!(grouped.map.extraction.budget(&args, limits.map.common),
            serial.map.extraction.budget(&args, baseline.map.common));
        let ordinary = grouped.map.validate(&args).unwrap();
        assert_eq!(limits.map.common.minimum_memory_bytes,
            ordinary.common.minimum_memory_bytes + limits.object_workspace_bytes);
        let mut short = args;
        short.memory_mib = limits.map.common.minimum_memory_bytes.div_ceil(MIB) - 1;
        assert_eq!(grouped.execute(short, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn exact_selected_rows_compose_with_object_cohorts_without_widening_any_other_allowance() {
    for width in [None, Some(1), Some(4)] {
        let (mut full, args) = fixture(&[]);
        full.map.cohort_chunks = width;
        let reference = full.validate(&args).unwrap();
        for cap in [1, 32, crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE] {
            let value = cap.to_string();
            let (mut selected, args) = fixture(&["--max-selected-rows", &value]);
            selected.map.cohort_chunks = width;
            let limits = selected.validate(&args).unwrap();
            assert_eq!(selected.map.extraction.selected_rows().unwrap().unwrap().max_rows_per_step, cap);
            assert_eq!(limits.map.common.minimum_memory_bytes, reference.map.common.minimum_memory_bytes);
            assert_eq!(limits.map.cohort, reference.map.cohort);
            assert_eq!(limits.map.mapping.mapping.max_model_work, reference.map.mapping.mapping.max_model_work);
            assert_eq!(limits.map.mapping.mapping.max_mask_visits, reference.map.mapping.mapping.max_mask_visits);
            assert_eq!(limits.map.mapping.verification, reference.map.mapping.verification);
            assert_eq!(limits.object, reference.object);
            let raw = r#"{"type":"object","properties":{"amount":{"type":"number","const":9007199254740993}},"required":["amount"],"additionalProperties":false}"#;
            let budget = selected.map.extraction.budget(&args, limits.map.common);
            let request = selected.request(raw.to_owned(), budget, limits, &mut Continue).unwrap();
            assert_eq!(request.schema, raw);
            assert_eq!(request.budget, full.map.extraction.budget(&args, reference.map.common));
        }
    }
}

#[test]
fn invalid_cohort_and_selected_bounds_refuse_before_private_input_or_model_io() {
    for flags in [vec!["--cohort-chunks", "0"], vec!["--cohort-chunks", "65"],
        vec!["--cohort-chunks", "3", "--max-chunks", "2"],
        vec!["--max-selected-rows", "0"], vec!["--max-selected-rows", "166145"]] {
        let (options, args) = fixture(&flags);
        assert_eq!(options.execute(args, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    let (mut options, args) = fixture(&[]);
    options.map.cohort_chunks = Some(usize::MAX);
    assert_eq!(options.execute(args, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn configured_object_execution_still_requires_runtime_before_input() {
    for flags in [vec!["--cohort-chunks", "2"], vec!["--max-selected-rows", "32"],
        vec!["--cohort-chunks", "2", "--max-selected-rows", "32"]] {
        let (options, args) = fixture(&flags);
        assert_eq!(options.execute(args, &mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    }
}

#[test]
fn complete_object_and_schema_workspace_are_admitted_before_any_io_without_changing_native_budget() {
    let (options, args) = fixture(&[]);
    let ordinary = options.map.validate(&args).unwrap();
    let composed = options.validate(&args).unwrap();
    assert!(composed.object_workspace_bytes > composed.object.workspace_bytes().unwrap());
    assert_eq!(composed.map.common.minimum_memory_bytes,
        ordinary.common.minimum_memory_bytes + composed.object_workspace_bytes);
    assert_eq!(composed.map.workspace_bytes, ordinary.workspace_bytes + composed.object_workspace_bytes);
    assert_eq!(options.map.extraction.budget(&args, composed.map.common), options.map.extraction.budget(&args, ordinary.common));
    assert_eq!(composed.map.mapping.mapping.max_model_work, ordinary.mapping.mapping.max_model_work);
    assert_eq!(composed.map.mapping.verification, ordinary.mapping.verification);
    let mut short = args;
    short.memory_mib = composed.map.common.minimum_memory_bytes.div_ceil(MIB) - 1;
    assert_eq!(options.execute(short, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[test]
fn all_object_bounds_and_remote_schema_paths_refuse_before_input_or_output() {
    for flags in [vec!["--max-object-properties", "0"], vec!["--max-object-origins", "1"],
        vec!["--max-object-nodes", "0"], vec!["--max-object-input-bytes", "1"],
        vec!["--max-object-json-bytes", "67108865"], vec!["--max-object-result-bytes", "67108865"],
        vec!["--max-object-json-bytes", "33554432"], vec!["--schema", "https://example.invalid/private.json"]] {
        let (options, args) = fixture(&flags);
        assert_eq!(options.execute(args, &mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn original_schema_bytes_survive_request_admission_and_grounding_modes_cannot_be_substituted() {
    let (options, args) = fixture(&[]);
    let limits = options.validate(&args).unwrap();
    let budget = options.map.extraction.budget(&args, limits.map.common);
    let schema = "{ \"type\":\"object\",\"additionalProperties\":false,\"properties\":{\"amount\":{\"type\":\"number\",\"const\":9007199254740993}},\"required\":[\"amount\"] }";
    let request = options.request(schema.to_owned(), budget, limits, &mut Continue).unwrap();
    assert_eq!(request.schema, schema);
    assert_eq!(request.budget, budget);
    assert_eq!(request.grounding, ExtractionBatchGrounding::Structural);
    let annotated = r#"{"type":"object","additionalProperties":false,"properties":{"name":{"type":"string","x-fnlp-source":"verbatim"}},"required":["name"]}"#;
    assert!(options.request(annotated.to_owned(), budget, limits, &mut Continue).is_err());
    let (grounded, grounded_args) = fixture(&["--source-membership"]);
    let gl = grounded.validate(&grounded_args).unwrap();
    assert!(grounded.request(schema.to_owned(), budget, gl, &mut Continue).is_err());
    let admitted = grounded.request(annotated.to_owned(), budget, gl, &mut Continue).unwrap();
    assert_eq!(admitted.schema, annotated);
    assert_eq!(admitted.grounding, ExtractionBatchGrounding::SourceMembership);
}

#[test]
fn one_chunk_output_cap_does_not_limit_the_original_full_required_object() {
    // The partial schema permits one 896-byte atomic string, while the full
    // required object is larger than the original 2048-byte per-chunk cap.
    let (options, args) = fixture(&["--max-result-bytes", "2048"]);
    let limits = options.validate(&args).unwrap();
    let value = "x".repeat(896);
    let schema = format!(r#"{{"type":"object","additionalProperties":false,"properties":{{"a":{{"type":"string","const":"{value}"}},"b":{{"type":"string","const":"{value}"}},"c":{{"type":"string","const":"{value}"}}}},"required":["a","b","c"]}}"#);
    let budget = options.map.extraction.budget(&args, limits.map.common);
    assert!(crate::grammar::runtime::JsonValidationSchema::compile(&schema, options.map.extraction.compiler()).is_err());
    assert!(options.request(schema.clone(), budget, limits, &mut Continue).is_ok());
    let request = options.request(schema, budget, limits, &mut Continue).unwrap();
    assert_eq!(request.budget.max_output_bytes, 2048);
}
