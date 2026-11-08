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
    for flags in [vec!["--root-array-union"], vec!["--cohort-chunks", "2"],
        vec!["--max-selected-rows", "7"], vec!["--input-encoding", "safe-bpe"],
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
