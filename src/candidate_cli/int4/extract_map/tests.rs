use super::*;

pub(super) fn command(extra: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", "extract-map", "--source-dir", "explicit-source",
        "--source-manifest", "explicit-manifest.json", "--memory-mib", "8192"];
    if !extra.contains(&"--schema") { argv.extend_from_slice(&["--schema", "schema.json"]); }
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong extraction map dispatcher"); };
    command
}
fn arguments(extra: &[&str]) -> (MapArgs, Int4Args) {
    let command = command(extra);
    let Int4Task::ExtractMap(map) = command.task else { panic!("wrong extraction map task"); };
    (map, command.args)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused extraction map reached input"); }
}

#[test]
fn source_and_pinned_snapshot_routes_keep_complete_document_limits() {
    let (map, args) = arguments(&[]);
    let source = map.validate(&args).unwrap();
    assert_eq!(args.max_input_bytes, 1_048_576);
    assert_eq!(source.mapping.mapping.chunks.max_input_bytes, args.max_input_bytes);
    assert_eq!(source.mapping.mapping.chunks.context_tokens, args.context_tokens);
    assert_eq!(source.max_result_bytes, 16_777_216);
    assert_eq!(source.common.result_bytes, 1_048_576);
    assert_eq!(source.common.preparation_bytes, 1024 * MIB);
    assert_eq!(source.mapping.mapping.mask_visits_per_chunk, map.extraction.mask_visits());
    assert_eq!(source.mapping.mapping.max_mask_visits, 128_000_000_000);
    assert!(source.common.prefill.is_none());
    let digest = "ab".repeat(32);
    let matches = definition().try_get_matches_from(["extract-map", "--schema", "schema.json",
        "--snapshot", "local.snapshot", "--expected-model-sha256", &digest, "--memory-mib", "8192"]).unwrap();
    let map = MapArgs::from_matches(&matches).unwrap();
    let args = Int4Args::from_arg_matches(&matches).unwrap();
    let snapshot = map.validate(&args).unwrap();
    assert!(snapshot.common.snapshot_pin.is_some());
    assert_eq!(source.mapping.mapping.max_model_work, snapshot.mapping.mapping.max_model_work);
    assert_eq!(source.workspace_bytes, snapshot.workspace_bytes);
}

#[test]
fn bad_chunk_reduction_work_and_verification_limits_refuse_before_io() {
    for flags in [vec!["--max-chunks", "0"], vec!["--max-chunks", "257"],
        vec!["--max-chunk-bytes", "3"], vec!["--max-tokenizer-calls", "0"],
        vec!["--max-tokenizer-calls", "1000001"], vec!["--max-map-result-bytes", "1048575"],
        vec!["--max-map-result-bytes", "67108865"], vec!["--max-live-value-bytes", "1"],
        vec!["--max-total-value-bytes", "1"], vec!["--reduction-reserve-mib", "0"],
        vec!["--max-total-mask-node-visits", "1"], vec!["--max-forward-positions", "0"],
        vec!["--max-forward-positions", "16777217"], vec!["--max-projected-logits", "0"],
        vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--max-extraction-fields", "0"],
        vec!["--max-extraction-evidence-spans", "1000001"], vec!["--max-extraction-scan-steps", "0"],
        vec!["--schema", "-"], vec!["--preparation-mib", "512"]] {
        let mut output = Vec::new();
        assert_eq!(command(&flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
}

#[test]
fn complete_schema_bytes_survive_and_source_language_waits_for_actual_chunks() {
    let (map, args) = arguments(&[]);
    let budget = map.extraction.budget(&args, map.validate(&args).unwrap().common);
    let raw = " {\"type\":\"number\",\"const\":12345678901234567890123456789012345678}\n";
    assert_eq!(map.request(raw.to_owned(), budget).unwrap().schema, raw);
    for raw in ["{", r#"{"type":"boolean","type":"string"}"#, r#"{"$ref":"remote"}"#] {
        assert!(map.request(raw.to_owned(), budget).is_err());
    }
    let (map, _) = arguments(&["--source-membership"]);
    let raw = r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#;
    let request = map.request(raw.to_owned(), budget).unwrap();
    assert_eq!(request.schema, raw);
    assert_eq!(request.grounding, crate::batch::extract::ExtractionBatchGrounding::SourceMembership);
}

#[test]
fn whole_document_memory_prices_all_prepared_chunks_and_full_evidence_workspace() {
    let (map, mut args) = arguments(&[]);
    let single = map.extraction.validate(&args).unwrap();
    let limits = map.validate(&args).unwrap();
    let budget = map.extraction.budget(&args, limits.common);
    let chunks = map.mapping.max_chunks as u64;
    let expected = (budget.max_output_bytes * chunks + map.mapping.max_live_value_bytes as u64) * 4
        + u64::from(budget.max_output_tokens) * chunks * 8
        + map.mapping.max_extraction_fields as u64 * std::mem::size_of::<SourceFieldEvidence>() as u64
        + map.mapping.max_extraction_evidence_spans as u64 * std::mem::size_of::<VerifiedSourceSpan>() as u64 * 2
        + limits.reduction_reserve_bytes;
    assert_eq!(limits.workspace_bytes, expected);
    assert!(limits.common.minimum_memory_bytes > single.minimum_memory_bytes + single.preparation_bytes + expected);
    args.memory_mib = limits.common.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert!(map.validate(&args).is_err());
    args.memory_mib = limits.common.minimum_memory_bytes.div_ceil(MIB);
    assert!(map.validate(&args).is_ok());
    assert!(workspace_bytes(budget, limits.mapping, 0, limits.reduction_reserve_bytes).is_err());
    assert!(workspace_bytes(budget, limits.mapping, map.mapping.max_chunks + 1, limits.reduction_reserve_bytes).is_err());
    let mut overflow = budget; overflow.max_output_bytes = u64::MAX;
    assert!(workspace_bytes(overflow, limits.mapping, 2, limits.reduction_reserve_bytes).is_err());
}

#[test]
fn native_group_partition_is_explicit_and_free_text_controls_cannot_leak() {
    let (map, mut args) = arguments(&[]);
    args.group_width = 16;
    let sixteen = map.validate(&args).unwrap();
    args.group_width = 128;
    let larger = map.validate(&args).unwrap();
    assert_eq!(sixteen.mapping.mapping.max_model_work, larger.mapping.mapping.max_model_work);
    assert!(sixteen.common.load.groups > larger.common.load.groups);
    for flags in [vec!["--max-output-bytes", "65536"], vec!["--prefill-rows", "1"],
        vec!["--min-new-tokens", "0"], vec!["--logit-bias", "42=0"]] {
        let mut argv = vec!["extract-map", "--schema", "schema.json", "--source-dir", "source",
            "--source-manifest", "manifest", "--memory-mib", "8192"];
        argv.extend(flags);
        let matches = definition().try_get_matches_from(argv).unwrap();
        assert!(MapArgs::from_matches(&matches).is_err());
    }
    let mut command = command(&[]); command.args.seed = Some("ab".repeat(32));
    assert_eq!(command.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_never_opens_source_schema_or_model() {
    let mut output = Vec::new();
    assert_eq!(command(&[]).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
