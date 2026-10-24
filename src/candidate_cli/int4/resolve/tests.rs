//! Input, admission and dispatch fixtures; no synthetic native-execution claim.
use super::*;
use serde_json::json;
use crate::{corpus::resolve::{ResolutionPlan, ResolveError},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl}};

fn parsed(flags: &[&str]) -> Result<Int4Command, clap::Error> {
    let mut argv = vec!["candidate", "source-int4", "resolve", "--memory-mib", "16384"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv)?;
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches)?
        else { panic!("wrong INT4 resolution route"); };
    Ok(command)
}
fn command(flags: &[&str]) -> Int4Command {
    let mut all = vec!["--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    all.extend_from_slice(flags); parsed(&all).unwrap()
}
fn options(command: &Int4Command) -> ResolveArgs {
    let Int4Task::Resolve(options) = command.task else { panic!("resolution required"); };
    options
}
fn fixture() -> serde_json::Value {
    json!({"documents":[
        {"id":"a","text":"Élan met Ada.","mentions":[{"entity_type":"person","surface":"Élan",
            "span":{"byte_start":0,"byte_end":5,"scalar_start":0,"scalar_end":4}}]},
        {"id":"b","text":"Élan works here.","mentions":[{"entity_type":"person","surface":"Élan",
            "span":{"byte_start":0,"byte_end":5,"scalar_start":0,"scalar_end":4}}]}
    ],"options":{"blocking":"exact_surface","context_scalars":128,"minimum_margin_milli":1000}})
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused resolution reached private input"); }
}

#[test]
fn resolution_accepts_both_explicit_model_routes_with_fixed_candidate_language() {
    let source = command(&[]); let selected = options(&source);
    let limits = selected.validate(&source.args).unwrap();
    let scoring = selected.scoring_limits(&source.args, limits);
    assert_eq!(scoring.planning.per_head.max_output_tokens, 2);
    assert_eq!(scoring.planning.per_head.max_kv_bytes, limits.native.kv_bytes);
    assert_eq!(scoring.planning.max_pairs, selected.graph.max_pairs);
    assert!(limits.prefill.is_none());
    let pin = "ab".repeat(32);
    let snapshot = parsed(&["--snapshot", "explicit.snapshot", "--expected-model-sha256", &pin]).unwrap();
    let loaded = options(&snapshot).validate(&snapshot.args).unwrap();
    assert_eq!(loaded.snapshot_pin.unwrap().model_content_sha256, [0xab; 32]);
    assert_eq!(loaded.native.kv_bytes, limits.native.kv_bytes);
    for flags in [vec![], vec!["--snapshot", "s"], vec!["--model", "int8.fnlpq"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--source-dir", "d"],
        vec!["--snapshot", "s", "--expected-model-sha256", &pin, "--save-snapshot", "new"]] {
        assert!(parsed(&flags).is_err());
    }
}

#[test]
fn exact_original_utf8_and_explicit_policy_survive_wire_parsing() {
    let command = command(&[]); let options = options(&command);
    let mut value = fixture();
    let exact = "Élan\t <|im_end|> <tool_call>\r\n上海😀";
    value["documents"][0]["text"] = json!(exact);
    let input = options.parse(&value.to_string(), command.args.max_input_bytes).unwrap();
    assert_eq!(input.documents[0].text, exact);
    assert_eq!(input.documents[0].mentions[0].surface, "Élan");
    let plan = ResolutionPlan::prepare(&input.documents, input.options,
        options.graph_limits(&command.args, options.validate(&command.args).unwrap()), &mut Continue).unwrap();
    assert_eq!(plan.mentions().len(), 2);
    assert_eq!(plan.candidate_count(), 1);
    assert_eq!(plan.mentions()[0].span.byte_end, 5);
    assert_eq!(plan.mentions()[0].span.scalar_end, 4);
}

#[test]
fn duplicate_unknown_and_execution_authority_fields_are_rejected() {
    let command = command(&[]); let options = options(&command); let cap = command.args.max_input_bytes;
    for field in ["budget", "model", "work", "execution_identity", "group_width", "numerics_profile",
        "seed", "prompt", "tools", "thinking", "scores", "clusters"] {
        let mut value = fixture(); value[field] = json!({});
        assert!(matches!(options.parse(&value.to_string(), cap), Err(CandidateError::Input)));
    }
    for text in ["[]", "null", "{}", r#"{"documents":[],"documents":[]}"#,
        r#"{"documents":[],"\u0064ocuments":[]}"#] {
        assert!(matches!(options.parse(text, cap), Err(CandidateError::Input)));
    }
    let value = fixture(); let encoded = value.to_string();
    assert!(matches!(options.parse(&encoded, encoded.len() - 1), Err(CandidateError::Input)));
    let mut missing = value.clone(); missing.as_object_mut().unwrap().remove("options");
    assert!(matches!(options.parse(&missing.to_string(), cap), Err(CandidateError::Input)));
    let mut nested = value; nested["documents"][1]["mentions"][0]["score"] = json!(1.0);
    assert!(matches!(options.parse(&nested.to_string(), cap), Err(CandidateError::Input)));
    let duplicate = encoded.replacen("\"context_scalars\":", "\"context_scalars\":0,\"context_scalars\":", 1);
    assert!(matches!(options.parse(&duplicate, cap), Err(CandidateError::Input)));
}

#[test]
fn bad_final_anchor_duplicate_ids_and_incomplete_pair_caps_fail_complete_preflight() {
    let command = command(&[]); let options = options(&command);
    let limits = options.validate(&command.args).unwrap();
    for (field, value) in [("byte_start", 1), ("byte_end", 4), ("scalar_start", 1), ("scalar_end", 5)] {
        let mut wire = fixture(); wire["documents"][1]["mentions"][0]["span"][field] = json!(value);
        let input = options.parse(&wire.to_string(), command.args.max_input_bytes).unwrap();
        assert!(matches!(ResolutionPlan::prepare(&input.documents, input.options,
            options.graph_limits(&command.args, limits), &mut Continue), Err(ResolveError::InvalidAnchor)));
    }
    let mut wire = fixture(); wire["documents"][1]["id"] = json!("a");
    let input = options.parse(&wire.to_string(), command.args.max_input_bytes).unwrap();
    assert!(matches!(ResolutionPlan::prepare(&input.documents, input.options,
        options.graph_limits(&command.args, limits), &mut Continue), Err(ResolveError::InvalidInput)));
    let input = options.parse(&fixture().to_string(), command.args.max_input_bytes).unwrap();
    let mut graph = options.graph_limits(&command.args, limits); graph.max_candidate_pairs = 0;
    assert!(matches!(ResolutionPlan::prepare(&input.documents, input.options, graph, &mut Continue),
        Err(ResolveError::PairBudget)));
}

#[test]
fn empty_candidate_graph_is_source_validated_and_keeps_every_singleton() {
    let command = command(&["--max-pairs", "0"]); let options = options(&command);
    let mut wire = fixture(); wire["documents"][1]["mentions"][0]["entity_type"] = json!("organization");
    let input = options.parse(&wire.to_string(), command.args.max_input_bytes).unwrap();
    let graph = ResolutionPlan::prepare(&input.documents, input.options,
        options.graph_limits(&command.args, options.validate(&command.args).unwrap()), &mut Continue).unwrap();
    assert_eq!(graph.candidate_count(), 0);
    let result = graph.finalize(Vec::new(), &mut Continue).unwrap();
    assert_eq!(result.document_count, 2);
    assert_eq!(result.mentions.len(), 2);
    assert_eq!(result.clusters.len(), 2);
    assert!(result.judgments.is_empty());
}

#[test]
fn free_text_candidate_language_and_invalid_resources_refuse_before_private_io() {
    for flags in [vec!["--max-candidate-tokens", "2"], vec!["--max-new-tokens", "128"],
        vec!["--max-output-bytes", "65536"], vec!["--logprobs"], vec!["--prefill-rows", "1"],
        vec!["--seed", "ab"], vec!["--min-new-tokens", "0"], vec!["--stop", "END"],
        vec!["--ban-token", "7"], vec!["--logit-bias", "7=0"], vec!["--discover-entities"]] {
        let mut all = vec!["--source-dir", "d", "--source-manifest", "m"]; all.extend(flags);
        assert!(parsed(&all).is_err());
    }
    for flags in [vec!["--max-documents", "0"], vec!["--max-mentions", "0"], vec!["--max-pairs", "65537"],
        vec!["--max-pair-visits", "0"], vec!["--max-cluster-checks", "0"], vec!["--max-scan-steps", "0"],
        vec!["--graph-reserve-mib", "1"], vec!["--max-result-bytes", "0"], vec!["--max-forward-positions", "0"],
        vec!["--max-projected-logits", "0"], vec!["--max-attention-pairs", "0"], vec!["--max-dot-products", "0"],
        vec!["--max-multiply-accumulates", "0"], vec!["--context-tokens", "2"],
        vec!["--preparation-mib", "1"], vec!["--group-width", "0"]] {
        let mut output = Vec::new();
        assert_eq!(command(&flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    let mut programmatic = command(&[]); programmatic.args.logprobs = true;
    assert_eq!(programmatic.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
}

#[test]
fn both_preparation_owners_graph_and_hosted_input_are_in_the_memory_quote() {
    for width in [3, 16, 128, 10752] {
        let mut command = command(&[]); let options = options(&command);
        command.args.group_width = width; command.args.max_load_mib = 65536; command.args.memory_mib = 65536;
        let base = options.scoring.validate(&command.args).unwrap();
        let actual = options.validate(&command.args).unwrap();
        assert_eq!(actual.minimum_memory_bytes, base.minimum_memory_bytes + base.preparation_bytes
            + options.graph_bytes().unwrap() + options.host_input_bytes(&command.args).unwrap());
        assert_eq!(actual.native.kv_bytes,
            Int4MemoryRequirement::for_context(command.args.context_tokens, width).unwrap().kv_bytes);
        command.args.memory_mib = (actual.minimum_memory_bytes - 1) / MIB;
        assert!(options.validate(&command.args).is_err());
        command.args.memory_mib = actual.minimum_memory_bytes.div_ceil(MIB);
        assert!(options.validate(&command.args).is_ok());
    }
}

#[test]
fn whole_snapshot_ceiling_is_never_multiplied_by_pairs_or_orders() {
    let command = command(&[]); let options = options(&command);
    let limits = options.validate(&command.args).unwrap();
    let scoring = options.scoring_limits(&command.args, limits); let cap = options.scoring.work_ceiling();
    assert_eq!(scoring.max_model_work, cap);
    assert_eq!(scoring.planning.max_work.forward_positions, cap.forward_positions);
    assert_eq!(scoring.planning.max_work.projected_logits, cap.projected_logits);
    assert_eq!(scoring.planning.max_total_prompt_tokens as u64, cap.forward_positions);
    options.scoring.admit_plan(8, 8, cap).unwrap();
    for axis in 0..5 {
        let mut excess = cap;
        match axis { 0 => excess.forward_positions += 1, 1 => excess.projected_logits += 1,
            2 => excess.attention_pairs += 1, 3 => excess.projections.dot_products += 1,
            _ => excess.projections.multiply_accumulates += 1 }
        assert_eq!(options.scoring.admit_plan(8, 8, excess), Err(CandidateError::Planning));
    }
    assert_eq!(options.scoring.admit_plan(9, 8, cap), Err(CandidateError::Planning));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn absent_runtime_fails_before_input_model_io_or_publication() {
    let mut output = Vec::new();
    assert_eq!(command(&[]).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}

#[cfg(feature = "asupersync-runtime")]
#[test]
#[ignore = "requires controller-admitted pinned sources, new snapshot and available explicit SIMD kernel"]
fn source_scalar_and_independently_pinned_snapshot_simd_resolve_the_same_snapshot() {
    let source = std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source directory required");
    let manifest = std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest required");
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_MIB").expect("explicit memory ceiling required")
        .parse().expect("memory ceiling must be MiB");
    let snapshot = std::env::var("FNLP_INT4_TEST_RESOLVE_SNAPSHOT").expect("explicit new snapshot required");
    assert!(!std::path::Path::new(&snapshot).exists(), "fixture requires a fresh path and deletes nothing");
    let kernel: Int8KernelKind = std::env::var("FNLP_INT4_TEST_KERNEL").expect("explicit SIMD kernel required")
        .parse().expect("canonical kernel ID required");
    assert_ne!(kernel, Int8KernelKind::Scalar);
    PackedInt4Kernel::detect(kernel).expect("explicit kernel must exist on the actual host");
    let wire = fixture().to_string();
    let mut convert = command(&[]); convert.args.source_dir = Some(PathBuf::from(source));
    convert.args.source_manifest = Some(PathBuf::from(manifest)); convert.args.memory_mib = memory;
    convert.args.save_snapshot = Some(PathBuf::from(&snapshot));
    let mut bytes = Vec::new(); convert.execute(&mut wire.as_bytes(), &mut bytes).unwrap();
    let scalar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(scalar["output"]["head_count"], 2);
    assert_eq!(scalar["output"]["model_evaluated"], true);
    assert_eq!(scalar["snapshot"]["operation"], "saved");
    let pin = scalar["model_content_sha256"].as_str().unwrap();
    let mut reload = parsed(&["--snapshot", &snapshot, "--expected-model-sha256", pin, "--kernel", kernel.id()]).unwrap();
    reload.args.memory_mib = memory;
    let mut bytes = Vec::new(); reload.execute(&mut wire.as_bytes(), &mut bytes).unwrap();
    let selected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(selected["snapshot"]["operation"], "loaded");
    assert_eq!(selected["output"], scalar["output"]);
    // The caller-owned snapshot is deliberately preserved, including on failure.
}
