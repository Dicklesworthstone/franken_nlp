//! One-off head capability and pinned planning; no native inference is claimed.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchDocument, extract::grouped_int4::Int4ExtractionBatchPlanner},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::extract::grouped_int4::{Int4ExtractPlan, INT4_EXTRACT_VERSION, INT4_SPARSE_EXTRACT_VERSION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

#[test]
fn only_supported_one_off_definitions_expose_selected_rows() {
    fn collect(command: &clap::Command, prefix: &str, found: &mut Vec<String>) {
        let path = format!("{prefix}/{}", command.get_name());
        if command.get_arguments().any(|arg| arg.get_id().as_str() == "max_selected_rows") {
            found.push(path.clone());
        }
        for child in command.get_subcommands() { collect(child, &path, found); }
    }
    let mut found = Vec::new();
    collect(&crate::candidate_cli::definition(), "", &mut found);
    assert_eq!(found, ["/candidate/source-int4/extract", "/candidate/source-int4/ner",
        "/candidate/source-int4/keyphrases", "/candidate/source-int4/summarize", "/candidate/source-int4/answer"]);
    for shared in [definition(), super::super::super::extract_batch::definition(),
        super::super::super::extract_map::definition(), super::super::super::extract_job::definition(),
        super::super::super::extract_document_job::definition(), super::super::super::semantic::definition()] {
        let mut args = vec![shared.get_name().to_owned()];
        if shared.get_subcommands().next().is_some() { args.push("start".to_owned()); }
        args.extend(["--max-selected-rows".to_owned(), "7".to_owned()]);
        assert!(shared.try_get_matches_from(args).is_err());
    }
}

#[test]
fn explicit_rows_and_prefill_preserve_task_and_mask_authority() {
    let (dense, dense_args) = arguments(&[]);
    let reference = dense.validate(&dense_args).unwrap();
    assert!(dense.selected_rows().unwrap().is_none());
    for rows in [1, 256, NANBEIGE_VOCAB_SIZE] {
        let count = rows.to_string();
        let (selected, args) = arguments(&["--max-selected-rows", &count]);
        let admitted = selected.validate_with_prefill(&args).unwrap();
        assert_eq!(selected.selected_rows().unwrap().unwrap().max_rows_per_step, rows);
        assert_eq!(admitted.minimum_memory_bytes, reference.minimum_memory_bytes);
        assert_eq!(selected.budget(&args, admitted), dense.budget(&dense_args, reference));
        assert_eq!(selected.mask_visits(), dense.mask_visits());
        assert_eq!(selected.validate(&args).unwrap_err(), CandidateError::Arguments);
    }
    let (selected, args) = arguments(&["--max-selected-rows", "256", "--prefill-rows", "4"]);
    let admitted = selected.validate_with_prefill(&args).unwrap();
    assert_eq!(admitted.prefill.unwrap().max_batch_rows, 4);
    assert_eq!(selected.budget(&args, admitted), dense.budget(&dense_args, reference));
    assert_eq!(selected.mask_visits(), dense.mask_visits());
}

#[test]
fn selected_rows_are_identical_on_source_and_pinned_snapshot_routes() {
    let (source, source_args) = arguments(&["--max-selected-rows", "7"]);
    let source_limits = source.validate_with_prefill(&source_args).unwrap();
    let digest = "ab".repeat(32);
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "source-int4", "extract", "--snapshot", "local.snapshot",
        "--expected-model-sha256", &digest, "--memory-mib", "8192", "--schema", "schema.json",
        "--max-selected-rows", "7",
    ]).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("INT4 snapshot required"); };
    let Int4Task::Extract(snapshot) = command.task else { panic!("extraction required"); };
    let snapshot_limits = snapshot.validate_with_prefill(&command.args).unwrap();
    assert!(snapshot_limits.snapshot_pin.is_some());
    assert_eq!(source, snapshot);
    assert_eq!(source.budget(&source_args, source_limits), snapshot.budget(&command.args, snapshot_limits));
}

#[test]
fn invalid_or_shared_programmatic_options_fail_before_private_io() {
    for rows in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        let count = rows.to_string();
        let mut output = Vec::new();
        assert_eq!(command(&["--max-selected-rows", &count]).execute(&mut NoRead, &mut output),
            Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    let matches = one_off_definition().try_get_matches_from(["extract", "--source-dir", "source",
        "--source-manifest", "manifest", "--memory-mib", "8192", "--schema", "schema.json",
        "--max-selected-rows", "7"]).unwrap();
    assert!(ExtractionArgs::from_matches(&matches).is_err());
    let (mut shared, args) = arguments(&[]);
    let limits = shared.validate(&args).unwrap();
    let budget = shared.budget(&args, limits);
    shared.max_selected_rows = Some(7);
    assert_eq!(shared.validate(&args).unwrap_err(), CandidateError::Arguments);
    assert_eq!(shared.shared_defaults(r#"{"type":"boolean"}"#.to_owned(), budget).err(),
        Some(CandidateError::Arguments));
}

fn plan(options: &ExtractionArgs, args: &Int4Args) -> Int4ExtractPlan {
    let limits = options.validate_with_prefill(args).unwrap();
    let budget = options.budget(args, limits);
    let width = args.group_width;
    let digest = Sha256Digest::of_bytes(b"metadata-only-selected-row-cli-fixture");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width,
    )).unwrap());
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: digest,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: packing, tokenizer_digest: digest, template_digest: digest,
        task_spec: "extract-v1".to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, width,
        budget, options.compiler(), options.source_limits(), None).unwrap();
    let source = "é 上海 上海";
    let schema = if options.source_membership {
        r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#
    } else { r#"{"type":"number","const":0.12345678901234567890123456789012345678}"# };
    let request = options.check_schema(schema.to_owned(), source, budget).unwrap();
    planner.prepare(BatchDocument { id: "cli".to_owned(), text: source.to_owned(), task_args: Some(request) })
        .unwrap().into_extraction_plan()
}

#[test]
fn cli_selection_binds_the_prepared_plan_before_host_admission() {
    for source_membership in [false, true] {
        let mut flags = vec!["--max-new-tokens", "64", "--group-width", "127"];
        if source_membership { flags.push("--source-membership"); }
        let (dense, args) = arguments(&flags);
        let original = plan(&dense, &args);
        let identity = original.execution_identity().clone();
        let work = original.planned_work();
        let default = dense.configure_plan(original).unwrap();
        assert_eq!(default.execution_identity(), &identity);
        assert_eq!(default.planned_work(), work);
        assert_eq!(default.execution_version(), INT4_EXTRACT_VERSION);
        flags.extend(["--max-selected-rows", "7", "--prefill-rows", "4"]);
        let (selected, args) = arguments(&flags);
        let sparse = selected.configure_plan(plan(&selected, &args)).unwrap();
        assert_eq!(sparse.selected_rows().unwrap().max_rows_per_step, 7);
        assert_eq!(sparse.execution_version(), INT4_SPARSE_EXTRACT_VERSION);
        assert_eq!(sparse.options().max_new_tokens, default.options().max_new_tokens);
        assert_eq!(sparse.max_kv_bytes(), default.max_kv_bytes());
        assert_eq!(sparse.max_result_bytes(), default.max_result_bytes());
        assert_eq!(sparse.planned_work().projected_logits, 7 * 64);
        assert_eq!(sparse.planned_work().forward_positions, work.forward_positions);
        assert_eq!(sparse.planned_work().attention_pairs, work.attention_pairs);
        assert!(sparse.planned_work().projections.dot_products < work.projections.dot_products);
        assert_eq!(sparse.execution_identity().prompt_digest, identity.prompt_digest);
        assert_eq!(sparse.execution_identity().schema_digest, identity.schema_digest);
        assert_ne!(sparse.execution_identity().decision_policy_digest, identity.decision_policy_digest);
        assert!(sparse.verify_identity(&identity).is_err());
        sparse.verify_identity(sparse.execution_identity()).unwrap();
        assert!(dense.configure_plan(sparse).is_err(), "a prior selection cannot escape the explicit CLI option");
    }
}

#[test]
fn full_vocabulary_row_cap_still_has_a_distinct_explicit_strategy_identity() {
    let (dense, args) = arguments(&["--max-new-tokens", "64"]);
    let original = plan(&dense, &args);
    let identity = original.execution_identity().clone(); let work = original.planned_work();
    let cap = NANBEIGE_VOCAB_SIZE.to_string();
    let (selected, _) = arguments(&["--max-new-tokens", "64", "--max-selected-rows", &cap]);
    let sparse = selected.configure_plan(original).unwrap();
    assert_eq!(sparse.planned_work(), work);
    assert_ne!(sparse.execution_identity(), &identity);
    assert_eq!(sparse.execution_version(), INT4_SPARSE_EXTRACT_VERSION);
    assert!(selected.configure_plan(sparse).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn selected_rows_still_require_the_owned_runtime_before_reading_private_input() {
    let mut output = Vec::new();
    assert_eq!(command(&["--max-selected-rows", "7", "--prefill-rows", "4"])
        .execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
