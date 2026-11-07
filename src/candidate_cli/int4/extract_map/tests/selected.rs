//! Explicit document policy, complete admission and real pinned plan binding.
use super::*;
use crate::{
    batch::extract::grouped_int4::Int4ExtractionBatchPlanner,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{constrained_int4::sparse::Int4JsonSparseLimits,
        generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

#[test]
fn selected_maps_keep_complete_physical_admission_and_every_chunk_task_allowance() {
    let (dense, mut args) = arguments(&["--max-result-bytes", "65536"]);
    args.memory_mib = 262_144;
    for width in [None, Some(1), Some(4)] {
        let mut dense = dense.clone(); dense.cohort_chunks = width;
        let reference = dense.validate(&args).unwrap();
        for cap in [1, 32, NANBEIGE_VOCAB_SIZE] {
            let count = cap.to_string();
            let (mut selected, _) = arguments(&["--max-result-bytes", "65536", "--max-selected-rows", &count]);
            selected.cohort_chunks = width;
            let actual = selected.validate(&args).unwrap();
            assert_eq!(selected.extraction.selected_rows().unwrap(), Some(Int4JsonSparseLimits { max_rows_per_step: cap }));
            assert_eq!(actual.common.minimum_memory_bytes, reference.common.minimum_memory_bytes);
            assert_eq!(actual.common.native, reference.common.native);
            assert_eq!(actual.workspace_bytes, reference.workspace_bytes);
            assert_eq!(actual.cohort, reference.cohort);
            assert_eq!(actual.mapping.mapping.max_model_work, reference.mapping.mapping.max_model_work);
            assert_eq!(actual.mapping.mapping.max_mask_visits, reference.mapping.mapping.max_mask_visits);
            assert_eq!(actual.mapping.verification, reference.mapping.verification);
            assert_eq!(selected.extraction.budget(&args, actual.common), dense.extraction.budget(&args, reference.common));
        }
    }
}

#[test]
fn document_projection_is_explicit_and_shared_by_population_definitions() {
    for definition in [definition(), corpus_definition(), document_job_definition()] {
        assert!(definition.get_arguments().any(|arg| arg.get_long() == Some("max-selected-rows")));
        assert!(definition.try_get_matches_from(["extract-map", "--schema", "schema.json",
            "--source-dir", "source", "--source-manifest", "manifest",
            "--memory-mib", "8192", "--max-selected-rows", "7"]).is_ok());
    }
    for operation in super::super::super::extract_document_job::definition().get_subcommands() {
        assert!(operation.get_arguments().any(|arg| arg.get_long() == Some("max-selected-rows")));
    }
    let (selected, args) = arguments(&["--max-selected-rows", "7"]);
    assert!(selected.validate(&args).is_ok());
    assert!(selected.validate_document_job(&args).is_ok());
    assert!(selected.validate_corpus(&args).is_ok());
}

#[test]
fn invalid_selected_caps_fail_before_schema_model_or_private_input_access() {
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        let count = cap.to_string(); let mut output = Vec::new();
        assert_eq!(command(&["--max-selected-rows", &count]).execute(&mut NoRead, &mut output),
            Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    let matches = definition().try_get_matches_from(["extract-map", "--schema", "schema.json",
        "--source-dir", "source", "--source-manifest", "manifest", "--memory-mib", "8192",
        "--max-selected-rows", "7", "--prefill-rows", "1"]).unwrap();
    assert!(MapArgs::from_matches(&matches).is_err());
}

#[test]
fn selected_source_and_snapshot_maps_share_schema_grounding_and_complete_limits() {
    let (source, source_args) = arguments(&["--max-selected-rows", "32", "--source-membership"]);
    let source_limits = source.validate(&source_args).unwrap();
    let digest = "ab".repeat(32);
    let matches = definition().try_get_matches_from(["extract-map", "--schema", "schema.json",
        "--snapshot", "local.snapshot", "--expected-model-sha256", &digest, "--memory-mib", "8192",
        "--max-selected-rows", "32", "--source-membership"]).unwrap();
    let snapshot = MapArgs::from_matches(&matches).unwrap();
    let snapshot_args = Int4Args::from_arg_matches(&matches).unwrap();
    let snapshot_limits = snapshot.validate(&snapshot_args).unwrap();
    assert!(snapshot_limits.common.snapshot_pin.is_some());
    assert_eq!(source, snapshot);
    assert_eq!(source.extraction.budget(&source_args, source_limits.common),
        snapshot.extraction.budget(&snapshot_args, snapshot_limits.common));
    assert_eq!(source_limits.mapping.mapping.max_model_work, snapshot_limits.mapping.mapping.max_model_work);
    assert_eq!(source_limits.workspace_bytes, snapshot_limits.workspace_bytes);
    let raw = " {\"type\":\"string\",\"const\":\"上海\",\"x-fnlp-source\":\"verbatim\"}\n";
    let request = source.request(raw.to_owned(), source.extraction.budget(&source_args, source_limits.common)).unwrap();
    assert_eq!(request.schema, raw);
    assert_eq!(request.grounding, crate::batch::extract::ExtractionBatchGrounding::SourceMembership);
}

fn planner(map: &MapArgs, args: &Int4Args) -> Int4ExtractionBatchPlanner {
    let limits = map.validate(args).unwrap();
    let budget = map.extraction.budget(args, limits.common);
    let width = args.group_width;
    let digest = Sha256Digest::of_bytes(b"selected-document-cli-synthetic-fixture");
    let packing = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width,
    )).unwrap());
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: crate::artifact::converter::PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
        tokenizer_digest: digest, template_digest: digest, task_spec: "extract-v1".to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "uncompiled".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "uncompiled".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: digest,
        decision_policy_digest: digest, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    };
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    Int4ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, width,
        budget, map.extraction.compiler(), map.extraction.source_limits(), None).unwrap()
}

struct Continue;
impl crate::native_engine::decode::DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<crate::native_engine::decode::DecodeCancellationKind> { None }
}

#[test]
fn selected_cli_planner_is_bound_before_sparse_document_work_and_host_replanning() {
    let source = "éAéA Alice Bob Carol";
    let raw = r#"{"type":"number","const":12345678901234567890123456789012345678}"#;
    let (dense, args) = arguments(&["--max-chunk-bytes", "6", "--max-new-tokens", "64"]);
    let dense_limits = dense.validate(&args).unwrap();
    let budget = dense.extraction.budget(&args, dense_limits.common);
    let request = dense.request(raw.to_owned(), budget).unwrap();
    let dense_planner = dense.extraction.configure_planner(planner(&dense, &args)).unwrap();
    let dense_plan = dense_planner.plan_document_with_control(source, &request, dense_limits.mapping, &mut Continue).unwrap();
    for cap in [7, NANBEIGE_VOCAB_SIZE] {
        let count = cap.to_string();
        let (selected, args) = arguments(&["--max-chunk-bytes", "6", "--max-new-tokens", "64",
            "--max-selected-rows", &count]);
        let limits = selected.validate(&args).unwrap();
        let selected_planner = selected.extraction.configure_planner(planner(&selected, &args)).unwrap();
        let request = selected.request(raw.to_owned(), budget).unwrap();
        let prepared = selected_planner.plan_document_with_control(source, &request, limits.mapping, &mut Continue).unwrap();
        let expected = prepared.preflight_metadata();
        assert!(expected.chunk_count() > 1);
        assert_eq!(expected.selected_rows().unwrap().max_rows_per_step, cap);
        for (selected, dense) in prepared.execution_identities().zip(dense_plan.execution_identities()) {
            assert_eq!(selected.schema_digest, Sha256Digest::of_bytes(raw.as_bytes()));
            assert_eq!(selected.prompt_digest, dense.prompt_digest);
            assert_ne!(selected.decision_policy_digest, dense.decision_policy_digest);
        }
        let mut tight = limits.mapping; tight.mapping.max_model_work = expected.planned_work();
        let replay = selected_planner.plan_document_with_control(source, &request, tight, &mut Continue).unwrap();
        assert_eq!(replay.preflight_metadata(), expected);
        if cap < NANBEIGE_VOCAB_SIZE {
            assert!(dense_planner.plan_document_with_control(source, &request, tight, &mut Continue).is_err());
        }
        assert!(dense.extraction.configure_planner(selected_planner).is_err());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn selected_document_commands_still_require_runtime_without_touching_input() {
    for flags in [&["--max-selected-rows", "7"][..],
        &["--max-selected-rows", "7", "--cohort-chunks", "2"][..]] {
        let mut command = command(flags); command.args.memory_mib = 262_144;
        let mut output = Vec::new();
        assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
