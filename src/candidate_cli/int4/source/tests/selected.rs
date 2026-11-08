//! One-off capability and pinned source compilation; no native inference.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::PlanContext, source_planning::{SourceTaskPlanner,
        grouped_int4::{PreparedInt4SourceTask, INT4_SOURCE_EXECUTION, INT4_SPARSE_SOURCE_EXECUTION}}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

#[test]
fn all_four_one_off_tasks_accept_bounded_selection_without_changing_other_authority() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (_, dense, args) = arguments(name, &[]);
        let reference = dense.validate(&args).unwrap();
        assert!(dense.selected_rows().unwrap().is_none());
        for cap in [1, 256, NANBEIGE_VOCAB_SIZE] {
            let count = cap.to_string();
            let (_, selected, selected_args) = arguments(name, &["--max-selected-rows", &count]);
            let admitted = selected.validate_with_prefill(&selected_args).unwrap();
            assert_eq!(selected.selected_rows().unwrap().unwrap().max_rows_per_step, cap);
            assert_eq!(admitted.minimum_memory_bytes, reference.minimum_memory_bytes);
            assert_eq!(selected.budget(&selected_args, admitted), dense.budget(&args, reference));
            assert_eq!(selected.mask_visits(), dense.mask_visits());
            assert_eq!(selected.validate(&selected_args).unwrap_err(), CandidateError::Arguments);
        }
        let (_, selected, args) = arguments(name, &["--max-selected-rows", "256", "--prefill-rows", "4"]);
        assert_eq!(selected.validate_with_prefill(&args).unwrap().prefill.unwrap().max_batch_rows, 4);
    }
}

#[test]
fn invalid_caps_and_programmatic_shared_selection_refuse_before_private_io() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
            let count = cap.to_string(); let mut output = Vec::new();
            assert_eq!(command(name, &["--max-selected-rows", &count]).execute(&mut NoRead, &mut output),
                Err(CandidateError::Arguments));
            assert!(output.is_empty());
        }
        let parsed = matches(name, &["--max-selected-rows", "7"]);
        let task = parsed.subcommand_matches("source-int4").unwrap().subcommand_matches(name).unwrap();
        assert!(SourceArgs::from_matches(task).is_err());
        let (_, mut shared, args) = arguments(name, &[]);
        shared.max_selected_rows = Some(NANBEIGE_VOCAB_SIZE);
        assert_eq!(shared.validate(&args).unwrap_err(), CandidateError::Arguments);
        let shared = definition(match name { "ner" => Kind::Ner, "keyphrases" => Kind::Keyphrases,
            "summarize" => Kind::Summarize, _ => Kind::Answer });
        assert!(!shared.get_arguments().any(|arg| arg.get_id().as_str() == "max_selected_rows"));
        assert!(shared.try_get_matches_from([name, "--max-selected-rows", "7"]).is_err());
    }
}

#[test]
fn selected_source_and_snapshot_routes_keep_identical_task_options() {
    let digest = "ab".repeat(32);
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (_, source, args) = arguments(name, &["--max-selected-rows", "7"]);
        let source_limits = source.validate_with_prefill(&args).unwrap();
        let parsed = crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "source-int4", name, "--snapshot", "local.snapshot",
            "--expected-model-sha256", &digest, "--memory-mib", "8192", "--max-selected-rows", "7",
        ]).unwrap();
        let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&parsed).unwrap()
            else { panic!("INT4 required") };
        let Int4Task::Source(_, snapshot) = command.task else { panic!("source task required") };
        let snapshot_limits = snapshot.validate_with_prefill(&command.args).unwrap();
        assert!(snapshot_limits.snapshot_pin.is_some());
        assert_eq!(source, snapshot);
        assert_eq!(source.budget(&args, source_limits), snapshot.budget(&command.args, snapshot_limits));
    }
}

pub(super) fn plan(kind: Kind, options: &SourceArgs, args: &Int4Args) -> PreparedInt4SourceTask {
    struct Continue;
    impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
    let limits = options.validate_with_prefill(args).unwrap(); let budget = options.budget(args, limits);
    let controls = pinned_controls::pinned().unwrap();
    let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
    let planner = SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap()
        .with_document_encoding(args.input_encoding.policy(args.max_input_bytes, limits.max_prompt_tokens).unwrap()).unwrap();
    let width = args.group_width; let digest = Sha256Digest::of_bytes(b"metadata-only-selected-source-cli-fixture");
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: digest,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width,
        )).unwrap()), tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task(kind).spec().identity(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "uncompiled".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    };
    let source = if kind == Kind::Answer {
        r#"{"question":"Where?","passages":[{"id":"one","text":"é上海"},{"id":"two","text":"上海"}]}"#
    } else { "é 上海 上海" };
    let request = shared::request(kind, source.to_owned(), None, budget, args.max_input_bytes).unwrap();
    planner.plan_int4_with_control(&request, &PlanContext::new(&identity, budget).unwrap(),
        options.planning(args), width, &mut Continue).unwrap()
}

#[test]
fn cli_binding_preserves_real_source_plans_and_finalizer_identity_before_admission() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let (kind, dense, args) = arguments(name, &["--max-new-tokens", "64", "--group-width", "127"]);
        let original = plan(kind, &dense, &args); let identity = original.execution_identity().clone();
        let work = original.planned_work(); let default = dense.configure_plan(original).unwrap();
        assert_eq!(default.execution_identity(), &identity);
        assert_eq!(default.execution_version(), INT4_SOURCE_EXECUTION);
        assert_eq!(default.planned_work(), work);
        let (kind, selected, args) = arguments(name, &["--max-new-tokens", "64", "--group-width", "127",
            "--max-selected-rows", "7", "--prefill-rows", "4"]);
        let sparse = selected.configure_plan(plan(kind, &selected, &args)).unwrap();
        assert_eq!(sparse.execution_version(), INT4_SPARSE_SOURCE_EXECUTION);
        assert_eq!(sparse.selected_rows().unwrap().max_rows_per_step, 7);
        assert_eq!(sparse.task_budget(), default.task_budget());
        assert_eq!(sparse.max_result_bytes(), default.max_result_bytes());
        assert_eq!(sparse.max_kv_bytes(), default.max_kv_bytes());
        assert_eq!(sparse.planned_work().projected_logits, 7 * 64);
        assert_eq!(sparse.planned_work().forward_positions, work.forward_positions);
        assert_eq!(sparse.planned_work().attention_pairs, work.attention_pairs);
        assert_eq!(sparse.execution_identity().prompt_digest, identity.prompt_digest);
        assert_eq!(sparse.execution_identity().schema_digest, identity.schema_digest);
        assert_ne!(sparse.execution_identity().decision_policy_digest, identity.decision_policy_digest);
        assert!(sparse.verify_identity(&identity).is_err());
        sparse.verify_identity(sparse.execution_identity()).unwrap();
        assert!(dense.configure_plan(sparse).is_err());
        let mut full = selected; full.max_selected_rows = Some(NANBEIGE_VOCAB_SIZE);
        let full = full.configure_plan(plan(kind, &dense, &args)).unwrap();
        assert_eq!(full.planned_work(), work);
        assert_ne!(full.execution_identity(), &identity);
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn selected_source_requires_the_owned_runtime_before_reading_private_input() {
    for name in ["ner", "keyphrases", "summarize", "answer"] {
        let mut output = Vec::new();
        assert_eq!(command(name, &["--max-selected-rows", "7", "--prefill-rows", "4"])
            .execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
