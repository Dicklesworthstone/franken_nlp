//! Shared-stream opt-in, pure admission and actual pinned task compilation.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchDocument, extract::grouped_int4::Int4ExtractionBatchPlanner},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::extract::grouped_int4::{INT4_EXTRACT_VERSION, INT4_SPARSE_EXTRACT_VERSION},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

#[test]
fn explicit_stream_rows_price_all_outputs_without_weakening_task_memory_or_masks() {
    let (dense, args) = arguments(&["--max-requests", "8"]); let reference = dense.validate(&args).unwrap();
    assert!(reference.selected_rows.is_none());
    for cap in [1, 7, NANBEIGE_VOCAB_SIZE] {
        let cap_string = cap.to_string();
        let (selected, args) = arguments(&["--max-requests", "8", "--max-selected-rows", &cap_string]);
        let admitted = selected.validate(&args).unwrap();
        assert_eq!(admitted.selected_rows.unwrap().max_rows_per_step, cap);
        assert_eq!(selected.extraction, dense.extraction);
        assert_eq!(admitted.common.minimum_memory_bytes, reference.common.minimum_memory_bytes);
        assert_eq!(admitted.common.native, reference.common.native);
        assert_eq!(selected.extraction.budget(&args, admitted.common), dense.extraction.budget(&args, reference.common));
        assert_eq!(admitted.native.masks.max_visits_per_item, reference.native.masks.max_visits_per_item);
        assert_eq!(admitted.native.masks.max_visits_per_run, reference.native.masks.max_visits_per_run);
        assert_eq!(admitted.native.masks.per_mask, reference.native.masks.per_mask);
        assert_eq!(admitted.native.max_model_work.forward_positions, reference.native.max_model_work.forward_positions);
        assert_eq!(admitted.native.max_model_work.attention_pairs, reference.native.max_model_work.attention_pairs);
        assert_eq!(admitted.native.max_model_work.projected_logits, 8 * args.max_new_tokens as u64 * cap as u64);
        assert_eq!(admitted.transport.max_work.projected_logits, admitted.native.max_model_work.projected_logits);
        assert_eq!(admitted.transport.max_output_line_bytes, reference.transport.max_output_line_bytes);
        assert_eq!(admitted.output_bytes, reference.output_bytes);
        if cap < NANBEIGE_VOCAB_SIZE {
            assert!(admitted.native.max_model_work.projections.dot_products < reference.native.max_model_work.projections.dot_products);
        } else { assert_eq!(admitted.native.max_model_work, reference.native.max_model_work); }
    }
}

#[test]
fn invalid_rows_and_packing_without_a_cohort_refuse_before_private_io() {
    for cap in [0, NANBEIGE_VOCAB_SIZE + 1, usize::MAX] {
        let count = cap.to_string(); let (batch, args) = arguments(&["--max-selected-rows", &count]);
        assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    }
    let (mut batch, args) = arguments(&["--max-selected-rows", "7"]);
    batch.stream.cohort_documents = Some(0);
    assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
    let command = definition().try_get_matches_from(["extract-batch", "--source-dir", "source",
        "--source-manifest", "manifest", "--memory-mib", "8192", "--schema", "schema.json",
        "--max-selected-rows", "7", "--prefill-rows", "1"]);
    assert!(command.is_err());
}

#[test]
fn selected_extraction_cohorts_and_packed_prefill_preserve_each_task_and_complete_memory_quote() {
    for documents in [1, 3, 64] {
        for rows in [None, Some(1), Some(64)] {
            let documents = documents.to_string();
            let packed = rows.map(|rows| rows.to_string());
            let mut flags = vec!["--cohort-documents", documents.as_str(), "--max-requests", "8",
                "--max-result-bytes", "65536"];
            if let Some(rows) = &packed { flags.extend(["--prefill-rows", rows.as_str()]); }
            let (dense, mut dense_args) = arguments(&flags); dense_args.memory_mib = 262_144;
            let reference = dense.validate(&dense_args).unwrap();
            for cap in [1, 7, NANBEIGE_VOCAB_SIZE] {
                let cap_string = cap.to_string(); let mut flags = flags.clone();
                flags.extend(["--max-selected-rows", &cap_string]);
                let (selected, mut args) = arguments(&flags); args.memory_mib = 262_144;
                let admitted = selected.validate(&args).unwrap();
                assert_eq!(admitted.selected_rows.unwrap().max_rows_per_step, cap);
                assert_eq!(admitted.common.native, reference.common.native);
                assert_eq!(admitted.common.minimum_memory_bytes, reference.common.minimum_memory_bytes);
                assert_eq!(selected.extraction.budget(&args, admitted.common),
                    dense.extraction.budget(&dense_args, reference.common));
                assert_eq!(admitted.native.max_model_work.projected_logits, 8 * args.max_new_tokens as u64 * cap as u64);
                assert_eq!(admitted.native.max_model_work.attention_pairs, reference.native.max_model_work.attention_pairs);
                assert_eq!(admitted.native.masks.max_visits_per_run, reference.native.masks.max_visits_per_run);
                let memory = Int4ExtractionCohortMemoryLimits { max_documents: admitted.cohort_documents.unwrap(),
                    context_tokens: args.context_tokens, preparation_reserve_bytes: admitted.preparation_reserve_bytes,
                    io_reserve_bytes: admitted.io_bytes, allocator_reserve_bytes: ALLOCATOR_BYTES };
                let quote = memory.requirements_for_execution(args.group_width, admitted.native, admitted.transport,
                    admitted.selected_rows, admitted.common.prefill).unwrap();
                let execution = quote.native.kv_bytes + quote.workspace_bytes + quote.input_bytes + quote.output_reservation_bytes;
                assert_eq!(admitted.common.minimum_memory_bytes, grouped_memory_floor(&args, admitted.common, execution).unwrap());
                args.memory_mib = admitted.common.minimum_memory_bytes.div_ceil(MIB);
                selected.validate(&args).unwrap(); args.memory_mib -= 1;
                assert_eq!(selected.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Arguments));
            }
        }
    }
}

#[test]
fn pinned_snapshot_and_source_keep_the_same_explicit_stream_cap() {
    let (source, source_args) = arguments(&["--max-selected-rows", "7"]);
    let source_limits = source.validate(&source_args).unwrap();
    let digest = "ab".repeat(32);
    let matches = definition().try_get_matches_from(["extract-batch", "--snapshot", "local.snapshot",
        "--expected-model-sha256", &digest, "--memory-mib", "8192", "--schema", "schema.json",
        "--max-selected-rows", "7"]).unwrap();
    let snapshot = BatchArgs::from_matches(&matches).unwrap();
    let args = Int4Args::from_arg_matches(&matches).unwrap();
    let snapshot_limits = snapshot.validate(&args).unwrap();
    assert!(snapshot_limits.common.snapshot_pin.is_some());
    assert_eq!(source, snapshot); assert_eq!(source_limits.selected_rows, snapshot_limits.selected_rows);
    assert_eq!(source_limits.native.max_model_work, snapshot_limits.native.max_model_work);
    assert_eq!(source_limits.transport, snapshot_limits.transport);
}

fn compiler(batch: &BatchArgs, args: &Int4Args) -> Int4ExtractionBatchPlanner {
    let admitted = batch.validate(args).unwrap();
    let budget = batch.extraction.budget(args, admitted.common);
    let schema = if batch.extraction.source_membership() {
        r#"{"type":"string","const":"上海","x-fnlp-source":"verbatim"}"#
    } else { r#"{"type":"boolean"}"# };
    let defaults = batch.load_defaults(schema.to_owned(), budget).unwrap();
    let controls = pinned_controls::pinned().unwrap(); let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let d = Sha256Digest::of_bytes(b"private CLI selected-row batch fixture");
    let packing = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, args.group_width,
    )).unwrap());
    let identity = ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={}", args.group_width), packing_set_digest: packing,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    Int4ExtractionBatchPlanner::pinned(controls.template_controls(), tokenizer.eos_token_id().unwrap(),
        identity, args.group_width, budget, batch.extraction.compiler(), batch.extraction.source_limits(),
        Some(defaults)).unwrap().require_shared_defaults().unwrap()
}

#[test]
fn shared_cli_factory_seals_each_document_before_host_admission() {
    for source in [false, true] {
        let mut flags = vec!["--group-width", "127", "--max-new-tokens", "64"];
        if source { flags.push("--source-membership"); }
        let (dense, args) = arguments(&flags);
        let prepare = |planner: &Int4ExtractionBatchPlanner| planner.prepare(BatchDocument {
            id: "private-id".to_owned(), text: "é 上海 上海 <think>".to_owned(), task_args: None }).unwrap();
        let original = dense.configure_planner(compiler(&dense, &args)).unwrap();
        let default = prepare(&original);
        assert_eq!(default.extraction_plan().execution_version(), INT4_EXTRACT_VERSION);
        flags.extend(["--max-selected-rows", "7"]);
        let (selected, args) = arguments(&flags);
        let changed = selected.configure_planner(compiler(&selected, &args)).unwrap();
        let sparse = prepare(&changed);
        assert_eq!(sparse.extraction_plan().execution_version(), INT4_SPARSE_EXTRACT_VERSION);
        assert_eq!(sparse.extraction_plan().selected_rows().unwrap().max_rows_per_step, 7);
        assert_eq!(sparse.planned_work().projected_logits, 7 * 64);
        assert_eq!(sparse.execution_identity().prompt_digest, default.execution_identity().prompt_digest);
        assert_eq!(sparse.execution_identity().schema_digest, default.execution_identity().schema_digest);
        assert_ne!(sparse.execution_identity().decision_policy_digest, default.execution_identity().decision_policy_digest);
        assert!(sparse.verify_identity(default.execution_identity()).is_err());
        sparse.verify_identity(sparse.execution_identity()).unwrap();
        assert!(dense.configure_planner(changed).is_err());
        assert!(selected.configure_planner(selected.configure_planner(compiler(&selected, &args)).unwrap()).is_err());
        let mut grouped = selected.clone(); grouped.stream.cohort_documents = Some(3);
        let mut grouped_args = args.clone(); grouped_args.memory_mib = 262_144;
        let cohort = grouped.configure_planner(compiler(&grouped, &grouped_args)).unwrap();
        let cohort = prepare(&cohort);
        cohort.verify_identity(sparse.execution_identity()).unwrap();
        assert_eq!(cohort.planned_work(), sparse.planned_work());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn selected_stream_still_requires_the_owned_runtime_before_input() {
    let (batch, args) = arguments(&["--max-selected-rows", "7"]);
    assert_eq!(batch.execute_owned(args, NoRead, NoWrite), Err(CandidateError::Unavailable));
}
