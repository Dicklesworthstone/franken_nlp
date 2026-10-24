//! Host admission and whole-frontier accounting; no model execution.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{Int4Work, STRICT_INT4_EXECUTION}},
    tasks::mapreduce::{ChunkLimits, ExecutionLimits},
    validation::grounded_fields::GroundingBudget,
};

pub(super) fn fixture(task: SourceMapTask, width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig) {
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap();
    let digest = Sha256Digest::of_bytes(b"int4-source-map-host-policy-fixture");
    let task_spec = match &task {
        SourceMapTask::Ner(_) => "ner-v1", SourceMapTask::Keyphrases(_) => "keyphrases-v1",
        SourceMapTask::Summarize(_) => "summarize-v1",
    };
    let context = 2048;
    let kv = Int4MemoryRequirement::for_context(context, width).unwrap().kv_bytes;
    let identity = ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width,
        )).unwrap()),
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task_spec.to_owned(), taskir_digest: digest, prompt_digest: digest,
        grammar_compiler_version: "none".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "constrained-json".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    };
    let config = Int4SourceMapConfig {
        identity, task,
        budget: TaskBudget { max_input_tokens: 1536, max_output_tokens: 512,
            max_output_bytes: 65_536, max_grammar_states: 8192, max_kv_bytes: kv },
        planning: SourcePlanningLimits { max_context_tokens: context, ..Default::default() },
        mapping: Int4SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 128,
                max_chunk_tokens: 512, context_tokens: context, reserved_tokens: 512,
                max_chunks: 8, max_tokenizer_calls: 1024 },
            reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 4,
                max_reduction_levels: 8, max_task_calls: 64, max_value_bytes: 65_536,
                max_live_value_bytes: 1 << 20, max_total_value_bytes: 1 << 22, max_result_bytes: 65_536 },
            max_model_work: Int4Work::for_sequence(width, 0, 4096, 128 * NANBEIGE_VOCAB_SIZE).unwrap(),
            mask_limits: MaskWorkLimits { max_trie_node_visits: 1000, checkpoint_interval_nodes: 16 },
            mask_visits_per_chunk: 4000, max_mask_visits: 32_000,
            verification: GroundingBudget { max_fields: 512, max_matches: 1024, max_scan_steps: 1 << 20 },
        },
        native: NativeLimits { context_tokens: context, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
                cleanup_reserve_bytes: 65_536 } },
        preparation_reserve_bytes: 1 << 20, reduction_reserve_bytes: 65_536,
    };
    (planner, config)
}

#[test]
fn all_three_source_tasks_keep_the_actual_group_and_pinned_assets() {
    for width in [16, 32, 128] {
        for task in [SourceMapTask::Ner(Default::default()), SourceMapTask::Keyphrases(Default::default()),
            SourceMapTask::Summarize(Default::default())] {
            let (planner, config) = fixture(task, width);
            let required = map_requirements(&config, 16, width, &planner).unwrap();
            assert_eq!(required.kv_bytes, config.budget.max_kv_bytes);
            assert!(map_requirements(&config, 16, if width == 16 { 32 } else { 16 }, &planner).is_err());
        }
    }
    for axis in 0..4 {
        let (planner, mut config) = fixture(SourceMapTask::Ner(Default::default()), 32);
        match axis {
            0 => config.identity.task_spec = "answer-v1".to_owned(),
            1 => config.identity.tokenizer_digest = Sha256Digest::of_bytes(b"changed-tokenizer"),
            2 => config.identity.template_digest = Sha256Digest::of_bytes(b"changed-template"),
            _ => config.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
        }
        assert!(map_requirements(&config, 16, 32, &planner).is_err());
    }
}

#[test]
fn complete_capacity_source_and_owned_storage_are_required_before_preparation() {
    for axis in 0..7 {
        let (planner, mut config) = fixture(SourceMapTask::Keyphrases(Default::default()), 32);
        match axis {
            0 => config.budget.max_kv_bytes -= 1,
            1 => config.native.context_tokens += 1,
            2 => config.planning.max_context_tokens += 1,
            3 => config.preparation_reserve_bytes = 0,
            4 => config.reduction_reserve_bytes = 0,
            5 => config.native.allocator_reserve_bytes = 0,
            _ => config.mapping.max_model_work.projected_logits = NANBEIGE_VOCAB_SIZE as u64 - 1,
        }
        assert!(map_requirements(&config, 16, 32, &planner).is_err());
    }
    let (planner, config) = fixture(SourceMapTask::Ner(Default::default()), 32);
    for bytes in [0, config.mapping.chunks.max_input_bytes + 1] {
        assert!(map_requirements(&config, bytes, 32, &planner).is_err());
    }
}

#[test]
fn frontier_quote_prices_all_chunks_local_and_lifted_evidence() {
    let (_, mut config) = fixture(SourceMapTask::Summarize(Default::default()), 32);
    let one = config.workspace_bytes(1).unwrap();
    let two = config.workspace_bytes(2).unwrap();
    assert!(two >= one + config.budget.max_output_bytes * 4
        + u64::from(config.budget.max_output_tokens) * 8);
    config.mapping.verification.max_fields += 1;
    assert_eq!(config.workspace_bytes(2).unwrap() - two, std::mem::size_of::<SourceFieldEvidence>() as u64);
    let before_spans = config.workspace_bytes(2).unwrap();
    config.mapping.verification.max_matches += 1;
    assert_eq!(config.workspace_bytes(2).unwrap() - before_spans,
        2 * std::mem::size_of::<VerifiedSourceSpan>() as u64);
    config.mapping.reduction.max_live_value_bytes += 1;
    assert_eq!(config.workspace_bytes(2).unwrap() - before_spans,
        2 * std::mem::size_of::<VerifiedSourceSpan>() as u64 + 4);
}

#[test]
fn partition_cardinality_and_workspace_overflow_cannot_weaken_reservation() {
    let (_, mut config) = fixture(SourceMapTask::Ner(Default::default()), 32);
    for count in [0, 9, 257, usize::MAX] { assert!(config.workspace_bytes(count).is_err()); }
    config.budget.max_output_bytes = u64::MAX;
    assert!(config.workspace_bytes(2).is_err());
    let (_, mut config) = fixture(SourceMapTask::Ner(Default::default()), 32);
    config.reduction_reserve_bytes = 0;
    assert!(config.workspace_bytes(1).is_err());
}

#[test]
fn source_map_and_summary_own_the_same_sendable_capture() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapConfig>();
    send::<Charged<SourceMapInput>>();
    let _map = NlpEngine::map_int4_source;
    let _summary = NlpEngine::summarize_int4_source;
}
