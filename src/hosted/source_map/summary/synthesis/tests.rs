//! Model-free admission/accounting definitions; no hosted neural execution.
use super::*;
use crate::{execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::{Int8Work, STRICT_INT8_EXECUTION},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits}, summarize::SummaryOptions,
        source_planning::quantized::long::summary::synthesis::SummarySynthesisLimits},
    validation::grounded_fields::GroundingBudget};
fn config() -> SourceMapConfig<SourceSummarySynthesis> {
    let d = Sha256Digest::of_bytes(b"host-summary-synthesis-model-free-fixture");
    SourceMapConfig {
        identity: ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
            tokenizer_digest: d, template_digest: d, task_spec: SUMMARIZE_TASK_VERSION.to_owned(), taskir_digest: d,
            prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
            sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
            host_class: None, compiler_identity: None },
        task: SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
            limits: SummarySynthesisLimits { max_evidence_segments: 32, max_evidence_bytes: 1024,
                verification: GroundingBudget::default() } },
        budget: TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
            max_grammar_states: 4096, max_kv_bytes: 1024 },
        planning: SourcePlanningLimits { max_context_tokens: 128, ..SourcePlanningLimits::default() },
        mapping: Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 1024, max_chunk_bytes: 16,
                max_chunk_tokens: 16, context_tokens: 128, reserved_tokens: 32, max_chunks: 16, ..ChunkLimits::default() },
            reduction: ExecutionLimits { max_value_bytes: 4096, max_live_value_bytes: 8192,
                max_result_bytes: 4096, ..ExecutionLimits::default() },
            max_model_work: Int8Work::for_sequence(0, 128, 128 * 166_144).unwrap(),
            mask_limits: Default::default(), mask_visits_per_chunk: 1000, max_mask_visits: 16_000 },
        native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 4096,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 100, cleanup_reserve_bytes: 4096 } },
        preparation_reserve_bytes: 4096, reduction_reserve_bytes: 4096,
    }
}
#[test]
fn host_checks_full_native_capacity_task_profile_and_both_summary_option_sets() {
    validate_synthesis(&config(), 10, 1024).unwrap();
    for axis in 0..6 {
        let mut c = config();
        match axis { 0 => c.identity.task_spec = "answer-v1".to_owned(),
            1 => c.identity.numerics_profile = NumericsProfile::HfBf16Eager,
            2 => c.planning.max_context_tokens = 129, 3 => c.budget.max_kv_bytes = 1023,
            4 => c.task.map_options.max_bullets = 0, _ => c.task.synthesis_options.max_citations_per_bullet = 0 }
        assert!(validate_synthesis(&c, 10, 1024).is_err(), "{axis}");
    }
    assert!(validate_synthesis(&config(), 0, 1024).is_err());
    assert!(validate_synthesis(&config(), 1025, 1024).is_err());
}
#[test]
fn evidence_reserve_covers_final_native_output_prefix_tables_and_origin_fanout() {
    let c = config(); let l = c.task.limits;
    let expected = reduction_bytes(&c, 3).unwrap() + 4096 * 4 + 16 * 8 + 1024 * 64 + 32 * 128
        + l.verification.max_matches as u64 * 128 + l.verification.max_fields as u64 * 128;
    assert_eq!(synthesis_bytes(&c, 3).unwrap(), expected);
    assert!(synthesis_bytes(&c, 0).is_err()); assert!(synthesis_bytes(&c, 17).is_err());
    let mut bad = config(); bad.budget.max_output_bytes = u64::MAX;
    assert!(synthesis_bytes(&bad, 3).is_err());
}
#[test]
fn evidence_limits_cannot_exceed_source_preparation_or_be_zero() {
    for axis in 0..5 {
        let mut c = config();
        match axis { 0 => c.task.limits.max_evidence_bytes = c.planning.max_input_bytes + 1,
            1 => c.task.limits.max_evidence_segments = 0, 2 => c.task.limits.verification.max_fields = 0,
            3 => c.task.limits.verification.max_matches = 1_000_001,
            _ => c.task.limits.verification.max_scan_steps = 0 }
        assert!(validate_synthesis(&c, 10, 1024).is_err(), "{axis}");
    }
}
