//! Admission and ownership fixtures. These do not execute a native model.
use super::*;
use crate::{
    execution_identity::NumericsProfile,
    tasks::source_planning::grouped_int4::long::summary::synthesis::SummarySynthesisLimits,
    validation::grounded_fields::GroundingBudget,
};

fn fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceSummarySynthesis>) {
    let (planner, base) = super::super::super::tests::fixture(
        SourceMapTask::Summarize(Default::default()), width);
    let config = Int4SourceMapConfig {
        identity: base.identity,
        task: SourceSummarySynthesis {
            map_options: Default::default(), synthesis_options: Default::default(),
            limits: SummarySynthesisLimits { max_evidence_segments: 32, max_evidence_bytes: 1024,
                verification: GroundingBudget { max_fields: 512, max_matches: 1024,
                    max_scan_steps: 1 << 20 } },
        },
        budget: base.budget, planning: base.planning, mapping: base.mapping, native: base.native,
        preparation_reserve_bytes: base.preparation_reserve_bytes,
        reduction_reserve_bytes: base.reduction_reserve_bytes,
    };
    (planner, config)
}

#[test]
fn synthesis_binds_pinned_assets_actual_width_and_task() {
    for width in [16, 32, 128] {
        let (planner, config) = fixture(width);
        let needed = synthesis_requirements(&config, 64, width, &planner).unwrap();
        assert_eq!(needed.kv_bytes, config.budget.max_kv_bytes);
        assert!(synthesis_requirements(&config, 64, if width == 16 { 32 } else { 16 }, &planner).is_err());
    }
    for axis in 0..4 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.identity.task_spec = "answer-v1".to_owned(),
            1 => config.identity.tokenizer_digest = Sha256Digest::of_bytes(b"wrong-tokenizer"),
            2 => config.identity.template_digest = Sha256Digest::of_bytes(b"wrong-template"),
            _ => config.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
        }
        assert!(synthesis_requirements(&config, 64, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn native_allocation_uses_full_resident_capacity_instead_of_prompt_length() {
    let (planner, mut config) = fixture(32);
    config.budget.max_kv_bytes -= 1;
    assert!(synthesis_requirements(&config, 1, 32, &planner).is_err());
    let (planner, mut config) = fixture(32);
    config.native.context_tokens += 1;
    assert!(synthesis_requirements(&config, 1, 32, &planner).is_err());
    let (planner, mut config) = fixture(32);
    config.planning.max_context_tokens += 1;
    assert!(synthesis_requirements(&config, 1, 32, &planner).is_err());
}

#[test]
fn discovery_and_final_options_are_both_validated_before_preparation() {
    for axis in 0..4 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.task.map_options.max_bullets = 0,
            1 => config.task.map_options.max_quote_scalars = 0,
            2 => config.task.synthesis_options.max_bullets = 0,
            _ => config.task.synthesis_options.max_citations_per_bullet = 0,
        }
        assert!(synthesis_requirements(&config, 64, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn evidence_and_independent_verification_limits_cannot_be_disabled() {
    for axis in 0..7 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.task.limits.max_evidence_segments = 0,
            1 => config.task.limits.max_evidence_bytes = 0,
            2 => config.task.limits.max_evidence_bytes = config.planning.max_input_bytes + 1,
            3 => config.task.limits.verification.max_fields = 0,
            4 => config.task.limits.verification.max_matches = 0,
            5 => config.task.limits.verification.max_matches = 1_000_001,
            _ => config.task.limits.verification.max_scan_steps = 0,
        }
        assert!(synthesis_requirements(&config, 64, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn source_and_complete_frontier_must_fit_before_any_model_call() {
    let (planner, config) = fixture(32);
    for bytes in [0, config.mapping.chunks.max_input_bytes + 1] {
        assert!(synthesis_requirements(&config, bytes, 32, &planner).is_err());
    }
    for axis in 0..4 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.preparation_reserve_bytes = 0,
            1 => config.reduction_reserve_bytes = 0,
            2 => config.native.allocator_reserve_bytes = 0,
            _ => config.mapping.chunks.max_chunks = 257,
        }
        assert!(synthesis_requirements(&config, 64, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn final_transcript_is_charged_in_addition_to_every_map_transcript() {
    assert_eq!(synthesis_output_tokens(1, 512).unwrap(), 1024);
    assert_eq!(synthesis_output_tokens(256, 512).unwrap(), 257 * 512);
    for count in [0, 257, usize::MAX] { assert!(synthesis_output_tokens(count, 512).is_err()); }
    assert!(synthesis_output_tokens(1, 0).is_err());
    assert_eq!(synthesis_output_tokens(256, u32::MAX).unwrap(), 257 * u64::from(u32::MAX));
}

#[test]
fn frontier_and_final_payloads_keep_distinct_memory_reservations() {
    let (_, config) = fixture(32);
    let base = config.workspace_bytes(3).unwrap();
    let synthesis = config.summary_synthesis_workspace_bytes(3).unwrap();
    assert!(synthesis >= base + 4 * config.budget.max_output_bytes
        + 8 * u64::from(config.budget.max_output_tokens));
    assert_eq!(config.summary_synthesis_workspace_bytes(4).unwrap() - synthesis,
        config.workspace_bytes(4).unwrap() - base);
}

#[test]
fn quote_origin_and_verifier_fanout_cannot_be_free() {
    for axis in 0..4 {
        let (_, mut config) = fixture(32);
        let before = config.summary_synthesis_workspace_bytes(2).unwrap();
        let expected = match axis {
            0 => { config.task.limits.max_evidence_bytes += 1; 64 },
            1 => { config.task.limits.max_evidence_segments += 1; 128 },
            2 => { config.task.limits.verification.max_matches += 1; 128 },
            _ => { config.task.limits.verification.max_fields += 1; 128 },
        };
        assert_eq!(config.summary_synthesis_workspace_bytes(2).unwrap() - before, expected);
    }
}

#[test]
fn workspace_arithmetic_refuses_overflow_and_unknown_partition_sizes() {
    let (_, config) = fixture(32);
    for count in [0, 9, 257, usize::MAX] {
        assert!(config.summary_synthesis_workspace_bytes(count).is_err());
    }
    for axis in 0..3 {
        let (_, mut config) = fixture(32);
        match axis {
            0 => config.budget.max_output_bytes = u64::MAX,
            1 => config.task.limits.max_evidence_bytes = usize::MAX,
            _ => config.task.limits.verification.max_matches = usize::MAX,
        }
        assert!(config.summary_synthesis_workspace_bytes(2).is_err(), "axis {axis}");
    }
}

#[test]
fn whole_synthesis_capture_is_sendable_and_uses_existing_host() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapConfig<SourceSummarySynthesis>>();
    send::<Charged<SourceMapInput<SourceSummarySynthesis>>>();
    let _method = NlpEngine::synthesize_int4_source_summary;
}
