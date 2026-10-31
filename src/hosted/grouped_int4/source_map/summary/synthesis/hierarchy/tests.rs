//! Whole-hierarchy admission and lifetime fixtures, without native execution.
use super::*;
use crate::{
    execution_identity::NumericsProfile,
    tasks::source_planning::grouped_int4::long::summary::synthesis::SummarySynthesisLimits,
    validation::grounded_fields::GroundingBudget,
};

fn fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceSummarySynthesis>) {
    let (planner, base) = super::super::super::super::tests::fixture(
        SourceMapTask::Summarize(Default::default()), width);
    (planner, Int4SourceMapConfig {
        identity: base.identity,
        task: SourceSummarySynthesis { map_options: Default::default(), synthesis_options: Default::default(),
            limits: SummarySynthesisLimits { max_evidence_segments: 32, max_evidence_bytes: 1024,
                verification: GroundingBudget { max_fields: 512, max_matches: 1024, max_scan_steps: 1 << 20 } } },
        budget: base.budget, planning: base.planning, mapping: base.mapping, native: base.native,
        preparation_reserve_bytes: base.preparation_reserve_bytes,
        reduction_reserve_bytes: base.reduction_reserve_bytes,
    })
}

#[test]
fn each_permitted_pass_prices_retained_output_and_complete_level_witnesses() {
    let (_, config) = fixture(32);
    let hierarchy = SummaryHierarchyLimits { max_passes: 4, max_levels: 3, ..Default::default() };
    let one = config.summary_hierarchy_workspace_bytes(2, hierarchy).unwrap();
    let more = SummaryHierarchyLimits { max_passes: 5, ..hierarchy };
    assert_eq!(config.summary_hierarchy_workspace_bytes(2, more).unwrap() - one,
        config.budget.max_output_bytes * 4 + u64::from(config.budget.max_output_tokens) * 8 + 8448);
    let single = SummaryHierarchyLimits { max_passes: 1, max_levels: 1, ..hierarchy };
    assert_eq!(config.summary_hierarchy_workspace_bytes(2, single).unwrap()
        - config.summary_synthesis_workspace_bytes(2).unwrap(), 8448 + 128);
}

#[test]
fn deeper_hierarchy_never_renews_the_shared_evidence_or_verification_reserve() {
    let (_, config) = fixture(32);
    let shallow = SummaryHierarchyLimits { max_passes: 8, max_levels: 2, ..Default::default() };
    let deep = SummaryHierarchyLimits { max_levels: 3, ..shallow };
    assert_eq!(config.summary_hierarchy_workspace_bytes(2, deep).unwrap()
        - config.summary_hierarchy_workspace_bytes(2, shallow).unwrap(), 128);
    assert_eq!(hierarchy_output_tokens(2, 512, shallow).unwrap(), hierarchy_output_tokens(2, 512, deep).unwrap());
}

#[test]
fn output_guard_keeps_all_maps_and_reduce_transcripts_after_native_drain() {
    let hierarchy = SummaryHierarchyLimits { max_passes: 16, ..Default::default() };
    assert_eq!(hierarchy_output_tokens(64, 512, hierarchy).unwrap(), 80 * 512);
    let maximum = SummaryHierarchyLimits { max_passes: 256, ..hierarchy };
    assert_eq!(hierarchy_output_tokens(256, u32::MAX, maximum).unwrap(), 512 * u64::from(u32::MAX));
    for chunks in [0, 257, usize::MAX] { assert!(hierarchy_output_tokens(chunks, 512, hierarchy).is_err()); }
    assert!(hierarchy_output_tokens(1, 0, hierarchy).is_err());
}

#[test]
fn invalid_levels_passes_or_tokenizer_work_cannot_enter_the_host() {
    for axis in 0..8 {
        let (planner, config) = fixture(32);
        let mut hierarchy = SummaryHierarchyLimits::default();
        match axis {
            0 => hierarchy.max_levels = 0, 1 => hierarchy.max_levels = 17,
            2 => hierarchy.max_passes = 0, 3 => hierarchy.max_passes = 257,
            4 => hierarchy.max_tokenizer_calls = 0, 5 => hierarchy.max_tokenizer_calls = 65_537,
            6 => hierarchy.max_tokenizer_bytes = 0, _ => hierarchy.max_tokenizer_bytes = 1_000_000_001,
        }
        assert!(hierarchy_requirements(&config, hierarchy, 32, 32, &planner).is_err(), "axis {axis}");
        assert!(config.summary_hierarchy_workspace_bytes(2, hierarchy).is_err());
        assert!(hierarchy_output_tokens(2, 512, hierarchy).is_err());
    }
}

#[test]
fn every_group_uses_actual_profile_assets_and_complete_resident_capacity() {
    let hierarchy = SummaryHierarchyLimits::default();
    for width in [16, 32, 128] {
        let (planner, config) = fixture(width);
        assert_eq!(hierarchy_requirements(&config, hierarchy, 32, width, &planner).unwrap().kv_bytes,
            config.budget.max_kv_bytes);
        assert!(hierarchy_requirements(&config, hierarchy, 32, if width == 16 { 32 } else { 16 }, &planner).is_err());
    }
    for axis in 0..7 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.identity.task_spec = "answer-v1".to_owned(),
            1 => config.identity.tokenizer_digest = Sha256Digest::of_bytes(b"different-tokenizer"),
            2 => config.identity.template_digest = Sha256Digest::of_bytes(b"different-template"),
            3 => config.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            4 => config.budget.max_kv_bytes -= 1,
            5 => config.native.context_tokens += 1,
            _ => config.planning.max_context_tokens += 1,
        }
        assert!(hierarchy_requirements(&config, hierarchy, 1, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn source_summary_options_and_charged_storage_remain_required_with_hierarchy() {
    let hierarchy = SummaryHierarchyLimits::default();
    let (planner, config) = fixture(32);
    for bytes in [0, config.mapping.chunks.max_input_bytes + 1] {
        assert!(hierarchy_requirements(&config, hierarchy, bytes, 32, &planner).is_err());
    }
    for axis in 0..6 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.task.map_options.max_bullets = 0,
            1 => config.task.synthesis_options.max_citations_per_bullet = 0,
            2 => config.task.limits.verification.max_matches = 0,
            3 => config.preparation_reserve_bytes = 0,
            4 => config.reduction_reserve_bytes = 0,
            _ => config.native.allocator_reserve_bytes = 0,
        }
        assert!(hierarchy_requirements(&config, hierarchy, 32, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn multi_pass_workspace_cannot_wrap_to_an_underpriced_admission() {
    let (_, mut config) = fixture(32);
    let hierarchy = SummaryHierarchyLimits::default();
    for count in [0, 9, 257, usize::MAX] {
        assert!(config.summary_hierarchy_workspace_bytes(count, hierarchy).is_err());
    }
    config.budget.max_output_bytes = u64::MAX / 8;
    assert!(config.summary_hierarchy_workspace_bytes(1, hierarchy).is_err());
    let (_, mut config) = fixture(32);
    config.reduction_reserve_bytes = u64::MAX;
    assert!(config.summary_hierarchy_workspace_bytes(1, hierarchy).is_err());
}

#[test]
fn increasing_map_population_retains_its_entire_native_frontier() {
    let (_, config) = fixture(128);
    let hierarchy = SummaryHierarchyLimits::default();
    assert_eq!(config.summary_hierarchy_workspace_bytes(3, hierarchy).unwrap()
        - config.summary_hierarchy_workspace_bytes(2, hierarchy).unwrap(),
        config.budget.max_output_bytes * 4 + u64::from(config.budget.max_output_tokens) * 8);
    assert_eq!(hierarchy_output_tokens(3, 512, hierarchy).unwrap()
        - hierarchy_output_tokens(2, 512, hierarchy).unwrap(), 512);
}

#[test]
fn hierarchy_reuses_the_same_sendable_charged_owner_and_blocking_host() {
    fn send<T: Send + 'static>() {}
    send::<Charged<SourceMapInput<SourceSummarySynthesis>>>();
    send::<SummaryHierarchyLimits>();
    send::<HostedOutput<Int4SummaryHierarchyCohortRun>>();
    let _method = NlpEngine::synthesize_int4_source_summary_hierarchy;
    let _cohorts = NlpEngine::synthesize_int4_source_summary_hierarchy_cohorts;
}

#[test]
fn discovery_cohort_adapter_keeps_every_hierarchy_pass_in_workspace_and_output() {
    use native_cohorts::SynthesisMode;
    let (planner, mut config) = fixture(32);
    let hierarchy = SummaryHierarchyLimits { max_passes: 4, max_levels: 3, ..Default::default() };
    let budget = config.budget;
    let workspace = config.summary_hierarchy_workspace_bytes(7, hierarchy).unwrap();
    config.mapping.reduction.map_batch_chunks = 4;
    let row = hierarchy.requirements(&config, 32, 32, &planner).unwrap();
    assert_eq!(config.synthesis_cohort_requirements(32).unwrap().native.kv_bytes, 4 * row.kv_bytes);
    assert_eq!(hierarchy.workspace_bytes(&config, 7).unwrap(), workspace);
    assert_eq!(hierarchy.output_tokens(&config, 7).unwrap(), 11 * u64::from(budget.max_output_tokens));
    assert_eq!(hierarchy.input_bytes(8192, &config).unwrap(), 8192 + config.preparation_reserve_bytes);
    assert_eq!(config.budget, budget);
    let invalid = SummaryHierarchyLimits { max_passes: 0, ..hierarchy };
    assert!(invalid.requirements(&config, 32, 32, &planner).is_err());
}
