//! Whole-hierarchy admission fixtures without native model execution.
use super::*;
use crate::{
    execution_identity::NumericsProfile,
    tasks::{answer::AnswerOptions,
        source_planning::grouped_int4::long::question::synthesis::QuestionSynthesisLimits},
    validation::grounded_fields::GroundingBudget,
};

fn fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceQuestionSynthesis>) {
    let (planner, base) = super::super::super::super::tests::fixture(
        SourceMapTask::Summarize(Default::default()), width);
    let mut identity = base.identity;
    identity.task_spec = ANSWER_TASK_VERSION.to_owned();
    (planner, Int4SourceMapConfig {
        identity,
        task: SourceQuestionSynthesis {
            question: SourceQuestion { question: "Who founded the company?".to_owned(),
                options: AnswerOptions::default(), verification: GroundingBudget {
                    max_fields: 512, max_matches: 1024, max_scan_steps: 1 << 20 } },
            limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 1024 },
        },
        budget: base.budget, planning: base.planning, mapping: base.mapping, native: base.native,
        preparation_reserve_bytes: base.preparation_reserve_bytes,
        reduction_reserve_bytes: base.reduction_reserve_bytes,
    })
}

#[test]
fn every_permitted_pass_prices_native_answers_and_level_witnesses() {
    let (_, config) = fixture(32);
    let hierarchy = QuestionHierarchyLimits { max_passes: 4, max_levels: 3, ..Default::default() };
    let before = config.question_hierarchy_workspace_bytes(2, hierarchy).unwrap();
    let more = QuestionHierarchyLimits { max_passes: 5, ..hierarchy };
    assert_eq!(config.question_hierarchy_workspace_bytes(2, more).unwrap() - before,
        config.budget.max_output_bytes * 4 + u64::from(config.budget.max_output_tokens) * 8 + 8448);
    let single = QuestionHierarchyLimits { max_passes: 1, max_levels: 1, ..hierarchy };
    assert_eq!(config.question_hierarchy_workspace_bytes(2, single).unwrap()
        - config.question_synthesis_workspace_bytes(2).unwrap(), 8448 + 128);
}

#[test]
fn deeper_levels_do_not_renew_evidence_or_verification_storage() {
    let (_, config) = fixture(32);
    let shallow = QuestionHierarchyLimits { max_passes: 8, max_levels: 2, ..Default::default() };
    let deep = QuestionHierarchyLimits { max_levels: 3, ..shallow };
    assert_eq!(config.question_hierarchy_workspace_bytes(2, deep).unwrap()
        - config.question_hierarchy_workspace_bytes(2, shallow).unwrap(), 128);
    assert_eq!(hierarchy_output_tokens(512, shallow).unwrap(), hierarchy_output_tokens(512, deep).unwrap());
}

#[test]
fn output_guard_retains_every_reduction_transcript_after_engine_drain() {
    let hierarchy = QuestionHierarchyLimits::default();
    assert_eq!(hierarchy_output_tokens(512, hierarchy).unwrap(), 16 * 512);
    let maximum = QuestionHierarchyLimits { max_passes: 256, ..hierarchy };
    assert_eq!(hierarchy_output_tokens(u32::MAX, maximum).unwrap(), 256 * u64::from(u32::MAX));
    assert!(hierarchy_output_tokens(0, hierarchy).is_err());
    let single = QuestionHierarchyLimits { max_passes: 1, ..hierarchy };
    assert_eq!(hierarchy_output_tokens(512, single).unwrap(), 512);
}

#[test]
fn invalid_level_pass_and_grouping_limits_refuse_before_dispatch() {
    for axis in 0..8 {
        let (planner, config) = fixture(32);
        let mut hierarchy = QuestionHierarchyLimits::default();
        match axis {
            0 => hierarchy.max_levels = 0, 1 => hierarchy.max_levels = 17,
            2 => hierarchy.max_passes = 0, 3 => hierarchy.max_passes = 257,
            4 => hierarchy.max_tokenizer_calls = 0, 5 => hierarchy.max_tokenizer_calls = 65_537,
            6 => hierarchy.max_tokenizer_bytes = 0, _ => hierarchy.max_tokenizer_bytes = 1_000_000_001,
        }
        assert!(hierarchy_requirements(&config, hierarchy, 32, 32, &planner).is_err(), "axis {axis}");
        assert!(config.question_hierarchy_workspace_bytes(2, hierarchy).is_err());
        assert!(hierarchy_output_tokens(512, hierarchy).is_err());
    }
}

#[test]
fn actual_group_assets_and_full_allocated_kv_remain_required() {
    let hierarchy = QuestionHierarchyLimits::default();
    for width in [16, 32, 128] {
        let (planner, config) = fixture(width);
        assert_eq!(hierarchy_requirements(&config, hierarchy, 32, width, &planner).unwrap().kv_bytes,
            config.budget.max_kv_bytes);
        assert!(hierarchy_requirements(&config, hierarchy, 32, if width == 16 { 32 } else { 16 }, &planner).is_err());
    }
    for axis in 0..7 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.identity.task_spec = "summarize-v1".to_owned(),
            1 => config.identity.tokenizer_digest = Sha256Digest::of_bytes(b"different-tokenizer"),
            2 => config.identity.template_digest = Sha256Digest::of_bytes(b"different-template"),
            3 => config.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            4 => config.budget.max_kv_bytes -= 1,
            5 => config.native.context_tokens += 1,
            _ => config.planning.max_context_tokens += 1,
        }
        assert!(hierarchy_requirements(&config, hierarchy, 32, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn hierarchy_preserves_question_evidence_and_whole_source_admission() {
    let hierarchy = QuestionHierarchyLimits::default();
    for axis in 0..10 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.task.question.question = " \n\t".to_owned(),
            1 => config.task.question.question = "q".repeat(config.budget.max_input_tokens as usize + 1),
            2 => config.task.question.options.max_citations = 0,
            3 => config.task.question.verification.max_fields = 0,
            4 => config.task.limits.max_evidence_passages = 1025,
            5 => config.task.limits.max_evidence_passages = config.planning.max_passages + 1,
            6 => config.task.limits.max_evidence_bytes = config.planning.max_input_bytes + 1,
            7 => config.preparation_reserve_bytes = 0,
            8 => config.reduction_reserve_bytes = 0,
            _ => config.native.allocator_reserve_bytes = 0,
        }
        assert!(hierarchy_requirements(&config, hierarchy, 32, 32, &planner).is_err(), "axis {axis}");
    }
    let (planner, config) = fixture(32);
    for bytes in [0, config.mapping.chunks.max_input_bytes + 1] {
        assert!(hierarchy_requirements(&config, hierarchy, bytes, 32, &planner).is_err());
    }
}

#[test]
fn both_private_string_capacities_stay_inside_the_owned_input_charge() {
    let (_, mut config) = fixture(32);
    let mut question = String::with_capacity(8192);
    question.push_str("Who?");
    config.task.question.question = question;
    assert_eq!(synthesis_input_bytes(4096, &config).unwrap(),
        4096 + config.task.question.question.capacity() as u64 + config.preparation_reserve_bytes);
    config.preparation_reserve_bytes = u64::MAX;
    assert!(synthesis_input_bytes(1, &config).is_err());
}

#[test]
fn overflows_and_invalid_discovery_populations_never_underreserve() {
    let (_, config) = fixture(32);
    let hierarchy = QuestionHierarchyLimits::default();
    for chunks in [0, 9, 257, usize::MAX] {
        assert!(config.question_hierarchy_workspace_bytes(chunks, hierarchy).is_err());
    }
    let (_, mut config) = fixture(32);
    config.budget.max_output_bytes = u64::MAX / 16;
    assert!(config.question_hierarchy_workspace_bytes(2, hierarchy).is_err());
    let (_, mut config) = fixture(32);
    config.reduction_reserve_bytes = u64::MAX;
    assert!(config.question_hierarchy_workspace_bytes(2, hierarchy).is_err());
}

#[test]
fn complete_map_results_and_shared_occurrence_fanout_remain_priced() {
    let (_, mut config) = fixture(32);
    let hierarchy = QuestionHierarchyLimits::default();
    let before = config.question_hierarchy_workspace_bytes(2, hierarchy).unwrap();
    assert_eq!(config.question_hierarchy_workspace_bytes(3, hierarchy).unwrap() - before,
        config.workspace_bytes(3).unwrap() - config.workspace_bytes(2).unwrap());
    config.task.question.verification.max_matches += 1;
    assert_eq!(config.question_hierarchy_workspace_bytes(2, hierarchy).unwrap() - before,
        128 + (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64);
}

#[test]
fn the_whole_private_owner_and_complete_result_cross_the_existing_host() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapConfig<SourceQuestionSynthesis>>();
    send::<Charged<SourceMapInput<SourceQuestionSynthesis>>>();
    send::<Int4QuestionHierarchyRun>();
    send::<HostedOutput<Int4QuestionHierarchyCohortRun>>();
    let _method = NlpEngine::synthesize_int4_document_answer_hierarchy;
    let _cohorts = NlpEngine::synthesize_int4_document_answer_hierarchy_cohorts;
}

#[test]
fn discovery_cohort_adapter_keeps_question_and_all_retained_hierarchy_transcripts() {
    use native_cohorts::SynthesisMode;
    let (planner, mut config) = fixture(32);
    let hierarchy = QuestionHierarchyLimits { max_passes: 4, max_levels: 3, ..Default::default() };
    let budget = config.budget;
    let workspace = config.question_hierarchy_workspace_bytes(7, hierarchy).unwrap();
    config.mapping.reduction.map_batch_chunks = 4;
    let row = hierarchy.requirements(&config, 32, 32, &planner).unwrap();
    assert_eq!(config.synthesis_cohort_requirements(32).unwrap().native.kv_bytes, 4 * row.kv_bytes);
    assert_eq!(hierarchy.workspace_bytes(&config, 7).unwrap(), workspace);
    assert_eq!(hierarchy.output_tokens(&config, 7).unwrap(), 4 * u64::from(budget.max_output_tokens));
    let mut question = String::with_capacity(8192); question.push_str("Who?");
    config.task.question.question = question;
    assert_eq!(hierarchy.input_bytes(4096, &config).unwrap(),
        4096 + config.task.question.question.capacity() as u64 + config.preparation_reserve_bytes);
    assert_eq!(config.budget, budget);
    let invalid = QuestionHierarchyLimits { max_passes: 0, ..hierarchy };
    assert!(invalid.requirements(&config, 32, 32, &planner).is_err());
}
