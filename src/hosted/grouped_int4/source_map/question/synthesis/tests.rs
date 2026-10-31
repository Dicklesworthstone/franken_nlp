//! Admission and private-owner fixtures without native model execution.
use super::*;
use crate::{
    execution_identity::NumericsProfile,
    native_engine::decode::DecodeCancellationKind,
    tasks::{answer::AnswerOptions, source_planning::grouped_int4::{Int4SourceError,
        long::question::synthesis::QuestionSynthesisLimits}},
    validation::grounded_fields::GroundingBudget,
};

fn fixture(width: usize) -> (SourceTaskPlanner, Int4SourceMapConfig<SourceQuestionSynthesis>) {
    let (planner, base) = super::super::super::tests::fixture(
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
fn synthesis_keeps_actual_group_and_pinned_question_assets() {
    for width in [16, 32, 128] {
        let (planner, config) = fixture(width);
        assert_eq!(synthesis_requirements(&config, 32, width, &planner).unwrap().kv_bytes,
            config.budget.max_kv_bytes);
        assert!(synthesis_requirements(&config, 32, if width == 16 { 32 } else { 16 }, &planner).is_err());
    }
    for axis in 0..4 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.identity.task_spec = "summarize-v1".to_owned(),
            1 => config.identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            2 => config.identity.tokenizer_digest = Sha256Digest::of_bytes(b"changed-tokenizer"),
            _ => config.identity.template_digest = Sha256Digest::of_bytes(b"changed-template"),
        }
        assert!(synthesis_requirements(&config, 32, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn even_a_short_question_requires_full_native_allocation() {
    for axis in 0..6 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.budget.max_kv_bytes -= 1,
            1 => config.native.context_tokens += 1,
            2 => config.planning.max_context_tokens += 1,
            3 => config.preparation_reserve_bytes = 0,
            4 => config.reduction_reserve_bytes = 0,
            _ => config.native.allocator_reserve_bytes = 0,
        }
        assert!(synthesis_requirements(&config, 1, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn question_and_shared_verification_require_finite_nonzero_authority() {
    for axis in 0..7 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.task.question.question = " \r\n".to_owned(),
            1 => config.task.question.question = "x".repeat(config.budget.max_input_tokens as usize + 1),
            2 => config.task.question.options.max_citations = 0,
            3 => config.task.question.verification.max_fields = 0,
            4 => config.task.question.verification.max_matches = 1_000_001,
            5 => config.task.question.verification.max_scan_steps = 0,
            _ => config.planning.max_input_bytes = 3,
        }
        assert!(synthesis_requirements(&config, 32, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn all_collected_evidence_must_fit_the_same_passage_and_byte_envelope() {
    for axis in 0..5 {
        let (planner, mut config) = fixture(32);
        match axis {
            0 => config.task.limits.max_evidence_passages = 0,
            1 => config.task.limits.max_evidence_passages = 1025,
            2 => config.task.limits.max_evidence_passages = config.planning.max_passages + 1,
            3 => config.task.limits.max_evidence_bytes = 0,
            _ => config.task.limits.max_evidence_bytes = config.planning.max_input_bytes + 1,
        }
        assert!(synthesis_requirements(&config, 32, 32, &planner).is_err(), "axis {axis}");
    }
}

#[test]
fn source_bounds_and_complete_partition_are_priced_before_dispatch() {
    let (planner, config) = fixture(32);
    for bytes in [0, config.mapping.chunks.max_input_bytes + 1] {
        assert!(synthesis_requirements(&config, bytes, 32, &planner).is_err());
    }
    for chunks in [0, 9, 257, usize::MAX] {
        assert!(config.question_synthesis_workspace_bytes(chunks).is_err());
    }
}

#[test]
fn retained_source_and_question_capacities_are_both_charged() {
    let (_, mut config) = fixture(32);
    let mut question = String::with_capacity(8192);
    question.push_str("Who?");
    config.task.question.question = question;
    assert_eq!(synthesis_input_bytes(4096, &config).unwrap(),
        4096 + config.task.question.question.capacity() as u64 + config.preparation_reserve_bytes);
    let before = synthesis_input_bytes(4096, &config).unwrap();
    assert_eq!(synthesis_input_bytes(8192, &config).unwrap() - before, 4096);
    config.preparation_reserve_bytes = u64::MAX;
    assert!(synthesis_input_bytes(1, &config).is_err());
}

#[test]
fn final_native_result_and_evidence_are_additional_to_complete_map_frontier() {
    let (_, config) = fixture(32);
    let before = config.question_synthesis_workspace_bytes(2).unwrap();
    assert!(before >= config.workspace_bytes(2).unwrap() + 4 * config.budget.max_output_bytes
        + 8 * u64::from(config.budget.max_output_tokens));
    assert_eq!(config.question_synthesis_workspace_bytes(3).unwrap() - before,
        config.workspace_bytes(3).unwrap() - config.workspace_bytes(2).unwrap());
}

#[test]
fn evidence_copies_and_original_occurrence_fanout_have_nonzero_cost() {
    for axis in 0..4 {
        let (_, mut config) = fixture(32);
        let before = config.question_synthesis_workspace_bytes(2).unwrap();
        let expected = match axis {
            0 => { config.task.limits.max_evidence_bytes += 1; 64 },
            1 => { config.task.limits.max_evidence_passages += 1; 1024 },
            2 => {
                config.task.question.verification.max_fields += 1;
                128 + (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64
            },
            _ => {
                config.task.question.verification.max_matches += 1;
                128 + (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64
            },
        };
        assert_eq!(config.question_synthesis_workspace_bytes(2).unwrap() - before, expected);
    }
}

#[test]
fn temporary_native_storage_cannot_overflow_into_a_small_reservation() {
    let (_, mut config) = fixture(32);
    config.budget.max_output_bytes = u64::MAX;
    assert!(config.question_synthesis_workspace_bytes(2).is_err());
    let (_, mut config) = fixture(32);
    config.reduction_reserve_bytes = u64::MAX;
    assert!(config.question_synthesis_workspace_bytes(2).is_err());
}

#[test]
fn cancellation_stays_typed_without_exposing_the_private_question() {
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable] {
        let error = map_error(Int4SourceMapError::Source(Int4SourceError::Cancelled(cause)));
        let HostedError::Int4(Int4HostError::SourceMap(source)) = &error else { panic!("wrong stage") };
        assert_eq!(source.cancellation(), Some(cause));
        assert!(!format!("{error:?}").contains("Who founded"));
    }
}

#[test]
fn entire_private_synthesis_owner_can_enter_the_existing_host() {
    fn send<T: Send + 'static>() {}
    send::<Int4SourceMapConfig<SourceQuestionSynthesis>>();
    send::<Charged<SourceMapInput<SourceQuestionSynthesis>>>();
    send::<HostedOutput<Int4QuestionSynthesisCohortRun>>();
    let _method = NlpEngine::synthesize_int4_document_answer;
    let _cohorts = NlpEngine::synthesize_int4_document_answer_cohorts;
}

#[test]
fn discovery_cohort_adapter_keeps_private_question_capacity_and_complete_frontier() {
    use native_cohorts::SynthesisMode;
    for width in [32, 128] {
        let (planner, mut config) = fixture(width);
        let mode = QuestionSynthesisMode;
        let budget = config.budget;
        let workspace = config.question_synthesis_workspace_bytes(7).unwrap();
        config.mapping.reduction.map_batch_chunks = 4;
        let row = mode.requirements(&config, 32, width, &planner).unwrap();
        assert_eq!(config.synthesis_cohort_requirements(width).unwrap().native.kv_bytes, 4 * row.kv_bytes);
        assert_eq!(mode.workspace_bytes(&config, 7).unwrap(), workspace);
        assert_eq!(mode.output_tokens(&config, 7).unwrap(), 0);
        let mut question = String::with_capacity(8192); question.push_str("Who?");
        config.task.question.question = question;
        assert_eq!(mode.input_bytes(4096, &config).unwrap(),
            4096 + config.task.question.question.capacity() as u64 + config.preparation_reserve_bytes);
        assert_eq!(config.budget, budget);
        config.task.question.verification.max_fields = 0;
        assert!(mode.requirements(&config, 32, width, &planner).is_err());
    }
}
