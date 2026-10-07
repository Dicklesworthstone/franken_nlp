//! Closed task composition: physical geometry and complete native-head handoff.
//! The task's existing semantic finalizer still owns decisions and labels.
use super::*;

pub(crate) fn contexts(schedules: &[CandidateSchedule]) -> Result<Vec<usize>, Int8ScoringError> {
    check_count(schedules.len())?;
    let mut contexts = reserve(schedules.len())?;
    for schedule in schedules {
        if schedule.context == 0 || schedule.context > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(Int8ScoringError::Input); }
        contexts.push(schedule.context);
    }
    Ok(contexts)
}

pub(crate) fn finish<T, E, F>(run: Int8CandidateCohortRun, schedules: &[CandidateSchedule], expected: Int8Work,
    cap: u64, finalize: F) -> Result<Int8ScoredCohort<T>, E>
where T: Serialize, E: From<Int8ScoringError>, F: FnOnce(&mut std::vec::IntoIter<Int8CandidateRun>) -> Result<T, E> {
    let validate = (|| -> Result<(), Int8ScoringError> {
        check_count(schedules.len())?;
        if run.schema_version != 1 || run.execution != INT8_SCORING_COHORT_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.model_work != expected
            || run.heads.len() != schedules.len() || run.group_steps == 0 || run.group_steps > expected.forward_positions {
            return Err(Int8ScoringError::Accounting);
        }
        let mut work = Int8Work::default(); let mut projections = 0_u64;
        for (head, schedule) in run.heads.iter().zip(schedules) {
            if head.schema_version != 1 || head.execution != INT8_SCORING_EXECUTION
                || head.numerics_profile != STRICT_INT8_PROFILE || head.model_work != schedule.model
                || head.scores.work != schedule.scoring || head.rewound_positions != schedule.rewound {
                return Err(Int8ScoringError::Accounting);
            }
            work = work.checked_add(head.model_work)?;
            projections = add(projections, schedule.scoring.prefix_evaluations as u64)?;
        }
        if work != expected || run.projection_groups == 0 || run.projection_groups > projections {
            return Err(Int8ScoringError::Accounting);
        }
        Ok(())
    })();
    validate.map_err(E::from)?;
    let mut heads = run.heads.into_iter();
    let output = finalize(&mut heads)?;
    if heads.next().is_some() { return Err(E::from(Int8ScoringError::Accounting)); }
    let result = Int8ScoredCohort { schema_version: 1, execution: INT8_SCORING_COHORT_EXECUTION.to_owned(),
        group_steps: run.group_steps, projection_groups: run.projection_groups, output };
    check_output(&result, cap).map_err(E::from)?;
    Ok(result)
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use crate::execution_identity::{NumericsProfile, ThinkingMode, ToolMode};
    pub(crate) struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    pub(crate) fn identity(task: &str, tokenizer: Sha256Digest, template: Sha256Digest) -> ExecutionIdentity {
        let d = Sha256Digest::of_bytes(b"task-cohort-synthetic-score-fixture");
        ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
            artifact_format: "synthetic-only".into(), quant_recipe: "fixture-int8".into(), packing_set_digest: d,
            tokenizer_digest: tokenizer, template_digest: template, task_spec: task.into(), taskir_digest: d,
            prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
            numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
            sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
            calibration_digest: d, decision_policy_digest: d,
            backend_semantic_version: crate::native_engine::strict_int8::STRICT_INT8_EXECUTION.into(),
            host_class: None, compiler_identity: None }
    }
    pub(crate) struct Constant;
    impl CandidateLogits for Constant {
        type Error = &'static str;
        fn project(&mut self, _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
            let width = match rows { ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size,
                ProjectionRows::Selected(ids) => ids.len() };
            Ok(vec![0.0; width])
        }
    }
    /// Synthetic heads for the private handoff only, NOT a native execution claim.
    pub(crate) fn run(heads: impl IntoIterator<Item = (CandidateScores, CandidateSchedule)>) -> Int8CandidateCohortRun {
        let mut work = Int8Work::default(); let mut projections = 0;
        let heads: Vec<_> = heads.into_iter().map(|(scores, schedule)| {
            work = work.checked_add(schedule.model).unwrap(); projections += schedule.scoring.prefix_evaluations as u64;
            Int8CandidateRun { schema_version: 1, execution: INT8_SCORING_EXECUTION.into(), numerics_profile: STRICT_INT8_PROFILE.into(),
                scores, model_work: schedule.model, rewound_positions: schedule.rewound }
        }).collect();
        Int8CandidateCohortRun { schema_version: 1, execution: INT8_SCORING_COHORT_EXECUTION.into(),
            numerics_profile: STRICT_INT8_PROFILE.into(), heads, group_steps: work.forward_positions,
            projection_groups: projections, model_work: work }
    }
    pub(crate) fn budget() -> crate::tasks::ir::TaskBudget {
        crate::tasks::ir::TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 1_000_000,
            max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
    }
}
