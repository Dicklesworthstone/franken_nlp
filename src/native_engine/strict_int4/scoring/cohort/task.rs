//! Closed task composition keeps semantic decisions in their existing finalizer.
use super::*;

pub(crate) fn contexts(schedules: &[CandidateSchedule]) -> Result<Vec<usize>, Int4ScoringError> {
    check_count(schedules.len())?;
    let width = schedules[0].group_width; check_group(width)?;
    let mut contexts = reserve(schedules.len())?;
    for schedule in schedules {
        if schedule.group_width != width { return Err(Int4ScoringError::Identity); }
        if schedule.context == 0 || schedule.context > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(Int4ScoringError::Input); }
        contexts.push(schedule.context);
    }
    Ok(contexts)
}
pub(crate) fn finish<T, E, F>(run: Int4CandidateCohortRun, schedules: &[CandidateSchedule], expected: Int4Work,
    cap: u64, finalize: F) -> Result<Int4ScoredCohort<T>, E>
where T: Serialize, E: From<Int4ScoringError>, F: FnOnce(&mut std::vec::IntoIter<Int4CandidateRun>) -> Result<T, E> {
    let validate = (|| -> Result<(), Int4ScoringError> {
        check_count(schedules.len())?;
        let width = schedules[0].group_width; check_group(width)?;
        if run.schema_version != 1 || run.execution != INT4_SCORING_COHORT_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width || run.model_work != expected
            || run.heads.len() != schedules.len() || run.group_steps == 0 || run.group_steps > expected.forward_positions {
            return Err(Int4ScoringError::Accounting);
        }
        let mut work = Int4Work::default(); let mut projections = 0_u64;
        for (head, schedule) in run.heads.iter().zip(schedules) {
            if schedule.group_width != width || head.group_width != width
                || head.schema_version != 1 || head.execution != INT4_SCORING_EXECUTION
                || head.numerics_profile != STRICT_INT4_PROFILE || head.model_work != schedule.model
                || head.scores.work != schedule.scoring || head.rewound_positions != schedule.rewound {
                return Err(Int4ScoringError::Accounting);
            }
            work = work.checked_add(head.model_work)?;
            projections = add(projections, schedule.scoring.prefix_evaluations as u64)?;
        }
        if work != expected || run.projection_groups == 0 || run.projection_groups > projections {
            return Err(Int4ScoringError::Accounting);
        }
        Ok(())
    })();
    validate.map_err(E::from)?;
    let mut heads = run.heads.into_iter();
    let output = finalize(&mut heads)?;
    if heads.next().is_some() { return Err(E::from(Int4ScoringError::Accounting)); }
    let result = Int4ScoredCohort { schema_version: 1, execution: INT4_SCORING_COHORT_EXECUTION.to_owned(),
        group_steps: run.group_steps, projection_groups: run.projection_groups, output };
    check_output(&result, cap).map_err(E::from)?;
    Ok(result)
}

#[cfg(test)] pub(crate) mod fixtures;
