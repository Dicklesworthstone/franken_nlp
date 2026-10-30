//! Explicit scored-head scheduling and charged output ownership.
use super::*;
use crate::native_engine::strict_int4::{Int4Work, StrictInt4Error,
    cohort::MAX_INT4_SEQUENCES, scoring::cohort::required_task_workspace_bytes};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Admission { pub heads: usize, pub workspace_bytes: u64 }
impl Admission {
    pub fn new(heads: usize, work: Int4Work, output: u64) -> Result<Self, CandidateError> {
        let workspace_bytes = required_task_workspace_bytes(work, heads, output)
            .map_err(|_| CandidateError::Arguments)?;
        Ok(Self { heads, workspace_bytes })
    }
    pub fn native(self, context: usize, width: usize) -> Result<Int4MemoryRequirement, StrictInt4Error> {
        if self.heads == 0 || self.heads > MAX_INT4_SEQUENCES { return Err(StrictInt4Error::Input); }
        let contexts = [context; MAX_INT4_SEQUENCES];
        Int4MemoryRequirement::for_cohort(&contexts[..self.heads], width)
    }
}

#[cfg(feature = "asupersync-runtime")]
pub(super) fn matches(task: &Int4Task, args: &Int4Args, limits: Int4Limits) -> Result<bool, CandidateError> {
    let Int4Task::Classify(scoring) = task else { return Ok(limits.cohort.is_none()); };
    let expected = scoring.validate_with_prefill(args)?;
    Ok(expected.cohort == limits.cohort && expected.native == limits.native
        && expected.minimum_memory_bytes == limits.minimum_memory_bytes)
}

#[derive(Serialize)]
pub(super) struct Summary<'a> {
    schema_version: u32, execution: &'a str, group_steps: u64, projection_groups: u64,
}
/// Keep the semantic response at the same location; physical statistics are
/// separate and absent for existing serial and prompt-prefill invocations.
#[derive(Serialize)]
pub(super) struct Response<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    pub response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohort: Option<Summary<'b>>,
}

#[cfg(feature = "asupersync-runtime")]
pub(super) enum Completion<T> {
    Serial(crate::hosted::HostedOutput<T>),
    Cohort(crate::hosted::HostedOutput<crate::native_engine::strict_int4::scoring::cohort::Int4ScoredCohort<T>>),
}
#[cfg(feature = "asupersync-runtime")]
impl<T> Completion<T> {
    pub fn result(&self) -> &T {
        match self { Self::Serial(result) => result.result(), Self::Cohort(result) => &result.result().output }
    }
    pub fn summary(&self) -> Option<Summary<'_>> {
        match self {
            Self::Serial(_) => None,
            Self::Cohort(result) => {
                let result = result.result();
                Some(Summary { schema_version: result.schema_version, execution: &result.execution,
                    group_steps: result.group_steps, projection_groups: result.projection_groups })
            }
        }
    }
}

#[cfg(test)] mod tests;
