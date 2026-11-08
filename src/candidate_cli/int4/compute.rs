//! Explicit physical team selection for complete single-request native tasks.
use super::*;
use crate::native_engine::strict_int4::projection::int4_team_workspace_bytes;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Args { threads: usize }
impl Default for Args { fn default() -> Self { Self { threads: 1 } } }
impl Args {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let threads = *matches.get_one::<usize>("compute_threads").ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::InvalidValue, "explicit compute width required"))?;
        Ok(Self { threads })
    }
    pub(super) fn threads(self) -> usize { self.threads }
    pub(super) fn admit(self, group_width: usize) -> Result<Admission, CandidateError> {
        let workspace_bytes = if self.threads == 1 { 0 }
            else { int4_team_workspace_bytes(self.threads, group_width).map_err(|_| CandidateError::Arguments)? };
        Ok(Admission { threads: self.threads, workspace_bytes })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Admission { pub threads: usize, pub workspace_bytes: u64 }

pub(super) fn definition(command: clap::Command) -> clap::Command {
    command.arg(clap::Arg::new("compute_threads").long("compute-threads")
        .value_name("N").value_parser(clap::value_parser!(usize)).default_value("1")
        .help("Coordinator plus persistent INT4 row workers, 1..=64; 1 is serial. One complete request, shared across its native heads or streamed tokens; cannot combine with --prefill-rows or --cohort-heads. Runtime threads are additional. Worker scratch is admitted before IO"))
}

#[cfg(feature = "asupersync-runtime")]
pub(super) fn matches(task: &Int4Task, args: &Int4Args, limits: Int4Limits)
    -> Result<bool, CandidateError> {
    Ok(limits.compute == args.compute.admit(args.group_width)?
        && (limits.compute.threads == 1
            || (matches!(task, Int4Task::Text(_) | Int4Task::Classify(_) | Int4Task::Sentiment(_)
                | Int4Task::Judge(_) | Int4Task::Extract(_) | Int4Task::Source(..))
                && limits.prefill.is_none() && limits.cohort.is_none())))
}

/// Keep each physical schedule's complete output owner alive while callers
/// serialize a shared semantic response. The result never becomes an unguarded
/// JSON value just to choose between serial, cohort and worker-team routes.
#[cfg(feature = "asupersync-runtime")]
pub(super) enum Completion<T> {
    Scheduled(cohort::Completion<T>),
    Team(crate::hosted::HostedInt4TeamResult<T>),
}
#[cfg(feature = "asupersync-runtime")]
impl<T> Completion<T> {
    pub fn serial(output: crate::hosted::HostedOutput<T>) -> Self {
        Self::Scheduled(cohort::Completion::Serial(output))
    }
    pub fn grouped(output: crate::hosted::HostedOutput<
        crate::native_engine::strict_int4::scoring::cohort::Int4ScoredCohort<T>>) -> Self {
        Self::Scheduled(cohort::Completion::Cohort(output))
    }
    pub fn result(&self) -> &T {
        match self { Self::Scheduled(output) => output.result(), Self::Team(output) => output.result() }
    }
    pub fn compute(&self) -> Option<&crate::hosted::Int4ComputeReceipt> {
        match self { Self::Scheduled(_) => None, Self::Team(output) => Some(output.compute()) }
    }
    pub fn cohort(&self) -> Option<cohort::Summary<'_>> {
        match self { Self::Scheduled(output) => output.summary(), Self::Team(_) => None }
    }
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(super) struct Response<'a, T: Serialize> {
    #[serde(flatten)]
    pub response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute: Option<&'a crate::hosted::Int4ComputeReceipt>,
}
