//! Explicit physical team selection for buffered single-request text only.
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
        .help("Coordinator plus persistent INT4 row workers, 1..=64; 1 is serial. Buffered single-request text only; cannot combine with --prefill-rows. Runtime threads are additional. Worker scratch is admitted before IO"))
}

#[cfg(feature = "asupersync-runtime")]
pub(super) fn matches(task: &Int4Task, args: &Int4Args, limits: Int4Limits)
    -> Result<bool, CandidateError> {
    Ok(limits.compute == args.compute.admit(args.group_width)?
        && (limits.compute.threads == 1
            || (matches!(task, Int4Task::Text(_)) && limits.prefill.is_none() && limits.cohort.is_none())))
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(super) struct Response<'a, T: Serialize> {
    #[serde(flatten)]
    pub response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute: Option<&'a crate::hosted::Int4ComputeReceipt>,
}
