//! Physical INT8 row teams are explicitly exposed only by buffered text routes.
use super::*;
use crate::native_engine::strict_int8::projection::int8_team_workspace_bytes;

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
    pub(super) fn admit(self, prefill: bool) -> Result<Admission, CandidateError> {
        if self.threads == 1 { return Ok(Admission { threads: 1, workspace_bytes: 0 }); }
        if prefill { return Err(CandidateError::Arguments); }
        let workspace_bytes = int8_team_workspace_bytes(self.threads).map_err(|_| CandidateError::Arguments)?;
        Ok(Admission { threads: self.threads, workspace_bytes })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Admission { pub threads: usize, pub workspace_bytes: u64 }

pub(super) fn definition(command: clap::Command) -> clap::Command {
    command.arg(clap::Arg::new("compute_threads").long("compute-threads")
        .value_name("N").value_parser(clap::value_parser!(usize)).default_value("1")
        .help("Coordinator plus persistent INT8 row workers, 1..=64; 1 keeps serial execution. One fixed team serves this complete buffered request across both decoder passes, prompt tokens and generation. Runtime threads are additional. Worker storage and native memory are admitted before input/model IO. Widths above 1 cannot combine with --prefill-rows; streams, batches and jobs do not expose this option"))
}

/// Price the entire simultaneously live threaded request, including the
/// immutable model, existing coordinator buffers and fixed child payload.
/// Validation precedes input/model IO; the host later reserves each component.
/// Width one preserves the established serial admission and output behavior.
pub(super) fn check_memory(args: &CandidateArgs, limits: Limits, native: Int8MemoryRequirement)
    -> Result<(), CandidateError> {
    if limits.compute.threads == 1 { return Ok(()); }
    let output = (limits.result_bytes as u64).checked_mul(4)
        .and_then(|bytes| bytes.checked_add(args.max_new_tokens as u64 * 8))
        .and_then(|bytes| bytes.checked_add(4096)).ok_or(CandidateError::Arguments)?;
    let peak = [limits.weight_bytes, 128 * MIB, 64 * MIB, limits.preparation_bytes,
        native.kv_bytes, native.rope_bytes, native.scratch_payload_bound, SAMPLER_BYTES,
        64 * MIB, limits.compute.workspace_bytes, output, 65_536, 64 * 1024]
        .into_iter().try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or(CandidateError::Arguments)?;
    if peak > limits.memory_bytes { return Err(CandidateError::Arguments); }
    Ok(())
}

#[cfg(feature = "asupersync-runtime")]
pub(super) fn resource_config(limits: Limits) -> crate::ResourceHostConfig {
    crate::ResourceHostConfig {
        runtime_preset: crate::RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: limits.compute.threads - 1,
        helper_threads: 0, thread_ceiling: limits.compute.threads + 1,
        memory_ceiling_bytes: limits.memory_bytes,
        leak_response_policy: crate::LeakResponsePolicy::RecordAndEscalate,
    }
}

#[cfg(feature = "asupersync-runtime")]
pub(super) enum Completion<T> {
    Serial(crate::hosted::HostedOutput<T>),
    Team(crate::hosted::HostedInt8TeamResult<T>),
}
#[cfg(feature = "asupersync-runtime")]
impl<T> Completion<T> {
    pub(super) fn result(&self) -> &T {
        match self { Self::Serial(output) => output.result(), Self::Team(output) => output.result() }
    }
    pub(super) fn compute(&self) -> Option<&crate::hosted::Int8ComputeReceipt> {
        match self { Self::Serial(_) => None, Self::Team(output) => Some(output.compute()) }
    }
}

#[cfg(feature = "asupersync-runtime")]
#[derive(Serialize)]
pub(super) struct Response<'a, T: Serialize> {
    #[serde(flatten)] pub response: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute: Option<&'a crate::hosted::Int8ComputeReceipt>,
}

#[cfg(test)] mod tests;
