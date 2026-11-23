//! Physical INT8 row teams are explicitly exposed by supported task routes.
use super::*;
use crate::native_engine::strict_int8::{prefill::Int8PrefillLimits,
    projection::{int8_team_workspace_bytes, int8_team_workspace_bytes_with_prefill}};

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
    /// One admitted team spans prompt morsels and subsequent per-token work.
    pub(super) fn admit_prefill(self, prefill: Option<Int8PrefillLimits>) -> Result<Admission, CandidateError> {
        if let Some(limits) = prefill { limits.validate().map_err(|_| CandidateError::Arguments)?; }
        if self.threads == 1 { return Ok(Admission { threads: 1, workspace_bytes: 0 }); }
        let workspace_bytes = match prefill {
            None => int8_team_workspace_bytes(self.threads),
            Some(limits) => int8_team_workspace_bytes_with_prefill(self.threads, limits.max_batch_rows),
        }.map_err(|_| CandidateError::Arguments)?;
        Ok(Admission { threads: self.threads, workspace_bytes })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Admission { pub threads: usize, pub workspace_bytes: u64 }

pub(super) fn definition(command: clap::Command) -> clap::Command {
    command.arg(clap::Arg::new("compute_threads").long("compute-threads")
        .value_name("N").value_parser(clap::value_parser!(usize)).default_value("1")
        .help("Coordinator plus persistent INT8 row workers, 1..=64; 1 keeps serial execution. One fixed team serves this complete request across both decoder passes, prompt tokens and generation. Runtime threads are additional. Worker storage and native memory are admitted before input/model IO. Buffered and streamed generate/chat, schema extraction, one-request source tasks, classification, sentiment and judgment can combine workers with --prefill-rows on the same team. Maps, batches and jobs do not expose this option"))
}

/// Price the entire simultaneously live threaded request, including the
/// immutable model, existing coordinator buffers and fixed child payload.
/// Validation precedes input/model IO; the host later reserves each component.
/// Width one preserves the established serial admission and output behavior.
pub(super) fn check_memory(args: &CandidateArgs, limits: Limits, native: Int8MemoryRequirement)
    -> Result<(), CandidateError> {
    if limits.compute.threads == 1 { return Ok(()); }
    let prefill_bytes = args.policy.prefill()?.map(|prefill| prefill.max_extra_scratch_bytes).unwrap_or(0);
    let peak = request_peak_bytes(limits, native, limits.result_bytes as u64,
        args.max_new_tokens as u64, SAMPLER_BYTES, 0)?
        .checked_add(prefill_bytes).ok_or(CandidateError::Arguments)?;
    if peak > limits.memory_bytes { return Err(CandidateError::Arguments); }
    Ok(())
}

/// Source execution retains the caller's preparation alongside a separately
/// charged transferred plan/vocabulary/source owner. Neither source bytes nor
/// that second reservation disappear when a compact or selected head is used.
pub(super) fn check_source_memory(args: &CandidateArgs, limits: Limits, max_result_bytes: usize,
    prefill: Option<Int8PrefillLimits>) -> Result<(), CandidateError> {
    if limits.compute.threads == 1 && prefill.is_none() { return Ok(()); }
    if source_peak_bytes(args, limits, max_result_bytes, prefill)? > limits.memory_bytes {
        return Err(CandidateError::Arguments);
    }
    Ok(())
}

pub(super) fn source_peak_bytes(args: &CandidateArgs, limits: Limits, max_result_bytes: usize,
    prefill: Option<Int8PrefillLimits>) -> Result<u64, CandidateError> {
    let native = Int8MemoryRequirement::for_context(args.context_tokens).map_err(|_| CandidateError::Arguments)?;
    // Prompt activations stay on the coordinator while children retain their
    // separately priced fixed batch buffers. Neither replaces native scratch.
    let extra = prefill.map(Int8PrefillLimits::validate).transpose()
        .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
    request_peak_bytes(limits, native, max_result_bytes as u64, args.max_new_tokens as u64,
        0, limits.preparation_bytes)?.checked_add(extra).ok_or(CandidateError::Arguments)
}

pub(super) fn request_peak_bytes(limits: Limits, native: Int8MemoryRequirement, result_bytes: u64,
    output_tokens: u64, sampler_bytes: u64, transferred_preparation_bytes: u64)
    -> Result<u64, CandidateError> {
    let output = result_bytes.checked_mul(4)
        .and_then(|bytes| bytes.checked_add(output_tokens.checked_mul(8)?))
        .and_then(|bytes| bytes.checked_add(4096)).ok_or(CandidateError::Arguments)?;
    [limits.weight_bytes, 128 * MIB, 64 * MIB, limits.preparation_bytes, transferred_preparation_bytes,
        native.kv_bytes, native.rope_bytes, native.scratch_payload_bound, sampler_bytes,
        64 * MIB, limits.compute.workspace_bytes, output, 65_536, 64 * 1024]
        .into_iter().try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or(CandidateError::Arguments)
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
