//! Explicit opt-in retention for the existing owned redaction NDJSON route.
//! Job authentication and pseudonymization use distinct caller-owned secrets.
use super::*;
use crate::jobs::{JobId, JobLimits};
use super::corpus::CorpusEnvelope;

#[derive(Args, Default)]
pub(in crate::candidate_cli) struct RetentionArgs {
    /// Retain private completed edits in an owner-only authenticated job spool.
    /// Stdout becomes metadata only. Original inputs are NOT stored for resume.
    #[arg(long, requires_all = ["ndjson", "job_dir", "job_id", "job_key_file", "job_limits"])]
    pub store_results: bool,
    /// Existing protected job directory; never created or adopted automatically.
    #[arg(long, requires = "store_results")]
    pub job_dir: Option<PathBuf>,
    /// Random 128-bit job ID, exactly 32 lowercase hexadecimal characters.
    #[arg(long, requires = "store_results", value_parser = parse_job_id)]
    pub job_id: Option<JobId>,
    /// Protected file with 32 RAW job-authentication bytes, not a pseudonym key.
    #[arg(long, requires = "store_results")]
    pub job_key_file: Option<PathBuf>,
    /// Immutable JobLimits JSON; includes lifetime attempts, native work and masks.
    #[arg(long, requires = "store_results")]
    pub job_limits: Option<PathBuf>,
    /// Authenticate the COMPLETE original population and run only pending items.
    #[arg(long, requires = "store_results")]
    pub resume: bool,
    /// Repair uncommitted tails only AFTER the original contract authenticates.
    #[arg(long, requires = "resume")]
    pub discard_uncommitted_tail: bool,
    /// Publish protected materialized.ndjson only when every item is committed.
    #[arg(long, requires = "store_results")]
    pub materialize: bool,
    /// Whole input line count including blanks (default 100000); no flush records.
    #[arg(long, requires = "store_results")]
    max_job_input_lines: Option<u64>,
    /// Modeled database RAM, additional to file-size bounds (default 64 MiB).
    #[arg(long, requires = "store_results")]
    journal_memory_mib: Option<u64>,
    /// Canonical serialization/allocator reserve (default 16 MiB).
    #[arg(long, requires = "store_results")]
    serialization_memory_mib: Option<u64>,
}
impl RetentionArgs {
    pub(in crate::candidate_cli) fn max_lines(&self) -> u64 { self.max_job_input_lines.unwrap_or(100_000) }
    pub(in crate::candidate_cli) fn journal_bytes(&self) -> Result<u64, CandidateError> {
        self.journal_memory_mib.unwrap_or(64).checked_mul(MIB).filter(|&n| n > 0).ok_or(CandidateError::Arguments)
    }
    pub(in crate::candidate_cli) fn serialization_bytes(&self) -> Result<u64, CandidateError> {
        self.serialization_memory_mib.unwrap_or(16).checked_mul(MIB).filter(|&n| n > 0).ok_or(CandidateError::Arguments)
    }
    pub(super) fn validate(&self, ndjson: bool, limits: Limits) -> Result<(), CandidateError> {
        if !self.store_results {
            if self.job_dir.is_some() || self.job_id.is_some() || self.job_key_file.is_some() || self.job_limits.is_some()
                || self.resume || self.discard_uncommitted_tail || self.materialize || self.max_job_input_lines.is_some()
                || self.journal_memory_mib.is_some() || self.serialization_memory_mib.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(());
        }
        let local = |p: Option<&PathBuf>| p.is_some_and(|p| !p.as_os_str().is_empty() && p.as_os_str() != "-");
        if !ndjson || self.job_id.is_none() || !local(self.job_dir.as_ref())
            || !local(self.job_key_file.as_ref()) || !local(self.job_limits.as_ref())
            || self.discard_uncommitted_tail && !self.resume || !(1..=1_000_000_000).contains(&self.max_lines())
            || self.journal_bytes()? > limits.memory_bytes || self.serialization_bytes()? > limits.memory_bytes {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    pub(in crate::candidate_cli) fn check_limits(&self, command: &RedactCommand,
        envelope: CorpusEnvelope, lifetime: JobLimits) -> Result<(), CandidateError> {
        lifetime.validate().map_err(|_| CandidateError::Arguments)?;
        // The durable input bound includes the entire original JSON envelope.
        // Keep it within the original-source cap: transformed-text headroom
        // must never authorize a larger original document through this route.
        let stored_bytes = lifetime.max_spool_bytes.checked_add(lifetime.max_materialized_bytes)
            .ok_or(CandidateError::Arguments)?;
        if !self.store_results || lifetime.max_items > envelope.transport.max_requests
            || lifetime.max_id_bytes > envelope.transport.max_id_bytes
            || lifetime.max_input_bytes_per_item > command.host.max_input_bytes
            || lifetime.max_input_bytes_per_item > envelope.transport.max_line_bytes
            || lifetime.max_snapshot_bytes > envelope.transport.max_input_bytes
            || command.host.max_result_bytes > lifetime.max_result_bytes || stored_bytes > envelope.output_bytes {
            return Err(CandidateError::Arguments);
        }
        // Lifetime work may exceed one invocation's cap to price explicit
        // retries. Neither the host nor JobRunner refunds earlier attempts.
        Ok(())
    }
}
fn parse_job_id(text: &str) -> Result<JobId, &'static str> {
    if text.len() != 32 || !text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("expected 32 lowercase hexadecimal characters");
    }
    let nibble = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
    let mut id = [0; 16];
    for (byte, pair) in id.iter_mut().zip(text.as_bytes().chunks_exact(2)) { *byte = nibble(pair[0]) * 16 + nibble(pair[1]); }
    Ok(JobId(id))
}

#[cfg(test)] pub(in crate::candidate_cli) mod tests;
