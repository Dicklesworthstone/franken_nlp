//! Explicit bounded refill selection; no inferred strategy or smaller fallback.
use super::*;
use crate::native_engine::{portable_int8::batch::MAX_BATCH_ROWS, strict_int8::prefill::Int8PrefillLimits};
impl TextBatchCommand {
    /// count is the actual current input window, including a partial final tail.
    /// The tail may have fewer members than active_rows; this is not a fallback
    /// after failed memory/native admission. Queue/output limits do not shrink.
    pub(in crate::candidate_cli) fn refill_strategy(&self, count: usize)
        -> Result<Option<(usize, Int8PrefillLimits)>, CandidateError> {
        let Some(active) = self.active_rows else { return Ok(None); };
        let queued = self.cohort_rows.ok_or(CandidateError::Arguments)?;
        if queued == 0 || queued > MAX_BATCH_ROWS || active == 0 || active > queued || count == 0 || count > queued {
            return Err(CandidateError::Arguments);
        }
        let active = active.min(count);
        let prefill = match self.common.policy.prefill()? {
            Some(prefill) => prefill,
            None => Int8PrefillLimits { max_batch_rows: active,
                max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(active)
                    .map_err(|_| CandidateError::Arguments)? },
        };
        Ok(Some((active, prefill)))
    }
}
#[cfg(test)] mod tests;
