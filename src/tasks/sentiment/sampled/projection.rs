//! A sealed physical projection policy, separate from semantic draw addresses.
//! Selected rows never prune the grammar or discount sampler/KV reservations.
use super::*;
use crate::native_engine::constrained_int4::{INT4_JSON_SAMPLED_SPARSE_EXECUTION,
    INT4_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION};

pub const SAMPLED_SENTIMENT_SPARSE_EXECUTION: &str =
    "grouped-int4-independent-sampled-sentiment-selected-rows-v1";
pub const SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION: &str =
    "grouped-int4-independent-sampled-sentiment-selected-rows-reused-prefill-v1";

impl SampledSentimentPlanner {
    /// Explicitly project every grammar-legal row under a hard per-step cap.
    /// An oversized legal set refuses the sample rather than pruning it or
    /// silently falling back to a more expensive full projection. Admission
    /// identities bind this policy; semantic sample addresses remain unchanged.
    pub fn with_selected_rows(mut self, limits: Int4JsonSparseLimits)
        -> Result<Self, SampledSentimentError> {
        limits.validate()?;
        self.selected_rows = Some(limits);
        self.refresh_projection_identity()?;
        Ok(self)
    }

    pub fn selected_rows(&self) -> Option<Int4JsonSparseLimits> { self.selected_rows }

    pub(super) fn refresh_projection_identity(&mut self) -> Result<(), SampledSentimentError> {
        self.admission_identity = self.selected_rows.map(|limits|
            bind_projection(&self.identity, limits)).transpose()?;
        Ok(())
    }
}

impl PreparedSampledSentiment {
    pub fn selected_rows(&self) -> Option<Int4JsonSparseLimits> { self.selected_rows }

    pub fn execution_version(&self, reuse_prefill: bool) -> &'static str {
        execution(self.selected_rows, reuse_prefill)
    }

    pub(super) fn with_projection(mut self, selected: Option<Int4JsonSparseLimits>)
        -> Result<Self, SampledSentimentError> {
        if self.selected_rows.is_some() { return Err(SampledSentimentError::Accounting); }
        let Some(limits) = selected else { return Ok(self); };
        limits.validate()?;
        let mut work = Int4Work::default();
        for axis in &mut self.axes {
            // Keep axis.identity exactly as compiled: it is the semantic
            // authority hashed by the addressed sampler, including item and
            // original-chunk coordinates applied later by population planners.
            axis.work = constrained_int4::sparse::planned_work(self.group_width,
                axis.prompt.len(), self.options.max_sample_tokens, limits)?;
            for _ in 0..self.options.samples { work = work.checked_add(axis.work)?; }
        }
        self.identity = bind_projection(&self.identity, limits)?;
        self.work = work;
        self.selected_rows = Some(limits);
        Ok(self)
    }
}

fn bind_projection(identity: &ExecutionIdentity, limits: Int4JsonSparseLimits)
    -> Result<ExecutionIdentity, SampledSentimentError> {
    limits.validate()?;
    let mut admitted = identity.clone();
    admitted.decision_policy_digest = digest(&("sampled-sentiment-selected-admission-v1",
        identity.receipt_identity().map_err(|_| SampledSentimentError::Identity)?,
        SAMPLED_SENTIMENT_SPARSE_EXECUTION, SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION,
        INT4_JSON_SAMPLED_SPARSE_EXECUTION, INT4_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION,
        limits.max_rows_per_step))?;
    admitted.validate().map_err(|_| SampledSentimentError::Identity)?;
    Ok(admitted)
}

pub(super) fn execution(selected: Option<Int4JsonSparseLimits>, reuse_prefill: bool) -> &'static str {
    match (selected.is_some(), reuse_prefill) {
        (false, false) => SAMPLED_SENTIMENT_EXECUTION,
        (false, true) => SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION,
        (true, false) => SAMPLED_SENTIMENT_SPARSE_EXECUTION,
        (true, true) => SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION,
    }
}
pub(super) fn native_execution(selected: Option<Int4JsonSparseLimits>, reuse_prefill: bool) -> &'static str {
    match (selected.is_some(), reuse_prefill) {
        (false, false) => INT4_JSON_SAMPLED_EXECUTION,
        (false, true) => INT4_JSON_SAMPLED_REUSED_PREFILL_EXECUTION,
        (true, false) => INT4_JSON_SAMPLED_SPARSE_EXECUTION,
        (true, true) => INT4_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION,
    }
}
