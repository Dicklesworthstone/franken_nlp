//! Explicit physical projection authority, separate from semantic draw keys.
//! Every grammar-legal row remains eligible. A row cap refuses an oversized
//! legal set; it never discounts the full sampler, grammar mask or KV memory.
use super::*;
use crate::native_engine::constrained_int8::{INT8_JSON_SAMPLED_SPARSE_EXECUTION,
    INT8_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION};

pub const INT8_SAMPLED_SENTIMENT_SPARSE_EXECUTION: &str =
    "portable-int8-independent-sampled-sentiment-selected-rows-v1";
pub const INT8_SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION: &str =
    "portable-int8-independent-sampled-sentiment-selected-rows-reused-prefill-v1";

impl Int8SampledSentimentPlanner {
    /// Project every grammar-legal token under an explicit per-step ceiling.
    /// The complete cold causal schedule remains admitted even when a later
    /// execution reuses a prefix. Only the head's declared row count changes.
    /// Admission binds this policy without changing an independent draw key.
    pub fn with_selected_rows(mut self, limits: Int8JsonSparseLimits)
        -> Result<Self, Int8SampledSentimentError> {
        limits.validate()?;
        self.selected_rows = Some(limits);
        self.refresh_projection_identity()?;
        Ok(self)
    }

    pub fn selected_rows(&self) -> Option<Int8JsonSparseLimits> { self.selected_rows }

    pub(super) fn refresh_projection_identity(&mut self) -> Result<(), Int8SampledSentimentError> {
        self.admission_identity = self.selected_rows.map(|limits|
            bind_projection(&self.identity, limits)).transpose()?;
        Ok(())
    }
}

impl PreparedInt8SampledSentiment {
    pub fn selected_rows(&self) -> Option<Int8JsonSparseLimits> { self.selected_rows }

    pub fn execution_version(&self, reuse_prefill: bool) -> &'static str {
        execution(self.selected_rows, reuse_prefill)
    }

    pub(super) fn with_projection(mut self, selected: Option<Int8JsonSparseLimits>)
        -> Result<Self, Int8SampledSentimentError> {
        if self.selected_rows.is_some() { return Err(SampledSentimentError::Accounting.into()); }
        let Some(limits) = selected else { return Ok(self); };
        limits.validate()?;
        let mut work = Int8Work::default();
        for axis in &mut self.inner.axes {
            // The native sampler hashes this exact semantic identity. Item and
            // original chunk addresses may extend it later, independently of
            // row caps, prefill morsels and physical worker scheduling.
            axis.work = constrained_int8::sparse::planned_work(axis.prompt.len(),
                self.inner.options.max_sample_tokens, limits)?;
            for _ in 0..self.inner.options.samples { work = work.checked_add(axis.work)?; }
        }
        self.inner.identity = bind_projection(&self.inner.identity, limits)?;
        self.inner.work = work;
        self.selected_rows = Some(limits);
        Ok(self)
    }
}

fn bind_projection(identity: &ExecutionIdentity, limits: Int8JsonSparseLimits)
    -> Result<ExecutionIdentity, Int8SampledSentimentError> {
    limits.validate()?;
    let mut admitted = identity.clone();
    admitted.decision_policy_digest = digest(&("int8-sampled-sentiment-selected-admission-v1",
        identity.receipt_identity().map_err(|_| SampledSentimentError::Identity)?,
        INT8_SAMPLED_SENTIMENT_SPARSE_EXECUTION, INT8_SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION,
        INT8_JSON_SAMPLED_SPARSE_EXECUTION, INT8_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION,
        limits.max_rows_per_step))?;
    admitted.validate().map_err(|_| SampledSentimentError::Identity)?;
    Ok(admitted)
}

pub(super) fn execution(selected: Option<Int8JsonSparseLimits>, reuse_prefill: bool) -> &'static str {
    match (selected.is_some(), reuse_prefill) {
        (false, false) => INT8_SAMPLED_SENTIMENT_EXECUTION,
        (false, true) => INT8_SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION,
        (true, false) => INT8_SAMPLED_SENTIMENT_SPARSE_EXECUTION,
        (true, true) => INT8_SAMPLED_SENTIMENT_SPARSE_REUSED_PREFILL_EXECUTION,
    }
}

pub(super) fn native_execution(selected: Option<Int8JsonSparseLimits>, reuse_prefill: bool) -> &'static str {
    match (selected.is_some(), reuse_prefill) {
        (false, false) => INT8_JSON_SAMPLED_EXECUTION,
        (false, true) => constrained_int8::INT8_JSON_SAMPLED_REUSED_PREFILL_EXECUTION,
        (true, false) => INT8_JSON_SAMPLED_SPARSE_EXECUTION,
        (true, true) => INT8_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION,
    }
}
