//! A pre-admission head choice sealed into the complete task identity.
use super::*;

pub const INT8_SPARSE_EXTRACT_VERSION: &str = "strict-int8-selected-row-schema-source-extraction-v1";

impl Int8ExtractPlan {
    /// Consume an unadmitted full-head plan and explicitly select the complete
    /// legal-row strategy. Exact prompt, schema, source, tokenizer and controls
    /// are unchanged. Old admissions and durable keys cannot authorize it.
    /// A second strategy selection is a refusal, not an identity rewrite loop.
    pub fn with_selected_rows(mut self, limits: selected_head::Int8JsonSparseLimits)
        -> Result<Self, Int8ExtractError> {
        limits.validate()?;
        if self.selected_rows.is_some() { return Err(Int8ExtractError::Identity); }
        let work = selected_head::planned_work(self.prompt_tokens(), self.options().max_new_tokens, limits)?;
        let policy = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
            INT8_SPARSE_EXTRACT_VERSION, selected_head::INT8_SPARSE_JSON_EXECUTION,
            self.identity.decision_policy_digest, limits.max_rows_per_step as u64,
        )).map_err(|_| ExtractError::Serialization)?);
        self.identity.decision_policy_digest = policy;
        self.identity.validate().map_err(|_| Int8ExtractError::Identity)?;
        self.work = work; self.selected_rows = Some(limits);
        Ok(self)
    }
    pub fn selected_rows(&self) -> Option<selected_head::Int8JsonSparseLimits> { self.selected_rows }
    pub fn execution_version(&self) -> &'static str {
        if self.selected_rows.is_some() { INT8_SPARSE_EXTRACT_VERSION } else { INT8_EXTRACT_VERSION }
    }
    pub(super) fn native_execution_version(&self) -> &'static str {
        if self.selected_rows.is_some() { selected_head::INT8_SPARSE_JSON_EXECUTION } else { INT8_JSON_EXECUTION }
    }
    /// The actual session reconciles EVERY requested row with its projection
    /// ledger. This closed task boundary additionally verifies all completed
    /// work axes and the immutable strategy's bounds; it is not a wire receipt
    /// authenticator or a second model evaluator.
    pub(super) fn completed_work(&self, count: usize, projected: u64) -> Result<Int8Work, Int8ExtractError> {
        if count == 0 || count > self.options().max_new_tokens { return Err(ExtractError::InvalidResult.into()); }
        match self.selected_rows {
            None => {
                let work = constrained_int8::planned_work(self.prompt_tokens(), count)?;
                if projected != work.projected_logits { return Err(Int8JsonError::WorkMismatch.into()); }
                Ok(work)
            }
            Some(limits) => {
                let rows = usize::try_from(projected).map_err(|_| Int8JsonError::WorkMismatch)?;
                let maximum = count.checked_mul(limits.max_rows_per_step).ok_or(Int8JsonError::WorkMismatch)?;
                if rows < count || rows > maximum { return Err(Int8JsonError::WorkMismatch.into()); }
                let positions = self.prompt_tokens().checked_add(count - 1).ok_or(Int8JsonError::WorkMismatch)?;
                Ok(Int8Work::for_sequence(0, positions, rows).map_err(Int8JsonError::from)?)
            }
        }
    }
}
