//! A complete legal-row strategy sealed before task admission.
use super::*;

pub const INT4_SPARSE_EXTRACT_VERSION: &str = "grouped-int4-selected-row-schema-source-extraction-v1";

impl Int4ExtractPlan {
    /// Consume an unadmitted full-head plan and explicitly select every legal
    /// grammar row. The ceiling refuses a complete set that is too large; it
    /// never prunes it. Prompt, schema, source and controls remain unchanged.
    /// Full-head admissions and durable keys cannot authorize this strategy.
    /// A second selection is refused instead of repeatedly rewriting identity.
    pub fn with_selected_rows(mut self, limits: selected_head::Int4JsonSparseLimits)
        -> Result<Self, Int4ExtractError> {
        limits.validate()?;
        if self.selected_rows.is_some() { return Err(Int4ExtractError::Identity); }
        let work = selected_head::planned_work(self.group_width, self.prompt_tokens(), self.options().max_new_tokens, limits)?;
        self.identity.decision_policy_digest = selected_policy(self.identity.decision_policy_digest, limits)?;
        // Paired comparisons still ignore only group width. They must retain
        // the exact head strategy, row ceiling and built-in finalizer policy.
        self.comparison_policy_digest = selected_policy(self.comparison_policy_digest, limits)?;
        self.identity.validate().map_err(|_| Int4ExtractError::Identity)?;
        self.work = work;
        self.selected_rows = Some(limits);
        Ok(self)
    }

    pub fn selected_rows(&self) -> Option<selected_head::Int4JsonSparseLimits> { self.selected_rows }
    pub fn execution_version(&self) -> &'static str {
        if self.selected_rows.is_some() { INT4_SPARSE_EXTRACT_VERSION } else { INT4_EXTRACT_VERSION }
    }
    pub(super) fn native_execution_version(&self) -> &'static str {
        if self.selected_rows.is_some() { selected_head::INT4_SPARSE_JSON_EXECUTION } else { INT4_JSON_EXECUTION }
    }

    /// The native session reconciles every projected row with the physical
    /// group ledger. This closed task boundary independently verifies every
    /// work axis and the sealed row bounds. It is not receipt authentication.
    pub(crate) fn completed_work(&self, count: usize, projected: u64) -> Result<Int4Work, Int4ExtractError> {
        if count == 0 || count > self.options().max_new_tokens { return Err(ExtractError::InvalidResult.into()); }
        match self.selected_rows {
            None => {
                let work = constrained_int4::planned_work(self.group_width, self.prompt_tokens(), count)?;
                if projected != work.projected_logits { return Err(Int4JsonError::WorkMismatch.into()); }
                Ok(work)
            }
            Some(limits) => {
                let rows = usize::try_from(projected).map_err(|_| Int4JsonError::WorkMismatch)?;
                let maximum = count.checked_mul(limits.max_rows_per_step).ok_or(Int4JsonError::WorkMismatch)?;
                // Every selection, including terminal EOS, projects at least
                // one row. No forced token may bypass native head accounting.
                if rows < count || rows > maximum { return Err(Int4JsonError::WorkMismatch.into()); }
                let positions = self.prompt_tokens().checked_add(count - 1).ok_or(Int4JsonError::WorkMismatch)?;
                Ok(Int4Work::for_sequence(self.group_width, 0, positions, rows).map_err(Int4JsonError::from)?)
            }
        }
    }
}

fn selected_policy(previous: Sha256Digest, limits: selected_head::Int4JsonSparseLimits)
    -> Result<Sha256Digest, Int4ExtractError> {
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SPARSE_EXTRACT_VERSION, selected_head::INT4_SPARSE_JSON_EXECUTION,
        previous, limits.max_rows_per_step as u64,
    )).map_err(|_| ExtractError::Serialization)?))
}
