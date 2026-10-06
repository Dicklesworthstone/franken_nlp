//! Closed coordinator revalidation, not a public native-receipt constructor.
use super::*;

impl Int8ExtractPlan {
    /// Recheck a native result against THIS sealed schema/source/policy before
    /// a composed task changes coordinates or aggregates it. The native driver
    /// remains the only public producer. This does not authenticate arbitrary
    /// serialized output or prove that tokens were evaluated by a model.
    pub(crate) fn verify_completed(&self, run: &Int8ExtractRun) -> Result<(), Int8ExtractError> {
        // Bound both the original envelope and the temporary validation copy.
        check_size(run, self.max_result_bytes())?;
        if run.schema_version != 1 || run.execution != self.execution_version() {
            return Err(ExtractError::InvalidResult.into());
        }
        let output = &run.result.output;
        let work = self.completed_work(output.token_ids.len(), output.projected_logits)?;
        if work != run.model_work || output.forward_positions != work.forward_positions
            || output.projected_logits != work.projected_logits {
            return Err(Int8JsonError::WorkMismatch.into());
        }
        // The existing finalizer checks EOS/control IDs, exact decimal JSON,
        // schema validity, optional/nested verbatim fields, every occurrence,
        // grounding/profile metadata and the complete per-task byte ceiling.
        let checked = self.extraction.finalize_profile(output.clone(), STRICT_INT8_PROFILE)?;
        if checked != run.result { return Err(ExtractError::InvalidResult.into()); }
        Ok(())
    }
}
