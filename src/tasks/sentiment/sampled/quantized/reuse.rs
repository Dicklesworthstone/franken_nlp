//! Opt-in same-axis prompt reuse for independently addressed INT8 samples.
//!
//! Cold admission and semantic identities are unchanged. Physical execution
//! receipts identify reuse and charge the new work of each causal branch.

use super::*;

pub const INT8_SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION: &str =
    "portable-int8-independent-sampled-sentiment-reused-prefill-v1";

impl PreparedInt8SampledSentiment {
    /// One sampler and one axis's exact sample-address array coexist. No KV,
    /// hidden-state snapshot or saved full-vocabulary logits are added. The
    /// host retains this allowance through native completion and separately
    /// owns preparation/results plus any layer-major prompt workspace.
    pub fn reuse_sampler_requirement_bytes(&self) -> Result<u64, Int8SampledSentimentError> {
        Ok(constrained_int8::json_sampling_reuse_workspace_bytes(self.inner.options.samples)?)
    }

    /// Prefill each exact axis prompt once, then replay its final token from
    /// the retained prefix for each later independent sample. The complete
    /// cold work ceiling must still be admitted before the first native call;
    /// actual receipts count only executed work. The optional prompt schedule
    /// affects the first sample only, and never enters a draw address.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_reusing_prefill_with_control<C: DecodeStepControl>(&self,
        engine: &mut StrictInt8Engine<'_>, admitted: &ExecutionIdentity,
        vocabulary: &SampledSentimentVocabulary, budget: Int8JsonBudget,
        max_sampler_bytes: u64, prefill: Option<Int8PrefillLimits>, control: &mut C)
        -> Result<Int8SampledSentimentRun, Int8SampledSentimentError> {
        checkpoint(control)?;
        if let Some(limits) = prefill { limits.validate()?; }
        self.verify_identity(admitted)?;
        check_model(self.execution_identity(), engine.artifact_identity())?;
        if engine.profile() != STRICT_INT8_PROFILE || engine.is_poisoned()
            || !engine.kv_cache().all_slots_have_len(0) || vocabulary.controls != self.inner.controls {
            return Err(SampledSentimentError::Identity.into());
        }
        let capacity = engine.kv_cache().capacity_positions();
        let resident_kv = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .ok_or(SampledSentimentError::WorkBudget)?;
        if capacity < self.required_context() || resident_kv > budget.json.max_kv_bytes
            || max_sampler_bytes < self.reuse_sampler_requirement_bytes()? {
            return Err(SampledSentimentError::WorkBudget.into());
        }
        // Deliberately retain the task's original cold ceiling. Reuse cannot
        // grant new task authority or turn saved forwards into more samples.
        let mask_charge = self.preflight(vocabulary, budget, max_sampler_bytes)?;
        let mut dimensions = reserved(self.inner.axes.len())?;
        let mut actual_work = Int8Work::default();
        let mut actual_mask = 0_u64;
        let mut actual_tokens = 0_u64;
        let mut retained_sample_bytes = 0_u64;
        for axis in &self.inner.axes {
            let mut addresses = reserved(self.inner.options.samples)?;
            let mut axis_work = Int8Work::default();
            for sample_index in 0..self.inner.options.samples {
                addresses.push(sampling(&self.inner.options, axis.axis, sample_index)?);
                axis_work = axis_work.checked_add(axis.work)?;
            }
            let axis_masks = mask_charge.checked_mul(self.inner.options.samples as u64)
                .ok_or(SampledSentimentError::WorkBudget)?;
            let axis_budget = Int8JsonBudget { native: Int8RunBudget::exact(axis_work),
                json: JsonWorkBudget { max_forward_positions: axis_work.forward_positions,
                    max_projected_logits: axis_work.projected_logits,
                    max_kv_bytes: budget.json.max_kv_bytes,
                    max_total_mask_node_visits: axis_masks, mask_limits: budget.json.mask_limits } };
            let mut finalize = |sample_index, run| -> Result<Int8SampledSentimentSample, Int8SampledSentimentError> {
                let sample = self.finalize_sample_scheduled(axis, sample_index, run, true)?;
                let bytes = canonjson::canonical_bytes(&sample).map_err(|_| SampledSentimentError::Serialization)?;
                retained_sample_bytes = retained_sample_bytes.checked_add(bytes.len() as u64)
                    .filter(|&n| n <= self.max_result_bytes()).ok_or(SampledSentimentError::OutputBudget)?;
                Ok(sample)
            };
            // The native owner finalizes each sample before its one session
            // can close; any failure aborts the entire axis, with no partial
            // result. Drop clears its prefix before a different axis starts.
            let samples = constrained_int8::decode_json_int8_sampled_reusing_prefill_with(
                engine, &axis.identity, &axis.prompt, &self.inner.program, &vocabulary.oracle,
                &self.inner.decode, &addresses, axis_budget, max_sampler_bytes, prefill,
                control, &mut finalize)?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(SampledSentimentError::Accounting.into());
            }
            for sample in &samples {
                actual_work = actual_work.checked_add(sample.model_work)?;
                actual_mask = actual_mask.checked_add(sample.mask_node_visit_charge)
                    .ok_or(SampledSentimentError::Accounting)?;
                actual_tokens = actual_tokens.checked_add(sample.generated_tokens as u64)
                    .ok_or(SampledSentimentError::Accounting)?;
            }
            dimensions.push(aggregate(axis.axis, samples)?);
            checkpoint(control)?;
        }
        let run = Int8SampledSentimentRun { schema_version: 1,
            execution: INT8_SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION.to_owned(),
            task_spec: "sentiment-v1".to_owned(), prompt_template_sha256: self.inner.identity.template_digest,
            numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            aggregation: SAMPLED_SENTIMENT_AGGREGATION_VERSION.to_owned(),
            interpretation: "generated_score_variation_not_calibrated_confidence".to_owned(),
            justification_status: "unverified_model_explanations_from_untrusted_document".to_owned(),
            sampler_version: JSON_SAMPLER_VERSION.to_owned(), samples_per_axis: self.inner.options.samples,
            dimensions, model_work: actual_work, generated_tokens: actual_tokens, mask_node_visit_charge: actual_mask };
        let per_token = vocabulary.oracle.trie().node_count().min(budget.json.mask_limits.max_trie_node_visits) as u64;
        self.verify_completed_scheduled_with_control(&run, per_token,
            budget.json.max_total_mask_node_visits, true, control)?;
        Ok(run)
    }
}
