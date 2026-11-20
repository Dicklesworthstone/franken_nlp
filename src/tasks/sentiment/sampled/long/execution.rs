//! One native allocation, complete cold admission, and all-or-error reduction.
use super::*;

impl PreparedSampledSentimentMap {
    /// The complete cold allowance is summed before the first forward. Early
    /// EOS and optional prefix reuse never authorize extra chunks or samples.
    pub fn complete_mask_charge(&self, vocabulary: &SampledSentimentVocabulary,
        budget: Int4JsonBudget, max_sampler_bytes: u64) -> Result<u64, SampledSentimentMapError> {
        let work = self.planned_work();
        if work.forward_positions > budget.native.max_forward_positions
            || work.attention_pairs > budget.native.max_attention_pairs
            || !work.projections.fits(budget.native.max_projection_work)
            || work.forward_positions > budget.json.max_forward_positions
            || work.projected_logits > budget.json.max_projected_logits {
            return Err(SampledSentimentMapError::Work);
        }
        let mut masks = 0_u64;
        for chunk in &self.chunks {
            masks = masks.checked_add(chunk.plan.complete_mask_charge(vocabulary, budget, max_sampler_bytes)?)
                .filter(|&n| n <= budget.json.max_total_mask_node_visits).ok_or(SampledSentimentMapError::Work)?;
        }
        Ok(masks)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_scheduled_with_control<C: DecodeStepControl>(&self, engine: &mut StrictInt4Engine<'_>,
        vocabulary: &SampledSentimentVocabulary, budget: Int4JsonBudget, max_sampler_bytes: u64,
        prefill: Option<Int4PrefillLimits>, reuse_prefill: bool, control: &mut C)
        -> Result<SampledSentimentMapRun, SampledSentimentMapError> {
        checkpoint(control)?;
        if let Some(prefill) = prefill { prefill.validate()?; }
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
            || engine.model_identity().group_width() != self.group_width()
            || engine.kv_cache().capacity_positions() < self.required_context() {
            return Err(SampledSentimentMapError::Admission);
        }
        let planned_masks = self.complete_mask_charge(vocabulary, budget, max_sampler_bytes)?;
        let mut mask_charges = map_reserved(self.chunks.len())?;
        // Check every model binding and every child's native/mask allowance
        // before any child can mutate KV. This pass never re-encodes source.
        for chunk in &self.chunks {
            checkpoint(control)?;
            constrained_int4::check_model(chunk.plan.execution_identity(), engine.model_identity())
                .map_err(SampledSentimentError::from)?;
            mask_charges.push(chunk.plan.complete_mask_charge(vocabulary, budget, max_sampler_bytes)?);
        }
        let mut chunks = map_reserved(self.chunks.len())?;
        let mut model_work = Int4Work::default();
        let mut masks = 0_u64; let mut tokens = 0_u64; let mut retained = 0_u64;
        for (chunk, mask_charge) in self.chunks.iter().zip(mask_charges) {
            checkpoint(control)?;
            let plan = &chunk.plan;
            let work = plan.planned_work();
            let child = Int4JsonBudget { native: Int4RunBudget::exact(work),
                json: crate::native_engine::constrained::JsonWorkBudget {
                    max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                    max_kv_bytes: budget.json.max_kv_bytes, max_total_mask_node_visits: mask_charge,
                    mask_limits: budget.json.mask_limits } };
            let run = if reuse_prefill {
                plan.execute_reusing_prefill_with_control(engine, plan.execution_identity(), vocabulary,
                    child, max_sampler_bytes, prefill, control)
            } else if let Some(prefill) = prefill {
                plan.execute_layer_major_with_control(engine, plan.execution_identity(), vocabulary,
                    child, max_sampler_bytes, prefill, control)
            } else {
                plan.execute_with_control(engine, plan.execution_identity(), vocabulary,
                    child, max_sampler_bytes, control)
            }?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(SampledSentimentMapError::Admission);
            }
            model_work = model_work.checked_add(run.model_work)?;
            masks = masks.checked_add(run.mask_node_visit_charge).filter(|&n| n <= planned_masks)
                .ok_or(SampledSentimentMapError::Work)?;
            tokens = tokens.checked_add(run.generated_tokens).filter(|&n| n <= self.max_output_tokens())
                .ok_or(SampledSentimentMapError::Work)?;
            if !fits_work(model_work, self.planned_work()) { return Err(SampledSentimentMapError::Work); }
            let result = SampledSentimentMapChunk { chunk_id: chunk.id, source_span: chunk.span,
                source_tokens: chunk.tokens, run };
            let remaining = self.limits.max_retained_result_bytes.checked_sub(retained)
                .ok_or(SampledSentimentMapError::Output)?;
            retained = retained.checked_add(binding::value_bytes(&result, remaining, control)?)
                .ok_or(SampledSentimentMapError::Output)?;
            chunks.push(result);
        }
        let aggregate = reduction::aggregate_with_control(&chunks, self.expected.source_span,
            self.expected.samples, control)?;
        let mut result = SampledSentimentMapRun { schema_version: 1,
            execution: execution(reuse_prefill).to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            group_width: self.group_width(), chunk_profile: CHUNK_PROFILE.to_owned(), semantics: SEMANTICS.to_owned(),
            single_context_equivalence: false, source_span: self.expected.source_span,
            chunk_limits: self.expected.chunk_limits, tokenizer_checks: self.expected.tokenizer_checks,
            samples_per_axis: self.expected.samples, planned_model_work: self.planned_work(), model_work,
            planned_mask_node_visit_charge: planned_masks, mask_node_visit_charge: masks, generated_tokens: tokens,
            retained_result_bytes: retained, chunks, aggregate, input_encoding: self.expected.encoding_record(),
            request_binding: self.expected.request_binding, result_binding: [0; 32] };
        result.result_binding = binding::result(&result, self.max_result_bytes(), control)?;
        self.expected.verify_completed_with_control(&result, reuse_prefill, control)?;
        checkpoint(control)?;
        Ok(result)
    }
}

impl SampledSentimentMapPreflight {
    /// Verify both private request ownership and the unchanged native payload.
    /// Deserialized or publicly mutated result fields cannot mint these seals.
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &SampledSentimentMapRun,
        reuse_prefill: bool, control: &mut C) -> Result<(), SampledSentimentMapError> {
        checkpoint(control)?;
        if run.request_binding != self.request_binding
            || run.result_binding != binding::result(run, self.max_result_bytes, control)?
            || run.schema_version != 1 || run.execution != execution(reuse_prefill)
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != self.group_width
            || run.chunk_profile != CHUNK_PROFILE || run.semantics != SEMANTICS || run.single_context_equivalence
            || run.source_span != self.source_span || run.chunk_limits != self.chunk_limits
            || run.tokenizer_checks != self.tokenizer_checks || run.samples_per_axis != self.samples
            || run.chunks.len() != self.chunks || run.planned_model_work != self.work
            || !fits_work(run.model_work, self.work) || run.generated_tokens > self.max_output_tokens
            || run.planned_mask_node_visit_charge == 0 || run.mask_node_visit_charge > run.planned_mask_node_visit_charge
            || run.retained_result_bytes > self.max_retained_bytes || run.input_encoding != self.encoding_record() {
            return Err(SampledSentimentMapError::Admission);
        }
        if binding::geometry_values(run.chunks.iter().map(|chunk|
            (chunk.chunk_id, chunk.source_span, chunk.source_tokens)), control)? != self.geometry {
            return Err(SampledSentimentMapError::Admission);
        }
        let mut work = Int4Work::default(); let mut masks = 0_u64;
        let mut tokens = 0_u64; let mut retained = 0_u64;
        let chunk_execution = if reuse_prefill { SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION } else { SAMPLED_SENTIMENT_EXECUTION };
        for chunk in &run.chunks {
            checkpoint(control)?;
            if chunk.source_tokens == 0
                || (self.encoding.is_none() && chunk.source_tokens != chunk.source_span.byte_end - chunk.source_span.byte_start)
                || chunk.run.execution != chunk_execution || chunk.run.group_width != self.group_width
                || chunk.run.samples_per_axis != self.samples || chunk.run.dimensions.len() != self.axes {
                return Err(SampledSentimentMapError::Admission);
            }
            work = work.checked_add(chunk.run.model_work)?;
            masks = masks.checked_add(chunk.run.mask_node_visit_charge).ok_or(SampledSentimentMapError::Work)?;
            tokens = tokens.checked_add(chunk.run.generated_tokens).ok_or(SampledSentimentMapError::Work)?;
            retained = retained.checked_add(binding::value_bytes(chunk,
                self.max_retained_bytes.checked_sub(retained).ok_or(SampledSentimentMapError::Output)?, control)?)
                .ok_or(SampledSentimentMapError::Output)?;
        }
        if work != run.model_work || masks != run.mask_node_visit_charge || tokens != run.generated_tokens
            || retained != run.retained_result_bytes
            || reduction::aggregate_with_control(&run.chunks, self.source_span, self.samples, control)? != run.aggregate {
            return Err(SampledSentimentMapError::Admission);
        }
        checkpoint(control)
    }
}
fn execution(reuse_prefill: bool) -> &'static str {
    if reuse_prefill { SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION } else { SAMPLED_SENTIMENT_MAP_EXECUTION }
}
