//! Native complete admission, exact per-chunk slices and sealed publication.
use super::*;

impl PreparedInt8SampledSentimentMap {
    pub fn complete_mask_charge(&self, vocabulary: &SampledSentimentVocabulary,
        budget: Int8JsonBudget, max_sampler_bytes: u64) -> Result<u64, Int8SampledSentimentMapError> {
        check_work_budget(self.planned_work(), budget)?;
        let mut masks = 0_u64;
        for chunk in &self.chunks {
            masks = masks.checked_add(chunk.plan.complete_mask_charge(vocabulary, budget, max_sampler_bytes)?)
                .filter(|&n| n <= budget.json.max_total_mask_node_visits).ok_or(Int8SampledSentimentMapError::Work)?;
        }
        Ok(masks)
    }

    /// The host owns one native allocation and optional persistent projection
    /// team throughout all independent chunk/axis/sample sessions.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_scheduled_with_control<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>,
        vocabulary: &SampledSentimentVocabulary, budget: Int8JsonBudget, max_sampler_bytes: u64,
        prefill: Option<Int8PrefillLimits>, control: &mut C)
        -> Result<Int8SampledSentimentMapRun, Int8SampledSentimentMapError> {
        checkpoint(control)?;
        if let Some(prefill) = prefill { prefill.validate()?; }
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
            || engine.profile() != STRICT_INT8_PROFILE
            || engine.kv_cache().capacity_positions() < self.required_context() {
            return Err(Int8SampledSentimentMapError::Admission);
        }
        let resident_kv = (engine.kv_cache().capacity_positions() as u64)
            .checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(Int8SampledSentimentMapError::Work)?;
        if resident_kv > budget.json.max_kv_bytes { return Err(Int8SampledSentimentMapError::Work); }
        self.complete_mask_charge(vocabulary, budget, max_sampler_bytes)?;
        // Check all model bindings before any native session can mutate KV.
        for chunk in &self.chunks {
            checkpoint(control)?;
            check_model(chunk.plan.execution_identity(), engine.artifact_identity())?;
        }
        let per_token = vocabulary.oracle.trie().node_count().min(budget.json.mask_limits.max_trie_node_visits) as u64;
        self.execute_chunks(per_token, budget, max_sampler_bytes, control, |plan, child, control| {
            let run = match prefill {
                Some(limits) => plan.execute_layer_major_with_control(engine, plan.execution_identity(),
                    vocabulary, child, max_sampler_bytes, limits, control),
                None => plan.execute_with_control(engine, plan.execution_identity(),
                    vocabulary, child, max_sampler_bytes, control),
            }?;
            if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                return Err(Int8SampledSentimentMapError::Admission);
            }
            Ok(run)
        })
    }

    // A private seam tests the actual completion/admission boundary without
    // exposing any public callback that could mint synthetic native results.
    pub(super) fn execute_chunks<C: DecodeStepControl, F>(&self, per_token: u64,
        budget: Int8JsonBudget, max_sampler_bytes: u64, control: &mut C, mut evaluate: F)
        -> Result<Int8SampledSentimentMapRun, Int8SampledSentimentMapError>
    where F: FnMut(&PreparedInt8SampledSentiment, Int8JsonBudget, &mut C)
        -> Result<Int8SampledSentimentRun, Int8SampledSentimentMapError> {
        checkpoint(control)?;
        check_work_budget(self.planned_work(), budget)?;
        let mut charges = map_reserved(self.chunk_count())?;
        let mut planned_masks = 0_u64;
        for chunk in &self.chunks {
            checkpoint(control)?;
            let one = chunk.plan.preflight_bounds(per_token, budget, max_sampler_bytes)?;
            let all = one.checked_mul(chunk.plan.sample_count() as u64).ok_or(Int8SampledSentimentMapError::Work)?;
            planned_masks = planned_masks.checked_add(all)
                .filter(|&n| n <= budget.json.max_total_mask_node_visits).ok_or(Int8SampledSentimentMapError::Work)?;
            charges.push(all);
        }
        let mut chunks = map_reserved(self.chunk_count())?;
        let mut model_work = Int8Work::default();
        let (mut masks, mut tokens, mut retained) = (0_u64, 0_u64, 0_u64);
        for (chunk, mask_charge) in self.chunks.iter().zip(charges) {
            checkpoint(control)?;
            let work = chunk.plan.planned_work();
            let child = Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
                max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                max_kv_bytes: budget.json.max_kv_bytes, max_total_mask_node_visits: mask_charge,
                mask_limits: budget.json.mask_limits } };
            let evaluated = evaluate(&chunk.plan, child, control);
            // An already observed terminal cause wins over a later callback
            // failure; otherwise checkpoint before accepting either outcome.
            let run = match evaluated {
                Err(error) if error.cancellation().is_some() => return Err(error),
                other => { checkpoint(control)?; other? },
            };
            chunk.plan.verify_completed_with_control(&run, per_token, mask_charge, control)?;
            model_work = model_work.checked_add(run.model_work)?;
            masks = masks.checked_add(run.mask_node_visit_charge).filter(|&n| n <= planned_masks)
                .ok_or(Int8SampledSentimentMapError::Work)?;
            tokens = tokens.checked_add(run.generated_tokens).filter(|&n| n <= self.max_output_tokens())
                .ok_or(Int8SampledSentimentMapError::Work)?;
            if !fits_work(model_work, self.planned_work()) { return Err(Int8SampledSentimentMapError::Work); }
            let row = Int8SampledSentimentMapChunk { chunk_id: chunk.id, source_span: chunk.span,
                source_tokens: chunk.tokens, run };
            let remaining = self.limits.max_retained_result_bytes.checked_sub(retained)
                .ok_or(Int8SampledSentimentMapError::Output)?;
            retained = retained.checked_add(binding::value_bytes(&row, remaining, control)?)
                .ok_or(Int8SampledSentimentMapError::Output)?;
            chunks.push(row);
        }
        let aggregate = reduction::aggregate_chunks_with_control(&chunks, self.expected.source_span,
            self.expected.samples, control)?;
        let mut run = Int8SampledSentimentMapRun { schema_version: 1,
            execution: INT8_SAMPLED_SENTIMENT_MAP_EXECUTION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            chunk_profile: CHUNK_PROFILE.to_owned(), semantics: SEMANTICS.to_owned(),
            single_context_equivalence: false, source_span: self.expected.source_span,
            chunk_limits: self.expected.chunk_limits, tokenizer_checks: self.expected.tokenizer_checks,
            samples_per_axis: self.expected.samples, planned_model_work: self.planned_work(), model_work,
            planned_mask_node_visit_charge: planned_masks, mask_node_visit_charge: masks, generated_tokens: tokens,
            retained_result_bytes: retained, chunks, aggregate, input_encoding: self.expected.encoding_record(),
            request_binding: self.expected.request_binding, result_binding: [0; 32] };
        run.result_binding = binding::result(&run, self.max_result_bytes(), control)?;
        self.expected.verify_completed_with_control(&run, control)?;
        checkpoint(control)?;
        Ok(run)
    }
}

impl Int8SampledSentimentMapPreflight {
    pub fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int8SampledSentimentMapRun,
        control: &mut C) -> Result<(), Int8SampledSentimentMapError> {
        checkpoint(control)?;
        if run.request_binding != self.request_binding
            || run.result_binding != binding::result(run, self.max_result_bytes, control)?
            || run.schema_version != 1 || run.execution != INT8_SAMPLED_SENTIMENT_MAP_EXECUTION
            || run.numerics_profile != STRICT_INT8_PROFILE || run.chunk_profile != CHUNK_PROFILE
            || run.semantics != SEMANTICS || run.single_context_equivalence
            || run.source_span != self.source_span || run.chunk_limits != self.chunk_limits
            || run.tokenizer_checks != self.tokenizer_checks || run.samples_per_axis != self.samples
            || run.chunks.len() != self.chunks || run.planned_model_work != self.work
            || !fits_work(run.model_work, self.work) || run.generated_tokens > self.max_output_tokens
            || run.planned_mask_node_visit_charge == 0 || run.mask_node_visit_charge > run.planned_mask_node_visit_charge
            || run.retained_result_bytes > self.max_retained_bytes || run.input_encoding != self.encoding_record() {
            return Err(Int8SampledSentimentMapError::Admission);
        }
        if binding::geometry(run.chunks.iter().map(|chunk|
            (chunk.chunk_id, chunk.source_span, chunk.source_tokens)), control)? != self.geometry {
            return Err(Int8SampledSentimentMapError::Admission);
        }
        let mut work = Int8Work::default();
        let (mut masks, mut tokens, mut retained) = (0_u64, 0_u64, 0_u64);
        for chunk in &run.chunks {
            checkpoint(control)?;
            if chunk.source_tokens == 0
                || (self.encoding.is_none() && chunk.source_tokens != chunk.source_span.byte_end - chunk.source_span.byte_start)
                || chunk.run.execution != INT8_SAMPLED_SENTIMENT_EXECUTION
                || chunk.run.samples_per_axis != self.samples || chunk.run.dimensions.len() != self.axes {
                return Err(Int8SampledSentimentMapError::Admission);
            }
            work = work.checked_add(chunk.run.model_work)?;
            masks = masks.checked_add(chunk.run.mask_node_visit_charge).ok_or(Int8SampledSentimentMapError::Work)?;
            tokens = tokens.checked_add(chunk.run.generated_tokens).ok_or(Int8SampledSentimentMapError::Work)?;
            retained = retained.checked_add(binding::value_bytes(chunk,
                self.max_retained_bytes.checked_sub(retained).ok_or(Int8SampledSentimentMapError::Output)?, control)?)
                .ok_or(Int8SampledSentimentMapError::Output)?;
        }
        if work != run.model_work || masks != run.mask_node_visit_charge || tokens != run.generated_tokens
            || retained != run.retained_result_bytes
            || reduction::aggregate_chunks_with_control(&run.chunks, self.source_span, self.samples, control)? != run.aggregate {
            return Err(Int8SampledSentimentMapError::Admission);
        }
        checkpoint(control)
    }
}

fn check_work_budget(work: Int8Work, budget: Int8JsonBudget) -> Result<(), Int8SampledSentimentMapError> {
    if work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work)
        || work.forward_positions > budget.json.max_forward_positions
        || work.projected_logits > budget.json.max_projected_logits {
        return Err(Int8SampledSentimentMapError::Work);
    }
    Ok(())
}

impl reduction::SampledMapChunk for Int8SampledSentimentMapChunk {
    type Work = Int8Work;
    fn id(&self) -> usize { self.chunk_id }
    fn span(&self) -> VerifiedSourceSpan { self.source_span }
    fn tokens(&self) -> usize { self.source_tokens }
    fn dimensions(&self) -> &[Int8SampledSentimentDimension] { &self.run.dimensions }
    fn compatible(&self, reference: &Self) -> bool {
        self.run.execution == reference.run.execution
            && self.run.prompt_template_sha256 == reference.run.prompt_template_sha256
            && self.run.dimensions.len() == reference.run.dimensions.len()
    }
    fn validate(&self, samples: usize, control: &mut impl DecodeStepControl)
        -> Result<(), SampledSentimentMapError> {
        let run = &self.run;
        if run.schema_version != 1 || run.task_spec != "sentiment-v1"
            || run.execution != INT8_SAMPLED_SENTIMENT_EXECUTION || run.numerics_profile != STRICT_INT8_PROFILE
            || run.aggregation != SAMPLED_SENTIMENT_AGGREGATION_VERSION
            || run.interpretation != "generated_score_variation_not_calibrated_confidence"
            || run.justification_status != "unverified_model_explanations_from_untrusted_document"
            || run.sampler_version != JSON_SAMPLER_VERSION || run.samples_per_axis != samples
            || run.dimensions.is_empty() || run.dimensions.len() > SentimentAxis::ALL.len()
            || run.dimensions.windows(2).any(|pair| pair[0].axis >= pair[1].axis) {
            return Err(SampledSentimentMapError::Aggregation);
        }
        let mut work = Int8Work::default(); let (mut tokens, mut masks) = (0_u64, 0_u64);
        for dimension in &run.dimensions {
            if let Some(cause) = control.prefill_checkpoint(0) { return Err(SampledSentimentMapError::Cancelled(cause)); }
            reduction::validate_dimension(dimension, samples)?;
            for (index, sample) in dimension.samples.iter().enumerate() {
                if let Some(cause) = control.prefill_checkpoint(0) { return Err(SampledSentimentMapError::Cancelled(cause)); }
                if sample.sample_index != index || !(-100..=100).contains(&sample.score)
                    || sample.justification.len() > 4 * 4096 || sample.justification.trim().is_empty()
                    || sample.justification.chars().count() > 4096
                    || !(1..=1024).contains(&sample.generated_tokens) || sample.mask_node_visit_charge == 0
                    || sample.model_work.forward_positions < sample.generated_tokens as u64
                    || sample.model_work.projected_logits != sample.generated_tokens as u64 * NANBEIGE_VOCAB_SIZE as u64 {
                    return Err(SampledSentimentMapError::Aggregation);
                }
                work = work.checked_add(sample.model_work).map_err(|_| SampledSentimentMapError::Aggregation)?;
                tokens = tokens.checked_add(sample.generated_tokens as u64).ok_or(SampledSentimentMapError::Aggregation)?;
                masks = masks.checked_add(sample.mask_node_visit_charge).ok_or(SampledSentimentMapError::Aggregation)?;
            }
        }
        if work != run.model_work || tokens != run.generated_tokens || masks != run.mask_node_visit_charge {
            return Err(SampledSentimentMapError::Aggregation);
        }
        Ok(())
    }
}
