//! Exact question-aware sizing and complete pre-model admission.
use super::*;

struct Capacity { chunks: ChunkLimits, scaffold: usize, non_source: usize }

impl SourceTaskPlanner {
    /// Read-only sizing before loading weights. Uses real source/question ids
    /// and the real canonical passage manifest, not a chars-per-token estimate.
    /// Grammar compilation and every actual identity are still checked later.
    #[allow(clippy::too_many_arguments)]
    pub fn preflight_int8_question_with_control<C: DecodeStepControl>(&self, source: &str,
        question: &SourceQuestion, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int8SourceMapLimits, control: &mut C)
        -> Result<Int8QuestionPreflight, Int8SourceMapError> {
        question.validate()?;
        mapping.validate(budget, planning)?;
        let cap = capacity(self, question, budget, context, planning, mapping.chunks, control)?;
        let chunks = partition(self, source, cap.chunks, control)?;
        let mut expected = metadata(&chunks, question.verification)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            if !has_text(chunk.text()) { continue; }
            let passages = [AnswerPassage { id: copy(PASSAGE_ID)?, text: copy(chunk.text())? }];
            let answer = AnswerContext::encode(&self.encoder, &question.question, &passages,
                AnswerInputLimits { max_passages: planning.max_passages, max_input_bytes: planning.max_input_bytes,
                    max_input_tokens: budget.max_input_tokens as usize }).map_err(Int8SourceError::from)?;
            let prompt = cap.scaffold.checked_add(answer.document().total_token_count())
                .ok_or(Int8SourceMapError::WorkLimit)?;
            check_prompt(&cap, chunk, prompt, budget)?;
            let work = constrained_int8::planned_work(prompt, budget.max_output_tokens as usize)
                .map_err(Int8SourceError::from)?;
            charge(&mut expected, work, mapping)?;
        }
        if expected.native_chunks == 0 { return Err(Int8SourceMapError::EmptySource); }
        checkpoint(control)?;
        Ok(expected)
    }

    /// Prepare every nonblank passage on the existing native answer compiler.
    /// Blank ranges are explicit metadata, never synthesized native abstentions.
    /// Source and all prepared grammars must be covered by the caller's guard.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int8_question_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        question: &SourceQuestion, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int8SourceMapLimits, control: &mut C)
        -> Result<PreparedInt8Question<'s>, Int8SourceMapError> {
        question.validate()?;
        mapping.validate(budget, planning)?;
        let cap = capacity(self, question, budget, context, planning, mapping.chunks, control)?;
        let chunks = partition(self, source, cap.chunks, control)?;
        let mut expected = metadata(&chunks, question.verification)?;
        let mut plans = Vec::new(); let mut indices = Vec::new();
        plans.try_reserve_exact(chunks.chunks().len()).map_err(|_| Int8SourceMapError::Allocation)?;
        indices.try_reserve_exact(chunks.chunks().len()).map_err(|_| Int8SourceMapError::Allocation)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            if !has_text(chunk.text()) { indices.push(None); continue; }
            let request = SourceTaskRequest::Answer { question: copy(&question.question)?,
                passages: vec![AnswerPassage { id: copy(PASSAGE_ID)?, text: copy(chunk.text())? }],
                options: question.options, budget };
            let plan = self.plan_int8_with_control(&request, context, planning, control)?;
            check_prompt(&cap, chunk, plan.prompt_tokens(), budget)?;
            charge(&mut expected, plan.planned_work(), mapping)?;
            indices.push(Some(plans.len())); plans.push(plan);
        }
        if expected.native_chunks == 0 { return Err(Int8SourceMapError::EmptySource); }
        checkpoint(control)?;
        Ok(PreparedInt8Question { chunks, plans, plan_indices: indices, mapping,
            options: question.options, verification: question.verification, expected })
    }
}

fn capacity<C: DecodeStepControl>(planner: &SourceTaskPlanner, question: &SourceQuestion,
    budget: TaskBudget, context: &PlanContext<'_>, planning: SourcePlanningLimits,
    mut chunks: ChunkLimits, control: &mut C) -> Result<Capacity, Int8SourceError> {
    checkpoint(control)?;
    constrained_int8::check_profile(context.execution_identity())?;
    planner.check_context_for_profile(BuiltInTask::Answer, context, budget, planning,
        NumericsProfile::StrictQuantized { version: 1 })?;
    let schema = question.options.schema_source()?;
    if schema.len() > planning.compiler.max_schema_bytes
        || question.question.len() > planning.max_input_bytes
        || question.question.len() > budget.max_input_tokens as usize {
        return Err(SourcePlanningError::InputBudget.into());
    }
    let mut scaffold = 0_usize;
    for fragment in render_fragments(BuiltInTask::Answer, &schema)? {
        checkpoint(control)?;
        let ids = planner.tokenizer.tokenizer().encode_ids_with_options(&fragment,
            EncodeOptions { add_bos: false, add_eos: false })
            .map_err(|_| SourcePlanningError::Contract("question map scaffold encoding"))?;
        scaffold = scaffold.checked_add(ids.len()).ok_or(SourcePlanningError::ContextBudget)?;
    }
    let question_tokens = planner.encoder.encode(&question.question, planning.max_input_bytes,
        budget.max_input_tokens as usize)?.total_token_count();
    // Coordinates grow with the chunk. Serialize their largest decimal widths
    // rather than pricing a placeholder passage. The byte-preserving encoder
    // needs at most one id per manifest byte. Every ACTUAL complete prompt is
    // independently checked below, so layout/tokenizer drift fails pre-model.
    let manifest = manifest_byte_bound(chunks.max_chunk_bytes)?;
    let non_source = scaffold.checked_add(question_tokens).and_then(|n| n.checked_add(manifest))
        .ok_or(SourcePlanningError::ContextBudget)?;
    let reserve = non_source.checked_add(budget.max_output_tokens as usize)
        .ok_or(SourcePlanningError::ContextBudget)?;
    let prompt_room = (budget.max_input_tokens as usize).checked_sub(non_source)
        .ok_or(SourcePlanningError::ContextBudget)?;
    let byte_room = planning.max_input_bytes.min(budget.max_input_tokens as usize)
        .checked_sub(question.question.len()).and_then(|n| n.checked_sub(manifest))
        .ok_or(SourcePlanningError::InputBudget)?;
    if scaffold == 0 || byte_room < 4 || prompt_room == 0 { return Err(SourcePlanningError::ContextBudget.into()); }
    chunks.context_tokens = chunks.context_tokens.min(planning.max_context_tokens);
    chunks.reserved_tokens = chunks.reserved_tokens.max(reserve);
    chunks.max_chunk_tokens = chunks.max_chunk_tokens.min(prompt_room);
    chunks.max_chunk_bytes = chunks.max_chunk_bytes.min(byte_room);
    chunks.effective_token_limit().map_err(|_| SourcePlanningError::ContextBudget)?;
    checkpoint(control)?;
    Ok(Capacity { chunks, scaffold, non_source })
}

fn manifest_byte_bound(max_bytes: usize) -> Result<usize, Int8SourceError> {
    #[derive(Serialize)]
    struct Extent { id: &'static str, span: VerifiedSourceSpan }
    // This is a size bound for the versioned manifest wire layout, not a
    // fabricated document, task plan, source citation or native result.
    let bound = Extent { id: PASSAGE_ID, span: VerifiedSourceSpan {
        byte_start: 0, byte_end: max_bytes, scalar_start: 0, scalar_end: max_bytes } };
    Ok(canonjson::canonical_bytes(&(ANSWER_PASSAGE_LAYOUT_VERSION, [bound]))
        .map_err(|_| SourcePlanningError::Serialization)?.len())
}
fn check_prompt(cap: &Capacity, chunk: &SourceChunk<'_>, actual: usize, budget: TaskBudget)
    -> Result<(), Int8SourceMapError> {
    if actual > budget.max_input_tokens as usize
        || cap.non_source.checked_add(chunk.tokens()).is_none_or(|n| actual > n)
        || actual.checked_add(budget.max_output_tokens as usize).is_none_or(|n| n > cap.chunks.context_tokens) {
        return Err(Int8SourceError::from(SourcePlanningError::ContextBudget).into());
    }
    Ok(())
}
fn partition<'s, C: DecodeStepControl>(planner: &SourceTaskPlanner, source: &'s str,
    chunks: ChunkLimits, control: &mut C) -> Result<ChunkPlan<'s>, Int8SourceMapError> {
    if source.is_empty() { return Err(Int8SourceMapError::EmptySource); }
    checkpoint(control)?;
    let mut cancelled = None;
    let plan = ChunkPlan::build_with_checkpoints(source, chunks, |text| {
        planner.encoder.encode(text, chunks.max_chunk_bytes, chunks.max_chunk_bytes)
            .map(|d| d.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
    }, || match control.prefill_checkpoint(0) {
        Some(cause) => { cancelled = Some(cause); Err(MapReduceError::Cancelled) }, None => Ok(()),
    });
    if let Some(cause) = cancelled { return Err(Int8SourceError::Cancelled(cause).into()); }
    plan.map_err(Int8SourceMapError::Chunk)
}
fn metadata(chunks: &ChunkPlan<'_>, verification: GroundingBudget) -> Result<Int8QuestionPreflight, Int8SourceMapError> {
    let first = chunks.chunks().first().ok_or(Int8SourceMapError::EmptySource)?.span();
    let last = chunks.chunks().last().ok_or(Int8SourceMapError::EmptySource)?.span();
    Ok(Int8QuestionPreflight { source_span: VerifiedSourceSpan { byte_start: first.byte_start,
        byte_end: last.byte_end, scalar_start: first.scalar_start, scalar_end: last.scalar_end },
        chunks: chunks.chunks().len(), native_chunks: 0, work: Int8Work::default(), masks: 0, verification })
}
fn charge(expected: &mut Int8QuestionPreflight, work: Int8Work, limits: Int8SourceMapLimits)
    -> Result<(), Int8SourceMapError> {
    expected.work = add_work(expected.work, work).ok_or(Int8SourceMapError::WorkLimit)?;
    expected.masks = expected.masks.checked_add(limits.mask_visits_per_chunk).ok_or(Int8SourceMapError::WorkLimit)?;
    expected.native_chunks += 1;
    if !within(expected.work, limits.max_model_work) || expected.masks > limits.max_mask_visits {
        return Err(Int8SourceMapError::WorkLimit);
    }
    Ok(())
}
fn copy(text: &str) -> Result<String, Int8SourceMapError> {
    let mut out = String::new();
    out.try_reserve_exact(text.len()).map_err(|_| Int8SourceMapError::Allocation)?;
    out.push_str(text); Ok(out)
}
