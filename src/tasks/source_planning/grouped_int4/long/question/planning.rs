//! Exact question/scaffold/manifest capacity and complete native admission.
use super::*;

struct Capacity { chunks: ChunkLimits, scaffold: usize, non_source: usize }

impl SourceTaskPlanner {
    /// Weight-free sizing uses actual question, manifest and source encodings.
    /// It is not an execution receipt: the native path still compiles and admits
    /// every real prompt/schema/identity before any forward can begin.
    #[allow(clippy::too_many_arguments)]
    pub fn preflight_int4_question_with_control<C: DecodeStepControl>(&self, source: &str,
        question: &SourceQuestion, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int4SourceMapLimits, group_width: usize, control: &mut C)
        -> Result<Int4QuestionPreflight, Int4SourceMapError> {
        question.validate()?;
        mapping.validate(budget, planning)?;
        let cap = capacity(self, question, budget, context, planning, mapping.chunks, group_width, control)?;
        let chunks = partition(self, source, cap.chunks, control)?;
        let mut expected = metadata(&chunks, question.verification, group_width)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            if !has_text(chunk.text()) { continue; }
            let passages = [AnswerPassage { id: copy(PASSAGE_ID)?, text: copy(chunk.text())? }];
            let answer = AnswerContext::encode(&self.encoder, &question.question, &passages,
                AnswerInputLimits { max_passages: planning.max_passages, max_input_bytes: planning.max_input_bytes,
                    max_input_tokens: budget.max_input_tokens as usize }).map_err(Int4SourceError::from)?;
            let prompt = cap.scaffold.checked_add(answer.document().total_token_count())
                .ok_or(Int4SourceMapError::WorkLimit)?;
            check_prompt(&cap, chunk, prompt, budget)?;
            let work = constrained_int4::planned_work(group_width, prompt, budget.max_output_tokens as usize)
                .map_err(Int4SourceError::from)?;
            charge(&mut expected, work, mapping)?;
        }
        if expected.native_chunks == 0 { return Err(Int4SourceMapError::EmptySource); }
        checkpoint(control)?;
        Ok(expected)
    }

    /// Prepare all nonblank passages on the sealed INT4 source task compiler.
    /// All native identities bind the actual group partition. Whitespace source
    /// remains in the complete partition without inventing a native abstention.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_int4_question_with_control<'s, C: DecodeStepControl>(&self, source: &'s str,
        question: &SourceQuestion, budget: TaskBudget, context: &PlanContext<'_>,
        planning: SourcePlanningLimits, mapping: Int4SourceMapLimits, group_width: usize, control: &mut C)
        -> Result<PreparedInt4Question<'s>, Int4SourceMapError> {
        question.validate()?;
        mapping.validate(budget, planning)?;
        let cap = capacity(self, question, budget, context, planning, mapping.chunks, group_width, control)?;
        let chunks = partition(self, source, cap.chunks, control)?;
        let mut expected = metadata(&chunks, question.verification, group_width)?;
        let mut plans = Vec::new(); let mut indices = Vec::new();
        plans.try_reserve_exact(chunks.chunks().len()).map_err(|_| Int4SourceMapError::Allocation)?;
        indices.try_reserve_exact(chunks.chunks().len()).map_err(|_| Int4SourceMapError::Allocation)?;
        for chunk in chunks.chunks() {
            checkpoint(control)?;
            if !has_text(chunk.text()) { indices.push(None); continue; }
            let request = SourceTaskRequest::Answer { question: copy(&question.question)?,
                passages: vec![AnswerPassage { id: copy(PASSAGE_ID)?, text: copy(chunk.text())? }],
                options: question.options, budget };
            let plan = self.plan_int4_with_control(&request, context, planning, group_width, control)?;
            if plan.group_width() != group_width || plan.execution_identity().task_spec != ANSWER_TASK_VERSION {
                return Err(Int4SourceMapError::Admission);
            }
            check_prompt(&cap, chunk, plan.prompt_tokens(), budget)?;
            charge(&mut expected, plan.planned_work(), mapping)?;
            indices.push(Some(plans.len())); plans.push(plan);
        }
        if expected.native_chunks == 0 { return Err(Int4SourceMapError::EmptySource); }
        checkpoint(control)?;
        Ok(PreparedInt4Question { chunks, plans, plan_indices: indices, mapping,
            options: question.options, verification: question.verification, expected,
            kv_ceiling: budget.max_kv_bytes })
    }
}

#[allow(clippy::too_many_arguments)]
fn capacity<C: DecodeStepControl>(planner: &SourceTaskPlanner, question: &SourceQuestion,
    budget: TaskBudget, context: &PlanContext<'_>, planning: SourcePlanningLimits,
    mut chunks: ChunkLimits, group_width: usize, control: &mut C) -> Result<Capacity, Int4SourceError> {
    checkpoint(control)?;
    constrained_int4::check_profile(context.execution_identity(), group_width)?;
    planner.check_context_for_profile(BuiltInTask::Answer, context, budget, planning,
        NumericsProfile::StrictQuantized { version: 2 })?;
    Int4MemoryRequirement::for_context(planning.max_context_tokens, group_width)
        .map_err(|_| SourcePlanningError::ContextBudget)?;
    let schema = question.options.schema_source()?;
    if schema.len() > planning.compiler.max_schema_bytes || question.question.len() > planning.max_input_bytes
        || question.question.len() > budget.max_input_tokens as usize {
        return Err(SourcePlanningError::InputBudget.into());
    }
    let mut scaffold = 0_usize;
    for fragment in render_fragments(BuiltInTask::Answer, &schema)? {
        checkpoint(control)?;
        let ids = planner.tokenizer.tokenizer().encode_ids_with_options(&fragment,
            EncodeOptions { add_bos: false, add_eos: false })
            .map_err(|_| SourcePlanningError::Contract("INT4 question scaffold encoding"))?;
        scaffold = scaffold.checked_add(ids.len()).ok_or(SourcePlanningError::ContextBudget)?;
    }
    let question_tokens = planner.encoder.encode(&question.question, planning.max_input_bytes,
        budget.max_input_tokens as usize)?.total_token_count();
    // Price the largest decimal coordinate widths in the versioned manifest,
    // not an artificial passage. The byte-preserving encoder needs at most one
    // id per manifest byte; each ACTUAL complete prompt is checked below too.
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
fn manifest_byte_bound(max_bytes: usize) -> Result<usize, Int4SourceError> {
    #[derive(Serialize)]
    struct Extent { id: &'static str, span: VerifiedSourceSpan }
    let bound = Extent { id: PASSAGE_ID, span: VerifiedSourceSpan {
        byte_start: 0, byte_end: max_bytes, scalar_start: 0, scalar_end: max_bytes } };
    Ok(canonjson::canonical_bytes(&(ANSWER_PASSAGE_LAYOUT_VERSION, [bound]))
        .map_err(|_| SourcePlanningError::Serialization)?.len())
}
fn check_prompt(cap: &Capacity, chunk: &SourceChunk<'_>, actual: usize, budget: TaskBudget)
    -> Result<(), Int4SourceMapError> {
    if actual > budget.max_input_tokens as usize
        || cap.non_source.checked_add(chunk.tokens()).is_none_or(|n| actual > n)
        || actual.checked_add(budget.max_output_tokens as usize).is_none_or(|n| n > cap.chunks.context_tokens) {
        return Err(Int4SourceError::from(SourcePlanningError::ContextBudget).into());
    }
    Ok(())
}
fn partition<'s, C: DecodeStepControl>(planner: &SourceTaskPlanner, source: &'s str,
    chunks: ChunkLimits, control: &mut C) -> Result<ChunkPlan<'s>, Int4SourceMapError> {
    if source.is_empty() { return Err(Int4SourceMapError::EmptySource); }
    checkpoint(control)?;
    let mut cancelled = None;
    let plan = ChunkPlan::build_with_checkpoints(source, chunks, |text| {
        planner.encoder.encode(text, chunks.max_chunk_bytes, chunks.max_chunk_bytes)
            .map(|document| document.total_token_count()).map_err(|_| MapReduceError::Tokenizer)
    }, || match control.prefill_checkpoint(0) {
        Some(cause) => { cancelled = Some(cause); Err(MapReduceError::Cancelled) }, None => Ok(()),
    });
    if let Some(cause) = cancelled { return Err(Int4SourceError::Cancelled(cause).into()); }
    plan.map_err(Int4SourceMapError::Chunk)
}
fn metadata(chunks: &ChunkPlan<'_>, verification: GroundingBudget, group_width: usize)
    -> Result<Int4QuestionPreflight, Int4SourceMapError> {
    let first = chunks.chunks().first().ok_or(Int4SourceMapError::EmptySource)?.span();
    let last = chunks.chunks().last().ok_or(Int4SourceMapError::EmptySource)?.span();
    Ok(Int4QuestionPreflight { source_span: VerifiedSourceSpan { byte_start: first.byte_start,
        byte_end: last.byte_end, scalar_start: first.scalar_start, scalar_end: last.scalar_end },
        chunks: chunks.chunks().len(), native_chunks: 0, group_width,
        work: Int4Work::default(), masks: 0, verification })
}
fn charge(expected: &mut Int4QuestionPreflight, work: Int4Work, limits: Int4SourceMapLimits)
    -> Result<(), Int4SourceMapError> {
    expected.work = add_work(expected.work, work).ok_or(Int4SourceMapError::WorkLimit)?;
    expected.masks = expected.masks.checked_add(limits.mask_visits_per_chunk).ok_or(Int4SourceMapError::WorkLimit)?;
    expected.native_chunks = expected.native_chunks.checked_add(1).ok_or(Int4SourceMapError::WorkLimit)?;
    if !fits(expected.work, limits.max_model_work) || expected.masks > limits.max_mask_visits {
        return Err(Int4SourceMapError::WorkLimit);
    }
    Ok(())
}
fn copy(text: &str) -> Result<String, Int4SourceMapError> {
    let mut out = String::new();
    out.try_reserve_exact(text.len()).map_err(|_| Int4SourceMapError::Allocation)?;
    out.push_str(text); Ok(out)
}
