//! Deterministic whole-segment grouping and bounded neural reduction.
use super::*;

pub(super) struct Native<'a, 'w> {
    pub planner: &'a SourceTaskPlanner, pub identity: &'a ExecutionIdentity,
    pub budget: TaskBudget, pub planning: SourcePlanningLimits, pub options: SummaryOptions,
    pub mapping: Int8SourceMapLimits, pub engine: &'a mut StrictInt8Engine<'w>, pub vocabulary: &'a ExtractionVocabulary,
}
// Private seam only. No public caller can inject native receipts or summaries.
pub(super) trait Driver {
    fn count<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<usize, Int8SourceMapError>;
    fn run<C: DecodeStepControl>(&mut self, text: &str, control: &mut C)
        -> Result<(Int8SourceTaskRun, Int8Work), Int8SourceMapError>;
}
impl Driver for Native<'_, '_> {
    fn count<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<usize, Int8SourceMapError> {
        checkpoint(control)?;
        let count = self.planner.source_encoder().encode(text, self.planning.max_input_bytes, self.planning.max_input_bytes)
            .map_err(|_| Int8SourceMapError::WorkLimit)?.total_token_count();
        checkpoint(control)?;
        Ok(count)
    }
    fn run<C: DecodeStepControl>(&mut self, text: &str, control: &mut C)
        -> Result<(Int8SourceTaskRun, Int8Work), Int8SourceMapError> {
        checkpoint(control)?;
        let context = PlanContext::new(self.identity, self.budget).map_err(|_| Int8SourceMapError::Admission)?;
        let request = SourceTaskRequest::Summarize { document: copy(text)?, options: self.options, budget: self.budget };
        let plan = self.planner.plan_int8_with_control(&request, &context, self.planning, control)?;
        // Recheck the actual dynamic plan against its reserved five-axis slot
        // BEFORE inference, in addition to its exact prompt/context compilation.
        let (_, ceiling) = reserve(self.budget, self.planning, self.mapping)?;
        if !within(plan.planned_work(), ceiling) { return Err(Int8SourceMapError::WorkLimit); }
        preflight_plans(std::slice::from_ref(&plan), std::slice::from_ref(plan.execution_identity()), self.engine)?;
        let mut native = NativeDriver { engine: self.engine, vocabulary: self.vocabulary, control, limits: self.mapping };
        let run = native.run(&plan, plan.execution_identity())?;
        check_run(&plan, &run, self.mapping.mask_visits_per_chunk)?;
        extract_int8::check_size(&run, plan.max_result_bytes()).map_err(Int8SourceError::from)?;
        Ok((run, plan.planned_work()))
    }
}
struct Group { segments: Range<usize>, bytes: usize, tokens: usize }
fn groups<C: DecodeStepControl, D: Driver>(frontier: &evidence::Collection, cap: usize,
    limits: SummaryHierarchyLimits, work: &mut HierarchyTokenizerWork, driver: &mut D, control: &mut C)
    -> Result<Vec<Group>, Int8SourceMapError> {
    let mut groups = vector(frontier.segment_count())?;
    let mut start = 0;
    while start < frontier.segment_count() {
        let mut accepted = None;
        for end in start + 1..=frontier.segment_count() {
            checkpoint(control)?;
            let text = frontier.range_text(start..end)?;
            work.calls = work.calls.checked_add(1).filter(|&n| n <= limits.max_tokenizer_calls)
                .ok_or(Int8SourceMapError::WorkLimit)?;
            work.bytes = work.bytes.checked_add(text.len() as u64).filter(|&n| n <= limits.max_tokenizer_bytes)
                .ok_or(Int8SourceMapError::WorkLimit)?;
            // Exact encoding of the actual joined source, not a sum of token
            // counts or a chars/token estimate. Stop at the first overflow;
            // no assumption that BPE counts are monotonic is needed for safety.
            let tokens = driver.count(text, control)?;
            if tokens > cap { break; }
            if tokens == 0 { return Err(invalid()); }
            accepted = Some(Group { segments: start..end, bytes: text.len(), tokens });
        }
        // Atomic quote segments are never split, truncated or silently skipped.
        let group = accepted.ok_or(Int8SourceMapError::WorkLimit)?;
        start = group.segments.end;
        groups.push(group);
    }
    checkpoint(control)?;
    Ok(groups)
}
pub(super) struct Reduced {
    pub status: SummarySynthesisStatus, pub final_pass: Option<usize>,
    pub planned: Int8Work, pub actual: Int8Work, pub masks: u64,
    pub tokenizer: HierarchyTokenizerWork, pub levels: Vec<SummaryHierarchyLevel>, pub passes: Vec<SummaryHierarchyPass>,
}
#[allow(clippy::too_many_arguments)]
pub(super) fn reduce<C: DecodeStepControl, D: Driver>(source: &str, mut frontier: evidence::Collection,
    request: SourceSummarySynthesis, expected: Int8SummaryHierarchyPreflight, max_result_bytes: u64,
    remaining: &mut GroundingBudget, driver: &mut D, control: &mut C) -> Result<Reduced, Int8SourceMapError> {
    let mut out = Reduced { status: SummarySynthesisStatus::NoEvidenceCollected, final_pass: None,
        planned: Int8Work::default(), actual: Int8Work::default(), masks: 0,
        tokenizer: HierarchyTokenizerWork::default(), levels: vector(expected.limits.max_levels)?,
        passes: vector(expected.limits.max_passes)? };
    checkpoint(control)?;
    if frontier.segment_count() == 0 { return Ok(out); }
    for level in 0..expected.limits.max_levels {
        checkpoint(control)?;
        let input = EvidenceSize::of(&frontier);
        let groups = groups(&frontier, expected.capacity.max_source_tokens(), expected.limits, &mut out.tokenizer, driver, control)?;
        let end = out.passes.len().checked_add(groups.len()).filter(|&n| n <= expected.limits.max_passes)
            .ok_or(Int8SourceMapError::WorkLimit)?;
        let terminal = groups.len() == 1;
        // Refuse an impossible nonterminal level BEFORE spending its calls.
        if !terminal && (end == expected.limits.max_passes || level + 1 == expected.limits.max_levels) {
            return Err(Int8SourceMapError::WorkLimit);
        }
        let first = out.passes.len();
        let mut next = evidence::Collection::empty();
        for (group_index, group) in groups.into_iter().enumerate() {
            checkpoint(control)?;
            let window = frontier.window(group.segments.clone(), control)?;
            let (native, planned) = driver.run(&window.text, control)?;
            if !within(planned, expected.discovery.synthesis_reserve) || !within(native.model_work, planned) {
                return Err(Int8SourceMapError::WorkLimit);
            }
            let SourceTaskResult::Summarize(raw) = &native.result else { return Err(invalid()); };
            let bullets = evidence::lift_summary(source, raw, &window, request.synthesis_options, remaining, control)?;
            let masks = mask_charge(&native.result)?;
            if masks > expected.discovery.synthesis_masks { return Err(Int8SourceMapError::WorkLimit); }
            let pass = SummaryHierarchyPass { level, group: group_index, input_segments: group.segments,
                input_bytes: group.bytes, input_tokens: group.tokens, planned_model_work: planned, native, bullets };
            receipt::verify_pass(&expected, &pass)?;
            out.planned = add_work(out.planned, planned).ok_or(Int8SourceMapError::WorkLimit)?;
            out.actual = add_work(out.actual, pass.native.model_work).ok_or(Int8SourceMapError::WorkLimit)?;
            out.masks = out.masks.checked_add(masks).ok_or(Int8SourceMapError::WorkLimit)?;
            // The only inter-level transport is the complete set of lifted
            // citations selected by this group. Never include bullet prose.
            if !terminal { next.append_verified(source, &pass.bullets, request.synthesis_options, request.limits, remaining, control)?; }
            out.passes.push(pass);
            extract_int8::check_size(&out.passes, max_result_bytes).map_err(Int8SourceError::from)?;
            checkpoint(control)?;
        }
        let next_size = EvidenceSize::of(&next);
        out.levels.push(SummaryHierarchyLevel { input, passes: first..end,
            next_evidence: if terminal { None } else { Some(next_size) } });
        if terminal {
            out.final_pass = Some(first);
            out.status = if out.passes[first].bullets.is_empty() { SummarySynthesisStatus::NoBulletsProduced }
                else { SummarySynthesisStatus::Synthesized };
            checkpoint(control)?;
            return Ok(out);
        }
        if next_size.segments == 0 {
            out.status = SummarySynthesisStatus::NoBulletsProduced;
            checkpoint(control)?;
            return Ok(out);
        }
        // A model may return unchanged/full-length quotes. Refuse rather than
        // looping, renewing budgets or pretending the reduction succeeded.
        if next_size.bytes >= input.bytes { return Err(Int8SourceMapError::WorkLimit); }
        frontier = next;
    }
    Err(Int8SourceMapError::WorkLimit)
}
