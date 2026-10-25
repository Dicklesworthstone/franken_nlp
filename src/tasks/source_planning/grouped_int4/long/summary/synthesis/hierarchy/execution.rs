//! Exact whole-segment grouping and complete-level native admission.
use super::*;

pub(super) struct Native<'a, 'w> {
    pub planner: &'a SourceTaskPlanner, pub identity: &'a ExecutionIdentity,
    pub budget: TaskBudget, pub planning: SourcePlanningLimits, pub options: SummaryOptions,
    pub mapping: Int4SourceMapLimits, pub engine: &'a mut StrictInt4Engine<'w>, pub vocabulary: &'a ExtractionVocabulary,
    pub group_width: usize,
}
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct Witness { identity: Sha256Digest, prompt: usize, work: Int4Work }
impl Witness {
    fn of(plan: &PreparedInt4SourceTask) -> Result<Self, Int4SourceMapError> {
        let bytes = plan.execution_identity().canonical_json_bytes().map_err(|_| Int4SourceMapError::Admission)?;
        Ok(Self { identity: Sha256Digest::of_bytes(&bytes), prompt: plan.prompt_tokens(), work: plan.planned_work() })
    }
}
// Private fixture seam; public execution always uses Native.
pub(super) trait Driver {
    type Witness;
    fn count<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<usize, Int4SourceMapError>;
    fn prepare<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<Self::Witness, Int4SourceMapError>;
    fn run<C: DecodeStepControl>(&mut self, text: &str, witness: &Self::Witness, control: &mut C)
        -> Result<(Int4SourceTaskRun, Int4Work), Int4SourceMapError>;
}
impl Native<'_, '_> {
    fn plan<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<PreparedInt4SourceTask, Int4SourceMapError> {
        checkpoint(control)?;
        let context = PlanContext::new(self.identity, self.budget).map_err(|_| Int4SourceMapError::Admission)?;
        let request = SourceTaskRequest::Summarize { document: copy(text)?, options: self.options, budget: self.budget };
        let plan = self.planner.plan_int4_with_control(&request, &context, self.planning, self.group_width, control)?;
        let (_, ceiling) = reserve(self.budget, self.planning, self.mapping, self.group_width)?;
        if !fits(plan.planned_work(), ceiling) { return Err(Int4SourceMapError::WorkLimit); }
        let driver = NativeDriver { engine: self.engine, vocabulary: self.vocabulary, control, limits: self.mapping };
        preflight(std::slice::from_ref(&plan), std::slice::from_ref(plan.execution_identity()), &driver)?;
        Ok(plan)
    }
}
impl Driver for Native<'_, '_> {
    type Witness = Witness;
    fn count<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<usize, Int4SourceMapError> {
        checkpoint(control)?;
        let count = self.planner.source_encoder().encode(text, self.planning.max_input_bytes, self.planning.max_input_bytes)
            .map_err(|_| Int4SourceMapError::WorkLimit)?.total_token_count();
        checkpoint(control)?;
        Ok(count)
    }
    fn prepare<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<Witness, Int4SourceMapError> {
        Witness::of(&self.plan(text, control)?)
    }
    fn run<C: DecodeStepControl>(&mut self, text: &str, witness: &Witness, control: &mut C)
        -> Result<(Int4SourceTaskRun, Int4Work), Int4SourceMapError> {
        let plan = self.plan(text, control)?;
        if Witness::of(&plan)? != *witness { return Err(Int4SourceMapError::Admission); }
        let mut driver = NativeDriver { engine: self.engine, vocabulary: self.vocabulary, control, limits: self.mapping };
        let run = driver.run(&plan, plan.execution_identity())?;
        check_run(&plan, &run, self.mapping.mask_visits_per_chunk)?;
        Ok((run, plan.planned_work()))
    }
}
struct Group { segments: Range<usize>, bytes: usize, tokens: usize }
fn groups<C: DecodeStepControl, D: Driver>(frontier: &evidence::Collection, cap: usize,
    limits: SummaryHierarchyLimits, work: &mut HierarchyTokenizerWork, driver: &mut D, control: &mut C)
    -> Result<Vec<Group>, Int4SourceMapError> {
    let mut groups = vector(frontier.segment_count())?;
    let mut start = 0;
    while start < frontier.segment_count() {
        let mut accepted = None;
        for end in start + 1..=frontier.segment_count() {
            checkpoint(control)?;
            let text = frontier.range_text(start..end)?;
            work.calls = work.calls.checked_add(1).filter(|&n| n <= limits.max_tokenizer_calls).ok_or(Int4SourceMapError::WorkLimit)?;
            work.bytes = work.bytes.checked_add(text.len() as u64).filter(|&n| n <= limits.max_tokenizer_bytes)
                .ok_or(Int4SourceMapError::WorkLimit)?;
            // Count the actual joined bytes. Stopping at first overflow does
            // not assume that BPE token counts are monotonic or additive.
            let tokens = driver.count(text, control)?;
            if tokens > cap { break; }
            if tokens == 0 { return Err(invalid()); }
            accepted = Some(Group { segments: start..end, bytes: text.len(), tokens });
        }
        // Atomic quotes are never split, truncated, or silently skipped.
        let group = accepted.ok_or(Int4SourceMapError::WorkLimit)?;
        start = group.segments.end; groups.push(group);
    }
    checkpoint(control)?;
    Ok(groups)
}
pub(super) struct Reduced {
    pub status: SummarySynthesisStatus, pub final_pass: Option<usize>,
    pub planned: Int4Work, pub actual: Int4Work, pub masks: u64,
    pub tokenizer: HierarchyTokenizerWork, pub levels: Vec<SummaryHierarchyLevel>, pub passes: Vec<SummaryHierarchyPass>,
}
#[allow(clippy::too_many_arguments)]
pub(super) fn reduce<C: DecodeStepControl, D: Driver>(source: &str, mut frontier: evidence::Collection,
    request: SourceSummarySynthesis, expected: Int4SummaryHierarchyPreflight, max_result_bytes: u64,
    remaining: &mut GroundingBudget, driver: &mut D, control: &mut C) -> Result<Reduced, Int4SourceMapError> {
    let mut out = Reduced { status: SummarySynthesisStatus::NoEvidenceCollected, final_pass: None,
        planned: Int4Work::default(), actual: Int4Work::default(), masks: 0,
        tokenizer: HierarchyTokenizerWork::default(), levels: vector(expected.limits.max_levels)?, passes: vector(expected.limits.max_passes)? };
    checkpoint(control)?;
    if frontier.segment_count() == 0 { return Ok(out); }
    for level in 0..expected.limits.max_levels {
        checkpoint(control)?;
        let input = EvidenceSize::of(&frontier);
        let groups = groups(&frontier, expected.capacity.max_source_tokens(), expected.limits, &mut out.tokenizer, driver, control)?;
        let end = out.passes.len().checked_add(groups.len()).filter(|&n| n <= expected.limits.max_passes)
            .ok_or(Int4SourceMapError::WorkLimit)?;
        let terminal = groups.len() == 1;
        if !terminal && (end == expected.limits.max_passes || level + 1 == expected.limits.max_levels) {
            return Err(Int4SourceMapError::WorkLimit);
        }
        // Compile and authenticate EVERY actual group before this level's first
        // native call. Compact witnesses retain no source text or identity strings.
        // Rebuilding one plan at a time keeps only one grammar/source owner live.
        let mut witnesses = vector(groups.len())?;
        for group in &groups {
            checkpoint(control)?;
            witnesses.push(driver.prepare(frontier.range_text(group.segments.clone())?, control)?);
        }
        let first = out.passes.len();
        let mut next = evidence::Collection::empty();
        for (group_index, (group, witness)) in groups.into_iter().zip(witnesses).enumerate() {
            checkpoint(control)?;
            let window = frontier.window(group.segments.clone(), control)?;
            let (native, planned) = driver.run(&window.text, &witness, control)?;
            if !fits(planned, expected.discovery.synthesis_reserve) || !fits(native.model_work, planned) {
                return Err(Int4SourceMapError::WorkLimit);
            }
            let SourceTaskResult::Summarize(raw) = &native.result else { return Err(invalid()); };
            let bullets = evidence::lift_summary(source, raw, &window, request.synthesis_options, remaining, control)?;
            let masks = mask_charge(&native.result);
            if masks > expected.discovery.synthesis_masks { return Err(Int4SourceMapError::WorkLimit); }
            let pass = SummaryHierarchyPass { level, group: group_index, input_segments: group.segments,
                input_bytes: group.bytes, input_tokens: group.tokens, planned_model_work: planned, native, bullets };
            receipt::verify_pass(&expected, &pass)?;
            out.planned = add_work(out.planned, planned).ok_or(Int4SourceMapError::WorkLimit)?;
            out.actual = add_work(out.actual, pass.native.model_work).ok_or(Int4SourceMapError::WorkLimit)?;
            out.masks = out.masks.checked_add(masks).ok_or(Int4SourceMapError::WorkLimit)?;
            if !terminal { next.append_verified(source, &pass.bullets, request.synthesis_options, request.limits, remaining, control)?; }
            out.passes.push(pass);
            check_size(&out.passes, max_result_bytes).map_err(Int4SourceError::from)?;
            checkpoint(control)?;
        }
        let next_size = EvidenceSize::of(&next);
        out.levels.push(SummaryHierarchyLevel { input, passes: first..end, next_evidence: if terminal { None } else { Some(next_size) } });
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
        if next_size.bytes >= input.bytes { return Err(Int4SourceMapError::WorkLimit); }
        frontier = next;
    }
    Err(Int4SourceMapError::WorkLimit)
}
