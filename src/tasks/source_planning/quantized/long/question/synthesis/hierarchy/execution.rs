//! Exact complete QA-context grouping and whole-level native admission.
use super::*;
use crate::tasks::answer::AnswerError;

pub(super) struct Native<'a, 'w> {
    pub planner: &'a SourceTaskPlanner, pub identity: &'a ExecutionIdentity, pub question: &'a str,
    pub budget: TaskBudget, pub planning: SourcePlanningLimits, pub options: AnswerOptions,
    pub mapping: Int8SourceMapLimits, pub engine: &'a mut StrictInt8Engine<'w>, pub vocabulary: &'a ExtractionVocabulary,
    pub expected: Int8QuestionHierarchyPreflight,
}
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct Witness { identity: Sha256Digest, prompt: usize, work: Int8Work }
impl Witness {
    pub(super) fn of(plan: &PreparedInt8SourceTask) -> Result<Self, Int8SourceMapError> {
        let bytes = plan.execution_identity().canonical_json_bytes().map_err(|_| Int8SourceMapError::Admission)?;
        Ok(Self { identity: Sha256Digest::of_bytes(&bytes), prompt: plan.prompt_tokens(), work: plan.planned_work() })
    }
}
// Private scripted-test seam. Every public execution uses pinned native plans.
pub(super) trait Driver {
    type Witness;
    fn count<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], bytes: usize, control: &mut C)
        -> Result<Option<usize>, Int8SourceMapError>;
    fn prepare<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], prompt: usize, control: &mut C)
        -> Result<Self::Witness, Int8SourceMapError>;
    fn run<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], witness: &Self::Witness, control: &mut C)
        -> Result<(Int8SourceTaskRun, Int8Work), Int8SourceMapError>;
}
/// Count question, canonical passage manifest and joined evidence on the
/// retained operation. Rejected probes spend their work; encoding exhaustion
/// and cancellation are failures, never hints to try a smaller renewed budget.
pub(super) fn count_context<C: DecodeStepControl>(planner: &SourceTaskPlanner, question: &str,
    passages: &[AnswerPassage], bytes: usize, mut limits: AnswerInputLimits,
    capacity: Capacity, control: &mut C) -> Result<Option<usize>, Int8SourceMapError> {
    checkpoint(control)?;
    if bytes > capacity.input_bytes || passages.len() > limits.max_passages { return Ok(None); }
    if matches!(planner.document_encoding(), crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(_)) {
        // Count actual compact tokens before deciding context fit. Every token
        // consumes at least one byte; original per-call and operation limits
        // remain in force even when this temporary counting ceiling is larger.
        limits.max_input_tokens = capacity.input_bytes;
    }
    let answer = match AnswerContext::encode_with_control(planner.source_encoder(), question, passages,
        limits, &mut || control.prefill_checkpoint(0)) {
        Ok(answer) => answer,
        Err(AnswerError::InputBudgetExceeded) => return Ok(None),
        Err(error) => return Err(Int8SourceError::from(error).into()),
    };
    let tokens = answer.document().total_token_count();
    if tokens == 0 { return Err(invalid()); }
    let prompt = capacity.scaffold.checked_add(tokens).ok_or(Int8SourceMapError::WorkLimit)?;
    checkpoint(control)?;
    Ok((prompt <= capacity.prompt).then_some(prompt))
}
impl Native<'_, '_> {
    fn plan<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], control: &mut C)
        -> Result<PreparedInt8SourceTask, Int8SourceMapError> {
        checkpoint(control)?;
        let context = PlanContext::new(self.identity, self.budget).map_err(|_| Int8SourceMapError::Admission)?;
        let mut owned = vector(passages.len())?;
        for passage in passages {
            checkpoint(control)?;
            owned.push(AnswerPassage { id: copy(&passage.id)?, text: copy(&passage.text)? });
        }
        let request = SourceTaskRequest::Answer { question: copy(self.question)?, passages: owned,
            options: self.options, budget: self.budget };
        let plan = self.planner.plan_int8_with_control(&request, &context, self.planning, control)?;
        if !within(plan.planned_work(), self.expected.base.synthesis_reserve) { return Err(Int8SourceMapError::WorkLimit); }
        preflight_plans(std::slice::from_ref(&plan), std::slice::from_ref(plan.execution_identity()), self.engine)?;
        Ok(plan)
    }
}
impl Driver for Native<'_, '_> {
    type Witness = Witness;
    fn count<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], bytes: usize, control: &mut C)
        -> Result<Option<usize>, Int8SourceMapError> {
        count_context(self.planner, self.question, passages, bytes, AnswerInputLimits {
            max_passages: self.planning.max_passages, max_input_bytes: self.planning.max_input_bytes,
            max_input_tokens: self.budget.max_input_tokens as usize,
        }, self.expected.capacity, control)
    }
    fn prepare<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], prompt: usize, control: &mut C)
        -> Result<Witness, Int8SourceMapError> {
        let plan = self.plan(passages, control)?;
        if plan.prompt_tokens() != prompt { return Err(Int8SourceMapError::Admission); }
        Witness::of(&plan)
    }
    fn run<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], witness: &Witness, control: &mut C)
        -> Result<(Int8SourceTaskRun, Int8Work), Int8SourceMapError> {
        let plan = self.plan(passages, control)?;
        if Witness::of(&plan)? != *witness { return Err(Int8SourceMapError::Admission); }
        let mut driver = NativeDriver { engine: self.engine, vocabulary: self.vocabulary, control, limits: self.mapping };
        let run = driver.run(&plan, plan.execution_identity())?;
        check_run(&plan, &run, self.mapping.mask_visits_per_chunk)?;
        extract_int8::check_size(&run, plan.max_result_bytes()).map_err(Int8SourceError::from)?;
        Ok((run, plan.planned_work()))
    }
}
struct Group { passages: Range<usize>, bytes: usize, context_bytes: usize, prompt: usize }
fn groups<C: DecodeStepControl, D: Driver>(frontier: &evidence::Collection, expected: Int8QuestionHierarchyPreflight,
    work: &mut HierarchyTokenizerWork, driver: &mut D, control: &mut C) -> Result<Vec<Group>, Int8SourceMapError> {
    let mut groups = vector(frontier.passages.len())?;
    let mut start = 0;
    while start < frontier.passages.len() {
        let mut accepted = None;
        for end in start + 1..=frontier.passages.len() {
            checkpoint(control)?;
            work.calls = work.calls.checked_add(1).filter(|&n| n <= expected.limits.max_tokenizer_calls)
                .ok_or(Int8SourceMapError::WorkLimit)?;
            let passages = &frontier.passages[start..end];
            let context_bytes = transport::context_bytes(expected.capacity.question_bytes,
                passages.iter().map(|p| (p.id.as_str(), p.text.as_str())), control)?;
            work.bytes = work.bytes.checked_add(context_bytes as u64).filter(|&n| n <= expected.limits.max_tokenizer_bytes)
                .ok_or(Int8SourceMapError::WorkLimit)?;
            let Some(prompt) = driver.count(passages, context_bytes, control)? else { break; };
            if prompt <= expected.capacity.scaffold || prompt > expected.capacity.prompt { return Err(invalid()); }
            let bytes = passages.iter().try_fold(0_usize, |sum, p| sum.checked_add(p.text.len()).ok_or_else(invalid))?;
            accepted = Some(Group { passages: start..end, bytes, context_bytes, prompt });
        }
        // Atomic quotations are never split, truncated or silently skipped.
        let group = accepted.ok_or(Int8SourceMapError::WorkLimit)?;
        start = group.passages.end; groups.push(group);
    }
    checkpoint(control)?;
    Ok(groups)
}
pub(super) struct Reduced {
    pub status: QuestionHierarchyStatus, pub final_pass: Option<usize>,
    pub planned: Int8Work, pub actual: Int8Work, pub masks: u64,
    pub tokenizer: HierarchyTokenizerWork, pub levels: Vec<QuestionHierarchyLevel>, pub passes: Vec<QuestionHierarchyPass>,
}
pub(super) fn empty<C: DecodeStepControl>(expected: Int8QuestionHierarchyPreflight, control: &mut C)
    -> Result<Reduced, Int8SourceMapError> {
    let out = Reduced { status: QuestionHierarchyStatus::NoEvidenceCollected, final_pass: None,
        planned: Int8Work::default(), actual: Int8Work::default(), masks: 0,
        tokenizer: HierarchyTokenizerWork::default(), levels: vector(expected.limits.max_levels)?, passes: vector(expected.limits.max_passes)? };
    checkpoint(control)?;
    Ok(out)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn reduce<C: DecodeStepControl, D: Driver>(source: &str, mut frontier: evidence::Collection,
    expected: Int8QuestionHierarchyPreflight, max_result_bytes: u64, remaining: &mut GroundingBudget,
    driver: &mut D, control: &mut C) -> Result<Reduced, Int8SourceMapError> {
    let mut out = empty(expected, control)?;
    if frontier.passages.is_empty() { return Ok(out); }
    for level in 0..expected.limits.max_levels {
        checkpoint(control)?;
        let input = EvidenceSize::of(&frontier);
        let groups = groups(&frontier, expected, &mut out.tokenizer, driver, control)?;
        let end = out.passes.len().checked_add(groups.len()).filter(|&n| n <= expected.limits.max_passes)
            .ok_or(Int8SourceMapError::WorkLimit)?;
        let terminal = groups.len() == 1;
        if !terminal && (end == expected.limits.max_passes || level + 1 == expected.limits.max_levels) {
            return Err(Int8SourceMapError::WorkLimit);
        }
        // Compile/admit EVERY actual context before this level's first forward.
        // Compact witnesses bind full identities, question, passage ids, source
        // and work. Prepared grammars are released immediately and one actual
        // group is rebuilt/rechecked at a time using the same operation ledger.
        let mut witnesses = vector(groups.len())?;
        for group in &groups {
            checkpoint(control)?;
            witnesses.push(driver.prepare(&frontier.passages[group.passages.clone()], group.prompt, control)?);
        }
        let first = out.passes.len();
        let mut next = evidence::Collection::empty();
        for (group_index, (group, witness)) in groups.into_iter().zip(witnesses).enumerate() {
            checkpoint(control)?;
            let window = frontier.window(group.passages.clone(), control)?;
            let (native, planned) = driver.run(&window.passages, &witness, control)?;
            if !within(planned, expected.base.synthesis_reserve) || !within(native.model_work, planned) {
                return Err(Int8SourceMapError::WorkLimit);
            }
            let SourceTaskResult::Answer(raw) = &native.result else { return Err(invalid()); };
            let answer = evidence::lift_answer(source, raw, &window, expected.options, remaining, control)?;
            let masks = raw.mask_node_visit_charge;
            if masks > expected.base.synthesis_masks { return Err(Int8SourceMapError::WorkLimit); }
            let pass = QuestionHierarchyPass { level, group: group_index, input_passages: group.passages,
                input_bytes: group.bytes, context_bytes: group.context_bytes, prompt_tokens: group.prompt,
                planned_model_work: planned, native, answer };
            receipt::verify_pass(&expected, &pass, control)?;
            out.planned = add_work(out.planned, planned).ok_or(Int8SourceMapError::WorkLimit)?;
            out.actual = add_work(out.actual, pass.native.model_work).ok_or(Int8SourceMapError::WorkLimit)?;
            out.masks = out.masks.checked_add(masks).ok_or(Int8SourceMapError::WorkLimit)?;
            if !terminal { next.append_verified(source, &pass.answer, expected, remaining, control)?; }
            out.passes.push(pass);
            extract_int8::check_size(&out.passes, max_result_bytes).map_err(Int8SourceError::from)?;
            checkpoint(control)?;
        }
        let next_size = EvidenceSize::of(&next);
        out.levels.push(QuestionHierarchyLevel { input, passes: first..end,
            next_evidence: if terminal { None } else { Some(next_size) } });
        if terminal {
            out.final_pass = Some(first);
            out.status = match out.passes[first].answer.status {
                SynthesisStatus::Answered => QuestionHierarchyStatus::Answered,
                SynthesisStatus::Abstained => QuestionHierarchyStatus::Abstained,
                SynthesisStatus::NoEvidenceCollected => return Err(invalid()),
            };
            checkpoint(control)?;
            return Ok(out);
        }
        if next_size.passages == 0 {
            out.status = QuestionHierarchyStatus::EvidenceExhausted;
            checkpoint(control)?;
            return Ok(out);
        }
        if next_size.bytes >= input.bytes { return Err(Int8SourceMapError::WorkLimit); }
        frontier = next;
    }
    Err(Int8SourceMapError::WorkLimit)
}
