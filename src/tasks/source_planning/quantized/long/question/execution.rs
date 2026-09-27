//! One native engine/controller, independent verification, lossless answer merge.
use super::*;
use std::cell::RefCell;

impl PreparedInt8Question<'_> {
    /// Every native identity, task variant and resident KV requirement is checked
    /// before the FIRST forward, including the final nonblank source passage.
    pub fn preflight(&self, admitted: &[ExecutionIdentity], engine: &StrictInt8Engine<'_>)
        -> Result<(), Int8SourceMapError> {
        self.check_plans(admitted)?;
        preflight_plans(&self.plans, admitted, engine)
    }
    fn check_plans(&self, admitted: &[ExecutionIdentity]) -> Result<(), Int8SourceMapError> {
        verify_identities(&self.plans, admitted)?;
        if self.plan_indices.len() != self.chunks.chunks().len()
            || self.expected.native_chunks != self.plans.len()
            || !self.plan_indices.iter().filter_map(|index| *index).eq(0..self.plans.len()) {
            return Err(Int8SourceMapError::Admission);
        }
        for plan in &self.plans {
            if !matches!(&plan.finalizer, Finalizer::Answer(_, options) if *options == self.options)
                || plan.execution_identity().task_spec != ANSWER_TASK_VERSION {
                return Err(Int8SourceMapError::Admission);
            }
        }
        Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int8QuestionRun, Int8SourceMapError> {
        checkpoint(control)?;
        self.preflight(admitted, engine)?;
        let driver = NativeDriver { engine, vocabulary, control, limits: self.mapping };
        self.execute_with_driver(admitted, driver)
    }
    // Private corruption-test seam only. Public entrypoints always use the SAME
    // native driver as the source portfolio, never a caller-supplied fake model.
    pub(super) fn execute_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity], driver: D)
        -> Result<Int8QuestionRun, Int8SourceMapError> {
        self.check_plans(admitted)?;
        let driver = RefCell::new(driver);
        let mut task = Executor { plans: &self.plans, indices: &self.plan_indices, admitted,
            driver: &driver, options: self.options, remaining: self.verification, next: 0,
            work: Int8Work::default(), masks: 0, mask_cap: self.mapping.mask_visits_per_chunk };
        let mut stopped = None;
        let mapped = mapreduce::execute(&self.chunks, &mut task, self.mapping.reduction, || {
            match driver.borrow_mut().checkpoint() {
                Ok(()) => Ok(()), Err(e) => { stopped = Some(e); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = stopped { return Err(error.into()); }
        let mapped = mapped.map_err(Int8SourceMapError::Reduction)?;
        driver.borrow_mut().checkpoint()?;
        let stats = statistics(mapped.root().value())?;
        if task.next != self.chunks.chunks().len() || stats.work != task.work || stats.masks != task.masks {
            return Err(Int8SourceError::InvalidResult.into());
        }
        let run = Int8QuestionRun { schema_version: 1, execution: INT8_QUESTION_EXECUTION,
            numerics_profile: STRICT_INT8_PROFILE, semantics: "independent-passage-answers-no-global-selection-v1",
            outcome: stats.outcome, distinct_answer_texts: stats.distinct,
            native_chunks: stats.answered + stats.abstained, whitespace_chunks: stats.blank,
            answered_chunks: stats.answered, abstained_chunks: stats.abstained,
            planned_model_work: self.expected.work, model_work: task.work,
            reserved_mask_node_visits: self.expected.masks, mask_node_visit_charge: task.masks,
            verification_scan_steps: self.verification.max_scan_steps - task.remaining.max_scan_steps,
            calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
            semantic_support: SummarySemanticSupport::NotAssessed, untrusted_fields: ["mapped.root.value"], mapped };
        self.expected.verify_completed(&run)?;
        extract_int8::check_size(&run, self.max_result_bytes()).map_err(Int8SourceError::from)?;
        driver.borrow_mut().checkpoint()?;
        Ok(run)
    }
}

struct Executor<'a, D> {
    plans: &'a [PreparedInt8SourceTask], indices: &'a [Option<usize>], admitted: &'a [ExecutionIdentity],
    driver: &'a RefCell<D>, options: AnswerOptions, remaining: GroundingBudget,
    next: usize, work: Int8Work, masks: u64, mask_cap: u64,
}
impl<D: SourceDriver> MapReduceTask for Executor<'_, D> {
    type Value = QuestionValue;
    type Error = Int8SourceError;
    fn policy(&self) -> ReductionPolicy {
        ReductionPolicy { id: "all-passage-answers-source-order-no-voting-v1", may_discard_information: false }
    }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<QuestionValue>>, Int8SourceError> {
        let mut values = Vec::new();
        values.try_reserve_exact(chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
        for chunk in chunks {
            self.driver.borrow_mut().checkpoint()?;
            if chunk.id() != self.next { return Err(Int8SourceError::InvalidResult); }
            let index = self.indices.get(chunk.id()).ok_or(Int8SourceError::InvalidResult)?;
            let value = if let Some(index) = *index {
                if !has_text(chunk.text()) { return Err(Int8SourceError::InvalidResult); }
                let plan = self.plans.get(index).ok_or(Int8SourceError::InvalidResult)?;
                let admitted = self.admitted.get(index).ok_or(Int8SourceError::InvalidResult)?;
                let native = self.driver.borrow_mut().run(plan, admitted)?;
                check_run(plan, &native, self.mask_cap)?;
                extract_int8::check_size(&native, plan.max_result_bytes())?;
                let work = native.model_work;
                let SourceTaskResult::Answer(raw) = native.result else { return Err(Int8SourceError::InvalidResult); };
                let value = convert(chunk, raw, work, self.options, &mut self.remaining,
                    &mut || self.driver.borrow_mut().checkpoint())?;
                self.work = add_work(self.work, value.model_work).ok_or(Int8SourceError::InvalidResult)?;
                self.masks = self.masks.checked_add(value.mask_node_visit_charge).ok_or(Int8SourceError::InvalidResult)?;
                value
            } else {
                if has_text(chunk.text()) { return Err(Int8SourceError::InvalidResult); }
                QuestionChunk { chunk_id: chunk.id(), source_span: chunk.span(), status: QuestionChunkStatus::WhitespaceOnly,
                    answer: None, citations: Vec::new(), model_work: Int8Work::default(), mask_node_visit_charge: 0 }
            };
            values.push(MapOutput { chunk_id: chunk.id(), value: QuestionValue { chunks: vec![Arc::new(value)] } });
            self.next += 1;
            self.driver.borrow_mut().checkpoint()?;
        }
        Ok(values)
    }
    fn reduce(&mut self, input: ReduceInput<'_, QuestionValue>) -> Result<QuestionValue, Int8SourceError> {
        self.driver.borrow_mut().checkpoint()?;
        let count = input.children.iter().try_fold(0_usize, |n, c| n.checked_add(c.value().chunks.len()))
            .ok_or(Int8SourceError::InvalidResult)?;
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(count).map_err(|_| SourcePlanningError::AllocationRefused)?;
        let mut next = input.children.first().ok_or(Int8SourceError::InvalidResult)?.chunk_range().start;
        for child in input.children {
            if child.chunk_range().start != next { return Err(Int8SourceError::InvalidResult); }
            for chunk in &child.value().chunks {
                self.driver.borrow_mut().checkpoint()?;
                if chunk.chunk_id != next { return Err(Int8SourceError::InvalidResult); }
                chunks.push(Arc::clone(chunk)); next += 1;
            }
            if child.chunk_range().end != next { return Err(Int8SourceError::InvalidResult); }
        }
        self.driver.borrow_mut().checkpoint()?;
        Ok(QuestionValue { chunks })
    }
}

fn convert<C: FnMut() -> Result<(), Int8SourceError>>(chunk: &SourceChunk<'_>, raw: AnswerResult,
    work: Int8Work, options: AnswerOptions, remaining: &mut GroundingBudget, checkpoint: &mut C)
    -> Result<QuestionChunk, Int8SourceError> {
    checkpoint()?;
    if raw.schema_version != 1 || raw.task_spec_version != ANSWER_TASK_VERSION || raw.numerics_profile != STRICT_INT8_PROFILE
        || raw.calibration != AnswerCalibration::Uncalibrated || raw.score_space != ScoreSpace::NotComputed
        || raw.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || raw.semantic_support != SummarySemanticSupport::NotAssessed
        || raw.untrusted_fields[0] != "answer" || raw.untrusted_fields[1] != "citations"
        || raw.citations.len() > options.max_citations {
        return Err(Int8SourceError::InvalidResult);
    }
    let status = match raw.status {
        AnswerStatus::Answered => {
            if !raw.answerable || !raw.answer.as_ref().is_some_and(|a| has_text(a)
                && a.chars().count() <= options.max_answer_scalars) || raw.citations.is_empty() {
                return Err(Int8SourceError::InvalidResult);
            }
            QuestionChunkStatus::Answered
        }
        AnswerStatus::Abstained => {
            if raw.answerable || raw.answer.is_some() || !raw.citations.is_empty() { return Err(Int8SourceError::InvalidResult); }
            QuestionChunkStatus::Abstained
        }
    };
    remaining.max_fields = remaining.max_fields.checked_sub(raw.citations.len())
        .ok_or(AnswerError::OutputBudgetExceeded)?;
    let mut citations = Vec::new();
    citations.try_reserve_exact(raw.citations.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
    for citation in raw.citations {
        citations.push(lift(chunk, citation, options, remaining, checkpoint)?);
    }
    checkpoint()?;
    Ok(QuestionChunk { chunk_id: chunk.id(), source_span: chunk.span(), status, answer: raw.answer,
        citations, model_work: work, mask_node_visit_charge: raw.mask_node_visit_charge })
}
fn lift<C: FnMut() -> Result<(), Int8SourceError>>(chunk: &SourceChunk<'_>, citation: PassageCitation,
    options: AnswerOptions, remaining: &mut GroundingBudget, checkpoint: &mut C) -> Result<SourceCitation, Int8SourceError> {
    checkpoint()?;
    if citation.quote.is_empty() || citation.quote.chars().count() > options.max_quote_scalars {
        return Err(Int8SourceError::InvalidResult);
    }
    let mut spans = scan_occurrences(chunk.text(), &citation.quote, remaining).map_err(|e| match e {
        crate::validation::grounded_fields::FieldGroundingError::AllocationRefused => Int8SourceError::from(AnswerError::AllocationRefused),
        crate::validation::grounded_fields::FieldGroundingError::FieldBudget
        | crate::validation::grounded_fields::FieldGroundingError::MatchBudget
        | crate::validation::grounded_fields::FieldGroundingError::WorkBudget => Int8SourceError::from(AnswerError::OutputBudgetExceeded),
        _ => Int8SourceError::InvalidResult,
    })?;
    if citation.spans.len() != spans.len() || citation.occurrence != (if spans.len() == 1 {
        SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous })
        || citation.spans.iter().zip(&spans).any(|(p, span)| p.passage_id != PASSAGE_ID || p.span != *span) {
        return Err(Int8SourceError::InvalidResult);
    }
    let origin = chunk.span();
    for span in &mut spans {
        checkpoint()?;
        span.byte_start = span.byte_start.checked_add(origin.byte_start).ok_or(Int8SourceError::InvalidResult)?;
        span.byte_end = span.byte_end.checked_add(origin.byte_start).ok_or(Int8SourceError::InvalidResult)?;
        span.scalar_start = span.scalar_start.checked_add(origin.scalar_start).ok_or(Int8SourceError::InvalidResult)?;
        span.scalar_end = span.scalar_end.checked_add(origin.scalar_start).ok_or(Int8SourceError::InvalidResult)?;
    }
    Ok(SourceCitation { quote: citation.quote, occurrence: citation.occurrence, spans })
}

pub(super) struct Statistics {
    pub answered: usize, pub abstained: usize, pub blank: usize, pub distinct: usize,
    pub outcome: QuestionOutcome, pub work: Int8Work, pub masks: u64,
    pub citations: usize, pub spans: usize,
}
pub(super) fn statistics(value: &QuestionValue) -> Result<Statistics, Int8SourceError> {
    let mut answers = Vec::new();
    answers.try_reserve_exact(value.chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
    let mut s = Statistics { answered: 0, abstained: 0, blank: 0, distinct: 0,
        outcome: QuestionOutcome::NoAnswerProposed, work: Int8Work::default(), masks: 0, citations: 0, spans: 0 };
    for chunk in value.chunks() {
        s.citations = s.citations.checked_add(chunk.citations.len()).ok_or(Int8SourceError::InvalidResult)?;
        for citation in &chunk.citations {
            s.spans = s.spans.checked_add(citation.spans.len()).ok_or(Int8SourceError::InvalidResult)?;
        }
        s.work = add_work(s.work, chunk.model_work).ok_or(Int8SourceError::InvalidResult)?;
        s.masks = s.masks.checked_add(chunk.mask_node_visit_charge).ok_or(Int8SourceError::InvalidResult)?;
        match chunk.status {
            QuestionChunkStatus::Answered => {
                s.answered += 1;
                answers.push(chunk.answer.as_deref().ok_or(Int8SourceError::InvalidResult)?);
            }
            QuestionChunkStatus::Abstained => s.abstained += 1,
            QuestionChunkStatus::WhitespaceOnly => s.blank += 1,
        }
    }
    answers.sort_unstable(); answers.dedup(); s.distinct = answers.len();
    s.outcome = match s.distinct { 0 => QuestionOutcome::NoAnswerProposed,
        1 => QuestionOutcome::OneAnswerText, _ => QuestionOutcome::MultipleAnswerTexts };
    Ok(s)
}
