//! One real grouped-INT4 engine, complete admission and independently lifted evidence.
use super::*;
use super::super::execution::{SourceDriver, NativeDriver, check_run};

impl PreparedInt4Question<'_> {
    fn check_plans(&self, admitted: &[ExecutionIdentity]) -> Result<(), Int4SourceMapError> {
        if self.plans.is_empty() || self.plans.len() != admitted.len()
            || self.plan_indices.len() != self.chunks.chunks().len()
            || self.expected.native_chunks != self.plans.len()
            || !self.plan_indices.iter().filter_map(|index| *index).eq(0..self.plans.len()) {
            return Err(Int4SourceMapError::Admission);
        }
        for (plan, identity) in self.plans.iter().zip(admitted) {
            plan.verify_identity(identity)?;
            if !matches!(&plan.finalizer, Finalizer::Answer(_, options) if *options == self.options)
                || plan.execution_identity().task_spec != ANSWER_TASK_VERSION
                || plan.group_width() != self.expected.group_width {
                return Err(Int4SourceMapError::Admission);
            }
        }
        Ok(())
    }
    /// The final native passage is admitted before the first forward. Full
    /// resident KV capacity, actual model identity and group partition must fit.
    pub fn preflight(&self, admitted: &[ExecutionIdentity], engine: &StrictInt4Engine<'_>)
        -> Result<(), Int4SourceMapError> {
        self.check_plans(admitted)?;
        if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
            || engine.model_identity().group_width() != self.expected.group_width {
            return Err(Int4SourceMapError::Admission);
        }
        let capacity = engine.kv_cache().capacity_positions();
        let required = Int4MemoryRequirement::for_context(capacity, self.expected.group_width)
            .map_err(|_| Int4SourceMapError::Admission)?;
        for (plan, identity) in self.plans.iter().zip(admitted) {
            constrained_int4::check_model(identity, engine.model_identity()).map_err(Int4SourceError::from)?;
            if plan.planned_work().forward_positions > capacity as u64
                || required.kv_bytes > plan.max_kv_bytes() || required.kv_bytes > plan.task_budget().max_kv_bytes {
                return Err(Int4SourceMapError::Admission);
            }
        }
        Ok(())
    }
    pub fn execute_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int4QuestionRun, Int4SourceMapError> {
        checkpoint(control)?;
        let limits = self.mapping;
        self.execute_with_driver(admitted, NativeDriver { engine, vocabulary, control, limits })
    }
    // Private fixture seam. All public execution uses the native INT4 driver.
    pub(super) fn execute_with_driver<D: SourceDriver>(self, admitted: &[ExecutionIdentity], mut driver: D)
        -> Result<Int4QuestionRun, Int4SourceMapError> {
        driver.checkpoint()?;
        self.check_plans(admitted)?;
        super::super::execution::preflight(&self.plans, admitted, &driver)?;
        let driver = RefCell::new(driver);
        let mut task = Executor { plans: &self.plans, indices: &self.plan_indices, admitted,
            driver: &driver, options: self.options, remaining: self.verification, next: 0,
            work: Int4Work::default(), masks: 0, mask_cap: self.mapping.mask_visits_per_chunk };
        let mut stopped = None;
        let mapped = mapreduce::execute(&self.chunks, &mut task, self.mapping.reduction, || {
            match driver.borrow_mut().checkpoint() {
                Ok(()) => Ok(()), Err(error) => { stopped = Some(error); Err(MapReduceError::Cancelled) }
            }
        });
        if let Some(error) = stopped { return Err(error.into()); }
        let mapped = mapped.map_err(Int4SourceMapError::Reduction)?;
        driver.borrow_mut().checkpoint()?;
        let stats = statistics(mapped.root().value())?;
        if task.next != self.chunks.chunks().len() || stats.work != task.work || stats.masks != task.masks {
            return Err(Int4SourceError::InvalidResult.into());
        }
        let run = Int4QuestionRun { schema_version: 1, execution: INT4_QUESTION_EXECUTION,
            numerics_profile: STRICT_INT4_PROFILE, group_width: self.expected.group_width,
            semantics: QUESTION_SEMANTICS, outcome: stats.outcome, distinct_answer_texts: stats.distinct,
            native_chunks: stats.answered + stats.abstained, whitespace_chunks: stats.blank,
            answered_chunks: stats.answered, abstained_chunks: stats.abstained,
            planned_model_work: self.expected.work, model_work: task.work,
            reserved_mask_node_visits: self.expected.masks, mask_node_visit_charge: task.masks,
            verification_used: SourceMapVerification::used(self.verification, task.remaining)?,
            calibration: AnswerCalibration::Uncalibrated,
            citation_guarantee: CitationGuarantee::StructuralSourceMembership,
            semantic_support: SummarySemanticSupport::NotAssessed, untrusted_fields: ["mapped.root.value"], mapped };
        self.expected.verify_completed(&run)?;
        check_size(&run, self.max_result_bytes()).map_err(Int4SourceError::from)?;
        driver.borrow_mut().checkpoint()?;
        Ok(run)
    }
}

struct Executor<'a, D> {
    plans: &'a [PreparedInt4SourceTask], indices: &'a [Option<usize>], admitted: &'a [ExecutionIdentity],
    driver: &'a RefCell<D>, options: AnswerOptions, remaining: GroundingBudget,
    next: usize, work: Int4Work, masks: u64, mask_cap: u64,
}
impl<D: SourceDriver> MapReduceTask for Executor<'_, D> {
    type Value = QuestionValue;
    type Error = Int4SourceError;
    fn policy(&self) -> ReductionPolicy {
        ReductionPolicy { id: "all-passage-answers-source-order-no-voting-v1", may_discard_information: false }
    }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<QuestionValue>>, Int4SourceError> {
        let mut values = Vec::new();
        values.try_reserve_exact(chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
        for chunk in chunks {
            let mut driver = self.driver.borrow_mut();
            driver.checkpoint()?;
            if !driver.clean() || chunk.id() != self.next { return Err(Int4SourceError::InvalidResult); }
            let index = self.indices.get(chunk.id()).ok_or(Int4SourceError::InvalidResult)?;
            let value = if let Some(index) = *index {
                if !has_text(chunk.text()) { return Err(Int4SourceError::InvalidResult); }
                let plan = self.plans.get(index).ok_or(Int4SourceError::InvalidResult)?;
                let admitted = self.admitted.get(index).ok_or(Int4SourceError::InvalidResult)?;
                let native = driver.run(plan, admitted)?;
                check_run(plan, &native, self.mask_cap)?;
                if !driver.clean() { return Err(Int4SourceError::InvalidResult); }
                let work = native.model_work;
                let SourceTaskResult::Answer(raw) = native.result else { return Err(Int4SourceError::InvalidResult); };
                let value = convert(chunk, raw, work, self.options, &mut self.remaining, &mut || driver.checkpoint())?;
                self.work = add_work(self.work, value.model_work).ok_or(Int4SourceError::InvalidResult)?;
                self.masks = self.masks.checked_add(value.mask_node_visit_charge).ok_or(Int4SourceError::InvalidResult)?;
                value
            } else {
                if has_text(chunk.text()) { return Err(Int4SourceError::InvalidResult); }
                QuestionChunk { chunk_id: chunk.id(), source_span: chunk.span(), status: QuestionChunkStatus::WhitespaceOnly,
                    answer: None, citations: Vec::new(), model_work: Int4Work::default(), mask_node_visit_charge: 0 }
            };
            values.push(MapOutput { chunk_id: chunk.id(), value: QuestionValue { chunks: vec![Arc::new(value)] } });
            self.next += 1;
            driver.checkpoint()?;
        }
        Ok(values)
    }
    fn reduce(&mut self, input: ReduceInput<'_, QuestionValue>) -> Result<QuestionValue, Int4SourceError> {
        let mut driver = self.driver.borrow_mut();
        driver.checkpoint()?;
        let count = input.children.iter().try_fold(0_usize, |n, child| n.checked_add(child.value().chunks.len()))
            .ok_or(Int4SourceError::InvalidResult)?;
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(count).map_err(|_| SourcePlanningError::AllocationRefused)?;
        let mut next = input.children.first().ok_or(Int4SourceError::InvalidResult)?.chunk_range().start;
        for child in input.children {
            if child.chunk_range().start != next { return Err(Int4SourceError::InvalidResult); }
            for chunk in &child.value().chunks {
                driver.checkpoint()?;
                if chunk.chunk_id != next { return Err(Int4SourceError::InvalidResult); }
                chunks.push(Arc::clone(chunk)); next += 1;
            }
            if child.chunk_range().end != next { return Err(Int4SourceError::InvalidResult); }
        }
        driver.checkpoint()?;
        Ok(QuestionValue { chunks })
    }
}

fn convert<C: FnMut() -> Result<(), Int4SourceError>>(chunk: &SourceChunk<'_>, raw: AnswerResult,
    work: Int4Work, options: AnswerOptions, remaining: &mut GroundingBudget, checkpoint: &mut C)
    -> Result<QuestionChunk, Int4SourceError> {
    checkpoint()?;
    if raw.schema_version != 1 || raw.task_spec_version != ANSWER_TASK_VERSION || raw.numerics_profile != STRICT_INT4_PROFILE
        || raw.calibration != AnswerCalibration::Uncalibrated || raw.score_space != ScoreSpace::NotComputed
        || raw.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || raw.semantic_support != SummarySemanticSupport::NotAssessed
        || raw.untrusted_fields[0] != "answer" || raw.untrusted_fields[1] != "citations"
        || raw.citations.len() > options.max_citations {
        return Err(Int4SourceError::InvalidResult);
    }
    let status = match raw.status {
        AnswerStatus::Answered => {
            if !raw.answerable || !raw.answer.as_ref().is_some_and(|answer| has_text(answer)
                && answer.chars().count() <= options.max_answer_scalars) || raw.citations.is_empty() {
                return Err(Int4SourceError::InvalidResult);
            }
            QuestionChunkStatus::Answered
        }
        AnswerStatus::Abstained => {
            if raw.answerable || raw.answer.is_some() || !raw.citations.is_empty() { return Err(Int4SourceError::InvalidResult); }
            QuestionChunkStatus::Abstained
        }
    };
    remaining.max_fields = remaining.max_fields.checked_sub(raw.citations.len())
        .ok_or(AnswerError::OutputBudgetExceeded)?;
    let mut citations = Vec::new();
    citations.try_reserve_exact(raw.citations.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
    for citation in raw.citations { citations.push(lift(chunk, citation, options, remaining, checkpoint)?); }
    checkpoint()?;
    Ok(QuestionChunk { chunk_id: chunk.id(), source_span: chunk.span(), status, answer: raw.answer,
        citations, model_work: work, mask_node_visit_charge: raw.mask_node_visit_charge })
}
fn lift<C: FnMut() -> Result<(), Int4SourceError>>(chunk: &SourceChunk<'_>, citation: PassageCitation,
    options: AnswerOptions, remaining: &mut GroundingBudget, checkpoint: &mut C) -> Result<SourceCitation, Int4SourceError> {
    checkpoint()?;
    if citation.quote.is_empty() || citation.quote.chars().count() > options.max_quote_scalars {
        return Err(Int4SourceError::InvalidResult);
    }
    let mut spans = scan_occurrences(chunk.text(), &citation.quote, remaining).map_err(|error| match error {
        FieldGroundingError::AllocationRefused => Int4SourceError::from(AnswerError::AllocationRefused),
        FieldGroundingError::FieldBudget | FieldGroundingError::MatchBudget | FieldGroundingError::WorkBudget =>
            Int4SourceError::from(AnswerError::OutputBudgetExceeded),
        _ => Int4SourceError::InvalidResult,
    })?;
    if citation.spans.len() != spans.len() || citation.occurrence != (if spans.len() == 1 {
        SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous })
        || citation.spans.iter().zip(&spans).any(|(passage, span)| passage.passage_id != PASSAGE_ID || passage.span != *span) {
        return Err(Int4SourceError::InvalidResult);
    }
    let origin = chunk.span();
    for span in &mut spans {
        checkpoint()?;
        span.byte_start = span.byte_start.checked_add(origin.byte_start).ok_or(Int4SourceError::InvalidResult)?;
        span.byte_end = span.byte_end.checked_add(origin.byte_start).ok_or(Int4SourceError::InvalidResult)?;
        span.scalar_start = span.scalar_start.checked_add(origin.scalar_start).ok_or(Int4SourceError::InvalidResult)?;
        span.scalar_end = span.scalar_end.checked_add(origin.scalar_start).ok_or(Int4SourceError::InvalidResult)?;
        if span.byte_end > origin.byte_end || span.scalar_end > origin.scalar_end { return Err(Int4SourceError::InvalidResult); }
    }
    Ok(SourceCitation { quote: citation.quote, occurrence: citation.occurrence, spans })
}

pub(super) struct Statistics {
    pub answered: usize, pub abstained: usize, pub blank: usize, pub distinct: usize,
    pub outcome: QuestionOutcome, pub work: Int4Work, pub masks: u64,
    pub citations: usize, pub spans: usize,
}
pub(super) fn statistics(value: &QuestionValue) -> Result<Statistics, Int4SourceError> {
    let mut answers = Vec::new();
    answers.try_reserve_exact(value.chunks.len()).map_err(|_| SourcePlanningError::AllocationRefused)?;
    let mut stats = Statistics { answered: 0, abstained: 0, blank: 0, distinct: 0,
        outcome: QuestionOutcome::NoAnswerProposed, work: Int4Work::default(), masks: 0, citations: 0, spans: 0 };
    for chunk in value.chunks() {
        stats.citations = stats.citations.checked_add(chunk.citations.len()).ok_or(Int4SourceError::InvalidResult)?;
        for citation in &chunk.citations {
            stats.spans = stats.spans.checked_add(citation.spans.len()).ok_or(Int4SourceError::InvalidResult)?;
        }
        stats.work = add_work(stats.work, chunk.model_work).ok_or(Int4SourceError::InvalidResult)?;
        stats.masks = stats.masks.checked_add(chunk.mask_node_visit_charge).ok_or(Int4SourceError::InvalidResult)?;
        match chunk.status {
            QuestionChunkStatus::Answered => {
                let answer = chunk.answer.as_deref().ok_or(Int4SourceError::InvalidResult)?;
                if !has_text(answer) || chunk.citations.is_empty() { return Err(Int4SourceError::InvalidResult); }
                stats.answered += 1; answers.push(answer);
            }
            QuestionChunkStatus::Abstained => {
                if chunk.answer.is_some() || !chunk.citations.is_empty() { return Err(Int4SourceError::InvalidResult); }
                stats.abstained += 1;
            }
            QuestionChunkStatus::WhitespaceOnly => {
                if chunk.answer.is_some() || !chunk.citations.is_empty() || chunk.model_work != Int4Work::default()
                    || chunk.mask_node_visit_charge != 0 { return Err(Int4SourceError::InvalidResult); }
                stats.blank += 1;
            }
        }
    }
    answers.sort_unstable(); answers.dedup(); stats.distinct = answers.len();
    stats.outcome = match stats.distinct { 0 => QuestionOutcome::NoAnswerProposed,
        1 => QuestionOutcome::OneAnswerText, _ => QuestionOutcome::MultipleAnswerTexts };
    Ok(stats)
}
