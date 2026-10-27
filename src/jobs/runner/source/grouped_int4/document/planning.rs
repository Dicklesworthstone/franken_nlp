//! Reconstruct the actual complete native plans and compare their sealed receipts.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Expected {
    Map(Int4SourceMapPreflight),
    Keyphrases(Int4SourceMapPreflight),
    Summary(Int4SourceMapPreflight),
    Question(Int4QuestionPreflight),
    SummarySynthesis(Int4SummarySynthesisPreflight),
    SummaryHierarchy(Int4SummaryHierarchyPreflight),
    QuestionSynthesis(Int4QuestionSynthesisPreflight),
    QuestionHierarchy(Int4QuestionHierarchyPreflight),
}
impl Expected {
    pub(super) fn chunk_count(self) -> usize {
        match self {
            Self::Map(p) | Self::Keyphrases(p) | Self::Summary(p) => p.chunk_count(),
            Self::Question(p) => p.chunk_count(),
            Self::SummarySynthesis(p) => p.chunk_count(),
            Self::SummaryHierarchy(p) => p.chunk_count(),
            Self::QuestionSynthesis(p) => p.discovery().chunk_count(),
            Self::QuestionHierarchy(p) => p.discovery().chunk_count(),
        }
    }
    fn native_chunks(self) -> usize {
        match self {
            Self::Question(p) => p.native_chunks(),
            Self::QuestionSynthesis(p) => p.discovery().native_chunks(),
            Self::QuestionHierarchy(p) => p.discovery().native_chunks(),
            _ => self.chunk_count(),
        }
    }
    fn group_width(self) -> usize {
        match self {
            Self::Map(p) | Self::Keyphrases(p) | Self::Summary(p) => p.group_width(),
            Self::Question(p) => p.group_width(),
            Self::SummarySynthesis(p) => p.group_width(),
            Self::SummaryHierarchy(p) => p.group_width(),
            Self::QuestionSynthesis(p) => p.discovery().group_width(),
            Self::QuestionHierarchy(p) => p.group_width(),
        }
    }
    pub(super) fn work(self) -> Result<JobWork, BatchItemFailure> {
        let (model, masks) = match self {
            Self::Map(p) | Self::Keyphrases(p) | Self::Summary(p) => (p.planned_work(), p.reserved_mask_visits()),
            Self::Question(p) => (p.planned_work(), p.reserved_mask_visits()),
            Self::SummarySynthesis(p) => (p.reserved_model_work().map_err(map_failure)?,
                p.reserved_mask_visits().map_err(map_failure)?),
            Self::SummaryHierarchy(p) => (p.reserved_model_work(), p.reserved_mask_visits()),
            Self::QuestionSynthesis(p) => (p.reserved_model_work().map_err(map_failure)?,
                p.reserved_mask_visits().map_err(map_failure)?),
            Self::QuestionHierarchy(p) => (p.reserved_model_work(), p.reserved_mask_visits()),
        };
        Ok(JobWork::from_int4(model, masks))
    }
    pub(super) fn verify(self, run: &Int4SourceDocumentRun, task: &Int4SourceDocumentTask)
        -> Result<(), BatchItemFailure> {
        match (self, run) {
            (Self::Map(p), Int4SourceDocumentRun::Map(run)) => p.verify_completed(run).map_err(map_failure),
            (Self::Keyphrases(p), Int4SourceDocumentRun::Keyphrases(run)) => {
                p.verify_keyphrases_completed(run).map_err(keyphrase_failure)?;
                let Int4SourceDocumentTask::Keyphrases { limits, .. } = task else {
                    return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
                };
                verify_keyphrase_limits(run, *limits)
            }
            (Self::Summary(p), Int4SourceDocumentRun::Summary(run)) => {
                p.verify_summary_completed(run).map_err(summary_failure)?;
                let Int4SourceDocumentTask::Summary { limits, .. } = task else {
                    return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
                };
                if run.summary.requested_max_bullets != limits.max_bullets
                    || run.verification_scan_steps > limits.aggregation.max_scan_steps {
                    return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
                }
                Ok(())
            }
            (Self::Question(p), Int4SourceDocumentRun::Question(run)) => p.verify_completed(run).map_err(map_failure),
            (Self::SummarySynthesis(p), Int4SourceDocumentRun::SummarySynthesis(run)) => p.verify_completed(run).map_err(map_failure),
            (Self::SummaryHierarchy(p), Int4SourceDocumentRun::SummaryHierarchy(run)) => p.verify_completed(run).map_err(map_failure),
            (Self::QuestionSynthesis(p), Int4SourceDocumentRun::QuestionSynthesis(run)) => p.verify_completed(run).map_err(map_failure),
            (Self::QuestionHierarchy(p), Int4SourceDocumentRun::QuestionHierarchy(run)) => p.verify_completed(run).map_err(map_failure),
            _ => Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
        }
    }
}

fn verify_keyphrase_limits(run: &Int4CorpusKeyphraseRun, limits: Int4KeyphraseLimits)
    -> Result<(), BatchItemFailure> {
    let phrases = &run.keyphrases;
    if run.requested_max_phrases != limits.max_phrases
        || phrases.phrases.len() > limits.max_phrases
        || run.verification_scan_work > limits.aggregation.max_scan_work
        || phrases.phrases.len().checked_add(phrases.omitted_candidates)
            .is_none_or(|count| count > limits.aggregation.max_unique_phrases) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    let spans = phrases.phrases.iter().flat_map(|phrase| &phrase.evidence)
        .try_fold(0_usize, |count, evidence| count.checked_add(evidence.spans.len()));
    if spans.is_none_or(|count| count > limits.aggregation.max_evidence_spans) {
        return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution));
    }
    crate::tasks::corpus_keyphrases::check_bytes(run, limits.max_result_bytes)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::OutputLineLimit))
}

pub(super) enum NativePlan<'s> {
    Map(PreparedInt4SourceMap<'s>),
    Keyphrases(PreparedInt4SourceMap<'s>, Int4KeyphraseLimits),
    Summary(PreparedInt4SourceMap<'s>, Int4SummaryLimits),
    Question(PreparedInt4Question<'s>),
    SummarySynthesis(PreparedInt4SummarySynthesis<'s>),
    SummaryHierarchy(PreparedInt4SummaryHierarchy<'s>),
    QuestionSynthesis(PreparedInt4QuestionSynthesis<'s>),
    QuestionHierarchy(PreparedInt4QuestionHierarchy<'s>),
}
impl NativePlan<'_> {
    fn expected(&self) -> Expected {
        match self {
            Self::Map(p) => Expected::Map(p.preflight_metadata()),
            Self::Keyphrases(p, _) => Expected::Keyphrases(p.preflight_metadata()),
            Self::Summary(p, _) => Expected::Summary(p.preflight_metadata()),
            Self::Question(p) => Expected::Question(p.preflight_metadata()),
            Self::SummarySynthesis(p) => Expected::SummarySynthesis(p.preflight_metadata()),
            Self::SummaryHierarchy(p) => Expected::SummaryHierarchy(p.preflight_metadata()),
            Self::QuestionSynthesis(p) => Expected::QuestionSynthesis(p.preflight_metadata()),
            Self::QuestionHierarchy(p) => Expected::QuestionHierarchy(p.preflight_metadata()),
        }
    }
    fn max_kv_bytes(&self) -> u64 {
        match self {
            Self::Map(p) | Self::Keyphrases(p, _) | Self::Summary(p, _) => p.max_kv_bytes(),
            Self::Question(p) => p.max_kv_bytes(),
            Self::SummarySynthesis(p) => p.max_kv_bytes(),
            Self::SummaryHierarchy(p) => p.max_kv_bytes(),
            Self::QuestionSynthesis(p) => p.max_kv_bytes(),
            Self::QuestionHierarchy(p) => p.max_kv_bytes(),
        }
    }
    fn identities<C: DecodeStepControl>(&self, control: &mut C) -> Result<Vec<ExecutionIdentity>, BatchItemFailure> {
        match self {
            Self::Map(p) | Self::Keyphrases(p, _) | Self::Summary(p, _) => copy_identities(p.execution_identities(), control),
            Self::Question(p) => copy_identities(p.execution_identities(), control),
            Self::SummarySynthesis(p) => copy_identities(p.execution_identities(), control),
            Self::SummaryHierarchy(p) => copy_identities(p.execution_identities(), control),
            Self::QuestionSynthesis(p) => copy_identities(p.execution_identities(), control),
            Self::QuestionHierarchy(p) => copy_identities(p.execution_identities(), control),
        }
    }
    pub(super) fn compare<C: DecodeStepControl>(&self, prepared: &PreparedInt4SourceDocument,
        control: &mut C) -> Result<(), BatchItemFailure> {
        if self.expected() != prepared.expected || self.expected().work()? != prepared.work
            || self.identities(control)? != prepared.identities {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(())
    }
    pub(super) fn preflight<C: DecodeStepControl>(&self, identities: &[ExecutionIdentity],
        engine: &StrictInt4Engine<'_>, control: &mut C) -> Result<(), BatchItemFailure> {
        let expected = self.expected();
        if identities.is_empty() || identities.len() != expected.native_chunks()
            || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0)
            || engine.model_identity().group_width() != expected.group_width() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        let required = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), expected.group_width())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if required.kv_bytes > self.max_kv_bytes() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        // Authenticate the complete ordered set before acquiring document output
        // storage. The native synthesis/hierarchy routes additionally preflight
        // every discovery plan and every dynamic level before its first forward.
        for identity in identities {
            poll(control)?;
            constrained_int4::check_model(identity, engine.model_identity())
                .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        match self {
            Self::Map(p) | Self::Keyphrases(p, _) | Self::Summary(p, _) => p.preflight(identities, engine).map_err(map_failure),
            Self::Question(p) => p.preflight(identities, engine).map_err(map_failure),
            _ => Ok(()),
        }
    }
    pub(super) fn execute<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, engine: &mut StrictInt4Engine<'_>, vocabulary: &ExtractionVocabulary,
        control: &mut C) -> Result<Int4SourceDocumentRun, BatchItemFailure> {
        match self {
            Self::Map(p) => p.execute_with_control(admitted, engine, vocabulary, control)
                .map(Int4SourceDocumentRun::Map).map_err(map_failure),
            Self::Keyphrases(p, limits) => p.execute_keyphrases_with_control(admitted, engine, vocabulary, limits, control)
                .map(Int4SourceDocumentRun::Keyphrases).map_err(keyphrase_failure),
            Self::Summary(p, limits) => p.execute_summary_with_control(admitted, engine, vocabulary, limits, control)
                .map(Int4SourceDocumentRun::Summary).map_err(summary_failure),
            Self::Question(p) => p.execute_with_control(admitted, engine, vocabulary, control)
                .map(Int4SourceDocumentRun::Question).map_err(map_failure),
            Self::SummarySynthesis(p) => p.execute_with_control(admitted, planner, engine, vocabulary, control)
                .map(Int4SourceDocumentRun::SummarySynthesis).map_err(map_failure),
            Self::SummaryHierarchy(p) => p.execute_with_control(admitted, planner, engine, vocabulary, control)
                .map(Int4SourceDocumentRun::SummaryHierarchy).map_err(map_failure),
            Self::QuestionSynthesis(p) => p.execute_with_control(admitted, planner, engine, vocabulary, control)
                .map(Int4SourceDocumentRun::QuestionSynthesis).map_err(map_failure),
            Self::QuestionHierarchy(p) => p.execute_with_control(admitted, planner, engine, vocabulary, control)
                .map(Int4SourceDocumentRun::QuestionHierarchy).map_err(map_failure),
        }
    }
}
fn copy_identities<'a, C: DecodeStepControl>(identities: impl ExactSizeIterator<Item = &'a ExecutionIdentity>,
    control: &mut C) -> Result<Vec<ExecutionIdentity>, BatchItemFailure> {
    let mut result = Vec::new();
    result.try_reserve_exact(identities.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
    for identity in identities { poll(control)?; result.push(identity.clone()); }
    Ok(result)
}
impl Int4SourceDocumentJobPlanner {
    pub(super) fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, control: &mut C)
        -> Result<NativePlan<'s>, BatchItemFailure> {
        poll(control)?;
        if source.is_empty() || source.len() > self.mapping.chunks.max_input_bytes {
            return Err(BatchItemFailure::fatal(BatchCode::DocumentLimit));
        }
        let context = PlanContext::new(&self.identity, self.budget)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let width = self.group_width;
        let plan = match self.task() {
            Int4SourceDocumentTask::Map(task) => NativePlan::Map(self.planner.plan_int4_map_with_control(
                source, task, self.budget, &context, self.planning, self.mapping, width, control).map_err(map_failure)?),
            Int4SourceDocumentTask::Keyphrases { options, limits, .. } => {
                let plan = self.planner.plan_int4_map_with_control(source, &SourceMapTask::Keyphrases(*options),
                    self.budget, &context, self.planning, self.mapping, width, control).map_err(map_failure)?;
                plan.check_keyphrases(*limits).map_err(keyphrase_failure)?;
                NativePlan::Keyphrases(plan, *limits)
            }
            Int4SourceDocumentTask::Summary { options, limits } => {
                let plan = self.planner.plan_int4_map_with_control(source, &SourceMapTask::Summarize(*options),
                    self.budget, &context, self.planning, self.mapping, width, control).map_err(map_failure)?;
                plan.check_summary(*limits).map_err(summary_failure)?;
                NativePlan::Summary(plan, *limits)
            }
            Int4SourceDocumentTask::Question(question) => NativePlan::Question(self.planner.plan_int4_question_with_control(
                source, question, self.budget, &context, self.planning, self.mapping, width, control).map_err(map_failure)?),
            Int4SourceDocumentTask::SummarySynthesis(request) => NativePlan::SummarySynthesis(
                self.planner.plan_int4_summary_synthesis_with_control(source, *request, self.budget, &context,
                    self.planning, self.mapping, width, control).map_err(map_failure)?),
            Int4SourceDocumentTask::SummaryHierarchy { request, limits } => NativePlan::SummaryHierarchy(
                self.planner.plan_int4_summary_hierarchy_with_control(source, *request, *limits, self.budget,
                    &context, self.planning, self.mapping, width, control).map_err(map_failure)?),
            Int4SourceDocumentTask::QuestionSynthesis(request) => NativePlan::QuestionSynthesis(
                self.planner.plan_int4_question_synthesis_with_control(source, request, self.budget, &context,
                    self.planning, self.mapping, width, control).map_err(map_failure)?),
            Int4SourceDocumentTask::QuestionHierarchy { request, limits } => NativePlan::QuestionHierarchy(
                self.planner.plan_int4_question_hierarchy_with_control(source, request, *limits, self.budget,
                    &context, self.planning, self.mapping, width, control).map_err(map_failure)?),
        };
        let expected = plan.expected();
        if expected.group_width() != width || expected.chunk_count() == 0
            || expected.chunk_count() > self.mapping.chunks.max_chunks
            || expected.native_chunks() == 0 || expected.native_chunks() > expected.chunk_count()
            || plan.max_kv_bytes() != self.budget.max_kv_bytes
            || !expected.work()?.fits(JobWork::from_int4(self.mapping.max_model_work, self.mapping.max_mask_visits)) {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        poll(control)?;
        Ok(plan)
    }
    /// Prepare the real complete discovery without filesystem or model work.
    /// The caller must retain preparation storage through processor teardown.
    pub fn prepare_with_control<C: DecodeStepControl>(&self, document: BatchDocument<SourceDocumentArgs>,
        control: &mut C) -> Result<PreparedInt4SourceDocument, BatchItemFailure> {
        let plan = self.plan(&document.text, control)?;
        let expected = plan.expected();
        let identities = plan.identities(control)?;
        if identities.len() != expected.native_chunks() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let work = expected.work()?;
        drop(plan);
        poll(control)?;
        Ok(PreparedInt4SourceDocument { source: document.text, identities, expected, work, owner: Arc::clone(&self.owner) })
    }
}
