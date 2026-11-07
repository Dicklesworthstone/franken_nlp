//! Fixed complete-document policies shared by live corpora and durable jobs.
use super::*;

/// One data-only policy for the complete ordered population. Questions retain
/// their private allocation and deliberately have no Debug implementation.
pub enum Int4SourceDocumentTask {
    Map(SourceMapTask),
    Keyphrases { options: KeyphraseOptions, limits: Int4KeyphraseLimits, max_lifetime_scan_work: u64 },
    Summary { options: SummaryOptions, limits: Int4SummaryLimits },
    Question(SourceQuestion),
    SummarySynthesis(SourceSummarySynthesis),
    SummaryHierarchy { request: SourceSummarySynthesis, limits: SummaryHierarchyLimits },
    QuestionSynthesis(SourceQuestionSynthesis),
    QuestionHierarchy { request: SourceQuestionSynthesis, limits: QuestionHierarchyLimits },
}
impl Int4SourceDocumentTask {
    pub(crate) fn kind(&self) -> BuiltInTask {
        match self {
            Self::Map(SourceMapTask::Ner(_)) => BuiltInTask::Ner,
            Self::Map(SourceMapTask::Keyphrases(_)) | Self::Keyphrases { .. } => BuiltInTask::Keyphrases,
            Self::Map(SourceMapTask::Summarize(_)) | Self::Summary { .. }
                | Self::SummarySynthesis(_) | Self::SummaryHierarchy { .. } => BuiltInTask::Summarize,
            Self::Question(_) | Self::QuestionSynthesis(_) | Self::QuestionHierarchy { .. } => BuiltInTask::Answer,
        }
    }
    /// Native semantic execution tag retained in each complete result.
    pub fn execution(&self) -> &'static str {
        match self {
            Self::Map(_) => INT4_SOURCE_MAP_EXECUTION,
            Self::Keyphrases { .. } => INT4_CORPUS_KEYPHRASE_EXECUTION,
            Self::Summary { .. } => INT4_CORPUS_SUMMARY_EXECUTION,
            Self::Question(_) => INT4_QUESTION_EXECUTION,
            Self::SummarySynthesis(_) => INT4_SUMMARY_SYNTHESIS_EXECUTION,
            Self::SummaryHierarchy { .. } => INT4_SUMMARY_HIERARCHY_EXECUTION,
            Self::QuestionSynthesis(_) => INT4_QUESTION_SYNTHESIS_EXECUTION,
            Self::QuestionHierarchy { .. } => INT4_QUESTION_HIERARCHY_EXECUTION,
        }
    }
    pub(crate) fn verification(&self, mapping: Int4SourceMapLimits) -> GroundingBudget {
        match self {
            Self::Map(_) | Self::Keyphrases { .. } | Self::Summary { .. } => mapping.verification,
            Self::SummarySynthesis(request) | Self::SummaryHierarchy { request, .. } => request.limits.verification,
            Self::Question(question) => question.verification,
            Self::QuestionSynthesis(request) | Self::QuestionHierarchy { request, .. } => request.question.verification,
        }
    }
    pub(crate) fn summary_scan_steps(&self) -> u64 {
        match self { Self::Summary { limits, .. } => limits.aggregation.max_scan_steps, _ => 0 }
    }
    pub(crate) fn keyphrase_scan_work(&self) -> u64 {
        match self { Self::Keyphrases { limits, .. } => limits.aggregation.max_scan_work, _ => 0 }
    }
    pub(crate) fn retained_bytes(&self) -> Result<u64, BatchFault> {
        let bytes = match self {
            Self::Map(SourceMapTask::Ner(options)) => options.types.capacity()
                .checked_mul(std::mem::size_of::<crate::tasks::ner::EntityType>()).ok_or(BatchCode::InvalidLimits)?,
            Self::Question(question) => question.question.capacity(),
            Self::QuestionSynthesis(request) | Self::QuestionHierarchy { request, .. } => request.question.question.capacity(),
            _ => 0,
        };
        u64::try_from(bytes).map_err(|_| BatchCode::InvalidLimits.into())
    }
    pub(crate) fn max_result_bytes(&self, mapping: Int4SourceMapLimits) -> u64 {
        match self {
            Self::Keyphrases { limits, .. } => limits.max_result_bytes as u64,
            Self::Summary { limits, .. } => limits.max_result_bytes as u64,
            _ => mapping.reduction.max_result_bytes as u64,
        }
    }
    pub(crate) fn output_token_ceiling(&self, mapping: Int4SourceMapLimits, budget: TaskBudget)
        -> Result<u64, BatchFault> {
        self.output_tokens(mapping.chunks.max_chunks, budget)
    }
    pub(crate) fn output_tokens(&self, chunks: usize, budget: TaskBudget) -> Result<u64, BatchFault> {
        let passes = match self {
            Self::Map(_) => chunks,
            Self::SummarySynthesis(_) => chunks.checked_add(1).ok_or(BatchCode::InvalidLimits)?,
            Self::SummaryHierarchy { limits, .. } => chunks.checked_add(limits.max_passes).ok_or(BatchCode::InvalidLimits)?,
            Self::QuestionHierarchy { limits, .. } => limits.max_passes,
            _ => 0,
        };
        (passes as u64).checked_mul(u64::from(budget.max_output_tokens)).ok_or_else(|| BatchCode::InvalidLimits.into())
    }
    pub(crate) fn verify_lifetime(&self, mapping: Int4SourceMapLimits, lifetime: Int4SourceDocumentLifetime,
        attempts: u64) -> Result<(), BatchFault> {
        verify_lifetime(self.verification(mapping), self.summary_scan_steps(), lifetime, attempts)?;
        if let Self::Keyphrases { limits, max_lifetime_scan_work, .. } = self {
            if limits.aggregation.max_scan_work.checked_mul(attempts).is_none_or(|work| work > *max_lifetime_scan_work) {
                return Err(BatchCode::InvalidLimits.into());
            }
        }
        Ok(())
    }
    pub(crate) fn validate(&self, planning: SourcePlanningLimits, mapping: Int4SourceMapLimits) -> Result<(), BatchFault> {
        match self {
            Self::Map(SourceMapTask::Ner(options)) => options.validate().map_err(|_| BatchCode::InvalidLimits)?,
            Self::Map(SourceMapTask::Keyphrases(options)) => options.validate().map_err(|_| BatchCode::InvalidLimits)?,
            Self::Map(SourceMapTask::Summarize(options)) => options.validate().map_err(|_| BatchCode::InvalidLimits)?,
            Self::Keyphrases { options, limits, .. } => {
                options.validate().map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate(mapping).map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::Summary { options, limits } => {
                options.validate().map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate(mapping).map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::Question(question) => validate_question(question, planning)?,
            Self::SummarySynthesis(request) => request.validate(planning).map_err(|_| BatchCode::InvalidLimits)?,
            Self::SummaryHierarchy { request, limits } => {
                request.validate(planning).map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::QuestionSynthesis(request) => {
                validate_question(&request.question, planning)?;
                request.limits.validate(planning).map_err(|_| BatchCode::InvalidLimits)?;
            }
            Self::QuestionHierarchy { request, limits } => {
                validate_question(&request.question, planning)?;
                request.limits.validate(planning).map_err(|_| BatchCode::InvalidLimits)?;
                limits.validate().map_err(|_| BatchCode::InvalidLimits)?;
            }
        }
        Ok(())
    }
}
fn validate_question(question: &SourceQuestion, planning: SourcePlanningLimits) -> Result<(), BatchFault> {
    question.validate().map_err(|_| BatchCode::InvalidLimits)?;
    if question.question.len() > planning.max_input_bytes { return Err(BatchCode::InvalidLimits.into()); }
    Ok(())
}

/// Durable population verification authority. Kept byte-for-byte serializable
/// in its original form; live corpora use independent incremental counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Int4SourceDocumentLifetime {
    pub verification: GroundingBudget,
    pub summary_scan_steps: u64,
}
pub(crate) fn verify_lifetime(per: GroundingBudget, scans: u64, max: Int4SourceDocumentLifetime,
    attempts: u64) -> Result<(), BatchFault> {
    let count = usize::try_from(attempts).map_err(|_| BatchCode::InvalidLimits)?;
    if count == 0 || per.max_fields.checked_mul(count).is_none_or(|n| n > max.verification.max_fields)
        || per.max_matches.checked_mul(count).is_none_or(|n| n > max.verification.max_matches)
        || per.max_scan_steps.checked_mul(attempts).is_none_or(|n| n > max.verification.max_scan_steps)
        || scans.checked_mul(attempts).is_none_or(|n| n > max.summary_scan_steps) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(())
}
