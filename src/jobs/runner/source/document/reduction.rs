//! Durable document-level selection on the existing real INT8 reducers.
//! No second neural pass, intermediate top-k or new runtime.
use super::*;
use crate::{
    corpus::summarize::SUMMARY_REDUCTION_POLICY,
    native_engine::decode::DecodeCancellationKind,
    tasks::{corpus_keyphrases::{CorpusKeyphraseLimits, CORPUS_KEYPHRASE_POLICY},
        source_planning::quantized::long::{
            summary::{Int8SummaryLimits, Int8CorpusSummaryRun, Int8CorpusSummaryError, INT8_CORPUS_SUMMARY_EXECUTION},
            keyphrase_reduction::{Int8KeyphraseLimits, Int8CorpusKeyphraseRun, Int8CorpusKeyphraseError,
                INT8_CORPUS_KEYPHRASE_EXECUTION}}},
};
mod execution;
pub use execution::{Int8ReducedSourceDocumentRun, Int8ReducedSourceDocumentJobProcessor};

/// Data-only policy. Construction grants no native, memory or job authority.
/// Every possible durable attempt must fit the lifetime evidence-scan ceiling,
/// including failed and acknowledgement-uncertain attempts.
#[derive(Clone, Copy, Debug)]
pub enum SourceDocumentReduction {
    Summary { limits: Int8SummaryLimits, max_lifetime_scan_work: u64 },
    Keyphrases { limits: Int8KeyphraseLimits, max_lifetime_scan_work: u64 },
}
impl SourceDocumentReduction {
    pub fn max_result_bytes(self) -> u64 {
        match self { Self::Summary { limits, .. } => limits.max_result_bytes as u64,
            Self::Keyphrases { limits, .. } => limits.max_result_bytes as u64 }
    }
    pub fn scan_work_per_attempt(self) -> u64 {
        match self { Self::Summary { limits, .. } => limits.aggregation.max_scan_steps,
            Self::Keyphrases { limits, .. } => limits.aggregation.max_scan_work }
    }
    pub fn execution(self) -> &'static str {
        match self { Self::Summary { .. } => INT8_CORPUS_SUMMARY_EXECUTION,
            Self::Keyphrases { .. } => INT8_CORPUS_KEYPHRASE_EXECUTION }
    }
    pub fn validate(self, task: &SourceMapTask, mapping: Int8SourceMapLimits, job: JobLimits) -> Result<(), BatchFault> {
        job.validate().map_err(|_| BatchCode::InvalidLimits)?;
        let lifetime = match (self, task) {
            (Self::Summary { limits, max_lifetime_scan_work }, SourceMapTask::Summarize(_)) => {
                limits.validate(mapping).map_err(|_| BatchCode::InvalidLimits)?;
                max_lifetime_scan_work
            }
            (Self::Keyphrases { limits, max_lifetime_scan_work }, SourceMapTask::Keyphrases(_)) => {
                limits.validate(mapping).map_err(|_| BatchCode::InvalidLimits)?;
                max_lifetime_scan_work
            }
            _ => return Err(BatchCode::InvalidLimits.into()),
        };
        if self.max_result_bytes() > job.max_result_bytes as u64
            || !scans_cover_attempts(self.scan_work_per_attempt(), lifetime, job.max_attempts) {
            return Err(BatchCode::InvalidLimits.into());
        }
        Ok(())
    }
    pub(super) fn check_plan(self, plan: &PreparedInt8SourceMap<'_>) -> Result<(), BatchItemFailure> {
        match self { Self::Summary { limits, .. } => plan.check_summary(limits).map_err(summary_failure),
            Self::Keyphrases { limits, .. } => plan.check_keyphrases(limits).map_err(keyphrase_failure) }
    }
    fn execute<C: DecodeStepControl>(self, plan: PreparedInt8SourceMap<'_>, identities: &[ExecutionIdentity],
        engine: &mut StrictInt8Engine<'_>, vocabulary: &ExtractionVocabulary, control: &mut C)
        -> Result<Int8ReducedSourceDocumentRun, BatchItemFailure> {
        match self {
            Self::Summary { limits, .. } => plan.execute_summary_with_control(identities, engine, vocabulary, limits, control)
                .map(Int8ReducedSourceDocumentRun::Summary).map_err(summary_failure),
            Self::Keyphrases { limits, .. } => plan.execute_keyphrases_with_control(identities, engine, vocabulary, limits, control)
                .map(Int8ReducedSourceDocumentRun::Keyphrases).map_err(keyphrase_failure),
        }
    }
}
fn scans_cover_attempts(per_attempt: u64, lifetime: u64, attempts: u64) -> bool {
    per_attempt != 0 && attempts != 0 && per_attempt.checked_mul(attempts).is_some_and(|n| n <= lifetime)
}

// Freeze every native limit by exhaustive destructuring. These projections are
// not executable recipes and cannot be deserialized into a prepared plan.
impl Serialize for SourceDocumentReduction {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Contract<A> {
            version: u32, execution: &'static str, reduction_policy: &'static str,
            aggregation: A, max_items: usize, max_result_bytes: usize, max_lifetime_scan_work: u64,
        }
        #[derive(Serialize)]
        struct PhraseLimits { max_unique_phrases: usize, max_evidence_spans: usize, max_value_bytes: usize, max_scan_work: u64 }
        match *self {
            Self::Summary { limits, max_lifetime_scan_work } => {
                let Int8SummaryLimits { aggregation, max_bullets, max_result_bytes } = limits;
                Contract { version: 1, execution: self.execution(), reduction_policy: SUMMARY_REDUCTION_POLICY,
                    aggregation, max_items: max_bullets, max_result_bytes, max_lifetime_scan_work }.serialize(serializer)
            }
            Self::Keyphrases { limits, max_lifetime_scan_work } => {
                let Int8KeyphraseLimits { aggregation, max_phrases, max_result_bytes } = limits;
                let CorpusKeyphraseLimits { max_unique_phrases, max_evidence_spans, max_value_bytes, max_scan_work } = aggregation;
                Contract { version: 1, execution: self.execution(), reduction_policy: CORPUS_KEYPHRASE_POLICY,
                    aggregation: PhraseLimits { max_unique_phrases, max_evidence_spans, max_value_bytes, max_scan_work },
                    max_items: max_phrases, max_result_bytes, max_lifetime_scan_work }.serialize(serializer)
            }
        }
    }
}

impl Int8SourceDocumentJobPlanner {
    /// Consume the map factory to opt into document-wide reduction. Invalidate
    /// old prepared objects even when their native prompts would be identical.
    /// Ordinary map jobs retain their exact v1 recipe bytes.
    pub fn with_reduction(mut self, policy: SourceDocumentReduction) -> Result<Self, BatchFault> {
        policy.validate(&self.task, self.mapping, self.limits)?;
        self.recipe.version = 2;
        self.recipe.execution = policy.execution();
        self.recipe.reduction_task = Some(policy);
        bounded_json(&self.recipe, 1024 * 1024 - 1024).map_err(|_| BatchCode::InvalidLimits)?;
        self.owner = Arc::new(());
        Ok(self)
    }
    pub fn reduction_policy(&self) -> Option<SourceDocumentReduction> { self.recipe.reduction_task }
}
fn reduced_failure(cancellation: Option<DecodeCancellationKind>, invalid_limits: bool) -> BatchItemFailure {
    if let Some(cause) = cancellation { return BatchItemFailure::fatal(BatchFault::cancelled(cause)); }
    BatchItemFailure::fatal(if invalid_limits { BatchCode::InvalidLimits } else { BatchCode::InvalidExecution })
}
fn summary_failure(error: Int8CorpusSummaryError) -> BatchItemFailure {
    reduced_failure(error.cancellation(), matches!(error, Int8CorpusSummaryError::InvalidLimits))
}
fn keyphrase_failure(error: Int8CorpusKeyphraseError) -> BatchItemFailure {
    reduced_failure(error.cancellation(), matches!(error, Int8CorpusKeyphraseError::InvalidLimits))
}

#[cfg(test)] mod tests;
