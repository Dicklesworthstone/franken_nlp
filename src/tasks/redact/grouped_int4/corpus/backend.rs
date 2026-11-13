//! Closed schedules for the real complete-document redactor. Cohort windows
//! have local native ownership and never substitute completed/injected output.
use super::*;
use crate::{
    batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES,
    native_engine::strict_int4::Int4MemoryRequirement,
    tasks::redact::grouped_int4::long::Int4LongRedactionCohortRun,
};

pub(super) enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel },
}
impl Backend<'_, '_> {
    pub(super) fn model_identity(&self) -> &Int4ModelIdentity {
        match self { Self::Serial(engine) => engine.model_identity(), Self::Cohorts { weights, .. } => weights.identity() }
    }
    pub(super) fn clean(&self) -> bool {
        match self {
            Self::Serial(engine) => super::clean(engine),
            // The concrete redactor drains every locally owned cohort before
            // returning, including unwind/error/cancellation paths.
            Self::Cohorts { .. } => true,
        }
    }
    pub(super) fn capacity(&self, config: &Int4DocumentRedactionBatchConfig, width: usize)
        -> Result<u64, BatchItemFailure> {
        match self {
            Self::Serial(engine) => super::capacity(engine, config, width),
            Self::Cohorts { weights, budget, .. } => {
                check_model(&config.ner_identity, weights.identity())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if weights.group_width() != width { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
                Ok(cohort_budget(config, width, *budget)?.native.kv_bytes)
            }
        }
    }
    pub(super) fn preflight<C: DecodeStepControl>(&self, redactor: &Int4DocumentRedactor<'_>,
        source: &str, control: &mut C) -> Result<Int4LongRedactionPreflight, Int4LongRedactionError> {
        match self {
            Self::Serial(_) => redactor.preflight(source, control),
            Self::Cohorts { budget, .. } => redactor.preflight_cohorts(source, *budget, control),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn redact<C: DecodeStepControl>(&mut self, redactor: &Int4DocumentRedactor<'_>,
        source: &str, config: &Int4DocumentRedactionBatchConfig, pseudonyms: Option<&Pseudonyms<'_>>,
        vocabulary: &ExtractionVocabulary, control: &mut C) -> Result<Int4LongRedactionRun, Int4LongRedactionError> {
        match self {
            Self::Serial(engine) => redactor.redact(source, &config.request, pseudonyms, engine, vocabulary, control),
            Self::Cohorts { weights, budget, kernel } => {
                let run = redactor.redact_cohorts(source, &config.request, pseudonyms,
                    weights, vocabulary, *budget, *kernel, control)?;
                completed_with_control(run, config.detector.mapping.reduction.map_batch_chunks, control)
            }
        }
    }
}

/// Price the full declared window, not whichever shorter original source was
/// supplied first. The edited source may need more simultaneous chunk rows.
pub(super) fn cohort_budget(config: &Int4DocumentRedactionBatchConfig, width: usize,
    budget: Int4SourceMapCohortBudget) -> Result<Int4SourceMapCohortRequirement, BatchItemFailure> {
    if budget.max_result_bytes > MAX_COHORT_RESULT_BYTES {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    let required = config.detector.cohort_requirements(budget.context_limit, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    let supplied = Int4SourceMapCohortRequirement {
        native: Int4MemoryRequirement { kv_bytes: budget.native_memory.max_kv_bytes,
            rope_bytes: budget.native_memory.max_rope_bytes,
            scratch_payload_bound: budget.native_memory.max_scratch_payload_bytes },
        workspace_bytes: budget.max_workspace_bytes, max_result_bytes: budget.max_result_bytes,
    };
    if required.native.kv_bytes > supplied.native.kv_bytes || required.native.rope_bytes > supplied.native.rope_bytes
        || required.native.scratch_payload_bound > supplied.native.scratch_payload_bound
        || required.workspace_bytes > supplied.workspace_bytes || required.max_result_bytes > supplied.max_result_bytes {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(required)
}
#[cfg(test)]
pub(super) fn completed(run: Int4LongRedactionCohortRun, width: usize)
    -> Result<Int4LongRedactionRun, Int4LongRedactionError> {
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<crate::native_engine::decode::DecodeCancellationKind> { None }
    }
    completed_with_control(run, width, &mut Continue)
}
fn completed_with_control<C: DecodeStepControl>(run: Int4LongRedactionCohortRun, width: usize, control: &mut C)
    -> Result<Int4LongRedactionRun, Int4LongRedactionError> {
    if run.cohort_chunks != width { return Err(Int4RedactionError::InvalidResult.into()); }
    run.validate_schedule_with_control(control)?;
    Ok(run.output)
}
