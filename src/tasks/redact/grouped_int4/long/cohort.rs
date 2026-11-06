//! Native NER chunk cohorts inside the unchanged two-stage redaction pipeline.
//! Original and edited-source maps share one peak and nonrenewable ledgers;
//! only physical scheduling counters are added around the semantic result.
use super::*;
use crate::{
    batch::extract::grouped_int4::memory::MAX_COHORT_RESULT_BYTES,
    native_engine::{int4::PackedInt4Kernel, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4MemoryRequirement, Int4ModelWeights}},
    tasks::source_planning::grouped_int4::long::cohort::{Int4SourceMapCohortBudget,
        Int4SourceMapCohortMemoryLimits, Int4SourceMapCohortRequirement, Int4SourceMapCohortStats},
};

pub const INT4_LONG_REDACTION_COHORT_EXECUTION: &str = "grouped-int4-document-redaction-chunk-cohorts-v1";
pub const INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES: u64 = 4096;

#[derive(Serialize)]
pub struct Int4LongRedactionCohortRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub cohort_chunks: usize,
    pub original_cohorts: Int4SourceMapCohortStats,
    pub verification_cohorts: Option<Int4SourceMapCohortStats>,
    /// Same edited text, policy/key scope, evidence and semantic stage receipts
    /// as the serial schedule. No native transcript or private source escapes.
    pub output: Int4LongRedactionRun,
}
impl Int4LongRedactionCohortRun {
    /// Check physical counters against the complete semantic stage receipts.
    /// This is a consistency check, not authentication of external results.
    pub fn validate_schedule(&self) -> Result<(), Int4LongRedactionError> {
        if self.schema_version != 1 || self.execution != INT4_LONG_REDACTION_COHORT_EXECUTION
            || !(1..=64).contains(&self.cohort_chunks) {
            return Err(Int4RedactionError::InvalidResult.into());
        }
        check_stats(self.original_cohorts, &self.output.original, self.cohort_chunks)?;
        match (&self.output.verification, self.verification_cohorts) {
            (Some(stage), Some(stats)) => check_stats(stats, stage, self.cohort_chunks)?,
            (None, None) => {},
            _ => return Err(Int4RedactionError::InvalidResult.into()),
        }
        Ok(())
    }
}

impl Int4LongRedactionConfig {
    /// Pure pre-IO quote for a complete declared window in EITHER stage.
    /// The original map dies before the edited-source map starts, so these
    /// native/cursor/result owners are reused at peak, not charged per stage.
    /// Preparation, map/coordinate/edit storage and final output are separate.
    /// Per-chunk task KV authority remains the original single-row ceiling.
    pub fn cohort_requirements(&self, context_tokens: usize, group_width: usize)
        -> Result<Int4SourceMapCohortRequirement, Int4LongRedactionError> {
        self.per_chunk.validate().map_err(|_| RedactError::InvalidOptions)?;
        self.mapping.chunks.effective_token_limit().map_err(Int4SourceMapError::Chunk)?;
        let chunks = self.mapping.reduction.map_batch_chunks;
        if chunks > self.mapping.chunks.max_chunks || !(1..=256).contains(&self.mapping.chunks.max_chunks)
            || self.mapping.chunks.context_tokens > self.planning.max_context_tokens
            || self.planning.max_context_tokens > context_tokens {
            return Err(Int4SourceMapError::InvalidLimits.into());
        }
        let row = Int4MemoryRequirement::for_context(context_tokens, group_width)
            .map_err(Int4RedactionError::from)?;
        if row.kv_bytes > self.per_chunk.max_kv_bytes { return Err(Int4SourceMapError::Admission.into()); }
        Ok(Int4SourceMapCohortMemoryLimits { max_chunks: chunks, context_tokens,
            max_output_tokens_per_chunk: u64::from(self.per_chunk.max_output_tokens),
            max_result_bytes_per_chunk: self.per_chunk.max_output_bytes }.requirements(group_width)?)
    }
}

impl Int4DocumentRedactor<'_> {
    /// Admit every original chunk's actual cohort geometry before optional
    /// snapshot persistence or native execution. Verification must separately
    /// admit the actual edited text using the SAME remaining operation limits.
    pub fn preflight_cohorts<C: DecodeStepControl>(&self, source: &str,
        budget: Int4SourceMapCohortBudget, control: &mut C)
        -> Result<Int4LongRedactionPreflight, Int4LongRedactionError> {
        poll(control)?;
        self.config.cohort_requirements(budget.context_limit, self.group_width)?;
        let prepared = self.prepare(source, self.config.mapping, control)?;
        check_budget(prepared.cohort_requirements(budget.context_limit)?, budget)?;
        poll(control)?;
        Ok(metadata(prepared.preflight_metadata()))
    }

    /// Execute both NER maps through real source-map cohorts. Each stage
    /// preflights all of its windows before the first forward. The edited text
    /// is freshly partitioned; no original boundaries or detections are reused.
    /// A failed later window, edit, verification or finalizer returns no result.
    #[allow(clippy::too_many_arguments)]
    pub fn redact_cohorts<C: DecodeStepControl>(&self, source: &str, request: &RedactionRequest,
        pseudonyms: Option<&Pseudonyms<'_>>, weights: &Int4ModelWeights,
        vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4LongRedactionCohortRun, Int4LongRedactionError> {
        poll(control)?;
        check_model(&self.identity, weights.identity())?;
        if weights.group_width() != self.group_width { return Err(Int4RedactionError::Identity.into()); }
        self.config.cohort_requirements(budget.context_limit, self.group_width)?;
        let mut driver = CohortDetection { weights, vocabulary, budget, kernel, control,
            stages: [None, None], completed: 0 };
        let output = self.run_pipeline(source, request, pseudonyms, &mut driver)?;
        let original_cohorts = driver.stages[0].ok_or(Int4RedactionError::InvalidResult)?;
        let run = Int4LongRedactionCohortRun { schema_version: 1,
            execution: INT4_LONG_REDACTION_COHORT_EXECUTION,
            cohort_chunks: self.config.mapping.reduction.map_batch_chunks,
            original_cohorts, verification_cohorts: driver.stages[1], output };
        if driver.completed != 1 + usize::from(request.verify) {
            return Err(Int4RedactionError::InvalidResult.into());
        }
        run.validate_schedule()?;
        let cap = self.config.max_result_bytes.checked_add(INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES)
            .ok_or(Int4SourceMapError::WorkLimit)?;
        crate::tasks::extract::quantized::check_size(&run, cap).map_err(Int4SourceError::from)?;
        poll(driver.control)?;
        Ok(run)
    }
}

struct CohortDetection<'a, C> {
    weights: &'a Int4ModelWeights, vocabulary: &'a ExtractionVocabulary,
    budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &'a mut C,
    stages: [Option<Int4SourceMapCohortStats>; 2], completed: usize,
}
impl<C: DecodeStepControl> DetectionDriver for CohortDetection<'_, C> {
    fn checkpoint(&mut self) -> Result<(), Int4LongRedactionError> { poll(self.control) }
    fn detect<'s>(&mut self, recipe: &Int4DocumentRedactor<'_>, source: &'s str,
        request: &RedactionRequest, accounting: &mut Accounting, grounding: &mut GroundingBudget)
        -> Result<(DetectedDocument<'s>, Int4LongRedactionStage), Int4LongRedactionError> {
        let slot = self.stages.get_mut(self.completed).ok_or(Int4RedactionError::InvalidResult)?;
        if slot.is_some() { return Err(Int4RedactionError::InvalidResult.into()); }
        let (weights, vocabulary, budget, kernel) = (self.weights, self.vocabulary, self.budget, self.kernel);
        let result = recipe.detect_with_map(source, request, accounting, grounding, self.control,
            |prepared, admitted, control| {
                let run = prepared.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?;
                *slot = Some(run.cohorts);
                Ok(run.document)
            })?;
        self.completed += 1;
        Ok(result)
    }
}

fn check_budget(need: Int4SourceMapCohortRequirement, budget: Int4SourceMapCohortBudget)
    -> Result<(), Int4LongRedactionError> {
    if need.native.kv_bytes > budget.native_memory.max_kv_bytes
        || need.native.rope_bytes > budget.native_memory.max_rope_bytes
        || need.native.scratch_payload_bound > budget.native_memory.max_scratch_payload_bytes
        || need.workspace_bytes > budget.max_workspace_bytes || need.max_result_bytes > budget.max_result_bytes
        || budget.max_result_bytes > MAX_COHORT_RESULT_BYTES {
        return Err(Int4SourceMapError::Admission.into());
    }
    Ok(())
}
fn check_stats(stats: Int4SourceMapCohortStats, stage: &Int4LongRedactionStage, width: usize)
    -> Result<(), Int4LongRedactionError> {
    let chunks = stage.preflight.chunks as u64;
    let heads = stage.model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64;
    if chunks == 0 || stats.chunks != chunks || stats.cohorts != chunks.div_ceil(width as u64)
        || stats.planned_work != stage.preflight.planned_model_work || stats.model_work != stage.model_work
        || stats.group_steps == 0 || stats.group_steps > stage.model_work.forward_positions
        || stats.group_steps < stage.model_work.forward_positions.div_ceil(width as u64)
        || stage.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || stats.projection_groups == 0 || stats.projection_groups > stats.group_steps
        || stats.projection_groups > heads || stats.projection_groups < heads.div_ceil(width as u64) {
        return Err(Int4RedactionError::InvalidResult.into());
    }
    Ok(())
}
