//! Sequential physical phases with one complete native peak reservation.
//! Discovery cohorts drain before scalar synthesis begins. Work, evidence,
//! transcripts and final semantics remain owned by each concrete task.
use super::*;
use crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE;

pub const INT4_SYNTHESIS_COHORT_EXECUTION: &str = "grouped-int4-discovery-cohorts-scalar-synthesis-v1";

#[derive(Serialize)]
pub struct Int4SynthesisCohortRun<T> {
    pub schema_version: u32,
    pub execution: &'static str,
    /// Complete existing synthesis or hierarchy result, including all phases.
    pub document: T,
    /// Physical discovery only. Synthesis work remains in the document's
    /// separate phase counters and is never reported as a shared map cohort.
    pub discovery_cohorts: Int4SourceMapCohortStats,
}

impl Int4SourceMapCohortRequirement {
    /// Admit one full scalar synthesis context alongside the discovery peak
    /// in time, not simultaneously in memory. Each component is their maximum.
    /// The scalar context must fit the original task's KV authority before
    /// any discovery can begin, even if discovery later collects no evidence.
    /// Discovery decoder workspace and the task's complete evidence/transcript
    /// frontier remain separately charged by the embedding host.
    pub fn with_synthesis_context(mut self, context_limit: usize, group_width: usize,
        task_kv_limit: u64) -> Result<Self, Int4SourceMapError> {
        let scalar = Int4MemoryRequirement::for_context(context_limit, group_width).map_err(native_error)?;
        if scalar.kv_bytes > task_kv_limit { return Err(Int4SourceMapError::Admission); }
        self.native.kv_bytes = self.native.kv_bytes.max(scalar.kv_bytes);
        self.native.rope_bytes = self.native.rope_bytes.max(scalar.rope_bytes);
        self.native.scratch_payload_bound = self.native.scratch_payload_bound.max(scalar.scratch_payload_bound);
        Ok(self)
    }
}

/// The caller has admitted the complete peak and physically dropped every
/// discovery engine. This function creates only the scalar continuation engine.
pub(in crate::tasks::source_planning::grouped_int4::long)
fn scalar_engine<'w, C: DecodeStepControl>(weights: &'w Int4ModelWeights,
    budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
    -> Result<StrictInt4Engine<'w>, Int4SourceMapError> {
    checkpoint(control)?;
    StrictInt4Engine::new(weights, budget.context_limit, budget.native_memory)
        .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
}

/// The concrete discovery executor has already verified its exact native
/// window count. Reconcile its ledger with the complete retained discovery
/// document, without counting QA whitespace as model work or synthesis as maps.
#[allow(clippy::too_many_arguments)]
pub(in crate::tasks::source_planning::grouped_int4::long)
fn finish<T: Serialize, C: DecodeStepControl>(document: T, discovery_cohorts: Int4SourceMapCohortStats,
    native_chunks: usize, source_windows: usize, planned: Int4Work, actual: Int4Work,
    max_result_bytes: u64, control: &mut C) -> Result<Int4SynthesisCohortRun<T>, Int4SourceMapError> {
    checkpoint(control)?;
    let stats = discovery_cohorts;
    if native_chunks == 0 || source_windows == 0 || stats.chunks != native_chunks as u64
        || stats.cohorts == 0 || stats.cohorts > source_windows as u64 || stats.cohorts > stats.chunks
        || stats.planned_work != planned || stats.model_work != actual || !fits(actual, planned)
        || stats.group_steps == 0 || stats.group_steps > actual.forward_positions
        || stats.projection_groups == 0 || stats.projection_groups > stats.group_steps
        || stats.projection_groups > actual.projected_logits / NANBEIGE_VOCAB_SIZE as u64 {
        return Err(Int4SourceError::InvalidResult.into());
    }
    let run = Int4SynthesisCohortRun { schema_version: 1, execution: INT4_SYNTHESIS_COHORT_EXECUTION,
        document, discovery_cohorts };
    let cap = max_result_bytes.checked_add(COHORT_ENVELOPE_BYTES).ok_or(Int4SourceMapError::WorkLimit)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
    checkpoint(control)?;
    Ok(run)
}

#[cfg(test)] mod tests;
